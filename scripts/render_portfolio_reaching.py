"""Re-render saved reaching states with synchronized telemetry; never simulate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
from PIL import Image, ImageDraw

from common import load_configs, make_model
from render_com_support_animation import _nearest_indices
from render_paired_comparison import overlay
from se3_whole_body_control.visualization.fonts import pil_font
from se3_whole_body_control.visualization.renderer import render_trial_frames
from se3_whole_body_control.visualization.video import _ffmpeg_executable


def metrics(arrays):
    return (1000*np.linalg.norm(arrays['reach_point_world']-arrays['reach_goal_world'], axis=1),
            np.rad2deg(arrays['torso_rotation_error_rad']), arrays['torque_utilization'])


def draw_chart(draw, values, times, index, top, title, unit, maximum, color, threshold=None):
    left, right, bottom = 1350, 1870, top+205
    draw.text((1320, top-50), title, font=pil_font(26, weight='bold'), fill='#20262a')
    draw.text((1320, top-15), f'{values[index]:.3f} {unit}', font=pil_font(22), fill=color)
    graph_top = top+28
    def point(t, value):
        return (left+float(t)/5*(right-left), bottom-float(value)/maximum*(bottom-graph_top))
    draw.rectangle((point(4.5,maximum)[0],graph_top,right,bottom), fill='#edf4ee')
    for value in (0,maximum/2,maximum):
        y = point(0,value)[1]
        draw.line((left,y,right,y), fill='#d7dde0', width=1)
        draw.text((1300,y-12), f'{value:g}', font=pil_font(18), fill='#505b61')
    if threshold is not None:
        y = point(0,threshold)[1]
        draw.line((left,y,right,y), fill='#9b4d50', width=2)
        label_y = y+4 if y<graph_top+30 else y-26
        draw.text((left+6,label_y), f'{threshold:g} {unit} limit', font=pil_font(18), fill='#9b4d50')
    points = [point(t,v) for t,v in zip(times,values)]
    draw.line(points, fill='#c1ccd1', width=2)
    if index:
        draw.line(points[:index+1], fill=color, width=3)
    x,y = points[index]
    draw.line((x,graph_top,x,bottom), fill='#687a84', width=1)
    draw.ellipse((x-5,y-5,x+5,y+5), fill=color)
    for t in (0,1,2,3,4,5):
        draw.text((point(t,0)[0]-5,bottom+8), str(t), font=pil_font(18), fill='#505b61')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=ROOT/'results/reaching_workspace/1c63402_final')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selections = json.loads((args.study/'montage_selection.json').read_text())
    records, total = [], 0
    with tempfile.TemporaryDirectory(prefix='portfolio-reaching-') as temp_name:
        temp = Path(temp_name)
        combined = temp/'combined'
        combined.mkdir()
        for _, row in selections:
            path = args.study/'trials'/row['trial_id']/'trajectory.npz'
            with np.load(path, allow_pickle=False) as payload:
                arrays = {k:payload[k] for k in payload.files}
            times = arrays['time_s']
            indices = _nearest_indices(times, np.arange(150)/30)
            assert len(times)==1250 and np.all(np.diff(times)>0)
            sampled = {k:v[indices] if v.ndim and len(v)==len(times) else v for k,v in arrays.items()}
            metadata = overlay(sampled, 'SE(3) WBC')
            for i,item in enumerate(metadata):
                item.update(reach_goal_world=arrays['reach_goal_world'],
                            reach_point_world=sampled['reach_point_world'][i])
            frames = render_trial_frames(make_model(load_configs(ROOT,robot_name='unitree_g1')),
                                         sampled['qpos_history'], temp/row['trial_id'],
                                         width=1280,height=920,overlay_data=metadata)
            values = metrics(arrays)
            outcome = row['failure_reason']
            for frame,index in zip(frames,indices):
                canvas = Image.new('RGB',(1920,1080),'#ffffff')
                with Image.open(frame) as robot:
                    canvas.paste(robot,(0,80))
                draw = ImageDraw.Draw(canvas)
                draw.text((28,22), f"Reach and balance | Front {row['distance_m']*100:g} cm",
                          font=pil_font(34,weight='bold'),fill='#20262a')
                draw.text((1330,26),f'Outcome: {outcome}',font=pil_font(28,weight='bold'),
                          fill='#257a59' if row['combined_success'] else '#ad3946')
                for data,top,title,unit,maximum,color,threshold in (
                    (values[0],155,'Target error','mm',150,'#167c9c',15),
                    (values[1],460,'Torso orientation error','deg',5,'#78569d',5),
                    (values[2],765,'Actuator torque utilization','',1,'#257a59',1)):
                    assert np.all(np.isfinite(data)) and np.max(data)<=maximum
                    draw_chart(draw,data,times,int(index),top,title,unit,maximum,color,threshold)
                draw.text((28,1023),'No push | Fixed-foot stance | Original simulation speed',font=pil_font(25),fill='#505b61')
                draw.text((1320,1023),'Time [s] | Shaded: final hold',font=pil_font(22),fill='#505b61')
                canvas.save(combined/f'frame_{total:06d}.png')
                if total==234:
                    canvas.save(args.output/'reaching-preview.jpg',quality=94)
                total += 1
            records.append(dict(trial_id=row['trial_id'],trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                frame_indices=indices.tolist(),outcome=outcome,source_version=row['source_version']))
        output = args.output/'reach-and-balance.mp4'
        subprocess.run([_ffmpeg_executable(),'-n','-loglevel','error','-framerate','30',
                        '-i',str(combined/'frame_%06d.png'),'-c:v','libx264','-crf','24',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
    assert total==450 and output.stat().st_size<10_000_000
    manifest = dict(trials=records,frames=total,fps=30,duration_s=15,resolution=[1920,1080],
                    dynamics_rerun=False,metric_definition='Distance to final goal, not moving reference',
                    files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in sorted(args.output.iterdir()) if p.is_file()])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
