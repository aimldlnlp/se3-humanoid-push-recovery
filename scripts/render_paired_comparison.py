'''Render a synchronized three-controller MP4 from saved canonical trajectories.'''

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get('SE3_REPO_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'src'))

from common import load_configs, make_model
from se3_whole_body_control.visualization.fonts import pil_font
from se3_whole_body_control.visualization.renderer import render_trial_frames
from se3_whole_body_control.visualization.video import encode_video


CONTROLLERS = (
    ('pure_pd', 'Pure PD'),
    ('pd_nominal_ff', 'PD + nominal FF'),
    ('se3_wbc', 'SE(3) WBC'),
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def overlay(arrays: dict[str, np.ndarray], label: str) -> list[dict]:
    force = arrays['push_force']
    active = np.linalg.norm(force[:, :2], axis=1) > 1e-9
    magnitude = float(np.linalg.norm(force[np.flatnonzero(active)[0], :2])) if np.any(active) else 0.0
    direction = float(np.rad2deg(np.arctan2(force[np.flatnonzero(active)[0], 1], force[np.flatnonzero(active)[0], 0])) % 360.0) if np.any(active) else 0.0
    return [
        {
            'time_s': arrays['time_s'][index],
            'controller': label,
            'push_magnitude_N': magnitude,
            'push_direction_deg': direction,
            'status': arrays['qp_status'][index],
            'com_world': arrays['com_world'][index],
            'feet_xy': arrays['foot_xy_world'][index],
            'push_point_world': arrays['torso_position'][index],
            'push_force': arrays['push_force'][index],
            'contact_left': arrays['contact_left'][index],
            'contact_right': arrays['contact_right'][index],
            'compact_overlay': True,
        }
        for index in range(len(arrays['qpos_history']))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configs = load_configs(ROOT)
    fps = int(configs['robot']['render_fps'])
    stride = max(1, int(round(1.0 / (float(configs['robot']['control_timestep']) * fps))))
    canonical = args.data_root.resolve() / 'raw' / 'canonical'
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_by_controller = {}
    with tempfile.TemporaryDirectory(prefix='paired-render-') as temp_name:
        temp = Path(temp_name)
        panels = []
        for controller, label in CONTROLLERS:
            matches = list(canonical.glob('*_' + controller + '.npz'))
            if len(matches) != 1:
                raise RuntimeError('expected one canonical trajectory for ' + controller)
            arrays = load_npz(matches[0])
            arrays_by_controller[controller] = arrays
            panel_dir = temp / controller
            panels.append(render_trial_frames(
                make_model(configs), arrays['qpos_history'], panel_dir,
                width=640, height=360, stride=stride,
                overlay_data=overlay(arrays, label),
            ))
        count = min(len(panel) for panel in panels)
        combined = temp / 'combined'
        combined.mkdir()
        title_font = pil_font(24, weight='bold')
        body_font = pil_font(18)
        reference = arrays_by_controller['se3_wbc']
        for index in range(count):
            images = [Image.open(panel[index]).convert('RGB') for panel in panels]
            header = 54
            canvas = Image.new('RGB', (sum(image.width for image in images), images[0].height + header), 'white')
            draw = ImageDraw.Draw(canvas)
            x = 0
            for image, (_, label) in zip(images, CONTROLLERS):
                canvas.paste(image, (x, header))
                draw.text((x + 14, 12), label, font=title_font, fill=(25, 35, 45))
                if x:
                    draw.line((x, header, x, canvas.height), fill=(150, 160, 170), width=1)
                x += image.width
            sample = min(index * stride, len(reference['time_s']) - 1)
            shared = 'same measured state + same 70 N push | t = ' + format(float(reference['time_s'][sample]), '.2f') + ' s'
            box = draw.textbbox((0, 0), shared, font=body_font)
            draw.text((canvas.width - (box[2] - box[0]) - 18, 16), shared, font=body_font, fill=(20, 20, 20))
            canvas.save(combined / ('frame_' + format(index, '06d') + '.png'))
        encode_video(combined, output, fps=fps)
    print(str(output))


if __name__ == '__main__':
    main()
