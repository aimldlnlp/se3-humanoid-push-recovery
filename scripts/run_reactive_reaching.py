"""Run and render a gated reactive reaching pilot; preserve every attempted trial."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
if os.name != 'nt' and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
from common import (execution_manifest, load_configs, make_model, make_push,
                    prepare_paired_initial_condition, save_paired_initial_condition,
                    load_paired_initial_condition, recovery_config, write_csv)
from reactive_reaching import EVENTS, ReactiveController
from reaching_pose_origin import PoseOriginController
from reaching_tracking_study import verify_pair
from se3_whole_body_control.config import load_yaml
from se3_whole_body_control.evaluation.metrics import save_trial_npz, summarize_trial
from se3_whole_body_control.evaluation.recovery import classify_recovery
from se3_whole_body_control.simulation.mujoco_sim import SimulationRunner


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')


def assess(arrays, initial, config, events, origin, duration):
    time = arrays['time_s']
    com = np.linalg.norm(arrays['com_world'][:, :2]-initial.com_reference[:2], axis=1)
    # Extend the observation window, not the physical thresholds. Evaluate from startup onward.
    cfg = replace(recovery_config(config), timeout_s=duration)
    physical = classify_recovery(time, arrays['torso_rotation_error_rad'], arrays['torso_angular_velocity_norm'],
        com, arrays['contact_left_post_step'], arrays['contact_right_post_step'], arrays['torso_height_m'],
        arrays['torque_abs_max_Nm'], arrays['qp_success'], arrays['actual_friction_margin'], config=cfg,
        actual_friction_utilization=arrays['actual_friction_utilization_post_step'],
        foot_tangent_velocity=arrays['foot_tangent_velocity_post_step'],
        foot_xy_displacement=arrays['foot_xy_displacement_post_step'], torque_utilization=arrays['torque_utilization'],
        joint_limit_violation=arrays['joint_limit_violation'], numerical_valid=arrays['numerical_valid'],
        require_final_double_support=True)
    waypoint_results = []
    for event in events:
        if event['hold'] is None:
            continue
        begin, end = event['hold']
        mask = (time >= begin-1e-9) & (time < end-1e-9)
        complete = bool(np.any(mask) and time[-1] >= end-.004-1e-9)
        errors = np.linalg.norm(arrays['reach_point_world'][mask]-origin-np.array(event['offset']), axis=1)
        maximum = float(np.max(errors)) if errors.size else None
        waypoint_results.append(dict(name=event['name'], window_s=[begin, end],
                                     max_error_mm=maximum*1000 if maximum is not None else None,
                                     passed=complete and maximum <= .015))
    moving = np.zeros(len(time), dtype=bool)
    for i, event in enumerate(events):
        end = min(event['time']+event['duration'], events[i+1]['time'] if i+1<len(events) else duration)
        moving |= (time >= event['time']) & (time < end)
    errors = np.linalg.norm(arrays['reach_point_world']-arrays['reach_reference_world'], axis=1)
    rmse = float(np.sqrt(np.mean(errors[moving]**2))) if np.any(moving) else float('inf')
    final = time >= duration-.5-1e-9
    final_ok = bool(time[-1] >= duration-.004-1e-9 and np.any(final)
        and np.all(arrays['torso_rotation_error_rad'][final] <= cfg.orientation_threshold_rad)
        and np.all(arrays['torso_angular_velocity_norm'][final] <= cfg.angular_velocity_threshold_rad_s)
        and np.all(com[final] <= cfg.com_displacement_threshold_m)
        and np.all(arrays['contact_left_post_step'][final]) and np.all(arrays['contact_right_post_step'][final]))
    rejected = int(np.sum(~arrays['qp_success'].astype(bool)))
    passed = physical.success and final_ok and rejected==0 and rmse <= .015 and all(w['passed'] for w in waypoint_results)
    reason = physical.failure_reason or ('QP_FAILURE' if rejected else 'FINAL_BALANCE' if not final_ok
              else 'WAYPOINT_TRACKING' if not all(w['passed'] for w in waypoint_results)
              else 'MOVING_TRACKING' if rmse>.015 else 'PASS')
    return dict(combined_success=bool(passed), failure_reason=reason, waypoint_results=waypoint_results,
        moving_tracking_rmse_mm=rmse*1000, maximum_tracking_error_mm=float(np.max(errors))*1000,
        final_balance_passed=final_ok, rejected_qps=rejected, physical=asdict(physical))


def trial(output, config, settings, initial, force, regression=False, push_start=None):
    output.mkdir(parents=True, exist_ok=False)
    model = make_model(config)
    model.reset(initial.qpos, initial.qvel)
    controller = (PoseOriginController if regression else ReactiveController)(model, config['controller'], settings)
    duration = 5 if regression else 16
    push_start = (3 if regression else 10.8) if push_start is None else push_start
    if not np.isfinite(push_start) or not 0 <= push_start <= duration-.15:
        raise ValueError('push must fit inside the observation window')
    push = make_push(config, magnitude=force, start=push_start, direction_deg=0) if force else None
    run = SimulationRunner(model, controller, duration, config['robot']['control_timestep'], warmup_duration_s=0).run(
        push=push, initial_qpos=initial.qpos, initial_qvel=initial.qvel, desired_torso=initial.desired_torso,
        desired_pelvis=initial.desired_pelvis, com_reference=initial.com_reference,
        joint_reference=initial.joint_reference, initial_condition_metadata=initial.metadata())
    arrays = run.log.arrays()
    extra = dict(qpos_history=np.asarray(run.qpos_history), qvel_history=np.asarray(run.qvel_history),
        reach_point_world=np.asarray(controller.points), reach_reference_world=np.asarray(controller.references),
        reach_goal_world=controller.goal, reach_weight_history=np.asarray(controller.weights),
        joint_reference_history=np.asarray(controller.joint_references))
    if not regression:
        extra.update(reach_goal_history=np.asarray(controller.goals), event_id=np.asarray(controller.event_ids),
                     reach_velocity_world=np.asarray(controller.velocities), reach_acceleration_world=np.asarray(controller.accelerations))
    arrays.update(extra)
    events = [dict(name='baseline', hold=[4.5, 5], offset=[.125, 0, 0], time=.5, duration=2)] if regression else EVENTS
    result = assess(arrays, initial, config, events, controller.start, duration)
    provenance = execution_manifest({'config':config}) | dict(settings=settings, initial_condition=initial.metadata(),
        duration_s=duration, events=events, push_N=force, push_start_s=push_start, push_direction_deg=0,
        actual_impulse_Ns=run.metadata['realized_impulse_Ns'])
    result.update(summarize_trial(run.log))
    result['provenance'] = provenance
    save_trial_npz(run.log, output/'trajectory.npz', provenance, extra)
    dump(output/'summary.json', result)
    print(output.name, result['combined_success'], result['failure_reason'], flush=True)
    return result


def render(path, output=None):
    import mujoco
    from PIL import Image, ImageDraw
    from se3_whole_body_control.visualization.fonts import pil_font
    from se3_whole_body_control.visualization.renderer import (_add_scene_annotations, _add_scene_geom,
        _rotation_from_z, _draw_overlay)
    from se3_whole_body_control.visualization.video import encode_video
    with np.load(path/'trajectory.npz', allow_pickle=False) as a:
        arrays = {k:a[k] for k in a.files}
    result = json.loads((path/'summary.json').read_text())
    model = make_model(load_configs(ROOT, robot_name='unitree_g1'))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.05, 0, .77]
    camera.distance, camera.azimuth, camera.elevation = 2.1, 140, -12
    hand_camera = mujoco.MjvCamera()
    hand_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    hand_camera.lookat[:] = arrays['reach_point_world'][0]+np.array([.05,-.02,.02])
    hand_camera.distance, hand_camera.azimuth, hand_camera.elevation = .55, 140, -12
    renderer = mujoco.Renderer(model.model, width=1280, height=900)
    frame_count = round(min(16, float(arrays['time_s'][-1])+.004)*30)
    indices = np.argmin(abs(arrays['time_s'][:,None]-np.arange(frame_count)[None,:]/30), axis=0)
    preview = output or path/'media'
    preview.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix='reactive-render-') as directory:
            frames = Path(directory)
            for frame, i in enumerate(indices):
                time = float(arrays['time_s'][i])
                model.data.qpos[:] = arrays['qpos_history'][i]
                mujoco.mj_forward(model.model, model.data)
                renderer.update_scene(model.data, camera=camera)
                stage = 'Easy: reach and hold' if time<4 else 'Harder: follow changing targets' if time<10 else 'Hero: redirect while moving'
                goal = arrays['reach_goal_history'][i]
                point = arrays['reach_point_world'][i]
                event_id = int(arrays['event_id'][i])
                target_label = EVENTS[event_id]['name'].replace('_cancelled',' (interrupted)').replace('_retarget',' (redirected)') if event_id>=0 else 'Waiting'
                metadata = dict(time_s=time, controller='SE(3) whole-body control', status=str(arrays['qp_status'][i]),
                    compact_overlay=True, task_label=stage+' | Target '+target_label, reach_goal_world=goal, reach_point_world=point,
                    contact_left=bool(arrays['contact_left'][i]), contact_right=bool(arrays['contact_right'][i]),
                    push_force=arrays['push_force'][i], push_point_world=arrays['torso_position'][i],
                    push_magnitude_N=result['provenance']['push_N'], push_direction_deg=0)
                if 10.8 <= time < 11.5:
                    metadata['event_label'] = 'Retarget A to B during motion'
                if np.linalg.norm(arrays['push_force'][i])>0:
                    metadata['event_label'] = (metadata.get('event_label','')+' | Torso push').strip(' |')
                _add_scene_annotations(renderer, mujoco, metadata)
                for key, color in [('reach_reference_world',[.2,.4,.85,.8]),('reach_point_world',[.1,.7,.4,.8])]:
                    trail = arrays[key][max(0,i-125):i+1:5]
                    for start, end in zip(trail[:-1],trail[1:]):
                        delta = end-start
                        if np.linalg.norm(delta)>1e-7:
                            _add_scene_geom(renderer,mujoco,mujoco.mjtGeom.mjGEOM_CAPSULE,
                                [.002,np.linalg.norm(delta)/2,0],(start+end)/2,_rotation_from_z(delta),color)
                image = Image.fromarray(renderer.render())
                renderer.update_scene(model.data, camera=hand_camera)
                _add_scene_annotations(renderer, mujoco, {'reach_goal_world':goal,'reach_point_world':point})
                closeup = Image.fromarray(renderer.render()).resize((440,310))
                image.paste(closeup,(810,160))
                _draw_overlay(image, metadata)
                draw = ImageDraw.Draw(image)
                font = pil_font(22)
                draw.text((760,35), f'Whole sequence: {result["failure_reason"]}', font=font, fill='#20262a')
                error = np.linalg.norm(point-arrays['reach_reference_world'][i])*1000
                draw.text((760,70),f'Tracking error: {error:.1f} mm',font=font,fill='#20262a')
                draw.text((760,105),'Pink: goal | Blue: reference | Green: hand',font=pil_font(18),fill='#20262a')
                draw.rectangle((810,160,1250,470),outline='#505b61',width=2)
                draw.text((820,172),'Hand close-up | Target '+target_label,font=pil_font(20),fill='#20262a')
                grf = arrays['actual_contact_wrench'][i, [2,8]]
                draw.text((945,660),'Measured foot load',font=font,fill='#20262a')
                for foot, force in enumerate(grf):
                    x = 965+foot*120
                    height = float(np.clip(force,0,450))/450*130
                    draw.rectangle((x,825-height,x+45,825),fill=('#257a59','#3671ae')[foot])
                    draw.text((x-5,835),f'{("L","R")[foot]} {force:.0f} N',font=pil_font(18),fill='#20262a')
                draw.text((420,867),'Fixed feet | Original simulation speed | Virtual position targets',font=pil_font(19),fill='#505b61')
                image.save(frames/f'frame_{frame:06d}.png')
                if frame in (90,210,325,330,479):
                    image.save(preview/f'frame_{frame:03d}.jpg',quality=94)
            encode_video(frames,preview/'reactive-reaching.mp4',fps=30)
    finally:
        renderer.close()
    dump(preview/'selection.json',dict(frames=frame_count,fps=30,duration_s=frame_count/30,frame_indices=indices.tolist(),
        trajectory_sha256=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest(),dynamics_rerun=False,
        renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hand_inset=True))


def timing_study(root, previous):
    """Three frozen push times; reuse the independently verified nominal pilot."""
    root.mkdir(parents=True,exist_ok=False)
    config = load_configs(ROOT,robot_name='unitree_g1')
    old_plan = json.loads((previous/'plan.json').read_text())
    assert old_plan['events']==EVENTS
    old_verification = json.loads((previous/'verification.json').read_text())
    assert old_verification['nominal_passed'] and old_verification['saved_baseline_dynamics_reproduced']
    # Freeze active dependencies, not unrelated unfinished experiments in the working tree.
    active_experiments={'common.py','reach_and_balance.py','reaching_pose_origin.py',
                        'reaching_task_audit.py','reaching_contact_mode.py','reactive_reaching.py'}
    frozen=[]
    for record in old_plan['source_files']:
        if record['path'].startswith(('src/','configs/','models/')) or (record['path'].startswith('experiments/') and Path(record['path']).name in active_experiments):
            raw=(ROOT/record['path']).read_bytes()
            normalized=raw.replace(b'\r\n',b'\n')
            candidates=(raw,normalized,normalized.replace(b'\n',b'\r\n'))
            assert record['sha256'] in {hashlib.sha256(b).hexdigest() for b in candidates}, record['path']
            frozen.append(dict(path=record['path'],lf_sha256=hashlib.sha256(normalized).hexdigest()))
    settings = old_plan['settings']
    initial = load_paired_initial_condition(previous/'initial_condition.npz')
    cases = [('before',10.55),('simultaneous',10.8),('after',11.05)]
    dump(root/'plan.json',dict(events=EVENTS,cases=cases,force_N=50,duration_s=.15,settings=settings,
        config=config,thresholds=old_plan['thresholds'],nominal_reference=previous.relative_to(ROOT).as_posix(),
        reference_trajectory_sha256=hashlib.sha256((previous/'nominal/trajectory.npz').read_bytes()).hexdigest(),
        source_version=execution_manifest()['source_version'],runner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_source_files=frozen))
    save_paired_initial_condition(initial,root/'initial_condition.npz')
    rows=[]
    for name,start in cases:
        result=trial(root/name,config,settings,initial,50,push_start=start)
        with np.load(previous/'nominal/trajectory.npz') as a,np.load(root/name/'trajectory.npz') as b:
            for key in ('time_s','reach_reference_world','reach_goal_history','reach_velocity_world','reach_acceleration_world'):
                np.testing.assert_array_equal(a[key],b[key])
            for key in ('qpos_history','qvel_history'):
                np.testing.assert_array_equal(a[key][0],b[key][0])
            expected=np.where((b['time_s']>=start-1e-9)&(b['time_s']<start+.15-1e-9),50,0)
            np.testing.assert_allclose(b['push_force'][:,0],expected,atol=1e-9)
        np.testing.assert_allclose(result['provenance']['actual_impulse_Ns'],7.5,atol=1e-8)
        rows.append(dict(trial=name,push_start_s=start,combined_success=result['combined_success'],
            failure_reason=result['failure_reason'],moving_rmse_mm=result['moving_tracking_rmse_mm'],
            final_hold_error_mm=result['waypoint_results'][-1]['max_error_mm'],
            rejected_qps=result['rejected_qps'],deadline_miss_percent=result['qp_deadline_miss_percent']))
        write_csv(rows,root/'study.csv')
        dump(root/'verification.json',dict(trials=rows,all_inputs_paired_with_nominal=True,
            all_passed=all(r['combined_success'] for r in rows),controller_source_unchanged=True))
    render(root/'simultaneous')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--media-output',type=Path,help='Fresh directory for a render-only replay of one trial')
    parser.add_argument('--previous',type=Path,help='Run the three-time study using a saved verified nominal pilot')
    args = parser.parse_args()
    root = args.output
    if args.media_output:
        if not args.render_only or args.previous:
            parser.error('--media-output requires --render-only on a single trial')
        render(root,args.media_output)
        return
    if args.previous:
        timing_study(root,args.previous.resolve())
        return
    if args.render_only:
        for name in ('nominal','hero'):
            if (root/name/'summary.json').exists():
                render(root/name)
        return
    root.mkdir(parents=True, exist_ok=False)
    config = load_configs(ROOT, robot_name='unitree_g1')
    settings = load_yaml(ROOT/'configs/reaching.yaml')
    settings.update(weight=3000,target_offset_world_m=[.125,0,0],contact_velocity_damping_s_inv=20,
                    contact_mode=True,pose_origin=True,experiment_duration_s=16)
    source_files = [p for directory in ('src','experiments','scripts','configs','models')
                    for p in sorted((ROOT/directory).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml','.xml')]
    dump(root/'plan.json',dict(events=EVENTS,duration_s=16,hero_push_N=50,push_start_s=10.8,push_duration_s=.15,
        settings=settings,config=config,thresholds=dict(hold_error_m=.015,moving_rmse_m=.015,
        hold_duration_s=.5,recovery=asdict(recovery_config(config))),
        source_files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_files]))
    initial = prepare_paired_initial_condition(config)
    save_paired_initial_condition(initial,root/'initial_condition.npz')
    regression_settings = settings | dict(experiment_duration_s=5)
    regression = trial(root/'regression_hold',config,regression_settings,initial,70,regression=True)
    verification = dict(regression_passed=regression['combined_success'])
    baseline = ROOT/'results/portfolio_reaching/trials/corrected_hold'
    if baseline.exists():
        verify_pair(baseline,root/'regression_hold',same_dynamics=True)
        verification['saved_baseline_dynamics_reproduced'] = True
    dump(root/'verification.json',verification)
    if not regression['combined_success']:
        return
    nominal = trial(root/'nominal',config,settings,initial,0)
    if nominal['combined_success']:
        hero = trial(root/'hero',config,settings,initial,50)
        with np.load(root/'nominal/trajectory.npz') as a, np.load(root/'hero/trajectory.npz') as b:
            for key in ('time_s','reach_reference_world','reach_goal_history','reach_velocity_world','reach_acceleration_world'):
                np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(a['qpos_history'][0],b['qpos_history'][0])
            np.testing.assert_array_equal(a['qvel_history'][0],b['qvel_history'][0])
        np.testing.assert_allclose(hero['provenance']['actual_impulse_Ns'],7.5,atol=1e-8)
        verification['nominal_hero_inputs_paired'] = True
    verification['nominal_passed'] = nominal['combined_success']
    verification['hero_passed'] = hero['combined_success'] if nominal['combined_success'] else None
    dump(root/'verification.json',verification)
    for name in ('nominal','hero'):
        if (root/name/'summary.json').exists():
            render(root/name)
    dump(root/'manifest.json',dict(files=[dict(path=p.relative_to(root).as_posix(),
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file()]))


if __name__ == '__main__':
    main()
