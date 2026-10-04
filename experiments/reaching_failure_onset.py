"""Saved-state onset and contact prediction audit; no new controller or rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from common import ROOT, load_configs, make_model, prepare_paired_initial_condition, recovery_config, write_csv
from reaching_pose_origin import PoseOriginController
from reaching_task_audit import objective_records
from reaching_wrench_audit import contact_cone, cone_residual
from se3_whole_body_control.control.tasks import com_jacobian


def event_start(times, mask, duration=0):
    """First qualifying run's onset, not its later confirmation time."""
    starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
    ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])
    return int(next((s for s,e in zip(starts,ends) if times[e]-times[s]>=duration),-1))


def events(a, summary, recovery):
    t=a['time_s']
    # Post-step channels describe the end of the control interval, not t itself.
    post=np.r_[t[1:],t[-1]+np.median(np.diff(t))]
    start=summary['provenance']['push_start_s']
    slip_speed=np.any(a['foot_tangent_velocity_post_step']>recovery.slip_tangent_velocity_threshold_m_s,axis=1)
    slip_distance=np.any(a['foot_xy_displacement_post_step']>recovery.slip_displacement_threshold_m,axis=1)
    friction=np.any(a['actual_friction_utilization_post_step']>recovery.friction_utilization_threshold,axis=1)
    loss=~(a['contact_left_post_step'] & a['contact_right_post_step'])
    com=np.linalg.norm(a['com_world'][:,:2]-np.array(summary['provenance']['com_reference_world'])[:2],axis=1)
    definitions=(('slip_speed_first',slip_speed,post,0),('slip_distance_first',slip_distance,post,0),
                 ('friction_first',friction,post,0),('contact_loss_first',loss,post,0),
                 ('slip_sustained',slip_speed|slip_distance|friction,post,recovery.slip_duration_s),
                 ('contact_loss_sustained',loss,post,recovery.contact_loss_duration_s),
                 ('com_band_first',com>recovery.com_displacement_threshold_m,t,0),
                 ('fall_height_first',a['torso_height_m']<recovery.torso_ground_height_m,t,0))
    result={}
    for name,mask,clock,duration in definitions:
        index=event_start(clock,mask & (clock>=start),duration)
        result[name]=None if index<0 else dict(index=index,time_s=float(clock[index]),
                                             confirmation_duration_s=duration,channel='post_step' if clock is post else 'pre_step')
    return result


def snapshot(path, a, summary, index, cfg, initial, controller_type=PoseOriginController, verify_control=True):
    i=index
    model=make_model(cfg)
    model.reset(a['qpos_history'][0],a['qvel_history'][0])
    controller=controller_type(model,cfg['controller'],summary['provenance']['reaching'])
    controller.q_des=a['joint_reference_history'][i].copy()
    controller.pd_fallback.q_des=controller.q_des.copy()
    controller.T_des_torso,controller.T_des_pelvis=initial.desired_torso.copy(),initial.desired_pelvis.copy()
    controller.com_des=initial.com_reference.copy()
    model.reset(a['qpos_history'][i],a['qvel_history'][i])
    model.data.time=float(a['time_s'][i])
    force=a['push_force'][i]
    if np.any(force):
        model.set_external_force(cfg['experiments']['push']['application_body'],force)
    model.data.ctrl[:]=a['control'][i]
    mujoco.mj_forward(model.model,model.data)
    result=controller.solve()
    if not result.success and not verify_control:
        return dict(success=False,time_s=float(a['time_s'][i]),message=result.message)
    assert result.success,result.message
    if verify_control:
        np.testing.assert_allclose(result.control,a['control'][i],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(result.contact_wrench,a['predicted_contact_wrench'][i],rtol=1e-6,atol=1e-6)
    objectives,cost,_=objective_records(controller)
    Jcom=com_jacobian(model)
    velocity=Jcom@model.data.qvel
    mass=float(np.sum(model.model.body_mass))
    gravity=np.array(model.model.opt.gravity)
    measured=a['actual_contact_wrench'][i].reshape(2,6)
    predicted=result.contact_wrench.reshape(2,6)
    local_points=[]
    feet=[]
    for foot,name in enumerate(('left_foot','right_foot')):
        G,points,dimensions=contact_cone(model,foot)
        pose=model.body_pose(name)
        spec=controller.specs[foot]
        J,bias=spec['J'],spec['bias']
        predicted_point_acceleration=(J@result.qdd+bias).reshape(-1,3)
        target=(-summary['provenance']['reaching']['contact_velocity_damping_s_inv']*(J@model.data.qvel)).reshape(-1,3)
        local=[pose[:3,:3].T@(p-pose[:3,3]) for p in spec['points']]
        local_points.append((name,local,(J@model.data.qvel).reshape(-1,3)))
        feet.append(dict(foot=name,point_count=len(points),motion_rank=spec['rank'],
                         predicted_wrench_world=predicted[foot].tolist(),measured_pre_step_wrench_world=measured[foot].tolist(),
                         measured_post_step_wrench_world=a['actual_contact_wrench_post_step'][i,6*foot:6*foot+6].tolist(),
                         force_prediction_error_N=float(np.linalg.norm(predicted[foot,:3]-measured[foot,:3])),
                         predicted_inner_cone_residual=cone_residual(spec['G'],predicted[foot]),
                         measured_inner_cone_residual=cone_residual(spec['G'],measured[foot]),
                         measured_outer_cone_residual=cone_residual(G,measured[foot]),
                         predicted_point_acceleration_m_s2=predicted_point_acceleration.tolist(),
                         point_target_acceleration_m_s2=target.tolist(),
                         predicted_point_target_residual_norm=float(np.linalg.norm(predicted_point_acceleration-target)),
                         actual_post_normal_force_N=float(a['actual_normal_force_post_step_N'][i,foot]),
                         actual_post_tangent_speed_m_s=float(a['foot_tangent_velocity_post_step'][i,foot])))
    dt=float(a['time_s'][i+1]-a['time_s'][i])
    model.reset(a['qpos_history'][i+1],a['qvel_history'][i+1])
    observed_com=(com_jacobian(model)@model.data.qvel-velocity)/dt
    for foot,(name,local,before) in enumerate(local_points):
        after=np.array([model.attached_point_kinematics(name,p)[1]@model.data.qvel for p in local]).reshape(-1,3)
        observed=(after-before)/dt
        feet[foot]['observed_interval_material_point_acceleration_m_s2']=observed.tolist()
        predicted_acc=np.array(feet[foot]['predicted_point_acceleration_m_s2']).reshape(-1,3)
        feet[foot]['point_acceleration_prediction_interval_difference_norm']=float(np.linalg.norm(observed-predicted_acc))
    return dict(success=True,index=i,time_s=float(a['time_s'][i]),interval_end_s=float(a['time_s'][i+1]),feet=feet,
                objectives=objectives,objective_sum=cost,max_qdd=float(np.max(abs(result.qdd))),
                torque_peak_Nm=float(np.max(abs(result.control))),
                relaxed_point_rows=getattr(controller,'relaxed_point_rows',[]),
                replay_control_max_error_Nm=float(np.max(abs(result.control-a['control'][i]))),
                replay_wrench_max_error=float(np.max(abs(result.contact_wrench-a['predicted_contact_wrench'][i]))),
                constraint_budget_ratio=result.diagnostics['constraint_budget_ratio'],
                external_force_oracle=result.diagnostics['external_force_oracle'],push_force_N=force.tolist(),
                predicted_force_com_acceleration_no_push_m_s2=(predicted[:,:3].sum(axis=0)/mass+gravity).tolist(),
                predicted_force_com_acceleration_with_known_push_m_s2=((predicted[:,:3].sum(axis=0)+force)/mass+gravity).tolist(),
                measured_pre_force_com_acceleration_with_push_m_s2=((measured[:,:3].sum(axis=0)+force)/mass+gravity).tolist(),
                observed_interval_com_acceleration_m_s2=observed_com.tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    root=ROOT/'results/reaching_contact/pose_disturbance_validation'
    study=json.loads((root/'study.json').read_text())
    records,flat=[],[]
    for row in study['trials']:
        if row['combined_success'] and row['case'] not in ('control','direction_45'):
            continue
        path=root/'trials'/row['trial_id']
        with np.load(path/'trajectory.npz',allow_pickle=False) as payload:
            a={k:payload[k] for k in payload.files}
        summary=json.loads((path/'summary.json').read_text())
        onset=events(a,summary,recovery_config(cfg))
        early=[v['index'] for k,v in onset.items() if v is not None and k.endswith('_first')]
        anchor=min(early) if early else int(np.argmin(abs(a['time_s']-3.152)))
        indices={int(np.argmin(abs(a['time_s']-t))) for t in (2.996,3.152)}
        indices.update(max(0,min(len(a['time_s'])-2,anchor+offset)) for offset in (-10,-1,0,5,20))
        samples=[snapshot(path,a,summary,i,cfg,initial) for i in sorted(indices)]
        records.append(dict(case=row['case'],physical_result=row['failure_reason'],events=onset,anchor_index=anchor,
                            trajectory_path=path.relative_to(ROOT).as_posix(),trajectory_sha256=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest(),
                            snapshots=samples))
        for sample in samples:
            flat.extend(dict(case=row['case'],time_s=sample['time_s'],**foot) for foot in sample['feet'])
        print(row['case'],onset,flush=True)
    write_csv(flat,args.output/'contacts.csv')
    report=dict(records=records,physical_trials_run=0,controller_changed=False,gains_changed=False,
                source_version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                method='Earliest threshold crossings and sustained-run onsets after push start; post-step clocks corrected to interval end. Replay with saved external force. Compare saved pre/post wrenches and fixed material-point interval accelerations.',
                limitation='Diagnostic snapshots only, not causal interventions. Geometric contact is not force-bearing support. Instantaneous predictions vs observed interval acceleration are distinct. Cone residual uses fixed 0.1 m moment scaling. Actual pre-step wrenches are saved solver outputs, not reconstructed history. Other non-foot contacts may affect force balance after fall.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
