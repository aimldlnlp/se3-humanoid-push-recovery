"""Contact failure timelines and frozen-state replay; no new physical trials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from common import ROOT, load_configs, make_model, prepare_paired_initial_condition, recovery_config, write_csv
from reach_and_balance import ReachingController


def first_event(times, mask, start, duration=0):
    active = np.asarray(mask, dtype=bool) & (times >= start)
    indices = np.flatnonzero(active)
    for group in np.split(indices, np.flatnonzero(np.diff(indices)>1)+1):
        if len(group) and times[group[-1]]-times[group[0]] >= duration:
            return float(times[group[0]])
    return None


def sources():
    track = ROOT/'results/reaching_tracking'
    for weight in (100,3000):
        for phase in ('no_push','moving','hold'):
            suffix = '' if phase=='no_push' else f'_{phase}_70N'
            folder = 'd240d2d_controls' if weight==100 and phase!='no_push' else 'fbc0b50_local'
            yield f'fixed_{weight}', phase, track/folder/'trials'/f'front_12.5cm_w{weight}{suffix}'
    for policy,folder in (('guard','07ffc0f_worker'),('guard_arm','11ebc63_arm_worker')):
        for phase in ('moving','hold'):
            yield policy, phase, ROOT/'results/reaching_guard'/folder/'trials'/f'front_12.5cm_{phase}_balance_guard'


def timeline(a, summary, cfg, start):
    t = a['time_s']
    com = np.linalg.norm(a['com_world'][:,:2]-np.array(summary['provenance']['com_reference_world'])[:2],axis=1)
    velocity = np.max(a['foot_tangent_velocity_post_step'],axis=1)
    displacement = np.max(a['foot_xy_displacement_post_step'],axis=1)
    contact_loss = ~(a['contact_left_post_step'] & a['contact_right_post_step'])
    slip = (velocity>cfg.slip_tangent_velocity_threshold_m_s) | (displacement>cfg.slip_displacement_threshold_m)
    masks = dict(angular_band=a['torso_angular_velocity_norm']>cfg.angular_velocity_threshold_rad_s,
                 orientation_band=a['torso_rotation_error_rad']>cfg.orientation_threshold_rad,
                 com_band=com>cfg.com_displacement_threshold_m,
                 slip_velocity=velocity>cfg.slip_tangent_velocity_threshold_m_s,
                 slip_displacement=displacement>cfg.slip_displacement_threshold_m,
                 contact_loss=contact_loss,torque_bound=a['torque_utilization']>=.999,
                 qp_failure=~a['qp_success'],fall_height=a['torso_height_m']<cfg.torso_ground_height_m)
    events = {key:first_event(t,mask,start) for key,mask in masks.items()}
    events['sustained_slip_onset'] = first_event(t,slip,start,cfg.slip_duration_s)
    events['sustained_contact_loss_onset'] = first_event(t,contact_loss,start,cfg.contact_loss_duration_s)
    stop = events['contact_loss'] if events['contact_loss'] is not None else float(t[-1])
    window = (t>=start) & (t<stop)
    events.update(pre_loss_max_foot_speed_m_s=float(np.max(velocity[window])),
                  pre_loss_max_foot_displacement_m=float(np.max(displacement[window])),
                  pre_loss_max_torque_utilization=float(np.max(a['torque_utilization'][window])))
    return events


def replay(path, a, summary, sample_times, initial, cfg):
    rows = []
    model = make_model(cfg)
    settings = dict(summary['provenance']['reaching'],balance_guard=None)
    for time in sample_times:
        i = int(np.argmin(abs(a['time_s']-time)))
        assert i+1<len(a['time_s'])
        settings['weight'] = float(a['reach_weight_history'][i]) if 'reach_weight_history' in a else settings['weight']
        model.reset(a['qpos_history'][0],a['qvel_history'][0])
        controller = ReachingController(model,cfg['controller'],settings)
        controller.q_des = a['joint_reference_history'][i].copy() if 'joint_reference_history' in a else initial.joint_reference.copy()
        controller.pd_fallback.q_des = controller.q_des.copy()
        controller.T_des_torso,controller.T_des_pelvis = initial.desired_torso.copy(),initial.desired_pelvis.copy()
        controller.com_des = initial.com_reference.copy()
        model.reset(a['qpos_history'][i],a['qvel_history'][i])
        model.data.time = float(a['time_s'][i])
        result = controller.solve()
        assert result.success, (path,i,result.message)
        np.testing.assert_allclose(result.control,a['control'][i],rtol=1e-6,atol=1e-6)
        J = model.contact_jacobian()
        bias = model.contact_bias_acceleration()
        velocity = (J@model.data.qvel).reshape(2,6)
        predicted_acceleration = (J@result.qdd+bias).reshape(2,6)
        model.reset(a['qpos_history'][i+1],a['qvel_history'][i+1])
        next_velocity = (model.contact_jacobian()@model.data.qvel).reshape(2,6)
        dt = a['time_s'][i+1]-a['time_s'][i]
        observed_acceleration = (next_velocity-velocity)/dt
        rows.append(dict(time_s=float(a['time_s'][i]),reach_weight=settings['weight'],
                         control_replay_max_error_Nm=float(np.max(abs(result.control-a['control'][i]))),
                         foot_velocity_world_m_s=velocity[:,:3].tolist(),
                         predicted_foot_acceleration_world_m_s2=predicted_acceleration[:,:3].tolist(),
                         observed_interval_foot_acceleration_world_m_s2=observed_acceleration[:,:3].tolist(),
                         predicted_normal_force_N=result.contact_wrench.reshape(2,6)[:,2].tolist(),
                         actual_pre_step_normal_force_N=a['actual_contact_wrench'][i].reshape(2,6)[:,2].tolist(),
                         actual_post_step_normal_force_N=a['actual_contact_wrench_post_step'][i].reshape(2,6)[:,2].tolist(),
                         contact_slack_norm=float(a['qp_slack_norm'][i]),
                         contact_residual_norm=float(a['contact_acceleration_residual_norm'][i]),
                         measured_double_support=bool(a['contact_left_post_step'][i] and a['contact_right_post_step'][i])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg = load_configs(ROOT,robot_name='unitree_g1')
    criteria = recovery_config(cfg)
    initial = prepare_paired_initial_condition(cfg)
    records, flat, arrays, paired = [], [], {}, {}
    for policy,phase,path in sources():
        with np.load(path/'trajectory.npz',allow_pickle=False) as payload:
            a = {key:payload[key] for key in payload.files}
        summary = json.loads((path/'summary.json').read_text())
        start = summary['provenance']['push_start_s'] if phase!='no_push' else criteria.startup_grace_period_s
        inputs = [a[k] for k in ('time_s','reach_reference_world','reach_goal_world','push_force')]+[a[k][0] for k in ('qpos_history','qvel_history')]
        if phase in paired:
            for old,new in zip(paired[phase],inputs):
                np.testing.assert_allclose(old,new,rtol=0,atol=1e-12)
        paired[phase] = inputs
        events = timeline(a,summary,criteria,start)
        stop = events['contact_loss'] or 4.9
        samples = sorted(set([max(0,start-.004),start+.076,start+.152, min(start+.4,stop-.052),stop-.052]))
        frozen = replay(path,a,summary,samples,initial,cfg)
        record = dict(policy=policy,phase=phase,outcome=summary['combined_success'],events=events,
                      frozen_replay=frozen,path=path.relative_to(ROOT).as_posix(),
                      source_version=summary['provenance']['source_version'],
                      trajectory_sha256=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest())
        records.append(record)
        flat.append(dict(policy=policy,phase=phase,combined_success=record['outcome'],**events))
        arrays[policy,phase] = a
        print(policy,phase,json.dumps(events),flush=True)
    write_csv(flat,args.output/'timeline.csv')
    report = dict(records=records,physical_trials_run=0,input_pairs_verified_atol=1e-12,
                  analysis_source_version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  observed_acceleration_definition='World foot-body linear velocity finite difference over next 4 ms; not an instantaneous contact-point measurement',
                  event_definition='First post-push threshold crossing; sustained onset reported separately; fall classification has precedence',
                  limitation='Association and frozen replay do not establish a unique cause or validate a controller change')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plot(args.output,arrays)
    files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(args.output.iterdir()) if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


def plot(root,arrays):
    import matplotlib.pyplot as plt
    from se3_whole_body_control.visualization.style import apply_style
    apply_style()
    fig,axes=plt.subplots(4,2,figsize=(12,10),constrained_layout=True)
    for col,phase in enumerate(('moving','hold')):
        start=1.5 if phase=='moving' else 3
        for policy,color in (('fixed_100','#167c9c'),('fixed_3000','#ad3946'),('guard','#257a59'),('guard_arm','#78569d')):
            a=arrays[policy,phase]
            curves=(np.max(a['foot_tangent_velocity_post_step'],axis=1),
                    1000*np.max(a['foot_xy_displacement_post_step'],axis=1),
                    np.rad2deg(a['torso_rotation_error_rad']),a['torque_utilization'])
            for ax,y in zip(axes[:,col],curves):
                ax.plot(a['time_s'],y,color=color,label=policy)
        for ax,label,threshold in zip(axes[:,col],('Foot tangential speed [m/s]','Foot XY displacement [mm]','Torso error [deg]','Torque utilization'),(.08,25,5,1)):
            ax.axhline(threshold,color='#777777',ls='--',lw=1)
            ax.axvspan(start,start+.15,color='#777777',alpha=.15)
            ax.set(xlim=(start-.1,5),ylabel=label)
        axes[0,col].set_title(phase.title()+' | Front 12.5 cm | 70 N')
        axes[0,col].legend(fontsize=9)
        axes[-1,col].set_xlabel('Time [s]')
    for suffix in ('png','pdf'):
        fig.savefig(root/f'contact_timeline.{suffix}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    main()
