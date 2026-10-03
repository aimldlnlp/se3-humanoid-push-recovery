"""Verify saved study outcomes and media without rerunning dynamics."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from reaching_workspace import assess_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    root = args.root
    study = json.loads((root/'study.json').read_text())
    rows = study['trials']
    states, actual_states, references = set(), set(), {}
    guard_policies_verified = 0
    for row in rows:
        path = root/'trials'/row['trial_id']
        s, passed, reason = assess_trial(path)
        assert passed == row['combined_success'] and reason == row['failure_reason'], row['trial_id']
        assert s['provenance']['source_version'] == row['source_version']
        states.add(row['initial_condition_sha256'])
        with np.load(path/'trajectory.npz', allow_pickle=False) as a:
            actual_states.add(hashlib.sha256(a['qpos_history'][0].tobytes()+a['qvel_history'][0].tobytes()).hexdigest())
            assert len(a['time_s']) == 1250
            assert np.all(np.diff(a['time_s']) > 0)
            for key in ('qpos_history','qvel_history','control','reach_point_world','reach_reference_world'):
                assert np.all(np.isfinite(a[key])), (row['trial_id'],key)
            key = tuple(a['reach_goal_world'])
            digest = hashlib.sha256(a['reach_reference_world'].tobytes()).hexdigest()
            if key in references:
                assert references[key] == digest
            references[key] = digest
            assert abs(s['provenance']['actual_impulse_Ns']-row['push_N']*.15) < 1e-8
            if 'balance_guard_policy' in study and row['policy']=='balance_guard':
                from reach_and_balance import ReachBalanceGuard
                settings = s['provenance']['reaching']
                guard = settings['balance_guard']
                com_error = np.linalg.norm(a['com_world'][:,:2]-np.asarray(s['provenance']['com_reference_world'])[:2],axis=1)
                risk = ((a['torso_rotation_error_rad'] > guard['orientation_threshold_rad'])
                        | (a['torso_angular_velocity_norm'] > guard['angular_velocity_threshold_rad_s'])
                        | (com_error > guard['com_displacement_threshold_m'])
                        | ~(a['contact_left'] & a['contact_right']))
                np.testing.assert_array_equal(risk, a['balance_guard_risk'])
                replay = ReachBalanceGuard(settings['weight'],guard['minimum_weight'],guard['stable_duration_s'])
                weights = np.array([replay.update(float(t),not bad) for t,bad in zip(a['time_s'],risk)])
                np.testing.assert_array_equal(weights,a['reach_weight_history'])
                if guard.get('track_arm_posture_during_recovery'):
                    from common import load_configs, make_model
                    model = make_model(load_configs(ROOT,robot_name='unitree_g1'))
                    arm = [i for i,name in enumerate(model.joint_names) if name.startswith('right_')
                           and any(part in name for part in ('shoulder','elbow','wrist'))]
                    other = np.setdiff1d(np.arange(model.nu),arm)
                    qref = a['joint_reference_history']
                    np.testing.assert_array_equal(qref[:,other],np.broadcast_to(qref[0,other],(len(qref),len(other))))
                    reduced = weights < settings['weight']
                    np.testing.assert_array_equal(qref[reduced][:,arm],a['qpos_history'][reduced][:,model.joint_qpos_indices[arm]])
                    for i in np.flatnonzero(~reduced):
                        if i:
                            np.testing.assert_array_equal(qref[i],qref[i-1])
                guard_policies_verified += 1
    assert len(states)==1 and len(actual_states)==1
    manifest = json.loads((root/'manifest.json').read_text())
    for file in manifest['files']:
        p=root/file['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == file['sha256'], p
    from PIL import Image
    plot = 'tracking_results.png' if 'selected_reach_weight' in study else 'workspace_results.png'
    if 'frozen_state_audit' in study:
        plot = 'conflict_results.png'
    if 'balance_guard_policy' in study:
        plot = 'guard_results.png'
    with Image.open(root/plot) as image:
        image.verify()
    videos = []
    for path in sorted((root/'videos').glob('*.mp4')):
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
        stream=info['streams'][0]
        assert len(info['streams'])==1 and stream['codec_name']=='h264'
        assert (stream['width'],stream['height'])==(1920,1080)
        assert stream['avg_frame_rate']=='30/1'
        expected=15 if path.name=='reaching_montage.mp4' else 5
        assert abs(float(info['format']['duration'])-expected)<.04
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'], check=True)
        videos.append(dict(path=path.relative_to(root).as_posix(),frames=int(stream['nb_frames']),duration_s=expected))
    report=dict(trials_verified=len(rows), shared_initial_state_verified=True,
                outcomes_recomputed=True, manifest_files_verified=len(manifest['files']),
                deterministic_baseline_regression=study.get('baseline_regression_passed', study.get('baseline_dynamics_reproduced', False)),
                target_reference_groups_verified=len(references), videos=videos)
    if 'balance_guard_policy' in study:
        report['guard_policies_replayed_from_measured_state'] = guard_policies_verified
    print(json.dumps(report,indent=2))
    if args.record:
        (root/'verification.json').write_text(json.dumps(report,indent=2))
        files=[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json']
        (root/'manifest.json').write_text(json.dumps(dict(files=files),indent=2))


if __name__=='__main__':
    main()
