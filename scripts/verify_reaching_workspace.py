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
            if row['stage'] in ('workspace', 'disturbance'):
                key = (row['direction'],row['distance_m'])
                digest = hashlib.sha256(a['reach_reference_world'].tobytes()).hexdigest()
                if key in references:
                    assert references[key] == digest
                references[key] = digest
            assert abs(s['provenance']['actual_impulse_Ns']-row['push_N']*.15) < 1e-8
    assert len(states)==1 and len(actual_states)==1
    manifest = json.loads((root/'manifest.json').read_text())
    for file in manifest['files']:
        p=root/file['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == file['sha256'], p
    from PIL import Image
    with Image.open(root/'workspace_results.png') as image:
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
                deterministic_baseline_regression=study['baseline_regression_passed'],
                target_reference_groups_verified=len(references), videos=videos)
    print(json.dumps(report,indent=2))
    if args.record:
        (root/'verification.json').write_text(json.dumps(report,indent=2))
        files=[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json']
        (root/'manifest.json').write_text(json.dumps(dict(files=files),indent=2))


if __name__=='__main__':
    main()
