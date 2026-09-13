'''Verify paired benchmark manifests, schemas, counts, hashes, and media.'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


EXPECTED = {'canonical.csv': 3, 'calibration.csv': 144, 'sweep.csv': 576, 'robustness.csv': 150}
REQUIRED = {
    'trial_id', 'condition_id', 'controller', 'success',
    'source_sha256', 'config_sha256', 'model_sha256',
    'initial_condition_sha256', 'force_trace_sha256',
    'realized_push_duration_s', 'realized_impulse_Ns',
    'trial_summary_path',
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_manifest(root: Path, manifest_name: str) -> int:
    manifest = json.loads((root / manifest_name).read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = root / record['path']
        if not path.is_file():
            raise RuntimeError('missing manifest file: ' + record['path'])
        if path.stat().st_size != int(record['size_bytes']):
            raise RuntimeError('size mismatch: ' + record['path'])
        if digest(path) != record['sha256']:
            raise RuntimeError('hash mismatch: ' + record['path'])
    return len(manifest['files'])


def verify_csvs(root: Path) -> dict:
    result = {}
    for name, expected in EXPECTED.items():
        with (root / 'data' / name).open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected or len({row['trial_id'] for row in rows}) != expected:
            raise RuntimeError('row count/uniqueness mismatch: ' + name)
        missing = REQUIRED - set(rows[0])
        if missing:
            raise RuntimeError('missing schema fields in ' + name + ': ' + ', '.join(sorted(missing)))
        result[name] = len(rows)
    return result


def ffmpeg_executable() -> str:
    system = shutil.which('ffmpeg')
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--skip-media', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_count = verify_manifest(root, 'curated_manifest.json')
    csv_counts = verify_csvs(root)
    video = root / 'videos' / 'canonical_three_controller.mp4'
    if not args.skip_media:
        subprocess.run([
            ffmpeg_executable(), '-v', 'error', '-i', str(video),
            '-f', 'null', '-',
        ], check=True)
    forbidden = ('/home/', '140.113.149.94', 'hucenrotia-ai')
    checked_text = 0
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.txt'):
            text = path.read_text(encoding='utf-8', errors='ignore')
            if any(item in text for item in forbidden):
                raise RuntimeError('forbidden operator value: ' + str(path))
            checked_text += 1
    print(json.dumps({
        'manifest_files_verified': manifest_count,
        'csv_rows_verified': csv_counts,
        'text_files_checked': checked_text,
        'media_decoded': not args.skip_media,
    }, indent=2))


if __name__ == '__main__':
    main()
