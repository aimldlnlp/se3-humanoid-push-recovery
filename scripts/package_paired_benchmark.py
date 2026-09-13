'''Validate, summarize, plot, and curate a completed paired benchmark.'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Keep the publication plotter runnable from a clean checkout without an
# editable install. All shipped figures use the repository-owned Latin
# Modern faces registered by the shared visualization style.
SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from se3_whole_body_control.visualization.style import (
    COLORS as STYLE_COLORS,
    apply_style,
    style_axes as shared_style_axes,
)


CONTROLLERS = ('pure_pd', 'pd_nominal_ff', 'se3_wbc')
LABELS = {'pure_pd': 'Pure PD', 'pd_nominal_ff': 'PD + nominal FF', 'se3_wbc': 'SE(3) WBC'}
COLORS = {
    'pure_pd': STYLE_COLORS['pd'],
    'pd_nominal_ff': STYLE_COLORS['left_foot'],
    'se3_wbc': STYLE_COLORS['wbc'],
}
FAILURE_COLORS = {
    'FALL': STYLE_COLORS['push'],
    'SLIP': STYLE_COLORS['left_foot'],
    'TORQUE_LIMIT': STYLE_COLORS['boundary'],
    'FRICTION_LIMIT': STYLE_COLORS['right_foot'],
    'NONFINITE': STYLE_COLORS['angular'],
}
EXPECTED = {'canonical': 3, 'calibration': 144, 'sweep': 576, 'robustness': 150}
PAIR_FIELDS = (
    'initial_condition_sha256', 'initial_qpos_sha256', 'initial_qvel_sha256',
    'torso_reference_sha256', 'pelvis_reference_sha256', 'com_reference_sha256',
    'joint_reference_sha256', 'force_trace_sha256', 'active_push_substeps',
    'realized_push_duration_s', 'realized_impulse_Ns',
)
REQUIRED_PROVENANCE = (
    'source_sha256', 'config_sha256', 'model_sha256',
    'initial_condition_sha256', 'force_trace_sha256', 'trial_summary_path',
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return str(value).strip().lower() == 'true'


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [float('nan'), float('nan')]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [center - radius, center + radius]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def validate_stage(stage: str, rows: list[dict]) -> dict:
    expected = EXPECTED[stage]
    if len(rows) != expected or len({row['trial_id'] for row in rows}) != expected:
        raise RuntimeError(stage + ' count/uniqueness gate failed')
    missing = [field for field in REQUIRED_PROVENANCE if any(not row.get(field) for row in rows)]
    if missing:
        raise RuntimeError(stage + ' missing provenance: ' + ', '.join(missing))
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition_id']].append(row)
    for condition_id, group in groups.items():
        if {row['controller'] for row in group} != set(CONTROLLERS):
            raise RuntimeError(stage + ' incomplete triplet: ' + condition_id)
        for field in PAIR_FIELDS:
            if field in ('realized_push_duration_s', 'realized_impulse_Ns'):
                values = np.asarray([float(row[field]) for row in group])
                equal = np.allclose(values, values[0], atol=1e-12, rtol=0.0)
            else:
                equal = len({row[field] for row in group}) == 1
            if not equal:
                raise RuntimeError(stage + ' paired mismatch: ' + condition_id + ' ' + field)
        if any(str(row['source_dirty']).lower() != 'false' for row in group):
            raise RuntimeError(stage + ' source was dirty: ' + condition_id)
    return {'rows': len(rows), 'conditions': len(groups), 'paired_equality': True}


def selected_conditions(stage: str, rows: list[dict]) -> set[str]:
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition_id']].append(row)
    selected = set()
    for condition_id, group in groups.items():
        outcomes = {truth(row['success']) for row in group}
        reasons = {row['failure_reason'] or 'RECOVERED' for row in group}
        if len(outcomes) > 1 or len(reasons) > 1:
            selected.add(condition_id)
    if stage not in ('calibration', 'sweep'):
        return selected
    for controller in CONTROLLERS:
        by_direction = defaultdict(list)
        for row in rows:
            if row['controller'] == controller:
                by_direction[float(row['push_direction_deg'])].append(row)
        for direction_rows in by_direction.values():
            ordered = sorted(direction_rows, key=lambda row: float(row['push_magnitude_N']))
            for left, right in zip(ordered, ordered[1:]):
                if truth(left['success']) != truth(right['success']):
                    selected.update((left['condition_id'], right['condition_id']))
    return selected


def sanitize(value):
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if key not in ('hostname', 'command')
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return Path(value).name
    return value


def copy_json_sanitized(source: Path, target: Path) -> None:
    write_json(target, sanitize(json.loads(source.read_text(encoding='utf-8'))))


def style_axes(axis) -> None:
    shared_style_axes(axis, grid=True)


def plot_decomposition(rows: list[dict], canonical: list[dict], output: Path) -> None:
    rates = [100.0 * np.mean([truth(row['success']) for row in rows if row['controller'] == controller]) for controller in CONTROLLERS]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    bars = axis.bar([LABELS[item] for item in CONTROLLERS], rates, color=[COLORS[item] for item in CONTROLLERS])
    axis.bar_label(bars, fmt='%.1f%%', padding=4)
    axis.set_ylabel('Recovered sweep trials [%]')
    axis.set_ylim(0, 105)
    axis.set_title('Controller decomposition under the corrected paired protocol')
    canonical_text = 'Canonical 70 N: ' + ' | '.join(
        LABELS[row['controller']] + '=' + ('recover' if truth(row['success']) else row['failure_reason'].lower())
        for row in sorted(canonical, key=lambda row: CONTROLLERS.index(row['controller']))
    )
    axis.text(0.5, -0.18, canonical_text, transform=axis.transAxes, ha='center', fontsize=9)
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(output / 'controller_decomposition.png', dpi=180)
    plt.close(figure)


def plot_envelope(rows: list[dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    directions = sorted({float(row['push_direction_deg']) for row in rows})
    for controller in CONTROLLERS:
        envelope = []
        for direction in directions:
            recovered = [
                float(row['push_magnitude_N']) for row in rows
                if row['controller'] == controller
                and float(row['push_direction_deg']) == direction and truth(row['success'])
            ]
            envelope.append(max(recovered, default=0.0))
        axis.plot(directions, envelope, marker='o', ms=3, lw=1.8, color=COLORS[controller], label=LABELS[controller])
    axis.set(xlabel='Push direction [deg]', ylabel='Largest recovered tested push [N]', xlim=(0, 345))
    axis.set_title('Measured recovery envelope')
    axis.legend(frameon=False, ncol=3)
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(output / 'recovery_envelope.png', dpi=180)
    plt.close(figure)


def plot_basin(rows: list[dict], output: Path) -> None:
    magnitudes = sorted({float(row['push_magnitude_N']) for row in rows})
    directions = sorted({float(row['push_direction_deg']) for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True, sharey=True)
    for axis, controller in zip(axes, CONTROLLERS):
        matrix = np.zeros((len(magnitudes), len(directions)))
        for row in rows:
            if row['controller'] == controller:
                matrix[magnitudes.index(float(row['push_magnitude_N'])), directions.index(float(row['push_direction_deg']))] = truth(row['success'])
        axis.imshow(matrix, origin='lower', aspect='auto', cmap='Blues', vmin=0, vmax=1)
        axis.set_title(LABELS[controller])
        axis.set_xticks(range(0, len(directions), 4), [int(directions[index]) for index in range(0, len(directions), 4)])
        axis.set_yticks(range(len(magnitudes)), [int(value) for value in magnitudes])
        axis.set_xlabel('Direction [deg]')
    axes[0].set_ylabel('Magnitude [N]')
    figure.suptitle('Corrected paired recovery basin (blue = recovered)')
    figure.tight_layout()
    figure.savefig(output / 'recovery_basin.png', dpi=180)
    plt.close(figure)


def plot_outcomes(rows: list[dict], output: Path) -> None:
    groups = defaultdict(dict)
    for row in rows:
        groups[row['condition_id']][row['controller']] = truth(row['success'])
    signatures = defaultdict(int)
    for group in groups.values():
        signatures['/'.join('R' if group[item] else 'F' for item in CONTROLLERS)] += 1
    labels, counts = zip(*sorted(signatures.items(), key=lambda item: (-item[1], item[0])))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(labels, counts, color='#5b8db8')
    axis.bar_label(bars, padding=3)
    axis.set_ylabel('Paired conditions')
    axis.set_xlabel('Pure PD / PD+FF / WBC outcome (R=recovered, F=failed)')
    axis.set_title('Paired outcome differences')
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(output / 'paired_outcome_differences.png', dpi=180)
    plt.close(figure)


def plot_robustness(rows: list[dict], output: Path) -> None:
    rates = [100.0 * np.mean([truth(row['success']) for row in rows if row['controller'] == item]) for item in CONTROLLERS]
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    bars = axis.bar([LABELS[item] for item in CONTROLLERS], rates, color=[COLORS[item] for item in CONTROLLERS])
    axis.bar_label(bars, fmt='%.1f%%', padding=4)
    axis.set(ylabel='Recovered robustness conditions [%]', ylim=(0, 105))
    axis.set_title('Paired robustness comparison (50 common conditions)')
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(output / 'robustness_comparison.png', dpi=180)
    plt.close(figure)


def plot_failure_modes(rows: list[dict], output: Path) -> None:
    reasons = sorted({row['failure_reason'] for row in rows if row['failure_reason']})
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(CONTROLLERS))
    for index, reason in enumerate(reasons):
        counts = np.asarray([
            sum(row['controller'] == controller and row['failure_reason'] == reason for row in rows)
            for controller in CONTROLLERS
        ])
        axis.bar(
            [LABELS[item] for item in CONTROLLERS],
            counts,
            bottom=bottom,
            label=reason,
            color=FAILURE_COLORS.get(reason, STYLE_COLORS['boundary']),
        )
        bottom += counts
    axis.set_ylabel('Failed sweep trials')
    axis.set_title('Failure-mode composition')
    axis.legend(frameon=False, ncol=max(1, min(4, len(reasons))))
    style_axes(axis)
    figure.tight_layout()
    figure.savefig(output / 'failure_modes.png', dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    apply_style()
    raw_root = args.raw_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError('refusing to overwrite publication root: ' + str(output))
    output.mkdir(parents=True)
    rows = {stage: read_rows(raw_root / 'data' / (stage + '.csv')) for stage in EXPECTED}
    validations = {stage: validate_stage(stage, stage_rows) for stage, stage_rows in rows.items()}

    shutil.copytree(raw_root / 'common_states', output / 'common_states')
    shutil.copytree(raw_root / 'data', output / 'data')
    summary_root = output / 'trial_summaries'
    trajectory_root = output / 'trajectories'
    for stage in ('gates', 'canonical', 'calibration', 'sweep', 'robustness'):
        for source in sorted((raw_root / 'raw' / stage).glob('*.json')):
            copy_json_sanitized(source, summary_root / stage / source.name)
    for stage in ('gates', 'canonical'):
        target = trajectory_root / stage
        target.mkdir(parents=True, exist_ok=True)
        for source in sorted((raw_root / 'raw' / stage).glob('*.npz')):
            shutil.copy2(source, target / source.name)
    selections = {}
    for stage in ('calibration', 'sweep', 'robustness'):
        selected = selected_conditions(stage, rows[stage])
        selections[stage] = sorted(selected)
        target = trajectory_root / stage
        target.mkdir(parents=True, exist_ok=True)
        for row in rows[stage]:
            if row['condition_id'] in selected:
                source = raw_root / 'raw' / stage / (row['trial_id'] + '.npz')
                shutil.copy2(source, target / source.name)
    provenance_root = output / 'provenance'
    for source in sorted((raw_root / 'logs').glob('*.json')):
        copy_json_sanitized(source, provenance_root / source.name)

    figure_root = output / 'figures'
    figure_root.mkdir()
    plot_decomposition(rows['sweep'], rows['canonical'], figure_root)
    plot_envelope(rows['sweep'], figure_root)
    plot_basin(rows['sweep'], figure_root)
    plot_outcomes(rows['sweep'], figure_root)
    plot_failure_modes(rows['sweep'], figure_root)
    plot_robustness(rows['robustness'], figure_root)
    if args.video is not None:
        video_root = output / 'videos'
        video_root.mkdir()
        shutil.copy2(args.video.resolve(), video_root / 'canonical_three_controller.mp4')

    rates = {
        stage: {
            controller: {
                **(lambda recovered, total: {
                    'recovered': recovered,
                    'total': total,
                    'rate': recovered / total,
                    'wilson_95': wilson_interval(recovered, total),
                })(
                    sum(truth(row['success']) for row in stage_rows if row['controller'] == controller),
                    sum(row['controller'] == controller for row in stage_rows),
                ),
            }
            for controller in CONTROLLERS
        }
        for stage, stage_rows in rows.items()
    }
    paired_differences = {}
    for stage in ('calibration', 'sweep', 'robustness'):
        grouped = defaultdict(dict)
        for row in rows[stage]:
            grouped[row['condition_id']][row['controller']] = truth(row['success'])
        paired_differences[stage] = {}
        for left, right in (('pure_pd', 'pd_nominal_ff'), ('pd_nominal_ff', 'se3_wbc'), ('pure_pd', 'se3_wbc')):
            paired_differences[stage][left + '_vs_' + right] = {
                'left_only': sum(group[left] and not group[right] for group in grouped.values()),
                'right_only': sum(group[right] and not group[left] for group in grouped.values()),
                'both_recovered': sum(group[left] and group[right] for group in grouped.values()),
                'both_failed': sum(not group[left] and not group[right] for group in grouped.values()),
            }
    write_json(output / 'summary.json', {
        'source_sha256': rows['canonical'][0]['source_sha256'],
        'validations': validations,
        'recovery_counts': rates,
        'paired_differences': paired_differences,
        'curated_condition_ids': selections,
        'canonical': rows['canonical'],
    })
    remote_files = [
        {'path': path.relative_to(raw_root).as_posix(), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}
        for path in sorted(item for item in raw_root.rglob('*') if item.is_file())
    ]
    write_json(output / 'remote_raw_manifest.json', {'files': remote_files})
    curated_files = [
        {'path': path.relative_to(output).as_posix(), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}
        for path in sorted(item for item in output.rglob('*') if item.is_file())
    ]
    write_json(output / 'curated_manifest.json', {'files': curated_files})
    forbidden = ('/home/', '140.113.149.94', 'hucenrotia-ai')
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.txt'):
            text = path.read_text(encoding='utf-8', errors='ignore')
            if any(token in text for token in forbidden):
                raise RuntimeError('operator-specific value in publication: ' + str(path))
    print(json.dumps({'validations': validations, 'recovery_counts': rates}, indent=2))


if __name__ == '__main__':
    main()
