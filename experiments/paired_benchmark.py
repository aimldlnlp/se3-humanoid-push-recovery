"""Fair, provenance-tracked G1 benchmark from shared measured start states."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    PairedInitialCondition,
    execution_manifest,
    flatten_result,
    load_configs,
    load_paired_initial_condition,
    make_model,
    make_push,
    prepare_paired_initial_condition,
    randomized_initial_state,
    run_trial,
    save_paired_initial_condition,
    save_run,
    write_csv,
)

CONTROLLERS = ('pure_pd', 'pd_nominal_ff', 'se3_wbc')
CALIBRATION_MAGNITUDES_N = (10.0, 20.0, 40.0, 60.0, 80.0, 100.0)
CALIBRATION_DIRECTIONS_DEG = tuple(float(value) for value in range(0, 360, 45))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')


def _duration(configs: dict, push) -> float:
    return push.start_time_s + push.duration_s + float(configs['experiments']['paired_benchmark']['post_push_observation_s'])


def _require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite existing stage output: {path}')


def _stage_manifest(root: Path, stage: str, configs: dict, **extra) -> None:
    payload = execution_manifest({'config': configs, 'run_id': f'paired-{stage}', 'model_path': configs['robot']['model_path']})
    payload.update({'stage': stage, 'controllers': list(CONTROLLERS), **extra})
    _write_json(root / 'logs' / f'{stage}_manifest.json', payload)


def _run_task(task: dict) -> dict:
    configs = task['configs']
    condition = load_paired_initial_condition(Path(task['condition_path']))
    push = make_push(
        configs,
        magnitude=task['magnitude_N'],
        direction_deg=task['direction_deg'],
        duration=task['duration_s'],
        start=task['start_time_s'],
    )
    _, run = run_trial(
        task['controller'], configs, push=push, duration=_duration(configs, push),
        classify=True, initial_condition=condition,
        mass_scale=task.get('mass_scale', 1.0),
        friction_coefficient=task.get('friction_coefficient'),
    )
    arrays = run.log.arrays()
    if not np.all(np.asarray(arrays['numerical_valid'], dtype=bool)):
        raise RuntimeError('non-finite runtime state: ' + str(task['trial_id']))
    provenance = execution_manifest({
        'config': configs,
        'model_path': configs['robot']['model_path'],
        'run_id': task['trial_id'],
    })
    trial_path = Path(task['trial_root']) / f"{task['trial_id']}.npz"
    save_run(run, trial_path, {
        'controller': task['controller'], 'push': push.__dict__, 'config': configs,
        'condition_id': task['condition_id'], 'trial_id': task['trial_id'],
    })
    return flatten_result(
        run, task['controller'], push, task['trial_id'], condition.seed,
        {
            'condition_id': task['condition_id'],
            'source_sha256': provenance['source_version'],
            'source_dirty': provenance['git_dirty'],
            'config_sha256': provenance['config_sha256'],
            'model_sha256': provenance['model_sha256'],
            'dependency_versions_json': json.dumps(provenance['dependency_versions'], sort_keys=True),
            'trial_summary_path': (
                Path('raw') / Path(task['trial_root']).name / (str(task['trial_id']) + '.json')
            ).as_posix(),
            **task.get('extra', {}),
        },
    )


def _execute(tasks: list[dict], workers: int) -> list[dict]:
    if workers == 1:
        rows = [_run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_task, tasks))
    return sorted(rows, key=lambda row: (row['condition_id'], row['controller']))


def _validate_pairs(rows: list[dict], expected_count: int) -> dict:
    if len(rows) != expected_count or len({row['trial_id'] for row in rows}) != expected_count:
        raise RuntimeError(f'incomplete or duplicate results: {len(rows)} rows, expected {expected_count}')
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row['condition_id']), []).append(row)
    for condition_id, group in groups.items():
        if {row['controller'] for row in group} != set(CONTROLLERS):
            raise RuntimeError(f'incomplete controller triplet: {condition_id}')
        for field in (
            'initial_condition_sha256', 'initial_qpos_sha256', 'initial_qvel_sha256',
            'torso_reference_sha256', 'pelvis_reference_sha256', 'com_reference_sha256',
            'joint_reference_sha256', 'force_trace_sha256', 'active_push_substeps',
        ):
            if len({str(row[field]) for row in group}) != 1:
                raise RuntimeError(f'paired equality failed for {condition_id}: {field}')
        for field in ('realized_push_duration_s', 'realized_impulse_Ns'):
            values = np.asarray([float(row[field]) for row in group])
            if not np.allclose(values, values[0], atol=1e-12, rtol=0.0):
                raise RuntimeError(f'paired equality failed for {condition_id}: {field}')
    return {'row_count': len(rows), 'condition_count': len(groups), 'paired_equality': True}


def prepare(root: Path, configs: dict) -> None:
    _require_new(root)
    root.mkdir(parents=True)
    cfg = configs['experiments']['paired_benchmark']
    condition = prepare_paired_initial_condition(configs, seed=0, setup_duration_s=float(cfg['setup_duration_s']))
    save_paired_initial_condition(condition, root / 'common_states' / 'nominal.npz')
    _stage_manifest(root, 'prepare', configs, initial_condition=condition.metadata())


def gates(root: Path, configs: dict) -> None:
    stage_root = root / 'raw' / 'gates'
    _require_new(stage_root)
    condition = load_paired_initial_condition(root / 'common_states' / 'nominal.npz')
    _, perturb_qvel, perturbation = randomized_initial_state(
        configs, int(configs['experiments']['perturbed_standing']['seed']),
        randomization=configs['experiments']['perturbed_standing'],
    )
    perturbed = replace(condition, qvel=condition.qvel + perturb_qvel)
    save_paired_initial_condition(perturbed, root / 'common_states' / 'perturbed.npz')
    records = []
    for label, state, duration, classify in (
        ('quiet', condition, float(configs['experiments']['standing_duration_s']), False),
        ('perturbed', perturbed, float(configs['experiments']['perturbed_standing']['duration_s']), True),
    ):
        for controller in CONTROLLERS:
            _, run = run_trial(controller, configs, duration=duration, classify=classify, initial_condition=state)
            trial_id = f'{label}_{controller}'
            save_run(run, stage_root / f'{trial_id}.npz', {'controller': controller, 'condition_id': label, 'config': configs})
            records.append({'trial_id': trial_id, 'controller': controller, 'condition_id': label, 'initial_condition_sha256': state.sha256, 'recovery': None if run.recovery is None else run.recovery.__dict__})
    _write_json(root / 'data' / 'gates.json', {'perturbation': perturbation, 'records': records})
    _stage_manifest(root, 'gates', configs, trial_count=len(records))


def _grid_tasks(root: Path, configs: dict, stage: str, magnitudes, directions) -> list[dict]:
    condition_path = root / 'common_states' / 'nominal.npz'
    sweep = configs['experiments']['sweep']
    tasks = []
    for magnitude in magnitudes:
        for direction in directions:
            condition_id = f'{float(magnitude):g}N_{float(direction):g}deg'
            for controller in CONTROLLERS:
                tasks.append({
                    'configs': configs, 'condition_path': str(condition_path),
                    'condition_id': condition_id, 'controller': controller,
                    'magnitude_N': float(magnitude), 'direction_deg': float(direction),
                    'duration_s': float(sweep['duration_s']), 'start_time_s': float(sweep['start_time_s']),
                    'trial_id': f'{condition_id}_{controller}', 'trial_root': str(root / 'raw' / stage),
                })
    return tasks


def grid(root: Path, configs: dict, stage: str, magnitudes, directions, workers: int) -> None:
    stage_root = root / 'raw' / stage
    _require_new(stage_root)
    stage_root.mkdir(parents=True)
    tasks = _grid_tasks(root, configs, stage, magnitudes, directions)
    rows = _execute(tasks, workers)
    validation = _validate_pairs(rows, len(tasks))
    write_csv(rows, root / 'data' / f'{stage}.csv')
    _stage_manifest(root, stage, configs, validation=validation)


def canonical(root: Path, configs: dict, workers: int) -> None:
    push = configs['experiments']['push']
    grid(root, configs, 'canonical', [push['magnitude_N']], [push['direction_deg']], workers)


def robustness(root: Path, configs: dict, workers: int) -> None:
    stage_root = root / 'raw' / 'robustness'
    _require_new(stage_root)
    stage_root.mkdir(parents=True)
    robust = configs['experiments']['robustness']
    random_cfg = configs['experiments']['robustness_randomization']
    tasks = []
    for factor, values in (('friction', robust['friction']), ('mass_scale', robust['mass_scale']), ('push_duration_s', robust['push_duration_s'])):
        for value in values:
            for seed in robust['seeds']:
                local = copy.deepcopy(configs)
                friction = float(value) if factor == 'friction' else None
                mass_scale = float(value) if factor == 'mass_scale' else 1.0
                if friction is not None:
                    local['controller']['friction_coefficient'] = friction
                rng = np.random.default_rng(seed)
                base = local['experiments']['push']
                magnitude = float(base['magnitude_N']) * (1.0 + rng.uniform(-float(random_cfg['push_magnitude_jitter_fraction']), float(random_cfg['push_magnitude_jitter_fraction'])))
                direction = float(base['direction_deg']) + rng.normal(0.0, float(random_cfg['push_direction_jitter_deg']))
                duration = float(value) if factor == 'push_duration_s' else float(base['duration_s']) + rng.uniform(-float(random_cfg['push_duration_jitter_s']), float(random_cfg['push_duration_jitter_s']))
                condition_id = f'{factor}_{float(value):g}_seed{int(seed)}'
                condition = prepare_paired_initial_condition(local, seed=int(seed), setup_duration_s=float(local['experiments']['paired_benchmark']['setup_duration_s']), mass_scale=mass_scale, friction_coefficient=friction)
                randomized_qpos, randomized_qvel, perturbation = randomized_initial_state(
                    local, int(seed), mass_scale=mass_scale,
                    friction_coefficient=friction, randomization=random_cfg,
                )
                nominal_model = make_model(
                    local, mass_scale=mass_scale, friction_coefficient=friction,
                )
                condition = replace(
                    condition,
                    qpos=condition.qpos + (randomized_qpos - nominal_model.qpos0),
                    qvel=condition.qvel + randomized_qvel,
                )
                condition_path = root / 'common_states' / 'robustness' / f'{condition_id}.npz'
                save_paired_initial_condition(condition, condition_path)
                for controller in CONTROLLERS:
                    tasks.append({
                        'configs': local, 'condition_path': str(condition_path), 'condition_id': condition_id,
                        'controller': controller, 'magnitude_N': magnitude, 'direction_deg': direction,
                        'duration_s': max(0.03, duration), 'start_time_s': float(base['start_time_s']),
                        'mass_scale': mass_scale, 'friction_coefficient': friction,
                        'trial_id': f'{condition_id}_{controller}', 'trial_root': str(stage_root),
                        'extra': {
                            'factor': factor, 'factor_value': value,
                            **{'initial_' + key: val for key, val in perturbation.items()},
                        },
                    })
    rows = _execute(tasks, workers)
    validation = _validate_pairs(rows, 150)
    write_csv(rows, root / 'data' / 'robustness.csv')
    _stage_manifest(root, 'robustness', configs, validation=validation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'gates', 'canonical', 'calibration', 'sweep', 'robustness'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=max(1, int(os.environ.get('SE3_PAIRED_WORKERS', '8'))))
    args = parser.parse_args()
    configs = load_configs(ROOT)
    if args.stage == 'prepare':
        prepare(args.output_root, configs)
    elif not args.output_root.exists():
        raise FileNotFoundError('run prepare before later stages')
    elif args.stage == 'gates':
        gates(args.output_root, configs)
    elif args.stage == 'canonical':
        canonical(args.output_root, configs, args.workers)
    elif args.stage == 'calibration':
        grid(args.output_root, configs, 'calibration', CALIBRATION_MAGNITUDES_N, CALIBRATION_DIRECTIONS_DEG, args.workers)
    elif args.stage == 'sweep':
        sweep = configs['experiments']['sweep']
        grid(args.output_root, configs, 'sweep', sweep['magnitudes_N'], sweep['directions_deg'], args.workers)
    elif args.stage == 'robustness':
        robustness(args.output_root, configs, args.workers)


if __name__ == '__main__':
    main()
