from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('mujoco')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))

from common import (
    load_configs,
    load_paired_initial_condition,
    make_model,
    make_push,
    prepare_paired_initial_condition,
    run_controller,
    run_trial,
    save_paired_initial_condition,
)


def _configs():
    return load_configs(ROOT, robot_name='unitree_g1')


def test_pd_controller_names_preserve_historical_alias_and_expose_pure_pd():
    configs = _configs()
    pure = run_controller('pure_pd', make_model(configs), configs)
    historical = run_controller('pd', make_model(configs), configs)
    explicit = run_controller('pd_nominal_ff', make_model(configs), configs)
    np.testing.assert_array_equal(pure.feedforward_control, np.zeros_like(pure.feedforward_control))
    np.testing.assert_allclose(historical.feedforward_control, explicit.feedforward_control, atol=0.0, rtol=0.0)
    assert np.linalg.norm(explicit.feedforward_control) > 0.0


def test_paired_initial_condition_roundtrip_and_conflict_guard(tmp_path):
    configs = _configs()
    condition = prepare_paired_initial_condition(configs, seed=3, setup_duration_s=0.02)
    path = tmp_path / 'condition.npz'
    save_paired_initial_condition(condition, path)
    restored = load_paired_initial_condition(path)
    assert restored.sha256 == condition.sha256
    np.testing.assert_array_equal(restored.qpos, condition.qpos)
    np.testing.assert_array_equal(restored.qvel, condition.qvel)
    with pytest.raises(ValueError, match='cannot be combined'):
        run_trial('pure_pd', configs, duration=0.01, initial_condition=condition, initial_qpos=condition.qpos)


def test_paired_controllers_share_state_force_trace_and_exact_impulse():
    configs = _configs()
    condition = prepare_paired_initial_condition(configs, setup_duration_s=0.02)
    push = make_push(configs)
    runs = []
    for controller in ('pure_pd', 'pd_nominal_ff', 'se3_wbc'):
        _, run = run_trial(controller, configs, push=push, duration=2.16, initial_condition=condition)
        runs.append(run)
        np.testing.assert_array_equal(run.qpos_history[0], condition.qpos)
        np.testing.assert_array_equal(run.qvel_history[0], condition.qvel)
        assert run.metadata['initial_condition_sha256'] == condition.sha256
        assert run.metadata['active_push_substeps'] == 75
        assert np.isclose(run.metadata['realized_push_duration_s'], 0.15, atol=1e-12)
        assert np.isclose(run.metadata['realized_impulse_Ns'], 10.5, atol=1e-12)
    assert len({run.metadata['force_trace_sha256'] for run in runs}) == 1


def test_trial_model_mismatch_keeps_nominal_controller_and_shared_initial_state():
    configs = _configs()
    condition = prepare_paired_initial_condition(configs, setup_duration_s=0.02)
    plant, run = run_trial(
        'se3_wbc', configs, duration=0.012, initial_condition=condition,
        mass_scale=1.1, friction_coefficient=0.4,
        controller_mass_scale=1.0, controller_friction_coefficient=0.7,
    )
    assert run.metadata['separate_internal_model']
    assert run.metadata['plant_mass_scale'] == 1.1
    assert run.metadata['controller_mass_scale'] == 1.0
    assert run.metadata['plant_friction_coefficient'] == 0.4
    assert run.metadata['controller_friction_coefficient'] == 0.7
    np.testing.assert_array_equal(run.qpos_history[0], condition.qpos)
    assert np.all(np.isfinite(plant.data.qpos))
    with pytest.raises(ValueError, match='only for se3_wbc'):
        run_trial('pure_pd', configs, controller_mass_scale=1.0)


def test_mismatch_stage_builds_five_seed_shared_knowledge_pairs(tmp_path, monkeypatch):
    import paired_benchmark
    configs = _configs()
    condition = prepare_paired_initial_condition(configs, setup_duration_s=0.02)
    save_paired_initial_condition(condition, tmp_path / 'common_states' / 'nominal.npz')
    captured = []

    def inspect_tasks(tasks, workers):
        captured.extend(tasks)
        rows = []
        for task in tasks:
            initial = load_paired_initial_condition(Path(task['condition_path']))
            rows.append({'condition_id': task['condition_id'], 'trial_id': task['trial_id'],
                         'initial_condition_sha256': initial.sha256,
                         'force_trace_sha256': str((task['magnitude_N'], task['direction_deg'])),
                         'realized_impulse_Ns': task['magnitude_N'] * task['duration_s'],
                         **task['extra']})
        return rows

    monkeypatch.setattr(paired_benchmark, '_execute', inspect_tasks)
    paired_benchmark.mismatch(tmp_path, configs, 1)
    assert len(captured) == 80
    hashes = {load_paired_initial_condition(Path(t['condition_path'])).sha256 for t in captured}
    assert len(hashes) == 5
    for task in captured:
        unknown = task['extra']['model_knowledge'] == 'unknown'
        assert task['controller_mass_scale'] == (1.0 if unknown else None)
        assert task['controller_friction_coefficient'] == (0.7 if unknown else None)
        assert not task['configs']['controller']['solver'].get('reuse_workspace', False)
