from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_contact_residuals import friction_audit
from package_paired_benchmark import limited_conditions


def test_force_scaled_budget_keeps_raw_violation_and_rejects_outer_square():
    result = friction_audit([[0.0118283334, -14.1688675, 20.2437305, 0, 0, 0]], .7, .001, .001)
    assert result['row_budget_pass']
    assert not result['strict_1mN_pass']
    assert result['max_l1_excess_N'] == pytest.approx(.0100844834)
    assert result['max_coulomb_excess_N'] == 0
    assert not friction_audit([[7, 7, 10, 0, 0, 0]], .7, .001, .001)['row_budget_pass']
    assert not friction_audit([[0, 0, -1, 0, 0, 0]], .7, .001, .001)['row_budget_pass']
    with pytest.raises(ValueError, match='finite'):
        friction_audit([[np.nan, 0, 10, 0, 0, 0]], .7, .001, .001)


def test_trajectory_curation_is_deterministic_and_retains_grid_endpoints():
    ids = {f'{i:03d}' for i in range(100)}
    selected = limited_conditions(ids, 8)
    assert len(selected) == 8
    assert {'000', '099'} <= selected
    assert selected == limited_conditions(set(reversed(sorted(ids))), 8)
    assert limited_conditions({'only'}, 8) == {'only'}
    with pytest.raises(ValueError):
        limited_conditions(ids, 0)
