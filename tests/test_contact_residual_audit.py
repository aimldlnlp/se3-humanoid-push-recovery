from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_contact_residuals import friction_audit


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
