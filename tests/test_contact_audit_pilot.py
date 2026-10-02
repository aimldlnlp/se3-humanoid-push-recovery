from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from contact_audit_pilot import timing_statistics


def test_timing_reports_entire_deadline_distribution_in_milliseconds():
    seconds = np.array([0.001, 0.004, 0.005, 0.010])
    result = timing_statistics(seconds, 0.004)
    assert result['mean_ms'] == 5.0
    assert result['max_ms'] == 10.0
    assert result['deadline_ms'] == 4.0
    assert result['deadline_miss_percent'] == 50.0
    assert result['p95_ms'] <= result['p99_ms'] <= result['max_ms']
    assert timing_statistics([], 0.004) == {}
