from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_paired_benchmark import plot_basin
from render_com_support_animation import _nearest_indices


def test_production_basin_labels_round_angular_float_noise(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, 'close', lambda fig: figures.append(fig))
    rows = [{'controller': c, 'push_magnitude_N': 20,
             'push_direction_deg': float(d) - 1e-12, 'success': True}
            for c in ('pure_pd', 'pd_nominal_ff', 'se3_wbc') for d in range(0, 360, 15)]
    plot_basin(rows, tmp_path)
    assert [t.get_text() for t in figures[-1].axes[0].get_xticklabels()] == ['0', '60', '120', '180', '240', '300']


def test_production_timestamp_selection_preserves_six_second_motion():
    times = np.arange(0, 6, .004)
    frame_times = np.arange(180) / 30
    indices = _nearest_indices(times, frame_times)
    assert len(indices) == len(np.unique(indices)) == 180
    assert np.max(np.abs(times[indices] - frame_times)) <= .002 + 1e-12
