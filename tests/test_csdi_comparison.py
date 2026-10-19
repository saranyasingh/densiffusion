"""Partition separation for the validation-selected training comparison."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "compare_csdi_training",
    Path(__file__).parents[1] / "scripts" / "compare_csdi_training.py",
)
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def test_partition_streams_are_disjoint_before_windowing(monkeypatch):
    calls = []

    def generate(*, n_series, n_steps, dt, rng, **kwargs):
        stream_id = int(rng.integers(10**9))
        calls.append((stream_id, n_series, n_steps, dt))
        # Each row is one tagged independent trajectory; windows retain its ID.
        return np.broadcast_to(
            (stream_id + np.arange(n_series))[:, None, None], (n_series, n_steps, 1)
        ).copy()

    monkeypatch.setattr(comparison, "generate", generate)
    seen = set()
    for name in ("train", "validation", "test"):
        values, times = comparison.partition(name, 3)
        ids = set(values[:, 0, 0])
        assert len(ids) == 3 and not seen & ids
        seen.update(ids)
        assert values.shape == (6, 17, 1)
        np.testing.assert_array_equal(values[::2, 0], values[1::2, 0])
        np.testing.assert_allclose(np.diff(times), 0.1)
        np.testing.assert_allclose(times[:2, 0], [0.0, 1.6])
    assert len({call[0] for call in calls}) == 3
    assert all(call[1:] == (3, 33, 0.1) for call in calls)


def test_paired_interval_resamples_whole_paths():
    # Constant CRPS improvement across independent paths must give a point interval.
    target = np.zeros((6, 17, 1))
    ensembles = {
        "original_csdi": np.full((6, 4, 17, 1), 2.0),
        "improved_csdi": np.full((6, 4, 17, 1), 1.0),
    }
    result = comparison.paired_improvement(ensembles, target, seed=42)
    assert result["independent_test_trajectories"] == 3
    assert result["original_minus_improved"] == 1.0
    assert result["paired_trajectory_bootstrap_95pct"] == [1.0, 1.0]
