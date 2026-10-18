"""Data separation and observed-only baseline checks for the local example."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "train_csdi_gp", Path(__file__).parents[1] / "scripts" / "train_csdi_gp.py"
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def test_example_splits_trajectories_before_windows_and_keeps_training_grid(
    monkeypatch,
):
    calls = []

    def generate(*, n_series, n_steps, dt, **kwargs):
        calls.append((n_series, n_steps, dt))
        values = 100 * np.arange(n_series)[:, None] + np.arange(n_steps) * dt
        return values[:, :, None]

    monkeypatch.setattr(example, "generate", generate)
    args = example.parse_args(
        [
            "--n-series",
            "10",
            "--n-steps",
            "9",
            "--window-length",
            "5",
            "--stride",
            "2",
            "--dt",
            "0.1",
        ]
    )
    windows, info = example.make_dataset(args)
    assert calls == [(10, 9, 0.1)]  # No finer-resolution truth is generated.
    seen = set()
    for name, (values, times) in zip(
        ("train", "validation", "test"), windows, strict=True
    ):
        ids = set(np.floor(values[:, 0, 0] / 100).astype(int))
        assert not seen & ids
        assert ids == set(info[name]["trajectory_ids"])
        assert len(values) == 3 * len(ids) == info[name]["n_windows"]
        np.testing.assert_allclose(np.diff(times), 0.1)
        np.testing.assert_allclose(times[:3, 0], [0, 0.2, 0.4])
        seen.update(ids)
    assert seen == set(range(10))


def test_example_interpolation_uses_only_observations():
    target = np.array([[[1.0], [200.0], [3.0], [400.0], [5.0]]])
    mask = np.array([[[False], [True], [False], [True], [False]]])
    times = np.array([[10.0, 10.1, 10.2, 10.3, 10.4]])
    samples = example.linear_interpolation(target, mask, times, 3)
    target[mask] = np.nan
    changed = example.linear_interpolation(target, mask, times, 3)
    np.testing.assert_array_equal(samples, changed)
    np.testing.assert_allclose(
        samples, np.broadcast_to(np.arange(1, 6)[None, None, :, None], (1, 3, 5, 1))
    )
