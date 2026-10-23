"""Original-grid data separation and reproducible scores for the forecast example."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.metrics import evaluate_forecast  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_csdi_forecast",
    Path(__file__).parents[1] / "scripts" / "train_csdi_forecast.py",
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def test_forecast_example_splits_paths_before_windows_and_generates_only_coarse_truth(
    monkeypatch,
):
    calls = []

    def generate(*, n_series, n_steps, dt, **kwargs):
        calls.append((n_series, n_steps, dt))
        return (100 * np.arange(n_series)[:, None] + np.arange(n_steps) * dt)[
            :, :, None
        ]

    monkeypatch.setattr(example, "generate", generate)
    args = example.parse_args(
        [
            "--n-series",
            "10",
            "--n-steps",
            "9",
            "--context-length",
            "3",
            "--horizon",
            "2",
            "--stride",
            "2",
            "--dt",
            "0.1",
            "--factor",
            "4",
        ]
    )
    windows, info = example.make_dataset(args)
    assert calls == [(10, 9, 0.1)]
    seen = set()
    for name, (history, future) in zip(
        ("train", "validation", "test"), windows, strict=True
    ):
        ids = set(np.floor(history[:, 0, 0] / 100).astype(int))
        assert not seen & ids
        assert ids == set(info[name]["trajectory_ids"])
        assert len(history) == 3 * len(ids) == info[name]["n_windows"]
        np.testing.assert_allclose(future[:, 0] - history[:, -1], args.dt)
        np.testing.assert_allclose(np.diff(history, axis=1), args.dt)
        assert set(np.floor(future[:, 0, 0] / 100).astype(int)) == ids
        seen.update(ids)
    assert seen == set(range(10))


def test_forecast_example_saves_exact_scored_ensembles(tmp_path):
    args = example.parse_args(
        [
            "--output",
            str(tmp_path / "forecast"),
            "--epochs",
            "1",
            "--densifier-epochs",
            "1",
            "--diffusion-steps",
            "2",
            "--n-series",
            "10",
            "--n-steps",
            "5",
            "--context-length",
            "3",
            "--horizon",
            "2",
            "--stride",
            "4",
            "--n-histories",
            "2",
            "--n-futures",
            "2",
            "--batch-size",
            "8",
        ]
    )
    scores = example.run_example(args)
    assert scores == json.loads((args.output / "scores.json").read_text())
    manifest = json.loads((args.output / "config.json").read_text())
    assert not manifest["finer_ground_truth_generated"]
    assert not manifest["test_targets_used_for_training_or_selection"]
    assert manifest["split_axis"] == "series_before_windowing"
    timings = json.loads((args.output / "timings.json").read_text())
    assert timings["densifier_training_seconds"] >= 0
    with np.load(args.output / "samples.npz", allow_pickle=False) as saved:
        for method in example.METHODS:
            samples = saved[method]
            assert samples.shape == (2, 4, 2, 1)
            assert np.isfinite(samples).all()
            assert scores[method] == evaluate_forecast(samples, saved["target"])
            assert timings["methods"][method]["forecast_seconds"] >= 0
            assert (args.output / f"{method}_forecaster.pt").is_file()
        np.testing.assert_allclose(
            saved["future_times"][:, 0] - saved["history_times"][:, -1], args.dt
        )
    assert (args.output / "densifier.pt").is_file()
    with pytest.raises(FileExistsError):
        example.run_example(args)
