"""Per-draw Chronos forecasting helpers in run_csdi_chronos2.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "run_csdi_chronos2",
    Path(__file__).parents[1] / "scripts" / "run_csdi_chronos2.py",
)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def test_hidden_runs():
    hidden = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert pipeline.hidden_runs(hidden) == [(1, 3), (5, 6), (7, 10)]
    assert pipeline.hidden_runs(np.zeros(4, dtype=bool)) == []


def test_quantile_samples_follow_the_quantile_function():
    levels = pipeline.FINE_QUANTILES
    quantiles = {q: np.full((3, 2), 10.0 * q) for q in levels}
    samples = pipeline.quantile_samples(quantiles, 200)
    assert samples.shape == (200, 3, 2)
    assert np.all(np.diff(samples[:, 0, 0]) >= 0)
    np.testing.assert_allclose(np.median(samples, axis=0), 5.0, atol=0.05)
    # Flat beyond the outermost levels.
    assert samples.min() == pytest.approx(10.0 * levels[0])
    assert samples.max() == pytest.approx(10.0 * levels[-1])


def test_quantile_samples_sort_crossed_quantiles():
    quantiles = {0.1: np.array([[2.0]]), 0.5: np.array([[1.0]]), 0.9: np.array([[3.0]])}
    samples = pipeline.quantile_samples(quantiles, 50)
    assert np.all(np.diff(samples[:, 0, 0]) >= 0)


def test_point_forecast_crps_is_mean_absolute_error():
    truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    point = np.array([[1.5, 2.0], [2.0, 4.0]])
    scores = pipeline.add_crps({}, truth, point[None], scale=2.0)
    assert scores["crps"] == pytest.approx(np.abs(point - truth).mean())
    assert scores["scaled_crps"] == pytest.approx(scores["crps"] / 2.0)


@pytest.mark.parametrize("stitch", ["average", "block"])
def test_block_stitching_keeps_each_gap_from_one_window(monkeypatch, stitch):
    window, stride, n_samples = 9, 4, 3

    def fake_impute(observed, sampler, *, mask, n_samples, seed, **kwargs):
        # Every hidden value names its window (seed) and draw (thousands).
        draws = np.arange(n_samples)[:, None, None] * 1000.0 + seed
        return np.broadcast_to(draws, (1, n_samples, *observed.shape[1:])).copy()

    monkeypatch.setattr(pipeline, "impute", fake_impute)
    hidden = np.zeros(21, dtype=bool)
    hidden[[2, 3, 6, 7, 10, 11, 14, 15, 18, 19]] = True
    values = np.full((1, 21, 2), -1.0)
    result = pipeline.impute_series(
        values,
        None,
        hidden_time=hidden,
        window=window,
        stride=stride,
        n_samples=n_samples,
        seed=0,
        stitch=stitch,
    )
    assert result.shape == (n_samples, 21, 2)
    np.testing.assert_array_equal(result[:, ~hidden], -1.0)
    for draw in range(n_samples):
        windows = result[draw, hidden, 0] - 1000.0 * draw
        whole = np.allclose(windows, np.round(windows))
        # Averaging blends windows in overlaps; block keeps whole windows.
        assert whole == (stitch == "block")
        for start, stop in pipeline.hidden_runs(hidden):
            assert len(set(result[draw, start:stop, 0])) == 1


def test_unknown_stitch_is_rejected():
    with pytest.raises(ValueError, match="stitch"):
        pipeline.impute_series(
            np.zeros((1, 9, 1)),
            None,
            hidden_time=np.zeros(9, dtype=bool),
            window=9,
            stride=4,
            n_samples=1,
            seed=0,
            stitch="nearest",
        )


@pytest.mark.parametrize("kind", ["points", "gaps"])
def test_ssl_targets_only_hide_observed_values(kind):
    rng = np.random.default_rng(0)
    available = np.repeat(rng.random((40, 97, 1)) > 0.4, 3, axis=2)
    targets = pipeline.ssl_targets(
        available, kind=kind, rate=0.25, min_gap=2, max_gap=12, seed=0
    )
    assert targets.shape == available.shape
    assert not (targets & ~available).any()
    # Shared across channels.
    np.testing.assert_array_equal(targets[..., 0], targets[..., 2])
    share = targets.sum() / available.sum()
    assert 0.1 < share < 0.4


def test_gap_targets_come_in_runs():
    available = np.ones((200, 97, 1), dtype=bool)
    targets = pipeline.ssl_targets(
        available, kind="gaps", rate=0.25, min_gap=4, max_gap=8, seed=1
    )
    assert targets.mean() == pytest.approx(0.25, abs=0.05)
    lengths = [
        stop - start
        for window in targets[..., 0]
        for start, stop in pipeline.hidden_runs(window)[1:-1]
    ]
    assert min(lengths) >= 4 and max(lengths) <= 8


def test_filter_ssl_windows_needs_targets_and_conditioning():
    values = np.arange(3.0)[:, None, None] * np.ones((3, 4, 1))
    available = np.ones((3, 4, 1), dtype=bool)
    targets = np.zeros((3, 4, 1), dtype=bool)
    targets[1, 1] = True
    targets[2] = True  # nothing left to condition on
    kept_values, _, kept_targets = pipeline.filter_ssl_windows(
        values, available, targets
    )
    assert kept_values[:, 0, 0].tolist() == [1.0]
    assert kept_targets.sum() == 1


def test_extra_targets_hide_one_short_observed_run_per_window():
    rng = np.random.default_rng(2)
    available = np.repeat(rng.random((300, 97, 1)) > 0.3, 2, axis=2)
    targets = pipeline.ssl_targets(
        available, kind="extra", rate=0.25, min_gap=2, max_gap=12, seed=0
    )
    assert not (targets & ~available).any()
    lengths = set()
    for window in targets[..., 0]:
        runs = pipeline.hidden_runs(window)
        assert len(runs) <= 1
        for start, stop in runs:
            assert 0 < start and stop < 97
            lengths.add(stop - start)
    assert lengths == {1, 2}
    # Nearly every window gets a target, and the windows stay almost as sparse.
    assert (targets.any(axis=(1, 2))).mean() > 0.95
    assert targets.sum() / available.sum() < 0.03
