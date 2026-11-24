"""Per-customer chunk pipeline (scripts/run_chronos_chunks.py)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "run_chronos_chunks",
    Path(__file__).parents[1] / "scripts" / "run_chronos_chunks.py",
)
chunks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chunks)


def test_client_split_is_disjoint_and_fixed():
    train, held = chunks.split_clients(50, 0.2, seed=0)
    assert len(held) == 10 and len(train) == 40
    assert not set(train) & set(held)
    np.testing.assert_array_equal(held, chunks.split_clients(50, 0.2, seed=0)[1])


def test_random_gaps_vary_and_leave_enough_observed():
    rng = np.random.default_rng(0)
    masks = np.stack([chunks.random_gaps(rng) for _ in range(500)])
    assert masks.shape == (500, chunks.HISTORY)
    assert (~masks).sum(axis=1).min() >= 24
    assert masks.any(axis=1).all()
    fractions = masks.mean(axis=1)
    assert fractions.min() < 0.05 and fractions.max() > 0.5


def test_heavy_gaps_hide_50_to_80_percent_in_long_runs():
    rng = np.random.default_rng(0)
    masks = np.stack([chunks.heavy_gaps(rng) for _ in range(300)])
    fractions = masks.mean(axis=1)
    assert fractions.min() >= 0.5 and fractions.max() <= 0.8
    assert fractions.max() - fractions.min() > 0.2
    edges = np.diff(masks.astype(int), axis=1)
    starts = np.argwhere(edges == 1)
    # every interior gap (not cut by either end) lasts at least 6 hours
    for row, col in starts:
        rest = masks[row, col + 1 :]
        if not rest.all():
            assert np.argmin(rest) >= 6


def test_daily_gaps_end_on_an_observed_hour_from_noon():
    hidden = chunks.daily_gaps(12)
    assert hidden[:12].all() and not hidden[12:24].any()
    assert not hidden[-1]
    assert hidden.mean() == pytest.approx(0.5)


def test_sample_chunks_are_aligned_inside_the_period_and_usable():
    values = np.random.default_rng(1).random((24 * 60, 5)) + 1
    values[:, 2] = 0.0  # a customer with no readings is never picked
    rng = np.random.default_rng(0)
    picked = chunks.sample_chunks(values, np.arange(5), 100, 24 * 50, 40, rng)
    assert (picked[:, 1] % 24 == 0).all()
    assert (picked[:, 1] >= 100).all()
    assert (picked[:, 1] + chunks.HISTORY + chunks.HORIZON <= 24 * 50).all()
    assert 2 not in picked[:, 0]
    assert len({tuple(p) for p in picked}) == 40


def test_scaling_ignores_hidden_values():
    rng = np.random.default_rng(2)
    history = rng.normal(size=(3, chunks.HISTORY))
    hidden = np.stack([chunks.random_gaps(rng) for _ in range(3)])
    changed = np.where(hidden, 1e6, history)
    for a, b in zip(
        chunks.observed_stats(history, hidden),
        chunks.observed_stats(changed, hidden),
        strict=True,
    ):
        np.testing.assert_allclose(a, b)


def test_self_supervised_inputs_never_contain_hidden_values():
    rng = np.random.default_rng(3)
    history = rng.normal(size=(20, chunks.HISTORY)) + 5
    hidden = np.stack([chunks.random_gaps(rng) for _ in range(20)])
    values, targets, available = chunks.training_arrays(history, hidden, "selfsup", 0)
    assert (values[~available] == 0).all()
    assert not (targets & ~available).any()
    runs = targets[..., 0].sum(axis=1)
    assert set(runs) <= {1, 2}
    supervised = chunks.training_arrays(history, hidden, "supervised", 0)
    np.testing.assert_array_equal(supervised[1][..., 0], hidden)
    assert supervised[2] is None


def test_ensemble_crps_matches_the_shared_metric():
    from densiffusion.metrics import crps

    rng = np.random.default_rng(4)
    samples = rng.normal(size=(30, 5, 24))
    truth = rng.normal(size=(5, 24))
    per_chunk = chunks.ensemble_crps(samples, truth)
    assert per_chunk.shape == (5,)
    expected = crps(np.moveaxis(samples, 0, 1)[..., None], truth[..., None])
    assert per_chunk.mean() == pytest.approx(expected)


def test_seasonal_naive_uses_the_latest_observed_same_hour():
    history = np.arange(chunks.HISTORY, dtype=float)[None]
    hidden = np.zeros((1, chunks.HISTORY), dtype=bool)
    hidden[0, -24:-20] = True  # yesterday's first four hours are missing
    forecast = chunks.seasonal_naive(history, hidden)
    assert forecast[0, 0] == chunks.HISTORY - 48
    assert forecast[0, 4] == chunks.HISTORY - 20


def test_linear_fill_draws_straight_lines():
    history = np.arange(chunks.HISTORY, dtype=float)[None] * 2
    hidden = np.zeros((1, chunks.HISTORY), dtype=bool)
    hidden[0, 10:20] = True
    filled = chunks.linear_fill(np.where(hidden, 0.0, history), hidden)
    np.testing.assert_allclose(filled, history)


class FakeChronos:
    """Quantile forecasts around the last finite context value."""

    def predict_df(self, frame, *, prediction_length, quantile_levels, **kwargs):
        import pandas as pd

        rows = []
        for item, group in frame.groupby("item_id", sort=False):
            values = group["target"].to_numpy()
            last = values[np.isfinite(values)][-1]
            stamps = pd.date_range(
                group["timestamp"].iloc[-1], periods=prediction_length + 1, freq="h"
            )[1:]
            for stamp in stamps:
                row = {"item_id": item, "timestamp": stamp, "target_name": "target"}
                row.update({str(q): last + (q - 0.5) for q in quantile_levels})
                rows.append(row)
        return pd.DataFrame(rows)


def test_end_to_end_with_a_stand_in_forecaster(tmp_path, monkeypatch):
    pytest.importorskip("pandas")  # in the chronos extra, not the research one
    rng = np.random.default_rng(5)
    hours = 24 * 120
    daily = np.sin(2 * np.pi * np.arange(hours) / 24)[:, None]
    values = 5 + daily * rng.uniform(1, 2, size=10) + 0.1 * rng.normal(size=(hours, 10))
    data = tmp_path / "elec.npz"
    np.savez_compressed(
        data,
        values=values[None],
        times=np.arange(hours, dtype=float),
        channels=np.asarray([f"c{i}" for i in range(10)]),
        config=json.dumps({"freq": "h", "start": "2012-01-01T00"}),
    )
    monkeypatch.setattr(chunks, "load_chronos", lambda *a, **k: FakeChronos())
    small = [
        "--data", str(data), "--train-chunks", "40", "--val-chunks", "8",
        "--test-chunks", "6", "--draws", "2", "--epochs", "1",
        "--hidden-channels", "8", "--residual-layers", "1", "--attention-heads", "1",
        "--embedding-dim", "8", "--holdout", "0.3",
    ]  # fmt: skip
    for protocol in ("selfsup", "supervised"):
        output = tmp_path / "runs" / f"random_{protocol}" / "seed0"
        results = chunks.run(
            chunks.parse_args([*small, "--output", str(output), "--protocol", protocol])
        )
        for group in ("seen", "held_out"):
            summary = results["summary"][group]
            for arm in (*chunks.ARMS, "seasonal_naive"):
                assert np.isfinite(summary[f"crps_{arm}"])
    chunks.compare(chunks.parse_args(["compare", "--output", str(tmp_path / "runs")]))
    text = (tmp_path / "runs" / "comparison.md").read_text()
    assert "C3 self-supervised beats supervised CSDI" in text
    assert "held-out customers" in text
