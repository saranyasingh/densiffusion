"""Unit tests for FX sparsification / Chronos helper utilities."""

import json
from pathlib import Path

import numpy as np
import pytest

from densiffusion.fx import (
    FX_CURRENCIES,
    block_observation_mask,
    build_observation_mask,
    expand_mask,
    fit_normalizer,
    forecast_metrics,
    linear_fill,
    load_multivariate_dataset,
    point_metrics,
    quantile_loss,
    seasonal_naive_forecast,
    seasonal_scale,
    sparse_observation_mask,
    weighted_quantile_loss,
)


def test_sparse_observation_mask_keeps_endpoints_and_grid():
    hidden = sparse_observation_mask(9, step=2, offset=0)
    assert hidden.tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    hidden4 = sparse_observation_mask(9, step=4, offset=0)
    assert not hidden4[0] and not hidden4[-1]
    assert hidden4[1:4].all()


def test_block_and_dispatch_masks():
    hidden = block_observation_mask(48, block_length=12, observe_length=12)
    assert not hidden[0] and not hidden[-1]
    assert hidden[12:24].all()
    assert not hidden[24:36].any()
    assert build_observation_mask(20, mode="grid", sparse_step=2).sum() > 0
    assert build_observation_mask(40, mode="random", keep_probability=0.3, seed=0).any()


def test_seasonal_metrics_and_naive():
    times = np.arange(48)
    history = np.stack([np.sin(2 * np.pi * times / 24), times / 10], axis=1)
    scale = seasonal_scale(history, season=24)
    assert scale > 0
    pred = seasonal_naive_forecast(history, horizon=12, season=24)
    assert pred.shape == (12, 2)
    assert np.allclose(pred[0], history[-24])
    truth = history[-12:]
    scores = forecast_metrics(
        truth,
        pred,
        scale=scale,
        quantile_forecasts={0.5: pred},
    )
    assert "mase" in scores and "sql" in scores and scores["mase"] >= 0


def test_fit_normalizer_uses_observed_only():
    values = np.arange(24, dtype=np.float64).reshape(1, 8, 3)
    observed = np.zeros_like(values, dtype=bool)
    observed[:, ::2] = True
    normalizer = fit_normalizer(values, observed=observed)
    transformed = normalizer.transform(values)
    known = transformed[:, ::2]
    assert np.allclose(known.mean(axis=(0, 1)), 0.0, atol=1e-7)
    restored = normalizer.inverse(transformed)
    assert np.allclose(restored, values)


def test_linear_fill_recovers_midpoints():
    values = np.linspace(0, 1, 5)[None, :, None]
    hidden = np.array([False, True, False, True, False])
    mask = expand_mask(hidden, batch=1, channels=1)
    filled = linear_fill(np.where(mask, 0.0, values), mask)
    assert np.allclose(filled, values)


def test_point_and_quantile_metrics():
    truth = np.ones((4, 2))
    pred = np.full((4, 2), 1.5)
    scores = point_metrics(truth, pred)
    assert scores["mae"] == pytest.approx(0.5)
    assert scores["rmse"] == pytest.approx(0.5)
    assert quantile_loss(truth, pred, 0.5) == pytest.approx(0.25)
    wql = weighted_quantile_loss(truth, {0.1: pred - 0.1, 0.5: pred, 0.9: pred + 0.1})
    assert wql > 0


def test_load_multivariate_dataset_roundtrip(tmp_path: Path):
    path = tmp_path / "elec.npz"
    names = [f"client_{i:03d}" for i in range(4)]
    values = np.random.default_rng(0).normal(size=(1, 20, 4))
    np.savez_compressed(
        path,
        values=values,
        times=np.arange(20, dtype=np.float64),
        channels=np.asarray(names),
        config=json.dumps(
            {"dt": 1.0, "dataset": "toy", "freq": "h", "start": "2012-01-01T00"}
        ),
    )
    loaded, times, meta = load_multivariate_dataset(path)
    assert loaded.shape == values.shape
    assert times.shape == (20,)
    assert meta["channels"] == names
    assert meta["freq"] == "h"


def test_legacy_currencies_key_still_loads(tmp_path: Path):
    path = tmp_path / "fx.npz"
    values = np.random.default_rng(0).normal(size=(1, 20, len(FX_CURRENCIES)))
    np.savez_compressed(
        path,
        values=values,
        times=np.arange(20, dtype=np.float64),
        currencies=np.asarray(FX_CURRENCIES),
        config=json.dumps({"dt": 1.0, "dataset": "toy", "freq": "D"}),
    )
    _, _, meta = load_multivariate_dataset(path)
    assert meta["channels"] == list(FX_CURRENCIES)
