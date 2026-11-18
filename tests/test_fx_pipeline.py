"""Unit tests for FX sparsification / Chronos helper utilities."""

import json
from pathlib import Path

import numpy as np
import pytest

from densiffusion.fx import (
    FX_CURRENCIES,
    expand_mask,
    fit_normalizer,
    linear_fill,
    load_multivariate_dataset,
    point_metrics,
    quantile_loss,
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
