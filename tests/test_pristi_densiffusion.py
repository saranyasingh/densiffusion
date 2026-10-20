import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.pristi_densiffusion import (  # noqa: E402
    CHUNK,
    densify,
    forecast,
    forecast_layout,
    imputation_windows,
    interpolate,
    normalise,
    refine,
    small_pristi,
    train_forecaster,
    train_imputer,
    window_scale,
)
from densiffusion.training import TrainingConfig  # noqa: E402

TINY = dict(width=8, heads=2, feedforward=8, time_dim=8, channel_dim=4, layers=1)
CONFIG = TrainingConfig(epochs=1, batch_size=8, diffusion_steps=3, beta_end=0.2)


def tiny_model():
    torch.manual_seed(0)
    model = small_pristi(step_dim=8, **TINY)
    torch.nn.init.normal_(model.denoiser.output.weight, std=0.01)
    return model.eval()


def test_imputation_windows_cover_each_stride():
    paths = np.arange(33.0)[None]
    windows = imputation_windows(paths)
    assert windows.shape == (7, 2 * CHUNK - 1)
    np.testing.assert_array_equal(windows[0], np.arange(9))
    np.testing.assert_array_equal(windows[4], np.arange(0, 17, 2))
    np.testing.assert_array_equal(windows[6], np.arange(0, 33, 4))


def test_normalisation_uses_reference_points_only():
    values = np.array([[0.0, 1.0, 5.0, 3.0, 100.0]])
    reference = np.array([True, True, False, True, False])
    location, scale = window_scale(values, reference)
    np.testing.assert_allclose(location, [4 / 3])
    np.testing.assert_allclose(scale, [np.sqrt((1 + 4) / 2)])
    flat = normalise(np.ones((1, 5)), np.ones(5, dtype=bool))
    np.testing.assert_allclose(flat, 0)


def test_refine_keeps_observations_and_never_mixes_chunks():
    model = tiny_model()
    rng = np.random.default_rng(0)
    series = rng.normal(size=(3, 9)).cumsum(axis=1)
    fine = refine(model, CONFIG, series, n_samples=2, seed=5)
    assert fine.shape == (3, 2, 17)
    np.testing.assert_array_equal(fine[:, :, ::2], np.repeat(series[:, None], 2, 1))
    assert np.isfinite(fine).all()
    changed = series.copy()
    changed[:, 5:] += 10
    refined = refine(model, CONFIG, changed, n_samples=2, seed=5)
    np.testing.assert_allclose(refined[:, :, :8], fine[:, :, :8])
    assert not np.allclose(refined[:, :, 9:], fine[:, :, 9:])
    with pytest.raises(ValueError, match="multiple"):
        refine(model, CONFIG, series[:, :8], n_samples=1, seed=0)


def test_densify_levels_double_resolution_and_keep_coarse_points():
    model = tiny_model()
    series = np.random.default_rng(1).normal(size=(2, 5))
    levels = densify(model, CONFIG, series, levels=3, n_samples=3, seed=2)
    assert [level.shape for level in levels] == [(2, 3, 9), (2, 3, 17), (2, 3, 33)]
    for index, level in enumerate(levels, start=1):
        np.testing.assert_array_equal(
            level[:, :, :: 2**index], np.repeat(series[:, None], 3, 1)
        )
    np.testing.assert_array_equal(levels[2][:, :, ::2], levels[1])


def test_interpolate_and_forecast_layout():
    np.testing.assert_allclose(
        interpolate(np.array([[0.0, 4.0, 0.0]]), 4)[0], [0, 1, 2, 3, 4, 3, 2, 1, 0]
    )
    hidden, reference = forecast_layout(3, 2, factor=2)
    np.testing.assert_array_equal(np.flatnonzero(hidden), [5, 6, 7, 8])
    np.testing.assert_array_equal(np.flatnonzero(reference), [0, 2, 4])


def test_forecast_returns_coarse_future_draws():
    model = tiny_model()
    histories = np.random.default_rng(3).normal(size=(4, 9)).cumsum(axis=1)
    samples = forecast(
        model, CONFIG, histories, horizon=2, factor=2, n_samples=3, seed=4
    )
    assert samples.shape == (4, 3, 2)
    assert np.isfinite(samples).all()


def test_training_entry_points_run_and_check_shapes():
    rng = np.random.default_rng(4)
    paths = rng.normal(size=(6, 17)).cumsum(axis=1)
    train_imputer(
        tiny_model(),
        imputation_windows(paths),
        imputation_windows(paths),
        CONFIG,
        seed=0,
    )
    windows = rng.normal(size=(6, 9)).cumsum(axis=1)
    train_forecaster(
        tiny_model(), windows, windows, CONFIG, history=3, horizon=2, factor=2, seed=0
    )
    with pytest.raises(ValueError, match="match"):
        train_forecaster(
            tiny_model(),
            windows,
            windows,
            CONFIG,
            history=5,
            horizon=2,
            factor=1,
            seed=0,
        )
