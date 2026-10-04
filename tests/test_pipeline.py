import numpy as np
import pytest

from densiffusion.pipeline import forecast


class StubDensifier:
    def sample(self, observed, *, factor, n_samples, dt, rng):
        batch, steps, channels = observed.shape
        dense = rng.normal(size=(batch, n_samples, (steps - 1) * factor + 1, channels))
        dense[:, :, ::factor, :] = observed[:, None, :, :]
        return dense


class StubForecaster:
    def __init__(self):
        self.calls = []

    def sample(self, observed, dense_history, *, horizon, n_samples, dt, rng):
        history_index = len(self.calls)
        self.calls.append((observed.copy(), dense_history.copy(), dt))
        return (
            observed[:, None, -1:, :]
            + np.arange(1, horizon + 1)[None, None, :, None] * dt
            + np.arange(n_samples)[None, :, None, None] * 10
            + history_index * 100
        )


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_forecasts_keep_grid_alignment_and_pool_histories(factor):
    observed = np.arange(12, dtype=float).reshape(2, 3, 2)
    forecaster = StubForecaster()
    samples = forecast(
        observed,
        StubDensifier(),
        forecaster,
        horizon=4,
        factor=factor,
        n_histories=2,
        n_futures=3,
        dt=0.5,
    )
    expected = (
        observed[:, None, -1:, :]
        + np.array([0, 10, 20, 100, 110, 120])[None, :, None, None]
        + np.arange(1, 5)[None, None, :, None] * 0.5
    )
    np.testing.assert_allclose(samples, expected)
    for original, dense, dt in forecaster.calls:
        np.testing.assert_array_equal(original, observed)
        np.testing.assert_array_equal(dense[:, ::factor, :], observed)
        assert dt == 0.5 / factor


def test_densifier_cannot_change_observations():
    class BrokenDensifier(StubDensifier):
        def sample(self, observed, **kwargs):
            observed += 1
            return super().sample(observed, **kwargs)

    observed = np.zeros((2, 3, 1))
    forecaster = StubForecaster()
    with pytest.raises(ValueError, match="preserve"):
        forecast(observed, BrokenDensifier(), forecaster, horizon=2)
    np.testing.assert_array_equal(observed, 0)
    assert not forecaster.calls


def test_forecaster_must_return_fine_resolution():
    class BrokenForecaster:
        def sample(self, observed, dense_history, **kwargs):
            return np.zeros((2, 1, 2, 1))

    with pytest.raises(ValueError, match="shape"):
        forecast(np.zeros((2, 3, 1)), StubDensifier(), BrokenForecaster(), horizon=2)


def test_pooling_handles_reused_forecast_buffers():
    class BufferedForecaster:
        def __init__(self):
            self.buffer = np.zeros((1, 1, 4, 1))

        def sample(self, observed, dense_history, **kwargs):
            self.buffer += 1
            return self.buffer

    samples = forecast(
        np.zeros((1, 3, 1)),
        StubDensifier(),
        BufferedForecaster(),
        horizon=2,
        n_histories=2,
    )
    np.testing.assert_array_equal(samples[0, :, :, 0], [[1, 1], [2, 2]])


def test_pipeline_seed_reproducibility_and_separate_stage_streams():
    class RandomForecaster:
        def sample(self, observed, dense_history, *, horizon, n_samples, dt, rng):
            return rng.normal(size=(observed.shape[0], n_samples, horizon, 1))

    class ExtraDrawsDensifier(StubDensifier):
        def sample(self, observed, **kwargs):
            kwargs["rng"].normal(size=100)
            return super().sample(observed, **kwargs)

    def run(densifier, seed):
        return forecast(
            np.zeros((2, 3, 1)), densifier, RandomForecaster(), horizon=3, seed=seed
        )

    baseline = run(StubDensifier(), 42)
    np.testing.assert_array_equal(baseline, run(StubDensifier(), 42))
    np.testing.assert_array_equal(baseline, run(ExtraDrawsDensifier(), 42))
    assert not np.array_equal(baseline, run(StubDensifier(), 43))
