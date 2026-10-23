import numpy as np
import pytest

from densiffusion.data import make_forecast_windows, split_data
from densiffusion.pipeline import forecast


class StubDensifier:
    def __init__(self, extra_draws=0):
        self.calls = []
        self.extra_draws = extra_draws

    def sample(self, observed, *, factor, n_samples, dt, rng):
        self.calls.append((observed.copy(), factor, n_samples, dt))
        batch, length, channels = observed.shape
        histories = rng.normal(
            size=(batch, n_samples, (length - 1) * factor + 1, channels)
        )
        histories[:, :, ::factor] = observed[:, None]
        rng.normal(size=self.extra_draws)
        return histories


class StubForecaster:
    def __init__(self):
        self.calls = []

    def sample(self, observed, dense_history, *, horizon, n_samples, dt, rng):
        self.calls.append(
            (observed.copy(), dense_history.copy(), horizon, n_samples, dt)
        )
        return rng.normal(size=(len(observed), n_samples, horizon, observed.shape[2]))


def test_forecast_windows_preserve_series_order_and_causal_boundary():
    values = np.arange(2 * 9 * 2).reshape(2, 9, 2)
    histories, futures = make_forecast_windows(
        values, context_length=3, horizon=2, stride=2
    )
    expected = np.stack(
        [series[start : start + 5] for series in values for start in (0, 2, 4)]
    )
    np.testing.assert_array_equal(histories, expected[:, :3])
    np.testing.assert_array_equal(futures, expected[:, 3:])
    assert not np.shares_memory(histories, values)
    assert not np.shares_memory(futures, values)
    assert not np.shares_memory(histories, futures)
    histories[0, 2] = -1
    np.testing.assert_array_equal(histories[1, 0], values[0, 2])


def test_forecast_windows_remain_inside_temporal_splits():
    values = np.arange(40).reshape(1, 40, 1)
    parts = split_data(
        values, train_fraction=0.5, validation_fraction=0.25, axis="time"
    )
    for part in parts:
        history, future = make_forecast_windows(part, context_length=3, horizon=2)
        assert history.min() == part.min()
        assert future.max() == part.max()
        np.testing.assert_array_equal(future[:, 0], history[:, -1] + 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_length": 0, "horizon": 1},
        {"context_length": 1.5, "horizon": 1},
        {"context_length": 1, "horizon": 0},
        {"context_length": 1, "horizon": True},
        {"context_length": 3, "horizon": 4},
        {"context_length": 2, "horizon": 2, "stride": 0},
    ],
)
def test_forecast_windows_reject_invalid_lengths(kwargs):
    with pytest.raises(ValueError):
        make_forecast_windows(np.zeros((2, 6, 1)), **kwargs)


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_forecast_preserves_history_conditioning_and_future_grid(factor):
    observed = np.arange(12).reshape(2, 3, 2)
    densifier = StubDensifier()

    class GridForecaster(StubForecaster):
        def sample(self, observed, dense_history, **kwargs):
            super().sample(observed, dense_history, **kwargs)
            horizon, n_samples, dt = (
                kwargs["horizon"],
                kwargs["n_samples"],
                kwargs["dt"],
            )
            return np.broadcast_to(
                observed[:, None, -1:, :]
                + np.arange(1, horizon + 1)[None, None, :, None] * dt,
                (len(observed), n_samples, horizon, observed.shape[2]),
            ).copy()

    forecaster = GridForecaster()
    samples = forecast(
        observed,
        densifier,
        forecaster,
        horizon=4,
        factor=factor,
        n_histories=2,
        n_futures=3,
        dt=0.75,
    )
    assert samples.shape == (2, 6, 4, 2)
    expected = observed[:, None, -1:, :] + np.arange(1, 5)[None, None, :, None] * 0.75
    np.testing.assert_array_equal(samples, np.broadcast_to(expected, samples.shape))
    coarse, dense, horizon, n_samples, dt = forecaster.calls[0]
    np.testing.assert_array_equal(coarse, np.repeat(observed, 2, axis=0))
    np.testing.assert_array_equal(dense[:, ::factor], coarse)
    assert dense.shape == (4, 2 * factor + 1, 2)
    assert (horizon, n_samples, dt) == (4 * factor, 3, 0.75 / factor)
    if factor == 1:
        assert densifier.calls == []
    else:
        assert len(densifier.calls) == 1
        np.testing.assert_array_equal(densifier.calls[0][0], observed)
        assert densifier.calls[0][1:] == (factor, 2, 0.75)


def test_forecast_pools_each_histories_future_samples_in_order():
    class HistoryDensifier(StubDensifier):
        def sample(self, observed, **kwargs):
            histories = super().sample(observed, **kwargs)
            histories[:, :, 1] = np.arange(kwargs["n_samples"])[None, :, None]
            return histories

    class HistoryForecaster:
        def sample(self, observed, dense_history, *, horizon, n_samples, **kwargs):
            values = (
                observed[:, None, -1:, :] * 100
                + dense_history[:, None, 1:2, :] * 10
                + np.arange(n_samples)[None, :, None, None]
            )
            return np.broadcast_to(values, (len(observed), n_samples, horizon, 1))

    result = forecast(
        np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
        HistoryDensifier(),
        HistoryForecaster(),
        horizon=2,
        n_histories=2,
        n_futures=3,
    )
    np.testing.assert_array_equal(result[0, :, 0, 0], [200, 201, 202, 210, 211, 212])
    np.testing.assert_array_equal(result[1, :, 0, 0], [400, 401, 402, 410, 411, 412])


def test_forecast_rng_is_reproducible_and_independent_of_densifier_consumption():
    observed = np.zeros((2, 3, 1))
    kwargs = dict(horizon=3, n_histories=2, n_futures=4, seed=42)
    baseline = forecast(observed, StubDensifier(), StubForecaster(), **kwargs)
    repeated = forecast(observed, StubDensifier(), StubForecaster(), **kwargs)
    extra = forecast(
        observed, StubDensifier(extra_draws=100), StubForecaster(), **kwargs
    )
    changed = forecast(
        observed, StubDensifier(), StubForecaster(), **(kwargs | {"seed": 43})
    )
    np.testing.assert_array_equal(baseline, repeated)
    np.testing.assert_array_equal(baseline, extra)
    assert not np.array_equal(baseline, changed)


def test_forecast_protects_inputs_and_copies_reused_output_buffers():
    class MutatingForecaster:
        def __init__(self):
            self.buffer = np.zeros((1, 1, 4, 1))

        def sample(self, observed, dense_history, **kwargs):
            observed += 7
            dense_history += 9
            self.buffer += 1
            return self.buffer

    observed = np.zeros((1, 3, 1))
    forecaster = MutatingForecaster()
    first = forecast(observed, StubDensifier(), forecaster, horizon=2)
    second = forecast(observed, StubDensifier(), forecaster, horizon=2)
    np.testing.assert_array_equal(observed, 0)
    np.testing.assert_array_equal(first, 1)
    np.testing.assert_array_equal(second, 2)


@pytest.mark.parametrize(
    "changes",
    [
        {"horizon": 0},
        {"factor": 1.5},
        {"n_histories": 0},
        {"n_futures": True},
        {"dt": np.nan},
        {"dt": 0},
        {"seed": -1},
        {"seed": True},
        {"dt": np.nextafter(0, 1), "factor": 2},
    ],
)
def test_forecast_rejects_invalid_parameters_before_sampling(changes):
    densifier, forecaster = StubDensifier(), StubForecaster()
    with pytest.raises(ValueError):
        forecast(
            np.zeros((1, 3, 1)), densifier, forecaster, **({"horizon": 2} | changes)
        )
    assert densifier.calls == []
    assert forecaster.calls == []


@pytest.mark.parametrize(
    "observed", [np.zeros((2, 3)), np.full((2, 3, 1), np.nan), np.zeros((0, 3, 1))]
)
def test_forecast_rejects_invalid_histories(observed):
    with pytest.raises(ValueError):
        forecast(observed, StubDensifier(), StubForecaster(), horizon=2)


@pytest.mark.parametrize(
    "invalid", [np.zeros((2, 1, 2, 1)), np.full((2, 1, 4, 1), np.nan)]
)
def test_forecast_rejects_invalid_forecaster_outputs(invalid):
    class BrokenForecaster:
        def sample(self, *args, **kwargs):
            return invalid

    with pytest.raises(ValueError):
        forecast(np.zeros((2, 3, 1)), StubDensifier(), BrokenForecaster(), horizon=2)
