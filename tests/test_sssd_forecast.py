"""Grid alignment and leakage checks for scripts/run_sssd_forecast.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "run_sssd_forecast", Path(__file__).parents[1] / "scripts" / "run_sssd_forecast.py"
)
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


class RecordingSampler:
    """Fill hidden entries with their fine-grid index; record every call."""

    def __init__(self):
        self.calls = []

    def sample(self, observed, *, factor, n_samples, dt, rng, mask=None, times=None):
        self.calls.append({"observed": observed.copy(), "dt": dt, "mask": mask})
        batch, steps, channels = observed.shape
        length = (steps - 1) * factor + 1
        index = np.arange(length, dtype=float)[None, None, :, None]
        draws = np.broadcast_to(index, (batch, n_samples, length, channels)).copy()
        if mask is None:
            draws[:, :, ::factor] = observed[:, None]
        else:
            draws = np.where(mask[:, None], draws, observed[:, None])
        return draws


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_future_mask_hides_everything_after_the_last_past_point(factor):
    mask = experiment.future_mask(context=5, horizon=3, factor=factor)
    assert mask.shape == ((5 + 3 - 1) * factor + 1,)
    assert not mask[: 4 * factor + 1].any()
    assert mask[4 * factor + 1 :].all()
    assert mask.sum() == 3 * factor


def test_densify_halves_spacing_each_level_and_keeps_observations():
    values = np.random.default_rng(0).normal(size=(2, 5, 1))
    sampler = RecordingSampler()
    dense = experiment.densify(values, sampler, levels=2, n_samples=3, dt=0.1, seed=0)
    assert dense.shape == (2, 3, 17, 1)
    np.testing.assert_array_equal(dense[:, :, ::4], np.repeat(values[:, None], 3, 1))
    assert [call["dt"] for call in sampler.calls] == [0.1, 0.05]
    # The second level refines each first-level draw once.
    assert sampler.calls[1]["observed"].shape == (6, 9, 1)


@pytest.mark.parametrize("levels", [0, 1, 2])
def test_forecast_sees_only_the_past_and_returns_the_original_grid(levels):
    past = np.random.default_rng(1).normal(size=(2, 5, 1)) + 10
    densifier, forecaster = RecordingSampler(), RecordingSampler()
    samples = experiment.forecast(
        past,
        densifier,
        forecaster,
        horizon=3,
        levels=levels,
        n_histories=2,
        n_futures=4,
        dt=0.1,
        seed=0,
    )
    factor = 2**levels
    histories = 2 if levels else 1
    assert samples.shape == (2, histories * 4, 3, 1)
    for call in densifier.calls:  # the densifier only ever receives past values
        assert call["observed"].shape[1] <= (5 - 1) * factor // 2 + 1
    (call,) = forecaster.calls
    window = (5 + 3 - 1) * factor + 1
    assert call["observed"].shape == (2 * histories, window, 1)
    assert call["dt"] == pytest.approx(0.1 / factor)
    np.testing.assert_array_equal(
        call["mask"][0, :, 0], experiment.future_mask(5, 3, factor)
    )
    assert (call["observed"][:, (5 - 1) * factor + 1 :] == 0).all()
    # Forecasts are read back at the original-grid future times only.
    expected = np.array([(5 - 1 + k) * factor for k in (1, 2, 3)], dtype=float)
    np.testing.assert_array_equal(samples[0, 0, :, 0], expected)


def test_training_windows_are_densified_with_measured_values_kept():
    values = np.random.default_rng(2).normal(size=(3, 8, 1))
    dense = experiment.dense_training_windows(
        values, RecordingSampler(), levels=1, dt=0.1, seed=0
    )
    assert dense.shape == (3, 15, 1)
    np.testing.assert_array_equal(dense[:, ::2], values)
