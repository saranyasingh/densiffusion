"""Masking, filling, and scoring checks for scripts/run_impute_then_forecast.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "run_impute_then_forecast", SCRIPTS / "run_impute_then_forecast.py"
)
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)

CONTEXT, HORIZON, LENGTH = experiment.CONTEXT, experiment.HORIZON, experiment.LENGTH


def test_stride_missing_keeps_every_other_point_ending_at_the_last_past_point():
    hidden = experiment.missing_mask((3, LENGTH, 1), "stride", 0.5, None)
    observed = np.flatnonzero(~hidden[0, :, 0])
    assert observed[observed < CONTEXT][-1] == CONTEXT - 1
    np.testing.assert_array_equal(np.diff(observed), 2)
    scored = experiment.scored_future("stride")
    np.testing.assert_array_equal(scored, ~hidden[0, CONTEXT:, 0])


def test_random_missing_hides_about_the_rate_but_never_the_last_past_point():
    rng = np.random.default_rng(0)
    hidden = experiment.missing_mask((200, LENGTH, 1), "random", 0.3, rng)
    assert hidden.mean() == pytest.approx(0.3, abs=0.02)
    assert not hidden[:, CONTEXT - 1].any()
    assert experiment.scored_future("random").all()


def test_linear_fill_interpolates_and_holds_the_ends():
    values = np.array([[0.0, 9, 2, 9, 6, 9]])[..., None]
    hidden = np.array([[True, False, True, False, False, True]])[..., None]
    values = np.where(hidden, 0, values)
    filled = experiment.linear_fill(values, hidden)
    np.testing.assert_array_equal(filled[0, :, 0], [9, 9, 9, 9, 6, 6])


def test_hide_some_only_hides_observed_points_and_leaves_some_of_each():
    rng = np.random.default_rng(0)
    available = rng.random((500, 12, 1)) < 0.3
    available[:, :2] = True
    extra = experiment.hide_some(available, rng)
    assert not (extra & ~available).any()
    assert extra.any(axis=(1, 2)).all()
    assert (available & ~extra).any(axis=(1, 2)).all()


def test_gaussian_crps_matches_the_sample_estimate():
    rng = np.random.default_rng(0)
    mean, std, target = 0.3, 1.7, -0.4
    draws = mean + std * rng.standard_normal(200_000)
    closed = experiment.gaussian_crps(
        torch.tensor(mean), torch.tensor(std), torch.tensor(target)
    )
    sampled = experiment.crps(draws[None, :, None, None], np.full((1, 1, 1), target))
    assert closed.item() == pytest.approx(sampled, rel=1e-2)


class RecordingSampler:
    """Fill hidden entries with -1 and record what the sampler saw."""

    def __init__(self):
        self.calls = []

    def sample(self, observed, *, factor, n_samples, dt, rng, mask=None, times=None):
        self.calls.append({"observed": observed.copy(), "mask": mask.copy()})
        draws = np.where(mask, -1.0, observed)
        return np.repeat(draws[:, None], n_samples, axis=1)


@pytest.mark.parametrize("arm", ["masked", "linear", "diffusion", "diffusion_aug"])
def test_filled_pasts_never_see_hidden_or_future_values(arm):
    rng = np.random.default_rng(1)
    values = rng.normal(size=(4, LENGTH, 1))
    missing = experiment.missing_mask(values.shape, "random", 0.5, rng)
    missing[:, CONTEXT:] = False  # the future is visible but must not be used
    values = np.where(missing, 100, values)
    values[:, CONTEXT:] = 100
    sampler = RecordingSampler()
    pasts = experiment.filled_pasts(values, missing, arm, sampler, copies=3, seed=0)
    copies = 3 if arm.startswith("diffusion") else 1
    assert pasts.shape == (4, copies, CONTEXT, 1)
    seen = np.broadcast_to(~missing[:, None, :CONTEXT], pasts.shape)
    np.testing.assert_array_equal(
        pasts[seen], np.repeat(values[:, None, :CONTEXT], copies, axis=1)[seen]
    )
    assert (np.abs(pasts[~seen]) < 50).all()  # hidden values (100) never leak
    for call in sampler.calls:
        assert call["observed"].shape[1] == CONTEXT
        assert (call["observed"][call["mask"]] == 0).all()


def test_forecaster_data_pairs_each_filled_past_with_its_own_future():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(2, LENGTH, 1))
    missing = experiment.missing_mask(values.shape, "stride", 0.5, None)
    pasts = rng.normal(size=(2, 3, CONTEXT, 1))
    scored = experiment.scored_future("stride")
    inputs, future, seen = experiment.forecaster_data(
        values, missing, pasts, scored, complete=False
    )
    assert inputs.shape == (6, CONTEXT, 2)
    np.testing.assert_array_equal(inputs[4, :, 0], pasts[1, 1, :, 0])
    np.testing.assert_array_equal(future[4], values[1, CONTEXT:])
    np.testing.assert_array_equal(inputs[4, :, 1], ~missing[1, :CONTEXT, 0])
    np.testing.assert_array_equal(seen[0], scored)
    _, _, everything = experiment.forecaster_data(
        values, missing, pasts, scored, complete=True
    )
    assert everything.all()


def test_gru_forecaster_outputs_a_positive_spread_per_future_step():
    model = experiment.GRUForecaster(hidden=8, layers=1)
    mean, std = model(torch.zeros(5, CONTEXT, 2))
    assert mean.shape == std.shape == (5, HORIZON)
    assert (std > 0).all()
    samples = experiment.gru_forecast(
        model, np.zeros((5, 2, CONTEXT, 2)), n_draws=3, seed=0
    )
    assert samples.shape == (5, 6, HORIZON, 1)


def test_observation_gaps_track_the_last_value_and_steps_since_it():
    values = torch.tensor([[5.0, 0.0, 7.0, 0.0, 0.0, 9.0]])
    observed = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
    last, gap, seen = experiment.observation_gaps(values, observed)
    np.testing.assert_array_equal(seen[0], [False, False, False, True, True, True])
    np.testing.assert_array_equal(last[0, 3:], [7, 7, 7])
    np.testing.assert_array_equal(gap[0], [0, 1, 2, 1, 2, 3])


def test_grud_ignores_values_at_missing_points():
    torch.manual_seed(0)
    model = experiment.GRUDForecaster(0.3, hidden=8)
    past = torch.randn(4, CONTEXT, 2)
    past[..., 1] = (torch.rand(4, CONTEXT) < 0.5).float()
    changed = past.clone()
    changed[..., 0] += 100 * (1 - past[..., 1])
    mean, std = model(past)
    assert mean.shape == std.shape == (4, HORIZON) and (std > 0).all()
    torch.testing.assert_close(model(changed)[0], mean)


def test_hide_future_scores_only_observed_future_points():
    available = np.random.default_rng(0).random((3, LENGTH, 1)) < 0.5
    extra = experiment.hide_future(available)
    assert not extra[:, :CONTEXT].any()
    np.testing.assert_array_equal(extra[:, CONTEXT:], available[:, CONTEXT:])


def test_arms_are_validated():
    assert experiment.parse_arms("grud, sssd_forecast") == ["grud", "sssd_forecast"]
    with pytest.raises(ValueError, match="brits"):
        experiment.parse_arms("linear,brits")
