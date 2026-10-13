import numpy as np
import pytest

from densiffusion.baselines import GPPosteriorSampler
from densiffusion.pipeline import impute


def brownian_posterior(obs_times, obs_values, query_times):
    covariance = np.minimum.outer(obs_times, obs_times)
    cross = np.minimum.outer(query_times, obs_times)
    weights = cross @ np.linalg.pinv(covariance)
    return weights @ obs_values, np.minimum.outer(
        query_times, query_times
    ) - weights @ cross.T


def test_gp_draws_match_brownian_bridge_moments_and_preserve_observations():
    observed = np.array([[[2.0], [4.0]]])
    sampler = GPPosteriorSampler(brownian_posterior)
    kwargs = dict(factor=4, n_samples=16000, times=np.array([1.0, 2.0]), seed=42)
    samples = impute(observed, sampler, **kwargs)
    draws = samples[0, :, 1:-1, 0]
    fractions = np.array([0.25, 0.5, 0.75])
    expected_covariance = np.minimum.outer(fractions, fractions) - np.outer(
        fractions, fractions
    )
    np.testing.assert_allclose(draws.mean(axis=0), 2 + 2 * fractions, atol=0.015)
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False), expected_covariance, atol=0.007
    )
    np.testing.assert_array_equal(
        samples[:, :, ::4], np.repeat(observed[:, None], 16000, axis=1)
    )
    np.testing.assert_array_equal(samples, impute(observed, sampler, **kwargs))


def test_gp_masks_use_original_times_and_condition_each_channel_separately():
    values = np.arange(16, dtype=float).reshape(2, 4, 2)
    times = np.array([[10, 10.1, 11, 15], [20, 20.2, 20.8, 21]])
    mask = np.array(
        [
            [[True, False], [False, True], [False, False], [True, True]],
            [[False, True], [True, False], [False, True], [True, False]],
        ]
    )
    calls = []

    def posterior(obs_times, obs_values, query_times):
        calls.append((obs_times.copy(), obs_values.copy(), query_times.copy()))
        return np.full(len(query_times), 5.0), np.zeros((len(query_times),) * 2)

    observed = np.where(mask, np.nan, values)
    samples = impute(
        observed,
        GPPosteriorSampler(posterior),
        factor=1,
        mask=mask,
        times=times,
        n_samples=3,
    )
    expected = np.repeat(np.where(mask, 5, values)[:, None], 3, axis=1)
    np.testing.assert_array_equal(samples, expected)
    for index, (obs_times, obs_values, query_times) in enumerate(calls):
        series, channel = divmod(index, 2)
        hidden = mask[series, :, channel]
        np.testing.assert_array_equal(obs_times, times[series, ~hidden])
        np.testing.assert_array_equal(obs_values, values[series, ~hidden, channel])
        np.testing.assert_array_equal(query_times, times[series, hidden])


def test_gp_no_hidden_points_skip_posterior_and_randomness():
    def posterior(*args):
        pytest.fail("No posterior call needed")

    values = np.ones((2, 3, 1))
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    samples = GPPosteriorSampler(posterior).sample(
        values, factor=1, n_samples=2, dt=1, rng=rng
    )
    np.testing.assert_array_equal(samples, np.repeat(values[:, None], 2, axis=1))
    assert rng.bit_generator.state == state


@pytest.mark.parametrize(
    "covariance", [np.array([[-1.0]]), np.array([[np.nan]]), np.zeros((2, 2))]
)
def test_gp_rejects_invalid_posterior_covariance(covariance):
    sampler = GPPosteriorSampler(lambda *args: (np.zeros(1), covariance))
    with pytest.raises(ValueError):
        impute(np.ones((1, 2, 1)), sampler)


def test_gp_requires_observations_in_each_imputed_channel():
    mask = np.array([[[False, True], [False, True], [False, True]]])
    with pytest.raises(ValueError, match="per channel"):
        impute(
            np.zeros((1, 3, 2)),
            GPPosteriorSampler(brownian_posterior),
            factor=1,
            mask=mask,
        )


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
@pytest.mark.parametrize("failure", ["negative", "indefinite", "asymmetric"])
def test_gp_rejects_invalid_covariance_at_any_scale(scale, failure):
    matrices = {
        "negative": -np.eye(2),
        "indefinite": np.array([[1.0, 2.0], [2.0, 1.0]]),
        "asymmetric": np.array([[1.0, 0.5], [0.0, 1.0]]),
    }
    sampler = GPPosteriorSampler(lambda *args: (np.zeros(2), scale * matrices[failure]))
    with pytest.raises(ValueError):
        impute(np.ones((1, 2, 1)), sampler, factor=3)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_gp_roundoff_eigenvalues_do_not_create_uncertainty(scale):
    covariance = scale * np.diag([-np.finfo(float).eps / 4, 1.0])
    mean = np.array([2.0, 3.0])
    sampler = GPPosteriorSampler(lambda *args: (mean, covariance))
    samples = impute(np.ones((1, 2, 1)), sampler, factor=3, n_samples=100)
    np.testing.assert_array_equal(samples[0, :, 1, 0], mean[0])
    assert np.ptp(samples[0, :, 2, 0]) > 0


def test_gp_explicit_tolerance_handles_posterior_cancellation():
    mean = np.array([2.0, 3.0])
    covariance = np.diag([-2e-15, 0.0])

    def posterior(*args):
        return mean, covariance

    with pytest.raises(ValueError, match="positive semidefinite"):
        impute(np.ones((1, 2, 1)), GPPosteriorSampler(posterior), factor=3)
    sampler = GPPosteriorSampler(posterior, covariance_atol=3e-15)
    samples = impute(np.ones((1, 2, 1)), sampler, factor=3, n_samples=100)
    np.testing.assert_array_equal(samples[0, :, 1:3, 0], np.tile(mean, (100, 1)))
    covariance[0, 0] = -1e-10
    with pytest.raises(ValueError, match="positive semidefinite"):
        impute(np.ones((1, 2, 1)), sampler, factor=3)


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_gp_rejects_invalid_covariance_tolerance(tolerance):
    with pytest.raises(ValueError, match="covariance_atol"):
        GPPosteriorSampler(brownian_posterior, covariance_atol=tolerance)
