import numpy as np
import pytest

from densiffusion.data import GenerationConfig, generate
from densiffusion.gp.kernels import exponential
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import sample_gp

PARAMS = {"variance": 2.0, "lengthscale": 1.5}


def brownian(t, s):
    # Var = t, so the value at t = 0 is known exactly: a zero row in K.
    return np.minimum.outer(t, s)


def test_noiseless_posterior_interpolates_observations():
    times = np.array([1.5, 0.0, 3.0, 0.5])
    values = sample_gp(times, "exponential", rng=0, mean=1.0, **PARAMS)[0]
    mean, cov = gp_posterior(times, values, times, "exponential", mean=1.0, **PARAMS)
    np.testing.assert_allclose(mean, values, atol=1e-10)
    np.testing.assert_allclose(cov, 0.0, atol=1e-10)


def test_matches_direct_conditioning():
    obs_times = np.array([0.0, 0.7, 2.0, 3.5])
    values = np.array([0.2, -0.5, 1.0, 0.4])
    query_times = np.array([0.3, 1.0, 5.0])
    prior_mean = 0.5
    obs_cov = exponential(obs_times, obs_times, **PARAMS)
    cross_cov = exponential(query_times, obs_times, **PARAMS)
    expected_mean = prior_mean + cross_cov @ np.linalg.solve(
        obs_cov, values - prior_mean
    )
    expected_cov = exponential(
        query_times, query_times, **PARAMS
    ) - cross_cov @ np.linalg.solve(obs_cov, cross_cov.T)

    mean, cov = gp_posterior(
        obs_times, values, query_times, "exponential", mean=prior_mean, **PARAMS
    )
    np.testing.assert_allclose(mean, expected_mean, atol=1e-10)
    np.testing.assert_allclose(cov, expected_cov, atol=1e-10)


def test_exponential_kernel_is_markov():
    # The exponential kernel is an Ornstein-Uhlenbeck process: between two
    # observations, observations beyond them carry no extra information.
    query = np.array([1.25, 1.5])
    near = gp_posterior(
        np.array([1.0, 2.0]), np.array([0.3, -0.4]), query, "exponential", **PARAMS
    )
    all_ = gp_posterior(
        np.array([-3.0, 1.0, 2.0, 6.0]),
        np.array([2.0, 0.3, -0.4, -1.0]),
        query,
        "exponential",
        **PARAMS,
    )
    np.testing.assert_allclose(near[0], all_[0], atol=1e-10)
    np.testing.assert_allclose(near[1], all_[1], atol=1e-10)


def test_brownian_bridge_closed_form():
    # Brownian motion pinned at 0 at t = 0 (a singular K), observed at 1 and 3:
    # at time 2 the bridge has mean (y1 + y3) / 2 and variance (2-1)(3-2)/(3-1).
    mean, cov = gp_posterior(
        np.array([0.0, 1.0, 3.0]), np.array([0.0, 0.4, 1.2]), np.array([2.0]), brownian
    )
    np.testing.assert_allclose(mean, [0.8])
    np.testing.assert_allclose(cov, [[0.5]])


def test_reverts_to_prior_far_from_observations():
    mean, cov = gp_posterior(
        np.array([0.0, 1.0]),
        np.array([3.0, -2.0]),
        np.array([1000.0]),
        "exponential",
        mean=0.5,
        **PARAMS,
    )
    np.testing.assert_allclose(mean, [0.5])
    np.testing.assert_allclose(cov, [[PARAMS["variance"]]])


def test_consistent_repeated_time_matches_single_observation():
    query = np.array([0.5, 1.5])
    repeated = gp_posterior(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.2, 0.7, 0.7]),
        query,
        "exponential",
        **PARAMS,
    )
    single = gp_posterior(
        np.array([0.0, 1.0]), np.array([0.2, 0.7]), query, "exponential", **PARAMS
    )
    np.testing.assert_allclose(repeated[0], single[0], atol=1e-10)
    np.testing.assert_allclose(repeated[1], single[1], atol=1e-10)


@pytest.mark.parametrize(
    "obs_times, values, kernel",
    [
        ([0.0, 1.0, 1.0], [0.2, 0.7, 0.9], "exponential"),
        ([0.0, 1.0], [0.1, 0.7], brownian),  # Brownian motion is 0 at t = 0
    ],
    ids=["repeated-time", "zero-variance-time"],
)
def test_noiseless_inconsistent_observations_are_rejected(obs_times, values, kernel):
    params = PARAMS if kernel == "exponential" else {}
    with pytest.raises(ValueError, match="inconsistent"):
        gp_posterior(
            np.array(obs_times), np.array(values), np.array([0.5]), kernel, **params
        )


def test_batched_values_match_rows_and_share_covariance():
    obs_times = np.array([0.0, 1.0, 2.5])
    query_times = np.array([0.5, 1.7, 3.0, 4.0])
    values = sample_gp(obs_times, "exponential", rng=0, n_samples=5, **PARAMS)
    mean, cov = gp_posterior(obs_times, values, query_times, "exponential", **PARAMS)
    assert mean.shape == (5, 4)
    assert cov.shape == (4, 4)
    for row, row_mean in zip(values, mean, strict=True):
        single_mean, single_cov = gp_posterior(
            obs_times, row, query_times, "exponential", **PARAMS
        )
        np.testing.assert_allclose(row_mean, single_mean, atol=1e-12)
        np.testing.assert_allclose(cov, single_cov, atol=1e-12)


def test_hidden_points_of_generated_dataset_are_calibrated():
    # Hide every other point of a generated dataset and condition on the rest
    # using the saved config: standardized errors at hidden points are N(0, 1).
    config = GenerationConfig(
        generator="densiffusion.gp.sampling:generate",
        n_series=2000,
        n_steps=41,
        dt=0.5,
        seed=3,
        params={"kernel": "exponential", "mean": 1.0, **PARAMS},
    )
    values = generate(config)[:, :, 0]
    times = np.arange(config.n_steps) * config.dt
    params = dict(config.params)
    kernel = params.pop("kernel")
    mean, cov = gp_posterior(times[::2], values[:, ::2], times[1::2], kernel, **params)
    errors = (values[:, 1::2] - mean) / np.sqrt(np.diag(cov))
    assert abs(errors.mean()) < 0.02
    assert abs(errors.std() - 1.0) < 0.02


def test_smooth_kernel_on_dense_grid_is_accepted():
    # Many eigenvalues of K are zero up to roundoff; genuine GP draws must not
    # be mistaken for inconsistent data.
    def squared_exponential(t, s):
        return np.exp(-(np.subtract.outer(t, s) ** 2) / (2 * 5.0**2))

    times = np.linspace(0, 10, 400)
    values = sample_gp(times, squared_exponential, rng=0, n_samples=50)
    mean, cov = gp_posterior(
        times[::2], values[:, ::2], times[1::2], squared_exponential
    )
    np.testing.assert_allclose(mean, values[:, 1::2], atol=1e-4)
    assert np.isfinite(cov).all()


@pytest.mark.parametrize(
    "changes",
    [
        {"obs_times": np.array([0.0, np.nan, 2.0])},
        {"obs_times": np.zeros((3, 1))},
        {"query_times": np.array([[0.5]])},
        {"query_times": np.array([])},
        {"obs_values": np.zeros(2)},
        {"obs_values": np.zeros((2, 2))},
        {"obs_values": np.zeros((1, 1, 3))},
        {"obs_values": np.array([0.0, np.inf, 1.0])},
        {"mean": np.zeros(3)},
        {"mean": float("nan")},
        {"kernel": "rbf"},
        {"lengthscale": 0.0},
    ],
)
def test_invalid_inputs_are_rejected(changes):
    arguments = {
        "obs_times": np.arange(3.0),
        "obs_values": np.zeros(3),
        "query_times": np.array([0.5]),
        "kernel": "exponential",
        **PARAMS,
    } | changes
    with pytest.raises(ValueError):
        gp_posterior(**arguments)


def test_misspelled_kernel_parameter_is_rejected():
    with pytest.raises(TypeError, match="length_scale"):
        gp_posterior(
            np.arange(3.0),
            np.zeros(3),
            np.array([0.5]),
            "exponential",
            variance=1.0,
            length_scale=1.0,
        )
