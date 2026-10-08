from pathlib import Path

import numpy as np
import pytest

from densiffusion.data import GenerationConfig, generate
from densiffusion.gp.kernels import KERNELS, exponential
from densiffusion.gp.sampling import covariance_factor, sample_gp

EXAMPLE_CONFIG = Path(__file__).parents[1] / "configs" / "gp.example.toml"


def test_exponential_kernel_values():
    t = np.array([0.0, 1.0])
    s = np.array([0.0, 0.5, 1.0])
    expected = 2.0 * np.exp(-np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]))
    np.testing.assert_allclose(
        exponential(t, s, variance=2.0, lengthscale=0.5), expected
    )
    assert KERNELS["exponential"] is exponential


def test_brownian_and_fractional_brownian_kernels():
    t = np.array([0.0, 1.0, 2.0])
    s = np.array([0.0, 1.5, 3.0])

    brownian_cov = KERNELS["brownian"](t, s, variance=2.0)
    expected_brownian = 2.0 * np.minimum.outer(t, s)
    np.testing.assert_allclose(brownian_cov, expected_brownian)

    fbm_cov = KERNELS["fractional_brownian"](t, s, variance=2.0, hurst=0.5)
    np.testing.assert_allclose(fbm_cov, expected_brownian)


def test_matern_kernels_match_reference_forms():
    t = np.array([0.0, 1.0])
    s = np.array([0.0, 0.5, 1.0])
    dt = np.abs(np.subtract.outer(t, s))
    lengthscale = 0.5

    expected_32 = 2.0 * (1.0 + np.sqrt(3.0) * dt / lengthscale) * np.exp(
        -np.sqrt(3.0) * dt / lengthscale
    )
    expected_52 = 2.0 * (1.0 + np.sqrt(5.0) * dt / lengthscale + 5.0 * dt**2 / (3.0 * lengthscale**2)) * np.exp(
        -np.sqrt(5.0) * dt / lengthscale
    )

    np.testing.assert_allclose(
        KERNELS["matern_3_2"](t, s, variance=2.0, lengthscale=lengthscale),
        expected_32,
    )
    np.testing.assert_allclose(
        KERNELS["matern_5_2"](t, s, variance=2.0, lengthscale=lengthscale),
        expected_52,
    )


def test_same_seed_reproduces_and_new_seed_differs():
    times = np.linspace(0.0, 5.0, 11)

    def draw(rng):
        return sample_gp(
            times, "exponential", rng=rng, n_samples=3, variance=1.0, lengthscale=2.0
        )

    first = draw(np.random.default_rng(7))
    assert first.shape == (3, 11)
    np.testing.assert_array_equal(first, draw(np.random.default_rng(7)))
    np.testing.assert_array_equal(first, draw(7))
    assert not np.array_equal(first, draw(8))


def test_draws_match_requested_mean_and_covariance():
    # Irregular, unsorted times: draws must follow the order given.
    times = np.array([1.5, 0.0, 3.0, 0.5])
    mean = np.array([1.0, -1.0, 0.0, 2.0])
    samples = sample_gp(
        times,
        "exponential",
        rng=0,
        n_samples=200_000,
        mean=mean,
        variance=2.0,
        lengthscale=2.0,
    )
    expected = 2.0 * np.exp(-np.abs(times[:, None] - times[None, :]) / 2.0)
    # Every off-diagonal covariance is well above zero, so sampling each time
    # independently would fail the covariance check below.
    assert expected[~np.eye(4, dtype=bool)].min() > 0.4
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.02)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), expected, atol=0.05)


def test_repeated_timestamp_gives_identical_values():
    # Two equal times make the covariance singular (two identical rows).
    times = np.array([0.0, 1.0, 1.0, 2.0])
    samples = sample_gp(
        times, "exponential", rng=0, n_samples=5, variance=1.0, lengthscale=1.0
    )
    assert np.isfinite(samples).all()
    np.testing.assert_allclose(samples[:, 1], samples[:, 2], atol=1e-6)


def cosine_covariance(times):
    # cos(t - s) has rank 2, like a periodic kernel on a long grid.
    return np.cos(np.subtract.outer(times, times))


@pytest.mark.parametrize(
    "covariance",
    [
        exponential(
            np.linspace(0, 10, 50),
            np.linspace(0, 10, 50),
            variance=1.5,
            lengthscale=2.0,
        ),
        exponential(
            np.array([0.0, 1.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 1.0, 2.0]),
            variance=1.0,
            lengthscale=1.0,
        ),
        np.ones((3, 3)),
        cosine_covariance(np.linspace(0, 5, 6)),
    ],
    ids=["positive-definite", "repeated-time", "rank-1", "rank-2"],
)
def test_factor_reconstructs_covariance(covariance):
    factor = covariance_factor(covariance)
    np.testing.assert_allclose(factor @ factor.T, covariance, atol=1e-10)


def test_smooth_kernel_on_dense_grid_is_accepted():
    # A very smooth kernel on a dense grid is singular up to roundoff: Cholesky
    # fails and tiny negative eigenvalues appear. That is still a valid GP.
    def squared_exponential(t, s):
        return np.exp(-(np.subtract.outer(t, s) ** 2) / (2 * 50.0**2))

    samples = sample_gp(np.linspace(0, 10, 1000), squared_exponential, rng=0)
    assert samples.shape == (1, 1000)
    assert np.isfinite(samples).all()


def not_symmetric(t, s, **params):
    return np.triu(np.ones((len(t), len(s))))


def not_positive_semidefinite(t, s, **params):
    return -np.ones((len(t), len(s)))


@pytest.mark.parametrize(
    "changes",
    [
        {"lengthscale": -1.0},
        {"lengthscale": float("inf")},
        {"variance": 0.0},
        {"times": np.array([0.0, np.nan])},
        {"times": np.zeros((2, 2))},
        {"times": np.array([])},
        {"kernel": "rbf"},
        {"mean": np.zeros(3)},
        {"n_samples": 0},
        {"kernel": not_symmetric},
        {"kernel": not_positive_semidefinite},
    ],
)
def test_invalid_inputs_are_rejected(changes):
    arguments = {
        "times": np.arange(4.0),
        "kernel": "exponential",
        "rng": 0,
        "variance": 1.0,
        "lengthscale": 1.0,
    } | changes
    with pytest.raises(ValueError):
        sample_gp(**arguments)


def test_misspelled_kernel_parameter_is_rejected():
    with pytest.raises(TypeError, match="length_scale"):
        sample_gp(np.arange(3.0), "exponential", rng=0, variance=1.0, length_scale=1.0)


def test_generate_plugin_matches_sample_gp():
    config = GenerationConfig(
        generator="densiffusion.gp.sampling:generate",
        n_series=3,
        n_steps=5,
        dt=0.5,
        seed=1,
        params={"kernel": "exponential", "variance": 1.0, "lengthscale": 2.0},
    )
    expected = sample_gp(
        np.arange(5) * 0.5,
        "exponential",
        rng=1,
        n_samples=3,
        variance=1.0,
        lengthscale=2.0,
    )
    np.testing.assert_array_equal(generate(config), expected[:, :, None])


def test_example_config_generates_data():
    config = GenerationConfig.from_toml(EXAMPLE_CONFIG)
    values = generate(config)
    assert values.shape == (config.n_series, config.n_steps, 1)
