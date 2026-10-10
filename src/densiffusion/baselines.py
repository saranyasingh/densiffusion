from collections.abc import Callable

import numpy as np

from densiffusion.conditioning import prepare_observations
from densiffusion.contracts import Array, Mask, positive_int, validate_array


class GPPosteriorSampler:
    """Adapt posterior(obs_times, obs_values, query_times) to imputation draws.

    The callable returns a mean vector and covariance matrix. Channels are
    conditioned independently; bind the kernel and its parameters beforehand.
    """

    def __init__(
        self,
        posterior: Callable[[Array, Array, Array], tuple[Array, Array]],
        *,
        covariance_atol: float = 0.0,
    ):
        if not np.isfinite(covariance_atol) or covariance_atol < 0:
            raise ValueError("covariance_atol must be finite and nonnegative")
        self.posterior = posterior
        self.covariance_atol = covariance_atol

    def sample(
        self,
        observed: Array,
        *,
        factor: int,
        n_samples: int,
        dt: float,
        rng: np.random.Generator,
        mask: Mask | None = None,
        times: Array | None = None,
    ) -> Array:
        positive_int("n_samples", n_samples)
        observed, hidden, times = prepare_observations(
            observed, factor=factor, dt=dt, mask=mask, times=times
        )
        samples = np.repeat(observed[:, None], n_samples, axis=1)
        for series in range(len(observed)):
            for channel in range(observed.shape[2]):
                missing = hidden[series, :, channel]
                if not missing.any():
                    continue
                if missing.all():
                    raise ValueError(
                        "GP conditioning requires observations per channel"
                    )
                mean, covariance = self.posterior(
                    times[series, ~missing],
                    observed[series, ~missing, channel],
                    times[series, missing],
                )
                size = int(missing.sum())
                mean = validate_array(mean, (size,))
                covariance = validate_array(covariance, (size, size))
                scale = np.abs(covariance).max()
                if not np.allclose(
                    covariance,
                    covariance.T,
                    rtol=0,
                    atol=max(1e-10 * scale, self.covariance_atol),
                ):
                    raise ValueError("Posterior covariance must be symmetric")
                eigenvalues, eigenvectors = np.linalg.eigh(
                    covariance / 2 + covariance.T / 2
                )
                roundoff = size * np.finfo(np.float64).eps * np.abs(eigenvalues).max()
                if eigenvalues.min() < -max(roundoff, self.covariance_atol):
                    raise ValueError(
                        "Posterior covariance must be positive semidefinite"
                    )
                covariance_factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
                draws = (
                    mean + rng.standard_normal((n_samples, size)) @ covariance_factor.T
                )
                samples[series, :, :, channel][:, missing] = draws
        return samples
