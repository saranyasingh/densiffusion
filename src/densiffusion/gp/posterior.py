"""Exact posterior of a Gaussian process (GP) given observed values.

Conditioning the prior x ~ N(mean, k) on values y at times t_obs gives a
Gaussian at query times t_q with

    posterior mean       = mean + k(t_q, t_obs) K^+ (y - mean)
    posterior covariance = k(t_q, t_q) - k(t_q, t_obs) K^+ k(t_obs, t_q)

where K = k(t_obs, t_obs) and K^+ is its pseudo-inverse. Observations are
noiseless, like the draws from densiffusion.gp.sampling, so with the same
kernel, parameters, and mean this describes those draws exactly. No jitter is
added: a singular K (a repeated timestamp, a zero-variance point such as
Brownian motion at t = 0, a smooth kernel on a dense grid) is handled by the
pseudo-inverse, and observations K cannot produce are rejected.
"""

import numpy as np

from densiffusion.contracts import Array, validate_array
from densiffusion.gp.kernels import Kernel, get_kernel
from densiffusion.gp.sampling import psd_eigh, symmetrize

# Noiseless GP draws vary along a direction with eigenvalue at most roundoff by
# about sqrt(roundoff); components far beyond that cannot come from the GP.
INCONSISTENCY_TOLERANCE = 1e3


def gp_posterior(
    obs_times: Array,
    obs_values: Array,
    query_times: Array,
    kernel: str | Kernel,
    *,
    mean: float = 0.0,
    **kernel_params: float,
) -> tuple[Array, Array]:
    """Return the posterior (mean, covariance) at query_times.

    obs_values is (n_obs,) or (batch, n_obs), one row per series observed at
    obs_times. The mean has the same leading shape with n_query values per row;
    the (n_query, n_query) covariance depends only on the times, so every row
    shares it. kernel is a name in KERNELS or a function kernel(t, s, **params),
    and mean is a scalar prior mean.
    """
    obs_times = validate_array(obs_times, (None,))
    query_times = validate_array(query_times, (None,))
    obs_values = np.asarray(obs_values)
    if obs_values.ndim not in (1, 2):
        raise ValueError("obs_values must be (n_obs,) or (batch, n_obs)")
    obs_values = validate_array(
        obs_values, (None,) * (obs_values.ndim - 1) + (obs_times.size,)
    )
    if np.ndim(mean) != 0 or not np.isfinite(mean):
        raise ValueError("mean must be a finite scalar")

    kernel = get_kernel(kernel)
    n_obs, n_query = obs_times.size, query_times.size
    obs_cov = kernel(obs_times, obs_times, **kernel_params)
    obs_cov = symmetrize(validate_array(obs_cov, (n_obs, n_obs)))
    cross_cov = kernel(query_times, obs_times, **kernel_params)
    cross_cov = validate_array(cross_cov, (n_query, n_obs))
    query_cov = kernel(query_times, query_times, **kernel_params)
    query_cov = symmetrize(validate_array(query_cov, (n_query, n_query)))

    eigenvalues, eigenvectors, roundoff = psd_eigh(obs_cov)
    kept = eigenvalues > roundoff
    inverse = np.zeros_like(eigenvalues)
    inverse[kept] = 1 / eigenvalues[kept]

    residual = obs_values - mean
    components = residual @ eigenvectors
    tolerance = INCONSISTENCY_TOLERANCE * np.sqrt(roundoff)
    if np.any(np.abs(components[..., ~kept]) > tolerance):
        raise ValueError(
            "obs_values are inconsistent with a noiseless GP under this kernel "
            "(for example, different values at a repeated time)"
        )

    projected = cross_cov @ eigenvectors
    posterior_mean = mean + (components * inverse) @ projected.T
    reduction = projected * np.sqrt(inverse)
    posterior_cov = query_cov - reduction @ reduction.T
    return posterior_mean, posterior_cov
