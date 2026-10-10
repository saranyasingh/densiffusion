"""Draw complete trajectories from a Gaussian process (GP).

Under a GP, the values x at times t_1, ..., t_T are jointly Gaussian:
x ~ N(mean, K) with K[i, j] = kernel(t_i, t_j). We factor K = A A^T once, then
each standard-normal vector z gives one trajectory x = mean + A z whose values
are correlated exactly as K says.

Draws are noiseless and come from K itself: no jitter is added to the diagonal,
so an exact posterior computed from the same kernel, parameters, and mean
describes these samples.
"""

import numpy as np

from densiffusion.contracts import Array, positive_int, validate_array
from densiffusion.gp.kernels import Kernel, get_kernel


def sample_gp(
    times: Array,
    kernel: str | Kernel,
    *,
    rng: np.random.Generator | int,
    n_samples: int = 1,
    mean: float | Array = 0.0,
    **kernel_params: float,
) -> Array:
    """Return (n_samples, len(times)) GP draws at times, in the order given.

    kernel is a name in KERNELS or a function kernel(t, s, **kernel_params).
    mean is a scalar or one value per time. rng is a NumPy Generator or a seed.
    """
    positive_int("n_samples", n_samples)
    times = validate_array(times, (None,))
    mean = np.asarray(mean)
    if mean.ndim == 0:
        mean = np.full(times.shape, mean)
    mean = validate_array(mean, times.shape)
    covariance = get_kernel(kernel)(times, times, **kernel_params)
    covariance = validate_array(covariance, (times.size, times.size))
    factor = covariance_factor(covariance)
    noise = np.random.default_rng(rng).standard_normal((n_samples, times.size))
    return mean + noise @ factor.T


def covariance_factor(covariance: Array) -> Array:
    """Return A with A @ A.T equal to covariance up to roundoff.

    Uses Cholesky when covariance is positive definite. A singular but valid
    covariance (for example from a repeated timestamp) falls back to the
    eigendecomposition K = V diag(w) V^T with A = V diag(sqrt(w)), treating
    eigenvalues within roundoff of zero as zero. Invalid input is rejected,
    never altered.
    """
    # Allow only roundoff-sized asymmetry, then make both triangles agree.
    scale = np.abs(covariance).max()
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-10 * scale):
        raise ValueError("Covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        pass
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Same roundoff threshold numpy.linalg.matrix_rank uses for "zero".
    roundoff = len(eigenvalues) * np.finfo(np.float64).eps * eigenvalues.max()
    if eigenvalues.min() < -roundoff:
        raise ValueError("Covariance must be positive semidefinite")
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def generate(
    *,
    n_series: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
    kernel: str,
    mean: float = 0.0,
    **kernel_params: float,
) -> Array:
    """Generator plug-in: (n_series, n_steps, 1) draws at arange(n_steps) * dt."""
    times = np.arange(n_steps) * dt
    samples = sample_gp(
        times, kernel, rng=rng, n_samples=n_series, mean=mean, **kernel_params
    )
    return samples[:, :, None]
