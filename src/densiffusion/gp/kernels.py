"""Covariance functions (kernels) shared by GP sampling and the exact posterior.

A kernel takes 1-D timestamp arrays t (length n) and s (length m) plus keyword
parameters and returns the n x m matrix K[i, j] = k(t[i], s[j]). Kernels take
timestamps rather than distances so non-stationary processes, such as fractional
Brownian motion, fit the same interface.

To add a kernel, write a function with this signature, check its parameters, and
register it in KERNELS so configs can select it by name.
"""

from collections.abc import Callable

import numpy as np

from densiffusion.contracts import Array

type Kernel = Callable[..., Array]


def exponential(t: Array, s: Array, *, variance: float, lengthscale: float) -> Array:
    """variance * exp(-|t - s| / lengthscale), the Matern 1/2 kernel.

    Not the squared-exponential (RBF) kernel: paths are continuous but rough.
    variance is the variance of each value, not a standard deviation, and
    lengthscale uses the same time units as t and s.
    """
    check_positive("variance", variance)
    check_positive("lengthscale", lengthscale)
    return variance * np.exp(-np.abs(np.subtract.outer(t, s)) / lengthscale)


def brownian(t: Array, s: Array, *, variance: float) -> Array:
    """Variance-scaled Brownian motion covariance: min(t, s).

    This is the covariance of standard Brownian motion started at 0, so the value
    at t = 0 is known exactly and the matrix is singular when the process is
    observed at a repeated or zero-variance timestamp.
    """
    check_positive("variance", variance)
    return variance * np.minimum.outer(t, s)


def fractional_brownian(
    t: Array, s: Array, *, variance: float, hurst: float
) -> Array:
    """Variance-scaled fractional Brownian motion covariance.

    The covariance is k(t, s) = 0.5 * variance * (|t|^(2H) + |s|^(2H)
    - |t-s|^(2H)). For H = 0.5 this reduces to Brownian motion started at 0.
    """
    check_positive("variance", variance)
    if not np.isfinite(hurst) or not 0.0 < hurst < 1.0:
        raise ValueError("hurst must be finite and strictly between 0 and 1")
    t_abs = np.abs(np.asarray(t, dtype=float))[:, None]
    s_abs = np.abs(np.asarray(s, dtype=float))[None, :]
    time_delta = np.abs(np.subtract.outer(t, s))
    power = 2.0 * hurst
    return 0.5 * variance * (t_abs**power + s_abs**power - time_delta**power)


def matern_3_2(
    t: Array, s: Array, *, variance: float, lengthscale: float
) -> Array:
    """Matérn 3/2 kernel: variance * (1 + sqrt(3) r / ell) * exp(-sqrt(3) r / ell)."""
    check_positive("variance", variance)
    check_positive("lengthscale", lengthscale)
    dist = np.abs(np.subtract.outer(t, s))
    scale = np.sqrt(3.0) * dist / lengthscale
    return variance * (1.0 + scale) * np.exp(-scale)


def matern_5_2(
    t: Array, s: Array, *, variance: float, lengthscale: float
) -> Array:
    """Matérn 5/2 kernel:
    variance * (1 + sqrt(5) r / ell + 5 r^2 / (3 ell^2)) * exp(-sqrt(5) r / ell).
    """
    check_positive("variance", variance)
    check_positive("lengthscale", lengthscale)
    dist = np.abs(np.subtract.outer(t, s))
    scale = np.sqrt(5.0) * dist / lengthscale
    return variance * (1.0 + scale + (5.0 / 3.0) * (dist / lengthscale) ** 2) * np.exp(
        -scale
    )


def periodic(
    t: Array,
    s: Array,
    *,
    variance: float,
    lengthscale: float,
    period: float,
) -> Array:
    """Periodic kernel based on the squared distance modulo the period."""
    check_positive("variance", variance)
    check_positive("lengthscale", lengthscale)
    check_positive("period", period)
    phase = np.abs(np.subtract.outer(t, s)) / period
    return variance * np.exp(-2.0 * np.sin(np.pi * phase) ** 2 / lengthscale**2)


KERNELS: dict[str, Kernel] = {
    "exponential": exponential,
    "brownian": brownian,
    "fractional_brownian": fractional_brownian,
    "matern_3_2": matern_3_2,
    "matern_5_2": matern_5_2,
    "periodic": periodic,
}


def get_kernel(kernel: str | Kernel) -> Kernel:
    """Return kernel if it is callable, otherwise look its name up in KERNELS."""
    if callable(kernel):
        return kernel
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}; choose from {sorted(KERNELS)}")
    return KERNELS[kernel]


def check_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
