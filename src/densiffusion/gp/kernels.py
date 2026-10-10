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


KERNELS: dict[str, Kernel] = {"exponential": exponential}


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
