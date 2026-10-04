from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]


class Generator(Protocol):
    def __call__(
        self,
        *,
        n_series: int,
        n_steps: int,
        dt: float,
        rng: np.random.Generator,
        **params: Any,
    ) -> Array:
        """Return (series, time, channels) at times arange(n_steps) * dt."""
        ...


class Densifier(Protocol):
    def sample(
        self,
        observed: Array,
        *,
        factor: int,
        n_samples: int,
        dt: float,
        rng: np.random.Generator,
    ) -> Array:
        """Return (batch, samples, (time - 1) * factor + 1, channels).

        dt is the observed spacing; every factor-th point must equal observed.
        """
        ...


class Forecaster(Protocol):
    def sample(
        self,
        observed: Array,
        dense_history: Array,
        *,
        horizon: int,
        n_samples: int,
        dt: float,
        rng: np.random.Generator,
    ) -> Array:
        """Return (batch, samples, horizon, channels) on the fine grid.

        dt is the fine spacing; the first forecast is dt after the last history.
        observed and dense_history have shape (batch, time, channels).
        """
        ...


def validate_array(values: Array, shape: tuple[int | None, ...]) -> Array:
    values = np.asarray(values)
    if values.ndim != len(shape) or any(
        actual == 0 or (expected is not None and actual != expected)
        for actual, expected in zip(values.shape, shape, strict=True)
    ):
        raise ValueError(f"Expected shape {shape}, got {values.shape}")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Arrays must contain finite real numbers")
    return values.astype(np.float64, copy=False)


def positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
