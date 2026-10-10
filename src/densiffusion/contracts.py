from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]
type Mask = NDArray[np.bool_]


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
        mask: Mask | None = None,
        times: Array | None = None,
    ) -> Array:
        """Return (batch, samples, (time - 1) * factor + 1, channels).

        dt is the observed spacing; every factor-th point must equal observed.
        With mask, observed is the full target grid and factor must be 1.
        mask=True marks hidden entries. times holds the input grid coordinates.
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


def validate_mask(mask: Mask, shape: tuple[int, int, int]) -> Mask:
    """Expand a time mask or a broadcastable (batch, time, channels) mask."""
    mask = np.asarray(mask)
    if mask.dtype.kind != "b" or mask.ndim not in (1, 3):
        raise ValueError("mask must be boolean with one or three dimensions")
    if mask.ndim == 1:
        if mask.shape != (shape[1],):
            raise ValueError("Time mask must match the number of time points")
        mask = mask[None, :, None]
    try:
        return np.broadcast_to(mask, shape).copy()
    except ValueError as error:
        raise ValueError(f"mask must broadcast to {shape}") from error


def validate_times(times: Array, shape: tuple[int, int]) -> Array:
    """Expand shared or per-series times without resetting their origins."""
    times = np.asarray(times)
    if times.ndim not in (1, 2) or times.shape[-1] != shape[1]:
        raise ValueError("times must have shape (time,) or (batch, time)")
    times = validate_array(times, (None,) * times.ndim)
    try:
        times = np.broadcast_to(times, shape).copy()
    except ValueError as error:
        raise ValueError(f"times must broadcast to {shape}") from error
    if not (np.diff(times, axis=1) > 0).all():
        raise ValueError("times must be strictly increasing")
    return times


def has_regular_spacing(times: Array, dt: float) -> bool:
    """Check spacing while allowing roundoff from absolute time coordinates."""
    scale = np.maximum(np.abs(times[..., :-1]), np.abs(times[..., 1:]))
    tolerance = 2 * np.finfo(np.float64).eps * scale
    return bool(
        np.isclose(np.diff(times, axis=-1), dt, rtol=1e-6, atol=tolerance).all()
    )
