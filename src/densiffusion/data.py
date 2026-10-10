import json
import tomllib
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

from densiffusion.contracts import (
    Array,
    Generator,
    positive_int,
    validate_array,
    validate_times,
)


@dataclass(frozen=True)
class GenerationConfig:
    generator: str
    n_series: int
    n_steps: int
    dt: float = 1.0
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive_int("n_series", self.n_series)
        positive_int("n_steps", self.n_steps)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.generator, str) or self.generator.count(":") != 1:
            raise ValueError("generator must be a module:function path")
        if not all(self.generator.split(":")):
            raise ValueError("generator must be a module:function path")
        if not isinstance(self.params, dict):
            raise ValueError("params must be a table")
        if self.params.keys() & {"n_series", "n_steps", "dt", "rng"}:
            raise ValueError("params cannot override n_series, n_steps, dt, or rng")

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        with path.open("rb") as source:
            return cls(**tomllib.load(source))


def generate(config: GenerationConfig) -> Array:
    module, name = config.generator.split(":")
    generator: Generator = getattr(import_module(module), name)
    if not callable(generator):
        raise TypeError(f"{config.generator} is not callable")
    values = generator(
        n_series=config.n_series,
        n_steps=config.n_steps,
        dt=config.dt,
        rng=np.random.default_rng(config.seed),
        **config.params,
    )
    return validate_array(values, (config.n_series, config.n_steps, None))


def save_dataset(path: Path, values: Array, config: GenerationConfig) -> None:
    """Save a dataset and its generation config without overwriting existing data."""
    values = validate_array(values, (config.n_series, config.n_steps, None))
    metadata = json.dumps(asdict(config), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = path.open("xb")
    try:
        with output:
            np.savez_compressed(
                output,
                values=values,
                times=np.arange(config.n_steps) * config.dt,
                config=metadata,
            )
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def split_data(
    values: Array,
    *,
    train_fraction: float,
    validation_fraction: float,
    axis: Literal["series", "time"] = "series",
    times: Array | None = None,
) -> tuple[Array, Array, Array] | tuple[tuple[Array, Array], ...]:
    """Split before windowing; supplied times return (values, times) per split."""
    values = validate_array(values, (None, None, None))
    if axis not in {"series", "time"}:
        raise ValueError("axis must be 'series' or 'time'")
    if not (
        0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Split fractions must be positive and sum to less than one")
    dimension = 0 if axis == "series" else 1
    length = values.shape[dimension]
    n_train = int(length * train_fraction)
    n_validation = int(length * validation_fraction)
    if min(n_train, n_validation, length - n_train - n_validation) < 1:
        raise ValueError("Each split must contain at least one series or time point")
    boundaries = [n_train, n_train + n_validation]
    parts = tuple(part.copy() for part in np.split(values, boundaries, axis=dimension))
    if times is None:
        return parts
    times = validate_times(times, values.shape[:2])
    time_parts = np.split(times, boundaries, axis=dimension)
    return tuple((part, t.copy()) for part, t in zip(parts, time_parts, strict=True))


def make_windows(
    values: Array,
    *,
    window_length: int,
    stride: int = 1,
    times: Array | None = None,
) -> Array | tuple[Array, Array]:
    """Return copied value windows, paired with time windows when supplied."""
    values = validate_array(values, (None, None, None))
    positive_int("window_length", window_length)
    positive_int("stride", stride)
    if values.shape[1] < window_length:
        raise ValueError("Each series must contain at least window_length points")
    windows = np.lib.stride_tricks.sliding_window_view(values, window_length, axis=1)
    windows = windows[:, ::stride].swapaxes(-1, -2)
    windows = windows.reshape(-1, window_length, values.shape[2]).copy()
    if times is None:
        return windows
    times = validate_times(times, values.shape[:2])
    time_windows = np.lib.stride_tricks.sliding_window_view(
        times, window_length, axis=1
    )
    return windows, time_windows[:, ::stride].reshape(-1, window_length).copy()


def make_forecast_windows(
    values: Array, *, context_length: int, horizon: int, stride: int = 1
) -> tuple[Array, Array]:
    """Return (history, future) windows on the input grid."""
    raise NotImplementedError("Forecast windowing is not implemented yet.")
