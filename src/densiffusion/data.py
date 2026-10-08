import json
import tomllib
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

from densiffusion.contracts import Array, Generator, positive_int, validate_array


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


def make_mask(
    length: int,
    *,
    step: int = 2,
    offset: int = 1,
    random: bool = False,
    probability: float | None = None,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Return a boolean mask with True at hidden positions.

    By default, the mask is deterministic: `step` controls the spacing and `offset`
    selects the first hidden position. Set `random=True` to draw a Bernoulli mask with
    `probability` of masking each element, optionally seeded by `rng`.
    """
    positive_int("length", length)
    if random:
        if probability is None:
            probability = 0.5
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if isinstance(rng, np.random.Generator):
            generator = rng
        else:
            generator = np.random.default_rng(rng)
        return generator.random(length) < probability

    positive_int("step", step)
    if not 0 <= offset < step:
        raise ValueError("offset must satisfy 0 <= offset < step")
    return np.arange(length) % step == offset


def make_random_mask(
    length: int,
    *,
    probability: float = 0.5,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Return a random Bernoulli mask with a tunable masking probability."""
    return make_mask(length, random=True, probability=probability, rng=rng)


def mask_data(
    values: Array,
    *,
    axis: Literal["series", "time"] = "time",
    step: int = 2,
    offset: int = 1,
    mask: np.ndarray | None = None,
    fill_value: float | None = np.nan,
    random: bool = False,
    probability: float | None = None,
    rng: int | np.random.Generator | None = None,
) -> tuple[Array, np.ndarray]:
    """Return a masked copy and the boolean mask of hidden positions.

    `axis` selects which dimension to hide along; for `(batch, time, channels)` data,
    the time axis is the second dimension. The returned mask is `True` where the
    data were hidden and its shape matches the selected axis length.

    Set `random=True` for Bernoulli masking with a tunable `probability` and RNG seed.
    """
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("values must have at least one dimension")
    if axis not in {"series", "time"}:
        raise ValueError("axis must be 'series' or 'time'")
    axis_index = 0 if axis == "series" else 1 if array.ndim > 1 else 0
    if mask is None:
        mask = make_mask(
            array.shape[axis_index],
            step=step,
            offset=offset,
            random=random,
            probability=probability,
            rng=rng,
        )
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (array.shape[axis_index],):
        raise ValueError(
            "mask must have the same length as the selected axis "
            f"({array.shape[axis_index]}), got {mask.shape}"
        )

    masked = array.copy()
    if fill_value is not None:
        index = [slice(None)] * array.ndim
        index[axis_index] = mask
        masked[tuple(index)] = fill_value
    return masked, mask


def split_data(
    values: Array,
    *,
    train_fraction: float,
    validation_fraction: float,
    axis: Literal["series", "time"] = "series",
) -> tuple[Array, Array, Array]:
    """Split train/validation/test before windowing, preserving order along axis."""
    raise NotImplementedError("Dataset splitting is not implemented yet.")


def make_windows(
    values: Array, *, context_length: int, horizon: int, stride: int = 1
) -> tuple[Array, Array]:
    """Return history/future arrays shaped (windows, steps, channels)."""
    raise NotImplementedError("Time-series windowing is not implemented yet.")
