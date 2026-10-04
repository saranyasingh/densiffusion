import json
import tomllib
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Self

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
