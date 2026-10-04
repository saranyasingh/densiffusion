# Densiffusion

Densify, diffuse, and downsample for time-series forecasting.

Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Use `uv sync --locked --extra research` to include SciPy and PyTorch.
Dependencies live in `pyproject.toml`; commit `uv.lock` after dependency changes.

Generators expose `generate(*, n_series, n_steps, dt, rng, **params)` and return
finite arrays shaped `(series, time, channels)`, including a channel axis for
univariate data. Times start at zero with spacing `dt`; use the supplied NumPy
`rng` for randomness. Put implementations in `src/densiffusion/generators/` or
any importable module.

Once a generator is available, copy `configs/generate.example.toml`, set its
`generator = "module:function"` and parameters, then run:

```sh
uv run densiffusion generate configs/local.toml --output data/train.npz
```

The `.npz` contains `values`, `times`, and a JSON `config` (including the seed),
readable with `numpy.load(path, allow_pickle=False)`. Existing files are not
overwritten; `data/` and `outputs/` are ignored by Git.

`contracts.py` defines generator, densifier, and forecaster interfaces.
`pipeline.forecast` pools sampled histories and futures, preserves observations,
and returns forecasts on the original grid. `metrics.evaluate` is a placeholder.
Generator implementations and model training are supplied separately;
the example config is a template.
