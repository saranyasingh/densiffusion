# Densiffusion

Diffusion-based time-series imputation research.

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

`contracts.py` defines generator and imputation sampler interfaces.
`pipeline.impute` samples intermediate points and checks observation preservation.
Generator implementations are supplied separately; the example config is a template.

Use `data.split_data` before `data.make_windows` to keep partitions separate.
`ImputationModel` supports a denoiser with an optional encoder. `run_experiment`
trains with a supplied objective, samples held-out imputations, and reports CRPS.
Supply the networks and sampler; see [docs/training.md](docs/training.md) for the
interfaces. Run `uv run --extra research pytest` for the full test suite.

Future forecasting interfaces are reserved in `Forecaster`, `make_forecast_windows`,
`train_forecaster`, `forecast`, and `run_forecast_experiment`; the functions are stubs.
