# Densiffusion

Diffusion-based time-series imputation and forecasting research.

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
Supply the networks and use `DDPMSampler` or a custom sampler; see
[docs/training.md](docs/training.md) for the interfaces. Run
`uv run --extra research pytest` for the full test suite.

CSDI is available as `densiffusion.architectures.csdi.CSDIDenoiser`. Run the small
univariate GP training-and-sampling example locally with:

```sh
uv run python scripts/train_csdi_gp.py --output outputs/csdi-gp
```

See [docs/csdi.md](docs/csdi.md) for the held-out refinement comparison with
interpolation and the exact GP posterior, configuration, and differences from
the paper. Use a fresh output directory; this example evaluates the training
resolution only.

A [stronger training comparison](docs/csdi.md#stronger-training-comparison) uses
more independent GP paths, quadratic noise scheduling, learning-rate decay,
and validation-only selection before scoring fresh test trajectories.

Forecasting implements Densiffusion's frozen-densifier augmentation and pooled
CSDI future sampling, evaluated at the original observation times. Run the small
comparison against direct CSDI and linear densification with:

```sh
uv run python scripts/train_csdi_forecast.py --output outputs/csdi-forecast
```

See [docs/forecasting.md](docs/forecasting.md) for the training/inference flow,
leakage boundaries, baselines, saved artifacts, and forecast APIs.
