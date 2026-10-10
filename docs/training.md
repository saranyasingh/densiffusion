# Imputation training

Install PyTorch with `uv sync --locked --extra research`. Supply an initialized
denoiser, an optional encoder, and a sampler factory. Both networks are ordinary
`torch.nn.Module` instances; set `encoder=None` for the denoiser-only path.

```python
from pathlib import Path

from densiffusion.data import make_windows, split_data
from densiffusion.experiments import run_experiment
from densiffusion.models import ImputationModel
from densiffusion.objectives import noise_mse
from densiffusion.training import TrainingConfig

splits = split_data(values, train_fraction=0.7, validation_fraction=0.15)
train, validation, test = (make_windows(part, window_length=65) for part in splits)
model = ImputationModel(denoiser, encoder=encoder)
scores = run_experiment(
    model,
    train,
    validation,
    test,
    sampler_factory=make_sampler,
    objective=noise_mse,
    config=TrainingConfig(epochs=100, batch_size=64),
    n_samples=100,
    dt=dt,
    seed=42,
    checkpoint=Path("outputs/imputer.pt"),
)
```

`values` comes from a generator or saved dataset; `dt` is its grid spacing.
Split independent paths along `axis="series"` or a continuous record along
`axis="time"`. Split before windowing to prevent overlap between partitions.
The runner selects weights using validation loss, then samples and evaluates
the test split in batches. The supplied model retains the selected weights.

The denoiser implements `forward(noisy, steps, *, observed, mask, dt)`.
An encoder implements `forward(observed, *, mask, dt)` and returns context;
the denoiser then accepts an additional `context` keyword. Context structure is
agreed between those two networks. Both networks train and checkpoint together.

Value tensors and the boolean mask have shape `(batch, time, channels)`.
`mask=True` marks hidden alternate points; `observed` is zero at those positions.
The encoder sees only `observed` and its mask. Only hidden points are noised.
Even-length windows lose their final point to keep both endpoints observed.
This trains `2*dt -> dt` imputation. `steps` contains zero-based diffusion indices;
`dt` passed to the networks is the target spacing. The beta schedule is linear.

Objectives implement `objective(prediction, batch) -> scalar_tensor` and return
a differentiable mean loss over hidden points. `DenoisingBatch` exposes `clean`,
`noisy`, `noise`, `observed`, `mask`, `steps`, `dt`, and the selected `alpha_bar`
values shaped `(batch, 1, 1)`. Built-ins are `noise_mse` and `reconstruction_mse`;
the latter trains the network to predict clean values. A custom objective can
use its own prediction layout, for example:

```python
def reconstruction_mae(prediction, batch):
    return (prediction[batch.mask] - batch.clean[batch.mask]).abs().mean()
```

`make_sampler(model, config)` returns a sampler bound to the trained model,
implementing `Densifier.sample` in `contracts.py`. Its reverse process must match
the objective's prediction type and the training schedule. The sampler receives
NumPy observations and an RNG; it returns `(batch, samples, time, channels)` arrays
that preserve observations exactly. Its `dt` argument is the observed spacing;
the model operates at `dt / factor`. The runner evaluates factor-two imputation.
Use `pipeline.impute` directly for other refinement factors or finer input grids.

`metrics.evaluate` reports mean marginal CRPS of the empirical sample distribution,
using the [ensemble CRPS definition](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_ensemble.html).
The runner scores hidden test entries only, averaging across windows, time, and
channels. Lower is better; use the same `n_samples` when comparing experiments.

Training uses Adam and gradient clipping. Seed network initialization separately;
the training seed controls shuffling and noise and resets PyTorch's RNG.
Validation noise is fixed across epochs. Enable `densiffusion.training` at INFO
for losses. Checkpoints save both networks, training configuration, objective name,
seed, and best validation loss; use a new path for each run. Optimizer state is
not saved. `train_densifier` is also available for training without sampling.

Forecasting retains the `Forecaster.sample` contract and stubs for
`data.make_forecast_windows`, `training.train_forecaster`, `pipeline.forecast`,
and `experiments.run_forecast_experiment`. Windows use the original grid;
`Forecaster.sample` returns fine-grid futures and `pipeline.forecast` will pool
samples and downsample to the original grid. Training will keep the densifier
fixed. The stub functions raise `NotImplementedError`; CRPS can reuse
`metrics.evaluate` with future targets.
