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
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig

splits = split_data(values, train_fraction=0.7, validation_fraction=0.15)
train, validation, test = (make_windows(part, window_length=65) for part in splits)
model = ImputationModel(denoiser, encoder=encoder)
scores = run_experiment(
    model,
    train,
    validation,
    test,
    sampler_factory=DDPMSampler,
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
Keep each paper's networks in `src/densiffusion/architectures/<paper>.py` and its
tests in `tests/test_<paper>.py`. Import them directly into experiment scripts;
the shared model, trainer, and sampler do not need per-architecture registration.

Value tensors and the boolean mask have shape `(batch, time, channels)`.
`mask=True` marks hidden points; `observed` is zero at those positions.
The encoder sees only `observed` and its mask. Only hidden points are noised.
By default, masks hide alternate interior points and even-length windows lose
their final point. This trains `2*dt -> dt` imputation. Pass `train_mask`,
`validation_mask`, and `test_mask` to `run_experiment` for other patterns;
`train_densifier` accepts the first two. Masks can be `(time,)` or boolean arrays
broadcastable from `(batch, time, channels)`. Supplied masks retain the full
window and must leave both hidden and observed entries in each example.
Generate masks separately, keeping validation and test masks fixed for comparisons.
Training and scoring require complete targets; do not replace their hidden values
with NaNs. `steps` contains zero-based diffusion indices;
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

`DDPMSampler` uses the training schedule and fixed posterior variance from
[DDPM](https://arxiv.org/abs/2006.11239). It keeps observations fixed at every step,
leaves values unclipped, and batches draws using `config.batch_size`. Its default
`prediction_type="noise"` matches `noise_mse`. For `reconstruction_mse`, use
`functools.partial(DDPMSampler, prediction_type="clean")` as the sampler factory.

A custom `sampler_factory(model, config)` returns a sampler bound to the trained model,
implementing `Densifier.sample` in `contracts.py`. Its reverse process must match
the objective's prediction type and the training schedule. The sampler receives
NumPy observations and an RNG; it returns `(batch, samples, time, channels)` arrays
that preserve observations exactly. Its `dt` argument is the observed spacing;
the model operates at `dt / factor`. The runner evaluates factor-two imputation.
With an explicit test mask it instead evaluates the full target grid at `factor=1`.
For direct masked sampling, use `impute(values, sampler, factor=1, mask=mask, ...)`.
Hidden inputs are cleared before calling the sampler, including NaNs. Observed
entries must be finite. Custom samplers accept optional `mask` and `times` keywords
when those features are used. `DDPMSampler` requires a regular grid with the given
spacing; absolute window origins may differ.

`metrics.evaluate` reports mean marginal CRPS of the empirical sample distribution,
using the [ensemble CRPS definition](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_ensemble.html).
The runner scores hidden test entries only. Loss and CRPS averages weight batches
by their hidden-entry counts. Lower CRPS is better; use the same `n_samples` when
comparing experiments.

For GP baselines, preserve the dataset's time coordinates through both splitting
and windowing. Nonstationary kernels such as Brownian motion depend on the
original time origin:

```python
splits = split_data(values, times=times, train_fraction=0.7, validation_fraction=0.15)
windows = [make_windows(v, times=t, window_length=65) for v, t in splits]
test_values, test_times = windows[2]
```

`GPPosteriorSampler` adapts a bound posterior function without a training step.
Once the GP branch is merged, use its posterior implementation directly:

```python
from functools import partial

from densiffusion.baselines import GPPosteriorSampler
from densiffusion.gp.posterior import gp_posterior
from densiffusion.metrics import evaluate
from densiffusion.pipeline import impute

sampler = GPPosteriorSampler(partial(gp_posterior, kernel="brownian", variance=1.0))
samples = impute(
    test_values,
    sampler,
    factor=1,
    mask=mask,
    times=test_times,
    n_samples=100,
    dt=dt,
    seed=42,
)
scores = evaluate(samples, test_values, mask=mask)
```

The adapter conditions each channel independently and needs observed values in
each channel being imputed. Match its kernel parameters and mean to the generator.
Covariances must be positive semidefinite. For cancellation in nearly deterministic
posteriors, set `GPPosteriorSampler(..., covariance_atol=...)` to a justified
absolute roundoff bound in variance units; it defaults to zero. Negative eigenvalues
within that bound are clipped to zero, without adding jitter.
Use the same targets, masks, and draw count as the neural experiment. Times default
to `arange(time) * dt` when omitted; pass actual times for windows with an offset.
`run_experiment` also accepts `test_times` with spacing `dt`. For irregular GP
queries, use `impute` directly with an explicit mask and time coordinates.

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
