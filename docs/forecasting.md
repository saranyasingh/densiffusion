# Forecasting with CSDI and Densiffusion

The forecasting pipeline implements the proposal's two-stage construction:
learn a densifier from the observed grid, freeze it, and train a separate CSDI
forecaster using synthetic dense histories and futures. At inference, sample
multiple histories, draw futures conditional on each history, and pool the
forecasts at the original observation times.

## Run the experiment

Install research dependencies and run from the repository root:

```sh
uv sync --locked --extra research
uv run python scripts/train_csdi_forecast.py --output outputs/csdi-forecast
```

Use a fresh output directory. The CPU defaults are 32 independent exponential-GP
trajectories, 25 observations per trajectory, original spacing `dt=0.1`, nine
history observations, four future observations, and stride 12. Paths are split
into train/validation/test **before windowing**, with fractions 0.7/0.15/remainder.
The generator uses only the original grid; there is no finer-resolution ground
truth or fitted normalization. All model selection uses validation noise MSE.
The default seed and data are development smoke data that have been inspected
during implementation, not an untouched confirmatory benchmark. Scientific
comparisons require fresh test trajectories and configurations fixed before
examining their test results.

The example uses the small CSDI architecture documented in [csdi.md](csdi.md):
16 hidden channels and two residual layers. It trains the densifier for 200
epochs and each forecaster for 200 epochs, with 50 linear diffusion steps from
beta 0.0001 to 0.5. These defaults exercise the workflow; they do not establish
convergence or forecasting improvement. For a smaller integration check:

```sh
uv run python scripts/train_csdi_forecast.py --epochs 2 --densifier-epochs 2 --n-series 10 --n-steps 5 --context-length 3 --horizon 2 --n-histories 2 --n-futures 2 --output outputs/csdi-forecast-quick
```

This very short check verifies the data and sampling path only; its samples can
be badly scaled because its denoisers are undertrained.

`--device mps` enables Apple GPU execution. `--epochs`, `--densifier-epochs`,
`--n-augmentations`, `--n-histories`, `--n-futures`, `--factor`, and other settings
are explicit command-line options. Defaults use factor two; applying a model
trained on `2*dt -> dt` at `dt -> dt/factor` assumes transfer across resolutions.
The implementation supports this experiment without claiming that the
assumption holds for a particular dataset or trained model.

## What is conditioned on and what is predicted

All value arrays use `(batch, time, channels)`. Start with a coarse history of
length `L` and a coarse future of length `H`, at spacing `dt`:

1. Fit `q` on training windows using `train_densifier`, hiding alternating
   interior points. Apply the same masking pattern to validation windows for
   checkpoint selection. This learns `2*dt -> dt`; even-length windows are
   cropped by the existing training helper. Restore its best validation
   checkpoint and freeze it.
2. Sample dense histories from `q(history)` alone. With factor `k`, these contain
   `(L-1)*k+1` points at spacing `dt/k`, preserving every original observation.
   Future training values never enter this history-sampling call.
3. Separately densify `[last_history_value, future]` and discard its first
   point. This produces `H*k` synthetic future targets, beginning `dt/k` after
   the last observed history point, including the original future observations
   at indices `k-1, 2*k-1, ...`. The shared endpoint provides the left anchor for
   the first future interval.
4. Concatenate the dense history and synthetic future, hide **the entire future
   suffix**, and train a distinct CSDI model with hidden-only noise MSE. The
   fixed augmentation set contains `n_augmentations` draws per original window.
   Validation augmentations come only from validation paths and are fixed too.
5. For held-out forecasting, draw `M=n_histories` dense histories using only
   the observed past, then `J=n_futures` fine-grid futures for each. Keep fine
   future indices `k-1::k` and pool the `M*J` draws. The result has shape
   `(batch, M*J, H, channels)` and is scored against original-grid future values.

The forecaster conditions on the entire sampled dense history. Original
observations are an exactly preserved subset of that history; this version
does not add a separate embedding identifying their provenance. Dense histories
and synthetic training futures are sampled separately conditional on their
available coarse observations and shared endpoint, rather than as a single
joint bridge allowed to expose future information to the history.

The same `CSDIDenoiser` architecture and `ImputationModel` wrapper serve both
stages, with separate parameter sets. `DDPMForecastSampler` constructs a hidden
future suffix, uses the existing DDPM reverse process, and returns only future
values. Its `dt` and `horizon` are **fine-grid** quantities. `pipeline.forecast`
instead takes the **original-grid** `dt` and `horizon` and handles expansion,
sampling, and downsampling. Factor one performs direct forecasting without
densification. This implementation retains CSDI's full temporal attention;
it does not reproduce the paper's separate linear-attention forecasting setup.

## Shared APIs

```python
from densiffusion.data import make_forecast_windows
from densiffusion.pipeline import forecast
from densiffusion.samplers import DDPMForecastSampler, DDPMSampler
from densiffusion.training import train_forecaster

# Split trajectories or time blocks before constructing these windows.
train = make_forecast_windows(train_values, context_length=9, horizon=4)
validation = make_forecast_windows(validation_values, context_length=9, horizon=4)
q = DDPMSampler(trained_densifier_model, densifier_config)
train_forecaster(
    forecast_model,
    q,
    train,
    validation,
    config=forecast_config,
    factor=2,
    dt=0.1,
    n_augmentations=1,
    seed=42,
)
samples = forecast(
    test_history,
    q,
    DDPMForecastSampler(forecast_model, forecast_config),
    horizon=4,
    factor=2,
    n_histories=2,
    n_futures=4,
    dt=0.1,
    seed=43,
)
# samples.shape == (len(test_history), 8, 4, channels)
```

`experiments.run_forecast_experiment` combines forecaster fitting, held-out
sampling, and evaluation when the densifier is already fitted. Use
`metrics.evaluate_forecast(samples, target)` to recompute the scores directly.
The command-line example trains all comparison methods before sampling any
test forecasts and saves exactly the ensembles it scores.

## Baselines, scores, and saved artifacts

The example compares three separately trained forecasters:

| Method | Training augmentation and inference history |
| --- | --- |
| `densiffusion` | Frozen learned CSDI densifier |
| `direct_csdi` | Original grid, factor one |
| `linear_densification` | Deterministic linear interpolation |

All use the same forecaster architecture, initialization seed, epoch count,
diffusion schedule, augmentation count, and scored ensemble size. Direct CSDI
draws `M*J` futures on the original grid; linear densification repeats a
deterministic history but samples stochastic CSDI futures. Sequence lengths,
augmentation cost, and total compute differ. Equal epochs and draw counts are
not an equal-FLOP comparison; timings are recorded explicitly.

All four metrics aggregate over future time points, channels, and test windows:

- `crps`: empirical-ensemble marginal CRPS, in original value units.
- `mse`: squared error of the ensemble mean, in squared value units.
- `coverage_90`: fraction of targets within empirical 5th/95th percentile bounds.
- `interval_width_90`: mean width of those bounds, in original value units.

Small ensembles give noisy and discretized quantiles and CRPS estimates.
Coverage near 0.9 is a calibration diagnostic, while narrow intervals alone
do not establish quality. These are marginal scores, not a joint path score.

Each fresh output directory contains:

- `config.json`: architecture, schedules, data parameters, all seeds, software
  versions, split trajectory IDs, window origins, and augmentation conventions.
- `densifier.pt` and `{method}_forecaster.pt`: four best-validation checkpoints.
  Load with `torch.load(path, map_location="cpu", weights_only=True)` and restore
  `model_state_dict` into the architecture recorded in `config.json`.
- `samples.npz`: `history`, `target`, `history_times`, `future_times`, and the
  exact original-grid ensembles under the three method names in the table.
- `scores.json`: the four metrics for each saved ensemble.
- `timings.json`: densifier training, augmentation plus forecaster training,
  total training by method, and test forecast wall times. MPS timings synchronize
  the device. These times describe this run and hardware, not a general speed claim.

No held-out test value is used for augmentation, training, or checkpoint
selection. The example reports whatever scores result and does not select an
architecture, training duration, or a winning method using those test scores.

## Recorded development result

The saved CPU run `csdi-forecast-trained-verified` uses the defaults above with
seed 42: 22/4/6 train/validation/test trajectories, 44/8/12 windows, 200 epochs
per model, and eight forecast draws per test window. Its configuration, exact
scores, timings, and scored ensembles are archived in
[`results/csdi-forecast-smoke/`](results/csdi-forecast-smoke/). The original model
checkpoints remain local under `outputs/csdi-forecast-trained-verified/`.

| Method | CRPS | MSE | 90% coverage | 90% interval width |
| --- | ---: | ---: | ---: | ---: |
| Densiffusion | 0.944900 | 8.412426 | 0.541667 | 6.427816 |
| Direct CSDI | 0.872554 | 1.886782 | 0.312500 | 1.975799 |
| Linear densification | 11.488330 | 2626.130858 | 0.604167 | 132.697686 |

All four metrics were recomputed from the archived ensembles and match
`scores.json` exactly. The stored histories, targets, and split metadata also
match the current data generator. Direct CSDI has lower CRPS and MSE in this
run, and all three methods under-cover their nominal 90% intervals. This small,
previously inspected development run verifies the saved-results workflow; it
does not establish forecasting improvement, convergence, or generalization.
