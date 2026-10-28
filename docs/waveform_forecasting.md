# Paired waveform forecasting experiment

This experiment compares direct CSDI, linear densification plus CSDI, and learned
Densiffusion plus CSDI. Its hypothesis concerns a useful finite-data history
representation and synthetic supervision. Generated history points provide no
additional information about the actual future beyond the observed history.

## Data and comparison

`generators.waveform.generate_waveforms` generates independent trajectories

```text
theta_i(t) = 2*pi*f_i*t + phi_i
X_i(t) = A_i*(sin(theta_i(t)) + 0.3*sin(2*theta_i(t) + psi_i)) + epsilon_i(t)
```

Each trajectory has independent, fixed latent parameters: amplitude uniform on
`[0.7,1.3]`, frequency on `[0.6,0.9]`, and both phases on `[0,2*pi]`. Observation
noise is independent normal with standard deviation `0.02`. Models receive only
observations. No normalization is fitted or applied.

The primary configuration is [`configs/waveform_forecast.json`](../configs/waveform_forecast.json):
three paired seeds, 1,024 training, 256 validation, and 1,024 test trajectories
per seed. Each trajectory supplies one 25-point history at `0:0.1:2.4` and one
8-point future at `2.5:0.1:3.2`. Splits use independent recorded RNG streams.
There are no overlapping windows.

All forecasters have the same 16-hidden-channel, two-residual-layer CSDI
architecture and identical initial weights within each seed. They train
separately for 600 epochs, with batch size 64, one cached augmentation per
trajectory, and 9,600 optimization updates each. They share example ordering,
Adam settings, and a 50-step quadratic diffusion schedule from `0.0001` to
`0.5`. The learning rate starts at `0.001` and drops by a factor of ten at 75%
and 90% of the budget; weight decay is `1e-6`. The densifier trains for 400
epochs. These budgets build on earlier 400-epoch repository experiments;
neither the budget nor the number of passing tests establishes convergence.

The same sample size is used for validation and test scoring: 64 draws, pooling
eight futures for each of eight sampled histories. Direct forecasting repeats
the observed history; linear interpolation repeats its deterministic history.
The sampling batch size is 256. Equal architecture and optimization updates do
not imply equal total compute: dense sequences are longer, and Densiffusion
also trains and samples a separate densifier. Costs are recorded separately.

## Preserved pipeline and selection

The [existing construction](forecasting.md) is retained:

1. The densifier hides alternating original-grid points, learning `0.2 -> 0.1`.
2. The frozen densifier independently samples the history using past observations
   only, and the training future using its original-grid values and the last
   history observation as a left anchor. Application uses `0.1 -> 0.05`.
3. The forecaster conditions on 49 dense history points and learns 16 synthetic
   future points. Evaluation takes future indices `1::2`, corresponding exactly
   to the eight original forecast times.

Original values are checked for exact preservation on every augmentation call
and by the sampling pipeline. Independently generated dense histories and
future targets are **not** assumed to be jointly sampled true fine trajectories.
Latent parameters and simulated fine-grid truth are excluded from training,
normalization, tuning, and checkpoint selection.

Forecaster checkpoints are selected by original-grid forecast CRPS on all 256
validation trajectories at epochs 100, 200, ..., 600, with fixed sampling seeds.
Every method gets the same six candidates. Denoising losses are logged, but
are not used to compare forecasting performance across target grids. Densifier
selection uses alternating-mask validation denoising loss every 50 epochs.
Optional trainer hooks preserve Torch RNG and module modes, so validation
sampling does not change subsequent optimization randomness.

All three seeds finish all training and selection before `FROZEN.json` records
configuration and checkpoint hashes. Only then does the evaluation phase
generate test observations. The evaluator rejects changed frozen artifacts and
existing test forecasts. No test-based tuning is performed.

## Reproduce

Run from the repository root with a fresh output directory:

```sh
uv run --locked python scripts/train_waveform_forecast.py --smoke --output outputs/waveform-smoke
uv run --locked python scripts/train_waveform_forecast.py --config configs/waveform_forecast.json --phase train --output outputs/waveform-primary
uv run --locked python scripts/train_waveform_forecast.py --phase evaluate --output outputs/waveform-primary
uv run --locked python scripts/analyze_waveform_forecast.py outputs/waveform-primary
uv run --locked python scripts/verify_waveform_forecast.py outputs/waveform-primary
```

The smoke run checks integration only; its accuracy has no scientific meaning.
The primary runner uses three independent CPU worker processes, each with one
Torch thread. Set `workers` in a copied configuration to change this. Accelerator
execution uses one worker. The recorded local timing check favored CPU over MPS
for this small model. Each method's times are wall time under concurrent seed
execution, not isolated hardware throughput or an equal-FLOP comparison.

## Artifacts and uncertainty

Each output directory contains the exact configuration, source snapshot, Git
diff/base revision, freeze hashes, and per-seed folders with:

- Recorded seeds and original-grid train/validation observations with split IDs.
- Four selected checkpoints, per-epoch logs, and validation forecast scores.
- Initialization hashes, equal-update audit, and augmentation preservation audit.
- Exact scored 64-draw ensembles, test histories/targets/IDs, and evaluation times.
- Aggregate metrics and training, validation, augmentation, and inference timings.

`analysis.json` recomputes CRPS, ensemble-mean MSE, empirical 90% interval
coverage/width, and each horizon's scores from saved ensembles. Paired percentile
bootstrap intervals resample whole test trajectories (10,000 replicates),
retaining their forecasts and all horizons together. These intervals condition
on the fitted models and saved forecast draws. Seed means, sample standard
deviations, and ranges are reported separately; they include training and data
seed variation. Three seeds do not establish a general training-variance law.
Intervals are unadjusted for multiple comparisons. The primary metric is CRPS;
other metrics describe forecast error and calibration.

Densiffusion's 64 draws share eight sampled histories, so their Monte Carlo
dependence differs from the direct and deterministic-linear ensembles. Equal
draw counts do not imply equal effective sample sizes. Scores evaluate the
saved empirical ensembles and are not unbiased exact-distribution scores.

Plots include the first six test trajectories in fixed order, per-horizon
performance, and validation CRPS curves. Finite-ensemble empirical quantiles can
affect coverage. Narrow intervals alone do not establish useful calibration.

The optional `waveform_diagnostics.run_diagnostics` requires completed frozen
test evaluation. It then separately compares interpolation at the trained and
transferred resolutions, generating fine-grid truth only for this diagnostic.
It cannot be used to revise this experiment's selected checkpoints or claims.
If the primary result favors learned densification, a separate history-only
ablation is needed to distinguish history representation from synthetic future
supervision. Further task or hyperparameter choices require fresh test data for
new confirmatory claims.

## Recorded local primary result

The completed 2026-10-27 run is saved under
`outputs/waveform-primary-20261027/`, with a full `REPORT.md`, exact ensembles,
paired bootstrap intervals, diagnostic samples, and artifact verification.
Mean CRPS was 0.146844 for direct CSDI, 0.136779 for linear densification, and
0.365331 for learned Densiffusion. Learned densification lost to both baselines
in every seed. All forecasters selected epoch 600, and the learned densifier
also lost to linear interpolation under its exact training geometry. These
findings limit the result to this fixed trained setup; convergence and useful
learned interpolation were not established. The conditional history-only
ablation was not triggered by this negative primary result.
