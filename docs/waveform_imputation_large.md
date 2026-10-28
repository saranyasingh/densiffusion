# Larger waveform imputation gate

This follow-up first tests whether the successful larger GP configuration learns
useful waveform interpolation. Forecasting is conditional on both validation
imputation and a separate frozen-model resolution-transfer diagnostic passing.
The earlier [waveform experiment](waveform_forecasting.md) and its results remain
unchanged. No GP weights are reused.

## Recovered recipe

`outputs/csdi-gp-better/plan.json` and `selection.json` identify the selected
`reference_size` configuration. The network has 413,521 parameters: four residual
blocks, 64 hidden channels, eight attention heads, 128-dimensional time and
diffusion embeddings, a 16-dimensional feature embedding, feedforward dimension
64, and dropout 0.1. Univariate feature attention is bypassed as in the existing
implementation.

The first attempt reproduces the saved 200-epoch training recipe: Adam with
learning rate 0.001, batch size 64, weight decay `1e-6`, gradient clipping at 1,
and 50 quadratic diffusion steps with betas from `1e-4` to `0.5`. Learning rate
drops by a factor of ten after epochs 150 and 180. All-step denoising validation
evaluates every diffusion step every ten epochs. The waveform task retains
1,024 training and 256 validation trajectories, each with 33 original-grid points,
for each seed 27101, 27102, and 27103. Original `train_validation.npz` files are
copied byte-for-byte. There is no normalization, latent input, or fine-grid truth
in training or validation selection.

The GP run selected checkpoints by denoising loss. This follow-up instead selects
by hidden-point empirical interpolation CRPS, evaluated on all 256 validation
trajectories with 64 samples every 50 epochs and at the final epoch in the initial
200-epoch attempt. Intermediate
all-step denoising losses cannot replace a CRPS-selected checkpoint. The trainer's
optional `validation_score_interval` separates those two validation schedules
while preserving RNG and model modes around custom callbacks.

## Gates and bounded adjustment

The exact training geometry observes the 17 even points at spacing 0.2 and
imputes the 16 odd points at spacing 0.1. All metrics score hidden points only;
observations must be preserved exactly. The linear baseline averages neighboring
observations and repeats that deterministic estimate, so its interval width and
continuous-target coverage are zero.

The validation gate was recorded before any new candidate scores were inspected:

- Learned CRPS must be lower than linear CRPS in each of the three seeds.
- The reduction in mean CRPS across seeds must be at least 5%.
- Each seed's paired trajectory-bootstrap 95% interval for learned minus linear
  CRPS must have a negative upper endpoint.

These are development criteria on reused validation data, conditional on fitted
models and sampled ensembles. They are not confirmatory test confidence claims.
The bootstrap retains all hidden times of each independent trajectory together.
Seed variation is reported separately. Checkpoint and attempt selection reuse
validation data, and the intervals are not adjusted for those decisions.

At most two training attempts are permitted: the exact 200-epoch recipe and,
if its gate fails after validation-only diagnostics, one fresh 600-epoch run with
the same architecture, distribution, splits, optimizer, and diffusion schedule.
The longer run stretches the learning-rate milestones to 450 and 540 and scores
CRPS at epochs 150, 300, 450, and 600. Both attempts therefore have four CRPS
checkpoint opportunities, while all-step denoising validation stays every ten
epochs. This bounds the substantial sampling cost. The longer run does
not resume a decayed optimizer or load GP weights. No transfer truth is generated
while these choices remain open.

After a passing validation gate, all three selected densifiers and their
configuration/analysis hashes are frozen. A separate diagnostic generates 256
fresh trajectories per seed on the fine grid. It evaluates the actual application
geometries: a 49-point history and a 17-point future with its left anchor. Each
task has original spacing 0.1 and output spacing 0.05. Simulator truth is used
only to score these frozen samples. For each seed and task, learned CRPS must
be at most 1.25 times linear CRPS and ensemble-mean MSE at most 1.5 times linear
MSE. This is an adequacy screen, not a requirement to beat linear or proof of
resolution invariance. Failure stops the follow-up; fine truth is never used to
adjust weights, hyperparameters, or checkpoint selection.

Only if both gates pass may forecasting proceed with paired larger forecasters,
the original independent history/future densification construction, and fresh
forecast test trajectories generated after all forecast choices are frozen.

## Reproduce and inspect

From the repository root, use fresh output directories:

```sh
uv run --locked python scripts/train_waveform_imputation.py --smoke --output outputs/waveform-large-smoke
uv run --locked python scripts/train_waveform_imputation.py --config configs/waveform_imputation_large.json --output outputs/waveform-large-imputation
uv run --locked python scripts/analyze_waveform_imputation.py outputs/waveform-large-imputation
```

The analysis module `waveform_imputation_analysis.analyze_imputation_run` reads
the saved validation ensembles, computes metrics and paired intervals, and
writes the validation gate and SVG plots. Then freeze the selected densifiers:

```sh
uv run --locked python scripts/train_waveform_imputation.py --phase freeze --output outputs/waveform-large-imputation
uv run --locked python scripts/verify_waveform_imputation.py outputs/waveform-large-imputation
```

For a failed validation gate, run the bounded original-grid diagnostic before
the optional longer attempt. The diagnostic refuses successful gates or
experiments that already generated transfer truth:

```sh
uv run --locked python scripts/diagnose_waveform_imputation.py --attempt outputs/waveform-large-imputation
uv run --locked python scripts/train_waveform_imputation.py --config configs/waveform_imputation_large_longer.json --output outputs/waveform-large-imputation-longer
uv run --locked python scripts/analyze_waveform_imputation.py outputs/waveform-large-imputation-longer
uv run --locked python scripts/train_waveform_imputation.py --phase freeze --output outputs/waveform-large-imputation-longer
uv run --locked python scripts/verify_waveform_imputation.py outputs/waveform-large-imputation-longer
```

If the validation gate passes, the separate transfer phase is available:

```sh
uv run --locked python scripts/train_waveform_imputation.py --phase transfer --output outputs/waveform-large-imputation
uv run --locked python scripts/analyze_waveform_imputation.py --transfer outputs/waveform-large-imputation
```

Artifacts include source/configuration snapshots, original data hashes and copies,
initialization hashes, selected checkpoints, every epoch's logs, exact scored
samples for every CRPS candidate, aggregate and per-hidden-time metrics,
bootstrap intervals, reconstruction plots, and wall timings. Three independent
CPU workers each use one Torch thread; reported worker times are measured under
concurrent execution. Training totals include checkpoint-validation costs, which
are recorded separately. A larger architecture or a completed epoch budget does
not establish convergence.

## Conditional forecasting reproduction

Run this stage only after both the validation and resolution-transfer gates pass
and the independent imputation verifier confirms the saved results. The example
below assumes the longer attempt passed; substitute the first attempt's directory
if it passed instead, and omit `--development-source` when there was no earlier
attempt. Use a fresh forecast output directory.

The selected waveform densifier and original train/validation arrays are imported
byte-for-byte. The densifier is frozen and is not retrained. Direct CSDI, linear
densification, and learned densification each train a separate larger forecaster
with paired initialization, example ordering, architecture, and optimizer recipe:
600 epochs, batch size 64, and 9,600 updates. All-step denoising validation runs
every ten epochs. Checkpoint selection has four original-grid forecast CRPS
opportunities at epochs 150, 300, 450, and 600, each using all 256 validation
trajectories and 64 draws. The prepared config uses MPS with one worker.

```sh
uv run --locked python scripts/verify_waveform_imputation.py outputs/waveform-large-imputation-longer
uv run --locked python scripts/train_waveform_large_forecast.py --phase train --config configs/waveform_forecast_large.json --imputation-source outputs/waveform-large-imputation-longer --development-source outputs/waveform-large-imputation --output outputs/waveform-large-forecast-new
uv run --locked python scripts/train_waveform_large_forecast.py --phase evaluate --output outputs/waveform-large-forecast-new
uv run --locked python scripts/analyze_waveform_forecast.py outputs/waveform-large-forecast-new
uv run --locked python scripts/verify_waveform_large_forecast.py outputs/waveform-large-forecast-new
```

Training freezes every seed and method before evaluation generates 1,024 fresh
test trajectories per seed from the separate recorded test namespace. Evaluation
pools eight sampled histories and eight futures per history, scoring the eight
original-grid future times. Saved validation ensembles, checkpoint/source hashes,
conditioning audits, and timings support independent verification; imported
densifier training cost and earlier development cost are reported separately.
This prepared implementation and its smoke tests are not evidence of an executed
scientific forecasting result.

## Recorded local result

The completed 2026-10-27 follow-up is saved in
[`outputs/waveform-large-20261027/REPORT.md`](../outputs/waveform-large-20261027/REPORT.md).
The exact 200-epoch GP recipe failed waveform interpolation. The sole permitted
fresh 600-epoch adjustment passed the training-geometry validation gate in all
three seeds: mean hidden-point CRPS was 0.018464 for learned imputation versus
0.097639 for linear interpolation, an 81.09% reduction. All selected checkpoints
were at epoch 600 and still improved over epoch 450; convergence was not
established. Empirical nominal-90% coverage was 84.55%.

After freezing those models, fresh diagnostic trajectories exposed poor
resolution transfer in both application geometries. History CRPS was
1.47–1.64 times linear and ensemble-mean MSE 4.20–4.85 times linear across seeds.
Anchored-future CRPS was 1.88–2.10 times linear and MSE 6.69–7.54 times linear.
Every seed failed both predeclared transfer thresholds. Fine-grid truth was not
used for further tuning, and the conditional larger forecasting comparison was
not run. This establishes a transfer failure for these frozen models; it does
not answer whether a successfully transferred densifier would improve forecasts.

Exact scored ensembles, all-metric paired trajectory-bootstrap intervals, seed
variation, per-time scores, checkpoints, timing records, and inspected plots are
retained in the new output tree. Independent verification recomputed the metrics
and gates and checked exact observation preservation and frozen hashes. Earlier
waveform and GP results remain unchanged.
