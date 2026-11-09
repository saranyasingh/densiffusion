# Quasi-periodic sweep across models

Section 3 of [pristi_experiments.md](pristi_experiments.md) found that
densifying helps forecast a barely resolved wave, but only with fine training
data. This experiment repeats that sweep with every denoiser in the
repository, to check that the findings are not specific to PriSTI, and adds
controls for where the fine-data gain comes from.

## Setup

Everything except the network is identical to section 3: the quasi-periodic
kernel, P/dt in {1.6, 2.5, 3, 4, 6, 8, 16}, 256 / 32 / 64 paths, 9 history
and 4 future coarse points, the training schedule and epochs, 100 draws per
test window (10 densified histories x 10 futures) and the fair ensemble CRPS.
Each network is built by `densify_forecast.small_model` at about 45k
parameters:

| Model | Configuration | Parameters |
|---|---|---|
| PriSTI | width 32, 4 heads, 2 layers | 44,201 |
| CSDI | 32 channels, 4 heads, 2 residual layers | 45,097 |
| SSSD | width 24, 2 S4 blocks, state size 32 | 45,601 |
| TimeAutoDiff | width 24, 2 recurrent layers, embedding 32 | 47,561 |

### Matched-factor baseline and controls

The fine-data forecaster differs from the sparse one in two ways: it trains
on true fine paths, and its test histories are densified. Three methods
separate these:

- **Interpolated x2:** trained and tested on linearly interpolated coarse
  windows, the matched-factor baseline for fine-data x2. Section 3 only had
  interpolated x8.
- **Fine-data x2, interpolated history:** the fine-data x2 forecaster, but
  with test histories filled in by linear interpolation instead of the
  fine-data imputers. Comparing it with fine-data x2 shows what the learned
  densifier contributes.
- **Sparse, all phase shifts:** a sparse forecaster trained on the fine paths
  seen at the coarse spacing from all 8 offsets, so it has the same
  information as the fine data but never densifies. It has 8 times as many
  windows and trains for the same number of updates as the sparse forecaster
  (18 epochs, validating every epoch). Comparing it with fine-data x2 shows
  whether densification is needed at all, or the fine data alone suffice.

These were added after the first sweep. `run --extend` adds them to a finished
run without retraining or resampling anything else. Each new method draws its
own sampling seed, so `--extend` and a fresh run give identical samples.

## Commands

```sh
for model in csdi timeautodiff; do
    for task in qp_1.6 qp_2.5 qp_3 qp_4 qp_6 qp_8 qp_16; do
        uv run --extra research python scripts/run_densiffusion.py run \
            --model $model --task $task \
            --output outputs/quasiperiodic/$model --threads 1 &
    done
done
for task in qp_1.6 qp_2.5 qp_3 qp_4 qp_6 qp_8 qp_16; do
    uv run --extra research python scripts/run_densiffusion.py run \
        --model sssd --device mps --train-device cpu --task $task \
        --output outputs/quasiperiodic/sssd --threads 1 &
done; wait
uv run --extra research --with matplotlib python \
    scripts/run_densiffusion.py compare --output outputs/quasiperiodic
```

PriSTI's results come from `outputs/densiffusion_quasiperiodic` (linked as
`outputs/quasiperiodic/pristi`), extended with the new methods. With
`--model pristi` the pipeline reproduces those samples exactly. SSSD samples
on the Apple GPU (`--device mps`), where it is about 7 times faster than on
the CPU, but trains on the CPU (`--train-device cpu`): training on the GPU
gave non-finite gradients at epoch 103 of the sparse forecaster for five of
seven tasks, while the same training on the CPU finished. P/dt = 8 and 16
were trained on the GPU before this was found.

`compare` writes `comparison.md`, with the finding checks, the controls and
per-method tables across models, and `comparison.png`, with each model's
sweep.

## Results (one seed)

Forecast CRPS of the x2 methods relative to each model's own sparse forecaster
(below 1 is better; `+` / `-` significantly better / worse than sparse):

| P/dt | Model | Sparse / exact GP | Coarse-data x2 | Interpolated x2 | Fine-data x2 | Oracle x2 |
|---|---|---|---|---|---|---|
| 1.6 | PriSTI | 1.69 | 0.86 + | **0.71 +** | 0.80 + | 0.76 + |
| 1.6 | CSDI | 2.55 | 1.19 - | **0.60 +** | 1.17 - | 1.15 - |
| 1.6 | TimeAutoDiff | 2.33 | 0.91 + | 0.68 + | 0.62 + | **0.53 +** |
| 1.6 | SSSD | 1.17 | 1.11 - | 1.01 | 1.05 | **0.95 +** |
| 2.5 | PriSTI | 1.80 | 0.86 + | 0.72 + | 0.68 + | **0.65 +** |
| 2.5 | CSDI | 2.70 | 1.13 - | **0.76 +** | 0.89 + | 0.77 + |
| 2.5 | TimeAutoDiff | 2.31 | 1.24 - | 0.83 + | 0.61 + | **0.56 +** |
| 2.5 | SSSD | 1.18 | 1.15 - | 1.05 - | 1.13 - | **0.95 +** |
| 3 | PriSTI | 1.46 | 1.56 - | 0.90 + | 0.88 + | **0.87 +** |
| 3 | CSDI | 3.08 | 1.05 - | 0.53 + | 0.58 + | **0.50 +** |
| 3 | TimeAutoDiff | 1.49 | 1.02 | 1.06 - | 1.00 | **0.99** |
| 3 | SSSD | 1.16 | 1.14 - | 1.06 - | 1.02 | **1.01** |
| 4 | PriSTI | 1.33 | 2.30 - | 0.96 + | 0.95 + | **0.94 +** |
| 4 | CSDI | 3.39 | 1.19 - | 0.54 + | 0.68 + | **0.47 +** |
| 4 | TimeAutoDiff | 1.25 | 1.87 - | 1.12 - | **1.05 -** | 1.06 - |
| 4 | SSSD | 1.18 | 1.39 - | **1.03** | 1.04 | 1.05 - |

`outputs/quasiperiodic/comparison.md` has every method, the finding checks
and the controls; `comparison.png` shows each model's full sweep, and
`meeting_x2_by_model.png` the x2 methods relative to each model's sparse
forecaster.

### What holds across models

- **A 2x grid helps in the most aliased cases, except for SSSD.** At
  P/dt = 1.6 and 2.5, interpolated x2 beats sparse for PriSTI, CSDI and
  TimeAutoDiff.
- **Coarse-only learned densification fails.** It is significantly worse than
  sparse at P/dt = 4 for every model, and at P/dt = 3 for every model except
  TimeAutoDiff.
- **Densifying does not help once the wave is resolved.** Every x8 densifier
  is significantly worse than sparse at P/dt >= 6 for every model, including
  SSSD. A few x2 methods beat sparse at P/dt = 8 for CSDI and TimeAutoDiff,
  scattered results among many comparisons.

### What does not hold

- **Learned fine-data densification against straight lines depends on the
  architecture.** At P/dt = 2.5-4, fine-data x2 is significantly better than
  interpolated x2 for TimeAutoDiff (by 0.51 of the GP's CRPS at 2.5),
  marginally better for PriSTI at 2.5 and SSSD at 3 only, and significantly
  worse for CSDI at all three.
- **The 2x gain depends on how good the sparse forecaster is.** It is large
  for CSDI, whose sparse forecaster is far from the exact GP (2.6-3.4x at
  P/dt <= 4), modest for PriSTI (1.3-1.8x), absent at P/dt = 3-4 for
  TimeAutoDiff, and absent everywhere for SSSD, whose sparse forecaster is
  closest (1.16-1.18x). For SSSD no densified forecaster beats sparse; even
  the oracle gains only 5% at P/dt = 1.6 and 2.5. SSSD's sparse forecaster
  (1.18x at P/dt = 2.5) also beats every other model's best densified
  forecast (PriSTI fine-data x2: 1.21x). Densification compensates for a weak
  coarse forecaster; a good one makes it unnecessary.
- **The joint model** (one network filling in and forecasting) is worse than
  the separate imputer and forecaster at P/dt = 2.5-4 for every model
  (not significantly for PriSTI at P/dt = 4 or SSSD at 2.5).

### Caveats

- One seed per model; the models differ greatly in how close their sparse
  forecasters get to the exact GP, so equal parameter counts do not mean
  equal quality.
- SSSD was trained on the CPU for P/dt <= 6 and on the Apple GPU for 8 and
  16, before the GPU's training instability was found.
