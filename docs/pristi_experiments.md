# PriSTI experiments

Two experiments with the temporal-only PriSTI in
`src/densiffusion/architectures/pristi.py`: alternate-point imputation under the
CSDI comparison protocol, and the densify-diffuse-downsample forecasting
benchmark from the proposal. All data are synthetic GP paths, so every method
is compared with the exact GP posterior, the best achievable result. Each
experiment uses one training seed.

## 1. Imputation under the CSDI protocol

`scripts/evaluate_pristi_imputation.py` reproduces the data, test seeds and
training plan of `scripts/compare_csdi_training.py`:

- 512 / 64 / 128 independent paths of 33 points, each split into two 17-point
  windows; alternate interior points hidden (2*dt -> dt).
- Default-size PriSTI (364k parameters), 200 epochs at batch size 64,
  quadratic schedule with 50 steps to 0.5, weight decay 1e-6, learning-rate
  decay at 75% and 90%, all-step validation every ten epochs.
- 32 draws per window, scored with `metrics.crps` like the CSDI comparison.

```sh
uv run --extra research --with matplotlib python \
    scripts/evaluate_pristi_imputation.py --output outputs/pristi
```

Each task takes about 6 minutes on CPU. On the exponential-GP task the exact GP
and interpolation scores reproduce the published CSDI comparison (0.1861 and
0.2543), so its test set is identical.

| Task | PriSTI | Exact GP | Interpolation | Gap closed | Spread ratio |
|---|---|---|---|---|---|
| Exponential GP (CSDI task) | 0.1887 | 0.1861 | 0.2543 | 96% | 0.88 |
| Brownian motion | 0.0423 | 0.0420 | 0.0567 | 98% | 0.78 |
| fBm, H = 0.7 | 0.0141 | 0.0137 | 0.0190 | 93% | 0.71 |
| fBm, H = 0.3 | 0.1161 | 0.1143 | 0.1582 | 96% | 0.86 |
| Matérn 5/2 | 0.0088 | 0.0055 | 0.0177 | 73% | 1.23 |

Hidden-point CRPS, lower is better. Gap closed is the share of the gap between
interpolation and the exact GP that PriSTI closes. Spread ratio is PriSTI's RMS
predictive standard deviation divided by the exact posterior's.

- PriSTI is within 1-3% of the exact GP on four tasks and matches CSDI on its
  own task (0.1887 vs. 0.1884 reported in `docs/csdi.md`).
- Its intervals are too narrow on all tasks except Matérn (spread ratio
  0.71-0.88), so coverage falls below the exact GP's.
- Matérn 5/2 is the weak case: intervals are too wide, the median error is
  1.6x the exact posterior mean's, and neighbouring imputed points correlate at
  +0.12 instead of -0.60.
- The best validation loss came at the final epoch on four tasks, so longer
  training would probably help.
- `metrics.crps` is a plug-in estimator that inflates a 32-draw ensemble's
  score by about 3% relative to a deterministic forecast, so the
  interpolation column is slightly flattered.

## 2. Densified vs sparse forecasting

`src/densiffusion/architectures/pristi_densiffusion.py` provides the pipeline
and `scripts/run_pristi_densiffusion.py` the benchmark. It stays PriSTI-specific;
the shared `Forecaster` interfaces remain stubs.

### Setup

- **Data:** for each task, GP paths are generated at spacing dt/8 and observed
  every 8th point (spacing dt). Everything is trained on the coarse
  observations only; the fine paths are used only for evaluation. Brownian
  motion and fBm use dt = 0.04; Matérn 5/2 uses lengthscale 1 and dt = 0.25.
  There are 256 / 32 / 64 train / validation / test paths of 33 coarse points.
- **Model:** a small PriSTI for every network (width 32, 4 heads, 2 layers;
  44k parameters).
- **Imputer:** trained on 9-point windows with alternate points hidden, taken
  at strides 1, 2 and 4 of the observations (2dt -> dt, 4dt -> 2dt,
  8dt -> 4dt). Each window is normalised by the mean and RMS increment of its
  observed points so the model can transfer to finer spacings. Applying it
  three times gives x8 by refining independent chunks of 5 observed points.
- **Forecasters:** windows of 9 history and 4 future coarse points.
  - *Sparse:* trained on the coarse windows.
  - *Densified x2, x4, x8:* history and future densified by the imputer (one
    realisation per training window).
  - *Interpolated x8:* history and future densified by linear interpolation.

  Every forecaster window is normalised by its coarse history points only, so
  all variants share one normalisation.
- **Training:** imputer 60 epochs, forecasters 150 epochs, batch size 64,
  quadratic schedule with 50 steps to 0.5, weight decay 1e-6, learning-rate
  decay at 75% and 90%, all-step validation every ten epochs.
- **Evaluation:** 128 test windows. Densified forecasters pool 4 densified
  histories x 4 futures; other methods draw 16 futures. Forecasts are
  downsampled to the coarse future times and scored with the fair ensemble
  CRPS, which is unbiased for any number of draws. Baselines are the exact GP
  forecast given the coarse history and repeating the last observation. Gains
  over the sparse forecaster have 95% bootstrap intervals over test paths.

```sh
for task in brownian fbm_h0.7 fbm_h0.3 matern52; do
    uv run --extra research python scripts/run_pristi_densiffusion.py run \
        --task $task --output outputs/densiffusion --threads 2 &
done; wait
uv run --extra research --with matplotlib python \
    scripts/run_pristi_densiffusion.py report --output outputs/densiffusion
```

Four tasks in parallel take about 21 minutes on an M4 CPU. `run --rescore`
reloads the saved checkpoints and recomputes samples and metrics without
retraining; it reproduces the original samples exactly.

### Results

Forecast CRPS relative to the exact GP forecast (1.00 is the best possible):

| Task | Sparse | Densified x2 | x4 | x8 | Interpolated x8 | Repeat last value |
|---|---|---|---|---|---|---|
| Brownian motion | **1.011** | 1.027 | 1.061 | 1.072 | 1.027 | 1.399 |
| fBm, H = 0.7 | 1.038 | 1.036 | **1.008** | 1.048 | 1.121 | 1.490 |
| fBm, H = 0.3 | 1.024 | 1.030 | 1.063 | **1.002** | 1.021 | 1.430 |
| Matérn 5/2 | **1.207** | 1.229 | 1.311 | 1.351 | 1.452 | 1.922 |

CRPS gain over the sparse forecaster with 95% intervals (positive is better):

| Task | Densified x2 | x4 | x8 | Interpolated x8 |
|---|---|---|---|---|
| Brownian motion | -0.003 [-0.013, +0.007] | -0.008 [-0.019, +0.002] | **-0.010 [-0.019, -0.001]** | -0.003 [-0.011, +0.005] |
| fBm, H = 0.7 | +0.000 [-0.004, +0.005] | +0.003 [-0.003, +0.010] | -0.001 [-0.011, +0.008] | **-0.009 [-0.017, -0.000]** |
| fBm, H = 0.3 | -0.002 [-0.016, +0.012] | -0.010 [-0.025, +0.004] | +0.006 [-0.006, +0.018] | +0.001 [-0.013, +0.014] |
| Matérn 5/2 | -0.006 [-0.026, +0.014] | **-0.027 [-0.050, -0.003]** | **-0.037 [-0.064, -0.011]** | **-0.063 [-0.098, -0.028]** |

Imputer one-step CRPS relative to the exact GP, refining true data at each
spacing (PriSTI / linear interpolation). It was trained only at 4dt, 2dt and dt:

| Task | 4dt | 2dt | dt | dt/2 | dt/4 | dt/8 |
|---|---|---|---|---|---|---|
| Brownian motion | 1.02 / 1.41 | 1.00 / 1.38 | 1.05 / 1.42 | 1.05 / 1.39 | 1.06 / 1.42 | 1.05 / 1.44 |
| fBm, H = 0.7 | 1.08 / 1.44 | 1.05 / 1.43 | 1.03 / 1.44 | 1.04 / 1.39 | 1.03 / 1.39 | 1.09 / 1.44 |
| fBm, H = 0.3 | 1.04 / 1.42 | 1.03 / 1.40 | 1.08 / 1.44 | 1.03 / 1.41 | 1.07 / 1.38 | 1.02 / 1.43 |
| Matérn 5/2 | 1.12 / 1.45 | 1.09 / 1.49 | 1.29 / 2.13 | 1.75 / 2.54 | 4.57 / 3.98 | 9.75 / 4.96 |

Densified x8 histories score 1.03-1.05x the exact GP posterior for Brownian
motion and fBm (interpolation 1.40-1.46x), but 2.53x for Matérn 5/2. The
coverage of the 16 draws' range is 82-89% for all forecasters, against 88% for
a calibrated forecaster.

### Findings

- No densified forecaster is significantly better than the sparse one on any
  task. The small advantages on fBm (x4 for H = 0.7, x8 for H = 0.3) are within
  their intervals.
- Densification significantly hurts Brownian motion at x8 and Matérn 5/2 at x4
  and x8.
- Learned densification beats interpolation at x8 for both fBm tasks and
  Matérn 5/2, but not for Brownian motion.
- Multi-scale training with per-window normalisation lets the imputer refine
  three halvings finer than it was trained on, near-optimally, for the
  self-similar processes. It fails for Matérn 5/2, which becomes smoother at
  finer spacings, and this explains the weak Matérn forecasts.
- Densification cannot add information: densified histories are functions of
  the same coarse observations. Any gain has to come from easier learning. Here
  the x8 forecaster instead faces 97-point windows with 32 future targets, at
  the same model size and number of updates. Most forecasters had their best
  validation loss in the final epoch, so they are likely undertrained.

### Caveats

- One seed, a small model, short training and 128 test windows.
- Per-window normalisation hides the absolute level, which the exact GP
  forecast uses. For the stationary Matérn process this alone may leave about a
  7% gap to the "best possible" line.
- The fair ensemble CRPS used here differs from the plug-in `metrics.crps`; it
  keeps 16-draw, 4-draw and deterministic forecasts comparable.
- An independent code review found no leakage or indexing errors. It led to
  the fair CRPS, the range-coverage measure and a fixed draw-count bug before
  these results were produced.

## 3. Quasi-periodic sweep: when could densification help?

Section 2 found no gain on processes that look alike at every scale. This
experiment uses a process where the sampling interval matters: a
quasi-periodic GP whose paths oscillate with slowly drifting amplitude and
phase. One quantity is varied: the period P divided by the coarse spacing dt.

- **P/dt = 1.6:** fewer than two samples per cycle, so the coarse data look
  like a slower wave (aliasing). This is the negative control.
- **P/dt = 2.5-4:** the wave is barely resolved. The coarse history is a
  jagged zig-zag, and a densified history would show the wave's shape. This
  is where densification is most plausibly helpful.
- **P/dt = 8-16:** the coarse points already trace the wave, so densifying
  should add little.

### Two data settings, two research questions

Densification cannot add information at test time: a densified history is a
function of the same coarse observations. Any gain must come from making the
forecaster easier to learn. The two settings differ in what extra data, if
any, the densifier can learn from.

| | Coarse-only | Fine training data |
|---|---|---|
| **Question** | Can a densifier learned from coarse data alone help forecast coarse data? | If fine-resolution training data exists, does densifying coarse test histories help forecast? |
| **Example use** | Only one sampling rate is ever recorded | A high-frequency archive, but a coarse live feed |
| **Data the models see** | Coarse paths only | Fine (dt/8) paths for training; coarse test histories |
| **Imputer** | One model, trained on coarse windows with alternate points hidden (2dt -> dt), applied three times (dt -> dt/2 -> dt/4 -> dt/8) | One model per level, each trained on fine paths at exactly the spacing it refines (dt -> dt/2, dt/2 -> dt/4, dt/4 -> dt/8) |
| **Scale mismatch** | Every application is at least one halving finer than training; for P/dt <= 4 the training inputs (spacing 2dt) are themselves aliased | None: training and use match |
| **Forecaster trained on** | Coarse windows densified by the imputer | True fine windows |
| **Test history** | Densified by the coarse-trained imputer | Densified by the fine-trained imputers |
| **Methods** | `learned_x2`, `learned_x8` | `fine_x2`, `fine_x8` |

Shared by both settings:

- **Sparse forecaster:** trained and tested on coarse windows. It is always
  coarse-only, so it is the baseline for both questions.
- **Oracle (`oracle_x2`, `oracle_x8`):** the fine-data forecaster, but with
  test histories densified by the exact GP posterior. This is the best any
  densifier could do. Comparing it with `fine_xK` shows what is lost by
  learning the densifier; comparing it with sparse shows whether perfect
  densification helps at all.
- **Interpolated x8, the exact GP forecast and repeating the last value**, as
  in section 2.

In the coarse-only setting the imputer trains at stride 1 only. Thinning the
coarse data, as section 2 did to teach several scales, would alias the wave.

### Setup

- **Kernel:** cos(2 pi tau / P) times Matérn 3/2 with lengthscale 12 dt,
  variance 1, dt = 1. The cosine gives a single frequency, so P/dt alone
  controls aliasing; the Matérn factor makes amplitude and phase drift.
- **Sweep:** P/dt in {1.6, 2.5, 3, 4, 6, 8, 16}.
- **Everything else as in section 2:** the same path counts, windows (9
  history and 4 future coarse points), small PriSTI, training schedule,
  epochs, 16 draws per test window and fair CRPS.
- **Equal imputer data:** each fine-data imputer is trained on a random
  subset of windows, the same number per path as the coarse imputer gets.
- **Shared forecaster:** `fine_xK` and `oracle_xK` share one forecaster
  trained on the true fine paths; only their test histories differ.

```sh
for task in qp_1.6 qp_2.5 qp_3 qp_4 qp_6 qp_8 qp_16; do
    uv run --extra research python scripts/run_pristi_densiffusion.py run \
        --task $task --output outputs/densiffusion_quasiperiodic --threads 1 &
done; wait
uv run --extra research --with matplotlib python \
    scripts/run_pristi_densiffusion.py report \
    --output outputs/densiffusion_quasiperiodic
```

The report adds `sweep.png`: CRPS relative to the exact GP, and gain over the
sparse forecaster, against P/dt.
