# PriSTI experiments

Two experiments with the temporal-only PriSTI in
`src/densiffusion/architectures/pristi.py`: alternate-point imputation under the
CSDI comparison protocol, and the densify-diffuse-downsample forecasting
benchmark from the proposal. All data are synthetic GP paths, so every method
is compared with the exact GP posterior, the best achievable result. Each
experiment uses one training seed and 100 draws per test window.

`scripts/plot_pristi_uncertainty.py` draws example predictions for each
experiment: each model's 90% interval at the filled-in points and over the
forecast, with the exact GP posterior's 90% interval overlaid. It uses one
canonical task per experiment (Brownian motion for sections 1 and 2,
P/dt = 2.5 for section 3) and writes `uncertainty_examples*.png` into each
output folder.

## 1. Imputation under the CSDI protocol

`scripts/evaluate_pristi_imputation.py` reproduces the data, test seeds and
training plan of `scripts/compare_csdi_training.py`:

- 512 / 64 / 128 independent paths of 33 points, each split into two 17-point
  windows; alternate interior points hidden (2*dt -> dt).
- Default-size PriSTI (364k parameters), 200 epochs at batch size 64,
  quadratic schedule with 50 steps to 0.5, weight decay 1e-6, learning-rate
  decay at 75% and 90%, all-step validation every ten epochs.
- 100 draws per window (the CSDI protocol uses 32), scored with
  `metrics.crps` like the CSDI comparison.

```sh
uv run --extra research --with matplotlib python \
    scripts/evaluate_pristi_imputation.py --output outputs/pristi
```

Training takes about 5 minutes per task on CPU. `--rescore` reloads the saved
checkpoints and only resamples; at 32 draws it reproduces the original samples
exactly. At 32 draws the exact GP and interpolation scores on the
exponential-GP task reproduce the published CSDI comparison (0.1861 and
0.2543), so its test set is identical.

| Task | PriSTI | Exact GP | Interpolation | Gap closed | Spread ratio |
|---|---|---|---|---|---|
| Exponential GP (CSDI task) | 0.1855 | 0.1837 | 0.2543 | 97% | 0.88 |
| Brownian motion | 0.0414 | 0.0413 | 0.0567 | 99% | 0.78 |
| fBm, H = 0.7 | 0.0139 | 0.0134 | 0.0190 | 91% | 0.70 |
| fBm, H = 0.3 | 0.1138 | 0.1122 | 0.1582 | 97% | 0.87 |
| Matérn 5/2 | 0.0086 | 0.0053 | 0.0177 | 73% | 1.23 |

Hidden-point CRPS, lower is better. Gap closed is the share of the gap between
interpolation and the exact GP that PriSTI closes. Spread ratio is PriSTI's RMS
predictive standard deviation divided by the exact posterior's.

- PriSTI is within 0.2-4% of the exact GP on four tasks. At the protocol's 32
  draws it matched CSDI on CSDI's own task (0.1887 vs. 0.1884 reported in
  `docs/csdi.md`); CSDI has not been scored at 100 draws.
- Its intervals are too narrow on all tasks except Matérn (spread ratio
  0.70-0.88), so 90% intervals cover 72-82% of hidden values, against 87-88%
  for the exact GP.
- Matérn 5/2 is the weak case: intervals are too wide, the median error is
  1.6x the exact posterior mean's, and neighbouring imputed points correlate at
  +0.12 instead of -0.60.
- The best validation loss came at the final epoch on four tasks, so longer
  training would probably help.
- `metrics.crps` is a plug-in estimator that inflates an ensemble's score
  relative to a deterministic forecast, by about 3% at 32 draws and 1% at
  100. That is why every sampled score fell from the 32-draw run, and why the
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
- **Evaluation:** 128 test windows. Densified forecasters pool 10 densified
  histories x 10 futures; other methods draw 100 futures. Forecasts are
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

Four tasks in parallel take about 21 minutes on an M4 CPU at the original 16
draws (4 x 4). `run --rescore` reloads the saved checkpoints and recomputes
samples and metrics without retraining; at the same draw counts it reproduces
the original samples exactly. The results below were rescored at 100 draws,
which took about 75 minutes per task with one thread each. At 16 draws,
sampling noise moved scores by up to 5%.

### Results

Forecast CRPS relative to the exact GP forecast (1.00 is the best possible):

| Task | Sparse | Densified x2 | x4 | x8 | Interpolated x8 | Repeat last value |
|---|---|---|---|---|---|---|
| Brownian motion | 1.016 | **1.006** | 1.010 | 1.037 | 1.038 | 1.389 |
| fBm, H = 0.7 | **1.045** | 1.068 | 1.067 | 1.098 | 1.130 | 1.512 |
| fBm, H = 0.3 | **1.024** | 1.032 | 1.027 | 1.031 | 1.029 | 1.437 |
| Matérn 5/2 | **1.256** | 1.303 | 1.332 | 1.409 | 1.476 | 1.953 |

CRPS gain over the sparse forecaster with 95% intervals (positive is better):

| Task | Densified x2 | x4 | x8 | Interpolated x8 |
|---|---|---|---|---|
| Brownian motion | +0.002 [-0.003, +0.006] | +0.001 [-0.003, +0.005] | -0.003 [-0.008, +0.001] | -0.004 [-0.009, +0.001] |
| fBm, H = 0.7 | -0.003 [-0.006, +0.001] | -0.002 [-0.006, +0.002] | -0.006 [-0.012, +0.001] | **-0.009 [-0.019, -0.000]** |
| fBm, H = 0.3 | -0.002 [-0.010, +0.005] | -0.001 [-0.007, +0.006] | -0.002 [-0.009, +0.004] | -0.001 [-0.008, +0.005] |
| Matérn 5/2 | **-0.012 [-0.023, -0.002]** | **-0.019 [-0.033, -0.006]** | **-0.039 [-0.059, -0.020]** | **-0.055 [-0.078, -0.034]** |

Imputer one-step CRPS relative to the exact GP, refining true data at each
spacing (PriSTI / linear interpolation). It was trained only at 4dt, 2dt and dt:

| Task | 4dt | 2dt | dt | dt/2 | dt/4 | dt/8 |
|---|---|---|---|---|---|---|
| Brownian motion | 1.05 / 1.40 | 1.03 / 1.40 | 1.04 / 1.41 | 1.03 / 1.33 | 1.03 / 1.41 | 1.02 / 1.43 |
| fBm, H = 0.7 | 1.04 / 1.43 | 1.04 / 1.42 | 1.05 / 1.46 | 1.05 / 1.41 | 1.03 / 1.41 | 1.05 / 1.42 |
| fBm, H = 0.3 | 1.05 / 1.43 | 1.04 / 1.43 | 1.07 / 1.42 | 1.03 / 1.40 | 1.04 / 1.40 | 1.03 / 1.42 |
| Matérn 5/2 | 1.11 / 1.42 | 1.08 / 1.47 | 1.29 / 2.12 | 1.81 / 2.57 | 4.38 / 4.05 | 9.39 / 4.91 |

Densified x8 histories score 1.05-1.06x the exact GP posterior for Brownian
motion and fBm (interpolation 1.42-1.44x), but 2.36x for Matérn 5/2. The 100
draws' range covers 95-100% of outcomes for every forecaster, against 98% for
a calibrated forecaster.

### Findings

- No densified forecaster is significantly better than the sparse one on any
  task. The small fBm advantages seen at 16 draws (x4 for H = 0.7, x8 for
  H = 0.3) vanished at 100 draws, so they were sampling noise. Brownian x2 is
  marginally ahead (1.006 vs. 1.016), well within its interval.
- Densification significantly hurts Matérn 5/2 at every factor, and
  interpolation hurts fBm with H = 0.7 (borderline).
- Learned densification beats interpolation at x8 for fBm with H = 0.7 and
  Matérn 5/2, and ties on the other two tasks.
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
  keeps 100-draw, 10-draw and deterministic forecasts comparable.
- The 16-draw results are kept in each output folder's `draws16/`.
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
  epochs, 100 draws per test window (10 histories x 10 futures) and fair
  CRPS.
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
sparse forecaster, against P/dt. `scripts/plot_quasiperiodic_figures.py
--output outputs/densiffusion_quasiperiodic` then draws example paths
(`figure_tasks.png`), the four pipelines on one test window
(`figure_setups.png`) and example forecasts (`figure_predictions.png`).

Seven tasks in parallel, one thread each, take about 55 minutes on an M4 CPU
at 16 draws. The results below were rescored at 100 draws with `--rescore`,
about 110 minutes per task.

### Results

Forecast CRPS relative to the exact GP forecast (1.00 is the best possible).
"Coarse-data" is the coarse-only setting:

| P/dt | Sparse | Coarse-data x2 | Coarse-data x8 | Interpolated x8 | Fine-data x2 | Fine-data x8 | Oracle x2 | Oracle x8 |
|---|---|---|---|---|---|---|---|---|
| 1.6 | 1.691 | 1.454 | 2.021 | 1.302 | 1.351 | 1.279 | 1.286 | **1.228** |
| 2.5 | 1.797 | 1.550 | 2.700 | 1.439 | 1.214 | 1.325 | **1.168** | 1.286 |
| 3 | 1.460 | 2.278 | 2.404 | 1.486 | 1.280 | 1.343 | **1.268** | 1.312 |
| 4 | 1.329 | 3.051 | 4.494 | 1.415 | 1.264 | 1.362 | **1.253** | 1.337 |
| 6 | **1.403** | 1.466 | 1.607 | 1.774 | 1.459 | 1.894 | 1.448 | 1.870 |
| 8 | **1.602** | 1.687 | 1.917 | 2.069 | 1.704 | 2.029 | 1.692 | 2.045 |
| 16 | **1.576** | 1.803 | 1.895 | 1.903 | 1.820 | 1.939 | 1.812 | 1.950 |

Which methods beat the sparse forecaster (95% bootstrap intervals over test
paths; full intervals are in the report's `results.md`):

| P/dt | Significantly better than sparse | Significantly worse |
|---|---|---|
| 1.6 | coarse-data x2, interpolated, fine-data x2 and x8, oracle x2 and x8 | coarse-data x8 |
| 2.5 | coarse-data x2, interpolated, fine-data x2 and x8, oracle x2 and x8 | coarse-data x8 |
| 3 | fine-data x2 and x8, oracle x2 and x8 | coarse-data x2 and x8 |
| 4 | fine-data x2, oracle x2 | coarse-data x2 and x8, interpolated |
| 6 | none | coarse-data x2, fine-data x2, all x8 methods |
| 8 | none | all densified methods |
| 16 | none | all densified methods |

Densified history CRPS relative to the exact GP given the coarse history, at
x8: fine-data imputers 1.4-2.3x; coarse-trained imputer 2.2x at P/dt = 16,
7-24x at 1.6, 2.5, 6 and 8, and 43-288x at 3 and 4. Linear interpolation is
3.2-23x.

### Findings

- **With fine training data, densification helps where predicted.** For P/dt
  from 2.5 to 4, fine-data x2 cuts the sparse forecaster's excess CRPS over the
  exact GP by 20-73%: 73% at P/dt = 2.5 (1.797 to 1.214), 39% at 3 and 20% at
  4, where only x2 is significantly better. At the same factor, fine-data x8
  beats interpolated x8 by 0.05-0.14 of the GP's CRPS, so the gain is not
  just from presenting a smoother input.
- **The learned densifier is not the bottleneck.** Fine-data forecasts are
  within 0.05 of the oracle's at x2 for P/dt >= 2.5 (0.07 at 1.6), although
  the fine-data histories score 1.4-2.4x the exact GP.
- **Below the Nyquist limit (P/dt = 1.6), fine training data still helps**,
  because it shows the true frequency. But interpolation does about as well
  there, so most of that gain is not from densification.
- **Once the wave is resolved (P/dt >= 6), densifying does not help.** x2 is
  slightly but significantly worse for every densifier except the oracle at
  P/dt = 6, and x8 is clearly worse for every densifier, including the oracle.
- **Coarse-only densification fails exactly where it would matter.** At
  P/dt = 3-4 the coarse-trained imputer's histories are 43-288x worse than the
  exact GP and the forecasts' CRPS is 1.6-3.4x the sparse forecaster's. The imputer is
  trained at 2dt -> dt, where the inputs are aliased, and then applied at
  finer, differently shaped scales. Its small gains at P/dt = 1.6 and 2.5
  (x2) are matched or beaten by interpolation.
- **x2 beats x8** for every densifier except at P/dt = 1.6. As in section 2, the x8 forecaster's
  97-point windows cost more than the finer resolution gains.

### Caveats

- The fine-data forecaster also trains on true fine futures, which the sparse
  one never sees. The gain could come from those richer training targets
  rather than from densified test histories. A fine-data forecaster tested on
  interpolated histories would separate the two; it has not been run.
- Every neural forecaster is far from the exact GP (1.2x at best, sparse
  1.4-1.8x), so all are undertrained or too small. One seed, 128 test windows.
- Per-window normalisation removes the absolute level, which the exact GP
  forecast uses for this stationary process.
