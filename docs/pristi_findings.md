# PriSTI densification: summary of findings

Results use 100 draws per test window: 10 densified histories x 10 futures,
or 100 futures for methods without a densifier. The full setup, all tables and
caveats are in [pristi_experiments.md](pristi_experiments.md).

## The question

Densiffusion's idea is to forecast a coarsely sampled series in three steps.
First fill in points between the observations with a diffusion model
(densify), then forecast at the fine resolution, then read off the coarse
future points. All data are synthetic Gaussian-process (GP) paths, so every
method can be compared with the exact GP answer, which is the best possible.
Scores are CRPS relative to that exact answer: 1.00 is perfect.

Three experiments, all with the temporal-only PriSTI:

1. **Gap-filling:** can PriSTI fill in hidden points as well as the exact GP?
2. **Self-similar processes:** does densifying help forecast Brownian motion
   and fractional Brownian motion, which look alike at every zoom level, or a
   smooth Matérn 5/2 process?
3. **A wave near its sampling limit:** does densifying help when the sampling
   rate matters?

## Headline

- PriSTI is a near-optimal gap-filler (within 0.2-4% of the exact GP on four
  of five tasks), but its uncertainty intervals are too narrow.
- Densifying never helps forecast self-similar or smooth processes.
- Densifying does help forecast a barely resolved wave, but only when
  fine-resolution training data exist. It then removes up to 73% of the
  sparse forecaster's excess error. A densifier learned from coarse data alone
  fails in exactly the cases where it would matter.

## 1. Gap-filling

Every other point is hidden and filled in, following the protocol of the CSDI
comparison in `docs/csdi.md`.

- PriSTI is within 0.2-4% of the exact GP's CRPS on the exponential GP,
  Brownian motion and both fBm tasks. At the protocol's 32 draws it matched
  CSDI on CSDI's own task (0.1887 vs. 0.1884).
- Matérn 5/2 is the weak case: PriSTI closes only 73% of the gap between
  linear interpolation and the exact GP, its intervals are too wide, and
  neighbouring filled-in points are wrongly positively correlated.
- On the other tasks it is overconfident: its 90% intervals cover 72-82% of
  hidden values, against 87-88% for the exact GP.

## 2. Densified vs. sparse forecasting on self-similar processes

An imputer trained on coarse data only is applied up to three times (x2, x4,
x8). Forecasters are trained on the densified windows and compared with a
forecaster trained directly on the coarse data (the "sparse" forecaster).

- **No densified forecaster significantly beats the sparse one on any task.**
  The closest is Brownian motion at x2 (1.006 vs. 1.016), well within noise.
- **Densifying significantly hurts Matérn 5/2 at every factor** (x8: 1.41 vs.
  1.26).
- Small fBm gains seen at 16 draws vanished at 100 draws; they were sampling
  noise.
- The imputer itself is not the problem for Brownian motion and fBm. Applied
  three halvings finer than it was trained on, it still scores 1.02-1.07x
  the exact GP. It fails for Matérn, which becomes smoother at finer scales.
- The reason densifying does not help: a densified history is a function of
  the same coarse points, so it adds no information. Any gain has to come from
  making the forecaster easier to train, and here the longer windows make it
  harder instead.

## 3. Quasi-periodic sweep

The process is a wave whose amplitude and phase drift slowly. The sweep varies
one quantity: the period P divided by the coarse spacing dt. At P/dt = 1.6 the
wave is aliased, at 2.5-4 it is barely resolved, and from 6 upwards the coarse
points already trace it.

Methods compared:

- **Sparse:** trained and tested on coarse data.
- **Coarse-data xK:** densifier learned from coarse data only.
- **Fine-data xK:** densifiers and forecaster trained on fine-resolution
  paths; test histories are still coarse and are densified before
  forecasting.
- **Oracle xK:** the fine-data forecaster with the exact GP as the
  densifier, the best any densifier could do.
- **Interpolated x8:** linear interpolation instead of a learned densifier.

Forecast CRPS relative to the exact GP (lower is better; 1.00 is perfect):

| P/dt | Sparse | Coarse-data x2 | Fine-data x2 | Oracle x2 | Interpolated x8 |
|---|---|---|---|---|---|
| 1.6 | 1.691 | 1.454 | 1.351 | 1.286 | 1.302 |
| 2.5 | 1.797 | 1.550 | **1.214** | 1.168 | 1.439 |
| 3 | 1.460 | 2.278 | **1.280** | 1.268 | 1.486 |
| 4 | 1.329 | 3.051 | **1.264** | 1.253 | 1.415 |
| 6 | 1.403 | 1.466 | 1.459 | 1.448 | 1.774 |
| 8 | 1.602 | 1.687 | 1.704 | 1.692 | 2.069 |
| 16 | 1.576 | 1.803 | 1.820 | 1.812 | 1.903 |

- **With fine training data, densifying helps exactly where predicted.** At
  P/dt = 2.5, 3 and 4, fine-data x2 removes 73%, 39% and 20% of the sparse
  forecaster's excess CRPS. It also beats interpolation, so the gain is not
  just from a smoother input.
- **The learned densifier is not the bottleneck.** Fine-data forecasts are
  within 0.05 of the oracle's for P/dt >= 2.5.
- **Coarse-only densification fails where it would matter.** At P/dt = 3-4
  its forecasts score 1.6-3.4x the sparse forecaster's CRPS. Its training data
  are themselves aliased, so it never sees the wave's true shape.
- **Once the wave is resolved (P/dt >= 6), densifying hurts**: slightly at
  x2, clearly at x8, even with the oracle densifier.
- **x2 beats x8 almost everywhere.** The longer x8 windows cost more than the
  finer resolution gains.
- **Below the sampling limit (P/dt = 1.6)** fine training data help, but
  interpolation does about as well, so that gain is not from densification.

## What this means

Densification adds no information at test time. It helped only when the
densifier carried knowledge the coarse data lack, here fine-resolution
training data showing the wave's true shape. The promising use case is a
high-frequency archive for training, a coarse live feed at test time, and a
signal near its sampling limit.

## Open questions and caveats

- **Confound:** the fine-data forecaster also trains on true fine futures,
  which the sparse forecaster never sees. Its gain could come from those
  richer training targets rather than from densified test histories. Two
  controls would separate them: the fine-data forecaster tested on
  interpolated histories, and a sparse forecaster trained on the eight
  phase-shifted coarse copies of each fine training path. Neither has been
  run.
- Every neural forecaster is far from the exact GP (1.17x at best, sparse
  1.33-1.80x in Experiment 3), so all are undertrained or too small.
- One seed, 128 test windows and a small (44k-parameter) PriSTI throughout.
  Other architectures have not been tested.

## Figures

The output folders are not committed. The figures are regenerated from saved
checkpoints:

- `scripts/plot_pristi_uncertainty.py`: each model's 90% interval at the
  filled-in points and over the forecast, against the exact GP's.
- `scripts/run_pristi_densiffusion.py report`: result tables and summary
  plots, including `sweep.png` for Experiment 3.
- `scripts/plot_quasiperiodic_figures.py`: example paths, the four pipelines
  on one test window, and example forecasts.
