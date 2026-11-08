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
- For a barely resolved wave, forecasting on a twice-as-fine grid removes up
  to 73% of the sparse forecaster's excess error. But most of that gain comes
  from the finer grid itself: filling it with straight lines, using coarse
  data only, does nearly as well. A densifier learned from fine training data
  adds a significant extra only at P/dt = 2.5, and one learned from coarse
  data alone fails where it would matter.

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
- **Interpolated xK:** gaps filled by straight lines; trained on interpolated
  coarse data, so it also uses coarse data only.
- **Fine-data xK:** densifiers and forecaster trained on fine-resolution
  paths; test histories are still coarse and are densified before
  forecasting.
- **Oracle xK:** the fine-data forecaster with the exact GP as the
  densifier, the best any densifier could do.

Two controls test where the fine-data gain comes from:

- **Fine-data x2, interpolated history:** the fine-data forecaster given
  straight-line histories instead of learned ones.
- **Sparse, all phase shifts:** a sparse forecaster trained on the fine paths
  seen at the coarse spacing from all 8 offsets, for the same number of
  updates as sparse. It has the fine data's information but never densifies.

Forecast CRPS relative to the exact GP (lower is better; 1.00 is perfect;
bold is the best non-oracle method):

| P/dt | Sparse | Coarse-data x2 | Interpolated x2 | Fine-data x2 | Oracle x2 |
|---|---|---|---|---|---|
| 1.6 | 1.691 | 1.454 | **1.201** | 1.351 | 1.286 |
| 2.5 | 1.797 | 1.550 | 1.288 | **1.214** | 1.168 |
| 3 | 1.460 | 2.278 | 1.313 | **1.280** | 1.268 |
| 4 | 1.329 | 3.051 | 1.272 | **1.264** | 1.253 |
| 6 | **1.403** | 1.466 | 1.467 | 1.459 | 1.448 |
| 8 | **1.602** | 1.687 | 1.724 | 1.704 | 1.692 |
| 16 | **1.576** | 1.803 | 1.775 | 1.820 | 1.812 |

- **A twice-as-fine grid helps for P/dt <= 4, however it is filled.**
  Interpolated x2 is significantly better than sparse at P/dt = 1.6-4,
  removing 71%, 64%, 32% and 17% of sparse's excess CRPS. Fine-data x2
  removes 49%, 73%, 39% and 20%.
- **Learned fine-data densification adds a significant extra only at
  P/dt = 2.5** (0.07 of the exact GP's CRPS better than interpolated x2). At
  3 and 4 the two are within noise, and at 1.6 interpolation is better.
- **Simply having fine data does not explain the gain.** The sparse forecaster
  trained on all phase shifts is significantly worse than fine-data x2 at
  P/dt = 2.5-4, and worse than sparse at 2.5 and 3.
- **The fine-data forecaster depends on learned histories.** Given
  straight-line histories it is far worse (2.03 instead of 1.21 at
  P/dt = 2.5), because it was trained on true fine shapes.
- **The learned densifier is not the bottleneck.** Fine-data forecasts are
  within 0.05 of the oracle's for P/dt >= 2.5.
- **Coarse-only learned densification fails.** At P/dt = 3-4 its forecasts
  score 1.6-3.4x the sparse forecaster's CRPS. Its training data are
  themselves aliased, so it never sees the wave's true shape.
- **Once the wave is resolved (P/dt >= 6), densifying hurts**: slightly at
  x2, clearly at x8, even with the oracle densifier.
- **x2 beats x8 almost everywhere.** The longer x8 windows cost more than the
  finer resolution gains.

## What this means

Densification adds no information at test time, and the experiments now
agree. For a barely resolved wave, a forecaster working on a finer grid
learns better than one on the coarse grid, even when the extra points are
straight lines. That is a learning effect: every neural forecaster here is
far from the exact GP (sparse 1.33-1.80x), so how the input is presented
matters. Fine training data and a learned densifier add a further,
significant gain only near P/dt = 2.5, where the wave is most ambiguous.

## Open questions and caveats

- **Why does a finer grid help?** Plausible reasons are the extra midpoint
  targets during training and PriSTI's attention working better on smoother
  inputs. Larger or longer-trained forecasters might close the gap without
  densifying.
- The phase-shifted sparse forecaster trains for only 18 epochs (the same
  number of updates as sparse) and may be undertrained.
- Every neural forecaster is far from the exact GP (1.17x at best), so all
  are undertrained or too small.
- One seed, 128 test windows and a small (44k-parameter) PriSTI.
  [quasiperiodic_models.md](quasiperiodic_models.md) repeats Experiment 3
  with CSDI, SSSD and TimeAutoDiff.

## Figures

The output folders are not committed. The figures are regenerated from saved
checkpoints:

- `scripts/plot_pristi_uncertainty.py`: each model's 90% interval at the
  filled-in points and over the forecast, against the exact GP's.
- `scripts/run_densiffusion.py report`: result tables and summary
  plots, including `sweep.png` for Experiment 3.
- `scripts/plot_quasiperiodic_figures.py`: example paths, the four pipelines
  on one test window, and example forecasts.
