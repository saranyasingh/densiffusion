# History polishing before noisy-target forecasting

**Complete: learned history polishing improved test CRPS over both baselines in
all three seeds under this fixed protocol.** Mean CRPS was **0.274400**, compared
with 0.285006 for raw histories and 0.282228 for median histories: relative
reductions of **3.72%** and **2.77%**, respectively, computed from equal-seed
means. All six learned-versus-baseline paired 95% CRPS intervals were below
zero. These intervals are conditional and unadjusted; they do not establish
general superiority across retraining or other distributions.

All methods undercovered the nominal 90% interval. Learned mean coverage was
82.22%, and its seed-28103 coverage was worse than both baselines despite lower
CRPS. The result supports a modest finite-budget forecasting improvement, with
substantial calibration limitations.

All nine models were globally frozen at **20:45:11 UTC on 2026-10-27**, before
fresh test generation. Evaluation finished at 22:10:08 UTC, followed by
saved-sample analysis and independent verification. The
[pipeline record](PIPELINE_COMPLETE.json), [analysis](primary/analysis.json),
[plain-text summary](primary/analysis.txt), and
[verification](primary/verification.json) are complete; the machine-readable
[outcome record](OUTCOME.json) summarizes the result. This was the full
scientific run, not a smoke pilot.

The question is whether a learned, history-only blind-spot MLP improves
probabilistic forecasting over both raw noisy observations and a seven-point
median. All three methods forecast the same original noisy future observations
with the same larger CSDI architecture. This is a new experiment: it does not
densify time grids, replace future targets with clean values, or reuse GP-trained
weights. Earlier saved waveform experiments remain separate.

The authoritative inputs are the [recorded protocol](protocol.json),
[run configuration](primary/config.json), and
[implementation documentation](../../docs/history_polishing_forecasting.md).
The scientific configuration was recorded on 2026-10-27 before scientific
training. No test-based tuning is permitted.

## Fresh-test results

All three seeds used 2,048 fresh independent trajectories and 64 draws per
trajectory. Each trajectory contributes eight noisy future targets. The table
below gives equal-seed means, sample standard deviations, and ranges; seed
variation is descriptive and separate from within-seed bootstrap uncertainty.
Coverage values and their SD/range are expressed in percent and percentage
points. CRPS and widths use original unnormalized observation units; MSE uses
squared observation units.

| Metric | Method | Mean | Seed SD | Seed range |
| --- | --- | ---: | ---: | ---: |
| CRPS | Raw | 0.285006 | 0.001026 | [0.283862, 0.285845] |
| CRPS | Learned | 0.274400 | 0.005654 | [0.269609, 0.280637] |
| CRPS | Median | 0.282228 | 0.004951 | [0.277060, 0.286929] |
| Mean-forecast MSE | Raw | 0.506439 | 0.017932 | [0.490280, 0.525732] |
| Mean-forecast MSE | Learned | 0.480770 | 0.036961 | [0.447757, 0.520701] |
| Mean-forecast MSE | Median | 0.501469 | 0.030482 | [0.468313, 0.528278] |
| 90% coverage (%) | Raw | 80.249 | 4.251 | [75.342, 82.794] |
| 90% coverage (%) | Learned | 82.222 | 5.868 | [76.514, 88.239] |
| 90% coverage (%) | Median | 79.537 | 1.773 | [78.040, 81.494] |
| 90% width | Raw | 1.314918 | 0.318973 | [1.059858, 1.672561] |
| 90% width | Learned | 1.493885 | 0.443313 | [1.164457, 1.997913] |
| 90% width | Median | 0.922863 | 0.059605 | [0.875283, 0.989721] |

The complete seed-level scores are:

| Seed | Method | CRPS | MSE | 90% coverage (%) | 90% width |
| --- | --- | ---: | ---: | ---: | ---: |
| 28101 | Raw | 0.283862 | 0.525732 | 82.794 | 1.212334 |
| 28101 | Learned | 0.280637 | 0.520701 | 88.239 | 1.997913 |
| 28101 | Median | 0.286929 | 0.528278 | 79.077 | 0.989721 |
| 28102 | Raw | 0.285845 | 0.490280 | 75.342 | 1.059858 |
| 28102 | Learned | 0.269609 | 0.473850 | 81.915 | 1.319285 |
| 28102 | Median | 0.277060 | 0.507816 | 81.494 | 0.875283 |
| 28103 | Raw | 0.285312 | 0.503306 | 82.611 | 1.672561 |
| 28103 | Learned | 0.272955 | 0.447757 | 76.514 | 1.164457 |
| 28103 | Median | 0.282694 | 0.468313 | 78.040 | 0.903586 |

Paired differences below are **first method minus second**; negative CRPS
differences favor the first method. Each interval uses 10,000 whole-trajectory
percentile bootstrap replicates within its seed, holding fitted models and
saved ensembles fixed. All intervals are 95% and unadjusted.

| Seed | Comparison | CRPS difference | Conditional 95% interval |
| --- | --- | ---: | ---: |
| 28101 | learned − raw | -0.003225 | [-0.005765, -0.000619] |
| 28101 | learned − median | -0.006292 | [-0.009519, -0.003011] |
| 28101 | median − raw | +0.003067 | [-0.000272, +0.006319] |
| 28102 | learned − raw | -0.016236 | [-0.018920, -0.013613] |
| 28102 | learned − median | -0.007451 | [-0.009571, -0.005291] |
| 28102 | median − raw | -0.008784 | [-0.011386, -0.006190] |
| 28103 | learned − raw | -0.012357 | [-0.016354, -0.008462] |
| 28103 | learned − median | -0.009739 | [-0.013861, -0.005720] |
| 28103 | median − raw | -0.002618 | [-0.005892, +0.000591] |

Across seeds, the CRPS differences have the following descriptive variation:

| Comparison | Mean difference | Seed SD | Seed range |
| --- | ---: | ---: | ---: |
| learned − raw | -0.010606 | 0.006680 | [-0.016236, -0.003225] |
| learned − median | -0.007828 | 0.001754 | [-0.009739, -0.006292] |
| median − raw | -0.002778 | 0.005927 | [-0.008784, +0.003067] |

Learned CRPS improved over both baselines in every seed. Median versus raw
was less consistent: median was worse by point estimate in seed 28101, better
in the other two, and only seed 28102 had a CRPS interval wholly below zero.
Learned point-forecast MSE was lower than both baselines in all seeds, but both
learned MSE comparisons in seed 28101 had intervals containing zero. The MSE
intervals in seeds 28102 and 28103 excluded zero in favor of learned polishing.
All metric-specific paired intervals are retained in the analysis JSON.

Calibration did not improve uniformly. Learned coverage was 88.24%, 81.92%,
and 76.51% across the three seeds. In seed 28103, raw coverage was 82.61% and
median coverage 78.04%, both higher than learned coverage. Learned intervals
were wider than median intervals in every seed, but narrower than raw in
seed 28103. All nine marginal coverage point estimates were below 90%. Lower
CRPS therefore does not imply calibrated intervals. Finite 64-draw quantiles
contribute uncertainty; this run does not isolate their contribution from
model or predictive-tail errors.

Equal-seed mean CRPS by future step is shown below. Learned mean CRPS was lower
than both baselines at all eight steps; this is a descriptive horizon summary,
not eight independent hypothesis tests.

| Future step | Time | Raw | Learned | Median |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32 | 0.264886 | 0.254911 | 0.261885 |
| 2 | 33 | 0.286665 | 0.275335 | 0.282972 |
| 3 | 34 | 0.282530 | 0.272922 | 0.281874 |
| 4 | 35 | 0.277111 | 0.266684 | 0.272841 |
| 5 | 36 | 0.287175 | 0.272771 | 0.281341 |
| 6 | 37 | 0.293643 | 0.281946 | 0.289864 |
| 7 | 38 | 0.294016 | 0.283607 | 0.290139 |
| 8 | 39 | 0.294024 | 0.287027 | 0.296908 |

Per-horizon MSE, coverage, width, and seed ranges are saved in
[analysis.json](primary/analysis.json) and the
[horizon plot](primary/plots/per_horizon.svg). The
[rendered horizon plot](preview_primary/per_horizon.svg.png) shows overlapping
seed ranges and undercoverage alongside the mean score improvement.

## Fixed data and history treatments

Seeds **28101, 28102, and 28103** each have 256 training, 256 validation, and
2,048 fresh test trajectories. Each independent trajectory supplies exactly one
32-point history and eight future points at integer times 0 through 39, with
spacing 1. Train, validation, and test use distinct recorded random streams;
the three methods within a seed use the same trajectories. Test trajectories
are generated only after all selected mediators and all nine forecasters have
been frozen. There is no signal normalization.

For each trajectory, independently draw amplitude $A\sim U[0.7,1.3]$, period
$P\sim U[24,40]$, and phase $\phi\sim U[0,2\pi]$. The observations are

$$
s_t=A\{\sin(2\pi t/P+\phi)
     +0.2\sin[2(2\pi t/P+\phi)+0.4]\},\qquad
y_t=s_t+\epsilon_t+B_tZ_t,
$$

with independent $\epsilon_t\sim N(0,0.15^2)$,
$B_t\sim\operatorname{Bernoulli}(0.1)$, and $Z_t\sim N(0,2^2)$.
The [generator](../../src/densiffusion/generators/noisy_periodic.py) exposes only
the noisy observations. No clean signals, latent parameters, or outlier labels
are available to model fitting, selection, preprocessing, or scoring.

| Method | History supplied to the forecaster |
| --- | --- |
| Raw | The unchanged 32 noisy observations. |
| Learned | One frozen blind-spot MLP prediction per history position. |
| Median | A seven-point, center-inclusive median with edge padding. |

The learned input contains six fixed neighbor values at offsets
`[-3, -2, -1, +1, +2, +3]`, followed by six availability indicators. Unavailable
neighbors have value and indicator zero. The center value is excluded; there
is no center residual connection or future input. All positions are polished
from the original history in one pass. The median uses only available history;
edge padding repeats the endpoint, so its first and last outputs equal the raw
endpoints. Each forecaster receives its assigned representation without a
second raw-history input. Future supervision remains `y[:, 32:]` for every
method.

For a fixed fitted mediator, all three histories are deterministic functions of
the same available noisy observations. Polishing adds no observation information
about the actual future, and may discard information. Any forecasting benefit
would concern the representation or optimization achievable with these finite
data and models. This is input polishing, without a fine grid or synthetic
future supervision.

## Selection and matched forecaster budget

Each seed fits four mediator candidates: hidden width 32 or 64, crossed with
MSE or Huber loss with delta 0.2. Each MLP has two ReLU hidden layers. Every
candidate receives 200 epochs of Adam, learning rate 0.001, weight decay 0.0001,
and batches of 256 center examples. The 256 histories provide 8,192 noisy-center
examples per split. Validation reconstruction MSE is recorded after epochs 1,
5, and every ten epochs through 200. The checkpoint and candidate with the
lowest masked noisy-center validation MSE are selected, including for the
Huber-trained candidates. Ties retain the first recorded candidate/checkpoint.
No forecasting score participates in mediator selection.

All three selected mediators are frozen before any forecaster is trained.
They receive no forecasting gradients or later adjustment. The four-candidate
search is part of the learned method's measured cost.

The mediator phase is complete. Every candidate received 6,400 optimizer
updates, and all three seeds selected width 64 with Huber loss:

| Seed | Selected mediator epoch | Noisy-center validation MSE | All four candidates: fit seconds |
| --- | ---: | ---: | ---: |
| 28101 | 160 | 0.497278 | 6.867 |
| 28102 | 140 | 0.456371 | 6.383 |
| 28103 | 50 | 0.486392 | 6.242 |

The independent mediator/protocol audit passed and checked all 69 frozen
mediator files. These are reconstruction selection scores against noisy
centers, not forecast-quality results or clean-signal recovery measurements.

The architecture is recovered from the previously selected `reference_size`
recipe, with fresh weights: **413,521 CSDI parameters**, 64 hidden channels,
four residual layers, eight attention heads, diffusion and time embeddings of
128, feature embedding 16, feedforward dimension 64, and dropout 0.1. The run
records the source recipe hashes; it loads no GP-trained checkpoint.

Every method and seed completed **1,200 epochs / 4,800 optimizer updates**, batch
size 64, Adam learning rate 0.001, weight decay $10^{-6}$, and gradient-norm
clipping at 1. The diffusion process has 50 steps with a quadratic beta schedule
from 0.0001 to 0.5. Learning rate falls by a factor of ten at epochs 900 and
1,080. All-step validation noise MSE is evaluated every ten epochs.

Forecaster selection uses noisy-future validation CRPS at exactly **10, 50,
100, 200, 400, 800, and 1,200 epochs**, with 64 draws for all 256 validation
trajectories at each opportunity. The first minimum wins. Validation noise MSE
is diagnostic and does not override CRPS selection. All candidate validation
ensembles are retained. Within a seed, initial weights, trajectory ordering,
diffusion timestep/noise streams, dropout seed, optimizer settings, selection
opportunities, and validation sampling streams are paired across methods.

The [frozen-training review](frozen_training_review.json) independently verified
all nine selections against the 63 saved validation candidate ensembles
(256 trajectories and 64 draws each):

| Seed | Method | Selected epoch | Selected validation CRPS |
| --- | --- | ---: | ---: |
| 28101 | Raw | 400 | 0.289567 |
| 28101 | Learned | 400 | 0.283496 |
| 28101 | Median | 200 | 0.286849 |
| 28102 | Raw | 400 | 0.295713 |
| 28102 | Learned | 400 | 0.278343 |
| 28102 | Median | 200 | 0.282286 |
| 28103 | Raw | 400 | 0.280140 |
| 28103 | Learned | 800 | 0.262438 |
| 28103 | Median | 200 | 0.276728 |

These scores chose the checkpoints; they are development-selection values,
not estimates from the fresh test set. No method selected the final epoch-1,200
candidate. Completing the fixed budget or selecting an earlier checkpoint
does not establish convergence. The [full validation curves](primary/plots/validation_crps.svg) include the
very poor epoch-10 candidates. A clearly labeled
[epoch-50-onward detail](preview_primary/validation_crps_epoch50_detail.svg.png)
makes later differences visible while retaining the original full plot.
Within each raw model, CRPS was worse at both 800 and 1,200 than at the selected
400. Every median model selected 200 and ended worse at 1,200. Learned CRPS
changed from 400 to 800 to 1,200 as follows: 0.283496 → 0.285672 → 0.290331
(seed 28101), 0.278343 → 0.284468 → 0.282407 (28102), and
0.273408 → 0.262438 → 0.262666 (28103). These are finite validation curves,
with scheduled learning-rate decreases and sampling noise, not a convergence
certificate or a causal diagnosis of later deterioration.

## Evaluation scope and plot inspection

The completed evaluation used only the original noisy future observations.
The primary score is empirical ensemble CRPS; MSE uses the ensemble mean.
Coverage is marginal inclusion in the empirical 5th-to-95th percentile interval,
and width is its mean size. No clean or latent simulator values were scored.
Saved forecast arrays have shape (2048, 64, 8, 1) per method and seed.

Each paired bootstrap replicate retained the trajectory's eight future
positions and saved ensemble; neither horizons nor draws were treated as
independent resampling units. The 95% percentile intervals condition on fitted,
validation-selected models, frozen preprocessing, and fixed ensembles. They
exclude retraining, checkpoint-selection, and Monte Carlo ensemble uncertainty
and have no multiplicity correction across comparisons or metrics. Bootstrap
base seed was 8675309, with recorded per-seed/comparison offsets. Three-seed
means, sample SDs and ranges remain separate descriptive summaries.

The [representative forecasts](primary/plots/representative_forecasts.svg) use
the first six test trajectories of seed 28101 (IDs 512–517), chosen without
score inspection. Their [rendered image](preview_primary/representative_forecasts.svg.png)
shows raw and supplied histories, the noisy future target, forecast mean,
empirical interval and forecast boundary. Learned histories attenuate visible
spikes; median histories retain endpoint spikes as prescribed. Learned bands
are often broad and jagged in these cases. These observations do not demonstrate
clean-signal recovery or explain the aggregate improvement.

All three original plots and the labeled validation detail were rendered and
inspected individually. Labels, axes, legends, bands and panels are intact.
The full-scale [validation image](preview_primary/validation_crps.svg.png)
preserves epoch 10; the separate detail prevents its large values from hiding
the later selection differences. Preview copies use square outer viewports
to avoid Quick Look cropping. Original SVG hashes are unchanged, as recorded
in [render metadata](preview_primary/render_metadata.json); see the
[visual QA record](preview_primary/plot_qa.txt) and
[detail derivation script](preview_primary/render_validation_detail.py).
The [root visual review](root_plot_review.json) independently records inspection
of all four PNGs and their hashes. The previews have intentional white margins.

The result is a modest improvement from this learned input representation
under the stated finite-data model and budget. It adds no future observations
and may discard information. It does not establish universal usefulness of
polishing, a mechanism for the gain, calibrated forecasting, or a result about
fine-grid densification. Earlier waveform experiments remain separate.

## Compute and checks

The recorded [environment](environment.json) is an Apple M1 Max with 64 GiB
memory, macOS 26.6.2 arm64, Python 3.12.15, NumPy 2.5.3, and PyTorch 2.14.1.
Scientific forecasters run on MPS with one sequential worker and one CPU
thread; mediator fitting and preprocessing run on CPU. Sampling batch size
is 256.

[Completed training](primary/TRAINED.json) took **8,272.914 seconds
(137.88 minutes)** of wall time, including 19.577 seconds for the mediator phase
and 8,253.332 seconds for the sequential forecaster phase. The total includes
phase overhead; these phase durations are not additional costs on top of it.
[Test evaluation](primary/EVALUATED.json) took **5,095.745 seconds
(84.93 minutes)**. The measured scientific training plus evaluation phases total
**13,368.660 seconds (222.81 minutes, about 3 h 43 min)**. This excludes coding,
the independent smoke run, report/plot work, and analysis/verification commands.
The [pipeline record](PIPELINE_COMPLETE.json) separately gives about 4.38 seconds
for analysis and 2.49 seconds for verification, including command startup.

The frozen per-seed timing records for [28101](primary/seed_28101/timings.json),
[28102](primary/seed_28102/timings.json), and
[28103](primary/seed_28103/timings.json) give the following descriptive means
and ranges across seeds:

| Method | Forecaster training seconds: mean [min, max] | Validation sampling subset: mean [min, max] | Train + validation preprocessing: mean milliseconds |
| --- | ---: | ---: | ---: |
| Raw | 913.728 [900.789, 932.488] | 497.929 [484.921, 514.482] | 0.00883 |
| Learned | 917.465 [893.763, 938.363] | 499.786 [479.954, 514.705] | 1.463 |
| Median | 919.645 [891.209, 936.479] | 500.481 [476.138, 517.880] | 2.616 |

The learned method additionally required **6.497 seconds of mediator fitting
per seed on average**, range 6.242–6.867 seconds, totaling 19.492 seconds across
the three four-candidate searches. Across all nine forecasters, 8,252.514
recorded training seconds included 4,494.588 validation-sampling seconds.
These small differences from phase wall times reflect measurement boundaries
and overhead. Preprocessing timings are single-run elapsed measurements and
should not be treated as stable microbenchmarks.

Completed test costs and total measured method costs are:

| Method | Test inference seconds: mean [min, max] | Test preprocessing: mean milliseconds | Method total seconds: mean [min, max] |
| --- | ---: | ---: | ---: |
| Raw | 582.746 [555.233, 606.018] | 0.02169 | 1496.474 [1456.022, 1519.476] |
| Learned | 560.542 [538.933, 582.589] | 6.01408 | 1484.512 [1460.255, 1509.734] |
| Median | 552.823 [537.620, 569.245] | 5.66496 | 1472.476 [1428.834, 1505.729] |

Method totals add training/validation preprocessing, forecaster training,
test preprocessing, and inference; learned additionally includes mediator
fit once. The scientific run used the same architecture and draw budget,
so modest observed timing differences are not evidence that one representation
is intrinsically faster. Timings are sequential measurements, not controlled
hardware benchmarks. Full timing distributions and per-seed components are
in the saved analysis and combined timing files.

Timing artifacts separate mediator fitting, training/validation preprocessing,
forecaster training, validation sampling, test preprocessing, and test inference.
Validation sampling is already included in forecaster training elapsed time;
the table shows it as a subset, and it is never added twice. The learned method alone
adds the complete mediator search cost. Frozen training timings are retained;
evaluation adds separate timing files. Wall times and per-method elapsed totals
are distinguished, including any overlapping phase markers.

[Implementation checks](implementation_checks.json) record **517 tests passed**,
Ruff checks passed, all 80 Python files passed formatting checks, and
`git diff --check` passed. The real MPS [smoke verification](smoke/verification.json)
and [saved-sample smoke analysis](smoke/analysis.json) passed integration checks.
These are implementation evidence, not scientific accuracy or convergence
evidence.

The [frozen-training review](frozen_training_review.json) passed at 20:47:18 UTC.
It checked 183 protected training artifacts, all 69 frozen mediator artifacts,
all nine measured 4,800-update budgets, all 63 validation ensembles, the
configured validation cadence, and unchanged noisy future targets. It replayed
the original noisy training/validation data and recomputed deterministic
mediator predictions. Every mediator was frozen before forecaster training,
and all frozen hashes were unchanged. This review accessed no test arrays or
scores and performed no forecast training or sampling. Complete scientific
test-analysis [verification](primary/verification.json) subsequently passed at
22:10:14 UTC. It independently replayed noisy observations, recomputed
deterministic mediator inference and saved test metrics, and verified selection,
pairing, preprocessing, budgets and frozen hashes. It did not train forecasters,
rerun diffusion sampling, or expose clean signals. The analysis reports complete
seed/marker inventory with no missing auxiliary artifacts, and every saved
runner score matches recomputation.

[Final preservation checks](final_checks.json) passed: all 476 protected older
result files are unchanged, frozen training hashes match, Git HEAD is unchanged,
and no commit or push was performed. The repeated 13 analysis tests are a subset
of the 517-test suite, not an additional test count.
The [independent final review](final_result_review.json) also recomputed all 36
paired metric intervals and checked the report's numerical and scope claims.
A [run-wide SHA-256 manifest](artifact_manifest.json) covers the delivered artifacts.

## Exact source provenance

The original frozen [training snapshot](primary/source.tar.gz) is preserved.
The [runtime source comparison](runtime_source_check.json) checked 89 relevant
files: 88 were byte-identical to that snapshot. The one different file,
polishing_analysis.py, received additional input/type/finite-score and
timing-subset validation guards at 18:27:28 UTC, shortly after the 18:27:18
training snapshot. There were **no training, sampling, metric-formula, or
bootstrap changes**. The exact analysis source used is archived at
[analysis_source/polishing_analysis.py](analysis_source/polishing_analysis.py),
with the [exact patch](analysis_source/validation_guards.patch), SHA-256 values,
and a repeated 13-test targeted analysis check recorded in the source review.
The original archive was not replaced. Documentation and README were finalized
later and are excluded from the byte-comparison claim.
The final repository source and documentation are also archived in the
[post-execution snapshot](source_final/source.tar.gz), with its
[snapshot notes](source_final/README.txt). This supplemental final archive does
not replace or claim byte identity with the original training snapshot.

## Reproduction and artifact guide

Run from the repository root in a fresh output directory; training refuses an
existing directory and evaluation refuses to overwrite completed test artifacts:

```sh
RUN=outputs/history-polishing-reproduction
uv run --locked python scripts/train_polishing_forecast.py \
  --phase train --output "$RUN" \
  --config configs/history_polishing_forecast.json \
  --recipe-source outputs/csdi-gp-better
uv run --locked python scripts/train_polishing_forecast.py \
  --phase evaluate --output "$RUN"
uv run --locked python scripts/analyze_polishing_forecast.py "$RUN" \
  --bootstrap-replicates 10000 --seed 8675309
uv run --locked python scripts/verify_polishing_forecast.py "$RUN"
```

The source entry points are the [runner](../../src/densiffusion/polishing_experiment.py),
[mediator implementation](../../src/densiffusion/history_polishing.py),
[saved-sample analysis](../../src/densiffusion/polishing_analysis.py), and
[independent verifier](../../scripts/verify_polishing_forecast.py).

The following paths under `primary/` contain the completed scientific artifacts:

| Artifact | Purpose/status |
| --- | --- |
| `config.json` | Recorded scientific configuration and random-stream allocation; present. |
| `MEDIATORS_FROZEN.json` | Present: all selected mediators and hashes, written before any forecaster training. |
| `TRAINED.json`, `FROZEN.json` | Present: completed training and global freeze before fresh test generation. |
| `EVALUATED.json` | Complete: test generation and forecast sampling. |
| `seed_<seed>/mediator/` | Four candidate logs/checkpoints, selection, and saved predictions. |
| `seed_<seed>/{raw,learned,median}_validation.json` | Recorded validation CRPS selection candidates. |
| `seed_<seed>/test.npz` | Canonical fresh noisy test histories/targets and trajectory IDs. |
| `seed_<seed>/{raw,learned,median}_samples.npz` | Paired forecasts, raw and supplied histories, times, and unchanged targets. |
| `seed_<seed>/scores.json` | Runner metric summaries, independently recomputed by analysis. |
| `seed_<seed>/timings.json` | Frozen training timing components. |
| `seed_<seed>/evaluation_timings.json`, `combined_timings.json` | Separate and combined test/training costs. |
| `analysis.json`, `analysis.txt` | Recomputed metrics, paired intervals, descriptive seed variation, costs, and completeness. |
| `plots/representative_forecasts.svg`, `per_horizon.svg`, `validation_crps.svg` | Saved-ensemble forecasts, horizon metrics, and validation selection curves. |
| `verification.json` | Independent replay/hash/selection/preprocessing/budget/metric checks. |

Analysis reads saved samples only and preserves source archives and freeze
markers. Verification checks data replay, frozen hashes, selection, preprocessing,
unchanged targets, paired initialization/update budgets, and saved metrics. It
does not retrain models or rerun diffusion inference. No commit or push is part
of this experiment.
