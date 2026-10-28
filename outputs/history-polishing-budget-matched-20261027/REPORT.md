The prospective resource-matched polishing comparison is complete. Learned polishing has lower test CRPS than raw inputs in **3/3 seeds** and lower CRPS than median filtering in **3/3 seeds**. The corresponding counts of paired 95% intervals lying wholly below zero are **3/3** and **1/3**. This comparison matches a specified matrix-work budget; it does not establish equal wall time, energy, an optimal allocation, or a densification advantage.

The [protocol](protocol.json) was fixed before scientific data generation and fitting. The [exact configuration](primary/config.json), [frozen model manifest](primary/FROZEN.json), [independent verification](primary/verification.json), [analysis](primary/analysis.json), and [structured outcome](OUTCOME.json) retain the audit trail. Earlier experiments remain unchanged.

| Method | CRPS | Ensemble-mean MSE | 90% coverage | 90% interval width |
|---|---:|---:|---:|---:|
| raw | 0.284813 | 0.485661 | 74.00% | 1.039768 |
| learned | 0.272510 | 0.492595 | 75.48% | 0.929023 |
| median | 0.275057 | 0.474049 | 80.75% | 1.158907 |

These are equal-seed means over 2,048 fresh test trajectories per seed, with 64 forecast draws per trajectory. Targets are the original noisy future observations. No clean simulator signals are used. Lower CRPS and MSE are better; nominal coverage is 90%, and width alone does not measure calibration.

**Interpretation of the completed comparison.** The learned model's CRPS advantage over raw inputs survives this resource definition in all three fresh seeds. The additional CRPS advantage over median filtering is small: its paired interval includes zero in two seeds. This does not establish a consistently superior learned preprocessing method. Learned polishing also has higher average ensemble-mean MSE than both baselines, while median filtering has better average coverage. All three methods substantially under-cover nominal 90% intervals. The learned model's narrower intervals are therefore not evidence of better calibration.

MSE improves with learned polishing in seeds 29101 and 29102, but deteriorates in seed 29103 (0.545192 versus 0.478749 raw and 0.474636 median). This third-seed deterioration reverses the average MSE comparison. Forecast CRPS was the predeclared selection metric; these secondary outcomes remain part of the result and did not trigger model revision.

| Contrast | Mean relative CRPS reduction from baseline |
|---|---:|
| learned versus raw | 4.320% |
| learned versus median | 0.926% |

**Per-seed evidence and uncertainty.** Intervals resample 2,048 whole paired trajectories 10,000 times, retaining all horizons. They condition on fitted models, preprocessing, and saved draws; they do not resample training or account for method search. They are unadjusted across comparisons and metrics. Three seeds provide limited evidence about training variation.

| Seed | Raw CRPS | Learned CRPS | Median CRPS | Learned minus raw (95% interval) | Learned minus median (95% interval) |
|---|---:|---:|---:|---|---|
| seed_29101 | 0.285171 | 0.272355 | 0.273320 | -0.012816 [-0.016426, -0.009111] | -0.000965 [-0.003720, +0.001781] |
| seed_29102 | 0.285793 | 0.272739 | 0.274644 | -0.013054 [-0.016617, -0.009551] | -0.001905 [-0.005414, +0.001612] |
| seed_29103 | 0.283475 | 0.272435 | 0.277207 | -0.011041 [-0.014068, -0.008004] | -0.004773 [-0.007421, -0.002097] |

| Method | Metric | Mean | Seed sample SD | Seed range |
|---|---|---:|---:|---|
| raw | crps | 0.284813 | 0.001200 | 0.283475–0.285793 |
| raw | mse | 0.485661 | 0.024033 | 0.465841–0.512393 |
| raw | coverage_90 | 0.739970 | 0.123462 | 0.597412–0.812195 |
| raw | interval_width_90 | 1.039768 | 0.270360 | 0.744302–1.274790 |
| learned | crps | 0.272510 | 0.000203 | 0.272355–0.272739 |
| learned | mse | 0.492595 | 0.045986 | 0.459979–0.545192 |
| learned | coverage_90 | 0.754781 | 0.078432 | 0.676025–0.832886 |
| learned | interval_width_90 | 0.929023 | 0.171475 | 0.830009–1.127025 |
| median | crps | 0.275057 | 0.001976 | 0.273320–0.277207 |
| median | mse | 0.474049 | 0.004781 | 0.469001–0.478509 |
| median | coverage_90 | 0.807495 | 0.025216 | 0.779602–0.828674 |
| median | interval_width_90 | 1.158907 | 0.218423 | 0.957233–1.390909 |

| Seed | Contrast | Metric | Paired difference | 95% interval |
|---|---|---|---:|---|
| seed_29101 | learned-minus-raw | crps | -0.012816 | [-0.016426, -0.009111] |
| seed_29101 | learned-minus-raw | mse | -0.039780 | [-0.049502, -0.030536] |
| seed_29101 | learned-minus-raw | coverage_90 | -0.054871 | [-0.063538, -0.046387] |
| seed_29101 | learned-minus-raw | interval_width_90 | -0.270175 | [-0.283480, -0.257122] |
| seed_29101 | learned-minus-median | crps | -0.000965 | [-0.003720, +0.001781] |
| seed_29101 | learned-minus-median | mse | -0.005896 | [-0.010902, -0.001025] |
| seed_29101 | learned-minus-median | coverage_90 | -0.058777 | [-0.066345, -0.051453] |
| seed_29101 | learned-minus-median | interval_width_90 | -0.127198 | [-0.139313, -0.115024] |
| seed_29101 | median-minus-raw | crps | -0.011851 | [-0.015155, -0.008571] |
| seed_29101 | median-minus-raw | mse | -0.033884 | [-0.042669, -0.025439] |
| seed_29101 | median-minus-raw | coverage_90 | +0.003906 | [-0.003113, +0.010864] |
| seed_29101 | median-minus-raw | interval_width_90 | -0.142977 | [-0.155075, -0.130938] |
| seed_29102 | learned-minus-raw | crps | -0.013054 | [-0.016617, -0.009551] |
| seed_29102 | learned-minus-raw | mse | -0.005862 | [-0.011489, -0.000254] |
| seed_29102 | learned-minus-raw | coverage_90 | +0.078613 | [+0.068481, +0.088991] |
| seed_29102 | learned-minus-raw | interval_width_90 | +0.085707 | [+0.066659, +0.104558] |
| seed_29102 | learned-minus-median | crps | -0.001905 | [-0.005414, +0.001612] |
| seed_29102 | learned-minus-median | mse | -0.009022 | [-0.015303, -0.002783] |
| seed_29102 | learned-minus-median | coverage_90 | -0.103577 | [-0.112794, -0.094482] |
| seed_29102 | learned-minus-median | interval_width_90 | -0.298569 | [-0.318507, -0.278745] |
| seed_29102 | median-minus-raw | crps | -0.011149 | [-0.014557, -0.007806] |
| seed_29102 | median-minus-raw | mse | +0.003160 | [-0.003153, +0.009425] |
| seed_29102 | median-minus-raw | coverage_90 | +0.182190 | [+0.172180, +0.192261] |
| seed_29102 | median-minus-raw | interval_width_90 | +0.384276 | [+0.361601, +0.407219] |
| seed_29103 | learned-minus-raw | crps | -0.011041 | [-0.014068, -0.008004] |
| seed_29103 | learned-minus-raw | mse | +0.066443 | [+0.048343, +0.085808] |
| seed_29103 | learned-minus-raw | coverage_90 | +0.020691 | [+0.013611, +0.027954] |
| seed_29103 | learned-minus-raw | interval_width_90 | -0.147765 | [-0.168271, -0.127916] |
| seed_29103 | learned-minus-median | crps | -0.004773 | [-0.007421, -0.002097] |
| seed_29103 | learned-minus-median | mse | +0.070556 | [+0.052921, +0.089901] |
| seed_29103 | learned-minus-median | coverage_90 | +0.004211 | [-0.002502, +0.010986] |
| seed_29103 | learned-minus-median | interval_width_90 | -0.263884 | [-0.282365, -0.245228] |
| seed_29103 | median-minus-raw | crps | -0.006268 | [-0.008888, -0.003693] |
| seed_29103 | median-minus-raw | mse | -0.004113 | [-0.009023, +0.000769] |
| seed_29103 | median-minus-raw | coverage_90 | +0.016479 | [+0.009705, +0.023132] |
| seed_29103 | median-minus-raw | interval_width_90 | +0.116118 | [+0.101170, +0.130878] |

**Per-horizon results.** These are equal-seed means. Step 1 is observation time 32 and step 8 is time 39. Learned CRPS is lower than raw at every horizon. Its advantage over median is confined to steps 1 through 6; median is lower at steps 7 and 8. Median has lower mean MSE at every horizon. All four metrics and seed ranges are also visible in the inspected per-horizon figure.

| Forecast step | Method | CRPS | Ensemble-mean MSE | 90% coverage | 90% width |
|---:|---|---:|---:|---:|---:|
| 1 | raw | 0.270818 | 0.466430 | 74.53% | 1.057134 |
| 1 | learned | 0.262809 | 0.474365 | 75.16% | 0.988356 |
| 1 | median | 0.269007 | 0.459221 | 79.59% | 1.203439 |
| 2 | raw | 0.275179 | 0.471197 | 74.07% | 1.030226 |
| 2 | learned | 0.267205 | 0.475839 | 74.50% | 0.928106 |
| 2 | median | 0.274028 | 0.472179 | 79.23% | 1.204555 |
| 3 | raw | 0.277134 | 0.465188 | 74.24% | 1.000133 |
| 3 | learned | 0.266816 | 0.481064 | 75.10% | 0.894486 |
| 3 | median | 0.271139 | 0.460758 | 80.88% | 1.193673 |
| 4 | raw | 0.276063 | 0.470335 | 75.46% | 1.052987 |
| 4 | learned | 0.264176 | 0.481817 | 76.95% | 0.846353 |
| 4 | median | 0.267127 | 0.466772 | 83.22% | 1.277283 |
| 5 | raw | 0.288618 | 0.483582 | 74.90% | 1.105038 |
| 5 | learned | 0.272515 | 0.470806 | 77.60% | 0.959257 |
| 5 | median | 0.273338 | 0.460628 | 82.86% | 1.255628 |
| 6 | raw | 0.300116 | 0.517643 | 71.96% | 1.022484 |
| 6 | learned | 0.281850 | 0.511322 | 75.57% | 0.897759 |
| 6 | median | 0.284494 | 0.495707 | 81.12% | 1.084039 |
| 7 | raw | 0.291206 | 0.495874 | 72.97% | 1.015988 |
| 7 | learned | 0.278179 | 0.510650 | 74.64% | 0.938976 |
| 7 | median | 0.276352 | 0.479495 | 79.85% | 1.010400 |
| 8 | raw | 0.299374 | 0.515039 | 73.86% | 1.034152 |
| 8 | learned | 0.286528 | 0.534894 | 74.30% | 0.978891 |
| 8 | median | 0.284974 | 0.497631 | 79.25% | 1.042238 |

**Resource matching.** The ceiling is 173,320,595,046,400 counted dense matrix operations per method and seed. One multiplication and one addition count as two operations. The budget includes all four mediator candidates, masked reconstruction validation, preprocessing, forecaster fitting, all-step noise validation, and every forecast selection pass. Counts were independently derived and audited using actual forward and backward hooks. Nonmatrix arithmetic, optimizer work, median comparisons, memory traffic, and I/O are excluded from this unit; wall times are reported separately.

| Method | Forecaster epochs | Updates | Total charged matrix work (trillion) | Unused ceiling |
|---|---:|---:|---:|---:|
| raw | 1200 | 4800 | 173.320595046 | 0.000000% |
| learned | 1192 | 4768 | 173.313085145 | 0.004333% |
| median | 1200 | 4800 | 173.320595046 | 0.000000% |

| Method | Auxiliary search + processing (trillion) | Forecaster optimization (trillion) | All-step noise validation (trillion) | Forecast selection (trillion) |
|---|---:|---:|---:|---:|
| raw | 0.000000000 | 19.486841242 | 32.500088832 | 121.333664973 |
| learned | 0.122402374 | 19.356928967 | 32.500088832 | 121.333664973 |
| median | 0.000000000 | 19.486841242 | 32.500088832 | 121.333664973 |

Validation work is identical and cancels in the comparison. Considering only forecaster optimization plus auxiliary search/preprocessing, learned uses 0.038538% less counted work than raw. Thus the close match does not rely on including the common, expensive validation workload in the denominator.

The predeclared allocation ceiling covers model development. Held-out evaluation is separate: each method uses 138.667045683 trillion matrix operations per seed for the same test sampler workload; learned test preprocessing adds 0.000645923 trillion. Adding these costs gives 311.987640730 trillion for raw/median and 311.980776751 trillion for learned (0.002200% less). This supplementary accounting does not change the frozen allocation or checkpoint selection. The timing pilot, source profiling, smoke tests, and implementation checks are research overhead outside these model budgets.

All three seeds selected width-64 Huber mediators, so these allocations are identical across seeds. Each method has seven forecast-CRPS candidates: 10, 50, 100, 200, 400, 800, and its final epoch. All 256 validation trajectories receive 64 draws per candidate. All-step denoising validation has 120 passes per method. Only forecast CRPS selects forecaster checkpoints. LR milestones occur at 75% and 90% of each method's allotted epochs.

| Seed | Raw selected epoch | Learned selected epoch | Median selected epoch | Mediator selected candidate / epoch |
|---|---:|---:|---:|---|
| 29101 | 200 | 800 | 400 | candidate_03_64_huber / 130 |
| 29102 | 1200 | 1192 | 800 | candidate_03_64_huber / 160 |
| 29103 | 400 | 200 | 400 | candidate_03_64_huber / 200 |

Completing the budgets or selecting the final checkpoint does not establish convergence. This tests fixed pipelines at one budget. It does not independently vary mediator effort, forecaster effort, or tuning policy to locate an optimum. Fresh training data and initializations also differ from the earlier run, so a changed outcome cannot be attributed solely to the eight-epoch allocation adjustment.

**Compute costs.** One MPS worker and one Torch thread were used on an Apple M1 Max with 64 GiB memory, with method order rotated across seeds; [environment versions](environment.json) are recorded. The data-free [batch pilot](batch_timing_pilot.json) selected 1,024 equally for all methods. Forecaster fitting times already include validation sampling; the validation column below is a subset and must not be added again.

| Method | Forecaster fit incl. validation (s/seed) | Forecast validation subset (s/seed) | Auxiliary fit/search (s/seed) | Train/val processing (s/seed) | Test inference (s/seed) | Test processing (s/seed) |
|---|---:|---:|---:|---:|---:|---:|
| raw | 745.540393 | 329.681187 | 0.000000 | 0.000005 | 368.974557 | 0.000021 |
| learned | 744.069075 | 328.388134 | 6.309135 | 0.001490 | 372.095896 | 0.006071 |
| median | 746.635248 | 329.770838 | 0.000000 | 0.002908 | 370.914834 | 0.004019 |

Training/selection phase: 112.14 minutes. Fresh-test evaluation phase: 55.72 minutes. Timing pilot, smoke testing, coding, analysis, and verification are separate research overhead. Measured wall times need not be identical under a matched matrix-work budget. Deployment still includes a small additional mediator or median preprocessing cost.

**Data, leakage controls, and reproduction.** The unchanged noisy-periodic generator draws independent amplitude Uniform(0.7,1.3), period Uniform(24,40), and phase Uniform(0,2π); the harmonic is 0.2 sin(2θ+0.4). Observations add Gaussian noise SD 0.15 and an independent Gaussian SD 2 spike with probability 0.1. Each split contains independent trajectories. Each history has 32 integer-time values and each future has eight original noisy observations. The mediator receives only the six fixed neighbors excluding the center, plus boundary availability masks; median uses edge padding on history alone. No future or clean values enter preprocessing. No normalization, densification, GP weights, or test-based tuning is used.

All forecasters use four residual blocks, 64 hidden channels, eight attention heads, 128-dimensional time and diffusion embeddings, feature embedding 16, feedforward width 64, and dropout 0.1: 413,521 parameters each. Adam uses LR 0.001, weight decay 1e-6, gradient clipping at 1, and the recovered 50-step quadratic schedule from 1e-4 to 0.5. Initial weights, shuffle streams, and noise streams are paired. All mediators and allocations froze before any forecaster fitting. All models froze globally before any new test generation.

[Representative forecasts](primary/plots/representative_forecasts.svg), [per-horizon performance](primary/plots/per_horizon.svg), [complete validation curves](primary/plots/validation_crps.svg), and the [epoch-50-onward detail](preview/validation_detail.svg) show fixed examples and all horizons. All four rendered plots were inspected; [visual-review notes](plot_review.json) record the observations. PNG previews are saved in `preview/`. Coverage is marginal and finite-ensemble quantiles affect it; narrow intervals do not establish useful calibration.

Independent verification replays every data split and boundary preprocessing, reconstructs mediator selection and allocations, checks actual update/forward counts, LR schedules, paired initialization, freeze ordering, and all validation ensembles, and recomputes test CRPS using the pairwise definition. A separate [bootstrap recomputation](bootstrap_recomputation.json) checks all 36 metric intervals. The full implementation suite passed 542 tests; final preservation and source checks are recorded in `final_checks.json`. No commit or push occurred.

```sh
uv run --locked python scripts/train_budgeted_polishing.py \
  --config configs/history_polishing_budget_matched.json \
  --output outputs/budgeted-polishing-reproduction
uv run --locked python scripts/verify_budgeted_polishing.py outputs/budgeted-polishing-reproduction
```
Use a fresh output directory. Full source snapshots, exact samples, configurations, logs, checkpoints, and timings are retained.
