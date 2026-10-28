**Final outcome — 27 October 2026: stopped at the resolution-transfer gate.** The larger waveform densifier learned useful interpolation at its training geometry after the sole permitted 600-epoch adjustment: validation CRPS was **0.018464 learned versus 0.097639 linear**, an **81.09% reduction**. However, the frozen model failed both predeclared transfer thresholds in every seed for both application geometries. History-transfer CRPS averaged **0.050077 versus 0.032411**; anchored-future interpolation CRPS averaged **0.065359 versus 0.032877**. **No larger scientific forecasting comparison or fresh forecast test generation proceeded.** This is a successful training-geometry validation followed by poor frozen transfer, not a forecast verdict or convergence claim. [OUTCOME.json](OUTCOME.json) records the stopping decision. The earlier [primary waveform forecasting report](../waveform-primary-20261027/REPORT.md) and its artifacts remain unchanged.

This follow-up asks whether the larger configuration previously successful on GP imputation can first learn useful waveform interpolation. Its [protocol](protocol.json), [implementation documentation](../../docs/waveform_imputation_large.md), and [attempt-01 frozen configuration](attempt_01_gp_recipe/config.json) record the recipe and decision rules. Model selection used development validation; separate fresh diagnostics evaluated resolution transfer. Neither stage is a forecasting test.

**Recipe and data.** The recovered `reference_size` recipe comes from `outputs/csdi-gp-better/plan.json` and `selection.json`; their hashes are recorded in the protocol. Each CSDI densifier has 413,521 parameters: four residual blocks, 64 hidden channels, eight attention heads, time and diffusion embeddings of dimension 128, feature embedding dimension 16, feedforward dimension 64, and dropout 0.1. Univariate feature attention is bypassed by the existing implementation. **No GP weights are reused**: each waveform model starts from its recorded fresh random initialization.

Attempt 01 uses Adam, learning rate 0.001, batch size 64, weight decay `1e-6`, gradient clipping at 1, and 50 quadratic diffusion steps with betas from `1e-4` to `0.5`. It runs 200 epochs, or 3,200 optimizer updates per seed, with tenfold learning-rate drops after epochs 150 and 180. All-step denoising validation evaluates all 50 diffusion steps every ten epochs. Checkpoint selection instead minimizes hidden-point empirical CRPS from 64 sampled reconstructions of all 256 validation trajectories, evaluated sparsely at epochs 50, 100, 150, and 200. Denoising-loss monitoring does not select an intermediate unscored checkpoint. The custom validation callback preserves training RNG state and model mode.

Seeds are 27101, 27102, and 27103. Each has 1,024 training and 256 validation trajectories with 33 original-grid measurements at spacing 0.1. The task observes the 17 even-indexed points, spaced by 0.2, and reconstructs the 16 hidden odd-indexed points. The original `train_validation.npz` files were copied **byte-for-byte**, preserving raw-file SHA-256 hashes; the full hashes appear in both attempt configurations and verification. Their prefixes are `5dd4e5ece495` (27101), `d1145fb2b9cf` (27102), and `89280516e463` (27103). There is no normalization, latent input, or fine-grid truth in training or validation selection.

The successful GP run used two 17-point windows per independent path; this waveform task retains the required single 33-point window. Matching the network and optimization recipe therefore does not make the distributions or interpolation geometries identical.

The validation gate requires learned CRPS below linear in every seed, at least 5% relative reduction calculated as `1 - mean_seed(learned CRPS) / mean_seed(linear CRPS)`, and a negative upper endpoint for every seed's paired 95% CRPS-difference interval. The threshold and maximum of two attempts were recorded before candidate scores were inspected. Attempt 01 has a relative reduction of **−325.465%**, equivalent to learned CRPS being 4.255 times linear. All three performance conditions fail; seed-inventory checks pass.

**Attempt 01 validation results.** Scores use only hidden points. MSE scores the ensemble mean; coverage and width use empirical 5th/95th percentiles of the 64 draws. Linear interpolation averages adjacent observations and repeats the deterministic estimate. Its zero-width intervals have zero coverage here; this is not a probabilistic-calibration benchmark. The following variation is descriptive across the three independent training/dataset seeds, separate from trajectory-bootstrap uncertainty.

| Metric | Method | Equal-seed mean | Seed sample SD | Seed range |
|---|---|---:|---:|---:|
| CRPS | Learned | 0.415418 | 0.010856 | 0.402986–0.423031 |
| CRPS | Linear | 0.097639 | 0.002617 | 0.095087–0.100317 |
| Ensemble-mean MSE | Learned | 0.553344 | 0.023999 | 0.526083–0.571285 |
| Ensemble-mean MSE | Linear | 0.015445 | 0.000809 | 0.014814–0.016357 |
| 90% interval coverage | Learned | 0.848551 | 0.006607 | 0.843262–0.855957 |
| 90% interval coverage | Linear | 0 | 0 | 0–0 |
| 90% interval width | Learned | 2.054697 | 0.008072 | 2.046897–2.063015 |
| 90% interval width | Linear | 0 | 0 | 0–0 |

| Seed | Learned CRPS | Linear CRPS | Learned minus linear | Paired 95% interval |
|---|---:|---:|---:|---:|
| 27101 | 0.420236 | 0.095087 | +0.325149 | [+0.317169, +0.333178] |
| 27102 | 0.402986 | 0.097512 | +0.305475 | [+0.297868, +0.313016] |
| 27103 | 0.423031 | 0.100317 | +0.322714 | [+0.315086, +0.330235] |

The mean CRPS difference is +0.317779, with descriptive seed SD 0.010725 and range +0.305475 to +0.325149. Intervals use 10,000 paired trajectory-bootstrap replicates, retaining each trajectory's hidden times and saved ensemble together. Bootstrap seeds are 8675309, 8675409, and 8675509. They are conditional on fitted models and saved draws, and are unadjusted across metrics, seeds, checkpoints, and attempts. Repeated validation selection can be optimistic; neither draws nor hidden times count as independent training replicates. Finite ensembles also affect estimated quantiles. Full per-trajectory, per-time, aggregate, seed-variation, and all-metric interval results are in [imputation_analysis.json](attempt_01_gp_recipe/imputation_analysis.json).

All three models select the final scored epoch, 200. Their CRPS trajectories are:

| Seed | Epoch 50 | Epoch 100 | Epoch 150 | Epoch 200 |
|---|---:|---:|---:|---:|
| 27101 | 0.422537 | 0.421290 | 0.421202 | 0.420236 |
| 27102 | 0.424441 | 0.421828 | 0.415510 | 0.402986 |
| 27103 | 0.425418 | 0.425474 | 0.424007 | 0.423031 |

Denoising losses fall sharply early and then improve slowly. CRPS improvements are small relative to the linear gap, although seed 27102 still improves appreciably late. Endpoint selection and a flattened curve under scheduled learning-rate decay do not establish convergence. The [training curves](attempt_01_gp_recipe/imputation_plots/training_curves.svg) explicitly show orange linear baselines so small movements near 0.42 are seen relative to baselines near 0.10. [First-six-trajectory reconstructions](attempt_01_gp_recipe/imputation_plots/validation_reconstructions.svg) show learned hidden-point means pulled toward central values, with broad bands, while linear interpolation follows the displayed waveforms closely. [Hidden-time score curves](attempt_01_gp_recipe/imputation_plots/validation_per_time.svg) show the learned CRPS/MSE deficit across the grid. All plots were rendered and visually inspected; [QA notes](attempt_01_plot_qa.txt) and [preview images](preview_attempt01/) are outside the frozen attempt directory.

**Bounded conditioning diagnostic.** After the failure, frozen selected models were sampled on the first 32 training and first 32 validation trajectories per seed, with 64 draws. Correct observations were compared with a one-row cyclic permutation of observations within the same split, using identical private sampling-noise streams. Targets remained the original row's hidden measurements. No training, checkpoint selection, test access, or fine-grid generation occurred; supplied observations were preserved and protected existing artifacts retained their hashes. This restricted descriptive diagnostic is saved in [conditioning/scores.json](attempt_01_gp_recipe/diagnostics/conditioning/scores.json).

| Split | Seed | Learned, correct CRPS | Learned, permuted CRPS | Permuted minus correct | Linear, correct CRPS |
|---|---|---:|---:|---:|---:|
| Train | 27101 | 0.428533 | 0.455241 | +0.026708 | 0.102512 |
| Train | 27102 | 0.415565 | 0.457329 | +0.041764 | 0.101020 |
| Train | 27103 | 0.414936 | 0.446318 | +0.031382 | 0.098144 |
| Train | Mean | 0.419678 | 0.452963 | +0.033285 | 0.100558 |
| Validation | 27101 | 0.422241 | 0.465624 | +0.043383 | 0.092095 |
| Validation | 27102 | 0.398162 | 0.449417 | +0.051254 | 0.096496 |
| Validation | 27103 | 0.428670 | 0.458229 | +0.029559 | 0.104052 |
| Validation | Mean | 0.416358 | 0.457756 | +0.041399 | 0.097548 |

Correct-conditioning learned MSE averages 0.558983 on training and 0.558119 on validation, versus linear 0.016529 and 0.015396. Permuting observations also degrades linear CRPS to 0.824486 and 0.813904. Learned ensemble means respond to changed observations, with mean absolute changes 0.075247 and 0.091695 on the two splits. Thus conditioning has a measurable effect in this check, but correct conditioning still produces poor interpolation even on the sampled training trajectories. This does not identify a causal failure mechanism, prove convergence, or establish a general impossibility result. Review found no concrete mask, schedule, or sampler defect. The [recorded decision](attempt_02_decision.json) therefore invokes the sole predeclared longer-budget adjustment rather than using new test or fine-grid evidence.

**Costs and verification of attempt 01.** Execution used an Apple M1 Max with 64 GiB memory, CPU, three parallel seed workers, and one Torch thread per worker. The [environment](environment.json) records Python 3.12.15, NumPy 2.5.3, and Torch 2.14.1; no new dependency was installed. Times below are worker elapsed times under concurrent execution, not isolated single-worker benchmarks.

| Seed | Training total, including validation (s) | CRPS validation sampling subset (s) |
|---|---:|---:|
| 27101 | 1,340.82 | 1,039.19 |
| 27102 | 1,345.40 | 1,038.11 |
| 27103 | 1,346.06 | 1,037.16 |
| Mean | 1,344.09 | 1,038.15 |

The three-worker training phase took 1,347.08 seconds wall time, about 22.45 minutes. Each full CRPS validation sample pass cost about 259.54 seconds per seed on average; four passes account for most of the total. The remaining mean 305.94 seconds includes optimization, all-step denoising validation, and overhead, so it is not pure optimization time. The separate conditioning diagnostic took 131.07 seconds wall time. Per-seed [summaries](attempt_01_gp_recipe/seed_27101/summary.json), candidate scores, epoch logs, and [TRAINED.json](attempt_01_gp_recipe/TRAINED.json) retain the measured costs.

[Independent verification passed](attempt_01_gp_recipe/verification.json): it reproduced fresh initialization hashes, checked the recovered recipe, parameter/update counts and validation cadence, verified original data copies and exact preserved observations, recomputed selected-sample metrics and the paired gate, and checked frozen configuration, analysis, and checkpoint hashes. It performed no new training or inference and loaded no GP weights. [DENSIFIERS_FROZEN.json](attempt_01_gp_recipe/DENSIFIERS_FROZEN.json) preserves the failed decision; a freeze marker does not mean the gate passed. Source snapshots, Git metadata, configurations, seeds, every CRPS candidate ensemble, and selected checkpoints are retained under each attempt directory.

**Attempt 02 — completed validation gate passes.** The [600-epoch configuration](attempt_02_longer/config.json) retains the architecture, byte-identical data files, initial paired seeds, optimizer, and diffusion recipe. It started afresh, with no GP weights or resumed decayed optimizer, and completed 9,600 updates per seed. Learning-rate milestones moved to 450 and 540. All-step denoising monitoring remained every ten epochs; CRPS was scored at 150, 300, 450, and 600. Both attempts therefore had exactly four CRPS checkpoint opportunities. This cadence was bounded before attempt 01 completed, to control the substantial sampling cost.

Every seed selects epoch 600. The same hidden-point estimands, 64-draw ensembles, and 10,000-replicate paired trajectory bootstrap used above yield:

| Metric | Learned mean | Learned seed SD | Learned seed range | Linear mean |
|---|---:|---:|---:|---:|
| CRPS | 0.018464 | 0.000817 | 0.017938–0.019405 | 0.097639 |
| Ensemble-mean MSE | 0.001071 | 0.000107 | 0.001000–0.001193 | 0.015445 |
| 90% interval coverage | 0.845459 | 0.010289 | 0.835693–0.856201 | 0 |
| 90% interval width | 0.095322 | 0.007422 | 0.090749–0.103886 | 0 |

Linear's seed variation is unchanged from the attempt-01 table because the data and deterministic baseline are identical. The CRPS ratio-of-means reduction is 81.088959%, exceeding the 5% gate threshold.

| Seed | Learned CRPS | Linear CRPS | Learned minus linear | Paired 95% interval |
|---|---:|---:|---:|---:|
| 27101 | 0.018049 | 0.095087 | −0.077038 | [−0.080197, −0.073868] |
| 27102 | 0.019405 | 0.097512 | −0.078106 | [−0.081119, −0.075160] |
| 27103 | 0.017938 | 0.100317 | −0.082378 | [−0.085699, −0.079093] |

The mean CRPS difference is −0.079174, with descriptive seed SD 0.002826 and range −0.082378 to −0.077038. These intervals remain conditional on the selected models and saved draws; they do not account for checkpoint/attempt selection or provide an independent test claim. The original-grid validation evidence supports useful interpolation in this configuration. It does not establish transfer performance, general superiority, or nominal calibration. Coverage is about 5.45 percentage points below 90% despite the substantial reduction in prediction error. Finite-ensemble quantiles can contribute to coverage error; no decomposition or calibration correction was attempted.

| Seed | Epoch 150 | Epoch 300 | Epoch 450 | Epoch 600 |
|---|---:|---:|---:|---:|
| 27101 | 0.421202 | 0.089862 | 0.026033 | 0.018049 |
| 27102 | 0.415510 | 0.036713 | 0.026222 | 0.019405 |
| 27103 | 0.424007 | 0.058690 | 0.023252 | 0.017938 |

The [training curves](attempt_02_longer/imputation_plots/training_curves.svg) show a pronounced loss decline after the early plateau, followed by continued improvement. All three final CRPS candidates improve over epoch 450, so completion of 600 epochs still does not prove convergence. The two attempts vary both the training duration and the timing of learning-rate decay; their contrast does not isolate those effects. [Reconstructions](attempt_02_longer/imputation_plots/validation_reconstructions.svg) now closely follow the six fixed representative waveforms and their extrema. [Per-hidden-time metrics](attempt_02_longer/imputation_plots/validation_per_time.svg) show learned mean CRPS below linear throughout the grid, ranging from 0.016874 to 0.023182; MSE ranges from 0.000886 to 0.001692. Per-time mean coverage ranges from 0.807292 to 0.871094, below 0.90 throughout. Errors and widths are slightly higher near the ends. These rendered plots passed [visual QA](attempt_02_plot_qa.txt); [PNG previews](preview_attempt02/) were saved outside the frozen directory. [Full analysis](attempt_02_longer/imputation_analysis.json) contains all per-time/per-trajectory metrics, seed variation, and intervals for all four metrics.

| Seed | Training total, including validation (s) | CRPS validation sampling subset (s) |
|---|---:|---:|
| 27101 | 1,963.48 | 1,041.49 |
| 27102 | 1,961.98 | 1,042.23 |
| 27103 | 1,965.45 | 1,041.70 |
| Mean | 1,963.63 | 1,041.81 |

[Attempt-02 wall time](attempt_02_longer/TRAINED.json) was 1,966.42 seconds, about 32.77 minutes, under the same three-CPU-worker setup. Four CRPS sample passes averaged 260.45 seconds per seed per pass. The mean remaining 921.83 seconds includes optimization, all-step validation, and overhead. Small separate hardware timing pilots also ran during this attempt, so these are observed concurrent-system costs rather than controlled hardware benchmarks. [Independent verification](attempt_02_longer/verification.json) passed after all three seeds completed, reproducing the paired gate and confirming epoch-600 selection, 9,600 updates, 413,521 parameters, fresh initialization hashes, exact preserved observations, and frozen artifact hashes. The [freeze manifest](attempt_02_longer/DENSIFIERS_FROZEN.json) records the passed validation gate before fine-grid transfer began.

**Completed transfer diagnostic — both tasks fail in every seed.** After model/configuration/validation-analysis hashes were frozen, 256 fresh 65-point trajectories per seed were generated at spacing 0.05. The actual application geometries were evaluated separately with 64 draws: history uses 25 known points at spacing 0.1 to fill 24 hidden points, giving 49 points over 0–2.4; the independently imputed future uses nine known original-grid points, including the left anchor, to fill eight hidden points, giving 17 points over a relative 0–0.8. The future task supplies original-grid future values for interpolation; it is not prediction from history alone. Both tasks use slices of the same fresh paths and are not pooled as independent observations. Fine-grid simulator observations only scored the frozen models; they never entered training, checkpoint choice, or later tuning.

The [predeclared transfer gate](attempt_02_longer/TRANSFER.json) requires CRPS ratio to linear at most 1.25 and ensemble-mean MSE ratio at most 1.5, for every seed and both tasks. All six task/seed pairs exceed both thresholds. In the table below, “mean ± SD [range]” describes the three seed point estimates; these are not confidence intervals. Linear coverage and width are zero in both tasks because its ensemble is deterministic.

| Task | Metric | Learned mean ± seed SD [range] | Linear mean ± seed SD [range] |
|---|---|---|---|
| History | CRPS | 0.050077 ± 0.002171 [0.048049, 0.052366] | 0.032411 ± 0.000424 [0.031924, 0.032697] |
| History | MSE | 0.007445 ± 0.000415 [0.007029, 0.007858] | 0.001658 ± 0.000034 [0.001619, 0.001682] |
| History | 90% coverage | 0.679688 ± 0.049908 [0.624512, 0.721680] | 0 ± 0 [0, 0] |
| History | 90% width | 0.171662 ± 0.009422 [0.161681, 0.180402] | 0 ± 0 [0, 0] |
| Anchored future | CRPS | 0.065359 ± 0.003029 [0.062617, 0.068611] | 0.032877 ± 0.000386 [0.032611, 0.033320] |
| Anchored future | MSE | 0.012129 ± 0.000582 [0.011589, 0.012746] | 0.001698 ± 0.000030 [0.001673, 0.001731] |
| Anchored future | 90% coverage | 0.580566 ± 0.043752 [0.535645, 0.623047] | 0 ± 0 [0, 0] |
| Anchored future | 90% width | 0.167762 ± 0.013100 [0.157960, 0.182641] | 0 ± 0 [0, 0] |

| Task | Seed | Learned/linear CRPS | Learned/linear MSE | CRPS difference | Paired 95% CRPS-difference interval |
|---|---|---:|---:|---:|---:|
| History | 27101 | 1.469511 | 4.199907 | +0.015352 | [+0.013977, +0.016765] |
| History | 27102 | 1.527539 | 4.429005 | +0.017204 | [+0.015542, +0.018908] |
| History | 27103 | 1.640365 | 4.852917 | +0.020443 | [+0.018855, +0.022105] |
| Anchored future | 27101 | 1.879259 | 6.693239 | +0.029297 | [+0.026823, +0.031881] |
| Anchored future | 27102 | 1.983150 | 7.202000 | +0.032150 | [+0.029393, +0.034991] |
| Anchored future | 27103 | 2.103951 | 7.540243 | +0.036000 | [+0.033170, +0.038814] |

Differences are learned minus linear. Mean CRPS differences are +0.017666 for history (seed SD 0.002577; range +0.015352 to +0.020443) and +0.032482 for anchored future (SD 0.003364; range +0.029297 to +0.036000). The gate uses the prespecified point-estimate ratios; bootstrap intervals provide separate uncertainty summaries. Each uses 10,000 paired whole-trajectory resamples, conditional on the frozen validation-selected model and saved 64-draw ensembles, with no multiplicity adjustment. History bootstrap seeds are 8675310/8675410/8675510; future seeds are 8675311/8675411/8675511. Both CRPS and MSE difference intervals are positive for every seed/task. [transfer_analysis.json](attempt_02_longer/transfer_analysis.json) contains all intervals, per-seed absolute scores, per-time/per-trajectory metrics, and descriptive variation, separately from the unchanged frozen validation analysis.

Coverage further deteriorates to 67.97% for history and 58.06% for anchored future, despite wider intervals than in original-grid validation. Finite 64-draw quantiles affect coverage, but no calibration correction or causal attribution is made. [History reconstructions](attempt_02_longer/imputation_plots/transfer_history_reconstructions.svg) show local offsets between preserved observations; [history per-time metrics](attempt_02_longer/imputation_plots/transfer_history_per_time.svg) show a particularly poor first hidden point and worsening near the right boundary. [Anchored-future reconstructions](attempt_02_longer/imputation_plots/transfer_future_reconstructions.svg) show local kinks and offsets, while [future per-time metrics](attempt_02_longer/imputation_plots/transfer_future_per_time.svg) remain worse than linear across the hidden grid. All four actual figures were rendered and inspected; [transfer QA](transfer_plot_qa.txt) and [PNG previews](preview_transfer/) are outside the attempt directory. QA notes the coarse two-decimal MSE tick labels; precise values are in the table/JSON.

Transfer sampling cost averaged 402.52 seconds per seed for history (range 402.21–402.80) and 133.56 seconds for anchored future (131.82–134.75). The three-worker phase took **538.06 seconds wall time**, about 8.97 minutes. Summing the measured initial training, conditioning diagnostic, longer training, and transfer phase times gives approximately **3,982.63 seconds / 66.38 minutes**, excluding coding, analysis/verification, and separate timing-pilot phases. Concurrent pilot activity can still influence the observed training wall time. The [final independent verifier](attempt_02_longer/verification.json) passed artifact checks and recomputed the failed transfer gate; passing verification means the artifacts and calculation agree, not that the scientific gate passed.

The predeclared stop rule was followed: no tuning after fine-grid truth, no larger scientific forecaster training/comparison, and no fresh scientific forecast test trajectories. The evidence is limited to useful validation interpolation at the trained geometry and inadequate frozen transfer to these two finer-grid geometries. Spacing and window length change together, so this diagnostic does not isolate their effects. It neither rules out other training strategies nor identifies a unique failure mechanism, establishes convergence, or supplies a forecast-performance verdict.

[Implementation checks](implementation_checks.json) record **434 tests passed**, Ruff passed, and `git diff --check` passed. A later formatting-only test change did not alter behavior. The [final implementation snapshot](final_source_snapshot/) supplements the retained per-attempt source snapshots. Scientific artifact verification is separate from these software checks. [Prior-result verification](prior_results_verification.json) confirms all **121 protected earlier files remain unchanged**.

To reproduce attempt 01 from the repository root, use a fresh destination so frozen artifacts remain intact:

```sh
RUN=outputs/waveform-large-reproduction/attempt_01_gp_recipe
.venv/bin/python scripts/train_waveform_imputation.py --config configs/waveform_imputation_large.json --output "$RUN" --data-source outputs/waveform-primary-20261027 --recipe-source outputs/csdi-gp-better
.venv/bin/python scripts/analyze_waveform_imputation.py "$RUN"
.venv/bin/python scripts/train_waveform_imputation.py --phase freeze --output "$RUN"
.venv/bin/python scripts/verify_waveform_imputation.py "$RUN"
# Only after a failed validation gate:
.venv/bin/python scripts/diagnose_waveform_imputation.py --attempt "$RUN"
```

The sole longer attempt uses `configs/waveform_imputation_large_longer.json` with a separate fresh output directory, followed by the same analysis, freeze, and verification sequence. Only after its validation gate passes, reproduce the separate transfer phase:

```sh
RUN=outputs/waveform-large-reproduction/attempt_02_longer
.venv/bin/python scripts/train_waveform_imputation.py --phase transfer --output "$RUN"
.venv/bin/python scripts/analyze_waveform_imputation.py --transfer "$RUN"
.venv/bin/python scripts/verify_waveform_imputation.py "$RUN"
```

The [documentation](../../docs/waveform_imputation_large.md) describes the guarded phases and recorded outcome. Relevant entry points are [training](../../scripts/train_waveform_imputation.py), [analysis](../../scripts/analyze_waveform_imputation.py), [independent verification](../../scripts/verify_waveform_imputation.py), and [conditioning diagnosis](../../scripts/diagnose_waveform_imputation.py). No commit or push was made.
