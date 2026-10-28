**Synthetic waveform forecasting: completed primary comparison**  
27 October 2026 · Seeds 27101, 27102, 27103

**Learned Densiffusion lost to direct CSDI and linear densification in all three seeds.** Mean original-grid CRPS was **0.365331** for learned Densiffusion, **0.146844** for direct CSDI, and **0.136779** for linear densification. Learned-minus-baseline trajectory-bootstrap intervals were positive in every seed. This is the full requested three-seed primary comparison, with 1,024 training, 256 validation, and 1,024 test trajectories per seed; it is not a pilot. The integration smoke run was completed separately and supplies no accuracy evidence.

The implementation adds a reproducible waveform generator, paired experiment runner, original-grid validation-CRPS checkpoint selection, saved-sample analysis, artifact verification, and forecast/validation plots. Existing results were preserved. No commit or push was made.

All values below use the eight original forecast times. The first table gives means across the three seeds; seed variability and within-seed bootstrap uncertainty are reported separately.

| Method | CRPS ↓ | MSE ↓ | 90% coverage | 90% width |
|---|---:|---:|---:|---:|
| Direct CSDI | 0.146844 | 0.085834 | 0.880819 | 0.866081 |
| Linear + CSDI | 0.136779 | 0.073357 | 0.825887 | 0.736608 |
| Learned Densiffusion | 0.365331 | 0.448605 | 0.851847 | 1.916089 |

Coverage is marginal coverage of linearly interpolated empirical 5th/95th percentiles. Direct CSDI was closest to 90% nominal coverage (88.08%). Linear interpolation improved mean CRPS but covered only 82.59%; learned Densiffusion covered 85.18% despite much wider intervals. Width alone is not a calibration criterion.

Descriptive seed variation is **sample standard deviation [minimum, maximum]**. These are not confidence intervals and combine variation in training, validation, and test datasets with model-training randomness. Three seeds do not characterize a general training-variance distribution.

| Metric | Direct CSDI | Linear + CSDI | Learned Densiffusion |
|---|---:|---:|---:|
| CRPS | 0.009088 [0.137007, 0.154928] | 0.009492 [0.128244, 0.147002] | 0.003969 [0.360757, 0.367876] |
| Ensemble-mean MSE | 0.011080 [0.074051, 0.096042] | 0.009877 [0.063757, 0.083489] | 0.009819 [0.437324, 0.455229] |
| 90% coverage | 0.010810 [0.868408, 0.888184] | 0.014374 [0.810669, 0.839233] | 0.012125 [0.844604, 0.865845] |
| 90% interval width | 0.026472 [0.837658, 0.890033] | 0.093284 [0.651352, 0.836248] | 0.028519 [1.883160, 1.932880] |

The individual seed CRPS scores were:

| Seed | Direct CSDI | Linear + CSDI | Learned Densiffusion |
|---|---:|---:|---:|
| 27101 | 0.148596 | 0.147002 | 0.367876 |
| 27102 | 0.137007 | 0.135092 | 0.360757 |
| 27103 | 0.154928 | 0.128244 | 0.367360 |

Paired CRPS differences below are **first method minus second [95% percentile interval]**. Each seed uses 10,000 paired bootstrap replicates of its 1,024 independent test trajectories, retaining each trajectory's complete forecast ensemble and eight horizons. Positive learned-minus-baseline differences favor the baseline.

| Seed | Learned − direct | Learned − linear | Linear − direct |
|---|---:|---:|---:|
| 27101 | +0.219280 [+0.211434, +0.227170] | +0.220874 [+0.213391, +0.228495] | -0.001594 [-0.004213, +0.001018] |
| 27102 | +0.223750 [+0.216157, +0.231549] | +0.225665 [+0.218038, +0.233339] | -0.001915 [-0.004876, +0.001064] |
| 27103 | +0.212431 [+0.205394, +0.219577] | +0.239116 [+0.231207, +0.246916] | -0.026684 [-0.029517, -0.023816] |

Across seeds, learned-minus-direct CRPS averaged **+0.218487** (seed SD 0.005701; range +0.212431 to +0.223750); learned-minus-linear averaged **+0.228552** (SD 0.009457; range +0.220874 to +0.239116). Linear-minus-direct averaged −0.010064 (SD 0.014394). Although its point estimate was lower in every seed, linear's first two paired intervals include zero; its advantage was substantially larger in seed 27103.

These intervals condition on the fitted models and saved ensembles, and are unadjusted across comparisons and metrics. They do not include training-seed variability or uncertainty from generating new ensembles. The 64 learned draws pool eight futures per sampled history across eight histories: draws sharing a history are dependent. Equal scored draw counts therefore do not imply equal Monte Carlo precision or finite-ensemble CRPS bias. Empirical interval coverage also depends on this finite ensemble. The reported primary metric remains the repository's empirical-ensemble CRPS.

Mean CRPS by horizon shows that the learned method's disadvantage extends across the whole evaluated interval:

| Future time | Direct CSDI | Linear + CSDI | Learned Densiffusion |
|---|---:|---:|---:|
| 2.5 | 0.122570 | 0.113502 | 0.434348 |
| 2.6 | 0.122210 | 0.110442 | 0.355482 |
| 2.7 | 0.115426 | 0.104842 | 0.306738 |
| 2.8 | 0.122062 | 0.119394 | 0.290104 |
| 2.9 | 0.143312 | 0.138171 | 0.309821 |
| 3.0 | 0.162681 | 0.149942 | 0.352930 |
| 3.1 | 0.181201 | 0.166897 | 0.409474 |
| 3.2 | 0.205289 | 0.191044 | 0.463750 |

All per-seed and per-horizon CRPS, MSE, coverage, width, and per-trajectory scores are in [analysis.json](analysis.json). See [per-horizon plots](plots/per_horizon.svg), [representative forecasts](plots/representative_forecasts.svg), and [validation curves](plots/validation_crps.svg). Forecast examples are the first six distinct test trajectories of the first seed, chosen without inspecting scores.

The task used `A[sin(2πft+φ)+0.3 sin(2(2πft+φ)+ψ)]+ε`, with independent trajectory-fixed `A~Uniform(0.7,1.3)`, `f~Uniform(0.6,0.9)`, `φ,ψ~Uniform(0,2π)`, and independent observation noise `ε~Normal(0,0.02²)`. Models received observations only. Each trajectory contributed one history of 25 observations at 0 through 2.4 and eight future observations at 2.5 through 3.2. Independent split RNG streams were recorded; no normalization or fine-grid truth was used in training or selection.

Every forecaster had the same two-layer, 16-hidden-channel CSDI architecture with **17,273 parameters**, identical initial weights within its paired seed, the same example-order stream, optimizer and diffusion schedule, and **600 epochs / 9,600 optimization updates**. Sequence-dependent diffusion noise and dropout were not fully paired across different grid sizes. Adam used learning rate 0.001, weight decay 10⁻⁶, and a 50-step quadratic diffusion schedule from 0.0001 to 0.5. Learning rates dropped tenfold after epochs 450 and 540. Each frozen learned densifier had **17,273 additional parameters** and trained for 400 epochs.

The original construction was preserved: densifiers learned alternating-point reconstruction at **0.2 → 0.1** spacing and were applied at **0.1 → 0.05**. Histories were densified using observed past only. Training futures were densified independently using future observations plus the last history observation as a left anchor; the repeated anchor was removed. Densified forecasters generated 16 fine future points and were evaluated at indices `1::2`, exactly the eight original times; direct CSDI generated the eight original-grid targets. Runtime checks preserved original observations exactly. Independently densified histories and futures were not treated as true joint fine trajectories. Generated history points add no information about the actual future beyond observed history.

All methods had six equally spaced validation candidates, selected by original-grid forecast CRPS using 64 draws on all 256 validation trajectories. All three seeds completed training and selection before checkpoints and configuration were frozen and any confirmatory test observations were generated. No test-based tuning occurred; [pretest_review.json](pretest_review.json) records the decision to retain the fixed budget without claiming convergence.

All nine forecasting checkpoints selected **epoch 600**. Mean validation CRPS at epochs 400 → 500 → 600 was:

| Method | Epoch 400 | Epoch 500 | Epoch 600 | 500 → 600 change |
|---|---:|---:|---:|---:|
| Direct CSDI | 0.168897 | 0.145414 | 0.142970 | -1.680% |
| Linear + CSDI | 0.160937 | 0.133369 | 0.129248 | -3.090% |
| Learned Densiffusion | 0.370486 | 0.365967 | 0.365522 | -0.122% |

Within each seed/method, mean training denoising loss over epochs 551–600 differed from epochs 501–550 by less than 2%; the learned method's changes were between −0.273% and +0.405%. These are within-method comparisons only. Denoising losses on different grids are not comparable forecasting scores. Densifier checkpoints selected epochs **400/350/350**, with validation denoising losses **0.349923/0.302823/0.335025**. Endpoint improvements support a near-plateau under the final learning rate of **10⁻⁵**, not a convergence proof; every forecaster still selected its final candidate. See [validation_review.json](validation_review.json) for complete curves, recent-loss windows, and checkpoint cross-checks.

Costs below are mean wall seconds per seed on an **Apple M1 Max with 64 GB memory**, using **three parallel seed workers and one Torch CPU thread per worker** for training and test evaluation. They include concurrent-workload contention and are not isolated throughput or FLOP measurements. Training totals include all forecast-validation candidates, augmentation, and the learned densifier where applicable; the validation column is a component of that total. Test inference covers 1,024 trajectories with 64 scored draws each.

| Method | Total training | Forecast-validation component | Test inference | Inference / trajectory |
|---|---:|---:|---:|---:|
| Direct CSDI | 331.62 s | 221.38 s | 153.66 s | 0.1501 s |
| Linear + CSDI | 661.13 s | 454.27 s | 300.60 s | 0.2936 s |
| Learned Densiffusion | 791.35 s | 497.86 s | 329.11 s | 0.3214 s |

The learned densifier added **77.91 s** of training per seed. Learned train/validation augmentation sampling took **6.53 s**, and test-history densification took **28.43 s** of the **329.11 s** inference total. The remaining inference time included future sampling and overhead. The baseline densifier costs are zero or negligible deterministic interpolation. Equal forecasting parameter counts and update budgets did not equalize total compute. Detailed cost components are saved in each seed's `timings.json` and in `analysis.json`; [hardware_pilot.json](hardware_pilot.json) records the CPU/MPS timing decision without forecast tuning.

Learned densification did not improve forecasting for this fixed small architecture, training budget, and preserved construction on this synthetic task. The result does not identify whether synthetic future supervision, model capacity, optimization, or sampling quality caused the loss. **The conditional history-only learned-densification ablation was not run:** the primary result was clearly unpromising, so its stated trigger was not met. Any further task or hyperparameter exploration must be labeled exploratory and use fresh test data for a new confirmatory claim.

**The learned densifier was already a poor interpolator at its training geometry.** Post-freeze diagnostics used the first 256 test trajectories per seed and 64 interpolation samples, scoring only hidden positions. The following are mean CRPS values across seeds against noisy observations:

| Separate interpolation task | Learned | Linear |
|---|---:|---:|
| Exact training geometry: 33 original points, 17 observed, 16 hidden, 0.2 → 0.1 | 0.424658 | 0.099076 |
| History window: 25 original points, 13 observed, 12 hidden, 0.2 → 0.1 | 0.426114 | 0.099447 |
| Fine history: 49 points, 25 observed, 24 hidden, 0.1 → 0.05 | 0.423684 | 0.032182 |

The two history diagnostics initially left a window-length mismatch with training. The additional exact-geometry check matched the 33-point window, alternating mask, and spacings and confirmed poor imputation. Spacing transfer alone therefore cannot explain the observed weakness. This weak densifier and the absence of demonstrated convergence substantially limit generalization beyond the trained setup; the run does not test the benefit of an accurate learned interpolator.

Fine-grid truth was generated only for the fine-history diagnostic, after frozen test evaluation. The exact-geometry check is full-window imputation: it conditions on alternating originals across 0–3.2, including some future-period observations, solely for this separate diagnostic. It never constructs a forecasting history, changes a model, or alters any primary forecast. All primary artifacts remained unchanged. The fine-history comparisons also change gap sizes, so they do not independently isolate a resolution-transfer effect. Linear interpolation is a point estimate with zero interval width; its zero coverage in these diagnostics is not a probabilistic calibration benchmark.

Diagnostic ensembles, noisy and noiseless truths, scores, and timings are in [diagnostics/scores.json](diagnostics/scores.json) and [matched-geometry scores](diagnostics/matched_geometry/scores.json). Both sets of scores were recomputed exactly from saved arrays; frozen hashes remained unchanged. The history diagnostics took 244.06 seconds sequentially, and the matched-geometry check took 39.02 seconds across three workers. These additional diagnostic costs are separate from the primary training/inference table above.

Verification passed for all three seeds: frozen hashes, dataset RNG replay, disjoint split IDs, shared initialization hashes, complete epoch logs and budgets, validation argmin checkpoints, finite `(1024,64,8,1)` forecast arrays, exact metric recomputation, and augmentation/inference conditioning audits. Observation-preservation evidence consists of saved runtime checks plus matching conditioning-input hashes; verification did not retrain models or resample forecasts. See [verification.json](verification.json). The saved check run reports **370 tests passed**, Ruff clean, and formatting clean in [checks.txt](checks.txt).

Reproducibility artifacts are all under this directory:

- [config.json](config.json), [protocol.json](protocol.json), [FROZEN.json](FROZEN.json), and [EVALUATED.json](EVALUATED.json): exact settings, seed streams, protocol, frozen hashes, and completion.
- [seed_27101](seed_27101/), [seed_27102](seed_27102/), [seed_27103](seed_27103/): original train/validation arrays and IDs, selected checkpoints, per-epoch logs, validation scores, pairing/augmentation/inference audits, exact scored `samples.npz`, metrics, and timings.
- [source.tar.gz](source.tar.gz): source at training start; [source_evaluation_analysis.tar.gz](source_evaluation_analysis.tar.gz): subsequent evaluation/analysis source; [git_head.txt](git_head.txt) and [git_diff.patch](git_diff.patch): repository baseline and tracked changes.
- [run_commands.txt](run_commands.txt), [console_training.log](console_training.log), and [evaluation.log](evaluation.log): recorded execution; [experiment documentation](../../docs/waveform_forecasting.md): interfaces and interpretation.
- [Smoke verification](../waveform-smoke-20261027/verification.json) and [parallel smoke verification](../waveform-parallel-smoke-20261027/verification.json): separate integration evidence, with no accuracy interpretation.

To rerun from the repository root, use a fresh output directory; completed evaluations refuse overwriting existing scored samples:

```sh
uv run --locked python scripts/train_waveform_forecast.py --smoke --output outputs/waveform-smoke-reproduction
uv run --locked python scripts/train_waveform_forecast.py --config configs/waveform_forecast.json --phase train --output outputs/waveform-primary-reproduction
uv run --locked python scripts/train_waveform_forecast.py --phase evaluate --output outputs/waveform-primary-reproduction
uv run --locked python scripts/analyze_waveform_forecast.py outputs/waveform-primary-reproduction --bootstrap-replicates 10000 --seed 8675309
uv run --locked python scripts/verify_waveform_forecast.py outputs/waveform-primary-reproduction
```

Recomputing the present primary metrics and intervals requires no training or forecast sampling: run the analysis command with `outputs/waveform-primary-20261027`. Full generator and sampling reproducibility uses the saved source snapshots, configuration, dependency lockfile, and environment versions.

For a fresh completed reproduction, run `run_diagnostics(output_path, n_examples=256, n_samples=64)` from `densiffusion.waveform_diagnostics`. The additional [matched-geometry script](diagnostics/matched_geometry.py) accepts `--output` pointing to that completed run. Diagnostics refuse existing diagnostic output directories. They never retrain or reselect models.
