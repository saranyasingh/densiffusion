# History polishing before noisy-target forecasting

This independent synthetic experiment compares raw, learned-polished, and
median-polished histories using the same larger CSDI forecaster. The completed
scientific run found lower fresh-test CRPS for the learned history in all three
seeds against both baselines. Earlier waveform experiments and their saved
results are separate and remain unchanged.

The scientific configuration is
[`configs/history_polishing_forecast.json`](../configs/history_polishing_forecast.json).
Each seed, 28101, 28102, and 28103, has 256 training, 256 validation, and 2,048
fresh test trajectories. A trajectory supplies one 32-point history and one
8-point future on the original integer-time grid, with spacing 1. No points are
inserted, and the noisy future targets are identical across methods. There is
no densification, clean-target substitution, or signal normalization.

## Completed scientific result

The 27 October 2026 run completed fresh-test evaluation and independent artifact
verification. Equal-seed means over the three seeds are:

| History | Test CRPS | Empirical 90% interval coverage |
| --- | ---: | ---: |
| Raw | 0.285006 | 80.25% |
| Learned | **0.274400** | 82.22% |
| Median | 0.282228 | 79.54% |

Learned polishing reduced mean CRPS by 3.72% versus raw and 2.77% versus median.
Its CRPS was lower than both baselines in every seed; all six corresponding
per-seed paired 95% bootstrap intervals for learned-minus-baseline CRPS were
strictly below zero. These intervals condition on the selected models and
saved 64-draw ensembles, are unadjusted for multiple comparisons, and do not
measure uncertainty from retraining or checkpoint selection. Seed variation
is reported separately.

All methods' mean coverage remains below the nominal 90%; learned coverage is
82.22% and varies substantially across seeds. Better CRPS therefore does not
establish nominal calibration. Finite ensembles affect both quantiles and
empirical CRPS. This result supports the learned representation in this fixed
synthetic experiment; it establishes neither convergence nor general
superiority. The [full report](../outputs/history-polishing-20261027/REPORT.md)
contains per-seed results, conditional intervals, validation curves, costs, and
links to the verified artifacts.

## Observations and data separation

For each independent trajectory, draw independent parameters

\[
A\sim\operatorname{Uniform}(0.7,1.3),\qquad
P\sim\operatorname{Uniform}(24,40),\qquad
\phi\sim\operatorname{Uniform}(0,2\pi).
\]

Keep them fixed for integer times \(t=0,\ldots,39\), and generate

\[
s_t=A\left[\sin(2\pi t/P+\phi)
    +0.2\sin\bigl(2(2\pi t/P+\phi)+0.4\bigr)\right],
\qquad
y_t=s_t+\epsilon_t+B_t Z_t,
\]

where \(\epsilon_t\sim\mathcal N(0,0.15^2)\),
\(B_t\sim\operatorname{Bernoulli}(0.1)\), and
\(Z_t\sim\mathcal N(0,2^2)\). These draws are independent across time,
trajectories, and latent parameters. The Gaussian outlier magnitude is drawn
even when its indicator is zero.

[`generate_noisy_periodic`](../src/densiffusion/generators/noisy_periodic.py)
returns only the noisy observation array, shaped `(trajectories, 40, 1)`.
There are no public clean-signal, latent-parameter, or outlier-label outputs.
The repository `generate` wrapper accepts only `dt=1` for this distribution.
History is `y[:, :32]`; forecasting targets are the untouched `y[:, 32:]`.

The RNG plan uses `SeedSequence([seed, 2026102705])` with distinct child streams
for training data, validation data, test data, mediator training, forecaster
initialization, forecaster training, validation sampling, and test sampling.
All methods within a seed use the same trajectories. Test streams are recorded
in advance, but their trajectories are generated only after the global model
freeze. Smoke runs require an independent namespace; the CLI smoke preset uses
2026102799.

## Three history representations

Every preprocessing API accepts only arrays shaped `(batch, 32, 1)` and returns
the same history shape. The forecaster receives the chosen representation
without an additional raw-history input.

| Method | Input to the forecaster |
| --- | --- |
| `raw` | An unchanged copy of the 32 noisy observations. |
| `learned` | A frozen blind-spot MLP prediction at each history position. |
| `median` | A seven-point median of the history, with edge padding. |

For a center position \(i\), the learned model takes the fixed offsets
`[-3, -2, -1, +1, +2, +3]`. Its 12 features are the six neighbor values followed
by six availability indicators. A neighbor outside the 32-point history has
value zero and indicator zero. Available neighbors retain their fixed slots;
boundary positions do not substitute more distant points. The model has no
center feature, residual connection from the center, normalization using the
center, or future input. Predictions are computed from the original supplied
history in one pass, not recursively from already polished values.

The median uses three points on either side and includes the center. At either
edge, it repeats the first or last available history value. Padding never reads
future observations, including when polishing the last history position. At an
endpoint, four of the seven window entries equal that endpoint, so its median
is exactly the raw endpoint value. This is the specified baseline, not a tuning
choice. The learned mediator instead has three available neighbors at an
endpoint, masks its other three slots, and never sees the endpoint center.

## Bounded mediator training and selection

[`history_polishing.py`](../src/densiffusion/history_polishing.py) trains four
candidates: hidden widths 32 and 64, each crossed with MSE and Huber loss
(delta 0.2). Each MLP has two fully connected hidden layers with ReLU activations
and a scalar output. Candidates use fresh initialization and independent,
recorded private initialization and shuffle streams.

Every candidate receives 200 epochs of Adam with learning rate 0.001, weight
decay 0.0001, and batches of 256 center examples. The 256 training histories
provide 8,192 center examples. Targets are their own noisy center observations;
features always exclude that center. The separate 256 validation histories
also provide 8,192 center examples, each excluded from its own input. Other
observed history positions remain available as the prescribed neighbors.

Selection scores noisy-center validation MSE after epochs 1, 5, and every ten
epochs through 200, including for candidates trained with Huber loss. Select
the lowest-MSE checkpoint within each candidate, then the candidate with the
lowest selected MSE. Exact ties choose the first checkpoint and then the first
candidate in recorded order. This masked-center validation criterion uses no
clean values, future observations, or forecasting score. It is a development
selection rule, not a claim that clean signals are recovered.

Each candidate saves its configuration, epoch log, validation scores, selected
checkpoint, and exact selected validation predictions. The mediator directory
also saves `selected.pt`, `selected_predictions.npz` with training/validation
predictions, and `summary.json` with selection, hashes, seeds, budgets, and
timings. All three seeds' mediators are fitted and frozen in
`MEDIATORS_FROZEN.json` before any forecaster is trained. They are never updated
using forecasting loss or forecasting validation performance.

## Paired larger forecasters

The runner reads the selected `reference_size` architecture and training recipe
from `outputs/csdi-gp-better/plan.json` and `selection.json`, records their hashes,
and initializes new weights. No GP-trained checkpoint is loaded. The recovered
CSDI architecture has 413,521 parameters:

| Setting | Value |
| --- | --- |
| Channels / hidden channels | 1 / 64 |
| Residual layers / attention heads | 4 / 8 |
| Diffusion / time embedding dimensions | 128 / 128 |
| Feature embedding / feedforward dimensions | 16 / 64 |
| Dropout | 0.1 |

The full recovered diffusion recipe uses Adam, learning rate 0.001, weight decay
\(10^{-6}\), gradient clipping at norm 1, and 50 diffusion steps. The quadratic
beta schedule runs from 0.0001 to 0.5. Batch size is 64. The scientific forecast
budget is 1,200 epochs, giving 4,800 optimizer updates per method per seed.
Learning-rate decay by a factor of 0.1 occurs at epochs 900 and 1,080, preserving
the recovered fractions 0.75 and 0.9. All-step validation noise MSE is evaluated
every ten epochs.

Checkpoint selection uses original-grid forecast CRPS, with 64 draws for all
256 validation trajectories, only at epochs **10, 50, 100, 200, 400, 800, and
1,200**. Every candidate forecast ensemble is saved. The first lowest-CRPS
checkpoint wins. Noise MSE is a training diagnostic; it does not override this
selection rule.

Within each seed, the three forecasters use paired initial weights, trajectory
order, diffusion timestep/noise streams, dropout seed, optimizer configuration,
update budget, and validation sampling stream. They receive equal checkpoint
selection opportunities and the same unchanged noisy future targets. The
processed histories are the experimental difference. Pairing and measured
optimizer updates are recorded in training audits.

The primary configuration uses MPS with one sequential forecaster worker and
one CPU thread; mediator fitting and history preprocessing run on CPU. Equal
forecaster budgets do not imply equal total compute: the learned method also
pays for the four-candidate mediator search and preprocessing. Timings retain
mediator fitting, preprocessing, forecaster training, validation sampling, and
test inference separately.

## Freeze, fresh test evaluation, and uncertainty

After all mediators and all nine forecasters are selected, `FROZEN.json` records
the training artifacts and their hashes. Evaluation checks this manifest before
generating the 2,048 fresh test trajectories per seed. Each method processes
only the available test history and draws 64 futures shaped
`(2048, 64, 8, 1)`. Test targets remain the original noisy observations. Neither
test scores nor clean simulator quantities choose checkpoints or alter models.

Reported metrics are empirical ensemble CRPS, MSE of the ensemble mean, coverage
of the empirical 5th-to-95th percentile interval, and its mean width. Analysis
recomputes these from saved ensembles and compares methods using 10,000 paired
whole-trajectory bootstrap replicates per seed. Each resampled trajectory keeps
all eight future positions and its saved ensemble together. Time positions and
draws are not independent bootstrap units.

Paired intervals condition on the fitted, validation-selected models and saved
ensembles. They do not capture uncertainty from retraining, checkpoint choice,
or finite Monte Carlo ensembles. Finite draws affect both empirical CRPS and
interval quantiles. Across the three independent training/dataset seeds, means,
sample standard deviations, and ranges describe seed variation separately;
they are not pooled trajectory confidence intervals. Comparisons and metrics
have no multiplicity correction. Validation curves and final-candidate
selection should be inspected before making any convergence claim.

## Reproduction and saved-artifact checks

Use a fresh output directory. Training refuses an existing directory, and
evaluation refuses to replace completed test artifacts. The commands below use
the fixed scientific configuration and the recovered local GP recipe:

```sh
RUN=outputs/history-polishing-forecast-fresh
uv run --locked python scripts/train_polishing_forecast.py \
  --phase train --output "$RUN" \
  --config configs/history_polishing_forecast.json \
  --recipe-source outputs/csdi-gp-better
uv run --locked python scripts/train_polishing_forecast.py \
  --phase evaluate --output "$RUN"
uv run --locked python scripts/analyze_polishing_forecast.py "$RUN"
uv run --locked python scripts/verify_polishing_forecast.py "$RUN"
```

`train` includes mediator fitting, the mediator freeze, paired forecaster
training/selection, and the global freeze. `evaluate` performs fresh test
generation and sampling, then writes `EVALUATED.json`. The analysis script
creates `analysis.json`, plots, and a report from saved ensembles. The verifier
checks saved data replay, frozen hashes, mediator selection and predictions,
history-only preprocessing, unchanged future targets, paired initialization and
update budgets, validation selection, and recomputed test metrics; it does not
train models or generate new diffusion forecasts.

For a wiring check, use a separate fresh output path and add `--smoke` to the
training command. The preset uses one seed, 16/8/8 train/validation/test paths,
five mediator epochs, two forecaster epochs and diffusion steps, and four
forecast draws in its independent RNG namespace. Smoke results are not evidence
about scientific forecast quality; the completed result above uses the full
scientific budget and fresh test evaluation.
