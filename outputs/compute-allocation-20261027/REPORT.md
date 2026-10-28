The resource concern is justified for actual densification. Equal epochs and
equal forecaster architecture gave the dense waveform methods substantially
more computation. The recent positive **history-polishing** result is different:
it used the original time grid, and the full mediator search was inexpensive.
This investigation finds useful evidence for polishing's efficiency, but **does
not establish a universally optimal allocation or a new prospective
equal-resource test result**.

I audited the existing three-seed runs, reconstructed budget-constrained
validation curves, recomputed saved validation metrics, and ran controlled
synthetic-tensor kernel benchmarks. There was no new scientific model training
or test evaluation. All 625 files under the three protected experiment roots
remain unchanged. The [machine-readable audit](audit.json), [protocol](protocol.json),
[work profiles](work_profiles.json), [native kernel timings](kernel_benchmarks.json),
and [verification](verification.json) are retained here. The
[budget figure](budget_frontiers.svg) was rendered and visually inspected;
the [PNG preview](budget_frontiers.svg.png) shows the same figure.

**What “same resources” means.** The primary audit unit is counted dense matrix
floating-point operations, where one multiplication plus one addition counts
as two. This hardware-independent arithmetic proxy includes the network's
attention products, projections, and forward/backward matrix operations. It
excludes activation, softmax, normalization, loss, random-number generation,
reductions, clipping, Adam arithmetic, median comparisons, interpolation, data
movement, and I/O. Zero matrix work for a median filter does not mean zero
runtime. Historical elapsed seconds provide a second, implementation-dependent
view that includes those costs. Neither measure is energy or memory usage.

The charged preparation/training cost is

\[
C_{\rm train}=C_{\rm auxiliary\ search+fit}
 +C_{\rm preprocessing/augmentation}
 +C_{\rm forecaster\ optimization}+C_{\rm validation}.
\]

Every preceding validation pass is charged, including unsuccessful checkpoint
candidates. The mediator's entire four-candidate search is charged to learned
polishing. The waveform densifier's full fitting cost and independent history
and future augmentation are charged to learned densification. Costs never
double-count validation already included in recorded training time. Researcher
time and the unknown cost of earlier project-wide development are outside this
ledger for every method.

For deployment, retain a separate constraint, or state the expected number
of forecasts \(Q\) and compare

\[
C_{\rm lifetime}(Q)=C_{\rm train}
 +Q\big(C_{\rm history\ processing}+C_{\rm forecast\ sampling}\big).
\]

The audit keeps 64 forecast draws fixed. Amortizing a trained densifier over
many forecasts reduces its fitting cost per forecast; it does not remove the
cost of longer-sequence forecasting or sampling a new history. Reusing a
densifier across tasks needs an explicit amortization rule and a comparable
opportunity to reuse baseline models.

**Why the same network costs more on a dense grid.** For the larger univariate
CSDI network, the profiled forward matrix work per trajectory of length \(L\)
agrees exactly with the independent matrix-product count

\[
F(L)=1024L^2+484736L+131072.
\]

The quadratic term is temporal attention; the linear term contains
pointwise projections and feedforward layers; the constant term contains
diffusion-embedding projections. Feature attention is bypassed for this
univariate implementation. Doubling the waveform grid from 33 to 65 points
raises forward work from 17.242496 million to 35.965312 million operations,
**2.086 times**, before training or sampling a densifier. Forward plus backward
has essentially the same ratio. For the smaller network used in the original
waveform forecast comparison the ratio is **2.319 times**. The cost is neither
one times nor automatically four times: the relative sizes of the linear and
quadratic terms matter.

For a hypothetical factor-two version of the 32-history/8-future task, the
larger forecaster's length would grow from 40 to 79, raising forward work from
21.158912 million to 44.816 million, **2.118 times**. That hypothetical
densification was not applied in the polishing experiment.

Native-kernel timing benchmarks used fresh random tensors, one Torch thread,
three warm-up steps, and five repetitions of three steps each. Accelerator
timings synchronize at repetition boundaries. For the larger model, increasing
length 33 to 65 multiplied CPU training-step time by 2.037 and MPS training-step
time by 1.224; MPS forward time at sampling batch size 256 increased by 2.395.
Training uses batch size 64 and includes backward, clipping, and Adam.
Different batching/kernel overhead explains why time and arithmetic need not
scale identically. These short benchmarks establish local kernel costs, not
end-to-end throughput, energy, peak memory, or forecast quality.

**Actual densification: the apparent linear gain was not resource-matched.**
The small waveform experiment used the same 600 epochs / 9,600 updates for
every forecaster. The full per-seed costs are:

| Pipeline | Counted training/selection work | Relative work | Recorded training/selection time, mean | 64-draw deployment work |
|---|---:|---:|---:|---:|
| Direct CSDI | 5.029 trillion | 1.000 | 331.6 s | 2.382 billion |
| Linear densification + CSDI | 11.666 trillion | 2.320 | 661.1 s | 5.524 billion |
| Learned densification + CSDI | 13.419 trillion | 2.668 | 791.3 s | 6.005 billion |

The recorded times are elapsed worker times with three concurrent CPU seed
workers, not isolated or equal-FLOP benchmarks. The original test means were
0.146844 direct, 0.136779 linear, and 0.365331 learned
([original report](../waveform-primary-20261027/REPORT.md)). The modest linear
advantage therefore cannot establish an advantage at equal resources.

At a **common ceiling of 5.029 trillion matrix operations per seed**, the
saved training traces make the following checkpoints affordable. Each method
selects its best validation CRPS among checkpoints available by that ceiling:

| Pipeline | Available through epoch | Mean validation CRPS | Seed sample SD |
|---|---:|---:|---:|
| Direct | 600 | **0.142970** | 0.011274 |
| Linear densification | 200 | 0.278019 | 0.056320 |
| Learned densification | 100 | 0.393893 | 0.008037 |

Direct wins in each seed at this ceiling. A time-based ceiling gives the same
qualitative result: at 400 recorded seconds per seed the means are 0.142970,
0.161004, and 0.379282. Linear's best saved mean first beats direct's full-run
mean after reaching its epoch-500 candidate, costing about 1.933 times the
direct full-run arithmetic budget. The raw trace ends at epoch 600, so its
flat curve beyond that point means “reuse the best known checkpoint,” not
“additional direct training cannot help.”

The [larger waveform follow-up](../waveform-large-20261027/REPORT.md) learned
good interpolation at its training geometry but failed the frozen resolution
transfer gate. It never produced a scientific forecasting comparison. There
is no larger-model densification forecasting advantage to explain or optimize
from those artifacts. This audit preserves that stopping decision.

**Polishing: the extra training is too small to be an obvious explanation.**
All three polishing forecasters have the same 40-point input/target sequence,
413,521 parameters, paired initialization, 4,800 optimizer updates, and seven
forecast validation opportunities. Each mediator's four-candidate search takes
6.50 seconds on average. Counting every candidate, its training, reconstruction
validation, and train/validation preprocessing costs 0.122402 trillion matrix
operations. That equals **7.54 forecaster optimization epochs**, compared with
the common 1,200-epoch forecaster budget.

The additional mediator work is **0.628% of forecaster optimization work**, or
**0.0706% of the full training/selection matrix work**. The different percentages
matter because noise validation costs 32.500 trillion operations and forecast
CRPS validation costs 121.334 trillion, versus 19.487 trillion for optimization.
Forecast validation sampling alone accounts for roughly 500 of 915 measured
forecaster seconds. Mediator search adds about **0.711%** relative to raw
forecaster training/selection wall time. Its CPU implementation and the
forecasters' MPS execution make that a single-host elapsed comparison, not an
energy equivalence.

| Pipeline | Full counted training/selection work | Recorded training/selection time, mean |
|---|---:|---:|
| Raw | 173.320595 trillion | 913.73 s |
| Learned polishing | 173.442997 trillion | 923.96 s |
| Median | 173.320595 trillion, plus uncounted median operations | 919.65 s |

Runtime differences between identical forecasters are noise/implementation
effects, not evidence that filtering inherently speeds forecasting. Learned
polishing adds only 315,392 counted operations per history at deployment,
versus 67.709 billion for its 64 forecast draws. Median and raw have the same
forecaster arithmetic.

The saved validation frontier remains favorable after charging the mediator:

| Common budget, fraction of full raw work | Raw CRPS | Learned CRPS | Median CRPS |
|---|---:|---:|---:|
| 0.25 | 0.360842 | **0.335664** | 0.340049 |
| 0.50 | 0.298343 | **0.280203** | 0.281955 |
| 0.75 | 0.288473 | **0.275734** | 0.281955 |
| 1.00 | 0.288473 | **0.274759** | 0.281955 |

These are three-seed means. At 0.75 and 1.00, learned polishing beats both
baselines in every seed. At 0.50 it beats raw in every seed but median in only
two. The epoch-10 models are unstable and poor: at a 100-second ceiling median
has the lowest mean, and all three means exceed 1.9. Polishing is not superior
at every tiny budget. Full seed selections, sample SDs, affordable epoch limits,
candidate counts, and unused budgets are in `audit.json`.

An additional check avoids comparing only nearly identical training budgets:
**learned epoch 200 beats the best raw checkpoint anywhere through epoch 1,200
in all three seeds**, using 78.120 trillion operations including all preceding
selection and mediator work. Raw's full search costs 173.321 trillion. The
learned-minus-best-raw validation differences are −0.006006, −0.004029, and
−0.014777. **Learned epoch 400 also beats the best median checkpoint through
epoch 1,200 in every seed**, with differences −0.003354, −0.003943, and −0.003320.
That learned prefix costs 104.118 trillion versus the median full search's
173.321 trillion. These comparisons weaken the explanation that the positive
polishing result merely purchased a little more forecaster training.

The [original frozen-test result](../history-polishing-20261027/REPORT.md)
remains a 3.72% CRPS improvement over raw and 2.77% over median, with all three
seed differences favorable. It is not relabeled as a new resource-matched
test. The audit did not load those test ensembles, select new models using
them, or generate replacement tests.

**What is and is not an optimal balance.** There is no task-independent optimal
fraction to spend on a densifier. With a fixed total ceiling, the relevant
question is the best achievable validation forecast score after assigning
resources to auxiliary fitting, augmentation, forecaster fitting, and
selection. If spending another unit on the auxiliary model improves forecasting
less than spending it on the forecaster, the allocation should move toward the
forecaster; the reverse favors the auxiliary model. This is a marginal
improvement criterion, not evidence that the response is smooth or convex.
Discrete architectures, sampling counts, setup overhead, and failures such as
resolution transfer can put the best observed allocation at zero auxiliary
training. A denoising-loss reduction alone does not measure that marginal
forecast benefit.

The present traces do not vary mediator training effort and forecaster training
effort independently. All mediator candidates used 200 epochs, and only one
selected representation per seed was passed to the forecasters. Selected
mediator checkpoints at 50–160 epochs do not establish that the earlier
selection/search could have been stopped there. Likewise, historical learning
rate decay remained tied to each original full-run budget. Truncating those
traces is not equivalent to designing the best shorter-budget schedule.

The curves impose **equal ceilings, not equal actual expenditures**. Sparse
checkpoint spacing leaves unused budget; at the full raw polishing ceiling,
learned can afford scoring through epoch 800 but not its epoch-1,200 score,
leaving about 20% unused. Different ceilings can also afford different numbers
of validation opportunities. Selection on the same validation data is
optimistic, and these descriptive seed summaries are not confidence intervals
for a newly selected compute policy. Completing a training budget proves
neither convergence nor an optimal allocation.

**A decisive prospective comparison requires these additional controls.**

1. Declare a hardware-independent training-work ceiling and a single-host
   elapsed-time ceiling separately, plus a deployment ceiling or an explicit
   forecast count. Hold original data, larger forecaster architecture, output
   grid, and scored sample count fixed. Do not compare experiments with
   different data distributions as though their resource frontiers pooled.
2. Reserve the same forecast-validation quota and sampling precision per method
   before assigning the remaining compute. Charge auxiliary search and
   preprocessing. Check feasibility first: one full 256-example, 64-draw
   polishing validation pass already costs 17.333 trillion matrix operations.
3. Sweep a bounded auxiliary allocation, including zero, while assigning the
   remainder to forecaster optimization. For polishing, a useful development
   grid is mediator training lengths 25/50/100/200, retaining masked noisy-history
   reconstruction selection. For true densification, vary fitting effort and
   cached augmentation count only within the original no-leakage construction;
   establish useful imputation and frozen transfer before a forecasting claim.
4. Give raw and median baselines equal tuning/search resources, including an
   opportunity to adjust learning-rate decay to their available optimization
   budget. Equal final fitting cost with unequal search effort is insufficient.
   Train to the ceiling within one optimizer update, report unused resources,
   and keep the validation candidate quota equal. Equal update count is not
   imposed when sequence lengths differ.
5. Freeze the allocation policy, schedules, preprocessing, and selected models
   across three paired seeds before generating a new test namespace. Score
   2,048 fresh noisy trajectories per seed for polishing with at least 64 draws,
   retaining exact samples, calibration metrics, per-seed differences, and
   paired trajectory-bootstrap intervals. Original test sets are already used
   evidence and cannot serve as a new confirmatory allocation test.

This prospective sweep was **not run** in this audit. The current actionable
conclusion is narrower: true densification has no demonstrated equal-resource
advantage in the saved experiments; inexpensive learned polishing has favorable
retrospective efficiency evidence that is difficult to explain solely by its
small added fitting budget. Its optimal allocation remains unmeasured.

**Verification and reproduction.** The implementation recomputed CRPS,
ensemble-mean MSE, central-90% coverage, and interval width from all 63 polishing
validation ensembles, and checked both experiments' validation scores against
their training logs. The waveform runner did not save per-checkpoint validation
ensembles, so its validation scores cannot be independently rescored here.
Matrix profiles pass batch-size linearity and independent forward-formula
checks. Artifact hashes protect all 625 prior files; no commit or push occurred.
Software and final artifact checks are recorded in [final_checks.json](final_checks.json).

Run from the repository root with a fresh destination:

```sh
uv run --locked python scripts/audit_compute_allocation.py \
  --output outputs/compute-allocation-reproduction
uv run --locked pytest tests/test_resource_allocation.py
```

The original source snapshot records the executed numerical audit. The
`source_final/` snapshot adds the independent formula check, supported-geometry
guards, documentation, and a square plotting canvas that fixes Quick Look
thumbnail cropping. Those post-run changes do not alter the numerical audit;
the final verification rechecks the saved profiles and reconstructed ledgers.
