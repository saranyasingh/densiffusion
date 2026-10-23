# CSDI for time-series densification

`CSDIDenoiser` in `src/densiffusion/architectures/csdi.py` implements the
conditional residual denoiser from [CSDI, Section 5 and Appendix E.1](https://arxiv.org/pdf/2107.03502v2),
following the [official network](https://github.com/ermongroup/CSDI/blob/main/diff_models.py)
and [conditioning implementation](https://github.com/ermongroup/CSDI/blob/main/main_model.py).
It plugs into the existing framework without architecture registration:

```python
from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.models import ImputationModel

model = ImputationModel(CSDIDenoiser(n_channels=1))
# Train with noise_mse; sample with DDPMSampler using the same TrainingConfig.
```

`forward(noisy, steps, *, observed, mask, dt)` takes and returns tensors shaped
`(batch, time, channels)`. The two input planes are observed values with hidden
entries cleared and noisy targets with observed entries cleared. Clearing uses
`masked_fill`, so even NaNs at unused positions cannot contaminate the network.
The conditional mask in CSDI is `~mask`: this repository's `True` means hidden.

Each residual block adds the projected diffusion embedding, applies temporal
attention separately for each feature and feature attention separately for each
time, then combines projected side information through a sigmoid/tanh gate.
Residual updates are scaled by `sqrt(2)` and summed skip outputs by
`sqrt(number_of_layers)`. Pointwise linear layers are equivalent to the
reference's kernel-size-one convolutions. The final projection weight starts at
zero. Singleton time/feature attention is bypassed as in the official code;
one-channel data is supported.

Side information contains sinusoidal measurement times `arange(time) * dt`,
learned feature IDs, and the observed mask. Measurement-time frequencies decrease
as `10000**(-2*i/time_embedding_dim)`, with interleaved sine/cosine as in the
official code (the paper writes these in concatenated order). Diffusion-step
frequencies increase as `10**(4*i/(diffusion_embedding_dim/2-1))`, with concatenated
sine/cosine followed by two Linear/SiLU layers. These are separate embeddings;
`dt` always denotes the target spacing. Local measurement times start at zero.

Network defaults are 64 hidden channels, four residual layers, eight attention
heads, diffusion/time embedding dimensions 128, feature embedding dimension 16,
feedforward dimension 64, and dropout 0.1. All are configurable constructor
arguments. Hidden channels must be divisible by attention heads; time embedding
dimensions must be even, and diffusion dimensions even and at least four.

## Local example

With the installed research dependencies, run from the repository root:

```sh
uv run python scripts/train_csdi_gp.py --output outputs/csdi-gp
```

Use a fresh output directory for each run. The CPU default uses one thread,
seed 42, 48 independent univariate exponential-GP trajectories of length 33 at
`dt=0.1`, and the existing GP generator with mean 0, variance 1, lengthscale 1.
It splits trajectories into 33 training, 7 validation, and 8 test paths **before**
forming length-17 windows with stride 16 (66/14/16 windows). It performs no
normalization. Shared window endpoints are observed; hidden targets do not
overlap between the default windows.

The small model has 16 hidden channels, two residual blocks, two heads, 32-wide
time/diffusion/feedforward dimensions and eight feature-embedding dimensions.
It trains for 200 epochs with batch size 32, Adam at 0.001 and gradient clipping.
This remains a small local smoke experiment, not a converged benchmark.
Initialization, generated paths, training noise, validation noise and sampling
are seeded. Repeating on the same device/software should reproduce the run;
CPU and MPS need not agree numerically.

For a very short pipeline check, or to use Apple GPU acceleration:

```sh
uv run python scripts/train_csdi_gp.py --epochs 2 --n-samples 2 --output outputs/csdi-gp-quick
uv run python scripts/train_csdi_gp.py --device mps --output outputs/csdi-gp-mps
```

Very short training can produce badly scaled samples even with finite losses.
Neither finite samples nor a passing smoke test establishes useful imputation
quality. The script reports the baseline comparison rather than assuming CSDI
outperforms it. CPU can be faster than MPS for this tiny workload.

`run_experiment` selects the best checkpoint using validation noise MSE and
evaluates held-out `0.2 -> 0.1` refinement. All methods use the same hidden
positions, original time coordinates, targets, and 16 draws:

- CSDI samples from the shared `DDPMSampler` through `pipeline.impute`.
- Linear interpolation is a point-mass ensemble; its CRPS equals its MAE.
- `GPPosteriorSampler(gp_posterior)` uses the generator's exact exponential
  kernel, parameters and mean, without jitter. The posterior is exact; CRPS is
  estimated from its finite ensemble, so it has Monte Carlo variation.

Scores are mean hidden-point ensemble CRPS in original value units, lower better.
On the verified M1 Max CPU run (PyTorch 2.14.1, seed 42, defaults above), the
200-epoch example took about four seconds and validation selected epoch 194.
CSDI scored 0.464840, interpolation 0.264188, and the exact GP posterior ensemble
0.190731. CSDI still trails both baselines: this verifies the workflow, not
convergence. A second seeded run produced bit-identical saved ensembles and
scores. The earlier shorter training checks were undertrained; no architecture,
schedule, or data settings were selected from these held-out comparisons.

The script verifies exact observation preservation for every saved ensemble.
It saves `config.json` (including path IDs, sizes, seeds, versions and schedule),
`model.pt` (selected shared checkpoint), `scores.json`, and `samples.npz`.
The NPZ contains `target`, `times`, `mask`, `csdi`, `linear_interpolation`, and
`exact_gp_posterior`; sample arrays have shape `(test_windows, draws, time, 1)`.
The saved CSDI samples are the exact draws scored by the experiment runner.
To load the model, instantiate `CSDIDenoiser` with `config.json["architecture"]`,
wrap it in `ImputationModel`, and load `model.pt["model_state_dict"]` using
`torch.load(..., map_location="cpu", weights_only=True)`.

No finer-than-0.1 ground truth is generated or used for normalization, training,
validation, checkpoint selection, or these scores. A future `0.1 -> 0.05`
scale-transfer experiment must be a separately labeled evaluation after freezing
the model and preprocessing. The example does not claim transfer performance.

## Reference compatibility and deliberate differences

The shared trainer and sampler use indices `j=0,...,T-1`, as does the official
code; index `j` corresponds to diffusion time `j+1` in the paper. In particular,
`alpha_bar[0] = 1-beta[0]`, not 1. The denoiser evaluates the reference embedding
formula on demand instead of storing a fixed-length step table, avoiding a second
independent diffusion-step-count setting.

Training uses hidden-only noise MSE. Sampling starts hidden entries from a unit
Gaussian and uses the reference noise-prediction reverse mean and posterior
variance `beta[j]*(1-alpha_bar[j-1])/(1-alpha_bar[j])`. The last step (`j=0`)
adds no noise. The sampler clamps observations at every reverse step and restores
the original NumPy values exactly on output, including precision beyond float32.
As in the official network, predictions at observed positions are unrestricted;
the objective masks them and the sampler discards them. This has the same
hidden-target effect as the output masking drawn in the paper's Figure 6.

The default shared schedule is **linear**. The original example uses 50 steps
from `1e-4` to `0.5`, giving terminal `alpha_bar` about `2.08e-7`. The paper's
default is a **quadratic** schedule over the same endpoints, with terminal
`alpha_bar` about `3.35e-5`. Keeping the shared default endpoint `0.02` while
reducing to 50 steps would instead leave `alpha_bar` about `0.603`, inconsistent
with near-pure-noise initialization. Architecture and sampler always share the
same training schedule. The example overrides endpoints/count, not schedule type.

Alternating hidden interior points are fixed densification masks, training
`2*dt -> dt`. This specializes the paper's known/test-pattern strategy rather
than reproducing its random or historical imputation experiments. Complete
targets are required. The small GP model/data and training budget also differ
from the paper's real-world benchmarks. The original smoke run used the shared
defaults. A subsequent comparison adds optional training settings described below;
the original example and all prior defaults retain their behavior.

## Stronger training comparison

Run a fixed two-candidate comparison with a larger independent training set:

```sh
uv run python scripts/compare_csdi_training.py --baseline outputs/csdi-gp-verified --output outputs/csdi-gp-better
```

The baseline directory must contain the original example's `config.json`,
`model.pt`, `samples.npz`, and `scores.json`. Use `--baseline outputs/csdi-gp` if
that is where you ran it, and choose a fresh output directory for each comparison.

Both candidates train on 512 independent exponential-GP paths, producing 1,024
length-17 windows. An independent stream supplies 64 validation paths. No paths
are shared across partitions, all have spacing 0.1, masks still alternate, and
there is no fitted normalization or finer-resolution ground truth.

Both use 50 quadratic diffusion steps from beta=0.0001 to 0.5, Adam with weight
decay 1e-6, learning-rate decays at 75% and 90% of training, and validation over
all diffusion steps every ten epochs. Gradient clipping at norm 1 and fixed
validation noise are retained. These optional settings now live in the shared
`TrainingConfig`; the sampler uses the same schedule helper as the trainer.

The fixed candidates are the original small network trained for 400 epochs
(6,400 updates) and the reference-size network trained for 200 epochs (3,200
updates), both at batch size 64. Each candidate restores its best validation
noise-MSE checkpoint. The candidate with lower validation CRPS from 32 draws is
selected; `selection.json` records that choice **before any test evaluation**.
This is a comparison of training plans, not an ablation isolating any one change.

The primary comparison uses 128 fresh independent test paths and 32 draws per
method. It evaluates the selected model, the original saved model, interpolation,
and the exact GP posterior on identical targets and masks. The paired CRPS
improvement interval resamples entire trajectories, keeping each path's two
windows together. It describes uncertainty across test paths conditional on
these trained models and ensembles, not variability across training seeds.
The previously inspected original test set is also scored separately using its
original draw count. No changes are chosen using either test comparison.

Outputs include a pre-training `plan.json` with seeds, configurations, versions
and code hashes; training logs; each candidate's checkpoint and validation
samples; the frozen selection; final test samples and `comparison.json`.

The verified M1 Max CPU comparison took about five minutes including training
and sampling. Validation CRPS selected `reference_size` (0.182720) over
`small_longer` (0.327650), using its epoch-200 checkpoint. Fresh test results
on 128 independent trajectories, with 32 draws per method, were:

| Method | Hidden-point CRPS |
| --- | ---: |
| Original saved CSDI | 0.447265 |
| Improved CSDI | 0.188437 |
| Linear interpolation | 0.254318 |
| Exact GP posterior ensemble | 0.186064 |

This is a 57.9% reduction from the original model on the same fresh targets.
The paired trajectory-bootstrap 95% interval for the absolute reduction is
[0.240184, 0.278303], conditional on this single training seed and the sampled
ensembles. On the original, previously examined test set, improved CSDI scored
0.197682 versus the saved original score 0.464840 (16 draws each).
The GP ensemble score differs between evaluations because of finite-sample
variation; its distribution and kernel parameters are unchanged.

All saved scores were recomputed from the output ensembles, and every observed
entry was preserved exactly. The original smoke example was also rerun with the
extended shared trainer and reproduced its old samples and scores bit-for-bit.
This comparison changes data quantity, model size and training settings together;
it does not establish which change contributed how much, or performance across
multiple training seeds. All evaluations remain `0.2 -> 0.1` refinement.

## Separate shared work for faithful benchmark reproduction

The stronger comparison implements a shared quadratic schedule, optional weight
decay and learning-rate decay, and all-step validation. Remaining extensions for
the original benchmarks include:

- Per-iteration random/historical/mixed masks and a separate availability mask
  to exclude genuinely missing training targets from the loss.
- Matching the reference's resampled validation noise and final-weight saving
  behavior, if an exact code reproduction is intended. Here validation noise is
  fixed and the best validation checkpoint is restored.
- Timestamp and padding-mask propagation for irregular grids or absolute-time
  models. The current denoiser interface supplies only regular local spacing.
- The paper's normalized, 19-quantile CRPS reporting, in addition to the
  repository's empirical ensemble CRPS in data units.

The larger network sizes are already configurable.
[Densiffusion forecasting](forecasting.md) now trains a separate CSDI model on
densified histories and hidden future suffixes, retaining the existing full
attention. The paper's
forecasting linear attention and unconditional CSDI remain separate architecture
options not included here.

## Checks

```sh
uv run pytest tests/test_csdi.py tests/test_csdi_example.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests cover univariate/multivariate shapes, both attention axes, finite/nonzero
gradients, conditioning leakage with nonconstant predictions, distinct time and
diffusion embeddings, reference step indexing, checkpoint round trips, and exact
observation preservation through the real sampling pipeline. Example tests
verify trajectory separation before overlapping windows and observed-only
interpolation.
