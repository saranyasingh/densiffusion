# MTSCI

`src/densiffusion/architectures/mtsci.py` implements the imputer of
[Zhou et al., "MTSCI: A Conditional Diffusion Model for Multivariate Time
Series Consistent Imputation" (CIKM 2024)](https://arxiv.org/abs/2408.05740)
as a denoiser-only network (no encoder). Layer layouts and the consistency
loss follow the [reference code](https://github.com/JeremyChou28/MTSCI)
(MIT licence); the module is an independent reimplementation. Where the paper
and the reference code differ, the reference code is followed.

```python
from densiffusion.architectures.mtsci import MTSCIDenoiser, MTSCIObjective
from densiffusion.models import ImputationModel
from densiffusion.training import TrainingConfig

denoiser = MTSCIDenoiser(n_channels=1)
model = ImputationModel(denoiser)
objective = MTSCIObjective(denoiser)  # noise MSE + 5 * consistency
# Reference point-missingness schedule: 50 quadratic steps up to 0.2.
config = TrainingConfig(
    diffusion_steps=50, beta_end=0.2, schedule="quadratic", batch_size=64
)
```

Pass `objective=objective` to `run_experiment` or `train_densifier`, and use
the default `DDPMSampler`. With `objective=noise_mse` the network trains as a
plain conditional noise predictor.

## Architecture

MTSCI is a CSDI-family conditional denoiser. Compared with our
`CSDIDenoiser`:

- **One input plane.** The network reads observations at observed entries and
  noisy values at hidden entries in a single channel (CSDI uses two).
- **Side information before attention.** Each residual layer adds the
  projected step embedding *and* the projected side information (time
  encoding, channel embedding, observed indicator) before its temporal and
  feature Transformer layers. CSDI adds side information after attention.
- **Input re-read by the decoder.** The skip sum is concatenated with the raw
  input plane and projected back to the hidden width before the two-layer
  output head.

Defaults are the reference ETT/Weather sizes: 64 hidden channels, 4 residual
layers, 8 heads, step embedding 128, time embedding 128, channel embedding 16
(380k parameters with one channel). Feature attention is skipped for
univariate data, as in the reference.

## Intra-consistency loss

During training, `MTSCIObjective` adds `consistency_weight` (default 5) times
an InfoNCE loss computed by `MTSCIDenoiser.consistency_loss`. Using the
batch's own noise draw, the network encodes three views of each window:

| View | Observed entries | Hidden entries | Observed indicator |
| --- | --- | --- | --- |
| forward | clean | noisy | observed entries |
| reverse | noisy | clean | hidden entries (complementary mask) |
| negative | noisy | noisy | none |

The forward view's decoder input at hidden entries and the reverse view's
encoding at observed entries are combined into one representation. InfoNCE
(temperature 0.1, cosine similarity) pulls it towards the negative view's
encoding of the same window, with the other windows in the batch as
negatives. The three views run as one batched pass. Validation runs in eval
mode and scores noise MSE only, as in the reference.

## Verification against the reference

With weights copied from the reference `denoising_network`, the noise
prediction and the consistency loss match to 6e-16 and 4e-16 in float64
(3 layers, 3 channels, random masks). The comparison script is not committed;
it imports the reference repository and the `info-nce-pytorch` package.

## Differences from the paper and reference

- **No inter-consistency term.** The reference mixes a 1x1-convolution of the
  *following* window's ground truth into the decoder input during training
  (random per-entry weights) and uses the window's own input at inference.
  Our trainer supplies independent windows with no neighbour, so this
  training-time branch and its `backward_projection` layer are omitted. The
  inference network is the same.
- **Consistency loss as in the code, not Eq. 12.** The paper describes a
  symmetric contrast between the two complementary views; the reference
  contrasts the combined forward/reverse representation with the fully noised
  view, mixing the forward decoder input (after the merge layer) with the
  reverse encoder output. We follow the code.
- **`dt` is unused.** As in the reference, the time encoding uses window
  positions `0..L-1`, so a model trained at one spacing sees identical side
  information at another.
- **Training setup comes from our trainer:** masks fixed per example rather
  than redrawn, Adam with gradient clipping, and checkpoint selection on
  validation noise MSE. The reference trains 300 epochs with batch size 16 on
  24-step windows.
- **No missing ground truth.** The reference also supports naturally missing
  values, which belong to neither the conditioning nor the target set; our
  targets are complete, so the two sets partition each window.

## Speed

On 2 CPU cores, a training step (batch 64, 65-point windows) takes ~1.0 s
with the consistency term and ~0.4 s without it; the consistency term costs
three extra encoder passes. Sampling 1,024 windows takes ~0.6 s per diffusion
step. Use a GPU for real runs.
