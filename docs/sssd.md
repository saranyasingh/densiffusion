# SSSD

`src/densiffusion/architectures/sssd.py` implements the SSSD^S4 imputer of
[Alcaraz & Strodthoff (TMLR 2023)](https://arxiv.org/abs/2208.09399) as a
denoiser-only network (no encoder). Layer layouts follow the
[reference code](https://github.com/AI4HealthUOL/SSSD) (MIT licence); the
module is an independent reimplementation.

```python
from densiffusion.architectures.sssd import SSSDDenoiser
from densiffusion.models import ImputationModel
from densiffusion.training import TrainingConfig

model = ImputationModel(SSSDDenoiser(channels=1))
# Paper schedule: 200 linear steps from 1e-4 to 0.02.
config = TrainingConfig(diffusion_steps=200, beta_start=1e-4, beta_end=0.02)
```

Use `noise_mse` and the default `DDPMSampler`. Train and sample on a GPU
(`device="mps"` or `"cuda"`): on Apple CPUs PyTorch routes these convolutions
through NNPACK, which is roughly 20 times slower.

## Apple GPU (MPS) gradient bug

With PyTorch 2.14.1, the MPS backend computes **ReLU and tanh gradients
incorrectly** for 3-D tensors with channels-last strides `(C*L, 1, C)`; GELU,
sigmoid, and SiLU are unaffected. Such strides come from transposing a
`(batch, time, features)` result back to `(batch, features, time)`, and from
MPS convolutions whose single-channel input has ambiguous strides. The forward
pass is correct, so nothing fails visibly.

Before the fix, SSSD's input-layer gradients on MPS were about 10,000 times
too large. Gradient clipping then shrank every other gradient accordingly, and
training collapsed to a constant output after about 40 epochs. The denoiser
now copies these tensors to the standard layout (`_standard`), and
`test_gradients_on_apple_gpu_match_cpu` checks every gradient against the CPU.

Minimal reproduction:

```python
import torch

x = torch.randn(4, 8, 5).transpose(1, 2).contiguous().transpose(1, 2)
up = torch.randn(4, 8, 5).transpose(1, 2).contiguous().transpose(1, 2)
for device in ("cpu", "mps"):
    leaf = x.detach().to(device).requires_grad_()
    torch.relu(leaf).backward(up.to(device))
    print(device, leaf.grad.cpu().sub(up * (x > 0)).abs().max())  # mps: nonzero
```

Other architectures trained on MPS can be affected the same way. Comparing
gradients with the CPU for identical weights and inputs, as the test does, is
a quick check.

## Architecture

- The input is the noisy series with observed entries set to their
  observations, projected to `width` channels.
- Each residual block adds a projection of the diffusion-step embedding, then
  applies a 3-tap convolution, a bidirectional S4 layer, a 1x1 projection of
  the conditioning input, a second S4 layer, and a `tanh * sigmoid` gate.
  Blocks pass a residual to the next block and a skip output to the head.
- The conditioning input is the observations concatenated with an indicator
  that is 1 at observed entries. This is SSSD's convention; our shared `mask`
  is `True` at hidden entries, and the denoiser converts between them.
- Each S4 layer is the reference S4 block (state-space convolution plus a skip
  term `D`, GELU, linear output) followed by a residual connection and
  LayerNorm. Its bidirectional form adds a second kernel that runs backwards in
  time, so each output depends on both earlier and later inputs.
- The output layer starts at zero, so an untrained network predicts zero noise.

## Deviations from the paper

1. **S4D kernel instead of S4.** The reference computes its kernels with the
   HiPPO-LegS normal-plus-low-rank (NPLR) parameterisation (about 1,200 lines
   and six extra dependencies). We use the diagonal S4D-Lin kernel of
   [Gu et al. (NeurIPS 2022)](https://arxiv.org/abs/2206.11893):
   `A_n = -1/2 + i pi n`, zero-order-hold discretisation, and step sizes drawn
   log-uniformly from [0.001, 0.1] as in the reference. The two directions of
   a bidirectional layer share `A` and the step size but have their own `C`,
   as in the reference. Everything around the kernel is unchanged.
2. **Smaller network.** Defaults are 4 blocks of width 64, state size 64, and
   a 128 -> 128 -> 128 step embedding: 538,497 parameters. The paper uses
   36 blocks of width 256 and a 128 -> 512 -> 512 embedding for 14-channel
   data. PriSTI's defaults have 363,601 parameters; `layers=3` gives 413,249.
3. **Observed entries of the noisy input.** Our trainer keeps observed entries
   clean. The reference's training input scales them by
   `sqrt(alpha_bar) + sqrt(1 - alpha_bar)`, while its sampler, like ours, uses
   the clean observations. The denoiser also overwrites observed entries of its
   input with the observations, so it never depends on them otherwise.
4. **Training setup comes from the shared trainer:** Adam at 1e-3 with gradient
   clipping (paper: 2e-4), batch size 64, and fixed masks per window (the paper
   draws random, missing-not-at-random, or blackout masks).
5. **`A` and the step size are trained**, as in S4D, at the network's learning
   rate. The reference leaves its S4 `A`, `B`, `P`, and step sizes fixed at
   initialisation and trains only `C`, `D`, and the surrounding layers.
   Freezing them (`requires_grad_(False)` on `log_dt`, `log_A_real`, and
   `A_imag`) is a one-line ablation that has not been compared yet.
6. **Initialisation.** Convolutions use weight normalisation with PyTorch's
   default initialisation. The reference calls Kaiming initialisation after
   weight normalisation, but its next forward pass recomputes the weight from
   the unchanged norm and direction, so the reference's effective
   initialisation is the same.

## Open questions

- **`dt` is unused,** as in the reference, so the network sees window indices
  only. This matters for the proposal's `dt -> dt/2` transfer: a model trained
  to fill gaps at one spacing may give the wrong amount of uncertainty at half
  that spacing, which GP data can check against the exact posterior. S4
  offers a specific option: its learned step size is a time scale, and the S4
  papers handle inputs at a new sampling rate by rescaling it. Dividing the
  step size by the refinement factor at test time is a cheap experiment.
- **Step embeddings differ between papers.** SSSD (like DiffWave) uses
  frequencies from 1 down to 1e-4; CSDI and PriSTI use 1 up to 1e4. A shared
  `architectures/common.py` should keep both.
- Channels interact through the input and conditioning projections, unlike
  temporal-only PriSTI.
