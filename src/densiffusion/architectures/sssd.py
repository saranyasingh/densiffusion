"""SSSD^S4 denoiser with diagonal state-space (S4D) layers.

SSSD (Alcaraz & Strodthoff, TMLR 2023, arXiv:2208.09399) is a DiffWave-style
noise predictor whose residual blocks use bidirectional S4 layers in place of
dilated convolutions. Every block adds a 1x1 projection of the conditioning
input: the observations concatenated with an indicator that is 1 at observed
entries. There is no separate encoder, so use it as
`ImputationModel(SSSDDenoiser(channels))` with `noise_mse` and the default
`DDPMSampler`. Channels are mixed by the input and conditioning projections.

Layer layouts follow the reference implementation
(github.com/AI4HealthUOL/SSSD, MIT licence); this module is an independent
reimplementation. The S4 layers replace the reference's HiPPO-LegS (NPLR)
kernel with the diagonal S4D-Lin kernel of Gu et al. (NeurIPS 2022,
arXiv:2206.11893). Everything around the kernel matches the reference S4 block.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class S4DKernel(nn.Module):
    """Convolution kernels of a diagonal state-space model (S4D-Lin).

    Each of the `features` channels has state_size // 2 complex modes with
    A_n = -1/2 + i pi n and a learned step size dt. Zero-order-hold
    discretization turns the continuous model into the recurrence
    x[k] = exp(dt A) x[k-1] + (exp(dt A) - 1) / A u[k], y[k] = 2 Re(C x[k]),
    whose impulse response is the kernel
    K[l] = 2 Re(sum_n C_n (exp(dt A_n) - 1) / A_n exp(dt A_n l)).
    The `channels` kernels share A and dt but have their own C.
    """

    def __init__(
        self,
        features: int,
        state_size: int,
        *,
        channels: int = 1,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        modes = state_size // 2
        log_dt = torch.rand(features) * math.log(dt_max / dt_min) + math.log(dt_min)
        C = torch.randn(channels, features, modes, dtype=torch.cfloat)
        A_imag = math.pi * torch.arange(modes, dtype=torch.float).repeat(features, 1)
        self.log_dt = nn.Parameter(log_dt)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.log_A_real = nn.Parameter(torch.full((features, modes), math.log(0.5)))
        self.A_imag = nn.Parameter(A_imag)

    def forward(self, length: int) -> Tensor:
        """Return kernels shaped (channels, features, length)."""
        A = torch.complex(-self.log_A_real.exp(), self.A_imag)
        dtA = A * self.log_dt.exp()[:, None]
        C = torch.view_as_complex(self.C) * (dtA.exp() - 1) / A
        positions = torch.arange(length, dtype=self.log_dt.dtype, device=A.device)
        powers = (dtA[..., None] * positions).exp()
        return 2 * torch.einsum("chn,hnl->chl", C, powers).real


class S4D(nn.Module):
    """Reference S4 block with an S4D kernel: convolution, skip D, GELU, linear.

    Inputs and outputs are shaped (batch, features, length). A bidirectional
    layer adds a second kernel that runs backwards in time, so every output
    depends on inputs both before and after it.
    """

    def __init__(self, features: int, *, state_size: int = 64, bidirectional=True):
        super().__init__()
        if type(state_size) is not int or state_size < 2 or state_size % 2:
            raise ValueError("state_size must be an even integer of at least 2")
        self.bidirectional = bidirectional
        channels = 2 if bidirectional else 1
        self.kernel = S4DKernel(features, state_size, channels=channels)
        self.D = nn.Parameter(torch.randn(features))
        self.output = nn.Conv1d(features, features, 1)

    def convolve(self, u: Tensor) -> Tensor:
        """Return the state-space output k * u + D u before the activation."""
        length = u.shape[-1]
        k = self.kernel(length)
        if self.bidirectional:
            # Forward kernel at lags 0..L-1, backward kernel at lags -1..-L.
            k = F.pad(k[0], (0, length)) + F.pad(k[1].flip(-1), (length, 0))
        else:
            k = k[0]
        # Zero-padding to 2L makes the FFT's circular convolution a linear one.
        spectrum = torch.fft.rfft(u, n=2 * length) * torch.fft.rfft(k, n=2 * length)
        y = torch.fft.irfft(spectrum, n=2 * length)[..., :length]
        return y + self.D[:, None] * u

    def forward(self, u: Tensor) -> Tensor:
        return self.output(F.gelu(self.convolve(u)))


class SSSDDenoiser(nn.Module):
    """Predict the noise at hidden points with S4-based residual blocks.

    Defaults are scaled down from the paper (36 blocks of width 256) for short
    univariate windows. dt is accepted for the shared interface but unused, as
    in the reference.
    """

    def __init__(
        self,
        channels: int = 1,
        *,
        width: int = 64,
        skip_width: int | None = None,
        layers: int = 4,
        state_size: int = 64,
        step_dim: int = 128,
        step_hidden: int | None = None,
        bidirectional: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        if type(layers) is not int or layers < 1:
            raise ValueError("layers must be a positive integer")
        skip_width = width if skip_width is None else skip_width
        step_hidden = 2 * width if step_hidden is None else step_hidden
        self.channels = channels
        self.input = _conv(channels, width, 1)
        self.step_embedding = _StepEmbedding(step_dim, step_hidden)
        self.blocks = nn.ModuleList(
            _ResidualBlock(
                width,
                skip_width,
                channels,
                step_hidden,
                state_size,
                bidirectional,
                layer_norm,
            )
            for _ in range(layers)
        )
        self.output_hidden = _conv(skip_width, skip_width, 1)
        self.output = nn.Conv1d(skip_width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy: Tensor,
        steps: Tensor,
        *,
        observed: Tensor,
        mask: Tensor,
        dt: float,
    ) -> Tensor:
        if noisy.ndim != 3 or noisy.shape[-1] != self.channels:
            raise ValueError(
                f"Expected (batch, time, {self.channels}) values for "
                f"{self.channels} channels, got {tuple(noisy.shape)}"
            )
        observed = observed.masked_fill(mask, 0)
        # Observed entries hold the observations, as in SSSD's reverse process.
        inputs = torch.where(mask, noisy, observed).transpose(1, 2)
        # SSSD's convention: the indicator is 1 at observed entries.
        condition = torch.cat((observed, (~mask).to(observed.dtype)), dim=-1)
        condition = condition.transpose(1, 2)
        step = self.step_embedding(steps)
        hidden = F.relu(_standard(self.input(inputs)))
        skip = 0
        for block in self.blocks:
            hidden, block_skip = block(hidden, condition, step)
            skip = skip + block_skip
        hidden = skip / math.sqrt(len(self.blocks))
        hidden = F.relu(_standard(self.output_hidden(hidden)))
        return self.output(hidden).transpose(1, 2)


class _S4Layer(nn.Module):
    """Reference S4Layer: S4 block, residual connection, then LayerNorm."""

    def __init__(
        self, features: int, state_size: int, bidirectional: bool, layer_norm: bool
    ):
        super().__init__()
        self.s4 = S4D(features, state_size=state_size, bidirectional=bidirectional)
        self.norm = nn.LayerNorm(features) if layer_norm else nn.Identity()

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.s4(hidden) + hidden
        return _standard(self.norm(hidden.transpose(1, 2)).transpose(1, 2))


class _ResidualBlock(nn.Module):
    """Gated residual block: conv, S4, add conditioning, S4, tanh * sigmoid."""

    def __init__(
        self,
        width: int,
        skip_width: int,
        channels: int,
        step_hidden: int,
        state_size: int,
        bidirectional: bool,
        layer_norm: bool,
    ):
        super().__init__()
        self.step_projection = nn.Linear(step_hidden, width)
        self.convolution = _conv(width, 2 * width, 3)
        self.s4_first = _S4Layer(2 * width, state_size, bidirectional, layer_norm)
        self.condition = _conv(2 * channels, 2 * width, 1)
        self.s4_second = _S4Layer(2 * width, state_size, bidirectional, layer_norm)
        self.residual = _conv(width, width, 1)
        self.skip = _conv(width, skip_width, 1)

    def forward(
        self, hidden: Tensor, condition: Tensor, step: Tensor
    ) -> tuple[Tensor, Tensor]:
        values = hidden + self.step_projection(step)[:, :, None]
        values = self.s4_first(self.convolution(values))
        values = self.s4_second(values + self.condition(condition))
        signal, gate = values.chunk(2, dim=1)
        values = torch.tanh(signal) * torch.sigmoid(gate)
        return (hidden + self.residual(values)) * math.sqrt(0.5), self.skip(values)


class _StepEmbedding(nn.Module):
    """SSSD diffusion-step embedding: sinusoids followed by two SiLU layers.

    Frequencies fall from 1 to 10^-4, as in DiffWave and the SSSD reference.
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        if type(dim) is not int or dim < 4 or dim % 2:
            raise ValueError("step_dim must be an even integer of at least 4")
        self.projection1 = nn.Linear(dim, hidden)
        self.projection2 = nn.Linear(hidden, hidden)

    def sinusoids(self, steps: Tensor) -> Tensor:
        weight = self.projection1.weight
        half = weight.shape[1] // 2
        exponents = torch.arange(half, dtype=weight.dtype, device=weight.device)
        frequencies = torch.exp(-math.log(10000.0) * exponents / (half - 1))
        angles = steps.to(device=weight.device, dtype=weight.dtype)[:, None]
        angles = angles * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def forward(self, steps: Tensor) -> Tensor:
        embedding = F.silu(self.projection1(self.sinusoids(steps)))
        return F.silu(self.projection2(embedding))


def _standard(values: Tensor) -> Tensor:
    """Return values in standard (batch, features, time) memory layout.

    Transposes and Apple-GPU convolutions can produce channels-last strides,
    for which PyTorch's MPS backend (2.14) computes ReLU and tanh gradients
    incorrectly. Copying to the standard layout keeps the gradients exact.
    """
    return values.contiguous()


def _conv(inputs: int, outputs: int, kernel_size: int) -> nn.Module:
    """Weight-normalized convolution with PyTorch's default initialization.

    The reference applies Kaiming initialization after weight normalization,
    but the next forward pass recomputes the weight from its unchanged norm and
    direction, so its effective initialization is also PyTorch's default.
    """
    padding = (kernel_size - 1) // 2
    return weight_norm(nn.Conv1d(inputs, outputs, kernel_size, padding=padding))
