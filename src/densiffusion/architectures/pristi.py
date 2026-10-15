"""PriSTI encoder and denoiser, restricted to temporal dependencies.

PriSTI (Liu et al., ICDE 2023, arXiv:2302.09746) linearly interpolates the
observations, extracts a noise-free conditional feature from that
interpolation, and uses the feature to compute the temporal attention weights
of a CSDI-style noise predictor. Layer layouts follow the reference
implementation (github.com/LMZZML/PriSTI); this module is an independent
reimplementation.

The paper's spatial components (graph convolution, spatial attention, virtual
nodes, and adjacency matrices) are omitted. Channels are modelled separately
with shared weights and a learned channel embedding, so the networks never
exchange information across channels. The encoder returns a PriSTIContext that
the denoiser consumes, and the denoiser predicts the added noise, matching
`noise_mse` and the default `DDPMSampler`.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PriSTIContext(NamedTuple):
    """Encoder output; rows are (batch, channel) pairs in row-major order."""

    interpolated: Tensor  # (rows, time) linear interpolation of observations
    prior: Tensor  # (rows, time, width) conditional feature H^pri
    side: Tensor  # (rows, time, time_dim + channel_dim) side information


def interpolate(values: Tensor, observed: Tensor) -> Tensor:
    """Linearly interpolate along the last axis between observed entries.

    Leading and trailing gaps repeat the nearest observation, and rows without
    observations become zero, matching `torchcde.linear_interpolation_coeffs`
    on a regular grid. Values at unobserved entries are ignored.
    """
    if values.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("observed must be a boolean tensor shaped like values")
    length = values.shape[-1]
    values = values.masked_fill(~observed, 0)
    index = torch.arange(length, device=values.device).expand_as(values)
    previous = torch.where(observed, index, -1).cummax(-1).values
    following = torch.where(observed, index, length).flip(-1).cummin(-1).values
    following = following.flip(-1)
    left = torch.where(previous < 0, following, previous).clamp(0, length - 1)
    right = torch.where(following == length, previous, following).clamp(0, length - 1)
    start, end = values.gather(-1, left), values.gather(-1, right)
    weight = (index - left).to(values.dtype) / (right - left).clamp(min=1).to(
        values.dtype
    )
    return torch.where(observed, values, start + weight * (end - start))


class PriSTIEncoder(nn.Module):
    """Interpolate observations and extract the conditional feature H^pri."""

    def __init__(
        self,
        *,
        channels: int = 1,
        width: int = 64,
        heads: int = 8,
        feedforward: int = 64,
        time_dim: int = 128,
        channel_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if time_dim % 2:
            raise ValueError("time_dim must be even")
        self.channels = channels
        self.time_dim = time_dim
        self.channel_embedding = nn.Embedding(channels, channel_dim)
        self.value_projection = _projection(1, width)
        self.side_projection = _projection(time_dim + channel_dim, width)
        self.guidance = _GuidanceBlock(width, heads, feedforward, dropout)

    def forward(self, observed: Tensor, *, mask: Tensor, dt: float) -> PriSTIContext:
        if observed.ndim != 3 or observed.shape[-1] != self.channels:
            raise ValueError(f"Expected (batch, time, {self.channels}) observations")
        batch, length, _ = observed.shape
        interpolated = interpolate(_fold(observed), ~_fold(mask))
        side = self._side_information(batch, length)
        hidden = self.value_projection(interpolated[..., None])
        prior = F.relu(self.guidance(hidden + self.side_projection(side)))
        return PriSTIContext(interpolated, prior, side)

    def _side_information(self, batch: int, length: int) -> Tensor:
        """Concatenate the time encoding U_tem with the channel embedding U_spa.

        U_spa replaces the paper's node embedding and lets shared weights tell
        channels apart. With one channel it is constant, so after the linear
        side projections it is equivalent to an extra bias.
        """
        embedding = self.channel_embedding.weight
        time = _time_embedding(length, self.time_dim, embedding)
        side = torch.cat(
            (
                time.expand(self.channels, -1, -1),
                embedding[:, None].expand(-1, length, -1),
            ),
            dim=-1,
        )
        return side.repeat(batch, 1, 1)


class PriSTIDenoiser(nn.Module):
    """Predict noise at hidden points with prior-guided temporal attention."""

    def __init__(
        self,
        *,
        width: int = 64,
        heads: int = 8,
        layers: int = 4,
        feedforward: int = 64,
        time_dim: int = 128,
        channel_dim: int = 16,
        step_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if type(layers) is not int or layers < 1:
            raise ValueError("layers must be a positive integer")
        self.width = width
        self.side_dim = time_dim + channel_dim
        self.input_projection = _projection(2, width)
        self.step_embedding = _StepEmbedding(step_dim)
        self.layers = nn.ModuleList(
            _NoiseLayer(width, heads, feedforward, self.side_dim, step_dim, dropout)
            for _ in range(layers)
        )
        self.output_hidden = _projection(width, width)
        self.output = nn.Linear(width, 1)
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
        context: PriSTIContext,
    ) -> Tensor:
        batch, length, channels = noisy.shape
        rows = batch * channels
        if (
            context.interpolated.shape != (rows, length)
            or context.prior.shape != (rows, length, self.width)
            or context.side.shape != (rows, length, self.side_dim)
        ):
            raise ValueError("PriSTI context does not match the denoiser input")
        target = _fold(noisy.masked_fill(~mask, 0))
        inputs = torch.stack((target, context.interpolated), dim=-1)
        hidden = F.relu(self.input_projection(inputs))
        step = self.step_embedding(steps).repeat_interleave(channels, dim=0)
        skips = []
        for layer in self.layers:
            hidden, skip = layer(hidden, step, context.prior, context.side)
            skips.append(skip)
        hidden = torch.stack(skips).sum(0) / math.sqrt(len(self.layers))
        output = self.output(F.relu(self.output_hidden(hidden)))
        return _unfold(output.squeeze(-1), channels)


class _AttentionLayer(nn.Module):
    """Post-norm Transformer layer; queries and keys may differ from values."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
        )
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query: Tensor, values: Tensor) -> Tensor:
        attended = self.attention(query, query, values, need_weights=False)[0]
        values = self.norm1(values + self.dropout1(attended))
        return self.norm2(values + self.dropout2(self.feedforward(values)))


class _GuidanceBlock(nn.Module):
    """Temporal branch of the reference conditional feature extraction."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float):
        super().__init__()
        self.temporal = _AttentionLayer(width, heads, feedforward, dropout)
        self.temporal_norm = nn.GroupNorm(4, width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width)
        )
        self.output_norm = nn.GroupNorm(4, width)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.temporal(hidden, hidden)
        hidden = _group_norm(self.temporal_norm, hidden)
        return _group_norm(self.output_norm, hidden + self.feedforward(hidden))


class _NoiseLayer(nn.Module):
    """Residual layer whose attention weights come from the prior alone."""

    def __init__(
        self,
        width: int,
        heads: int,
        feedforward: int,
        side_dim: int,
        step_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.step_projection = nn.Linear(step_dim, width)
        self.temporal = _AttentionLayer(width, heads, feedforward, dropout)
        self.mid_projection = _projection(width, 2 * width)
        self.side_projection = _projection(side_dim, 2 * width)
        self.output_projection = _projection(width, 2 * width)

    def forward(
        self, hidden: Tensor, step: Tensor, prior: Tensor, side: Tensor
    ) -> tuple[Tensor, Tensor]:
        values = hidden + self.step_projection(step)[:, None]
        values = self.temporal(prior, values)
        values = self.mid_projection(values) + self.side_projection(side)
        gate, signal = values.chunk(2, dim=-1)
        values = self.output_projection(torch.sigmoid(gate) * torch.tanh(signal))
        residual, skip = values.chunk(2, dim=-1)
        return (hidden + residual) / math.sqrt(2.0), skip


class _StepEmbedding(nn.Module):
    """CSDI diffusion-step embedding: sinusoids followed by two SiLU layers.

    Steps are scaled by frequencies from 1 to 10^4. PriSTI uses this embedding
    (paper Fig. 2); each noise layer adds its own projection before attention.
    Sinusoids use at least float32 because these angles overflow float16.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("step_dim must be an even integer of at least 4")
        self.projection1 = nn.Linear(dim, dim)
        self.projection2 = nn.Linear(dim, dim)

    def forward(self, steps: Tensor) -> Tensor:
        weight = self.projection1.weight
        dtype = torch.promote_types(weight.dtype, torch.float32)
        half = weight.shape[1] // 2
        exponents = torch.arange(half, dtype=dtype, device=weight.device)
        frequencies = 10.0 ** (exponents / (half - 1) * 4.0)
        angles = steps.to(device=weight.device, dtype=dtype)[:, None] * frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        embedding = F.silu(self.projection1(embedding.to(weight.dtype)))
        return F.silu(self.projection2(embedding))


def _projection(inputs: int, outputs: int) -> nn.Linear:
    """Pointwise projection with the reference's Kaiming-normal weights."""
    layer = nn.Linear(inputs, outputs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _time_embedding(length: int, dim: int, like: Tensor) -> Tensor:
    """Transformer sine-cosine encoding U_tem of window positions 0..length-1."""
    position = torch.arange(length, dtype=like.dtype, device=like.device)[:, None]
    exponents = torch.arange(0, dim, 2, dtype=like.dtype, device=like.device)
    angles = position * 10000.0 ** (-exponents / dim)
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


def _group_norm(norm: nn.GroupNorm, hidden: Tensor) -> Tensor:
    return norm(hidden.transpose(1, 2)).transpose(1, 2)


def _fold(values: Tensor) -> Tensor:
    """(batch, time, channels) -> (batch * channels, time)."""
    return values.transpose(1, 2).reshape(-1, values.shape[1])


def _unfold(values: Tensor, channels: int) -> Tensor:
    """(batch * channels, time) -> (batch, time, channels)."""
    return values.reshape(-1, channels, values.shape[-1]).transpose(1, 2)
