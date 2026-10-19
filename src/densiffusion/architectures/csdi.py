"""CSDI conditional noise predictor for regular-grid time series.

Architecture: https://arxiv.org/abs/2107.03502 (Section 5, Appendix E.1).
Reference implementation: https://github.com/ermongroup/CSDI.
See docs/csdi.md for conventions and reproduction differences.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from densiffusion.contracts import positive_int


def _projection(inputs: int, outputs: int) -> nn.Linear:
    # A pointwise Linear is the reference's kernel-size-one Conv1d.
    layer = nn.Linear(inputs, outputs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class _DiffusionEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        half = dimension // 2
        self.register_buffer(
            "frequencies",
            10.0 ** (torch.arange(half) / (half - 1) * 4.0),
            persistent=False,
        )
        self.projection1 = nn.Linear(dimension, dimension)
        self.projection2 = nn.Linear(dimension, dimension)

    def forward(self, steps: Tensor) -> Tensor:
        # Evaluate the reference's zero-indexed lookup formula on demand, so
        # architecture and TrainingConfig cannot disagree on the table length.
        phase = steps[:, None] * self.frequencies[None]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return F.silu(self.projection2(F.silu(self.projection1(embedding))))


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        side: int,
        diffusion: int,
        heads: int,
        feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion, hidden)
        self.time_attention = nn.TransformerEncoderLayer(
            hidden,
            heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.feature_attention = nn.TransformerEncoderLayer(
            hidden,
            heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.mid_projection = _projection(hidden, 2 * hidden)
        self.side_projection = _projection(side, 2 * hidden)
        self.output_projection = _projection(hidden, 2 * hidden)

    def forward(
        self, values: Tensor, side: Tensor, diffusion: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, length, features, hidden = values.shape
        y = values + self.diffusion_projection(diffusion)[:, None, None, :]
        if length > 1:
            y = y.transpose(1, 2).reshape(batch * features, length, hidden)
            y = self.time_attention(y)
            y = y.reshape(batch, features, length, hidden).transpose(1, 2)
        # As in the reference, bypass feature attention for univariate data.
        if features > 1:
            y = self.feature_attention(y.reshape(batch * length, features, hidden))
            y = y.reshape(batch, length, features, hidden)
        gate, value = (self.mid_projection(y) + self.side_projection(side)).chunk(
            2, dim=-1
        )
        residual, skip = self.output_projection(gate.sigmoid() * value.tanh()).chunk(
            2, dim=-1
        )
        return (values + residual) / math.sqrt(2), skip


class CSDIDenoiser(nn.Module):
    """Predict noise with CSDI's temporal/feature residual attention network.

    Inputs and output are (batch, time, n_channels). ``mask=True`` is hidden;
    hidden conditioning values are cleared defensively, including NaNs.
    ``steps`` is a (batch,) tensor of zero-based integer diffusion indices.
    Measurement times are local ``arange(time) * dt`` with *target* spacing dt,
    embedded separately from diffusion steps. Use with ImputationModel,
    noise_mse and DDPMSampler; no encoder is needed.

    Defaults match the reference network sizes. Smaller sizes are useful for
    local smoke runs. Time embedding dimensions must be even; the diffusion
    embedding dimension must be even and at least four.
    """

    def __init__(
        self,
        n_channels: int,
        *,
        hidden_channels: int = 64,
        residual_layers: int = 4,
        attention_heads: int = 8,
        diffusion_embedding_dim: int = 128,
        time_embedding_dim: int = 128,
        feature_embedding_dim: int = 16,
        feedforward_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        for name, value in (
            ("n_channels", n_channels),
            ("hidden_channels", hidden_channels),
            ("residual_layers", residual_layers),
            ("attention_heads", attention_heads),
            ("diffusion_embedding_dim", diffusion_embedding_dim),
            ("time_embedding_dim", time_embedding_dim),
            ("feature_embedding_dim", feature_embedding_dim),
            ("feedforward_dim", feedforward_dim),
        ):
            positive_int(name, value)
        if hidden_channels % attention_heads:
            raise ValueError("hidden_channels must be divisible by attention_heads")
        if diffusion_embedding_dim < 4 or diffusion_embedding_dim % 2:
            raise ValueError("diffusion_embedding_dim must be even and at least 4")
        if time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.n_channels = n_channels
        self.register_buffer(
            "time_frequencies",
            10000.0 ** (-torch.arange(0, time_embedding_dim, 2) / time_embedding_dim),
            persistent=False,
        )
        self.feature_embedding = nn.Embedding(n_channels, feature_embedding_dim)
        self.diffusion_embedding = _DiffusionEmbedding(diffusion_embedding_dim)
        self.input_projection = _projection(2, hidden_channels)
        self.residual_layers = nn.ModuleList(
            _ResidualBlock(
                hidden_channels,
                time_embedding_dim + feature_embedding_dim + 1,
                diffusion_embedding_dim,
                attention_heads,
                feedforward_dim,
                dropout,
            )
            for _ in range(residual_layers)
        )
        self.output_projection1 = _projection(hidden_channels, hidden_channels)
        self.output_projection2 = _projection(hidden_channels, 1)
        nn.init.zeros_(self.output_projection2.weight)

    def forward(
        self,
        noisy: Tensor,
        steps: Tensor,
        *,
        observed: Tensor,
        mask: Tensor,
        dt: float,
    ) -> Tensor:
        if noisy.ndim != 3 or noisy.shape[-1] != self.n_channels or 0 in noisy.shape:
            raise ValueError("noisy must have shape (batch, time, n_channels)")
        if observed.shape != noisy.shape or mask.shape != noisy.shape:
            raise ValueError("observed and mask must have the same shape as noisy")
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean, with True at hidden points")
        if steps.shape != (len(noisy),) or steps.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("steps must contain one integer diffusion index per batch")
        if (steps < 0).any():
            raise ValueError("steps must be zero-based nonnegative indices")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        batch, length, features = noisy.shape
        # Two separate value planes, never clean targets in the hidden plane.
        inputs = torch.stack(
            (observed.masked_fill(mask, 0), noisy.masked_fill(~mask, 0)), dim=-1
        )
        values = F.relu(self.input_projection(inputs))

        times = torch.arange(length, dtype=noisy.dtype, device=noisy.device) * dt
        phase = times[:, None] * self.time_frequencies[None]
        time_embedding = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)
        feature_embedding = self.feature_embedding.weight
        side = torch.cat(
            (
                time_embedding[None, :, None, :].expand(batch, -1, features, -1),
                feature_embedding[None, None, :, :].expand(batch, length, -1, -1),
                (~mask).to(noisy.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        diffusion = self.diffusion_embedding(steps)
        skips = []
        for layer in self.residual_layers:
            values, skip = layer(values, side, diffusion)
            skips.append(skip)
        values = torch.stack(skips).sum(0) / math.sqrt(len(skips))
        return self.output_projection2(F.relu(self.output_projection1(values))).squeeze(
            -1
        )
