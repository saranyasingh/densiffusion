"""Two-dimensional U-Net noise predictor for image-shaped imputation.

Images use the layout of the image scripts: ``(batch, rows, columns * planes)``
tensors whose rows stand in for time and whose columns hold the planes of each
pixel (for example RGB, or a residual and its baseline) in adjacent channels.
The network reshapes this to ``(batch, planes, rows, columns)`` so that its
convolutions act on the pixel grid, instead of attending down columns and along
rows as CSDI does when it is given an image.

Conditioning follows CSDI: the observed plane with hidden entries cleared, the
noisy plane with observed entries cleared, and the observed mask are stacked as
input channels, and each residual block adds a learned embedding of the
diffusion step. Any row and column counts are supported, including odd sizes:
levels downsample with stride-two convolutions and upsample by interpolating
back to the size of the skip connection. The final convolution starts at zero.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from densiffusion.contracts import positive_int


class _StepEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        half = dimension // 2
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(10000.0) * torch.arange(half) / half),
            persistent=False,
        )
        self.projection1 = nn.Linear(dimension, dimension)
        self.projection2 = nn.Linear(dimension, dimension)

    def forward(self, steps: Tensor) -> Tensor:
        phase = steps[:, None] * self.frequencies[None]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return self.projection2(F.silu(self.projection1(embedding)))


class _ResidualBlock(nn.Module):
    def __init__(
        self, inputs: int, outputs: int, embedding: int, groups: int, dropout: float
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.embedding = nn.Linear(embedding, outputs)
        self.norm2 = nn.GroupNorm(groups, outputs)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.skip = (
            nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)
        )

    def forward(self, values: Tensor, embedding: Tensor) -> Tensor:
        y = self.conv1(F.silu(self.norm1(values)))
        y = y + self.embedding(embedding)[:, :, None, None]
        y = self.conv2(self.dropout(F.silu(self.norm2(y))))
        return self.skip(values) + y


class UNet2DDenoiser(nn.Module):
    """Predict noise on a pixel grid with a small convolutional U-Net.

    Inputs and output are ``(batch, rows, columns * planes)``; ``mask=True`` is
    hidden and hidden conditioning values are cleared defensively, including
    NaNs. ``steps`` is a (batch,) tensor of zero-based integer diffusion
    indices. ``dt`` is accepted for interface compatibility and does not affect
    the prediction, since rows are pixels rather than measurement times. Use
    with ImputationModel, noise_mse and DDPMSampler; no encoder is needed.

    ``hidden_channels`` times each ``channel_multipliers`` entry is the width of
    one resolution level, from full resolution downwards; every width must be
    divisible by ``groups`` (GroupNorm groups). Each level has
    ``blocks_per_level`` residual blocks on the way down and up, and two more
    at the bottom.
    """

    def __init__(
        self,
        planes: int = 1,
        *,
        hidden_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        blocks_per_level: int = 1,
        diffusion_embedding_dim: int = 128,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        for name, value in (
            ("planes", planes),
            ("hidden_channels", hidden_channels),
            ("blocks_per_level", blocks_per_level),
            ("diffusion_embedding_dim", diffusion_embedding_dim),
            ("groups", groups),
        ):
            positive_int(name, value)
        if not channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")
        for multiplier in channel_multipliers:
            positive_int("channel_multipliers", multiplier)
        if diffusion_embedding_dim % 2:
            raise ValueError("diffusion_embedding_dim must be even")
        widths = [hidden_channels * multiplier for multiplier in channel_multipliers]
        if any(width % groups for width in widths):
            raise ValueError("Every level width must be divisible by groups")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.planes = planes
        self.step_embedding = _StepEmbedding(diffusion_embedding_dim)
        self.input_projection = nn.Conv2d(3 * planes, hidden_channels, 3, padding=1)

        def block(inputs: int, outputs: int) -> _ResidualBlock:
            return _ResidualBlock(
                inputs, outputs, diffusion_embedding_dim, groups, dropout
            )

        self.encoder = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        channels = hidden_channels
        for level, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(block(channels, width))
                channels = width
            self.encoder.append(blocks)
            if level < len(widths) - 1:
                self.downsamplers.append(
                    nn.Conv2d(width, width, 3, stride=2, padding=1)
                )
        self.middle = nn.ModuleList(
            [block(channels, channels), block(channels, channels)]
        )
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level in reversed(range(len(widths))):
            width = widths[level]
            blocks = nn.ModuleList()
            for index in range(blocks_per_level):
                blocks.append(block(channels + width if index == 0 else width, width))
                channels = width
            self.decoder.append(blocks)
            if level > 0:
                self.upsamplers.append(nn.Conv2d(width, width, 3, padding=1))
        self.output_norm = nn.GroupNorm(groups, hidden_channels)
        self.output_projection = nn.Conv2d(hidden_channels, planes, 3, padding=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy: Tensor,
        steps: Tensor,
        *,
        observed: Tensor,
        mask: Tensor,
        dt: float,
    ) -> Tensor:
        if noisy.ndim != 3 or 0 in noisy.shape or noisy.shape[-1] % self.planes:
            raise ValueError("noisy must have shape (batch, rows, columns * planes)")
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

        batch, rows, width = noisy.shape
        grid = (batch, rows, width // self.planes, self.planes)

        def planes(values: Tensor) -> Tensor:
            return values.reshape(grid).permute(0, 3, 1, 2)

        # Two separate value planes, never clean targets in the hidden plane.
        inputs = torch.cat(
            (
                planes(observed.masked_fill(mask, 0)),
                planes(noisy.masked_fill(~mask, 0)),
                planes((~mask).to(noisy.dtype)),
            ),
            dim=1,
        )
        embedding = self.step_embedding(steps)
        values = self.input_projection(inputs)
        skips = []
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                values = block(values, embedding)
            skips.append(values)
            if level < len(self.downsamplers):
                values = self.downsamplers[level](values)
        for block in self.middle:
            values = block(values, embedding)
        for index, blocks in enumerate(self.decoder):
            values = torch.cat((values, skips.pop()), dim=1)
            for block in blocks:
                values = block(values, embedding)
            if index < len(self.upsamplers):
                values = F.interpolate(
                    values, size=skips[-1].shape[-2:], mode="nearest"
                )
                values = self.upsamplers[index](values)
        values = self.output_projection(F.silu(self.output_norm(values)))
        return values.permute(0, 2, 3, 1).reshape(batch, rows, width)
