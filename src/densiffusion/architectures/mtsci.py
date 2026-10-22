"""MTSCI conditional noise predictor and its intra-consistency objective.

MTSCI (Zhou et al., CIKM 2024, arXiv:2408.05740) is a CSDI-style conditional
denoiser for multivariate imputation. Each residual layer adds the side
information before temporal and feature Transformer layers, and the decoder
re-reads the single-plane input before predicting noise. Training adds an
intra-consistency term: the network also encodes the window with the roles
of conditioning and target swapped (a complementary mask), and an InfoNCE
loss aligns the combined forward/reverse representation with that of the
fully noised window. Layer layouts and the loss follow the reference
implementation (github.com/JeremyChou28/MTSCI); this module is an independent
reimplementation.

The paper's inter-consistency term, which mixes the following window's
observations into the decoder input during training, is omitted: the shared
trainer supplies independent windows. At inference both versions are the
same network. See docs/mtsci.md for all differences.

Use the denoiser without an encoder. It predicts noise, so it samples with
the default `DDPMSampler`; train it with `MTSCIObjective` (or `noise_mse`
alone, which disables the consistency term).
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from densiffusion.contracts import positive_int
from densiffusion.objectives import DenoisingBatch, noise_mse


def _projection(inputs: int, outputs: int) -> nn.Linear:
    """Pointwise projection with the reference's Kaiming-normal Conv1d weights."""
    layer = nn.Linear(inputs, outputs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _transformer(width: int, heads: int, feedforward: int, dropout: float):
    return nn.TransformerEncoderLayer(
        width,
        heads,
        dim_feedforward=feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )


class _DiffusionEmbedding(nn.Module):
    """CSDI/MTSCI step embedding: sinusoids from 1 to 10^4, two SiLU layers."""

    def __init__(self, dimension: int):
        super().__init__()
        self.projection1 = nn.Linear(dimension, dimension)
        self.projection2 = nn.Linear(dimension, dimension)

    def forward(self, steps: Tensor) -> Tensor:
        # Sinusoids use at least float32 because these angles overflow float16.
        weight = self.projection1.weight
        dtype = torch.promote_types(weight.dtype, torch.float32)
        half = weight.shape[1] // 2
        exponents = torch.arange(half, dtype=dtype, device=weight.device)
        frequencies = 10.0 ** (exponents / (half - 1) * 4.0)
        phase = steps.to(device=weight.device, dtype=dtype)[:, None] * frequencies
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        embedding = embedding.to(self.projection1.weight.dtype)
        return F.silu(self.projection2(F.silu(self.projection1(embedding))))


class _ResidualBlock(nn.Module):
    """Reference block: step and side information enter before attention."""

    def __init__(
        self,
        width: int,
        side: int,
        diffusion: int,
        heads: int,
        feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion, width)
        self.side_projection = _projection(side, width)
        self.time_attention = _transformer(width, heads, feedforward, dropout)
        self.feature_attention = _transformer(width, heads, feedforward, dropout)
        self.mid_projection = _projection(width, 2 * width)
        self.output_projection = _projection(width, 2 * width)

    def forward(
        self, values: Tensor, side: Tensor, diffusion: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, length, features, width = values.shape
        y = values + self.diffusion_projection(diffusion)[:, None, None]
        y = y + self.side_projection(side)
        if length > 1:
            y = y.transpose(1, 2).reshape(batch * features, length, width)
            y = self.time_attention(y)
            y = y.reshape(batch, features, length, width).transpose(1, 2)
        # As in the reference, bypass feature attention for univariate data.
        if features > 1:
            y = self.feature_attention(y.reshape(batch * length, features, width))
            y = y.reshape(batch, length, features, width)
        gate, signal = self.mid_projection(y).chunk(2, dim=-1)
        y = self.output_projection(gate.sigmoid() * signal.tanh())
        residual, skip = y.chunk(2, dim=-1)
        return (values + residual) / math.sqrt(2.0), skip


class MTSCIDenoiser(nn.Module):
    """Predict noise at hidden points with MTSCI's residual attention network.

    Inputs and output are (batch, time, n_channels); ``mask=True`` is hidden.
    The network reads one value plane: observations at observed entries and
    noisy values at hidden entries. Side information is the sine-cosine
    encoding of window positions 0..time-1 (as in the reference, ``dt`` is not
    used), a learned channel embedding, and the observed-entry indicator.

    Defaults match the reference ETT/Weather configuration (64 channels,
    4 layers, 8 heads; 50 quadratic steps up to beta 0.2 for point missingness).
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
        self.hidden_channels = hidden_channels
        self.register_buffer(
            "time_frequencies",
            10000.0 ** (-torch.arange(0, time_embedding_dim, 2) / time_embedding_dim),
            persistent=False,
        )
        self.feature_embedding = nn.Embedding(n_channels, feature_embedding_dim)
        self.diffusion_embedding = _DiffusionEmbedding(diffusion_embedding_dim)
        self.input_projection = _projection(1, hidden_channels)
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
        # Decoder: merge the encoder output with the raw input plane, then
        # two pointwise layers. Only the final weight is zeroed, as in the
        # reference, so an untrained network predicts a constant bias.
        self.input_merge = _projection(hidden_channels + 1, hidden_channels)
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
        self._check(noisy, steps, observed, mask, dt)
        values = torch.where(mask, noisy, observed.masked_fill(mask, 0))
        hidden = self.encode(values, steps, conditioning=~mask)
        merged = self.merge(hidden, values)
        output = self.output_projection2(F.relu(self.output_projection1(merged)))
        return output.squeeze(-1)

    def encode(self, values: Tensor, steps: Tensor, *, conditioning: Tensor) -> Tensor:
        """Sum of residual skips, (batch, time, channels, hidden_channels).

        ``conditioning`` is True where ``values`` holds clean observations.
        """
        batch, length, features = values.shape
        hidden = F.relu(self.input_projection(values[..., None]))
        positions = torch.arange(length, dtype=values.dtype, device=values.device)
        phase = positions[:, None] * self.time_frequencies.to(values.dtype)[None]
        time_embedding = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(1)
        side = torch.cat(
            (
                time_embedding[None, :, None].expand(batch, -1, features, -1),
                self.feature_embedding.weight[None, None].expand(batch, length, -1, -1),
                conditioning.to(values.dtype)[..., None],
            ),
            dim=-1,
        )
        diffusion = self.diffusion_embedding(steps)
        skips = []
        for layer in self.residual_layers:
            hidden, skip = layer(hidden, side, diffusion)
            skips.append(skip)
        return torch.stack(skips).sum(0) / math.sqrt(len(skips))

    def merge(self, hidden: Tensor, values: Tensor) -> Tensor:
        """Decoder input: encoder output concatenated with the input plane."""
        return self.input_merge(torch.cat((hidden, values[..., None]), dim=-1))

    def consistency_loss(
        self, batch: DenoisingBatch, *, temperature: float = 0.1
    ) -> Tensor:
        """Reference intra-consistency InfoNCE loss for one training batch.

        Three views of each window share the batch's noise draw:
        forward (observations clean, targets noisy), reverse (targets clean,
        observations noisy) and negative (everything noisy, no conditioning).
        The forward decoder input at targets and the reverse encoding at
        observations form one representation; InfoNCE pulls it towards the
        negative view's encoding of the same window, with the other windows
        in the batch as negatives.
        """
        clean, mask, steps = batch.clean, batch.mask, batch.steps
        noisy = batch.alpha_bar.sqrt() * clean + (1 - batch.alpha_bar).sqrt() * (
            batch.noise
        )
        forward = torch.where(mask, noisy, clean)
        reverse = torch.where(mask, clean, noisy)
        views = torch.cat((forward, reverse, noisy))
        conditioning = torch.cat((~mask, mask, torch.zeros_like(mask)))
        encoded = self.encode(views, steps.repeat(3), conditioning=conditioning)
        forward_hidden, reverse_hidden, negative_hidden = encoded.chunk(3)
        forward_hidden = self.merge(forward_hidden, forward)
        combined = torch.where(mask[..., None], forward_hidden, reverse_hidden)
        return info_nce(
            negative_hidden.flatten(1), combined.flatten(1), temperature=temperature
        )

    def _check(self, noisy, steps, observed, mask, dt) -> None:
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


def info_nce(query: Tensor, positive: Tensor, *, temperature: float) -> Tensor:
    """InfoNCE with in-batch negatives, as `info_nce.InfoNCE()` with no negatives.

    Rows are L2-normalised; row i of ``positive`` is the positive for row i of
    ``query`` and every other row is a negative.
    """
    if query.shape != positive.shape or query.ndim != 2:
        raise ValueError("query and positive must be matching (batch, dim) tensors")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = F.normalize(query, dim=-1) @ F.normalize(positive, dim=-1).T
    labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels)


class MTSCIObjective:
    """``noise_mse`` plus the weighted intra-consistency loss during training.

    Bind it to the denoiser being trained. Validation runs in eval mode and
    scores noise MSE only, as the reference does. The default weight of 5
    and temperature of 0.1 are the reference values (loss = 1 * noise + 5 *
    consistency).
    """

    def __init__(
        self,
        denoiser: MTSCIDenoiser,
        *,
        consistency_weight: float = 5.0,
        temperature: float = 0.1,
    ):
        if not math.isfinite(consistency_weight) or consistency_weight < 0:
            raise ValueError("consistency_weight must be finite and nonnegative")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.denoiser = denoiser
        self.consistency_weight = consistency_weight
        self.temperature = temperature

    def __call__(self, prediction: Tensor, batch: DenoisingBatch) -> Tensor:
        loss = noise_mse(prediction, batch)
        if not self.denoiser.training or self.consistency_weight == 0:
            return loss
        consistency = self.denoiser.consistency_loss(
            batch, temperature=self.temperature
        )
        return loss + self.consistency_weight * consistency
