"""TimeAutoDiff-style encoder and denoiser for masked imputation.

TimeAutoDiff (Suh et al., 2024, arXiv:2406.16028) embeds the conditioning
series with sine-cosine features and a bidirectional GRU, then predicts
diffusion noise with a bidirectional LSTM over the noisy values, a timestamp
encoding, and a diffusion-step encoding (`BiRNN_score` in the reference
implementation, github.com/namjoonsuh/TimeAutoDiff). This module is an
independent reimplementation of those networks.

The reference diffuses in the latent space of a separately trained VAE. Here
diffusion runs directly on the hidden values, because the shared trainer and
`DDPMSampler` noise only hidden entries and keep observations fixed. The
encoder returns per-time context of width `hidden`; the denoiser predicts the
added noise, matching `noise_mse` and the default `DDPMSampler`.

Other differences from the reference: timestamps are window positions scaled
to [0, 1], calendar features (`time_info`) are omitted, the encoder also sees
raw observed values and the observation indicator, and the denoiser also sees
the hidden-entry mask so it can be trained without an encoder.
"""

import math

import torch
from torch import Tensor, nn


def sine_cosine(values: Tensor, terms: int) -> Tensor:
    """Return sin and cos of 2^k * pi * values for k < terms, per feature.

    Values of shape (..., features) become (..., 2 * terms * features). The
    features have period 2 in each value, so callers should also pass values
    whose range exceeds one period.
    """
    frequencies = 2 ** torch.arange(terms, device=values.device, dtype=values.dtype)
    angles = (math.pi * values[..., None] * frequencies).flatten(-2)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class PositionalEncoding(nn.Module):
    """Concatenate a learned linear map with fixed sinusoids of a scalar input."""

    def __init__(self, dim: int, max_value: float):
        super().__init__()
        if type(dim) is not int or dim < 2:
            raise ValueError("dim must be an integer of at least 2")
        if not math.isfinite(max_value) or max_value <= 0:
            raise ValueError("max_value must be finite and positive")
        self.max_value = max_value
        linear_dim = dim // 2
        periodic_dim = dim - linear_dim
        index = torch.arange(periodic_dim, dtype=torch.float32)
        shift = torch.zeros(periodic_dim)
        shift[::2] = math.pi / 2
        self.register_buffer(
            "scale", torch.exp(-2 * index * math.log(max_value) / periodic_dim)
        )
        self.register_buffer("shift", shift)
        self.linear = nn.Linear(1, linear_dim)

    def forward(self, values: Tensor) -> Tensor:
        """Encode values shaped (..., 1) as (..., dim)."""
        periodic = torch.sin(values * self.scale + self.shift)
        return torch.cat((self.linear(values / self.max_value), periodic), dim=-1)


class TimeAutoDiffEncoder(nn.Module):
    """Embed observed values and summarize them with a bidirectional GRU."""

    def __init__(
        self,
        *,
        channels: int = 1,
        hidden: int = 64,
        embedding: int = 128,
        layers: int = 2,
        terms: int = 8,
    ):
        super().__init__()
        for name, value in (
            ("channels", channels),
            ("hidden", hidden),
            ("embedding", embedding),
            ("layers", layers),
            ("terms", terms),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.channels = channels
        self.terms = terms
        features = 2 * terms * channels
        self.numeric = nn.Sequential(
            nn.Linear(features, features), nn.SiLU(), nn.Linear(features, features)
        )
        self.output_embedding = nn.Sequential(
            nn.Linear(features + 2 * channels, embedding),
            nn.ReLU(),
            nn.Linear(embedding, embedding),
        )
        self.gru = nn.GRU(
            embedding, hidden, layers, batch_first=True, bidirectional=True
        )
        self.output = nn.Linear(2 * hidden, hidden)

    def forward(self, observed: Tensor, *, mask: Tensor, dt: float) -> Tensor:
        """Return (batch, time, hidden) context from observed entries only."""
        if observed.ndim != 3 or observed.shape[-1] != self.channels:
            raise ValueError(f"Expected (batch, time, {self.channels}) observations")
        present = (~mask).to(observed.dtype)
        values = observed * present
        numeric = self.numeric(sine_cosine(values, self.terms))
        numeric = numeric * present.repeat_interleave(2 * self.terms, dim=-1)
        features = torch.cat((numeric, values, present), dim=-1)
        summary, _ = self.gru(self.output_embedding(features))
        return self.output(summary)


class TimeAutoDiffDenoiser(nn.Module):
    """Predict noise at hidden points with a bidirectional LSTM score network."""

    def __init__(
        self,
        *,
        channels: int = 1,
        hidden: int = 64,
        layers: int = 2,
        diffusion_steps: int = 1000,
    ):
        super().__init__()
        for name, value in (
            ("channels", channels),
            ("hidden", hidden),
            ("layers", layers),
            ("diffusion_steps", diffusion_steps),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.channels = channels
        self.input_projection = nn.Linear(2 * channels, hidden)
        self.time_encoding = PositionalEncoding(hidden, max_value=1.0)
        self.step_encoding = PositionalEncoding(hidden, max_value=diffusion_steps)
        self.projection = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU())
        self.lstm = nn.LSTM(
            hidden, hidden, layers, batch_first=True, bidirectional=True
        )
        self.norm = nn.LayerNorm(2 * hidden)
        self.output = nn.Linear(2 * hidden, channels)

    def forward(
        self,
        noisy: Tensor,
        steps: Tensor,
        *,
        observed: Tensor,
        mask: Tensor,
        dt: float,
        context: Tensor | None = None,
    ) -> Tensor:
        if noisy.ndim != 3 or noisy.shape[-1] != self.channels:
            raise ValueError(f"Expected (batch, time, {self.channels}) values")
        batch, length, _ = noisy.shape
        values = self.input_projection(torch.cat((noisy, mask.to(noisy.dtype)), dim=-1))
        if context is not None:
            values = values + context
        positions = torch.linspace(0, 1, length, dtype=noisy.dtype, device=noisy.device)
        times = self.time_encoding(positions[None, :, None].expand(batch, -1, -1))
        step_values = steps.to(noisy.dtype)[:, None, None].expand(-1, length, -1)
        hidden = self.projection(
            torch.cat((values, times, self.step_encoding(step_values)), dim=-1)
        )
        summary, _ = self.lstm(hidden)
        return self.output(self.norm(summary))
