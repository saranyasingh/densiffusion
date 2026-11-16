"""PSNR-oriented helpers for the self-trained CSDI bootstrap pipeline."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def geometric_augment(images: np.ndarray, rng) -> np.ndarray:
    """Horizontal flips and 90-degree rotations (odd sizes stay odd)."""
    del rng  # deterministic dihedral set; kept for call-site symmetry with shifted()
    output = [images]
    flipped = images[:, :, ::-1]
    output.append(flipped.copy())
    for turns in (1, 2, 3):
        output.append(np.rot90(images, turns, axes=(1, 2)).copy())
        output.append(np.rot90(flipped, turns, axes=(1, 2)).copy())
    return np.concatenate(output)


def aggregate(draws: np.ndarray, how: str) -> np.ndarray:
    """Collapse the sample axis of (images, samples, ...)."""
    if how == "mean":
        return draws.mean(axis=1)
    if how == "median":
        return np.median(draws, axis=1)
    raise ValueError(f"Unknown aggregate {how!r}")


def clamp_observed(pred: np.ndarray, low: np.ndarray, factor: int = 2) -> np.ndarray:
    """Force observed grid pixels to the low-resolution input (..., H, W, C)."""
    out = np.array(pred, copy=True)
    out[..., ::factor, ::factor, :] = low
    return out


def clamp_layout(
    pred: np.ndarray, low_layout: np.ndarray, colours: int, factor: int = 2
) -> np.ndarray:
    """Clamp observed pixels for (..., size, size * colours) layouts."""
    *leading, size, width = pred.shape
    pixels = pred.reshape(*leading, size, width // colours, colours)
    low = low_layout.reshape(
        *low_layout.shape[:-1], low_layout.shape[-1] // colours, colours
    )
    return clamp_observed(pixels, low, factor).reshape(*leading, size, width)


def blend(model: np.ndarray, bilinear: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * model + (1 - alpha) * bilinear."""
    if not 0 <= alpha <= 1:
        raise ValueError("blend alpha must be in [0, 1]")
    return alpha * model + (1 - alpha) * bilinear


def residual_pair(layout: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Stack (layout - baseline, baseline) into doubled channel width."""
    return np.stack((layout - baseline, baseline), axis=-1).reshape(
        *layout.shape[:-1], -1
    )


def from_residual_pair(values: np.ndarray) -> np.ndarray:
    """Inverse of residual_pair: residual + baseline."""
    return values.reshape(*values.shape[:-1], -1, 2).sum(axis=-1)


def residual_mask(spatial_mask: np.ndarray) -> np.ndarray:
    """Hide residual channels where spatial_mask is True; keep baseline observed."""
    return np.stack((spatial_mask, np.zeros_like(spatial_mask)), axis=-1).reshape(
        spatial_mask.shape[0], -1
    )


def flip_layout(layout: np.ndarray, colours: int) -> np.ndarray:
    """Horizontal flip of a (..., size, size * colours) layout."""
    *leading, size, width = layout.shape
    pixels = layout.reshape(*leading, size, width // colours, colours)
    return pixels[..., :, ::-1, :].reshape(*leading, size, width)


class RefineCNN(nn.Module):
    """Tiny CNN: concat(bilinear, estimate) -> residual correction."""

    def __init__(self, colours: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * colours, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, colours, 3, padding=1),
        )

    def forward(self, bilinear: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        return estimate + self.net(torch.cat((bilinear, estimate), dim=1))


def train_refine(
    bilinear: np.ndarray,
    estimate: np.ndarray,
    target: np.ndarray,
    *,
    colours: int,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    path,
) -> RefineCNN:
    """Train RefineCNN with L2 on [0, 1] pixel images (N, H, W, C)."""
    torch.manual_seed(seed)
    model = RefineCNN(colours).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        return model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bi = torch.as_tensor(
        bilinear.transpose(0, 3, 1, 2), dtype=torch.float32, device=device
    )
    est = torch.as_tensor(
        estimate.transpose(0, 3, 1, 2), dtype=torch.float32, device=device
    )
    tgt = torch.as_tensor(
        target.transpose(0, 3, 1, 2), dtype=torch.float32, device=device
    )
    n = len(bi)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            pred = model(bi[index], est[index])
            loss = (pred - tgt[index]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    torch.save(model.state_dict(), path)
    return model


@torch.no_grad()
def apply_refine(
    model: RefineCNN, bilinear: np.ndarray, estimate: np.ndarray, device: str
) -> np.ndarray:
    bi = torch.as_tensor(
        bilinear.transpose(0, 3, 1, 2), dtype=torch.float32, device=device
    )
    est = torch.as_tensor(
        estimate.transpose(0, 3, 1, 2), dtype=torch.float32, device=device
    )
    return model(bi, est).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
