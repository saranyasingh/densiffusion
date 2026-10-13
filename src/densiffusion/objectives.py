from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class DenoisingBatch:
    clean: Tensor
    noisy: Tensor
    noise: Tensor
    observed: Tensor
    mask: Tensor
    steps: Tensor
    alpha_bar: Tensor
    dt: float


type Objective = Callable[[Tensor, DenoisingBatch], Tensor]


def noise_mse(prediction: Tensor, batch: DenoisingBatch) -> Tensor:
    """Mean squared noise prediction error at hidden points."""
    return _masked_mse(prediction, batch.noise, batch.mask)


def reconstruction_mse(prediction: Tensor, batch: DenoisingBatch) -> Tensor:
    """Mean squared clean-value prediction error at hidden points."""
    return _masked_mse(prediction, batch.clean, batch.mask)


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not isinstance(prediction, Tensor) or prediction.shape != target.shape:
        raise ValueError("Prediction must have the same shape as its target")
    return (prediction[mask] - target[mask]).square().mean()
