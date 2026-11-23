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
    loss_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.loss_mask is None:
            object.__setattr__(self, "loss_mask", self.mask)


type Objective = Callable[[Tensor, DenoisingBatch], Tensor]


def noise_mse(prediction: Tensor, batch: DenoisingBatch) -> Tensor:
    """Mean squared noise prediction error at loss-mask points."""
    return _masked_mse(prediction, batch.noise, batch.loss_mask)


def reconstruction_mse(prediction: Tensor, batch: DenoisingBatch) -> Tensor:
    """Mean squared clean-value prediction error at loss-mask points."""
    return _masked_mse(prediction, batch.clean, batch.loss_mask)


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not isinstance(prediction, Tensor) or prediction.shape != target.shape:
        raise ValueError("Prediction must have the same shape as its target")
    return (prediction[mask] - target[mask]).square().mean()
