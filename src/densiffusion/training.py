import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import Tensor, nn

from densiffusion.contracts import Array, Densifier, positive_int, validate_array
from densiffusion.objectives import DenoisingBatch, Objective, noise_mse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    max_grad_norm: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "diffusion_steps"):
            positive_int(name, getattr(self, name))
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.beta_start <= self.beta_end < 1:
            raise ValueError("Require 0 < beta_start <= beta_end < 1")


def train_densifier[T: nn.Module](
    model: T,
    train: Array,
    validation: Array,
    *,
    config: TrainingConfig | None = None,
    objective: Objective = noise_mse,
    dt: float = 1.0,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> T:
    """Fit 2*dt -> dt imputation and return the best validation weights.

    forward(noisy, steps, *, observed, mask, dt) returns objective predictions.
    Value tensors are (batch, time, channels); mask is True at hidden points.
    Even-length series lose the final point to keep both endpoints observed.
    """
    train = validate_array(train, (None, None, None))
    validation = validate_array(validation, (None, *train.shape[1:]))
    if train.shape[1] < 3:
        raise ValueError("Densifier training requires at least three time points")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if checkpoint is not None and checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")
    config = config or TrainingConfig()
    length = 2 * ((train.shape[1] - 1) // 2) + 1
    train, validation = train[:, :length], validation[:, :length]
    model.to(config.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model must have trainable parameters")
    dtype, device = parameters[0].dtype, parameters[0].device
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    beta = torch.linspace(
        config.beta_start,
        config.beta_end,
        config.diffusion_steps,
        dtype=dtype,
        device=device,
    )
    alpha_bar = (1 - beta).cumprod(0)
    seeds = np.random.SeedSequence(seed).generate_state(3, dtype=np.uint64)
    order_rng = np.random.default_rng(seeds[0])
    train_noise = torch.Generator().manual_seed(int(seeds[1]))
    torch.manual_seed(seed)
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        losses = []
        for training, data in ((True, train), (False, validation)):
            model.train(training)
            size = len(data)
            order = order_rng.permutation(size) if training else np.arange(size)
            noise_rng = (
                train_noise
                if training
                else torch.Generator().manual_seed(int(seeds[2]))
            )
            total = 0.0
            for start in range(0, size, config.batch_size):
                indices = order[start : start + config.batch_size]
                target = torch.as_tensor(data[indices], dtype=dtype, device=device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    loss = _loss(model, target, alpha_bar, noise_rng, dt, objective)
                if not isinstance(loss, Tensor) or loss.ndim != 0:
                    raise ValueError("Objective must return a scalar tensor")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite objective loss")
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        parameters, config.max_grad_norm, error_if_nonfinite=True
                    )
                    optimizer.step()
                total += loss.item() * len(indices)
            losses.append(total / size)

        logger.info("epoch=%d train_loss=%.6g validation_loss=%.6g", epoch, *losses)
        if losses[1] < best_loss:
            best_loss = losses[1]
            best_state = deepcopy(model.state_dict())
            if checkpoint is not None:
                _save_checkpoint(
                    checkpoint,
                    {
                        "model_state_dict": best_state,
                        "config": asdict(config),
                        "objective": getattr(
                            objective, "__qualname__", type(objective).__qualname__
                        ),
                        "stage": "densifier",
                        "factor": 2,
                        "target_dt": dt,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": losses[0],
                        "validation_loss": losses[1],
                    },
                )
    model.load_state_dict(best_state)
    model.eval()
    return model


def train_forecaster[T: nn.Module](
    model: T,
    densifier: Densifier,
    train: tuple[Array, Array],
    validation: tuple[Array, Array],
    *,
    config: TrainingConfig | None = None,
    objective: Objective = noise_mse,
    factor: int = 2,
    dt: float = 1.0,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> T:
    """Train on (history, future) windows with the trained densifier fixed."""
    raise NotImplementedError("Forecaster training is not implemented yet.")


def _loss(model, target, alpha_bar, rng, dt, objective):
    steps = torch.randint(len(alpha_bar), (len(target),), generator=rng, device="cpu")
    noise = torch.randn(
        target.shape, generator=rng, dtype=target.dtype, device="cpu"
    ).to(target.device)
    steps = steps.to(target.device)
    alpha = alpha_bar[steps, None, None]
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, 1::2] = True
    observed = target.masked_fill(mask, 0)
    noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    noisy = torch.where(mask, noisy, target)
    prediction = model(noisy, steps, observed=observed, mask=mask, dt=dt)
    batch = DenoisingBatch(
        clean=target,
        noisy=noisy,
        noise=noise,
        observed=observed,
        mask=mask,
        steps=steps,
        alpha_bar=alpha,
        dt=dt,
    )
    return objective(prediction, batch)


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(state, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
