import logging
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from densiffusion.conditioning import prepare_targets
from densiffusion.contracts import Array, Densifier, Mask, positive_int, validate_array
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
    schedule: Literal["linear", "quadratic"] = "linear"
    weight_decay: float = 0.0
    lr_decay_fractions: tuple[float, ...] = ()
    lr_decay_factor: float = 0.1
    validation_steps: Literal["sampled", "all"] = "sampled"
    validation_interval: int = 1

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "diffusion_steps", "validation_interval"):
            positive_int(name, getattr(self, name))
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.beta_start <= self.beta_end < 1:
            raise ValueError("Require 0 < beta_start <= beta_end < 1")
        if self.schedule not in ("linear", "quadratic"):
            raise ValueError("schedule must be 'linear' or 'quadratic'")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 < self.lr_decay_factor <= 1:
            raise ValueError("lr_decay_factor must be in (0, 1]")
        if any(not 0 < f < 1 for f in self.lr_decay_fractions) or any(
            a >= b
            for a, b in zip(
                self.lr_decay_fractions, self.lr_decay_fractions[1:], strict=False
            )
        ):
            raise ValueError("lr_decay_fractions must increase strictly within (0, 1)")
        if self.validation_steps not in ("sampled", "all"):
            raise ValueError("validation_steps must be 'sampled' or 'all'")

    def betas(self, *, dtype: torch.dtype, device: torch.device | str) -> Tensor:
        """The single schedule definition used by training and sampling."""
        if self.schedule == "quadratic":
            return torch.linspace(
                self.beta_start**0.5,
                self.beta_end**0.5,
                self.diffusion_steps,
                dtype=dtype,
                device=device,
            ).square()
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.diffusion_steps,
            dtype=dtype,
            device=device,
        )


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
    train_mask: Mask | None = None,
    validation_mask: Mask | None = None,
    validation_score: Callable[[T, int], float] | None = None,
    validation_score_interval: int | None = None,
    validation_score_epochs: Sequence[int] | None = None,
    epoch_callback: Callable[[dict], None] | None = None,
) -> T:
    """Fit masked imputation and return the best validation weights.

    forward(noisy, steps, *, observed, mask, dt) returns objective predictions.
    Value tensors are (batch, time, channels); mask is True at hidden points.
    Without masks, use alternate interior points, cropping even-length series.
    An optional validation_score(model, epoch) selects checkpoints by a finite
    lower-is-better score, evaluated without gradients in evaluation mode at
    validation intervals. Its Torch RNG use and module modes are isolated from
    training. epoch_callback receives one metrics dictionary per epoch.
    validation_score_interval can evaluate expensive custom scores less often
    than denoising validation. It must be a multiple of validation_interval;
    the final epoch is always scored. Only custom-score epochs select weights.
    Alternatively, validation_score_epochs supplies strictly increasing epochs,
    each a multiple of validation_interval and no greater than config.epochs.
    It requires validation_score and is mutually exclusive with the interval.
    The final epoch is still scored, including when the sequence is empty.
    """
    train = validate_array(train, (None, None, None))
    validation = validate_array(validation, (None, *train.shape[1:]))
    factor = 2 if train_mask is None else None
    train, train_mask = prepare_targets(train, train_mask)
    validation, validation_mask = prepare_targets(validation, validation_mask)
    _validate_training_options(dt, seed, checkpoint)
    return _fit_masked(
        model,
        train,
        validation,
        train_mask,
        validation_mask,
        config=config or TrainingConfig(),
        objective=objective,
        dt=dt,
        seed=seed,
        checkpoint=checkpoint,
        metadata={"stage": "densifier", "factor": factor, "target_dt": dt},
        validation_score=validation_score,
        validation_score_interval=validation_score_interval,
        validation_score_epochs=validation_score_epochs,
        epoch_callback=epoch_callback,
    )


def _fit_masked[T: nn.Module](
    model: T,
    train: Array,
    validation: Array,
    train_mask: Mask,
    validation_mask: Mask,
    *,
    config: TrainingConfig,
    objective: Objective,
    dt: float,
    seed: int,
    checkpoint: Path | None,
    metadata: dict,
    validation_score: Callable[[T, int], float] | None = None,
    validation_score_interval: int | None = None,
    validation_score_epochs: Sequence[int] | None = None,
    epoch_callback: Callable[[dict], None] | None = None,
) -> T:
    """Shared masked-diffusion optimizer for imputation and suffix forecasting."""
    validation_score_epochs = _validate_score_schedule(
        config, validation_score, validation_score_interval, validation_score_epochs
    )
    model.to(config.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Model must have trainable parameters")
    dtype, device = parameters[0].dtype, parameters[0].device
    optimizer = torch.optim.Adam(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(f * config.epochs)) for f in config.lr_decay_fractions],
        gamma=config.lr_decay_factor,
    )
    beta = config.betas(dtype=dtype, device=device)
    alpha_bar = (1 - beta).cumprod(0)
    seeds = np.random.SeedSequence(seed).generate_state(3, dtype=np.uint64)
    order_rng = np.random.default_rng(seeds[0])
    train_noise = torch.Generator().manual_seed(int(seeds[1]))
    torch.manual_seed(seed)
    best_score = float("inf")
    best_state = None
    selection_metric = (
        getattr(validation_score, "__qualname__", type(validation_score).__qualname__)
        if validation_score is not None
        else "validation_loss"
    )

    for epoch in range(1, config.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        validate = epoch % config.validation_interval == 0 or epoch == config.epochs
        losses = []
        for training, data, masks in (
            (True, train, train_mask),
            (False, validation, validation_mask),
        ):
            if not training and not validate:
                continue
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
                mask = torch.as_tensor(masks[indices], device=device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    if not training and config.validation_steps == "all":
                        loss = (
                            sum(
                                _loss(
                                    model,
                                    target,
                                    mask,
                                    alpha_bar,
                                    noise_rng,
                                    dt,
                                    objective,
                                    step=step,
                                )
                                for step in range(config.diffusion_steps)
                            )
                            / config.diffusion_steps
                        )
                    else:
                        loss = _loss(
                            model, target, mask, alpha_bar, noise_rng, dt, objective
                        )
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
                total += loss.item() * int(masks[indices].sum())
            losses.append(total / int(masks.sum()))

        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": losses[0],
            "learning_rate": learning_rate,
        }
        if not validate:
            logger.info(
                "epoch=%d train_loss=%.6g lr=%.6g", epoch, losses[0], learning_rate
            )
            if epoch_callback is not None:
                with _isolated_hook(model, device):
                    epoch_callback(record)
            continue
        score = losses[1] if validation_score is None else None
        score_interval = validation_score_interval or config.validation_interval
        score_due = epoch == config.epochs or (
            epoch in validation_score_epochs
            if validation_score_epochs is not None
            else epoch % score_interval == 0
        )
        if validation_score is not None and score_due:
            with _isolated_hook(model, device):
                model.eval()
                score = validation_score(model, epoch)
            if isinstance(score, (Tensor, np.ndarray)) and score.ndim != 0:
                raise ValueError("validation_score must return a scalar")
            score = float(score)
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite validation_score")
        record.update(
            validation_loss=losses[1],
            selection_metric=selection_metric,
        )
        if score is not None:
            record["selection_score"] = score
        logger.info(
            "epoch=%d train_loss=%.6g validation_loss=%.6g lr=%.6g%s",
            epoch,
            *losses,
            learning_rate,
            f" selection_score={score:.6g}" if score is not None else "",
        )
        if score is not None and score < best_score:
            best_score = score
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
                        **metadata,
                        "seed": seed,
                        **record,
                    },
                )
        if epoch_callback is not None:
            with _isolated_hook(model, device):
                epoch_callback(record)
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
    n_augmentations: int = 1,
    validation_score: Callable[[T, int], float] | None = None,
    validation_score_interval: int | None = None,
    validation_score_epochs: Sequence[int] | None = None,
    epoch_callback: Callable[[dict], None] | None = None,
) -> T:
    """Fit future-only diffusion to cached frozen-densifier augmentations.

    Inputs are pairs of coarse (batch, time, channels) history/future arrays;
    dt is their observed spacing. Histories are densified without future values.
    Futures are densified with the final history point as a left bridge anchor,
    then that anchor is removed. Thus dense futures have factor * horizon points
    and measured targets occur at factor - 1 :: factor. Each window receives
    n_augmentations independent paired draws, held fixed across training epochs.
    Validation augmentation and diffusion noise are also fixed. At factor=1,
    train directly on coarse windows without calling the densifier. Models may
    not share parameters, including at factor=1, to preserve densifier weights.
    validation_score, validation_score_interval, validation_score_epochs, and
    epoch_callback behave as in train_densifier; scoring can evaluate forecasts
    at original-grid times.
    """
    positive_int("factor", factor)
    positive_int("n_augmentations", n_augmentations)
    _validate_training_options(dt, seed, checkpoint)
    fine_dt = dt / factor
    if not np.isfinite(fine_dt) or fine_dt <= 0:
        raise ValueError("dt / factor must be finite and positive")
    train = _validate_forecast_pair(train)
    validation = _validate_forecast_pair(validation)
    if any(
        left.shape[1:] != right.shape[1:]
        for left, right in zip(train, validation, strict=True)
    ):
        raise ValueError("Training and validation window dimensions must match")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Model must have trainable parameters")
    config = config or TrainingConfig()
    validation_score_epochs = _validate_score_schedule(
        config, validation_score, validation_score_interval, validation_score_epochs
    )
    frozen_model = getattr(densifier, "model", densifier)
    if isinstance(frozen_model, nn.Module):
        _reject_shared_parameters(model, frozen_model)
        modes = [(module, module.training) for module in frozen_model.modules()]
    else:
        modes = []
    seeds = np.random.SeedSequence(seed).spawn(4)
    try:
        if modes and factor != 1:
            frozen_model.eval()
        with torch.no_grad():
            augmented_train = _augment_forecast_pair(
                train,
                densifier,
                factor,
                n_augmentations,
                dt,
                config.batch_size,
                np.random.default_rng(seeds[0]),
                np.random.default_rng(seeds[1]),
            )
            augmented_validation = _augment_forecast_pair(
                validation,
                densifier,
                factor,
                n_augmentations,
                dt,
                config.batch_size,
                np.random.default_rng(seeds[2]),
                np.random.default_rng(seeds[3]),
            )
    finally:
        for module, training in modes:
            module.training = training

    context_length = train[0].shape[1]
    horizon = train[1].shape[1]
    dense_context_length = (context_length - 1) * factor + 1
    train_mask = np.zeros(augmented_train.shape, dtype=bool)
    validation_mask = np.zeros(augmented_validation.shape, dtype=bool)
    train_mask[:, dense_context_length:] = True
    validation_mask[:, dense_context_length:] = True
    return _fit_masked(
        model,
        augmented_train,
        augmented_validation,
        train_mask,
        validation_mask,
        config=config,
        objective=objective,
        dt=fine_dt,
        seed=seed,
        checkpoint=checkpoint,
        metadata={
            "stage": "forecaster",
            "factor": factor,
            "observed_dt": dt,
            "target_dt": fine_dt,
            "context_length": context_length,
            "horizon": horizon,
            "dense_context_length": dense_context_length,
            "dense_horizon": factor * horizon,
            "n_augmentations": n_augmentations,
            "augmentation": {
                "cached": True,
                "history_conditioning": "history_only",
                "future_conditioning": "last_history_point_and_future",
                "independent_history_future_draws": True,
                "densifier_bypassed": factor == 1,
            },
        },
        validation_score=validation_score,
        validation_score_interval=validation_score_interval,
        validation_score_epochs=validation_score_epochs,
        epoch_callback=epoch_callback,
    )


@contextmanager
def _isolated_hook(model: nn.Module, device: torch.device):
    """Prevent optional monitoring from changing dropout RNG or module modes."""
    modes = [(module, module.training) for module in model.modules()]
    cuda_devices = [device.index] if device.type == "cuda" else []
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        for module, training in modes:
            module.training = training


def _validate_training_options(dt: float, seed: int, checkpoint: Path | None) -> None:
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if checkpoint is not None and checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}")


def _validate_score_interval(config, validation_score, interval):
    if interval is None:
        return
    positive_int("validation_score_interval", interval)
    if validation_score is None:
        raise ValueError("validation_score_interval requires validation_score")
    if interval % config.validation_interval:
        raise ValueError(
            "validation_score_interval must be a multiple of validation_interval"
        )


def _validate_score_schedule(config, validation_score, interval, epochs):
    _validate_score_interval(config, validation_score, interval)
    if epochs is None:
        return None
    if interval is not None:
        raise ValueError(
            "validation_score_epochs and validation_score_interval "
            "are mutually exclusive"
        )
    if validation_score is None:
        raise ValueError("validation_score_epochs requires validation_score")
    if not isinstance(epochs, Sequence) or isinstance(epochs, (str, bytes)):
        raise ValueError("validation_score_epochs must be a sequence of integers")
    # Copy caller-owned lists so callbacks cannot alter a running schedule.
    epochs = tuple(epochs)
    for epoch in epochs:
        positive_int("validation_score_epochs entry", epoch)
        if epoch > config.epochs:
            raise ValueError("validation_score_epochs entries must not exceed epochs")
        if epoch % config.validation_interval:
            raise ValueError(
                "validation_score_epochs entries must be "
                "multiples of validation_interval"
            )
    if any(left >= right for left, right in zip(epochs, epochs[1:], strict=False)):
        raise ValueError("validation_score_epochs must be strictly increasing")
    return epochs


def _validate_forecast_pair(values: tuple[Array, Array]) -> tuple[Array, Array]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise ValueError("Forecast data must be a (history, future) tuple")
    history = validate_array(values[0], (None, None, None))
    future = validate_array(values[1], (len(history), None, history.shape[2]))
    return history, future


def _reject_shared_parameters(model: nn.Module, frozen: nn.Module) -> None:
    """Reject tied parameters, including distinct Parameters sharing storage."""
    parameters = list(frozen.parameters())
    identities = {id(parameter) for parameter in parameters}
    storages = {
        (parameter.device, parameter.untyped_storage().data_ptr())
        for parameter in parameters
        if parameter.numel()
    }
    for parameter in model.parameters():
        if id(parameter) in identities or (
            parameter.numel()
            and (parameter.device, parameter.untyped_storage().data_ptr()) in storages
        ):
            raise ValueError("Forecaster and densifier must not share parameters")


def _augment_forecast_pair(
    data: tuple[Array, Array],
    densifier: Densifier,
    factor: int,
    n_augmentations: int,
    dt: float,
    batch_size: int,
    history_rng: np.random.Generator,
    future_rng: np.random.Generator,
) -> Array:
    history, future = data
    if factor == 1:
        return np.repeat(np.concatenate((history, future), axis=1), n_augmentations, 0)
    length = (history.shape[1] + future.shape[1] - 1) * factor + 1
    augmented = np.empty((len(history) * n_augmentations, length, history.shape[2]))
    for start in range(0, len(augmented), batch_size):
        stop = min(start + batch_size, len(augmented))
        indices = np.arange(start, stop) // n_augmentations
        past = _sample_dense(densifier, history[indices], factor, dt, history_rng)
        bridge = np.concatenate((history[indices, -1:], future[indices]), axis=1)
        target = _sample_dense(densifier, bridge, factor, dt, future_rng)
        augmented[start:stop] = np.concatenate((past, target[:, 1:]), axis=1)
    return augmented


def _sample_dense(
    densifier: Densifier,
    observed: Array,
    factor: int,
    dt: float,
    rng: np.random.Generator,
) -> Array:
    samples = densifier.sample(
        observed.copy(), factor=factor, n_samples=1, dt=dt, rng=rng
    )
    samples = validate_array(
        samples,
        (len(observed), 1, (observed.shape[1] - 1) * factor + 1, observed.shape[2]),
    )[:, 0]
    if not np.array_equal(samples[:, ::factor], observed):
        raise ValueError("Densifier must preserve every observed value")
    return samples.copy()


def _loss(model, target, mask, alpha_bar, rng, dt, objective, *, step=None):
    steps = (
        torch.randint(len(alpha_bar), (len(target),), generator=rng, device="cpu")
        if step is None
        else torch.full((len(target),), step, dtype=torch.long)
    )
    noise = torch.randn(
        target.shape, generator=rng, dtype=target.dtype, device="cpu"
    ).to(target.device)
    steps = steps.to(target.device)
    alpha = alpha_bar[steps, None, None]
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
