from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from densiffusion.contracts import (
    Array,
    Densifier,
    Forecaster,
    positive_int,
    validate_array,
)
from densiffusion.metrics import evaluate
from densiffusion.objectives import Objective, noise_mse
from densiffusion.pipeline import impute
from densiffusion.training import TrainingConfig, train_densifier


def run_experiment[T: nn.Module](
    model: T,
    train: Array,
    validation: Array,
    test: Array,
    *,
    sampler_factory: Callable[[T, TrainingConfig], Densifier],
    config: TrainingConfig | None = None,
    objective: Objective = noise_mse,
    n_samples: int = 100,
    dt: float = 1.0,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> dict[str, float]:
    """Train on prepared splits and evaluate hidden test points with CRPS.

    sampler_factory binds the best validation model and its training config.
    The supplied model is updated in place; test values never enter training.
    """
    train = validate_array(train, (None, None, None))
    validation = validate_array(validation, (None, *train.shape[1:]))
    test = validate_array(test, (None, *train.shape[1:]))
    positive_int("n_samples", n_samples)
    config = config or TrainingConfig()
    train_densifier(
        model,
        train,
        validation,
        config=config,
        objective=objective,
        dt=dt,
        seed=seed,
        checkpoint=checkpoint,
    )
    sampler = sampler_factory(model, config)
    test = test[:, : 2 * ((test.shape[1] - 1) // 2) + 1]
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(test), config.batch_size):
            target = test[start : start + config.batch_size]
            samples = impute(
                target[:, ::2],
                sampler,
                factor=2,
                n_samples=n_samples,
                dt=2 * dt,
                seed=int(rng.integers(2**63)),
            )
            mask = np.zeros(target.shape, dtype=bool)
            mask[:, 1::2] = True
            scores = evaluate(samples, target, mask=mask)
            total += scores["crps"] * len(target)
    return {"crps": total / len(test)}


def run_forecast_experiment[T: nn.Module](
    model: T,
    densifier: Densifier,
    train: tuple[Array, Array],
    validation: tuple[Array, Array],
    test: tuple[Array, Array],
    *,
    sampler_factory: Callable[[T, TrainingConfig], Forecaster],
    config: TrainingConfig | None = None,
    objective: Objective = noise_mse,
    factor: int = 2,
    n_histories: int = 1,
    n_futures: int = 100,
    dt: float = 1.0,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> dict[str, float]:
    """Train a forecaster and evaluate (history, future) test windows."""
    raise NotImplementedError("Forecast experiments are not implemented yet.")
