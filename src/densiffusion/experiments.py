from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from densiffusion.conditioning import prepare_targets
from densiffusion.contracts import (
    Array,
    Densifier,
    Forecaster,
    Mask,
    has_regular_spacing,
    positive_int,
    validate_array,
    validate_times,
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
    train_mask: Mask | None = None,
    validation_mask: Mask | None = None,
    test_mask: Mask | None = None,
    test_times: Array | None = None,
) -> dict[str, float]:
    """Train on prepared splits and evaluate hidden test points with CRPS.

    sampler_factory binds the best validation model and its training config.
    The supplied model is updated in place; test values never enter training.
    """
    train = validate_array(train, (None, None, None))
    validation = validate_array(validation, (None, *train.shape[1:]))
    test = validate_array(test, (None, *train.shape[1:]))
    positive_int("n_samples", n_samples)
    factor = 2 if test_mask is None else 1
    if test_times is not None:
        test_times = validate_times(test_times, test.shape[:2])
        if not has_regular_spacing(test_times, dt):
            raise ValueError("test_times must be regularly spaced by dt")
    test, test_mask = prepare_targets(test, test_mask)
    if test_times is not None:
        test_times = test_times[:, : test.shape[1]]
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
        train_mask=train_mask,
        validation_mask=validation_mask,
    )
    sampler = sampler_factory(model, config)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(test), config.batch_size):
            target = test[start : start + config.batch_size]
            mask = test_mask[start : start + config.batch_size]
            options = {}
            if factor == 1:
                options["mask"] = mask
            if test_times is not None:
                options["times"] = test_times[start : start + len(target), ::factor]
            samples = impute(
                target[:, ::factor],
                sampler,
                factor=factor,
                n_samples=n_samples,
                dt=factor * dt,
                seed=int(rng.integers(2**63)),
                **options,
            )
            scores = evaluate(samples, target, mask=mask)
            total += scores["crps"] * int(mask.sum())
    return {"crps": total / int(test_mask.sum())}


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
