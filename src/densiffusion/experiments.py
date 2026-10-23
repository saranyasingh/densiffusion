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
from densiffusion.metrics import evaluate, evaluate_forecast
from densiffusion.objectives import Objective, noise_mse
from densiffusion.pipeline import forecast, impute
from densiffusion.training import TrainingConfig, train_densifier, train_forecaster


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
    n_augmentations: int = 1,
) -> dict[str, float]:
    """Fit a forecaster with a fixed densifier and score original-grid futures.

    Split trajectories before windowing. Test futures only enter scoring, after
    selecting the forecast checkpoint on validation noise loss. Pool all
    n_histories * n_futures draws before computing marginal metrics.
    """
    for name, value in (
        ("factor", factor),
        ("n_histories", n_histories),
        ("n_futures", n_futures),
        ("n_augmentations", n_augmentations),
    ):
        positive_int(name, value)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    checked = []
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if not isinstance(split, tuple) or len(split) != 2:
            raise ValueError(f"{name} must be a (history, future) tuple")
        history = validate_array(split[0], (None, None, None))
        future = validate_array(split[1], (len(history), None, history.shape[2]))
        if checked:
            validate_array(history, (None, *checked[0][0].shape[1:]))
            validate_array(future, (None, *checked[0][1].shape[1:]))
        checked.append((history, future))
    train, validation, (test_history, test_future) = checked
    config = config or TrainingConfig()
    train_forecaster(
        model,
        densifier,
        train,
        validation,
        config=config,
        objective=objective,
        factor=factor,
        dt=dt,
        seed=seed,
        checkpoint=checkpoint,
        n_augmentations=n_augmentations,
    )
    sampler = sampler_factory(model, config)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    totals = {}
    with torch.no_grad():
        for start in range(0, len(test_history), config.batch_size):
            history = test_history[start : start + config.batch_size]
            target = test_future[start : start + config.batch_size]
            samples = forecast(
                history,
                densifier,
                sampler,
                horizon=target.shape[1],
                factor=factor,
                n_histories=n_histories,
                n_futures=n_futures,
                dt=dt,
                seed=int(rng.integers(2**63)),
            )
            for name, value in evaluate_forecast(samples, target).items():
                totals[name] = totals.get(name, 0.0) + value * target.size
    return {name: total / test_future.size for name, total in totals.items()}
