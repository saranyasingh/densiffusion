"""Benchmark every imputation architecture against the exact GP posterior.

    uv run --extra research python scripts/benchmark_gp.py --output outputs/bench

For each GP dataset, independent windows of WINDOW points at spacing DT are drawn
separately for train, validation, and test. Every architecture is trained with
the same TrainingConfig and scored by CRPS on the same hidden test entries and
draw count. The exact GP posterior is the best achievable imputer for this data;
linear interpolation is a deterministic reference whose CRPS equals its MAE.

`--mask alternate` hides every other point (2*dt -> dt densification);
`--mask random` hides entries independently with `--probability`. Windows start
at time zero, so nonstationary kernels (Brownian motion) start at zero too.
Results are written after each model so long runs keep partial scores.
"""

import argparse
import json
import logging
import time
import zlib
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.architectures.pristi import PriSTIDenoiser, PriSTIEncoder
from densiffusion.architectures.sssd import SSSDDenoiser
from densiffusion.architectures.timeautodiff import (
    TimeAutoDiffDenoiser,
    TimeAutoDiffEncoder,
)
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import sample_gp
from densiffusion.metrics import evaluate
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

DT, WINDOW = 0.1, 33

DATASETS = {
    "exponential": dict(kernel="exponential", variance=1.0, lengthscale=1.0),
    "matern_3_2": dict(kernel="matern_3_2", variance=1.0, lengthscale=1.0),
    "matern_5_2": dict(kernel="matern_5_2", variance=1.0, lengthscale=1.0),
    "periodic": dict(kernel="periodic", variance=1.0, lengthscale=1.0, period=1.0),
    "brownian": dict(kernel="brownian", variance=1.0),
    "fbm_h0.3": dict(kernel="fractional_brownian", variance=1.0, hurst=0.3),
    "fbm_h0.7": dict(kernel="fractional_brownian", variance=1.0, hurst=0.7),
}

MODELS = {
    "csdi": lambda config: ImputationModel(CSDIDenoiser(n_channels=1)),
    "pristi": lambda config: ImputationModel(
        PriSTIDenoiser(), encoder=PriSTIEncoder(channels=1)
    ),
    "sssd": lambda config: ImputationModel(SSSDDenoiser(1)),
    "timeautodiff": lambda config: ImputationModel(
        TimeAutoDiffDenoiser(channels=1, diffusion_steps=config.diffusion_steps),
        encoder=TimeAutoDiffEncoder(channels=1),
    ),
}

# Noiseless periodic draws repeat exactly, so posterior covariances are singular
# and roundoff produces tiny negative eigenvalues (variance units, variance 1).
COVARIANCE_ATOL = 1e-8


def make_masks(shape, args, rng):
    """Return hidden-entry masks, keeping a hidden and an observed entry each."""
    if args.mask == "alternate":
        masks = np.zeros(shape, dtype=bool)
        masks[:, 1::2] = True
        return masks
    masks = rng.random(shape) < args.probability
    rows = np.arange(shape[0])
    masks[rows, rng.integers(shape[1], size=shape[0]), 0] = True
    masks[rows, rng.integers(shape[1], size=shape[0]), 0] = False
    return masks


def make_dataset(name, args):
    """Independent (values, mask) pairs per split from separate random streams."""
    streams = np.random.SeedSequence([args.seed, zlib.crc32(name.encode())]).spawn(4)
    times = np.arange(WINDOW) * DT
    mask_rng = np.random.default_rng(streams[3])
    splits = {}
    for stream, (split, n_series) in zip(
        streams[:3],
        (
            ("train", args.train_series),
            ("validation", args.validation_series),
            ("test", args.test_series),
        ),
        strict=True,
    ):
        values = sample_gp(
            times,
            rng=np.random.default_rng(stream),
            n_samples=n_series,
            **DATASETS[name],
        )[:, :, None]
        splits[split] = (values, make_masks(values.shape, args, mask_rng))
    return splits


def linear_interpolation(target, mask):
    times = np.arange(target.shape[1]) * DT
    interpolated = target.copy()
    for series in range(len(target)):
        known = ~mask[series, :, 0]
        interpolated[series, :, 0] = np.interp(
            times, times[known], target[series, known, 0]
        )
    return interpolated[:, None]


def score_baselines(name, target, mask, args):
    sampler = GPPosteriorSampler(
        partial(gp_posterior, **DATASETS[name]), covariance_atol=COVARIANCE_ATOL
    )
    samples = impute(
        target,
        sampler,
        factor=1,
        mask=mask,
        n_samples=args.n_samples,
        dt=DT,
        seed=args.seed,
    )
    return {
        "gp_posterior": evaluate(samples, target, mask=mask)["crps"],
        "linear_interpolation": evaluate(
            linear_interpolation(target, mask), target, mask=mask
        )["crps"],
    }


def train_and_score(model_name, splits, config, args):
    (train, train_mask), (validation, validation_mask), (test, test_mask) = (
        splits[split] for split in ("train", "validation", "test")
    )
    torch.manual_seed(args.seed)
    model = MODELS[model_name](config)
    parameters = sum(p.numel() for p in model.parameters())
    started = time.perf_counter()
    train_densifier(
        model,
        train,
        validation,
        config=config,
        dt=DT,
        seed=args.seed,
        train_mask=train_mask,
        validation_mask=validation_mask,
    )
    trained = time.perf_counter()
    samples = impute(
        test,
        DDPMSampler(model, replace(config, batch_size=args.sample_batch_size)),
        factor=1,
        mask=test_mask,
        n_samples=args.n_samples,
        dt=DT,
        seed=args.seed,
    )
    return {
        "crps": evaluate(samples, test, mask=test_mask)["crps"],
        "parameters": parameters,
        "train_seconds": trained - started,
        "sample_seconds": time.perf_counter() - trained,
    }


def summary_table(results, models):
    columns = ["gp_posterior", "linear_interpolation", *models]
    lines = [
        "CRPS (lower is better); parentheses give the ratio to the GP posterior.",
        "",
        "| dataset | " + " | ".join(columns) + " | best model |",
        "|" + "---|" * (len(columns) + 2),
    ]
    for dataset, scores in results.items():
        optimum = scores["gp_posterior"]
        cells = []
        for column in columns:
            if column not in scores:
                cells.append("-")
                continue
            value = scores[column]
            value = value["crps"] if isinstance(value, dict) else value
            cells.append(f"{value:.4f} ({value / optimum:.2f})")
        trained = [m for m in models if m in scores]
        best = min(trained, key=lambda m: scores[m]["crps"]) if trained else "-"
        lines.append(f"| {dataset} | " + " | ".join(cells) + f" | {best} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--mask", choices=("alternate", "random"), default="alternate")
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--train-series", type=int, default=1000)
    parser.add_argument("--validation-series", type=int, default=200)
    parser.add_argument("--test-series", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--sample-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.output / "training.log"),
            logging.StreamHandler(),
        ],
    )
    # CSDI's quadratic schedule, shared so differences come from the networks.
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        beta_start=1e-4,
        beta_end=args.beta_end,
        schedule="quadratic",
        lr_decay_fractions=(0.75, 0.9),
        validation_interval=max(1, args.epochs // 20),
        device=args.device,
    )
    plan = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "dt": DT,
        "window": WINDOW,
        "datasets": {name: DATASETS[name] for name in args.datasets},
        "training": asdict(config),
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    results = {}
    for dataset in args.datasets:
        splits = make_dataset(dataset, args)
        test, test_mask = splits["test"]
        results[dataset] = score_baselines(dataset, test, test_mask, args)
        logging.info("%s baselines: %s", dataset, results[dataset])
        for model_name in args.models:
            logging.info("Training %s on %s", model_name, dataset)
            results[dataset][model_name] = train_and_score(
                model_name, splits, config, args
            )
            logging.info("%s %s: %s", dataset, model_name, results[dataset][model_name])
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2, allow_nan=False) + "\n"
            )
    table = summary_table(results, args.models)
    (args.output / "results.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
