"""Train the TimeAutoDiff-style imputer on a generated dataset and report CRPS.

Run from the repository root after generating data, for example:

    uv run densiffusion generate configs/gp.example.toml --output data/gp.npz
    uv run --extra research python scripts/run_timeautodiff.py data/gp.npz

With the default alternate mask, every other point of each test window is
hidden (2*dt -> dt densification). `--mask random` hides entries
independently with `--probability`. For datasets from the GP generator, the
exact GP posterior is scored on the same targets, masks, and draw count; it
is the best achievable imputer for that data.
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.timeautodiff import (
    TimeAutoDiffDenoiser,
    TimeAutoDiffEncoder,
)
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.conditioning import prepare_targets
from densiffusion.data import make_windows, split_data
from densiffusion.experiments import run_experiment
from densiffusion.gp.posterior import gp_posterior
from densiffusion.metrics import evaluate
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig

GP_GENERATOR = "densiffusion.gp.sampling:generate"


def random_masks(
    shape: tuple[int, int, int], probability: float, rng: np.random.Generator
) -> np.ndarray:
    """Hide entries independently, keeping a hidden and an observed entry each."""
    masks = rng.random(shape) < probability
    batch, length, _ = shape
    rows = np.arange(batch)
    masks[rows, rng.integers(length, size=batch), 0] = True
    masks[rows, rng.integers(length, size=batch), 0] = False
    return masks


def gp_crps(
    test: np.ndarray,
    test_times: np.ndarray,
    mask: np.ndarray | None,
    generation: dict,
    *,
    n_samples: int,
    dt: float,
    seed: int,
) -> float:
    params = dict(generation["params"])
    sampler = GPPosteriorSampler(partial(gp_posterior, **params))
    target, hidden = prepare_targets(test, mask)
    times = test_times[:, : target.shape[1]]
    if mask is None:
        samples = impute(
            target[:, ::2],
            sampler,
            factor=2,
            n_samples=n_samples,
            dt=2 * dt,
            seed=seed,
            times=times[:, ::2],
        )
    else:
        samples = impute(
            target,
            sampler,
            factor=1,
            mask=hidden,
            n_samples=n_samples,
            dt=dt,
            seed=seed,
            times=times,
        )
    return evaluate(samples, target, mask=hidden)["crps"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help=".npz from `densiffusion generate`")
    parser.add_argument("--window-length", type=int, default=65)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--mask", choices=("alternate", "random"), default="alternate")
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="Write scores as JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with np.load(args.data, allow_pickle=False) as dataset:
        values = dataset["values"]
        times = np.broadcast_to(dataset["times"], values.shape[:2])
        generation = json.loads(str(dataset["config"]))
    dt = float(generation["dt"])

    splits = split_data(
        values, times=times, train_fraction=0.7, validation_fraction=0.15
    )
    (train, _), (validation, _), (test, test_times) = (
        make_windows(v, times=t, window_length=args.window_length, stride=args.stride)
        for v, t in splits
    )
    masks = {}
    if args.mask == "random":
        rng = np.random.default_rng(args.seed)
        masks = {
            f"{name}_mask": random_masks(split.shape, args.probability, rng)
            for name, split in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        }
    print(f"windows: train={len(train)} validation={len(validation)} test={len(test)}")

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    channels = values.shape[2]
    torch.manual_seed(args.seed)
    model = ImputationModel(
        TimeAutoDiffDenoiser(
            channels=channels,
            hidden=args.hidden,
            layers=args.layers,
            diffusion_steps=args.diffusion_steps,
        ),
        encoder=TimeAutoDiffEncoder(
            channels=channels, hidden=args.hidden, layers=args.layers
        ),
    )
    scores = {
        "timeautodiff_crps": run_experiment(
            model,
            train,
            validation,
            test,
            sampler_factory=DDPMSampler,
            config=config,
            n_samples=args.n_samples,
            dt=dt,
            seed=args.seed,
            checkpoint=args.checkpoint,
            test_times=test_times,
            **masks,
        )["crps"]
    }
    if generation["generator"] == GP_GENERATOR:
        scores["gp_posterior_crps"] = gp_crps(
            test,
            test_times,
            masks.get("test_mask"),
            generation,
            n_samples=args.n_samples,
            dt=dt,
            seed=args.seed,
        )
    print(json.dumps(scores, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**scores, **vars(args)}, default=str))


if __name__ == "__main__":
    main()
