"""Impute synthetic images with CSDI and score them against the exact GP posterior.

    uv run --extra research --with matplotlib python scripts/impute_images_csdi.py \
        --output outputs/csdi_images

Images are SIZE x SIZE draws of a stationary Gaussian random field whose
covariance is a repository kernel applied to the Euclidean distance between
pixels (in pixel units). `--mask grid` keeps every other row and column, the
2D analogue of 2*dt -> dt densification (x2 super-resolution, 3 of 4 pixels
hidden); `--mask random` hides pixels independently with `--probability`.
`--mask upsample` observes the whole SIZE x SIZE test image of the grid run
(same seed) and adds a hidden pixel between every pair of neighbours, giving a
(2*SIZE-1)-pixel grid. The lengthscale is doubled in fine pixels so observed
pixels keep their distribution, and the hidden truth is an exact GP draw.

CSDI needs no changes: an image is passed as (batch, time, channels) with rows
as time and columns as channels, so its temporal and feature attention become
attention down columns and along rows. Training and sampling reuse
`train_densifier` and `DDPMSampler`, which keep observed pixels fixed.

The exact GP posterior is the best achievable imputer for these images, and
linear interpolation is a deterministic reference whose CRPS equals its MAE.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import griddata

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.gp.kernels import get_kernel
from densiffusion.gp.sampling import covariance_factor
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

STATIONARY_KERNELS = ("exponential", "matern_3_2", "matern_5_2")
KERNEL_NAMES = {
    "exponential": "Exponential",
    "matern_3_2": "Matérn 3/2",
    "matern_5_2": "Matérn 5/2",
}


def pixel_covariance(size: int, kernel: str, **params: float) -> np.ndarray:
    """Covariance between all pixels of a size x size image, in row-major order."""
    rows, columns = np.divmod(np.arange(size * size), size)
    distances = np.hypot(
        np.subtract.outer(rows, rows), np.subtract.outer(columns, columns)
    )
    # Stationary kernels depend only on |t - s|, so distances stand in for t.
    values = get_kernel(kernel)(distances.ravel(), np.zeros(1), **params)
    return values.reshape(distances.shape)


def sample_fields(covariance, size, n_images, rng) -> np.ndarray:
    factor = covariance_factor(covariance)
    noise = rng.standard_normal((n_images, size * size))
    return (noise @ factor.T).reshape(n_images, size, size)


def make_masks(n_images, size, args, rng) -> np.ndarray:
    """Boolean (images, size, size) masks with True at hidden pixels."""
    if args.mask in ("grid", "upsample"):
        mask = np.ones((size, size), dtype=bool)
        mask[::2, ::2] = False
        return np.broadcast_to(mask, (n_images, size, size)).copy()
    masks = rng.random((n_images, size, size)) < args.probability
    flat = masks.reshape(n_images, -1)
    rows = np.arange(n_images)
    flat[rows, rng.integers(size * size, size=n_images)] = False
    flat[rows, rng.integers(size * size, size=n_images)] = True
    return masks


def gp_posterior_samples(images, masks, covariance, n_samples, rng) -> np.ndarray:
    """Exact posterior draws of hidden pixels, sharing work across equal masks."""
    n_images, size, _ = images.shape
    samples = np.repeat(images[:, None], n_samples, axis=1)
    flat_masks = masks.reshape(n_images, -1)
    groups = {}
    for index, mask in enumerate(flat_masks):
        groups.setdefault(mask.tobytes(), []).append(index)
    for indices in groups.values():
        hidden = flat_masks[indices[0]]
        observed = ~hidden
        cholesky = np.linalg.cholesky(covariance[np.ix_(observed, observed)])
        cross = covariance[np.ix_(hidden, observed)]
        solved = np.linalg.solve(cholesky, cross.T)
        posterior = covariance[np.ix_(hidden, hidden)] - solved.T @ solved
        factor = covariance_factor(posterior)
        values = images[indices].reshape(len(indices), -1)
        weights = np.linalg.solve(cholesky.T, solved).T
        means = values[:, observed] @ weights.T
        noise = rng.standard_normal((len(indices), n_samples, int(hidden.sum())))
        draws = means[:, None] + noise @ factor.T
        flat = samples[indices].reshape(len(indices), n_samples, -1)
        flat[:, :, hidden] = draws
        samples[indices] = flat.reshape(len(indices), n_samples, size, size)
    return samples


def linear_interpolation(images, masks) -> np.ndarray:
    """Piecewise-linear fill of hidden pixels, nearest-neighbour outside the hull."""
    size = images.shape[1]
    grid = np.stack(np.divmod(np.arange(size * size), size), axis=-1)
    filled = images.copy()
    for image, mask, output in zip(images, masks, filled, strict=True):
        known = ~mask.ravel()
        points, values = grid[known], image.ravel()[known]
        query = grid[mask.ravel()]
        estimate = griddata(points, values, query, method="linear")
        missing = np.isnan(estimate)
        if missing.any():
            estimate[missing] = griddata(
                points, values, query[missing], method="nearest"
            )
        output[mask] = estimate
    return filled[:, None]


def score(samples, images, masks) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "crps": crps(samples, images, mask=masks),
        "rmse": float(np.sqrt(np.mean((mean - images)[masks] ** 2))),
    }


def plot_examples(path, images, masks, samples, n_examples=4, observed=None) -> None:
    from matplotlib.figure import Figure

    columns = [
        ("Truth", images),
        ("Observed", np.where(masks, np.nan, images) if observed is None else observed),
        ("Linear interp.", samples["linear_interpolation"][:, 0]),
        ("GP posterior mean", samples["gp_posterior"].mean(axis=1)),
        ("CSDI sample", samples["csdi"][:, 0]),
        ("CSDI mean", samples["csdi"].mean(axis=1)),
    ]
    figure = Figure(figsize=(2 * len(columns), 2.1 * n_examples))
    axes = figure.subplots(n_examples, len(columns))
    limit = np.abs(images[:n_examples]).max()
    for row in range(n_examples):
        for column, (title, values) in enumerate(columns):
            axis = axes[row, column]
            axis.imshow(values[row], cmap="viridis", vmin=-limit, vmax=limit)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--kernel", choices=STATIONARY_KERNELS, default="matern_3_2")
    parser.add_argument("--lengthscale", type=float, default=5.0)
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument(
        "--mask", choices=("grid", "random", "upsample"), default="grid"
    )
    parser.add_argument("--probability", type=float, default=0.75)
    parser.add_argument("--train-images", type=int, default=2000)
    parser.add_argument("--validation-images", type=int, default=200)
    parser.add_argument("--test-images", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--residual-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.output / "training.log"),
            logging.StreamHandler(),
        ],
    )
    streams = np.random.SeedSequence(args.seed).spawn(6)
    covariance = pixel_covariance(
        args.size, args.kernel, variance=args.variance, lengthscale=args.lengthscale
    )
    observed_size = args.size
    if args.mask == "upsample":
        coarse_covariance = covariance
        args.size = 2 * observed_size - 1
        covariance = pixel_covariance(
            args.size,
            args.kernel,
            variance=args.variance,
            lengthscale=2 * args.lengthscale,
        )
    splits = {}
    for stream, (split, n_images) in zip(
        streams[:3],
        (
            ("train", args.train_images),
            ("validation", args.validation_images),
            ("test", args.test_images),
        ),
        strict=True,
    ):
        rng = np.random.default_rng(stream)
        if args.mask == "upsample" and split == "test":
            coarse = sample_fields(coarse_covariance, observed_size, n_images, rng)
            masks = make_masks(n_images, args.size, args, rng)
            images = np.zeros((n_images, args.size, args.size))
            images[:, ::2, ::2] = coarse
            images = gp_posterior_samples(
                images, masks, covariance, 1, np.random.default_rng(streams[5])
            )[:, 0]
        else:
            images = sample_fields(covariance, args.size, n_images, rng)
            masks = make_masks(n_images, args.size, args, rng)
        splits[split] = (images, masks)
    (train, train_mask), (validation, validation_mask), (test, test_mask) = (
        splits[split] for split in ("train", "validation", "test")
    )
    logging.info(
        "%d train / %d validation / %d test images of %dx%d, %.0f%% hidden",
        len(train),
        len(validation),
        len(test),
        args.size,
        args.size,
        100 * test_mask.mean(),
    )

    samples = {
        "gp_posterior": gp_posterior_samples(
            test,
            test_mask,
            covariance,
            args.n_samples,
            np.random.default_rng(streams[3]),
        ),
        "linear_interpolation": linear_interpolation(test, test_mask),
    }

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
    torch.manual_seed(args.seed)
    model = ImputationModel(
        CSDIDenoiser(
            n_channels=args.size,
            hidden_channels=args.hidden_channels,
            residual_layers=args.residual_layers,
            attention_heads=args.attention_heads,
            time_embedding_dim=64,
        )
    )
    parameters = sum(p.numel() for p in model.parameters())
    logging.info("CSDI with %d parameters", parameters)
    started = time.perf_counter()
    train_densifier(
        model,
        train,
        validation,
        config=config,
        dt=1.0,
        seed=args.seed,
        checkpoint=args.output / "csdi.pt",
        train_mask=train_mask,
        validation_mask=validation_mask,
    )
    trained = time.perf_counter()
    samples["csdi"] = impute(
        test,
        DDPMSampler(model, replace(config, batch_size=args.sample_batch_size)),
        factor=1,
        mask=test_mask,
        n_samples=args.n_samples,
        dt=1.0,
        seed=int(np.random.default_rng(streams[4]).integers(2**63)),
    )
    sampled = time.perf_counter()

    scores = {name: score(draws, test, test_mask) for name, draws in samples.items()}
    results = {
        "scores": scores,
        "parameters": parameters,
        "train_seconds": trained - started,
        "sample_seconds": sampled - trained,
        "args": {key: str(value) for key, value in vars(args).items()},
        "training": asdict(config),
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(
        args.output / "samples.npz",
        images=test,
        masks=test_mask,
        **{name: draws[:8] for name, draws in samples.items()},
    )
    logging.info("%s", json.dumps(scores, indent=2))

    if args.no_plot:
        return
    plot_examples(
        args.output / "examples.png",
        test,
        test_mask,
        samples,
        observed=test[:, ::2, ::2] if args.mask == "upsample" else None,
    )
    from plot_forecast_table import render_table

    best = scores["gp_posterior"]["crps"]
    rows = ("gp_posterior", "csdi", "linear_interpolation")
    hidden = {
        "grid": "every other row and column observed (x2 super-resolution)",
        "upsample": (
            f"full {observed_size}×{observed_size} images observed, "
            "new pixels added between them (x2 super-resolution)"
        ),
        "random": f"{100 * args.probability:.0f}% of pixels hidden at random",
    }[args.mask]
    render_table(
        [
            "Exact GP posterior\n(best possible)",
            "CSDI" + (" ×2\ndensified" if args.mask != "random" else ""),
            "Linear interpolation",
        ],
        [
            (
                KERNEL_NAMES[args.kernel],
                [
                    (
                        scores[name]["crps"],
                        None
                        if name == "gp_posterior"
                        else scores[name]["crps"] / best - 1,
                    )
                    for name in rows
                ],
            )
        ],
        candidates=(1,),
        heading="Imputer",
        title="CSDI image imputation",
        subtitle=(
            f"{KERNEL_NAMES[args.kernel]} Gaussian random field, lengthscale "
            f"{args.lengthscale:g} px of the {observed_size}×{observed_size} "
            f"grid, variance {args.variance:g}\n"
            f"{len(test)} test images of {args.size}×{args.size} pixels; {hidden}"
        ),
        output=args.output / "table.png",
    )


if __name__ == "__main__":
    main()
