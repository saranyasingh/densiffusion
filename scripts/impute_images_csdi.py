"""Impute synthetic images with CSDI or a U-Net and score them against the GP posterior.

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
`--denoiser unet` swaps CSDI for a 2D convolutional U-Net on the same pixel
grid. `--residual` diffuses only the hidden residual around linear
interpolation: the interpolant becomes a second, fully observed plane of every
pixel, so the network sees it and observed pixels stay exact by construction.

The exact GP posterior is the best achievable imputer for these images, and
linear interpolation is a deterministic reference whose CRPS equals its MAE.
Scores report the shared ensemble CRPS, its fair (unbiased) variant, and for
the GP posterior also the exact CRPS of its Gaussian marginals.
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
from scipy.stats import norm

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.architectures.unet2d import UNet2DDenoiser
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
DENOISER_NAMES = {"csdi": "CSDI", "unet": "U-Net"}


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


def _mask_groups(masks: np.ndarray) -> dict[bytes, list[int]]:
    """Indices of the images sharing each distinct flattened mask."""
    groups = {}
    for index, mask in enumerate(masks.reshape(len(masks), -1)):
        groups.setdefault(mask.tobytes(), []).append(index)
    return groups


def gp_posterior_samples(images, masks, covariance, n_samples, rng) -> np.ndarray:
    """Exact posterior draws of hidden pixels, sharing work across equal masks."""
    n_images, size, _ = images.shape
    samples = np.repeat(images[:, None], n_samples, axis=1)
    flat_masks = masks.reshape(n_images, -1)
    for indices in _mask_groups(masks).values():
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


def gp_posterior_crps(images, masks, covariance) -> float:
    """Exact mean CRPS of the Gaussian posterior marginals at hidden pixels."""
    n_images = len(images)
    flat_masks = masks.reshape(n_images, -1)
    values = images.reshape(n_images, -1)
    total, count = 0.0, 0
    for indices in _mask_groups(masks).values():
        hidden = flat_masks[indices[0]]
        observed = ~hidden
        cholesky = np.linalg.cholesky(covariance[np.ix_(observed, observed)])
        cross = covariance[np.ix_(hidden, observed)]
        solved = np.linalg.solve(cholesky, cross.T)
        variance = np.diag(covariance)[hidden] - np.sum(solved**2, axis=0)
        weights = np.linalg.solve(cholesky.T, solved).T
        means = values[indices][:, observed] @ weights.T
        sigma = np.sqrt(np.clip(variance, np.finfo(float).tiny, None))
        z = (values[indices][:, hidden] - means) / sigma
        scores = sigma * (
            z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)
        )
        total += scores.sum()
        count += scores.size
    return float(total / count)


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


def residual_layout(images, masks, baseline) -> tuple[np.ndarray, np.ndarray]:
    """Pair each pixel's hidden residual with its fully observed interpolant.

    Returns (images, size, 2 * size) values and masks: the residual of a pixel
    sits next to its baseline value, in the layout the denoisers expect for
    multi-plane pixels. The residual is exactly zero at observed pixels.
    """
    n_images, size, _ = images.shape
    values = np.stack((images - baseline, baseline), axis=-1)
    hidden = np.stack((masks, np.zeros_like(masks)), axis=-1)
    return values.reshape(n_images, size, -1), hidden.reshape(n_images, size, -1)


def from_residual_layout(values: np.ndarray) -> np.ndarray:
    """Inverse of residual_layout for any leading shape: residual + baseline."""
    return values.reshape(*values.shape[:-1], -1, 2).sum(axis=-1)


def score(samples, images, masks) -> dict[str, float]:
    mean = samples.mean(axis=1)
    scores = {
        "crps": crps(samples, images, mask=masks),
        "rmse": float(np.sqrt(np.mean((mean - images)[masks] ** 2))),
    }
    if samples.shape[1] > 1:
        scores["crps_fair"] = crps(samples, images, mask=masks, fair=True)
    return scores


def plot_examples(
    path, images, masks, samples, *, model, label, n_examples=4, observed=None
) -> None:
    from matplotlib.figure import Figure

    columns = [
        ("Truth", images),
        ("Observed", np.where(masks, np.nan, images) if observed is None else observed),
        ("Linear interp.", samples["linear_interpolation"][:, 0]),
        ("GP posterior mean", samples["gp_posterior"].mean(axis=1)),
        (f"{label} sample", samples[model][:, 0]),
        (f"{label} mean", samples[model].mean(axis=1)),
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
    parser.add_argument("--denoiser", choices=tuple(DENOISER_NAMES), default="csdi")
    parser.add_argument(
        "--residual",
        action="store_true",
        help="diffuse the hidden residual around linear interpolation",
    )
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--residual-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--channel-multipliers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="U-Net level widths as multiples of --hidden-channels",
    )
    parser.add_argument("--blocks-per-level", type=int, default=1)
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
    name = args.denoiser + ("_residual" if args.residual else "")
    label = DENOISER_NAMES[args.denoiser] + (" residual" if args.residual else "")
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
    test, test_mask = splits["test"]
    logging.info(
        "%d train / %d validation / %d test images of %dx%d, %.0f%% hidden",
        len(splits["train"][0]),
        len(splits["validation"][0]),
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

    # The denoiser sees (images, rows, columns * planes); planes are the
    # residual and its interpolant in residual mode, otherwise the pixel itself.
    planes = 1
    data = dict(splits)
    if args.residual:
        planes = 2
        for split, (images, masks) in splits.items():
            if split == "test":
                baseline = samples["linear_interpolation"][:, 0]
            else:
                baseline = linear_interpolation(images, masks)[:, 0]
            data[split] = residual_layout(images, masks, baseline)
    (train, train_mask), (validation, validation_mask), (inputs, hidden) = (
        data[split] for split in ("train", "validation", "test")
    )

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
    if args.denoiser == "csdi":
        denoiser = CSDIDenoiser(
            n_channels=planes * args.size,
            hidden_channels=args.hidden_channels,
            residual_layers=args.residual_layers,
            attention_heads=args.attention_heads,
            time_embedding_dim=64,
        )
    else:
        denoiser = UNet2DDenoiser(
            planes,
            hidden_channels=args.hidden_channels,
            channel_multipliers=tuple(args.channel_multipliers),
            blocks_per_level=args.blocks_per_level,
        )
    model = ImputationModel(denoiser)
    parameters = sum(p.numel() for p in model.parameters())
    logging.info("%s with %d parameters", label, parameters)
    started = time.perf_counter()
    train_densifier(
        model,
        train,
        validation,
        config=config,
        dt=1.0,
        seed=args.seed,
        checkpoint=args.output / f"{name}.pt",
        train_mask=train_mask,
        validation_mask=validation_mask,
    )
    trained = time.perf_counter()
    draws = impute(
        inputs,
        DDPMSampler(model, replace(config, batch_size=args.sample_batch_size)),
        factor=1,
        mask=hidden,
        n_samples=args.n_samples,
        dt=1.0,
        seed=int(np.random.default_rng(streams[4]).integers(2**63)),
    )
    samples[name] = from_residual_layout(draws) if args.residual else draws
    sampled = time.perf_counter()

    scores = {key: score(draws, test, test_mask) for key, draws in samples.items()}
    scores["gp_posterior"]["crps_exact"] = gp_posterior_crps(
        test, test_mask, covariance
    )
    results = {
        "model": name,
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
        **{key: draws[:8] for key, draws in samples.items()},
    )
    logging.info("%s", json.dumps(scores, indent=2))

    if args.no_plot:
        return
    plot_examples(
        args.output / "examples.png",
        test,
        test_mask,
        samples,
        model=name,
        label=label,
        observed=test[:, ::2, ::2] if args.mask == "upsample" else None,
    )
    from plot_forecast_table import render_table

    best = scores["gp_posterior"]["crps"]
    rows = ("gp_posterior", name, "linear_interpolation")
    hidden_description = {
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
            label + (" ×2\ndensified" if args.mask != "random" else ""),
            "Linear interpolation",
        ],
        [
            (
                KERNEL_NAMES[args.kernel],
                [
                    (
                        scores[key]["crps"],
                        None
                        if key == "gp_posterior"
                        else scores[key]["crps"] / best - 1,
                    )
                    for key in rows
                ],
            )
        ],
        candidates=(1,),
        heading="Imputer",
        title=f"{label} image imputation",
        subtitle=(
            f"{KERNEL_NAMES[args.kernel]} Gaussian random field, lengthscale "
            f"{args.lengthscale:g} px of the {observed_size}×{observed_size} "
            f"grid, variance {args.variance:g}\n"
            f"{len(test)} test images of {args.size}×{args.size} pixels; "
            f"{hidden_description}"
        ),
        output=args.output / "table.png",
    )


if __name__ == "__main__":
    main()
