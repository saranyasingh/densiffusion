"""Super-resolve cat photos with CSDI and compare against classical upscaling.

    uv run --extra research --with matplotlib python scripts/impute_cats_csdi.py \
        --data data/afhq_cats65.npz --output outputs/cats_csdi --device cuda

Reads the uint8 RGB crops written by `prepare_cats.py`. `--factor 2` keeps
every other row and column (33x33 -> 65x65 super-resolution, 3 of 4 pixels
hidden); `--factor 4` keeps every fourth. Sizes of the form factor * k + 1
keep the last row and column observed, so no method extrapolates. This is
the 2D analogue of
2*dt -> dt densification: low-resolution pixels are a subsample of the
high-resolution image and stay fixed during sampling.

CSDI is unchanged. Pixel values are scaled to [-1, 1] and each image becomes
(rows, columns * 3) with the RGB values of a pixel in adjacent channels, so
CSDI's temporal attention runs down columns and its feature attention along
rows across all colours. Training and sampling reuse `train_densifier` and
`DDPMSampler`. Real photos have no exact posterior, so CSDI is compared with
bilinear and bicubic interpolation through the same observed pixels.

Scores use pixel values in [0, 1]. CRPS covers hidden values only; for the
deterministic baselines it equals the mean absolute error. PSNR and SSIM
(Gaussian window, sigma 1.5, averaged over colours) cover whole images and are
reported for the mean of the CSDI draws and for a single draw: the mean
scores higher but is blurrier, while a single draw keeps realistic texture.
Existing checkpoints in OUTPUT are reused, so a rerun skips training.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier


def to_layout(images: np.ndarray) -> np.ndarray:
    """uint8 (images, size, size, colours) -> float (images, size, size * colours)."""
    n_images, size, _, colours = images.shape
    return (images / 127.5 - 1).reshape(n_images, size, size * colours)


def to_pixels(values: np.ndarray, colours: int) -> np.ndarray:
    """Inverse of to_layout for any leading shape, clipped to [0, 1]."""
    *leading, size, width = values.shape
    pixels = values.reshape(*leading, size, width // colours, colours)
    return np.clip((pixels + 1) / 2, 0, 1)


def grid_mask(size: int, colours: int, factor: int) -> np.ndarray:
    """(size, size * colours) mask hiding all but every factor-th row and column."""
    mask = np.ones((size, size), dtype=bool)
    mask[::factor, ::factor] = False
    return np.repeat(mask, colours, axis=1)


def interpolate(images: np.ndarray, factor: int, method: str) -> np.ndarray:
    """Upscale the observed subgrid of uint8 images back to full size in [-1, 1]."""
    n_images, size, _, colours = images.shape
    known = np.arange(0, size, factor)
    query = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing="ij"), -1)
    output = np.empty((n_images, size, size, colours))
    for index, image in enumerate(images / 127.5 - 1):
        interpolator = RegularGridInterpolator(
            (known, known),
            image[::factor, ::factor],
            method=method,
            bounds_error=False,
            fill_value=None,
        )
        output[index] = interpolator(query)
    return np.clip(output, -1, 1).reshape(n_images, 1, size, size * colours)


def psnr(estimate: np.ndarray, truth: np.ndarray) -> float:
    mse = ((estimate - truth) ** 2).mean(axis=(1, 2, 3))
    return float((10 * np.log10(1 / np.maximum(mse, 1e-12))).mean())


def ssim(estimate: np.ndarray, truth: np.ndarray) -> float:
    def blur(values):
        return gaussian_filter(values, sigma=(0, 1.5, 1.5, 0))

    c1, c2 = 0.01**2, 0.03**2
    mean_x, mean_y = blur(estimate), blur(truth)
    var_x = blur(estimate**2) - mean_x**2
    var_y = blur(truth**2) - mean_y**2
    covariance = blur(estimate * truth) - mean_x * mean_y
    index = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(index.mean())


def plot_examples(path, truth, low, bicubic, draws, n_examples=4) -> None:
    from matplotlib.figure import Figure

    spread = draws.std(axis=1).mean(axis=-1)
    columns = [
        ("High-res truth", truth),
        ("Low-res input", low),
        ("Bicubic", bicubic),
        ("CSDI sample", draws[:, 0]),
        ("CSDI mean", draws.mean(axis=1)),
        ("CSDI uncertainty", spread),
    ]
    figure = Figure(figsize=(2 * len(columns), 2.1 * n_examples))
    axes = figure.subplots(n_examples, len(columns))
    for row in range(n_examples):
        for column, (title, images) in enumerate(columns):
            axis = axes[row, column]
            if images.ndim == 3:
                axis.imshow(images[row], cmap="magma", vmin=0, vmax=spread.max())
            else:
                axis.imshow(images[row], interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/afhq_cats65.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--no-flip", action="store_true", help="Skip mirrored copies")
    parser.add_argument("--train-images", type=int, help="Use only the first N")
    parser.add_argument("--test-images", type=int, help="Use only the first N")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.output / "training.log"),
            logging.StreamHandler(),
        ],
    )
    with np.load(args.data, allow_pickle=False) as dataset:
        train, validation, test = (
            dataset[split] for split in ("train", "validation", "test")
        )
        data_config = json.loads(str(dataset["config"]))
    train, test = train[: args.train_images], test[: args.test_images]
    if not args.no_flip:
        train = np.concatenate((train, train[:, :, ::-1]))
    _, size, _, colours = test.shape
    mask = grid_mask(size, colours, args.factor)
    logging.info(
        "%d train / %d validation / %d test images of %dx%d, x%d (%d%% hidden), %s",
        len(train),
        len(validation),
        len(test),
        size,
        size,
        args.factor,
        round(100 * mask.mean()),
        args.device,
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
    model = ImputationModel(
        CSDIDenoiser(
            n_channels=size * colours,
            hidden_channels=args.hidden_channels,
            residual_layers=args.residual_layers,
            attention_heads=args.attention_heads,
        )
    )
    parameters = sum(p.numel() for p in model.parameters())
    logging.info("CSDI with %d parameters", parameters)
    checkpoint = args.output / "csdi.pt"
    started = time.perf_counter()
    if checkpoint.exists():
        logging.info("Loading %s", checkpoint)
        state = torch.load(checkpoint, map_location=args.device)
        model.to(args.device).load_state_dict(state["model_state_dict"])
        model.eval()
    else:
        train_densifier(
            model,
            to_layout(train),
            to_layout(validation),
            config=config,
            dt=1.0,
            seed=args.seed,
            checkpoint=checkpoint,
            train_mask=np.broadcast_to(mask, (len(train), *mask.shape)),
            validation_mask=np.broadcast_to(mask, (len(validation), *mask.shape)),
        )
    trained = time.perf_counter()
    target = to_layout(test)
    test_mask = np.broadcast_to(mask, target.shape)
    draws = impute(
        target,
        DDPMSampler(model, replace(config, batch_size=args.sample_batch_size)),
        factor=1,
        mask=test_mask,
        n_samples=args.n_samples,
        dt=1.0,
        seed=args.seed,
    )
    sampled = time.perf_counter()

    truth = test / 255
    estimates = {
        "bilinear": interpolate(test, args.factor, "linear"),
        "bicubic": interpolate(test, args.factor, "cubic"),
    }
    scores = score_methods(estimates, draws, target, test_mask, colours)
    results = {
        "scores": scores,
        "parameters": parameters,
        "train_seconds": trained - started,
        "sample_seconds": sampled - trained,
        "data": data_config,
        "args": {key: str(value) for key, value in vars(args).items()},
        "training": asdict(config),
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(
        args.output / "samples.npz",
        truth=test[:16],
        csdi=(to_pixels(draws[:16], colours) * 255).round().astype(np.uint8),
        bicubic=(to_pixels(estimates["bicubic"][:16, 0], colours) * 255)
        .round()
        .astype(np.uint8),
    )
    logging.info("%s", json.dumps(scores, indent=2))

    if args.no_plot:
        return
    low = np.repeat(
        np.repeat(truth[:, :: args.factor, :: args.factor], args.factor, 1),
        args.factor,
        2,
    )
    plot_examples(
        args.output / "examples.png",
        truth,
        low[:, :size, :size],
        to_pixels(estimates["bicubic"][:, 0], colours),
        to_pixels(draws, colours),
    )
    low_size = len(range(0, size, args.factor))
    scores_table(
        args.output / "table.png",
        scores,
        {
            "bilinear": "Bilinear",
            "bicubic": "Bicubic",
            "csdi_mean": f"CSDI ×{args.factor}\n(mean of {args.n_samples})",
            "csdi_sample": f"CSDI ×{args.factor}\n(single sample)",
        },
        title="CSDI super-resolution on cat photos",
        subtitle=(
            f"{data_config['dataset']}: {len(test)} test images, "
            f"{low_size}×{low_size} → {size}×{size} RGB (×{args.factor})\n"
            "CRPS on hidden pixels (= MAE for deterministic methods); "
            "PSNR and SSIM on whole images"
        ),
    )


def score_methods(estimates, draws, target, mask, colours) -> dict:
    """Score deterministic estimates and CSDI draws in [0, 1] pixel units.

    estimates maps names to (images, 1, rows, columns * colours) values in
    [-1, 1]; draws holds (images, samples, ...) CSDI values; target is the
    true layout. CRPS covers masked values; PSNR and SSIM whole images.
    """
    truth = to_pixels(target, colours)
    scores = {}
    for name, values in estimates.items():
        pixels = to_pixels(values[:, 0], colours)
        scores[name] = {
            "crps": crps(values, target, mask=mask) / 2,
            "psnr": psnr(pixels, truth),
            "ssim": ssim(pixels, truth),
        }
    mean = to_pixels(draws.mean(axis=1), colours)
    single = to_pixels(draws[:, 0], colours)
    scores["csdi_mean"] = {
        "crps": crps(draws, target, mask=mask) / 2,
        "psnr": psnr(mean, truth),
        "ssim": ssim(mean, truth),
    }
    scores["csdi_sample"] = {"psnr": psnr(single, truth), "ssim": ssim(single, truth)}
    return scores


def scores_table(path, scores, labels, *, title, subtitle) -> None:
    """Draw CRPS, PSNR, and SSIM for the methods in labels, bolding each best."""
    from plot_forecast_table import draw_table

    names = list(labels)
    metrics = [
        ("crps", "{:.4f}", min),
        ("psnr", "{:.2f}", max),
        ("ssim", "{:.3f}", max),
    ]
    bold = set()
    for column, (metric, _, best) in enumerate(metrics, start=1):
        ranked = [name for name in names if metric in scores[name]]
        bold.add((names.index(best(ranked, key=lambda n: scores[n][metric])), column))
    draw_table(
        [
            [labels[name]]
            + [
                form.format(scores[name][metric]) if metric in scores[name] else "—"
                for metric, form, _ in metrics
            ]
            for name in names
        ],
        ["Method", "CRPS ↓", "PSNR ↑", "SSIM ↑"],
        bold=bold,
        widths=[1, 1, 1],
        title=title,
        subtitle=subtitle,
        output=path,
    )


if __name__ == "__main__":
    main()
