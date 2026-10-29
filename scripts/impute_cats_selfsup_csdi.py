"""Self-supervised CSDI super-resolution: train on low-resolution cats only.

    uv run --extra research --with matplotlib python \
        scripts/impute_cats_selfsup_csdi.py --output outputs/cats_selfsup --device cuda

The model never sees a high-resolution image. Each 65x65 photo from
`prepare_cats.py` is reduced to its low-resolution grid (every other row and
column, 33x33), and only those images are used for training: CSDI learns to
restore a 33x33 image from every other pixel of it (17x17 -> 33x33). Assuming
scale invariance, the statistics of how detail fills in between pixels are
the same one level up, so the same model then turns 33x33 test images into
65x65 ones.

CSDI's learned column embedding fixes the image width, so the 65x65 output is
filled as four overlapping 33x33 tiles whose corners start at even pixels.
Each tile keeps its own low-resolution pixels fixed and also conditions on
pixels filled by earlier tiles, so seams stay consistent.

The held-out 65x65 photos are used only for scoring, with the metrics of
`impute_cats_csdi.py`. Pass that script's results.json with `--supervised`
to add the supervised model, which trains on high-resolution images, as an
upper bound. The same-scale task (17x17 -> 33x33 on test images) is also
scored, to separate how well CSDI learned from what transfer across scales
costs.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from impute_cats_csdi import (
    grid_mask,
    interpolate,
    plot_examples,
    score_methods,
    scores_table,
    to_layout,
    to_pixels,
)

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier


def tile_starts(size: int, tile: int) -> list[int]:
    """Tile corners that share their edge rows, all at even pixels."""
    starts = list(range(0, size - tile + 1, tile - 1))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def upscale(low, sampler, *, colours, n_samples, seed) -> np.ndarray:
    """Double (images, n, n * colours) layouts to (images, samples, 2n - 1, ...).

    Low-resolution pixels land on even rows and columns. Tiles of the trained
    size n are filled in turn; pixels filled by earlier tiles count as
    observed, so each hidden pixel is drawn once per sample.
    """
    n_images, tile, _ = low.shape
    size = 2 * tile - 1
    batch = n_images * n_samples
    canvas = np.zeros((batch, size, size, colours))
    canvas[:, ::2, ::2] = np.repeat(
        low.reshape(n_images, tile, tile, colours), n_samples, axis=0
    )
    filled = np.zeros((batch, size, size), dtype=bool)
    filled[:, ::2, ::2] = True
    starts = tile_starts(size, tile)
    for index, (row, column) in enumerate(
        (row, column) for row in starts for column in starts
    ):
        window = (slice(None), slice(row, row + tile), slice(column, column + tile))
        hidden = np.repeat(~filled[window], colours, axis=2)
        draws = impute(
            canvas[window].reshape(batch, tile, tile * colours),
            sampler,
            factor=1,
            mask=hidden,
            n_samples=1,
            dt=1.0,
            seed=seed + index,
        )[:, 0]
        canvas[window] = draws.reshape(batch, tile, tile, colours)
        filled[window] = True
    return canvas.reshape(n_images, n_samples, size, size * colours)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/cats65.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--supervised", type=Path, help="results.json from impute_cats_csdi.py"
    )
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
    _, size, _, colours = test.shape
    if size % 4 != 1:
        parser.error("Image size must be 4k + 1 so both scales keep their edges")
    # Everything the model trains on is low resolution.
    train, validation = train[:, ::2, ::2], validation[:, ::2, ::2]
    low_test = test[:, ::2, ::2]
    if not args.no_flip:
        train = np.concatenate((train, train[:, :, ::-1]))
    low_size = low_test.shape[1]
    mask = grid_mask(low_size, colours, 2)
    logging.info(
        "Training on %d low-res %dx%d images (%d validation) to fill %dx%d -> "
        "%dx%d; testing %dx%d -> %dx%d on %d images; %s",
        len(train),
        low_size,
        low_size,
        len(validation),
        len(range(0, low_size, 2)),
        len(range(0, low_size, 2)),
        low_size,
        low_size,
        low_size,
        low_size,
        size,
        size,
        len(test),
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
            n_channels=low_size * colours,
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
    sampler = DDPMSampler(model, replace(config, batch_size=args.sample_batch_size))

    low_target = to_layout(low_test)
    low_mask = np.broadcast_to(mask, low_target.shape)
    same_scale = impute(
        low_target,
        sampler,
        factor=1,
        mask=low_mask,
        n_samples=args.n_samples,
        dt=1.0,
        seed=args.seed,
    )
    same_scale_scores = {
        "csdi_crps": crps(same_scale, low_target, mask=low_mask) / 2,
        "bicubic_crps": crps(
            interpolate(low_test, 2, "cubic"), low_target, mask=low_mask
        )
        / 2,
    }
    logging.info("Same-scale (training task) scores: %s", same_scale_scores)

    draws = upscale(
        low_target,
        sampler,
        colours=colours,
        n_samples=args.n_samples,
        seed=args.seed + 1,
    )
    sampled = time.perf_counter()
    target = to_layout(test)
    test_mask = np.broadcast_to(grid_mask(size, colours, 2), target.shape)
    estimates = {
        "bilinear": interpolate(test, 2, "linear"),
        "bicubic": interpolate(test, 2, "cubic"),
    }
    scores = score_methods(estimates, draws, target, test_mask, colours)
    labels = {
        "bilinear": "Bilinear",
        "bicubic": "Bicubic",
        "csdi_mean": f"CSDI self-supervised\n(mean of {args.n_samples})",
        "csdi_sample": "CSDI self-supervised\n(single sample)",
    }
    if args.supervised is not None:
        supervised = json.loads(args.supervised.read_text())
        if supervised["args"]["factor"] != "2" or supervised["args"].get(
            "test_images"
        ) != str(args.test_images):
            parser.error("--supervised must be a x2 run on the same test images")
        scores["supervised_mean"] = supervised["scores"]["csdi_mean"]
        scores["supervised_sample"] = supervised["scores"]["csdi_sample"]
        labels["supervised_mean"] = "CSDI supervised\n(mean, high-res training)"
        labels["supervised_sample"] = "CSDI supervised\n(single sample)"
    results = {
        "scores": scores,
        "same_scale_scores": same_scale_scores,
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
    )
    logging.info("%s", json.dumps(scores, indent=2))

    if args.no_plot:
        return
    truth = test / 255
    low_display = np.repeat(np.repeat(truth[:, ::2, ::2], 2, 1), 2, 2)
    plot_examples(
        args.output / "examples.png",
        truth,
        low_display[:, :size, :size],
        to_pixels(estimates["bicubic"][:, 0], colours),
        to_pixels(draws, colours),
    )
    scores_table(
        args.output / "table.png",
        scores,
        labels,
        title="Self-supervised CSDI super-resolution on cat photos",
        subtitle=(
            f"{data_config['dataset']}: trained on {low_size}×{low_size} images "
            f"only; tested {low_size}×{low_size} → {size}×{size} RGB on "
            f"{len(test)} images\nCRPS on hidden pixels (= MAE for "
            "deterministic methods); PSNR and SSIM on whole images"
        ),
    )


if __name__ == "__main__":
    main()
