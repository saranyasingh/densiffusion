"""Self-trained CSDI super-resolution of MNIST digits from blurry data only.

    uv run --extra research --with matplotlib python \
        scripts/bootstrap_digits_csdi.py --output outputs/digits --device cuda

Only low-resolution digits are used for training: each 29x29 digit from
`prepare_digits.py` is reduced to every other row and column (15x15), and the
29x29 versions of training digits are never read (except by `--oracle`).

Round 1 learns imputation from the blurry data. CSDI is trained to restore a
15x15 digit from every other pixel of it (8x8 -> 15x15). MNIST digits are
centred, so a model that only sees whole digits learns that borders are
blank and erases strokes cut by tile edges in the next step; `--shift-copies`
adds copies of the blurry digits translated by up to half their size, so
strokes appear at every position, including cut off at the border.

Round 1 then imputes the data. Assuming scale invariance, the same model
doubles every blurry training and validation digit to 29x29, in overlapping
15x15 tiles (see `impute_cats_selfsup_csdi.upscale`). The original
low-resolution pixels are kept exactly; only the pixels between them are
imputed. `--imputations k` draws k versions of each digit.

Round 2 trains on the imputed data. A new CSDI learns 15x15 -> 29x29 directly
from the imputed digits, with the imputed pixels as targets, and is applied
to whole test images without tiles to produce the clear digits.

Test digits are scored against their true 29x29 versions with the metrics of
`impute_cats_csdi.py`, next to bilinear and bicubic interpolation and the
round 1 model applied to test digits directly. `--oracle` adds an upper bound
trained on the true 29x29 training digits. Checkpoints and the imputed data
are saved in --output and reused, so a rerun continues where it stopped.
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
    score_methods,
    scores_table,
    to_layout,
    to_pixels,
)
from impute_cats_selfsup_csdi import upscale

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier


def fit(name, train, validation, mask, *, args, config) -> ImputationModel:
    """Train CSDI to fill the masked pixels, or load the checkpoint if present."""
    torch.manual_seed(args.seed)
    model = ImputationModel(
        CSDIDenoiser(
            n_channels=train.shape[-1],
            hidden_channels=args.hidden_channels,
            residual_layers=args.residual_layers,
            attention_heads=args.attention_heads,
        )
    )
    checkpoint = args.output / f"{name}.pt"
    if checkpoint.exists():
        logging.info("Loading %s", checkpoint)
        state = torch.load(checkpoint, map_location=args.device)
        model.to(args.device).load_state_dict(state["model_state_dict"])
        model.eval()
        return model
    size = train.shape[1]
    logging.info("Training %s on %d %dx%d digits", name, len(train), size, size)
    train_densifier(
        model,
        train,
        validation,
        config=config,
        dt=1.0,
        seed=args.seed,
        checkpoint=checkpoint,
        train_mask=np.broadcast_to(mask, train.shape),
        validation_mask=np.broadcast_to(mask, validation.shape),
    )
    return model


def plot_rows(path, rows, n_examples) -> None:
    """One row per method, one column per digit, all in [0, 1] gray."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(1.1 * n_examples + 1.6, 1.15 * len(rows)))
    axes = figure.subplots(len(rows), n_examples, squeeze=False)
    for row, (label, images) in enumerate(rows):
        for column in range(n_examples):
            axis = axes[row, column]
            axis.imshow(images[column], cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def shifted(images: np.ndarray, copies: int, rng) -> np.ndarray:
    """Images plus copies translated by up to half their size, zero filled."""
    n_images, size = images.shape[:2]
    reach = size // 2
    padded = np.pad(images, ((0, 0), (reach, reach), (reach, reach), (0, 0)))
    output = [images]
    for _ in range(copies):
        rows, columns = rng.integers(0, 2 * reach + 1, size=(2, n_images))
        output.append(
            np.stack(
                [
                    image[row : row + size, column : column + size]
                    for image, row, column in zip(padded, rows, columns, strict=True)
                ]
            )
        )
    return np.concatenate(output)


def enlarge(images: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour display of low-resolution digits at full size."""
    return np.repeat(np.repeat(images, 2, 1), 2, 2)[:, :size, :size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/mnist29.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=20000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--imputations", type=int, default=1)
    parser.add_argument(
        "--shift-copies",
        type=int,
        default=4,
        help="Translated copies of each blurry digit for round 1",
    )
    parser.add_argument(
        "--oracle", action="store_true", help="Also train on true 29x29 digits"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--sample-batch-size", type=int, default=256)
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
        train = dataset["train"][: args.train_images]
        validation = dataset["validation"]
        test = dataset["test"][: args.test_images]
        data_config = json.loads(str(dataset["config"]))
    size = test.shape[1]
    if size % 4 != 1:
        parser.error("Digit size must be 4k + 1 so both scales keep their edges")
    # The blurry dataset: everything training sees from here on.
    low_train, low_validation = train[:, ::2, ::2], validation[:, ::2, ::2]
    low_size = low_train.shape[1]
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
    sampling = replace(config, batch_size=args.sample_batch_size)
    timings = {}
    started = time.perf_counter()

    round1 = fit(
        "round1",
        to_layout(
            shifted(low_train, args.shift_copies, np.random.default_rng(args.seed))
        ),
        to_layout(low_validation),
        grid_mask(low_size, 1, 2),
        args=args,
        config=config,
    )
    round1_sampler = DDPMSampler(round1, sampling)
    timings["round1_train_seconds"] = time.perf_counter() - started

    imputed_path = args.output / "imputed.npz"
    if imputed_path.exists():
        logging.info("Loading %s", imputed_path)
        with np.load(imputed_path) as imputed:
            imputed_train, imputed_validation = imputed["train"], imputed["validation"]
    else:
        logging.info(
            "Imputing %d training digits %dx%d -> %dx%d (%d each)",
            len(low_train),
            low_size,
            low_size,
            size,
            size,
            args.imputations,
        )
        start = time.perf_counter()
        imputed_train = upscale(
            to_layout(low_train),
            round1_sampler,
            colours=1,
            n_samples=args.imputations,
            seed=args.seed + 1,
        ).reshape(-1, size, size)
        imputed_validation = upscale(
            to_layout(low_validation),
            round1_sampler,
            colours=1,
            n_samples=1,
            seed=args.seed + 2,
        )[:, 0]
        timings["impute_seconds"] = time.perf_counter() - start
        np.savez_compressed(
            imputed_path,
            train=imputed_train.astype(np.float32),
            validation=imputed_validation.astype(np.float32),
        )

    start = time.perf_counter()
    mask = grid_mask(size, 1, 2)
    round2 = fit(
        "round2",
        imputed_train,
        imputed_validation,
        mask,
        args=args,
        config=config,
    )
    timings["round2_train_seconds"] = time.perf_counter() - start
    oracle = None
    if args.oracle:
        start = time.perf_counter()
        oracle = fit(
            "oracle",
            to_layout(train),
            to_layout(validation),
            mask,
            args=args,
            config=config,
        )
        timings["oracle_train_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    target = to_layout(test)
    test_mask = np.broadcast_to(mask, target.shape)
    estimates = {
        "bilinear": interpolate(test, 2, "linear"),
        "bicubic": interpolate(test, 2, "cubic"),
    }
    draws = {
        "round1": upscale(
            to_layout(test[:, ::2, ::2]),
            round1_sampler,
            colours=1,
            n_samples=args.n_samples,
            seed=args.seed + 3,
        ),
        "round2": impute(
            target,
            DDPMSampler(round2, sampling),
            factor=1,
            mask=test_mask,
            n_samples=args.n_samples,
            dt=1.0,
            seed=args.seed + 4,
        ),
    }
    if oracle is not None:
        draws["oracle"] = impute(
            target,
            DDPMSampler(oracle, sampling),
            factor=1,
            mask=test_mask,
            n_samples=args.n_samples,
            dt=1.0,
            seed=args.seed + 4,
        )
    timings["test_sample_seconds"] = time.perf_counter() - start

    scores = {}
    for name, values in draws.items():
        method = score_methods(estimates, values, target, test_mask, 1)
        scores.update({key: method.pop(key) for key in estimates})
        scores[f"{name}_mean"] = method["csdi_mean"]
        scores[f"{name}_sample"] = method["csdi_sample"]
    labels = {
        "bilinear": "Bilinear",
        "bicubic": "Bicubic",
        "round1_mean": f"Round 1, tiles\n(mean of {args.n_samples})",
        "round1_sample": "Round 1, tiles\n(single sample)",
        "round2_mean": f"Round 2, self-trained\n(mean of {args.n_samples})",
        "round2_sample": "Round 2, self-trained\n(single sample)",
    }
    if oracle is not None:
        labels["oracle_mean"] = f"Oracle, clear digits\n(mean of {args.n_samples})"
        labels["oracle_sample"] = "Oracle, clear digits\n(single sample)"
    results = {
        "scores": scores,
        "timings": timings,
        "data": data_config,
        "args": {key: str(value) for key, value in vars(args).items()},
        "training": asdict(config),
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(
        args.output / "samples.npz",
        truth=test[:32],
        **{
            name: (to_pixels(values[:32], 1)[..., 0] * 255).round().astype(np.uint8)
            for name, values in draws.items()
        },
    )
    logging.info("%s", json.dumps(scores, indent=2))

    if args.no_plot:
        return
    n_examples = min(10, len(test))
    truth = test[:n_examples, ..., 0] / 255
    rows = [
        ("Clear truth", truth),
        (f"Blurry input\n{low_size}×{low_size}", enlarge(truth[:, ::2, ::2], size)),
        ("Bicubic", to_pixels(estimates["bicubic"][:n_examples, 0], 1)[..., 0]),
    ]
    for name, label in (
        ("round1", "Round 1"),
        ("round2", "Round 2"),
        ("oracle", "Oracle"),
    ):
        if name in draws:
            pixels = to_pixels(draws[name][:n_examples], 1)[..., 0]
            rows += [
                (f"{label}\nsample", pixels[:, 0]),
                (f"{label}\nmean", pixels.mean(1)),
            ]
    plot_rows(args.output / "examples.png", rows, n_examples)
    plot_rows(
        args.output / "imputed_train.png",
        [
            (
                "Blurry training\ndigit",
                enlarge(low_train[:n_examples, ..., 0] / 255, size),
            ),
            (
                "Round 1 imputation\n(round 2 trains on)",
                to_pixels(
                    imputed_train.reshape(len(low_train), -1, size, size)[
                        :n_examples, 0
                    ],
                    1,
                )[..., 0],
            ),
        ],
        n_examples,
    )
    scores_table(
        args.output / "table.png",
        scores,
        labels,
        title="Self-trained CSDI super-resolution on MNIST digits",
        subtitle=(
            f"Trained on {low_size}×{low_size} digits only ({len(low_train)} images); "
            f"tested {low_size}×{low_size} → {size}×{size} on {len(test)} digits\n"
            "CRPS on hidden pixels (= MAE for deterministic methods); "
            "PSNR and SSIM on whole images"
        ),
    )


if __name__ == "__main__":
    main()
