"""Self-trained CSDI super-resolution from blurry images only.

    uv run --extra research --with matplotlib python scripts/bootstrap_csdi.py \
        --data data/mnist29.npz --output outputs/digits --device cuda
    uv run --extra research --with matplotlib python scripts/bootstrap_csdi.py \
        --data data/div2k65.npz --shift-copies 0 --output outputs/div2k --device cuda

PSNR-oriented options (opt-in; defaults match earlier runs):

    --augment --residual --imputations 3 --imputation-aggregate mean \\
    --n-samples 30 --aggregate median --self-ensemble --blend 0.5 --refine-epochs 20

Reads the uint8 (images, size, size, colours) splits written by
`prepare_digits.py` (MNIST, 29x29 grey) or `prepare_div2k.py` (DIV2K crops,
65x65 RGB). Only low-resolution images are used for training: each image is
reduced to every other row and column, and the full-size training images are
never read (except by `--oracle`).

Round 1 learns imputation from the blurry data. CSDI is trained to restore a
low-resolution image from every other pixel of it. `--shift-copies` adds
translated copies; `--augment` adds flips and 90-degree rotations.

Round 1 then imputes the training set to full size in overlapping tiles.
`--imputations k` draws k versions; `--imputation-aggregate mean|median` collapses
them into one Round 2 target per image (better for PSNR than stacking noisy draws).

Round 2 / `--oracle` can use `--residual` (diffuse the residual around bilinear).
At test time, draws are aggregated (`--aggregate`), optionally self-ensembled with
a horizontal flip, observed pixels are clamped, optionally blended with bilinear
(`--blend`), and optionally refined by a tiny L2 CNN (`--refine-epochs`).
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
    psnr,
    score_methods,
    scores_table,
    ssim,
    to_layout,
    to_pixels,
)
from impute_cats_selfsup_csdi import upscale
from psnr_hacks import (
    aggregate,
    apply_refine,
    blend,
    clamp_layout,
    clamp_observed,
    flip_layout,
    from_residual_pair,
    geometric_augment,
    residual_mask,
    residual_pair,
    train_refine,
)

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.metrics import crps
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
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        model.to(args.device).load_state_dict(state["model_state_dict"])
        model.eval()
        return model
    size = train.shape[1]
    logging.info("Training %s on %d %dx%d images", name, len(train), size, size)
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
    """One row per method, one column per image; (images, size, size, colours)."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(1.1 * n_examples + 1.6, 1.15 * len(rows)))
    axes = figure.subplots(len(rows), n_examples, squeeze=False)
    for row, (label, images) in enumerate(rows):
        for column in range(n_examples):
            axis = axes[row, column]
            image = np.clip(images[column], 0, 1)
            if image.shape[-1] == 1:
                axis.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(image, interpolation="nearest")
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
    """Nearest-neighbour display of low-resolution images at full size."""
    return np.repeat(np.repeat(images, 2, 1), 2, 2)[:, :size, :size]


def maybe_augment(images: np.ndarray, enabled: bool, rng) -> np.ndarray:
    return geometric_augment(images, rng) if enabled else images


def score_estimate(
    name, estimate_layout, draws, target, mask, colours, scores, labels, label
):
    """PSNR/SSIM of a deterministic estimate; CRPS from draws if provided."""
    truth = to_pixels(target, colours)
    pixels = to_pixels(estimate_layout, colours)
    entry = {
        "psnr": psnr(pixels, truth),
        "ssim": ssim(pixels, truth),
        "crps": crps(
            draws if draws is not None else estimate_layout[:, None],
            target,
            mask=mask,
        )
        / 2,
    }
    scores[name] = entry
    labels[name] = label


def method_label(name: str) -> str:
    return {"round1": "Round 1", "round2": "Round 2", "oracle": "Oracle"}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/mnist29.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=20000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--imputations", type=int, default=1)
    parser.add_argument(
        "--imputation-aggregate",
        choices=("none", "mean", "median"),
        default="none",
        help="Collapse Round 1 imputations into one Round 2 target per image",
    )
    parser.add_argument(
        "--shift-copies",
        type=int,
        default=4,
        help="Translated copies of each blurry image for round 1",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Add flips and 90-degree rotations to Round 1/2/oracle training",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Round 2/oracle: diffuse residual around bilinear (PSNR-oriented)",
    )
    parser.add_argument(
        "--aggregate",
        choices=("mean", "median"),
        default="mean",
        help="Collapse test draws for the PSNR estimate",
    )
    parser.add_argument(
        "--self-ensemble",
        action="store_true",
        help="Average test draws with a horizontally flipped pass (Round 2/oracle)",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="alpha * model + (1-alpha) * bilinear for the PSNR estimate",
    )
    parser.add_argument(
        "--no-clamp-observed",
        action="store_true",
        help="Do not force observed grid pixels to the low-res input",
    )
    parser.add_argument(
        "--refine-epochs",
        type=int,
        default=0,
        help="Train a tiny L2 CNN on (bilinear, estimate) -> refined image",
    )
    parser.add_argument(
        "--oracle", action="store_true", help="Also train on true full-size images"
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
    if not 0 <= args.blend <= 1:
        parser.error("--blend must be in [0, 1]")
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
    if args.imputations > 1 and args.imputation_aggregate == "none":
        logging.warning(
            "imputations=%d with imputation-aggregate=none stacks noisy draws; "
            "mean/median is usually better for PSNR",
            args.imputations,
        )
    with np.load(args.data, allow_pickle=False) as dataset:
        train = dataset["train"][: args.train_images]
        validation = dataset["validation"]
        test = dataset["test"][: args.test_images]
        data_config = json.loads(str(dataset["config"]))
    size, colours = test.shape[1], test.shape[-1]
    if size % 4 != 1:
        parser.error("Image size must be 4k + 1 so both scales keep their edges")
    rng = np.random.default_rng(args.seed)
    # The blurry dataset: everything training sees from here on.
    low_train, low_validation = train[:, ::2, ::2], validation[:, ::2, ::2]
    low_test = test[:, ::2, ::2]
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

    round1_train = maybe_augment(
        shifted(low_train, args.shift_copies, rng), args.augment, rng
    )
    round1 = fit(
        "round1",
        to_layout(round1_train),
        to_layout(low_validation),
        grid_mask(low_size, colours, 2),
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
            "Imputing %d training images %dx%d -> %dx%d (%d each, aggregate=%s)",
            len(low_train),
            low_size,
            low_size,
            size,
            size,
            args.imputations,
            args.imputation_aggregate,
        )
        start = time.perf_counter()
        imputed_draws = upscale(
            to_layout(low_train),
            round1_sampler,
            colours=colours,
            n_samples=args.imputations,
            seed=args.seed + 1,
        )
        if args.imputation_aggregate == "none":
            imputed_train = imputed_draws.reshape(-1, size, size * colours)
        else:
            imputed_train = aggregate(imputed_draws, args.imputation_aggregate)
        imputed_validation = upscale(
            to_layout(low_validation),
            round1_sampler,
            colours=colours,
            n_samples=1,
            seed=args.seed + 2,
        )[:, 0]
        timings["impute_seconds"] = time.perf_counter() - start
        np.savez_compressed(
            imputed_path,
            train=imputed_train.astype(np.float32),
            validation=imputed_validation.astype(np.float32),
        )

    spatial_mask = grid_mask(size, colours, 2)
    planes = 2 * colours if args.residual else colours

    def baseline_from_low(low_uint8: np.ndarray) -> np.ndarray:
        canvas = np.zeros((len(low_uint8), size, size, colours), dtype=np.uint8)
        canvas[:, ::2, ::2] = low_uint8
        return interpolate(canvas, 2, "linear")[:, 0]

    if args.residual:
        if len(imputed_train) == len(low_train):
            low_for_imputed = low_train
        else:
            low_for_imputed = np.repeat(
                low_train, len(imputed_train) // len(low_train), 0
            )
        round2_train = residual_pair(imputed_train, baseline_from_low(low_for_imputed))
        round2_val = residual_pair(
            imputed_validation, baseline_from_low(low_validation)
        )
        round2_mask = residual_mask(spatial_mask)
    else:
        round2_train = imputed_train
        round2_val = imputed_validation
        round2_mask = spatial_mask
    if args.augment:
        round2_train = np.concatenate((round2_train, flip_layout(round2_train, planes)))
    start = time.perf_counter()
    round2 = fit(
        "round2" + ("_residual" if args.residual else ""),
        round2_train,
        round2_val,
        round2_mask,
        args=args,
        config=config,
    )
    timings["round2_train_seconds"] = time.perf_counter() - start

    oracle = None
    if args.oracle:
        start = time.perf_counter()
        if args.residual:
            oracle_train = residual_pair(
                to_layout(train), baseline_from_low(train[:, ::2, ::2])
            )
            oracle_val = residual_pair(
                to_layout(validation), baseline_from_low(validation[:, ::2, ::2])
            )
            oracle_mask = residual_mask(spatial_mask)
            if args.augment:
                oracle_train = np.concatenate(
                    (oracle_train, flip_layout(oracle_train, planes))
                )
        else:
            oracle_train = to_layout(maybe_augment(train, args.augment, rng))
            oracle_val = to_layout(validation)
            oracle_mask = spatial_mask
        oracle = fit(
            "oracle" + ("_residual" if args.residual else ""),
            oracle_train,
            oracle_val,
            oracle_mask,
            args=args,
            config=config,
        )
        timings["oracle_train_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    target = to_layout(test)
    test_mask = np.broadcast_to(spatial_mask, target.shape)
    bilinear = interpolate(test, 2, "linear")
    bicubic = interpolate(test, 2, "cubic")
    estimates = {"bilinear": bilinear, "bicubic": bicubic}
    low_layout = to_layout(low_test)

    round1_draws = upscale(
        low_layout,
        round1_sampler,
        colours=colours,
        n_samples=args.n_samples,
        seed=args.seed + 3,
    )

    def run_full(model, seed):
        sampler = DDPMSampler(model, sampling)
        if args.residual:
            observed = residual_pair(target, baseline_from_low(low_test))
            mask = np.broadcast_to(residual_mask(spatial_mask), observed.shape)
            pack = planes
        else:
            observed, mask, pack = target, test_mask, colours
        orientations = [False, True] if args.self_ensemble else [False]
        parts = []
        for index, flip in enumerate(orientations):
            obs = flip_layout(observed, pack) if flip else observed
            m = (
                flip_layout(mask.astype(np.float32), pack).astype(bool)
                if flip
                else mask
            )
            sampled = impute(
                obs,
                sampler,
                factor=1,
                mask=m,
                n_samples=args.n_samples,
                dt=1.0,
                seed=seed + index,
            )
            if args.residual:
                sampled = from_residual_pair(sampled)
            if flip:
                sampled = flip_layout(sampled, colours)
            parts.append(sampled)
        return np.concatenate(parts, axis=1)

    draws = {"round1": round1_draws, "round2": run_full(round2, args.seed + 4)}
    if oracle is not None:
        draws["oracle"] = run_full(oracle, args.seed + 5)
    timings["test_sample_seconds"] = time.perf_counter() - start

    def psnr_estimate(values: np.ndarray) -> np.ndarray:
        estimate = aggregate(values, args.aggregate)
        if not args.no_clamp_observed:
            estimate = clamp_layout(estimate, low_layout, colours)
        if args.blend < 1:
            estimate = blend(estimate, bilinear[:, 0], args.blend)
        return estimate

    scores = {}
    labels = {
        "bilinear": "Bilinear",
        "bicubic": "Bicubic",
    }
    # Baselines via score_methods once
    base = score_methods(estimates, draws["round2"], target, test_mask, colours)
    scores["bilinear"] = base["bilinear"]
    scores["bicubic"] = base["bicubic"]

    processed = {}
    for name, values in draws.items():
        method = score_methods({}, values, target, test_mask, colours)
        estimate = psnr_estimate(values)
        processed[name] = estimate
        # PSNR/SSIM from the processed estimate; CRPS from the raw draws.
        est_pixels = to_pixels(estimate, colours)
        truth_pixels = to_pixels(target, colours)
        scores[f"{name}_mean"] = {
            "crps": method["csdi_mean"]["crps"],
            "psnr": psnr(est_pixels, truth_pixels),
            "ssim": ssim(est_pixels, truth_pixels),
        }
        scores[f"{name}_sample"] = method["csdi_sample"]
        hack = args.aggregate
        if args.blend < 1:
            hack += f", blend {args.blend}"
        if args.self_ensemble:
            hack += ", ensemble"
        labels[f"{name}_mean"] = f"{method_label(name)}\n({hack} of {values.shape[1]})"
        labels[f"{name}_sample"] = f"{method_label(name)}\n(single sample)"

    if args.refine_epochs > 0:
        start = time.perf_counter()
        refine_sampler = DDPMSampler(round2, sampling)
        canvas_f = np.zeros((len(low_train), size, size, colours))
        canvas_f[:, ::2, ::2] = low_train / 127.5 - 1
        conditioned = canvas_f.reshape(len(low_train), size, size * colours)
        if args.residual:
            observed = residual_pair(conditioned, baseline_from_low(low_train))
            mask = np.broadcast_to(residual_mask(spatial_mask), observed.shape)
            refine_draws = impute(
                observed,
                refine_sampler,
                factor=1,
                mask=mask,
                n_samples=max(4, args.n_samples // 2),
                dt=1.0,
                seed=args.seed + 7,
            )
            refine_est = from_residual_pair(aggregate(refine_draws, args.aggregate))
        else:
            mask = np.broadcast_to(spatial_mask, conditioned.shape)
            refine_draws = impute(
                conditioned,
                refine_sampler,
                factor=1,
                mask=mask,
                n_samples=max(4, args.n_samples // 2),
                dt=1.0,
                seed=args.seed + 7,
            )
            refine_est = aggregate(refine_draws, args.aggregate)
        if not args.no_clamp_observed:
            refine_est = clamp_layout(refine_est, to_layout(low_train), colours)
        canvas_u8 = np.zeros((len(low_train), size, size, colours), dtype=np.uint8)
        canvas_u8[:, ::2, ::2] = low_train
        refine_bi = interpolate(canvas_u8, 2, "linear")
        if args.oracle:
            refine_target = train / 255
        elif len(imputed_train) == len(low_train):
            refine_target = to_pixels(imputed_train, colours)
        else:
            refine_target = to_pixels(
                aggregate(
                    imputed_train.reshape(len(low_train), -1, size, size * colours),
                    "mean",
                ),
                colours,
            )
        refine_model = train_refine(
            to_pixels(refine_bi[:, 0], colours),
            to_pixels(refine_est, colours),
            refine_target,
            colours=colours,
            epochs=args.refine_epochs,
            batch_size=min(args.batch_size, 32),
            device=args.device,
            seed=args.seed,
            path=args.output / "refine.pt",
        )
        refined = apply_refine(
            refine_model,
            to_pixels(bilinear[:, 0], colours),
            to_pixels(processed["round2"], colours),
            args.device,
        )
        if not args.no_clamp_observed:
            refined = clamp_observed(refined, low_test / 255)
        refined_layout = (refined * 2 - 1).reshape(len(test), size, size * colours)
        score_estimate(
            "round2_refined",
            refined_layout,
            draws["round2"],
            target,
            test_mask,
            colours,
            scores,
            labels,
            f"Round 2 refined\n(L2 CNN, {args.refine_epochs} ep)",
        )
        processed["round2_refined"] = refined_layout
        timings["refine_seconds"] = time.perf_counter() - start

    results = {
        "scores": scores,
        "timings": timings,
        "data": data_config,
        "args": {key: str(value) for key, value in vars(args).items()},
        "training": asdict(config),
        "hacks": {
            "augment": args.augment,
            "residual": args.residual,
            "imputation_aggregate": args.imputation_aggregate,
            "aggregate": args.aggregate,
            "self_ensemble": args.self_ensemble,
            "blend": args.blend,
            "clamp_observed": not args.no_clamp_observed,
            "refine_epochs": args.refine_epochs,
        },
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(
        args.output / "samples.npz",
        truth=test[:32],
        **{
            name: (to_pixels(values[:32], colours) * 255).round().astype(np.uint8)
            for name, values in draws.items()
        },
    )
    logging.info("%s", json.dumps(scores, indent=2))

    if args.no_plot:
        return
    n_examples = min(10, len(test))
    truth = test[:n_examples] / 255
    rows = [
        ("Clear truth", truth),
        (f"Blurry input\n{low_size}×{low_size}", enlarge(truth[:, ::2, ::2], size)),
        ("Bicubic", to_pixels(bicubic[:n_examples, 0], colours)),
        (
            f"Round 2\n{args.aggregate}",
            to_pixels(processed["round2"][:n_examples], colours),
        ),
    ]
    if "round2_refined" in processed:
        rows.append(
            (
                "Round 2\nrefined",
                to_pixels(processed["round2_refined"][:n_examples], colours),
            )
        )
    for name, label in (
        ("round1", "Round 1"),
        ("round2", "Round 2"),
        ("oracle", "Oracle"),
    ):
        if name in draws:
            pixels = to_pixels(draws[name][:n_examples], colours)
            rows += [
                (f"{label}\nsample", pixels[:, 0]),
                (f"{label}\nraw mean", pixels.mean(1)),
            ]
    plot_rows(args.output / "examples.png", rows, n_examples)
    if len(imputed_train) == len(low_train):
        show = imputed_train
    else:
        show = imputed_train.reshape(len(low_train), -1, size, size * colours)[:, 0]
    plot_rows(
        args.output / "imputed_train.png",
        [
            ("Blurry training\nimage", enlarge(low_train[:n_examples] / 255, size)),
            (
                "Round 1 imputation\n(round 2 trains on)",
                to_pixels(show[:n_examples], colours),
            ),
        ],
        n_examples,
    )
    scores_table(
        args.output / "table.png",
        scores,
        labels,
        title=f"Self-trained CSDI super-resolution on {data_config['dataset']}",
        subtitle=(
            f"Trained on {low_size}×{low_size} images only ({len(low_train)} images); "
            f"tested {low_size}×{low_size} → {size}×{size} on {len(test)} images\n"
            "CRPS on hidden pixels (= MAE for deterministic methods); "
            "PSNR and SSIM on whole images"
        ),
    )


if __name__ == "__main__":
    main()
