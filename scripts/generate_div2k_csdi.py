"""Round 3: generate new 65x65 RGB crops from a model trained on self-trained data.

    uv run --extra research --with matplotlib python scripts/generate_div2k_csdi.py \
        --output outputs/div2k_60201471 --device cuda --oracle

Reads a finished `bootstrap_csdi.py` DIV2K output directory (same --data,
--train-images, and model sizes). With `--source round2` (default) the round 2
model super-resolves every blurry training and validation crop; with
`--source imputed` the round 1 imputations in imputed.npz are used instead. A
new CSDI then learns to draw whole crops from those images: every pixel is
hidden except the bottom-right one, so sampling starts from almost pure noise
(one RGB value from a random training crop). The clear training crops are never
read, except by `--oracle`, which trains the same generator on them as an upper
bound, and by the evaluation encoder.

Generated crops are compared with held-out test crops through a small CNN
trained to predict image rotations on the clear training crops (evaluation only,
like the Inception network of FID): Frechet distance between its penultimate
features (FID). Novelty is the RMS pixel distance from each crop to its nearest
neighbour in the generator's own training set, next to the same distance for
real test crops; generated crops far closer than that are copies. Checkpoints
are reused, so a rerun continues where it stopped.
"""

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from bootstrap_csdi import fit, plot_rows
from generate_digits_csdi import (
    corner_mask,
    frechet,
    generate,
    nearest,
    super_resolve,
)
from impute_cats_csdi import to_layout, to_pixels
from plot_forecast_table import draw_table
from torch import nn

from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig


class Encoder(nn.Module):
    """Small RGB CNN; rotation head is for training features only."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, 4)

    def forward(self, images):
        return self.head(self.features(images))


def as_input(pixels: np.ndarray, device) -> torch.Tensor:
    """(images, size, size, 3) pixels in [0, 1] -> (images, 3, size, size)."""
    return torch.as_tensor(pixels, dtype=torch.float32, device=device).permute(
        0, 3, 1, 2
    )


def train_encoder(path, images, *, epochs, device, seed) -> Encoder:
    """Rotation-prediction CNN on clear training crops, used only to score."""
    torch.manual_seed(seed)
    model = Encoder().to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    pixels = images / 255
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        batches = max(1, len(pixels) // 64)
        for indices in np.array_split(rng.permutation(len(pixels)), batches):
            batch = pixels[indices]
            rotation = int(rng.integers(4))
            rotated = np.rot90(batch, rotation, axes=(1, 2)).copy()
            labels = torch.full(
                (len(batch),), rotation, dtype=torch.long, device=device
            )
            loss = nn.functional.cross_entropy(model(as_input(rotated, device)), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            count += len(batch)
        logging.info("encoder epoch=%d loss=%.4g", epoch, total / count)
    torch.save(model.state_dict(), path)
    return model.eval()


@torch.no_grad()
def embed(encoder, pixels, device) -> np.ndarray:
    """Penultimate features for pixels in [0, 1]."""
    features = []
    for start in range(0, len(pixels), 256):
        features.append(
            encoder.features(as_input(pixels[start : start + 256], device))
            .cpu()
            .numpy()
        )
    return np.concatenate(features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/div2k65.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("round2", "imputed"), default="round2")
    parser.add_argument("--train-images", type=int, default=10000)
    parser.add_argument("--n-generate", type=int, default=500)
    parser.add_argument(
        "--oracle", action="store_true", help="Also train on clear training crops"
    )
    parser.add_argument("--encoder-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
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
    if not args.output.is_dir():
        parser.error(f"{args.output} is not a bootstrap_csdi.py output directory")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.output / "generation.log"),
            logging.StreamHandler(),
        ],
    )
    with np.load(args.data, allow_pickle=False) as dataset:
        train = dataset["train"][: args.train_images]
        all_train = dataset["train"]
        validation = dataset["validation"]
        test = dataset["test"]
    size, colours = test.shape[1], test.shape[-1]
    if colours != 3 or 2 * args.n_generate > len(test):
        parser.error("Needs RGB DIV2K crops and 2 * --n-generate test images")
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

    source_path = args.output / f"round3_{args.source}_data.npz"
    if source_path.exists():
        with np.load(source_path) as source:
            source_train, source_validation = source["train"], source["validation"]
    elif args.source == "imputed":
        with np.load(args.output / "imputed.npz") as imputed:
            source_train, source_validation = imputed["train"], imputed["validation"]
    else:
        checkpoint = args.output / "round2.pt"
        if not checkpoint.exists():
            parser.error(f"{checkpoint} not found; run bootstrap_csdi.py first")
        layout_shape = np.zeros((1, size, size * colours))
        round2 = fit("round2", layout_shape, None, None, args=args, config=config)
        logging.info("Super-resolving %d blurry training crops", len(train))
        start = time.perf_counter()
        round2_sampler = DDPMSampler(round2, sampling)
        source_train, source_validation = (
            super_resolve(
                images[:, ::2, ::2],
                round2_sampler,
                size=size,
                colours=colours,
                seed=args.seed + offset,
            ).astype(np.float32)
            for offset, images in ((5, train), (6, validation))
        )
        timings["round2_train_set_seconds"] = time.perf_counter() - start
        np.savez_compressed(
            source_path, train=source_train, validation=source_validation
        )

    mask = corner_mask(size, colours)
    generators = {}
    start = time.perf_counter()
    generators[f"round3_{args.source}"] = (
        fit(
            f"round3_{args.source}",
            source_train,
            source_validation,
            mask,
            args=args,
            config=config,
        ),
        source_train,
    )
    timings["round3_train_seconds"] = time.perf_counter() - start
    if args.oracle:
        start = time.perf_counter()
        clear_train = to_layout(train)
        generators["oracle_generator"] = (
            fit(
                "oracle_generator",
                clear_train,
                to_layout(validation),
                mask,
                args=args,
                config=config,
            ),
            clear_train,
        )
        timings["oracle_train_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    generated = {
        name: to_pixels(
            generate(
                model, reference, args.n_generate, config=sampling, seed=args.seed + 7
            ),
            colours,
        )
        for name, (model, reference) in generators.items()
    }
    timings["generate_seconds"] = time.perf_counter() - start

    encoder = train_encoder(
        args.output / "encoder.pt",
        all_train,
        epochs=args.encoder_epochs,
        device=args.device,
        seed=args.seed,
    )

    n = args.n_generate
    reference_features = embed(encoder, test[:n] / 255, args.device)
    sets = {"real": (test[n : 2 * n] / 255, train / 255)}
    sets |= {
        name: (pixels, to_pixels(generators[name][1], colours))
        for name, pixels in generated.items()
    }
    scores, neighbours = {}, {}
    for name, (pixels, training_set) in sets.items():
        features = embed(encoder, pixels, args.device)
        distance, index = nearest(pixels, training_set, args.device)
        neighbours[name] = training_set[index]
        scores[name] = {
            "fid": frechet(features, reference_features),
            "nearest_distance": float(np.median(distance)),
        }
    results = {
        "scores": scores,
        "timings": timings,
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    (args.output / "generation.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(
        args.output / "generated.npz",
        **{
            name: (pixels * 255).round().astype(np.uint8)
            for name, pixels in generated.items()
        },
    )
    logging.info("%s", json.dumps(scores, indent=2))

    if args.no_plot:
        return
    labels = {
        "real": "Real test crops",
        f"round3_{args.source}": f"Round 3, trained on\n{args.source} crops",
        "oracle_generator": "Oracle, trained on\nclear crops",
    }
    n_examples = 8
    rows = []
    for name in generated:
        rows += [
            (f"{labels[name]}\n(new)", generated[name][:n_examples]),
            ("Nearest\ntraining crop", neighbours[name][:n_examples]),
        ]
    plot_rows(args.output / "generated.png", rows, n_examples)
    plot_rows(
        args.output / "generated_grid.png",
        [
            (labels[name], generated[name][row * n_examples : (row + 1) * n_examples])
            for name in generated
            for row in range(4)
        ],
        n_examples,
    )
    metrics = [("fid", "{:.2f}", min), ("nearest_distance", "{:.3f}", None)]
    names = list(sets)
    bold = {
        (names.index(best(generated, key=lambda key: scores[key][metric])), column)
        for column, (metric, _, best) in enumerate(metrics, start=1)
        if best is not None
    }
    draw_table(
        [
            [labels[name]]
            + [form.format(scores[name][metric]) for metric, form, _ in metrics]
            for name in names
        ],
        ["Crops", "FID ↓", "NN distance"],
        bold=bold,
        widths=[1.4, 1, 1],
        title="New 65×65 DIV2K crops from self-trained CSDI",
        subtitle=(
            f"{n} generated crops vs {n} held-out test crops\n"
            "NN distance: median RMS pixel distance to the nearest image "
            "the generator trained on"
        ),
        output=args.output / "generation_table.png",
    )


if __name__ == "__main__":
    main()
