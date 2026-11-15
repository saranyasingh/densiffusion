"""Round 3: generate new 29x29 digits from a model trained on self-trained data.

    uv run --extra research --with matplotlib python scripts/generate_digits_csdi.py \
        --output outputs/digits_60074338 --device cuda --oracle

Reads a finished `bootstrap_csdi.py` output directory (same --data,
--train-images, and model sizes). With `--source round2` (default) the round 2
model super-resolves every blurry training and validation digit; with
`--source imputed` the round 1 imputations in imputed.npz are used instead. A
new CSDI then learns to draw whole digits from those images: every pixel is
hidden except the bottom-right one, which `prepare_digits.py` pads blank, so
sampling starts from pure noise and nothing about the digit is given. The
clear training digits are never read, except by `--oracle`, which trains the
same generator on them as an upper bound, and by the evaluation classifier.

Generated digits are compared with held-out test digits through a small CNN
trained on the clear training digits (evaluation only, like the Inception
network of FID): Frechet distance between its penultimate features (FID),
mean top-class probability (confidence), and the entropy of the predicted
class histogram (ln 10 = 2.303 when all classes appear equally often).
Novelty is the RMS pixel distance from each digit to its nearest neighbour in
the generator's own training set, next to the same distance for real test
digits; generated digits far closer than that are copies. Checkpoints are
reused, so a rerun continues where it stopped.
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
from impute_cats_csdi import grid_mask, to_layout, to_pixels
from plot_forecast_table import draw_table
from scipy.linalg import sqrtm
from torch import nn

from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig


def corner_mask(size: int, colours: int) -> np.ndarray:
    """Hide every value except the bottom-right pixel."""
    mask = np.ones((size, size), dtype=bool)
    mask[-1, -1] = False
    return np.repeat(mask, colours, axis=1)


def super_resolve(low, sampler, *, size, colours, seed) -> np.ndarray:
    """Round 2 draw at full size for uint8 (images, n, n, colours) blurry images."""
    full = np.zeros((len(low), size, size, colours), dtype=np.uint8)
    full[:, ::2, ::2] = low
    layout = to_layout(full)
    mask = np.broadcast_to(grid_mask(size, colours, 2), layout.shape)
    return impute(layout, sampler, factor=1, mask=mask, n_samples=1, seed=seed)[:, 0]


def generate(model, reference, n_images, *, config, seed) -> np.ndarray:
    """Draw n_images layouts, keeping only the blank corner of reference images."""
    rng = np.random.default_rng(seed)
    start = reference[rng.integers(len(reference), size=n_images)]
    size, width = start.shape[1:]
    mask = np.broadcast_to(corner_mask(size, width // size), start.shape)
    return impute(
        start, DDPMSampler(model, config), factor=1, mask=mask, n_samples=1, seed=seed
    )[:, 0]


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, images):
        return self.head(self.features(images))


def as_input(pixels: np.ndarray, device) -> torch.Tensor:
    """(images, 29, 29, 1) pixels in [0, 1] -> (images, 1, 29, 29) tensor."""
    return torch.as_tensor(pixels[..., 0], dtype=torch.float32, device=device)[:, None]


def train_classifier(path, images, labels, *, epochs, device, seed) -> Classifier:
    """CNN on clear training digits, used only to score generated digits."""
    torch.manual_seed(seed)
    model = Classifier().to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    pixels = images / 255
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for indices in np.array_split(rng.permutation(len(pixels)), len(pixels) // 256):
            loss = nn.functional.cross_entropy(
                model(as_input(pixels[indices], device)),
                torch.as_tensor(labels[indices], dtype=torch.long, device=device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        logging.info("classifier epoch=%d loss=%.4g", epoch, total / len(pixels))
    torch.save(model.state_dict(), path)
    return model.eval()


@torch.no_grad()
def embed(classifier, pixels, device) -> tuple[np.ndarray, np.ndarray]:
    """Penultimate features and class probabilities for pixels in [0, 1]."""
    features, probabilities = [], []
    for start in range(0, len(pixels), 1024):
        hidden = classifier.features(as_input(pixels[start : start + 1024], device))
        features.append(hidden.cpu().numpy())
        probabilities.append(classifier.head(hidden).softmax(1).cpu().numpy())
    return np.concatenate(features), np.concatenate(probabilities)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = sqrtm(cov_a @ cov_b).real
    difference = a.mean(0) - b.mean(0)
    return float(difference @ difference + np.trace(cov_a + cov_b - 2 * root))


@torch.no_grad()
def nearest(pixels, reference, device) -> tuple[np.ndarray, np.ndarray]:
    """RMS distance to, and index of, each image's nearest reference image."""
    reference = torch.as_tensor(
        reference.reshape(len(reference), -1), dtype=torch.float32, device=device
    )
    distances, indices = [], []
    for start in range(0, len(pixels), 256):
        batch = torch.as_tensor(
            pixels[start : start + 256].reshape(-1, reference.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        distance, index = torch.cdist(batch, reference).min(1)
        distances.append(distance.cpu().numpy())
        indices.append(index.cpu().numpy())
    scale = reference.shape[1] ** 0.5
    return np.concatenate(distances) / scale, np.concatenate(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/mnist29.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=("round2", "imputed"), default="round2")
    parser.add_argument("--train-images", type=int, default=20000)
    parser.add_argument("--n-generate", type=int, default=1000)
    parser.add_argument(
        "--oracle", action="store_true", help="Also train on clear training digits"
    )
    parser.add_argument("--classifier-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
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
        all_train, train_labels = dataset["train"], dataset["train_labels"]
        validation = dataset["validation"]
        test, test_labels = dataset["test"], dataset["test_labels"]
    size, colours = test.shape[1], test.shape[-1]
    if colours != 1 or 2 * args.n_generate > len(test):
        parser.error("Needs grey digits and 2 * --n-generate test images")
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
        logging.info("Super-resolving %d blurry training digits", len(train))
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

    classifier = train_classifier(
        args.output / "classifier.pt",
        all_train,
        train_labels,
        epochs=args.classifier_epochs,
        device=args.device,
        seed=args.seed,
    )
    _, test_probabilities = embed(classifier, test / 255, args.device)
    accuracy = float((test_probabilities.argmax(1) == test_labels).mean())
    logging.info("classifier test accuracy=%.4f", accuracy)

    n = args.n_generate
    reference_features, _ = embed(classifier, test[:n] / 255, args.device)
    sets = {"real": (test[n : 2 * n] / 255, train / 255)}
    sets |= {
        name: (pixels, to_pixels(generators[name][1], colours))
        for name, pixels in generated.items()
    }
    scores, neighbours = {}, {}
    for name, (pixels, training_set) in sets.items():
        features, probabilities = embed(classifier, pixels, args.device)
        histogram = np.bincount(probabilities.argmax(1), minlength=10) / len(pixels)
        distance, index = nearest(pixels, training_set, args.device)
        neighbours[name] = training_set[index]
        scores[name] = {
            "fid": frechet(features, reference_features),
            "confidence": float(probabilities.max(1).mean()),
            "class_entropy": float(
                -(histogram * np.log(np.maximum(histogram, 1e-12))).sum()
            ),
            "nearest_distance": float(np.median(distance)),
            "class_histogram": histogram.round(4).tolist(),
        }
    results = {
        "scores": scores,
        "classifier_test_accuracy": accuracy,
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
        "real": "Real test digits",
        f"round3_{args.source}": f"Round 3, trained on\n{args.source} digits",
        "oracle_generator": "Oracle, trained on\nclear digits",
    }
    n_examples = 10
    rows = []
    for name in generated:
        rows += [
            (f"{labels[name]}\n(new)", generated[name][:n_examples]),
            ("Nearest\ntraining digit", neighbours[name][:n_examples]),
        ]
    plot_rows(args.output / "generated.png", rows, n_examples)
    plot_rows(
        args.output / "generated_grid.png",
        [
            (labels[name], generated[name][row * n_examples : (row + 1) * n_examples])
            for name in generated
            for row in range(5)
        ],
        n_examples,
    )
    metrics = [
        ("fid", "{:.2f}", min),
        ("confidence", "{:.3f}", max),
        ("class_entropy", "{:.3f}", max),
        ("nearest_distance", "{:.3f}", None),
    ]
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
        ["Digits", "FID ↓", "Confidence ↑", "Class entropy ↑", "NN distance"],
        bold=bold,
        widths=[1, 1, 1, 1],
        title="New 29×29 digits from self-trained CSDI",
        subtitle=(
            f"{n} generated digits vs {n} held-out test digits; classifier "
            f"accuracy {accuracy:.3f}\nNN distance: median RMS pixel distance "
            "to the nearest image the generator trained on"
        ),
        output=args.output / "generation_table.png",
    )


if __name__ == "__main__":
    main()
