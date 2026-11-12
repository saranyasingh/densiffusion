"""Download MNIST and save 29x29 digits for the self-trained CSDI pipeline.

    uv run python scripts/prepare_digits.py --output data/mnist29.npz

Run this where there is internet access (for example a cluster login node);
training reads only the saved .npz. The four archives (about 12 MB) are cached
in --cache. Each 28x28 digit gets one blank row and column at the bottom and
right, making 29x29 = 4 * 7 + 1, so keeping every other pixel gives 15x15 and
then 8x8 with the last row and column always kept. Validation images come
from MNIST's training split and test images from its test split. Arrays are
uint8 of shape (images, 29, 29, 1).

Dataset: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning
applied to document recognition", Proc. IEEE 1998.
"""

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def fetch(name: str, cache: Path) -> bytes:
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL + name}")
        urllib.request.urlretrieve(URL + name, path)
    return gzip.decompress(path.read_bytes())


def load(split: str, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    images, labels = (fetch(name, cache) for name in FILES[split])
    images = np.frombuffer(images, np.uint8, offset=16).reshape(-1, 28, 28)
    labels = np.frombuffer(labels, np.uint8, offset=8)
    return np.pad(images, ((0, 0), (0, 1), (0, 1)))[..., None], labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/raw/mnist"))
    parser.add_argument("--validation-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")

    rng = np.random.default_rng(args.seed)
    train, train_labels = load("train", args.cache)
    test, test_labels = load("test", args.cache)
    order, test_order = rng.permutation(len(train)), rng.permutation(len(test))
    train, train_labels = train[order], train_labels[order]
    test, test_labels = test[test_order], test_labels[test_order]
    n = args.validation_images
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train=train[n:],
        train_labels=train_labels[n:],
        validation=train[:n],
        validation_labels=train_labels[:n],
        test=test,
        test_labels=test_labels,
        config=json.dumps(
            {"dataset": "MNIST", "url": URL, "size": 29, "seed": args.seed}
        ),
    )
    print(
        f"Saved {len(train) - n} train, {n} validation, {len(test)} test "
        f"29x29 digits to {args.output}"
    )


if __name__ == "__main__":
    main()
