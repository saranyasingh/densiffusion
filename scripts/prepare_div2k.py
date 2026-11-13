"""Download DIV2K and save random square RGB crops for the self-trained pipeline.

    uv run --with pillow python scripts/prepare_div2k.py --output data/div2k65.npz

Run this where there is internet access (for example a cluster login node);
training reads only the saved .npz. The two archives (about 4 GB) are cached
in --cache. Each 2K photo is first reduced by --downscale with Lanczos
filtering, so a crop holds a scene rather than a few blurry pixels, and then
random --size crops are cut from it. Crops of the first 750 DIV2K training
photos form the train split, the last --validation-photos (50) the validation
split, and the 100
DIV2K validation photos (whose own test set is unreleased) the test split, so
no photo contributes to two splits. Arrays are uint8 of shape
(crops, size, size, 3).

Dataset: E. Agustsson, R. Timofte, "NTIRE 2017 Challenge on Single Image
Super-Resolution: Dataset and Study", CVPR Workshops 2017,
https://data.vision.ee.ethz.ch/cvl/DIV2K/ (research use only).
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/"
ARCHIVES = {"train": "DIV2K_train_HR.zip", "test": "DIV2K_valid_HR.zip"}


def download(name: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        print(f"Downloading {URL + name}")
        with urllib.request.urlopen(URL + name) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(path)
    return path


def photos(archive: Path, downscale: int):
    """Yield (name, downscaled RGB array) for each PNG in the archive."""
    with zipfile.ZipFile(archive) as zip_file:
        for name in sorted(n for n in zip_file.namelist() if n.endswith(".png")):
            with zip_file.open(name) as file, Image.open(file) as image:
                width, height = image.size
                small = image.convert("RGB").resize(
                    (width // downscale, height // downscale),
                    Image.Resampling.LANCZOS,
                )
            yield Path(name).name, np.asarray(small)


def crops(image: np.ndarray, size: int, count: int, rng) -> list[np.ndarray]:
    rows = rng.integers(0, image.shape[0] - size + 1, count)
    columns = rng.integers(0, image.shape[1] - size + 1, count)
    return [
        image[r : r + size, c : c + size] for r, c in zip(rows, columns, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/raw/div2k"))
    parser.add_argument(
        "--size",
        type=int,
        default=65,
        help="65 = 4k + 1 keeps the last row and column observed at both scales",
    )
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--validation-photos", type=int, default=50)
    parser.add_argument("--train-crops", type=int, default=32, help="Per photo")
    parser.add_argument("--validation-crops", type=int, default=2, help="Per photo")
    parser.add_argument("--test-crops", type=int, default=10, help="Per photo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")

    rng = np.random.default_rng(args.seed)
    splits = {"train": [], "validation": [], "test": []}
    names = {split: [] for split in splits}
    for source, archive in ARCHIVES.items():
        path = download(archive, args.cache)
        found = list(photos(path, args.downscale))
        for index, (name, image) in enumerate(found):
            if source == "test":
                split, count = "test", args.test_crops
            elif index >= len(found) - args.validation_photos:
                split, count = "validation", args.validation_crops
            else:
                split, count = "train", args.train_crops
            splits[split] += crops(image, args.size, count, rng)
            names[split] += [name] * count
    arrays = {}
    for split, images in splits.items():
        order = rng.permutation(len(images))
        arrays[split] = np.stack(images)[order]
        arrays[f"{split}_names"] = np.array(names[split])[order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **arrays,
        config=json.dumps(
            {
                "dataset": "DIV2K",
                "url": URL,
                "size": args.size,
                "downscale": args.downscale,
                "seed": args.seed,
            }
        ),
    )
    counts = {split: len(images) for split, images in splits.items()}
    print(f"Saved {counts} {args.size}x{args.size} RGB crops to {args.output}")


if __name__ == "__main__":
    main()
