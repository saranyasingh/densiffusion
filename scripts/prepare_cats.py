"""Download the cat photos of Oxford-IIIT Pet and save square RGB crops.

    uv run --with pillow python scripts/prepare_cats.py --output data/cats65.npz

Run this where there is internet access (for example a cluster login node);
training reads only the saved .npz. The archive (about 790 MB) is cached in
--cache and reused. Each photo is center-cropped to a square and resized with
Lanczos filtering. Images are split into train, validation, and test sets in a
seeded random order and stored as uint8 arrays of shape (images, size, size, 3).

Dataset: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and
Dogs", CVPR 2012, https://www.robots.ox.ac.uk/~vgg/data/pets/ (CC BY-SA 4.0).
"""

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

URL = "https://thor.robots.ox.ac.uk/pets/images.tar.gz"
CAT_BREEDS = (
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
)


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        while chunk := response.read(1 << 20):
            output.write(chunk)
    partial.replace(path)


def square_crop(image: Image.Image, size: int) -> np.ndarray:
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return np.asarray(image.resize((size, size), Image.Resampling.LANCZOS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--size",
        type=int,
        default=65,
        help="65 = 8k + 1 keeps the last row and column observed at x2, x4, x8",
    )
    parser.add_argument("--validation-images", type=int, default=100)
    parser.add_argument("--test-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists")

    archive = args.cache / "oxford_pets_images.tar.gz"
    download(URL, archive)
    images, names = [], []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if not (member.isfile() and name.endswith(".jpg")):
                continue
            if name.rsplit("_", 1)[0] not in CAT_BREEDS:
                continue
            with Image.open(tar.extractfile(member)) as image:
                images.append(square_crop(image.convert("RGB"), args.size))
            names.append(name)
    order = np.random.default_rng(args.seed).permutation(len(images))
    images = np.stack(images)[order]
    names = np.array(names)[order]
    n_test, n_validation = args.test_images, args.validation_images
    splits = {
        "test": slice(0, n_test),
        "validation": slice(n_test, n_test + n_validation),
        "train": slice(n_test + n_validation, None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{split: images[part] for split, part in splits.items()},
        **{f"{split}_names": names[part] for split, part in splits.items()},
        config=json.dumps(
            {
                "dataset": "Oxford-IIIT Pet (cats)",
                "url": URL,
                "size": args.size,
                "seed": args.seed,
            }
        ),
    )
    counts = {split: len(images[part]) for split, part in splits.items()}
    print(f"Saved {counts} {args.size}x{args.size} RGB cat images to {args.output}")


if __name__ == "__main__":
    main()
