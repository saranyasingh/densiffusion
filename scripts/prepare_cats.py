"""Download cat photos and save square RGB crops for the CSDI cat scripts.

    uv run --with pillow python scripts/prepare_cats.py --output data/afhq_cats65.npz
    uv run --with pillow python scripts/prepare_cats.py --dataset oxford-pets \
        --output data/cats65.npz

Run this where there is internet access (for example a cluster login node);
training reads only the saved .npz. Archives are cached in --cache and reused.
Each photo is center-cropped to a square and resized with Lanczos filtering,
and stored as uint8 arrays of shape (images, size, size, 3).

afhq-cat (default): the 5,558 cat faces of AFHQ-v2, 512x512 and closely
cropped, so fur and whiskers survive downscaling. The archive is about 7 GB
(cats, dogs, and wildlife); only the cats are read. The test set is drawn from
AFHQ's own test split and validation from its train split.
    Y. Choi, Y. Uh, J. Yoo, J.-W. Ha, "StarGAN v2: Diverse Image Synthesis for
    Multiple Domains", CVPR 2020, https://github.com/clovaai/stargan-v2
    (CC BY-NC 4.0).

oxford-pets: the cat breeds of Oxford-IIIT Pet (about 790 MB), whole scenes in
which the cat is often small. Splits are drawn in a seeded random order.
    O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs",
    CVPR 2012, https://www.robots.ox.ac.uk/~vgg/data/pets/ (CC BY-SA 4.0).
"""

import argparse
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

AFHQ_URL = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
OXFORD_URL = "https://thor.robots.ox.ac.uk/pets/images.tar.gz"
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
    return np.asarray(
        image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    )


def afhq_cats(cache: Path, size: int) -> dict[str, tuple[list, list]]:
    """Cat images and names from AFHQ-v2, keyed by its split (train, test)."""
    archive = cache / "afhq_v2.zip"
    download(AFHQ_URL, archive)
    splits = {"train": ([], []), "test": ([], [])}
    with zipfile.ZipFile(archive) as zip_file:
        for name in sorted(zip_file.namelist()):
            split, kind, *_ = name.split("/")
            if kind != "cat" or not name.endswith(".png"):
                continue
            with zip_file.open(name) as file, Image.open(file) as image:
                splits[split][0].append(square_crop(image, size))
            splits[split][1].append(Path(name).name)
    return splits


def oxford_cats(cache: Path, size: int) -> tuple[list, list]:
    archive = cache / "oxford_pets_images.tar.gz"
    download(OXFORD_URL, archive)
    images, names = [], []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if not (member.isfile() and name.endswith(".jpg")):
                continue
            if name.rsplit("_", 1)[0] not in CAT_BREEDS:
                continue
            with Image.open(tar.extractfile(member)) as image:
                images.append(square_crop(image, size))
            names.append(name)
    return images, names


def shuffled(images: list, names: list, rng) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(images))
    return np.stack(images)[order], np.array(names)[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("afhq-cat", "oxford-pets"), default="afhq-cat"
    )
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

    rng = np.random.default_rng(args.seed)
    n_test, n_validation = args.test_images, args.validation_images
    if args.dataset == "afhq-cat":
        found = afhq_cats(args.cache, args.size)
        train, train_names = shuffled(*found["train"], rng)
        test, test_names = shuffled(*found["test"], rng)
        splits = {
            "train": (train[n_validation:], train_names[n_validation:]),
            "validation": (train[:n_validation], train_names[:n_validation]),
            "test": (test[:n_test], test_names[:n_test]),
        }
        label, url = "AFHQ-v2 (cats)", AFHQ_URL
    else:
        images, names = shuffled(*oxford_cats(args.cache, args.size), rng)
        parts = {
            "test": slice(0, n_test),
            "validation": slice(n_test, n_test + n_validation),
            "train": slice(n_test + n_validation, None),
        }
        splits = {split: (images[part], names[part]) for split, part in parts.items()}
        label, url = "Oxford-IIIT Pet (cats)", OXFORD_URL

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{split: images for split, (images, _) in splits.items()},
        **{f"{split}_names": names for split, (_, names) in splits.items()},
        config=json.dumps(
            {"dataset": label, "url": url, "size": args.size, "seed": args.seed}
        ),
    )
    counts = {split: len(images) for split, (images, _) in splits.items()}
    print(f"Saved {counts} {args.size}x{args.size} RGB cat images to {args.output}")


if __name__ == "__main__":
    main()
