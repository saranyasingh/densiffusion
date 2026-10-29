"""Side-by-side examples from the supervised and self-supervised cat runs.

    uv run --with matplotlib python scripts/plot_cats_comparison.py \
        --supervised outputs/cats_csdi_<job id> \
        --self-supervised outputs/cats_selfsup_<job id> \
        --output outputs/cats_comparison.png

Reads the samples.npz written by impute_cats_csdi.py and
impute_cats_selfsup_csdi.py, which hold the same first test images.
"""

import argparse
from pathlib import Path

import numpy as np
from impute_cats_csdi import interpolate, to_pixels
from matplotlib.figure import Figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--supervised", type=Path, required=True)
    parser.add_argument("--self-supervised", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--images", type=int, nargs="+", default=[0, 1, 2, 3], help="Test indices"
    )
    args = parser.parse_args()

    with np.load(args.supervised / "samples.npz") as samples:
        truth, supervised = samples["truth"], samples["csdi"] / 255
    with np.load(args.self_supervised / "samples.npz") as samples:
        if not np.array_equal(samples["truth"], truth):
            parser.error("The two runs saved different test images")
        self_supervised = samples["csdi"] / 255
    truth = truth[args.images]
    supervised, self_supervised = supervised[args.images], self_supervised[args.images]
    size, colours = truth.shape[1], truth.shape[-1]
    low = np.repeat(np.repeat(truth[:, ::2, ::2], 2, 1), 2, 2)[:, :size, :size]
    bicubic = to_pixels(interpolate(truth, 2, "cubic")[:, 0], colours)

    columns = [
        ("High-res truth", truth / 255),
        ("Low-res input", low / 255),
        ("Bicubic", bicubic),
        ("Supervised\nsample", supervised[:, 0]),
        ("Supervised\nmean of 10", supervised.mean(axis=1)),
        ("Self-supervised\nsample", self_supervised[:, 0]),
        ("Self-supervised\nmean of 10", self_supervised.mean(axis=1)),
    ]
    figure = Figure(figsize=(2 * len(columns), 2.1 * len(args.images) + 0.4))
    axes = figure.subplots(len(args.images), len(columns), squeeze=False)
    for row in range(len(args.images)):
        for column, (title, images) in enumerate(columns):
            axis = axes[row, column]
            axis.imshow(images[row], interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=10)
    figure.suptitle(
        f"Cat super-resolution, {(size + 1) // 2}×{(size + 1) // 2} → {size}×{size}: "
        "supervised (trained on high-res) vs self-supervised (low-res only)",
        fontsize=11,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
