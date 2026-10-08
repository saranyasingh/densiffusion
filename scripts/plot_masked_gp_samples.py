"""Plot a GP sample with a masked subset of time points.

Run from the repository root:

    uv run --with matplotlib python scripts/plot_masked_gp_samples.py --random --probability 0.3 --seed 0

Saves outputs/gp_masked_samples.png.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from densiffusion.data import mask_data
from densiffusion.gp.sampling import sample_gp

OUTPUT = Path("outputs/gp_masked_samples.png")
TIMES = np.linspace(0.0, 10.0, 401)
KERNEL = "exponential"
KERNEL_PARAMS = {"variance": 1.0, "lengthscale": 2.0}
MASK_STEP = 2
MASK_OFFSET = 1
DEFAULT_RANDOM = True
DEFAULT_PROBABILITY = 0.3


def plot_masked_gp(
    times: np.ndarray,
    *,
    kernel: str = KERNEL,
    output: Path = OUTPUT,
    mask_step: int = MASK_STEP,
    mask_offset: int = MASK_OFFSET,
    random: bool = DEFAULT_RANDOM,
    probability: float | None = DEFAULT_PROBABILITY,
    rng: int = 0,
    **kernel_params: float,
) -> Path:
    """Draw one GP sample, hide points by either a fixed pattern or random masking."""
    sample = sample_gp(
        times,
        kernel,
        rng=rng,
        n_samples=1,
        **kernel_params,
    )[0]
    masked_sample, mask = mask_data(
        sample,
        axis="time",
        step=mask_step,
        offset=mask_offset,
        fill_value=np.nan,
        random=random,
        probability=probability,
        rng=rng,
    )

    figure, axis = plt.subplots(figsize=(10, 4), layout="constrained")
    axis.plot(times, sample, color="#2a78d6", linewidth=2, label="full GP sample")
    axis.plot(
        times,
        masked_sample,
        color="#0b0b0b",
        linewidth=2,
        label="visible points",
    )
    axis.scatter(
        times[~mask],
        sample[~mask],
        color="#2a78d6",
        s=18,
        label="observed",
    )
    axis.scatter(
        times[mask],
        sample[mask],
        color="#eb6834",
        s=18,
        label="hidden",
    )
    axis.set_xlabel("time")
    axis.set_ylabel("value")
    axis.set_title("GP sample with masked time points")
    axis.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a masked GP sample.")
    parser.add_argument("--kernel", default=KERNEL, help="GP kernel name")
    parser.add_argument("--probability", type=float, default=DEFAULT_PROBABILITY)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and masking")
    parser.add_argument("--random", action="store_true", default=DEFAULT_RANDOM)
    parser.add_argument("--step", type=int, default=MASK_STEP)
    parser.add_argument("--offset", type=int, default=MASK_OFFSET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    output = plot_masked_gp(
        TIMES,
        kernel=args.kernel,
        output=args.output,
        mask_step=args.step,
        mask_offset=args.offset,
        random=args.random,
        probability=args.probability,
        rng=args.seed,
        **KERNEL_PARAMS,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
