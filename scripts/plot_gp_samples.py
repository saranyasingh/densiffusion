"""Plot a few GP trajectories drawn with the exponential kernel.

Run from the repository root:

    uv run --with matplotlib python scripts/plot_gp_samples.py

Saves outputs/gp_samples.png. Both panels use the same variance; only the
lengthscale differs, which sets how quickly values stop being correlated.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from densiffusion.gp.sampling import sample_gp

OUTPUT = Path("outputs/gp_samples.png")
TIMES = np.linspace(0.0, 10.0, 401)
VARIANCE = 1.0
LENGTHSCALES = (0.1, 10.0)
N_SAMPLES = 3
SEED = 0

# First three categorical slots of the reference palette, plus neutral inks.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#ecebe6"


def main() -> None:
    figure = Figure(figsize=(10, 3.8), layout="constrained")
    axes = figure.subplots(1, len(LENGTHSCALES), sharey=True)
    band = 2 * np.sqrt(VARIANCE)
    for ax, lengthscale in zip(axes, LENGTHSCALES, strict=True):
        samples = sample_gp(
            TIMES,
            "exponential",
            rng=SEED,
            n_samples=N_SAMPLES,
            variance=VARIANCE,
            lengthscale=lengthscale,
        )
        ax.axhspan(-band, band, color=BAND, linewidth=0, zorder=0)
        ax.axhline(0.0, color=MUTED, linewidth=1, zorder=1)
        for path, color in zip(samples, COLORS, strict=True):
            ax.plot(TIMES, path, color=color, linewidth=1.5, zorder=2)
        ax.set_title(f"lengthscale = {lengthscale:g}", loc="left", color=INK)
        ax.set_xlabel("time", color=MUTED)
        ax.set_xlim(TIMES[0], TIMES[-1])
        ax.tick_params(colors=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
    axes[0].set_ylabel("value", color=MUTED)
    axes[-1].text(
        TIMES[-1] - 0.15,
        band - 0.12,
        "mean ± 2 sd",
        ha="right",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    figure.suptitle(
        f"{N_SAMPLES} draws from a GP with exponential kernel "
        f"(mean 0, variance {VARIANCE:g})",
        x=0.01,
        ha="left",
        color=INK,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=150, facecolor="white")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
