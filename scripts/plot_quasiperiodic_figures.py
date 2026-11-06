"""Illustrate the quasi-periodic sweep: tasks, setups and example predictions.

Reads the outputs of run_pristi_densiffusion.py for the qp_* tasks and writes
three figures next to them:

    uv run --extra research --with matplotlib python \
        scripts/plot_quasiperiodic_figures.py \
        --output outputs/densiffusion_quasiperiodic
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
from run_pristi_densiffusion import (
    AXIS,
    COLOURS,
    FINE,
    GRID,
    HISTORIES,
    HISTORY,
    INK,
    MUTED,
    SECONDARY,
    SURFACE,
    TASKS,
    _style,
    exact_history,
    generate,
    kernel_params,
)

from densiffusion.baselines import GPPosteriorSampler
from densiffusion.gp.posterior import gp_posterior

TASK_RATIOS = ("1.6", "2.5", "4", "16")
PREDICTION_RATIOS = ("1.6", "2.5", "4", "8", "16")
SETUP_RATIO = "2.5"
WINDOW = 0
PIPELINES = (
    (
        "Sparse",
        "sparse",
        None,
        "No densifier. Forecaster trained on coarse windows.",
    ),
    (
        "Coarse-data",
        "learned_x8",
        "densified",
        "One imputer learns 2dt -> dt from coarse data, applied three times.\n"
        "Forecaster trained on coarse windows densified by it.",
    ),
    (
        "Fine-data",
        "fine_x8",
        "densified_fine",
        "One imputer per level, trained on fine paths at the gap it fills.\n"
        "Forecaster trained on true fine windows.",
    ),
    (
        "Oracle",
        "oracle_x8",
        "exact",
        "Exact GP posterior (knows the kernel), no training.\n"
        "Uses the fine-data forecaster.",
    ),
)
PREDICTED = ("sparse", "learned_x2", "fine_x2", "oracle_x2", "gp")
SHORT = {
    "sparse": "Sparse",
    "learned_x2": "Coarse-data x2",
    "fine_x2": "Fine-data x2",
    "oracle_x2": "Oracle x2",
    "gp": "Exact GP",
    "learned_x8": "Coarse-data x8",
    "fine_x8": "Fine-data x8",
    "oracle_x8": "Oracle x8",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _style(plt)
    plot_tasks(plt, args.output)
    plot_setups(plt, args.output)
    plot_predictions(plt, args.output)
    print(f"Figures written to {args.output}")


def load(output, ratio):
    folder = output / f"qp_{ratio}"
    results = json.loads((folder / "results.json").read_text())
    return results, np.load(folder / "samples.npz")


def plot_tasks(plt, output):
    """One test path per ratio: fine truth, coarse samples, what they suggest."""
    figure, axes = plt.subplots(len(TASK_RATIOS), 1, figsize=(10, 8.4), sharex=True)
    notes = {
        "1.6": "aliased: the coarse points trace a slower wave",
        "2.5": "barely resolved: the coarse points zig-zag",
        "4": "resolved, but corners are cut",
        "16": "well resolved: the coarse points trace the wave",
    }
    span = 16
    for ax, ratio in zip(axes, TASK_RATIOS, strict=True):
        paths, times = generate(f"qp_{ratio}", 1, "test")
        fine = paths[0, : span * FINE + 1]
        steps = times[: span * FINE + 1] / TASKS[f"qp_{ratio}"][1]
        ax.plot(steps, fine, color=AXIS, linewidth=2, label="Fine path (dt/8)")
        ax.plot(
            steps[::FINE],
            fine[::FINE],
            "--",
            color=COLOURS["sparse"],
            linewidth=1.5,
            label="Joining the coarse points",
        )
        ax.plot(
            steps[::FINE],
            fine[::FINE],
            "o",
            color=INK,
            ms=5,
            mec=SURFACE,
            mew=1,
            label="Coarse observations (dt)",
        )
        ax.set_title(
            f"P/dt = {ratio}: {float(ratio):g} samples per cycle, {notes[ratio]}",
            loc="left",
        )
        ax.set_yticks([])
    axes[-1].set_xlabel("Time (coarse steps dt)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
    figure.suptitle(
        "Quasi-periodic GP tasks: the period-to-spacing ratio P/dt sets how well "
        "coarse data show the wave",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(output / "figure_tasks.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_setups(plt, output):
    """Each pipeline on one test window: what it learns, its history, its forecast."""
    task = f"qp_{SETUP_RATIO}"
    results, data = load(output, SETUP_RATIO)
    truth, window = data["truth"][WINDOW], data["windows"][WINDOW]
    gp = kernel_params(TASKS[task][2])
    sampler = GPPosteriorSampler(partial(gp_posterior, mean=0.0, **gp))
    exact = exact_history(
        sampler,
        data["truth"][WINDOW : WINDOW + 1],
        data["times"][WINDOW : WINDOW + 1],
        FINE,
        HISTORIES,
        0,
    )[0]
    histories = {
        "densified": data["densified"][WINDOW],
        "densified_fine": data["densified_fine"][WINDOW],
        "exact": exact,
    }
    train_paths, _ = generate(task, 1, "train")
    fine_steps = np.arange(truth.size) / FINE
    coarse_steps = np.arange(window.size)
    gp_crps = results["forecast"]["gp"]["crps"]

    figure, axes = plt.subplots(
        len(PIPELINES),
        3,
        figsize=(14, 11),
        gridspec_kw={"width_ratios": (1, 1.5, 1.5)},
    )
    headings = (
        "What the densifier learns from",
        "Test history given to the forecaster",
        "Forecast at coarse future points",
    )
    for ax, heading in zip(axes[0], headings, strict=True):
        ax.set_title(heading, loc="left", fontsize=11)
    for row, (name, method, history_key, description) in zip(
        axes, PIPELINES, strict=True
    ):
        colour = COLOURS[method]
        _learns_from(row[0], name, train_paths[0], colour)
        row[0].set_ylabel(name, color=INK, fontsize=11)
        row[0].text(
            0.0,
            -0.2,
            description,
            transform=row[0].transAxes,
            fontsize=7.5,
            color=SECONDARY,
            va="top",
        )

        ax = row[1]
        ax.plot(fine_steps, truth, color=AXIS, linewidth=2)
        if history_key is not None:
            draws = histories[history_key]
            dense_steps = np.arange(draws.shape[-1]) / FINE
            for index, path in enumerate(draws[:2]):
                ax.plot(
                    dense_steps,
                    path,
                    color=colour,
                    linewidth=1.3,
                    alpha=0.85,
                    label="Densified draws" if index == 0 else None,
                )
        _coarse(ax, coarse_steps, window)
        ax.axvspan(HISTORY - 1, coarse_steps[-1] + 0.3, color=GRID, alpha=0.5)
        ax.set_xlim(-0.3, coarse_steps[-1] + 0.3)

        ax = row[2]
        ax.plot(fine_steps, truth, color=AXIS, linewidth=2)
        _coarse(ax, coarse_steps, window)
        for offset, key in ((-0.12, method), (0.12, "gp")):
            _interval(ax, coarse_steps[HISTORY:] + offset, data[key][WINDOW], key)
        ratio = results["forecast"][method]["crps"] / gp_crps
        ax.text(
            0.02,
            0.95,
            f"{SHORT.get(method, name)}: CRPS {ratio:.2f}x exact GP (all windows)",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=SECONDARY,
        )
        ax.set_xlim(HISTORY - 4.5, coarse_steps[-1] + 0.5)
        for ax in row:
            ax.set_yticks([])
    for ax in axes[-1, 1:]:
        ax.set_xlabel("Time (coarse steps dt; shaded: future)")

    handles = [
        plt.Line2D([], [], color=AXIS, linewidth=2, label="Fine truth"),
        plt.Line2D([], [], marker="o", color=INK, ls="none", label="Coarse history"),
        plt.Line2D(
            [], [], marker="o", mfc=SURFACE, mec=INK, ls="none", label="True future"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            mfc=SURFACE,
            mec=MUTED,
            ls="none",
            label="Hidden in training (target)",
        ),
        plt.Line2D([], [], color=COLOURS["gp"], linewidth=2.5, label="Exact GP: 90%"),
        plt.Line2D([], [], color=SECONDARY, linewidth=2.5, label="Method: 90%"),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=6, frameon=False)
    figure.suptitle(
        f"Four ways to forecast coarse data, P/dt = {SETUP_RATIO}, x8 densification "
        f"(test window {WINDOW}, not selected)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=2)
    figure.savefig(output / "figure_setups.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def _learns_from(ax, name, path, colour):
    """An example training window for each densifier."""
    if name in ("Sparse", "Oracle"):
        message = (
            "Nothing: no densifier"
            if name == "Sparse"
            else "Nothing: the exact kernel\nis given"
        )
        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=SECONDARY,
            fontsize=9,
        )
        ax.set_xticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return
    if name == "Coarse-data":
        spacing, label = FINE, "coarse window: 2dt -> dt"
    else:
        spacing, label = (
            FINE // 2,
            "fine window: dt -> dt/2\n(+ dt/2 -> dt/4, dt/4 -> dt/8)",
        )
    points = path[: 8 * spacing + 1 : spacing]
    steps = np.arange(points.size) * spacing / FINE
    ax.plot(
        np.arange(8 * spacing + 1) / FINE,
        path[: 8 * spacing + 1],
        color=AXIS,
        linewidth=1.5,
    )
    ax.plot(steps[::2], points[::2], "o", color=INK, ms=5, mec=SURFACE, mew=1)
    ax.plot(steps[1::2], points[1::2], "o", mfc=SURFACE, mec=colour, ms=6, mew=1.6)
    ax.set_title(label, loc="left", fontsize=8, color=SECONDARY)


def _coarse(ax, steps, window):
    ax.plot(steps[:HISTORY], window[:HISTORY], "o", color=INK, ms=5, mec=SURFACE, mew=1)
    ax.plot(steps[HISTORY:], window[HISTORY:], "o", mfc=SURFACE, mec=INK, ms=6, mew=1.4)


def _interval(ax, x, draws, key):
    low, mid, high = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    ax.vlines(x, low, high, color=COLOURS[key], linewidth=2.5)
    ax.plot(x, mid, "o", color=COLOURS[key], ms=4, mec=SURFACE, mew=1)


def plot_predictions(plt, output):
    """x2 forecasts and the exact GP on one test window per ratio."""
    figure, axes = plt.subplots(
        len(PREDICTION_RATIOS), 1, figsize=(10, 2.5 * len(PREDICTION_RATIOS))
    )
    for ax, ratio in zip(axes, PREDICTION_RATIOS, strict=True):
        results, data = load(output, ratio)
        truth, window = data["truth"][WINDOW], data["windows"][WINDOW]
        ax.plot(np.arange(truth.size) / FINE, truth, color=AXIS, linewidth=2)
        steps = np.arange(window.size)
        _coarse(ax, steps, window)
        ax.axvspan(HISTORY - 1, steps[-1] + 0.6, color=GRID, alpha=0.5)
        width = 0.11
        for index, key in enumerate(PREDICTED):
            offset = (index - (len(PREDICTED) - 1) / 2) * width
            _interval(ax, steps[HISTORY:] + offset, data[key][WINDOW], key)
        gp = results["forecast"]["gp"]["crps"]
        scores = ", ".join(
            f"{SHORT[key]} {results['forecast'][key]['crps'] / gp:.2f}"
            for key in PREDICTED[:-1]
        )
        ax.set_title(
            f"P/dt = {ratio}    CRPS / exact GP over all windows: {scores}",
            loc="left",
            fontsize=9,
        )
        ax.set_xlim(-0.3, steps[-1] + 0.6)
        ax.set_yticks([])
    axes[-1].set_xlabel("Time (coarse steps dt; shaded: forecast horizon)")
    handles = [
        plt.Line2D([], [], color=AXIS, linewidth=2, label="Fine truth"),
        plt.Line2D([], [], marker="o", color=INK, ls="none", label="Coarse history"),
        plt.Line2D(
            [], [], marker="o", mfc=SURFACE, mec=INK, ls="none", label="True future"
        ),
    ] + [
        plt.Line2D([], [], color=COLOURS[key], linewidth=2.5, label=f"{SHORT[key]}")
        for key in PREDICTED
    ]
    figure.legend(handles=handles, loc="lower center", ncols=4, frameon=False)
    figure.suptitle(
        f"Example forecasts: 90% intervals of 16 draws (test window {WINDOW}, "
        "not selected)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(output / "figure_predictions.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    main()
