"""Example predictions with uncertainty bands for each PriSTI experiment.

On one canonical task per experiment, shade each model's 90% interval at the
filled-in points between observations and over the forecast horizon, and
overlay the exact GP posterior's 90% interval given the same observations.
Panels show the first test windows (not selected).

- Imputation: the saved evaluation draws of evaluate_pristi_imputation.py.
- Forecasting: the saved checkpoints of run_pristi_densiffusion.py, resampled
  for the shown windows with one future per densified history, which samples
  the same predictive distribution as the benchmark's histories x futures.

    uv run --extra research --with matplotlib python \
        scripts/plot_pristi_uncertainty.py
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from run_pristi_densiffusion import (
    COLOURS,
    HISTORY,
    HORIZON,
    INK,
    MUTED,
    NAMES,
    SECONDARY,
    SURFACE,
    TASKS,
    _style,
    config,
    exact_history,
    forecast_windows,
    generate,
    kernel_params,
)

from densiffusion.architectures.pristi_densiffusion import (
    densify,
    forecast,
    interpolate,
    sampling_config,
    small_pristi,
)
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.gp.posterior import gp_posterior

Z90 = 1.6449
DENSIFIED_COLUMNS = ("sparse", "learned_x2", "learned_x8", "interp_x8")
QUASI_FIGURES = {
    "x2": ("sparse", "learned_x2", "fine_x2", "oracle_x2"),
    "x8": ("interp_x8", "learned_x8", "fine_x8", "oracle_x8"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--imputation", type=Path, default=Path("outputs/pristi"))
    parser.add_argument("--imputation-task", default="brownian")
    parser.add_argument(
        "--forecasting", type=Path, default=Path("outputs/densiffusion")
    )
    parser.add_argument("--forecasting-task", default="brownian")
    parser.add_argument(
        "--quasiperiodic", type=Path, default=Path("outputs/densiffusion_quasiperiodic")
    )
    parser.add_argument("--quasiperiodic-task", default="qp_2.5")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _style(plt)
    torch.set_num_threads(2)
    plot_imputation(plt, args)
    plot_forecasts(
        plt,
        args,
        args.forecasting,
        args.forecasting_task,
        DENSIFIED_COLUMNS,
        "uncertainty_examples.png",
    )
    for suffix, columns in QUASI_FIGURES.items():
        plot_forecasts(
            plt,
            args,
            args.quasiperiodic,
            args.quasiperiodic_task,
            columns,
            f"uncertainty_examples_{suffix}.png",
        )


def plot_imputation(plt, args):
    """Alternate-point imputation: PriSTI's draws against the exact posterior."""
    from evaluate_pristi_imputation import TASKS as IMPUTATION_TASKS

    label, _, gp = IMPUTATION_TASKS[args.imputation_task]
    data = np.load(args.imputation / args.imputation_task / "samples.npz")
    hidden = data["mask"][0, :, 0]
    posterior = partial(gp_posterior, mean=0.0, **gp)
    figure, axes = plt.subplots(
        args.windows, 1, figsize=(10, 2.6 * args.windows), squeeze=False
    )
    for index, ax in enumerate(axes[:, 0]):
        values, times = data["test"][index, :, 0], data["times"][index]
        draws = data["pristi"][index, :, :, 0]
        steps = np.arange(values.size)
        mean, covariance = posterior(times[~hidden], values[~hidden], times)
        sd = np.sqrt(np.clip(np.diag(covariance), 0, None))
        _band(ax, steps, draws, COLOURS["pristi"])
        _truth(ax, steps, values)
        _gp_band(ax, steps, mean, sd)
        ax.plot(
            steps[~hidden], values[~hidden], "o", color=INK, ms=5, mec=SURFACE, mew=1
        )
        ax.plot(steps[hidden], values[hidden], "o", mfc=SURFACE, mec=INK, ms=5, mew=1.2)
        ax.set_ylabel(f"Test window {index}")
        ax.set_xlim(steps[0] - 0.3, steps[-1] + 0.3)
    axes[-1, 0].set_xlabel("Time step (every other point hidden)")
    draws = data["pristi"].shape[1]
    _legend(
        plt,
        figure,
        [
            _line(plt, MUTED, "True path"),
            _dot(plt, "Observed", filled=True),
            _dot(plt, "True hidden value", filled=False),
            _patch(plt, COLOURS["pristi"], f"PriSTI 90% interval ({draws} draws)"),
            _line(plt, COLOURS["pristi"], "PriSTI median"),
            _line(plt, COLOURS["gp"], "Exact GP posterior 90% interval", "--"),
        ],
        ncols=3,
    )
    figure.suptitle(
        f"Experiment 1, {label}: PriSTI fills the hidden half-way points",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.97))
    output = args.imputation / "uncertainty_examples.png"
    figure.savefig(output, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"Wrote {output}")


def plot_forecasts(plt, args, folder, task, columns, filename):
    """Rows are test windows, columns methods; history and forecast bands."""
    label, dt, gp = TASKS[task]
    paths, times = generate(task, 64, "test")
    windows, truth, window_times = forecast_windows(paths, times)
    shown = np.arange(args.windows)
    history = windows[shown, :HISTORY]
    results = json.loads((folder / task / "results.json").read_text())
    gp_crps = results["forecast"]["gp"]["crps"]
    params = kernel_params(gp)
    posterior = partial(gp_posterior, mean=0.0, **params)
    models = Models(folder / task, GPPosteriorSampler(posterior), args)
    figure, axes = plt.subplots(
        args.windows,
        len(columns),
        figsize=(4.1 * len(columns), 2.7 * args.windows + 0.6),
        squeeze=False,
        sharey="row",
    )
    envelope = np.full((args.windows, 2), [np.inf, -np.inf])
    for column, method in enumerate(columns):
        factor = 1 if method == "sparse" else int(method.split("_x")[1])
        paths_drawn = models.sample(method, history, truth[shown], window_times[shown])
        grid = np.arange((HISTORY + HORIZON - 1) * factor + 1) / factor
        for row, index in enumerate(shown):
            ax = axes[row, column]
            mean, covariance = posterior(
                window_times[index, :HISTORY],
                history[row],
                window_times[index, 0] + grid * dt,
            )
            sd = np.sqrt(np.clip(np.diag(covariance), 0, None))
            fine_steps = np.linspace(0, grid[-1], truth.shape[1])
            _band(ax, grid, paths_drawn[row], COLOURS[method])
            _truth(ax, fine_steps, truth[index])
            _gp_band(ax, grid, mean, sd)
            envelope[row] = (
                min(envelope[row, 0], (mean - Z90 * sd).min(), truth[index].min()),
                max(envelope[row, 1], (mean + Z90 * sd).max(), truth[index].max()),
            )
            coarse = np.arange(windows.shape[1])
            ax.axvline(HISTORY - 1, color=SECONDARY, linewidth=0.8, linestyle=":")
            ax.plot(
                coarse[:HISTORY],
                windows[index, :HISTORY],
                "o",
                color=INK,
                ms=4.5,
                mec=SURFACE,
                mew=1,
            )
            ax.plot(
                coarse[HISTORY:],
                windows[index, HISTORY:],
                "o",
                mfc=SURFACE,
                mec=INK,
                ms=5,
                mew=1.2,
            )
            ax.set_xlim(-0.3, grid[-1] + 0.3)
            if column == 0:
                ax.set_ylabel(f"Test window {index}")
        ratio = results["forecast"][method]["crps"] / gp_crps
        axes[0, column].set_title(
            f"{NAMES[method]}\nforecast CRPS {ratio:.2f}x exact GP (all windows)",
            loc="left",
        )
    for (low, high), row in zip(envelope, axes, strict=True):
        pad = 0.15 * (high - low)
        row[0].set_ylim(low - pad, high + pad)
    for ax in axes[-1]:
        ax.set_xlabel("Time (coarse steps; dotted: forecast origin)")
    _legend(
        plt,
        figure,
        [
            _line(plt, MUTED, "True path"),
            _dot(plt, "Observed history", filled=True),
            _dot(plt, "True future", filled=False),
            _patch(
                plt, SECONDARY, f"Model 90% interval and median ({args.draws} draws)"
            ),
            _line(plt, COLOURS["gp"], "Exact GP posterior 90% interval", "--"),
        ],
    )
    figure.suptitle(
        f"{label}: uncertainty at filled-in history points and over the forecast "
        "(model colour per column; wider bands run off the panel)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(folder / filename, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"Wrote {folder / filename}")


class Models:
    """Reload a task's checkpoints and sample (history, future) paths."""

    def __init__(self, folder, gp_sampler, args):
        self.folder, self.gp_sampler, self.args = folder, gp_sampler, args
        self.sampler = sampling_config(config(1))
        self.rng = np.random.default_rng(args.seed)

    def load(self, name):
        model = small_pristi()
        state = torch.load(
            self.folder / f"{name}.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state["model_state_dict"])
        return model.eval()

    def seed(self):
        return int(self.rng.integers(2**62))

    def sample(self, method, history, truth, times):
        """(windows, draws, points) on the method's grid: history then future."""
        draws = self.args.draws
        if method == "sparse":
            histories = np.repeat(history[:, None], draws, axis=1)
            factor, checkpoint = 1, "sparse"
        else:
            kind, factor = method.split("_x")
            factor = int(factor)
            level = factor.bit_length() - 1
            checkpoint = f"truefine_x{factor}" if kind in ("fine", "oracle") else method
            if kind == "learned":
                histories = densify(
                    self.load("imputer"),
                    self.sampler,
                    history,
                    levels=level,
                    n_samples=draws,
                    seed=self.seed(),
                )[-1]
            elif kind == "fine":
                histories = densify(
                    [self.load(f"imputer_fine_level{k}") for k in range(1, level + 1)],
                    self.sampler,
                    history,
                    levels=level,
                    n_samples=draws,
                    seed=self.seed(),
                )[-1]
            elif kind == "oracle":
                histories = exact_history(
                    self.gp_sampler, truth, times, factor, draws, self.seed()
                )
            else:
                histories = np.repeat(
                    interpolate(history, factor)[:, None], draws, axis=1
                )
        windows, _, length = histories.shape
        futures = forecast(
            self.load(checkpoint),
            self.sampler,
            histories.reshape(-1, length),
            horizon=HORIZON,
            factor=factor,
            n_samples=1,
            seed=self.seed(),
            downsample=False,
        ).reshape(windows, draws, -1)
        return np.concatenate([histories, futures], axis=2)


def _band(ax, x, draws, colour):
    low, median, high = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    ax.fill_between(x, low, high, color=colour, alpha=0.45, linewidth=0, zorder=2)
    ax.plot(x, median, color=colour, linewidth=1.2, zorder=2.5)


def _truth(ax, x, values):
    ax.plot(x, values, color=MUTED, linewidth=1.1, zorder=2.6)


def _gp_band(ax, x, mean, sd):
    for edge in (mean - Z90 * sd, mean + Z90 * sd):
        ax.plot(x, edge, "--", color=COLOURS["gp"], linewidth=1.4, zorder=3)


def _line(plt, colour, label, style="-"):
    return plt.Line2D([], [], color=colour, linewidth=1.6, linestyle=style, label=label)


def _dot(plt, label, *, filled):
    if filled:
        return plt.Line2D(
            [], [], marker="o", color=INK, mec=SURFACE, ls="none", label=label
        )
    return plt.Line2D(
        [], [], marker="o", mfc=SURFACE, mec=INK, mew=1.2, ls="none", label=label
    )


def _patch(plt, colour, label):
    from matplotlib.patches import Patch

    return Patch(color=colour, alpha=0.45, linewidth=0, label=label)


def _legend(plt, figure, handles, ncols=None):
    figure.legend(
        handles=handles,
        loc="lower center",
        ncols=ncols or len(handles),
        frameon=False,
    )


if __name__ == "__main__":
    main()
