"""Train PriSTI on alternate-point imputation of GP paths and evaluate it.

Defaults follow the CSDI comparison protocol in scripts/compare_csdi_training.py:
independent 33-point paths at spacing dt, split into two 17-point windows each;
512 training, 64 validation and 128 test paths from fixed data seeds; 200
epochs at batch size 64; a quadratic 50-step schedule to 0.5 with weight decay,
learning-rate decay and all-step validation every ten epochs; 32 draws. The
`exponential` task reproduces that protocol's data and test seeds exactly.
Hidden points alternate (2*dt -> dt). PriSTI is compared with the exact GP
posterior (the best possible imputer) and linear interpolation.

Run from the repository root:

    uv run --extra research --with matplotlib python \
        scripts/evaluate_pristi_imputation.py --output outputs/pristi

Writes metrics, plots, samples and checkpoints to the output directory.
"""

import argparse
import json
import logging
import time
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from densiffusion.architectures.pristi import PriSTIDenoiser, PriSTIEncoder
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.conditioning import prepare_targets
from densiffusion.data import make_windows
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import generate
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.objectives import noise_mse
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

TASKS = {
    "exponential": (
        "Exponential GP (CSDI task)",
        0.1,
        {"kernel": "exponential", "variance": 1.0, "lengthscale": 1.0},
    ),
    "brownian": ("Brownian motion", 0.01, {"kernel": "brownian", "variance": 1.0}),
    "fbm_h0.7": (
        "fBm, H = 0.7",
        0.01,
        {"kernel": "fractional_brownian", "variance": 1.0, "hurst": 0.7},
    ),
    "fbm_h0.3": (
        "fBm, H = 0.3",
        0.01,
        {"kernel": "fractional_brownian", "variance": 1.0, "hurst": 0.3},
    ),
    "matern52": (
        "Matérn 5/2",
        0.01,
        {"kernel": "matern_5_2", "variance": 1.0, "lengthscale": 0.1},
    ),
}
DATA_SEEDS = {"train": 22001, "validation": 22002, "test": 22003}
SAMPLE_SEED, GP_SEED = 31002, 31003
LEVELS = np.round(np.linspace(0.1, 0.9, 9), 2)

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
PRISTI, GP, INTERP = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--path-length", type=int, default=33)
    parser.add_argument("--window", type=int, default=17)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--train-paths", type=int, default=512)
    parser.add_argument("--validation-paths", type=int, default=64)
    parser.add_argument("--test-paths", type=int, default=128)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--schedule", default="quadratic")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-decay", type=float, nargs="*", default=[0.75, 0.9])
    parser.add_argument("--validation-steps", default="all")
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    config = training_config(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps({**vars(args), "output": str(args.output)}, indent=2)
    )
    _style()

    results = {}
    for name in args.tasks:
        results[name] = run(name, config, args)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
    write_table(results, args)
    plot_examples(results, args)
    plot_training(results, args)
    plot_crps(results, args)
    plot_calibration(results, args)
    print(f"Done. Results in {args.output}", flush=True)


def training_config(args):
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        beta_start=1e-4,
        beta_end=args.beta_end,
        schedule=args.schedule,
        weight_decay=args.weight_decay,
        lr_decay_fractions=tuple(args.lr_decay),
        validation_steps=args.validation_steps,
        validation_interval=args.validation_interval,
        device=args.device,
    )


def partition(name, count, dt, gp, args):
    """Generate independent paths for one partition, then window them."""
    params = {key: value for key, value in gp.items() if key != "kernel"}
    values = generate(
        n_series=count,
        n_steps=args.path_length,
        dt=dt,
        rng=np.random.default_rng(DATA_SEEDS[name]),
        kernel=gp["kernel"],
        mean=0.0,
        **params,
    )
    return make_windows(
        values,
        times=np.arange(args.path_length) * dt,
        window_length=args.window,
        stride=args.stride,
    )


def run(name, config, args):
    label, dt, gp = TASKS[name]
    folder = args.output / name
    folder.mkdir()
    train, _ = partition("train", args.train_paths, dt, gp, args)
    validation, _ = partition("validation", args.validation_paths, dt, gp, args)
    test, test_times = partition("test", args.test_paths, dt, gp, args)
    test, mask = prepare_targets(test)
    hidden = mask[0, :, 0]

    print(f"[{name}] training on {args.device}: {len(train)} windows", flush=True)
    epochs = _EpochLog()
    logger = logging.getLogger("densiffusion.training")
    logger.setLevel(logging.INFO)
    logger.addHandler(epochs)
    torch.manual_seed(args.seed)
    model = ImputationModel(PriSTIDenoiser(), encoder=PriSTIEncoder())
    start = time.perf_counter()
    try:
        train_densifier(
            model,
            train,
            validation,
            config=config,
            objective=noise_mse,
            dt=dt,
            seed=args.seed,
            checkpoint=folder / "model.pt",
        )
    finally:
        logger.removeHandler(epochs)
    training_seconds = time.perf_counter() - start

    print(f"[{name}] sampling PriSTI and the exact GP posterior", flush=True)
    start = time.perf_counter()
    options = dict(factor=1, mask=mask, times=test_times, n_samples=args.draws, dt=dt)
    pristi = impute(test, DDPMSampler(model, config), seed=SAMPLE_SEED, **options)
    sampling_seconds = time.perf_counter() - start
    posterior = partial(gp_posterior, mean=0.0, **gp)
    exact_gp = impute(test, GPPosteriorSampler(posterior), seed=GP_SEED, **options)
    interpolation = test.copy()
    interpolation[:, 1::2] = (test[:, :-1:2] + test[:, 2::2]) / 2
    np.savez_compressed(
        folder / "samples.npz",
        test=test,
        times=test_times,
        mask=mask,
        pristi=pristi,
        gp=exact_gp,
    )

    windows_per_path = len(test) // args.test_paths
    summary = {
        "label": label,
        "dt": dt,
        "gp": gp,
        "pristi": score(pristi, test, hidden, windows_per_path),
        "gp_posterior": score(exact_gp, test, hidden, windows_per_path),
        "interpolation": score(interpolation[:, None], test, hidden, windows_per_path),
        "exact": exact_posterior(test, test_times, hidden, posterior),
        "epochs": epochs.records,
        "training_seconds": training_seconds,
        "sampling_seconds": sampling_seconds,
    }
    pristi_crps, gp_crps, interpolation_crps = (
        summary[key]["crps"] for key in ("pristi", "gp_posterior", "interpolation")
    )
    summary["gap_closed"] = (interpolation_crps - pristi_crps) / (
        interpolation_crps - gp_crps
    )
    print(
        f"[{name}] CRPS PriSTI {pristi_crps:.4f}, exact GP {gp_crps:.4f}, "
        f"interpolation {interpolation_crps:.4f} "
        f"({training_seconds / 60:.1f} min training, "
        f"{sampling_seconds / 60:.1f} min sampling)",
        flush=True,
    )
    return summary


def score(samples, test, hidden, windows_per_path):
    """Hidden-point scores; samples are (windows, draws, time, channels).

    Standard errors treat each path (its adjacent windows together) as one
    independent unit.
    """
    windows = np.array(
        [
            crps(samples[i : i + 1], test[i : i + 1], mask=hidden)
            for i in range(len(test))
        ]
    )
    paths = windows.reshape(-1, windows_per_path).mean(axis=1)
    draws = samples[:, :, hidden, 0]
    truth = test[:, hidden, 0]
    median = np.median(draws, axis=1)
    result = {
        "crps": float(windows.mean()),
        "crps_se": float(paths.std(ddof=1) / np.sqrt(len(paths))),
        "median_mae": float(np.abs(median - truth).mean()),
        "median_rmse": float(np.sqrt(((median - truth) ** 2).mean())),
    }
    if draws.shape[1] < 2:
        return result
    coverage = []
    for level in LEVELS:
        lower = np.quantile(draws, (1 - level) / 2, axis=1)
        upper = np.quantile(draws, (1 + level) / 2, axis=1)
        coverage.append(float(((truth >= lower) & (truth <= upper)).mean()))
    centred = draws - draws.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1)
    neighbours = (centred[:, :, :-1] * centred[:, :, 1:]).mean(axis=1)
    result.update(
        coverage=dict(zip(map(str, LEVELS), coverage, strict=True)),
        rms_sd=float(np.sqrt(np.mean(draws.var(axis=1, ddof=1)))),
        neighbour_correlation=float(
            (neighbours / (spread[:, :-1] * spread[:, 1:])).mean()
        ),
    )
    return result


def exact_posterior(test, times, hidden, posterior):
    """Exact posterior spread, neighbour correlation and mean error per window."""
    variances, correlations, errors = [], [], []
    for values, window_times in zip(test[..., 0], times, strict=True):
        mean, covariance = posterior(
            window_times[~hidden], values[~hidden], window_times[hidden]
        )
        sd = np.sqrt(np.clip(np.diag(covariance), 1e-300, None))
        variances.append(sd**2)
        correlations.append(np.diag(covariance / np.outer(sd, sd), 1))
        errors.append(mean - values[hidden])
    errors = np.array(errors)
    return {
        "rms_sd": float(np.sqrt(np.mean(variances))),
        "neighbour_correlation": float(np.mean(correlations)),
        "mean_mae": float(np.abs(errors).mean()),
        "mean_rmse": float(np.sqrt((errors**2).mean())),
    }


def write_table(results, args):
    lines = [
        "# PriSTI alternate-point imputation",
        "",
        f"{args.train_paths}/{args.validation_paths}/{args.test_paths} "
        f"train/validation/test paths of {args.path_length} points, windows of "
        f"{args.window} (stride {args.stride}); alternate interior points hidden. "
        f"{args.epochs} epochs, batch {args.batch_size}, {args.diffusion_steps} "
        f"{args.schedule} steps to {args.beta_end}; {args.draws} draws; "
        f"seed {args.seed}; device {args.device}.",
        "",
        "| Task | CRPS PriSTI | CRPS exact GP | CRPS interpolation | Gap closed "
        "| 90% coverage PriSTI | 90% coverage GP | Spread ratio "
        "| Neighbour corr. PriSTI | Neighbour corr. exact | Train min |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results.values():
        pristi, gp, exact = result["pristi"], result["gp_posterior"], result["exact"]
        lines.append(
            f"| {result['label']} | {pristi['crps']:.4f} ± {pristi['crps_se']:.4f} "
            f"| {gp['crps']:.4f} ± {gp['crps_se']:.4f} "
            f"| {result['interpolation']['crps']:.4f} "
            f"| {result['gap_closed']:.0%} "
            f"| {pristi['coverage']['0.9']:.1%} | {gp['coverage']['0.9']:.1%} "
            f"| {pristi['rms_sd'] / exact['rms_sd']:.2f} "
            f"| {pristi['neighbour_correlation']:+.2f} "
            f"| {exact['neighbour_correlation']:+.2f} "
            f"| {result['training_seconds'] / 60:.1f} |"
        )
    lines += [
        "",
        "Gap closed: share of the CRPS gap between linear interpolation and the "
        "exact GP posterior that PriSTI closes. Spread ratio: PriSTI's RMS "
        "predictive standard deviation divided by the exact posterior's. "
        "Neighbour correlation: posterior correlation between adjacent hidden "
        "points. ± is one standard error across test paths. Coverage from "
        f"{args.draws} draws is slightly below nominal even for the exact "
        "posterior.",
    ]
    (args.output / "results.md").write_text("\n".join(lines) + "\n")


def plot_examples(results, args):
    names = list(results)
    figure, axes = plt.subplots(
        len(names),
        2,
        figsize=(13, 2.9 * len(names)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.4, 1]},
    )
    for row, name in zip(axes, names, strict=True):
        data = np.load(args.output / name / "samples.npz")
        test = data["test"][0, :, 0]
        _window(row[0], test, data["pristi"][0, :, :, 0])
        row[0].set_title(f"{results[name]['label']}: test window 1", loc="left")
        _intervals(row[1], test, data["pristi"][0, :, :, 0], data["gp"][0, :, :, 0])
        row[1].set_title("90% intervals at hidden points", loc="left")
    _legend(
        figure,
        [
            ("observed", "Observed"),
            ("truth", "True hidden value"),
            ("median", "PriSTI median"),
            ("band", "PriSTI 90% interval"),
            ("sample", "PriSTI samples (3)"),
            ("gp", "Exact GP 90% interval"),
        ],
    )
    figure.suptitle(
        "PriSTI imputations of the first test window per task (not selected)",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(args.output / "examples.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(names), 3, figsize=(15, 2.7 * len(names)), squeeze=False
    )
    for row, name in zip(axes, names, strict=True):
        data = np.load(args.output / name / "samples.npz")
        for column, ax in enumerate(row, start=1):
            _window(ax, data["test"][column, :, 0], data["pristi"][column, :, :, 0])
            ax.set_title(
                f"{results[name]['label']}: test window {column + 1}", loc="left"
            )
    _legend(
        figure,
        [
            ("observed", "Observed"),
            ("truth", "True hidden value"),
            ("median", "PriSTI median"),
            ("band", "PriSTI 90% interval"),
            ("sample", "PriSTI samples (3)"),
        ],
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(args.output / "examples_more.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def _window(ax, test, samples):
    steps = np.arange(len(test))
    observed = steps % 2 == 0
    lower, median, upper = np.quantile(samples, [0.05, 0.5, 0.95], axis=0)
    ax.fill_between(steps, lower, upper, color=PRISTI, alpha=0.15, linewidth=0)
    for draw in samples[:3]:
        ax.plot(steps, draw, color=PRISTI, linewidth=0.8, alpha=0.45)
    ax.plot(steps, median, color=PRISTI, linewidth=2)
    ax.plot(steps[~observed], test[~observed], "o", mfc=SURFACE, mec=INK, ms=5, mew=1.2)
    ax.plot(steps[observed], test[observed], "o", color=INK, ms=5, mec=SURFACE, mew=1)
    ax.set_xlabel("Time step (fine grid)")


def _intervals(ax, test, pristi, gp):
    steps = np.arange(len(test))
    hidden = steps % 2 == 1
    for samples, colour, offset in ((pristi, PRISTI, -0.18), (gp, GP, 0.18)):
        lower, median, upper = np.quantile(
            samples[:, hidden], [0.05, 0.5, 0.95], axis=0
        )
        x = steps[hidden] + offset
        ax.vlines(x, lower, upper, color=colour, linewidth=2.5)
        ax.plot(x, median, "o", color=colour, ms=4, mec=SURFACE, mew=1)
    ax.plot(steps[hidden], test[hidden], "o", mfc=SURFACE, mec=INK, ms=5, mew=1.2)
    ax.plot(steps[~hidden], test[~hidden], "o", color=INK, ms=5, mec=SURFACE, mew=1)
    ax.set_xlabel("Time step (fine grid)")


def plot_training(results, args):
    names = list(results)
    figure, axes = plt.subplots(
        1, len(names), figsize=(3.4 * len(names), 3.3), squeeze=False
    )
    for ax, name in zip(axes[0], names, strict=True):
        records = results[name]["epochs"]
        epochs = [record["epoch"] for record in records]
        ax.plot(
            epochs,
            [record["train_loss"] for record in records],
            color=MUTED,
            linewidth=2,
            label="Training",
        )
        validated = [record for record in records if "validation_loss" in record]
        x = [record["epoch"] for record in validated]
        y = [record["validation_loss"] for record in validated]
        ax.plot(x, y, "o-", color=PRISTI, linewidth=2, ms=4, label="Validation")
        best = int(np.argmin(y))
        ax.plot(x[best], y[best], "o", color=PRISTI, ms=8, mec=SURFACE, mew=2)
        ax.annotate(
            f"best: epoch {x[best]:.0f}",
            (x[best], y[best]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color=SECONDARY,
            fontsize=8,
        )
        ax.set_yscale("log")
        ax.set_title(results[name]["label"], loc="left")
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Noise MSE at hidden points")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        "Training and validation loss (best validation weights kept)",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(args.output / "training_curves.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_crps(results, args):
    names = list(results)
    x = np.arange(len(names))
    ratios = {
        key: [
            results[name][key]["crps"] / results[name]["gp_posterior"]["crps"]
            for name in names
        ]
        for key in ("pristi", "interpolation")
    }
    figure, ax = plt.subplots(figsize=(1.9 * len(names) + 2, 4.2))
    for offset, key, colour, label in (
        (-0.15, "pristi", PRISTI, "PriSTI"),
        (0.15, "interpolation", INTERP, "Linear interpolation"),
    ):
        bars = ax.bar(x + offset, ratios[key], width=0.28, color=colour, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, color=SECONDARY, fontsize=9)
    ax.axhline(1, color=GP, linewidth=2, label="Exact GP posterior (optimal)")
    ax.set_xticks(x, [results[name]["label"] for name in names])
    ax.set_ylabel("CRPS relative to exact GP posterior")
    ax.set_ylim(0, max(max(values) for values in ratios.values()) * 1.18)
    ax.legend(frameon=False, loc="upper left", ncols=3)
    ax.set_title(
        "CRPS at hidden test points (lower is better; 1 = best possible)", loc="left"
    )
    figure.tight_layout()
    figure.savefig(args.output / "crps.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_calibration(results, args):
    names = list(results)
    figure, axes = plt.subplots(
        1, len(names), figsize=(3.2 * len(names), 3.6), squeeze=False
    )
    for ax, name in zip(axes[0], names, strict=True):
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        for key, colour, label in (
            ("pristi", PRISTI, "PriSTI"),
            ("gp_posterior", GP, "Exact GP"),
        ):
            coverage = list(results[name][key]["coverage"].values())
            ax.plot(
                LEVELS,
                coverage,
                "o-",
                color=colour,
                linewidth=2,
                ms=5,
                mec=SURFACE,
                mew=1,
                label=label,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(results[name]["label"], loc="left")
        ax.set_xlabel("Nominal interval level")
    axes[0, 0].set_ylabel("Empirical coverage")
    axes[0, 0].legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Calibration of central prediction intervals (diagonal = perfect)",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(args.output / "calibration.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def _legend(figure, entries):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = {
        "observed": Line2D([], [], marker="o", ls="", color=INK, ms=5),
        "truth": Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK, ms=5, mew=1.2),
        "median": Line2D([], [], color=PRISTI, lw=2),
        "band": Patch(color=PRISTI, alpha=0.15),
        "sample": Line2D([], [], color=PRISTI, lw=0.8, alpha=0.45),
        "gp": Line2D([], [], color=GP, lw=2.5),
    }
    figure.legend(
        [handles[key] for key, _ in entries],
        [label for _, label in entries],
        loc="lower center",
        ncols=len(entries),
        frameon=False,
    )


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "legend.labelcolor": SECONDARY,
            "font.family": "sans-serif",
        }
    )


class _EpochLog(logging.Handler):
    """Collect the trainer's per-epoch key=value log lines."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("epoch="):
            fields = dict(item.split("=") for item in message.split())
            self.records.append({key: float(value) for key, value in fields.items()})


if __name__ == "__main__":
    main()
