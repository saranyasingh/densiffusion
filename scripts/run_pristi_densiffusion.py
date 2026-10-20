"""Benchmark densified vs sparse PriSTI forecasting on GP paths.

For each task, fine ground-truth paths are generated at spacing dt / 8 and
observed every 8th point (spacing dt). Everything is trained on the coarse
observations only:

1. an imputer trained at strides 1, 2 and 4 of the observations (2*dt -> dt,
   4*dt -> 2*dt, 8*dt -> 4*dt) and applied three times (x8);
2. forecasters from densified history to densified future at x2, x4 and x8,
   plus x8 with linear interpolation instead of the imputer;
3. a forecaster from the sparse history to the sparse future.

Forecasts are scored at the coarse future times against the exact GP forecast
(the best possible) and repeating the last observation. Run one task per
process, then combine:

    uv run --extra research python scripts/run_pristi_densiffusion.py run \
        --task brownian --output outputs/densiffusion
    uv run --extra research --with matplotlib python \
        scripts/run_pristi_densiffusion.py report --output outputs/densiffusion
"""

import argparse
import json
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.pristi_densiffusion import (
    densify,
    forecast,
    forecast_layout,
    imputation_windows,
    interpolate,
    refine,
    sampling_config,
    small_pristi,
    train_forecaster,
    train_imputer,
)
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import sample_gp
from densiffusion.pipeline import impute
from densiffusion.training import TrainingConfig

TASKS = {
    "brownian": ("Brownian motion", 0.04, {"kernel": "brownian", "variance": 1.0}),
    "fbm_h0.7": (
        "fBm, H = 0.7",
        0.04,
        {"kernel": "fractional_brownian", "variance": 1.0, "hurst": 0.7},
    ),
    "fbm_h0.3": (
        "fBm, H = 0.3",
        0.04,
        {"kernel": "fractional_brownian", "variance": 1.0, "hurst": 0.3},
    ),
    "matern52": (
        "Matérn 5/2",
        0.25,
        {"kernel": "matern_5_2", "variance": 1.0, "lengthscale": 1.0},
    ),
}
LEVELS, FINE = 3, 8
COARSE_POINTS = 33
HISTORY, HORIZON = 9, 4
WINDOW_STARTS = (0, 20)
DATA_SEEDS = {"train": 22001, "validation": 22002, "test": 22003}
HISTORIES, FUTURES = 4, 4
DRAWS = HISTORIES * FUTURES
METHODS = (
    "persistence",
    "sparse",
    "learned_x2",
    "learned_x4",
    "learned_x8",
    "interp_x8",
)
LEVELS_TESTED = (-2, -1, 0, 1, 2, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("run", "report"))
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-paths", type=int, default=256)
    parser.add_argument("--validation-paths", type=int, default=32)
    parser.add_argument("--test-paths", type=int, default=64)
    parser.add_argument("--imputer-epochs", type=int, default=60)
    parser.add_argument("--forecaster-epochs", type=int, default=150)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rescore", action="store_true", help="Reuse checkpoints")
    args = parser.parse_args()
    if args.mode == "run":
        torch.set_num_threads(args.threads)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        run_task(args)
    else:
        report(args)


def config(epochs):
    return TrainingConfig(
        epochs=epochs,
        batch_size=64,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=0.5,
        schedule="quadratic",
        weight_decay=1e-6,
        lr_decay_fractions=(0.75, 0.9),
        validation_steps="all",
        validation_interval=10,
        device="cpu",
    )


def generate(task, count, part):
    _, dt, gp = TASKS[task]
    params = {key: value for key, value in gp.items() if key != "kernel"}
    times = np.arange((COARSE_POINTS - 1) * FINE + 1) * dt / FINE
    paths = sample_gp(
        times, gp["kernel"], rng=DATA_SEEDS[part], n_samples=count, **params
    )
    return paths, times


def forecast_windows(fine, times):
    """Coarse windows, their fine truth, and absolute coarse times."""
    length = HISTORY + HORIZON - 1
    coarse = fine[:, ::FINE]
    windows = np.concatenate([coarse[:, s : s + length + 1] for s in WINDOW_STARTS])
    truth = np.concatenate(
        [fine[:, FINE * s : FINE * (s + length) + 1] for s in WINDOW_STARTS]
    )
    window_times = np.concatenate(
        [
            np.broadcast_to(times[::FINE][s : s + length + 1], (len(fine), length + 1))
            for s in WINDOW_STARTS
        ]
    )
    return windows, truth, window_times


def run_task(args):
    task = args.task
    label, dt, gp = TASKS[task]
    folder = args.output / task
    folder.mkdir(parents=True, exist_ok=args.rescore)
    previous = folder / "results.json"
    previous = json.loads(previous.read_text()) if args.rescore else {}
    logger = logging.getLogger("densiffusion.training")
    timings = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timings[name] = now - clock
        clock = now
        print(f"[{task}] {name}: {timings[name] / 60:.1f} min", flush=True)

    fine = {
        part: generate(task, count, part)
        for part, count in (
            ("train", args.train_paths),
            ("validation", args.validation_paths),
            ("test", args.test_paths),
        )
    }
    coarse = {part: paths[:, ::FINE] for part, (paths, _) in fine.items()}
    posterior = partial(gp_posterior, mean=0.0, **gp)
    gp_sampler = GPPosteriorSampler(posterior)
    rng = np.random.default_rng(args.seed)

    def seed():
        return int(rng.integers(2**62))

    def fit(model, checkpoint, train):
        """Train, or with --rescore load the saved best weights instead."""
        if args.rescore:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            return None
        log = _EpochLog()
        logger.addHandler(log)
        try:
            train()
        finally:
            logger.removeHandler(log)
        return log.records

    imputer_config = config(args.imputer_epochs)
    sampler = sampling_config(imputer_config)
    torch.manual_seed(args.seed)
    imputer = small_pristi()
    imputer_epochs = fit(
        imputer,
        folder / "imputer.pt",
        lambda: train_imputer(
            imputer,
            imputation_windows(coarse["train"]),
            imputation_windows(coarse["validation"]),
            imputer_config,
            seed=args.seed,
            checkpoint=folder / "imputer.pt",
        ),
    )
    lap("imputer training")

    results = {"task": task, "label": label, "dt": dt, "gp": gp}
    results["scale_transfer"] = scale_transfer(
        imputer, sampler, fine["test"], gp_sampler, seed
    )
    lap("scale-transfer diagnostics")

    windows = {
        part: forecast_windows(paths, times) for part, (paths, times) in fine.items()
    }
    train_windows, _, _ = windows["train"]
    validation_windows, _, _ = windows["validation"]
    test_windows, test_truth, test_times = windows["test"]
    test_history = test_windows[:, :HISTORY]
    future = test_windows[:, HISTORY:]

    dense_train = densify(
        imputer, sampler, train_windows, levels=LEVELS, n_samples=1, seed=seed()
    )
    dense_validation = densify(
        imputer, sampler, validation_windows, levels=LEVELS, n_samples=1, seed=seed()
    )
    dense_test = densify(
        imputer, sampler, test_history, levels=LEVELS, n_samples=HISTORIES, seed=seed()
    )
    lap("densification")
    results["densified_history"] = history_quality(
        dense_test, test_history, test_truth, test_times, gp_sampler, seed
    )

    hidden, _ = forecast_layout(HISTORY, HORIZON)
    oracle = impute(
        test_windows[..., None],
        gp_sampler,
        factor=1,
        mask=hidden,
        times=test_times,
        n_samples=DRAWS,
        dt=dt,
        seed=seed(),
    )[:, :, HISTORY:, 0]
    persistence = np.repeat(test_history[:, -1:, None], HORIZON, axis=2)
    samples = {"gp": oracle, "persistence": persistence}

    forecaster_config = config(args.forecaster_epochs)
    forecaster_sampler = sampling_config(forecaster_config)
    variants = {"sparse": (1, train_windows, validation_windows, test_history, 1)}
    for level in range(1, LEVELS + 1):
        factor = 2**level
        variants[f"learned_x{factor}"] = (
            factor,
            dense_train[level - 1][:, 0],
            dense_validation[level - 1][:, 0],
            dense_test[level - 1].reshape(-1, dense_test[level - 1].shape[-1]),
            HISTORIES,
        )
    factor = 2**LEVELS
    variants[f"interp_x{factor}"] = (
        factor,
        interpolate(train_windows, factor),
        interpolate(validation_windows, factor),
        interpolate(test_history, factor),
        1,
    )
    epochs = {"imputer": imputer_epochs}
    for name, (factor, train, validation, histories, per_window) in variants.items():
        torch.manual_seed(args.seed)
        model = small_pristi()
        epochs[name] = fit(
            model,
            folder / f"{name}.pt",
            partial(
                train_forecaster,
                model,
                train,
                validation,
                forecaster_config,
                history=HISTORY,
                horizon=HORIZON,
                factor=factor,
                seed=args.seed,
                checkpoint=folder / f"{name}.pt",
            ),
        )
        draws = DRAWS // per_window
        predicted = forecast(
            model,
            forecaster_sampler,
            histories,
            horizon=HORIZON,
            factor=factor,
            n_samples=draws,
            seed=seed(),
        )
        samples[name] = predicted.reshape(len(test_windows), DRAWS, HORIZON)
        lap(f"forecaster {name}")

    results["forecast"] = score_forecasts(samples, future, args.test_paths)
    if args.rescore:
        results["epochs"] = previous["epochs"]
        results["timings"] = previous["timings"]
        results["rescore_timings"] = timings
    else:
        results["epochs"] = epochs
        results["timings"] = timings
    (folder / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        folder / "samples.npz",
        truth=test_truth,
        windows=test_windows,
        times=test_times,
        densified=dense_test[-1],
        **samples,
    )
    print(f"[{task}] done", flush=True)


def scale_transfer(imputer, sampler, fine_test, gp_sampler, seed):
    """One-step refinement from true data at each spacing dt * 2**-level."""
    paths, times = fine_test
    rows = {}
    alternate = np.arange(2 * 5 - 1) % 2 == 1
    for level in LEVELS_TESTED:
        stride = FINE // 2**level if level >= 0 else FINE * 2**-level
        span = 8 * stride
        start = paths.shape[1] - 1 - span
        truth = paths[:, start : start + span + 1 : stride]
        window_times = np.broadcast_to(
            times[start : start + span + 1 : stride], truth.shape
        )
        pristi = refine(imputer, sampler, truth[:, ::2], n_samples=DRAWS, seed=seed())
        exact = impute(
            truth[..., None],
            gp_sampler,
            factor=1,
            mask=alternate,
            times=window_times,
            n_samples=DRAWS,
            dt=float(window_times[0, 1] - window_times[0, 0]),
            seed=seed(),
        )[..., 0]
        midpoint = (truth[:, :-1:2] + truth[:, 2::2]) / 2
        rows[str(level)] = {
            "trained_scale": level <= 0,
            "pristi": float(ensemble_crps(pristi[:, :, 1::2], truth[:, 1::2]).mean()),
            "gp": float(ensemble_crps(exact[:, :, 1::2], truth[:, 1::2]).mean()),
            "interpolation": float(np.abs(midpoint - truth[:, 1::2]).mean()),
        }
    return rows


def history_quality(dense_test, history, truth, times, gp_sampler, seed):
    """CRPS of densified history points against the GP given coarse history."""
    rows = {}
    for level, densified in enumerate(dense_test, start=1):
        factor = 2**level
        length = (HISTORY - 1) * factor + 1
        fine_truth = truth[:, : (HISTORY - 1) * FINE + 1 : FINE // factor]
        fine_times = (
            times[:, :1] + np.arange(length) * (times[0, 1] - times[0, 0]) / factor
        )
        new = np.arange(length) % factor != 0
        exact = impute(
            fine_truth[..., None],
            gp_sampler,
            factor=1,
            mask=new,
            times=fine_times,
            n_samples=densified.shape[1],
            dt=float(fine_times[0, 1] - fine_times[0, 0]),
            seed=seed(),
        )[..., 0]
        interpolated = interpolate(history, factor)
        rows[f"x{factor}"] = {
            "pristi": float(
                ensemble_crps(densified[:, :, new], fine_truth[:, new]).mean()
            ),
            "gp": float(ensemble_crps(exact[:, :, new], fine_truth[:, new]).mean()),
            "interpolation": float(
                np.abs(interpolated[:, new] - fine_truth[:, new]).mean()
            ),
            "draws": int(densified.shape[1]),
        }
    return rows


def ensemble_crps(samples, truth):
    """Fair ensemble CRPS per entry for (windows, draws, time) samples.

    Unlike the plug-in estimator in metrics.crps, its expectation does not
    depend on the number of draws, so 4-draw, 16-draw and deterministic
    (one draw: absolute error) forecasts are directly comparable.
    """
    draws = samples.shape[1]
    errors = np.abs(samples - truth[:, None]).mean(axis=1)
    if draws == 1:
        return errors
    spread = np.abs(samples[:, :, None] - samples[:, None]).sum(axis=(1, 2))
    return errors - spread / (2 * draws * (draws - 1))


def score_forecasts(samples, future, n_paths):
    """CRPS, error and coverage per method, plus paired gains over sparse."""
    scores = {}
    per_window = {}
    for name, draws in samples.items():
        crps = ensemble_crps(draws, future)
        per_window[name] = crps.mean(axis=1)
        mean = draws.mean(axis=1)
        lower, upper = draws.min(axis=1), draws.max(axis=1)
        scores[name] = {
            "crps": float(crps.mean()),
            "crps_by_lead": crps.mean(axis=0).tolist(),
            "rmse": float(np.sqrt(((mean - future) ** 2).mean())),
            "range_coverage": (
                float(((future >= lower) & (future <= upper)).mean())
                if draws.shape[1] > 1
                else None
            ),
        }
    rng = np.random.default_rng(0)
    paths = np.arange(len(future)) % n_paths
    for name in samples:
        if name in ("sparse", "gp"):
            continue
        gain = per_window["sparse"] - per_window[name]
        by_path = np.array([gain[paths == path].mean() for path in range(n_paths)])
        resampled = rng.integers(n_paths, size=(4000, n_paths))
        interval = np.quantile(by_path[resampled].mean(axis=1), [0.025, 0.975])
        scores[name]["gain_over_sparse"] = float(gain.mean())
        scores[name]["gain_over_sparse_95"] = interval.tolist()
    return scores


class _EpochLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("epoch="):
            fields = dict(item.split("=") for item in message.split())
            self.records.append({key: float(value) for key, value in fields.items()})


INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOURS = {
    "gp": "#eb6834",
    "persistence": "#c3c2b7",
    "sparse": "#eda100",
    "learned_x2": "#86b6ef",
    "learned_x4": "#3987e5",
    "learned_x8": "#1c5cab",
    "interp_x8": "#1baf7a",
    "pristi": "#2a78d6",
    "interpolation": "#1baf7a",
}
NAMES = {
    "gp": "Exact GP forecast (optimal)",
    "persistence": "Repeat last value",
    "sparse": "Sparse forecaster",
    "learned_x2": "Densified x2",
    "learned_x4": "Densified x4",
    "learned_x8": "Densified x8",
    "interp_x8": "Interpolated x8",
}


def report(args):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _style(plt)
    results = {
        task: json.loads((args.output / task / "results.json").read_text())
        for task in TASKS
        if (args.output / task / "results.json").exists()
    }
    if not results:
        raise FileNotFoundError(f"No task results in {args.output}")
    write_tables(results, args.output)
    plot_forecast_crps(plt, results, args.output)
    plot_gains(plt, results, args.output)
    plot_by_lead(plt, results, args.output)
    plot_scale_transfer(plt, results, args.output)
    plot_examples(plt, results, args.output)
    print(f"Report written to {args.output}")


def write_tables(results, output):
    lines = [
        "# Densified vs sparse PriSTI forecasting",
        "",
        f"Coarse spacing dt, x{FINE} ground truth; {HISTORY} history and {HORIZON} "
        f"future coarse points; {DRAWS} draws per window "
        f"({HISTORIES} densified histories x {FUTURES} futures).",
        "",
        "## Forecast CRPS relative to the exact GP forecast (1 = best possible)",
        "",
        "| Task | GP CRPS | " + " | ".join(NAMES[m] for m in METHODS) + " |",
        "|---|---|" + "---|" * len(METHODS),
    ]
    for result in results.values():
        forecast = result["forecast"]
        gp = forecast["gp"]["crps"]
        cells = [f"{forecast[m]['crps'] / gp:.3f}" for m in METHODS]
        lines.append(f"| {result['label']} | {gp:.4f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## CRPS gain over the sparse forecaster (positive = better than sparse)",
        "",
        "Mean per-window CRPS reduction with a 95% bootstrap interval over test paths.",
        "",
        "| Task | " + " | ".join(NAMES[m] for m in METHODS if m != "sparse") + " |",
        "|---|" + "---|" * (len(METHODS) - 1),
    ]
    for result in results.values():
        cells = []
        for method in METHODS:
            if method == "sparse":
                continue
            entry = result["forecast"][method]
            low, high = entry["gain_over_sparse_95"]
            cells.append(f"{entry['gain_over_sparse']:+.4f} [{low:+.4f}, {high:+.4f}]")
        lines.append(f"| {result['label']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Forecast RMSE of the ensemble mean and coverage of the draws' range",
        "",
        f"Coverage of [min, max] of the {DRAWS} draws; a calibrated forecaster "
        f"covers {(DRAWS - 1) / (DRAWS + 1):.0%} of outcomes.",
        "",
        "| Task | " + " | ".join(NAMES[m] for m in ("gp", *METHODS)) + " |",
        "|---|" + "---|" * (len(METHODS) + 1),
    ]
    for result in results.values():
        forecast = result["forecast"]
        cells = [
            f"{forecast[m]['rmse']:.4f} ({_percent(forecast[m]['range_coverage'])})"
            for m in ("gp", *METHODS)
        ]
        lines.append(f"| {result['label']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Imputer scale transfer: one-step CRPS relative to the exact GP",
        "",
        "Spacing of the refined grid relative to dt; the imputer was trained at "
        "4, 2 and 1 only. PriSTI / interpolation ratios.",
        "",
        "| Task | " + " | ".join(_spacing(level) for level in LEVELS_TESTED) + " |",
        "|---|" + "---|" * len(LEVELS_TESTED),
    ]
    for result in results.values():
        rows = result["scale_transfer"]
        cells = [
            f"{rows[str(level)]['pristi'] / rows[str(level)]['gp']:.2f} / "
            f"{rows[str(level)]['interpolation'] / rows[str(level)]['gp']:.2f}"
            for level in LEVELS_TESTED
        ]
        lines.append(f"| {result['label']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Densified history quality: CRPS relative to the exact GP given the "
        "coarse history",
        "",
        f"{HISTORIES} densified draws per window, scored with the fair ensemble "
        "CRPS, which is unbiased for any number of draws. PriSTI / interpolation "
        "ratios.",
        "",
        "| Task | x2 | x4 | x8 |",
        "|---|---|---|---|",
    ]
    for result in results.values():
        rows = result["densified_history"]
        cells = [
            f"{rows[k]['pristi'] / rows[k]['gp']:.2f} / "
            f"{rows[k]['interpolation'] / rows[k]['gp']:.2f}"
            for k in ("x2", "x4", "x8")
        ]
        lines.append(f"| {result['label']} | " + " | ".join(cells) + " |")
    lines += ["", "## Wall-clock (minutes)", ""]
    for result in results.values():
        total = sum(result["timings"].values()) / 60
        lines.append(f"- {result['label']}: {total:.1f}")
    (output / "results.md").write_text("\n".join(lines) + "\n")


def _percent(value):
    return "n/a" if value is None else f"{value:.0%}"


def _spacing(level):
    return f"{2.0**-level:g} dt" if level <= 0 else f"dt/{2**level}"


def plot_forecast_crps(plt, results, output):
    labels = [result["label"] for result in results.values()]
    x = np.arange(len(labels))
    width = 0.13
    figure, ax = plt.subplots(figsize=(11, 4.6))
    for index, method in enumerate(METHODS):
        ratios = [
            result["forecast"][method]["crps"] / result["forecast"]["gp"]["crps"]
            for result in results.values()
        ]
        offset = (index - (len(METHODS) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            ratios,
            width=width * 0.92,
            color=COLOURS[method],
            label=NAMES[method],
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, color=SECONDARY, fontsize=7)
    ax.axhline(1, color=COLOURS["gp"], linewidth=2, label=NAMES["gp"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Forecast CRPS / exact GP CRPS")
    ax.set_ylim(0.9 * min(ax.get_ylim()[1], 1), None)
    ax.legend(frameon=False, ncols=4, loc="upper left", fontsize=8)
    ax.set_title(
        "Forecast CRPS at coarse future points (lower is better; 1 = best possible)",
        loc="left",
    )
    figure.tight_layout()
    figure.savefig(output / "forecast_crps.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_gains(plt, results, output):
    methods = [m for m in METHODS if m not in ("sparse", "persistence")]
    labels = [result["label"] for result in results.values()]
    figure, ax = plt.subplots(figsize=(9, 3.8))
    ax.axhline(0, color=AXIS, linewidth=1)
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * 0.16
        for row, result in enumerate(results.values()):
            entry = result["forecast"][method]
            scale = result["forecast"]["gp"]["crps"]
            low, high = (value / scale for value in entry["gain_over_sparse_95"])
            ax.vlines(row + offset, low, high, color=COLOURS[method], linewidth=2.5)
            ax.plot(
                row + offset,
                entry["gain_over_sparse"] / scale,
                "o",
                color=COLOURS[method],
                ms=7,
                mec=SURFACE,
                mew=2,
                label=NAMES[method] if row == 0 else None,
            )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("CRPS gain over sparse / GP CRPS")
    ax.legend(frameon=False, ncols=4, loc="upper left", fontsize=8)
    ax.set_title(
        "Gain of densified forecasters over the sparse forecaster "
        "(above 0 = better; 95% interval over test paths)",
        loc="left",
    )
    figure.tight_layout()
    figure.savefig(output / "forecast_gain.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_by_lead(plt, results, output):
    figure, axes = plt.subplots(
        1, len(results), figsize=(3.4 * len(results), 3.4), squeeze=False
    )
    leads = np.arange(1, HORIZON + 1)
    for ax, result in zip(axes[0], results.values(), strict=True):
        for method in ("gp", "persistence", "sparse", "learned_x8", "interp_x8"):
            ax.plot(
                leads,
                result["forecast"][method]["crps_by_lead"],
                "o-",
                color=COLOURS[method],
                linewidth=2,
                ms=5,
                mec=SURFACE,
                mew=1,
                label=NAMES[method],
            )
        ax.set_xticks(leads)
        ax.set_xlabel("Lead time (coarse steps)")
        ax.set_title(result["label"], loc="left")
    axes[0, 0].set_ylabel("CRPS")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=5, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output / "forecast_by_lead.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_scale_transfer(plt, results, output):
    figure, axes = plt.subplots(
        1, len(results), figsize=(3.4 * len(results), 3.4), squeeze=False
    )
    levels = np.array(LEVELS_TESTED)
    for ax, result in zip(axes[0], results.values(), strict=True):
        rows = result["scale_transfer"]
        ax.axvspan(0.5, levels.max() + 0.5, color=GRID, alpha=0.5, linewidth=0)
        for key in ("pristi", "interpolation"):
            ratios = [
                rows[str(level)][key] / rows[str(level)]["gp"] for level in levels
            ]
            ax.plot(
                levels,
                ratios,
                "o-",
                color=COLOURS[key],
                linewidth=2,
                ms=5,
                mec=SURFACE,
                mew=1,
                label="PriSTI imputer" if key == "pristi" else "Linear interpolation",
            )
        ax.axhline(1, color=COLOURS["gp"], linewidth=2, label="Exact GP")
        ax.set_xticks(levels, [_spacing(level) for level in levels], fontsize=7)
        ax.set_xlabel("Refined spacing (shaded: finer than trained)")
        ax.set_title(result["label"], loc="left")
    axes[0, 0].set_ylabel("One-step CRPS / exact GP CRPS")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output / "imputer_scale_transfer.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_examples(plt, results, output):
    figure, axes = plt.subplots(
        len(results), 1, figsize=(11, 3.2 * len(results)), squeeze=False
    )
    for ax, (task, result) in zip(axes[:, 0], results.items(), strict=True):
        data = np.load(output / task / "samples.npz")
        truth, window = data["truth"][0], data["windows"][0]
        fine_steps = np.arange(truth.size) / FINE
        ax.plot(fine_steps, truth, color=AXIS, linewidth=1, label="Fine ground truth")
        dense = data["densified"][0]
        dense_steps = np.arange(dense.shape[-1]) / 2**LEVELS
        for index, path in enumerate(dense[:2]):
            ax.plot(
                dense_steps,
                path,
                color=COLOURS["learned_x8"],
                linewidth=1,
                alpha=0.7,
                label="Densified x8 history (2 draws)" if index == 0 else None,
            )
        coarse_steps = np.arange(window.size)
        ax.plot(
            coarse_steps[:HISTORY],
            window[:HISTORY],
            "o",
            color=INK,
            ms=5,
            mec=SURFACE,
            mew=1,
            label="Observed history",
        )
        ax.plot(
            coarse_steps[HISTORY:],
            window[HISTORY:],
            "o",
            mfc=SURFACE,
            mec=INK,
            ms=6,
            mew=1.4,
            label="True future",
        )
        for offset, method in ((-0.18, "sparse"), (0.0, "learned_x8"), (0.18, "gp")):
            draws = data[method][0]
            low, mid, high = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
            x = coarse_steps[HISTORY:] + offset
            ax.vlines(x, low, high, color=COLOURS[method], linewidth=2.5)
            ax.plot(
                x,
                mid,
                "o",
                color=COLOURS[method],
                ms=4,
                mec=SURFACE,
                mew=1,
                label=f"{NAMES[method]}: 90% interval",
            )
        ax.set_title(f"{result['label']}: first test window (not selected)", loc="left")
        ax.set_xlabel("Time (coarse steps)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=4, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output / "forecast_examples.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def _style(plt):
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


if __name__ == "__main__":
    main()
