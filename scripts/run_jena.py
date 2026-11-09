"""Densify-then-forecast on measured solar radiation (Jena rooftop station).

The real-data counterpart of the quasi-periodic sweep in run_densiffusion.py.
The signal is shortwave downward radiation (SWDR) from the Max Planck
Institute for Biogeochemistry's rooftop station, measured every 10 minutes
(scripts/prepare_jena.py; CC-BY-4.0). Its dominant period is P = 24 h, so the
coarse spacing dt sets P/dt. Each spacing is a multiple of 80 minutes, so the
fine grid dt / 8 lies on the 10-minute measurements.

Paths are 33 coarse points (256 fine steps), taken from 2004-2018 for
training, 2019-2020 for validation and 2021-2024 for testing, evenly spaced
in time and skipping any with missing values. Windows, normalisation, model
sizes, training schedule and 100 draws per window match the synthetic sweep.

Windows are normalised as in the synthetic sweep, but with the scale floored
at --scale-floor (20 W/m^2, about 2% of peak radiation): a history entirely
at night would otherwise have a near-zero scale.

There is no exact GP here, so scores are relative to the sparse forecaster,
and the oracle uses the true fine history (one history, 100 futures) instead
of exact posterior draws. Methods, as in the synthetic sweep:

- sparse, persistence (repeat the last observation);
- learned_xK: densifier and forecaster trained on coarse data only;
- interp_xK: linear interpolation;
- fine_xK: per-level imputers and a forecaster trained on the fine paths;
- fineinterp_x2: the fine-data forecaster given interpolated histories;
- oracle_xK: the fine-data forecaster given the true fine history;
- sparse_shifted: a sparse forecaster trained on the fine paths at every
  coarse phase, for the same number of updates as sparse.

    uv run --extra research python scripts/run_jena.py run --task jena_3 \\
        --output outputs/jena/pristi
    uv run --extra research --with matplotlib python scripts/run_jena.py \\
        report --output outputs/jena/pristi
"""

import argparse
import dataclasses
import json
import logging
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from run_densiffusion import (
    COARSE_POINTS,
    COLOURS,
    DRAWS,
    FINE,
    FUTURES,
    HISTORIES,
    HISTORY,
    HORIZON,
    LEVELS,
    MARKERS,
    NAMES,
    SURFACE,
    _clip_markers,
    _EpochLog,
    _style,
    config,
    ensemble_crps,
    forecast_windows,
    level_windows,
    method_seed,
    phase_shifted_windows,
    score_forecasts,
)

from densiffusion import densify_forecast
from densiffusion.densify_forecast import (
    MODEL_LABELS,
    SMALL_MODELS,
    densify,
    forecast,
    imputation_windows,
    interpolate,
    sampling_config,
    small_model,
    train_forecaster,
    train_imputer,
)

PERIOD_MINUTES = 24 * 60
SPACINGS = (880, 560, 480, 400, 240, 160, 80)  # minutes; multiples of 80
TASKS = {f"jena_{PERIOD_MINUTES / dt:.3g}": dt for dt in SPACINGS}
SPLITS = {"train": (2004, 2018), "validation": (2019, 2020), "test": (2021, 2024)}
METHODS = (
    "persistence",
    "sparse",
    "sparse_shifted",
    "learned_x2",
    "learned_x8",
    "interp_x2",
    "interp_x8",
    "fine_x2",
    "fine_x8",
    "fineinterp_x2",
    "oracle_x2",
    "oracle_x8",
)
LABELS = {
    **NAMES,
    "oracle_x2": "Oracle x2 (true history)",
    "oracle_x8": "Oracle x8 (true history)",
}


def load(path: Path):
    data = np.load(path)
    start = datetime.fromisoformat(str(data["start"]))
    return start, data["values"], int(data["step_minutes"])


def extract_paths(values, start, step, years, spacing, count):
    """count fine paths at spacing / FINE, evenly spaced within years.

    A path whose start would include a missing value moves forward one fine
    step at a time until it is complete. Returns (paths, times in hours).
    """
    stride = spacing // FINE // step
    span = (COARSE_POINTS - 1) * FINE * stride
    first = int((datetime(years[0], 1, 1) - start) / timedelta(minutes=step))
    last = int((datetime(years[1] + 1, 1, 1) - start) / timedelta(minutes=step))
    last = min(last, values.size) - span - 1
    starts = np.linspace(max(first, 0), last, count).astype(int)
    paths, offsets = [], []
    for begin in starts:
        while np.isnan(values[begin : begin + span + 1 : stride]).any():
            begin += stride
        paths.append(values[begin : begin + span + 1 : stride])
        offsets.append(begin)
    if len({*offsets}) < count or offsets[-1] > last:
        raise ValueError("Not enough complete paths in this split")
    times = np.arange(span // stride + 1) * spacing / FINE / 60
    return np.stack(paths), times, np.array(offsets)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("run", "report"))
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/jena/swdr.npz"))
    parser.add_argument("--model", choices=list(SMALL_MODELS), default="pristi")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--train-device",
        help="Device for training if not --device (see run_densiffusion)",
    )
    parser.add_argument("--train-paths", type=int, default=256)
    parser.add_argument("--validation-paths", type=int, default=32)
    parser.add_argument("--test-paths", type=int, default=64)
    parser.add_argument("--imputer-epochs", type=int, default=60)
    parser.add_argument("--forecaster-epochs", type=int, default=150)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale-floor",
        type=float,
        default=20.0,
        help="Smallest window normalisation scale in W/m^2 (night histories)",
    )
    args = parser.parse_args()
    densify_forecast.SCALE_FLOOR = args.scale_floor
    if args.mode == "run":
        torch.set_num_threads(args.threads)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        run_task(args)
    else:
        report(args)


def run_task(args):
    task, spacing = args.task, TASKS[args.task]
    folder = args.output / task
    folder.mkdir(parents=True)
    logger = logging.getLogger("densiffusion.training")
    timings, clock = {}, time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timings[name], clock = now - clock, now
        print(f"[{task}] {name}: {timings[name] / 60:.1f} min", flush=True)

    def seed(name):
        return method_seed(args.seed, name)

    def fit(model, checkpoint, train):
        log = _EpochLog()
        logger.addHandler(log)
        try:
            train(checkpoint=checkpoint)
        finally:
            logger.removeHandler(log)
        model.to(args.device)
        return log.records

    start, values, step = load(args.data)
    counts = {
        "train": args.train_paths,
        "validation": args.validation_paths,
        "test": args.test_paths,
    }
    fine, path_starts = {}, {}
    for part, years in SPLITS.items():
        paths, times, offsets = extract_paths(
            values, start, step, years, spacing, counts[part]
        )
        fine[part], path_starts[part] = (paths, times), offsets
    coarse = {part: paths[:, ::FINE] for part, (paths, _) in fine.items()}
    windows = {part: forecast_windows(*data) for part, data in fine.items()}
    train_windows, train_truth, _ = windows["train"]
    validation_windows, validation_truth, _ = windows["validation"]
    test_windows, test_truth, _ = windows["test"]
    test_history = test_windows[:, :HISTORY]
    future = test_windows[:, HISTORY:]
    results = {
        "task": task,
        "label": f"P/dt = {PERIOD_MINUTES / spacing:.3g}",
        "spacing_minutes": spacing,
        "ratio": PERIOD_MINUTES / spacing,
        "model": args.model,
        "device": args.device,
        "train_device": args.train_device or args.device,
        "scale_floor": args.scale_floor,
        "methods": METHODS,
        "first_test_start": str(
            start + timedelta(minutes=step * int(path_starts["test"][0]))
        ),
    }
    lap("data")

    train_device = args.train_device or args.device
    imputer_config = config(args.imputer_epochs, train_device)
    sampler = sampling_config(imputer_config)
    epochs = {}
    torch.manual_seed(args.seed)
    imputer = small_model(args.model)
    epochs["imputer"] = fit(
        imputer,
        folder / "imputer.pt",
        partial(
            train_imputer,
            imputer,
            imputation_windows(coarse["train"], (1,)),
            imputation_windows(coarse["validation"], (1,)),
            imputer_config,
            seed=args.seed,
        ),
    )
    fine_imputers = []
    for level in range(1, LEVELS + 1):
        pick = np.random.default_rng(args.seed + level)
        torch.manual_seed(args.seed)
        model = small_model(args.model)
        epochs[f"imputer_fine_level{level}"] = fit(
            model,
            folder / f"imputer_fine_level{level}.pt",
            partial(
                train_imputer,
                model,
                level_windows(fine["train"][0], level, pick),
                level_windows(fine["validation"][0], level, pick),
                imputer_config,
                seed=args.seed,
            ),
        )
        fine_imputers.append(model)
    lap("imputer training")

    def dense(models, series, draws, name):
        return densify(
            models, sampler, series, levels=LEVELS, n_samples=draws, seed=seed(name)
        )

    dense_train = dense(imputer, train_windows, 1, "dense_train")
    dense_validation = dense(imputer, validation_windows, 1, "dense_validation")
    dense_test = dense(imputer, test_history, HISTORIES, "dense_test")
    dense_test_fine = dense(fine_imputers, test_history, HISTORIES, "dense_test_fine")
    results["densified_history"] = history_quality(
        {"coarse-trained": dense_test, "fine-data": dense_test_fine},
        test_history,
        test_truth,
    )
    lap("densification")

    def true_history(factor):
        return test_truth[:, : (HISTORY - 1) * FINE + 1 : FINE // factor]

    def variant(name):
        """(forecaster key, factor, train, validation, test histories,
        histories per window); see run_densiffusion.run_task."""
        if name == "sparse":
            return name, 1, train_windows, validation_windows, test_history, 1
        if name == "sparse_shifted":
            return (
                name,
                1,
                phase_shifted_windows(fine["train"][0]),
                phase_shifted_windows(fine["validation"][0]),
                test_history,
                1,
            )
        kind, factor = name.split("_x")
        factor = int(factor)
        level = factor.bit_length() - 1
        truth = (
            train_truth[:, :: FINE // factor],
            validation_truth[:, :: FINE // factor],
        )
        if kind == "learned":
            draws = dense_test[level - 1]
            train = (dense_train[level - 1][:, 0], dense_validation[level - 1][:, 0])
            return (name, factor, *train, draws.reshape(-1, draws.shape[-1]), HISTORIES)
        if kind == "fine":
            draws = dense_test_fine[level - 1]
            histories = draws.reshape(-1, draws.shape[-1])
            return (f"truefine_x{factor}", factor, *truth, histories, HISTORIES)
        if kind == "fineinterp":
            histories = interpolate(test_history, factor)
            return (f"truefine_x{factor}", factor, *truth, histories, 1)
        if kind == "oracle":
            return (f"truefine_x{factor}", factor, *truth, true_history(factor), 1)
        return (
            name,
            factor,
            interpolate(train_windows, factor),
            interpolate(validation_windows, factor),
            interpolate(test_history, factor),
            1,
        )

    forecaster_config = config(args.forecaster_epochs, train_device)
    forecaster_sampler = sampling_config(forecaster_config)
    shifted_config = dataclasses.replace(
        forecaster_config,
        epochs=max(1, args.forecaster_epochs // FINE),
        validation_interval=1,
    )
    samples = {"persistence": np.repeat(test_history[:, -1:, None], HORIZON, axis=2)}
    forecasters = {}
    for name in METHODS[1:]:
        key, factor, train, validation, histories, per_window = variant(name)
        if key not in forecasters:
            torch.manual_seed(args.seed)
            forecasters[key] = small_model(args.model)
            epochs[key] = fit(
                forecasters[key],
                folder / f"{key}.pt",
                partial(
                    train_forecaster,
                    forecasters[key],
                    train,
                    validation,
                    shifted_config if key == "sparse_shifted" else forecaster_config,
                    history=HISTORY,
                    horizon=HORIZON,
                    factor=factor,
                    seed=args.seed,
                ),
            )
        predicted = forecast(
            forecasters[key],
            forecaster_sampler,
            histories,
            horizon=HORIZON,
            factor=factor,
            n_samples=DRAWS // per_window,
            seed=seed(name),
        )
        samples[name] = predicted.reshape(len(test_windows), DRAWS, HORIZON)
        lap(f"forecaster {name}")

    results["forecast"] = score_forecasts(samples, future, args.test_paths)
    results["epochs"] = epochs
    results["timings"] = timings
    (folder / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        folder / "samples.npz",
        truth=test_truth,
        windows=test_windows,
        densified=dense_test[-1],
        densified_fine=dense_test_fine[-1],
        **samples,
    )
    print(f"[{task}] done", flush=True)


def history_quality(densified, history, truth):
    """CRPS (W/m^2) of densified history points against the true fine values,
    and linear interpolation's absolute error, at each refinement level."""
    rows = {}
    for level in range(1, LEVELS + 1):
        factor = 2**level
        length = (HISTORY - 1) * factor + 1
        fine_truth = truth[:, : (HISTORY - 1) * FINE + 1 : FINE // factor]
        new = np.arange(length) % factor != 0
        row = {
            name: float(
                ensemble_crps(draws[level - 1][:, :, new], fine_truth[:, new]).mean()
            )
            for name, draws in densified.items()
        }
        row["interpolation"] = float(
            np.abs(interpolate(history, factor)[:, new] - fine_truth[:, new]).mean()
        )
        rows[f"x{factor}"] = row
    return rows


def report(args):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _style(plt)
    results = sorted(
        (
            json.loads((args.output / task / "results.json").read_text())
            for task in TASKS
            if (args.output / task / "results.json").exists()
        ),
        key=lambda result: result["ratio"],
    )
    if not results:
        raise FileNotFoundError(f"No task results in {args.output}")
    write_tables(results, args.output)
    every = [m for m in METHODS if m not in ("persistence", "sparse")]
    plot_sweep(plt, results, args.output / "sweep.png", every)
    plot_sweep(plt, results, args.output / "sweep_focus.png", FOCUS)
    print(f"Report written to {args.output}")


def _relative(result, method):
    forecast = result["forecast"]
    return forecast[method]["crps"] / forecast["sparse"]["crps"]


def write_tables(results, output):
    model = MODEL_LABELS[results[0]["model"]]
    methods = [m for m in METHODS if m != "sparse"]
    lines = [
        f"# Densified vs sparse {model} forecasting of Jena solar radiation",
        "",
        f"Coarse spacing dt with P = 24 h, x{FINE} measured truth; {HISTORY} "
        f"history and {HORIZON} future coarse points; {DRAWS} draws per window "
        f"({HISTORIES} densified histories x {FUTURES} futures).",
        "",
        "## Forecast CRPS relative to the sparse forecaster (below 1 = better)",
        "",
        "| P/dt | dt | Sparse CRPS (W/m^2) | "
        + " | ".join(LABELS[m] for m in methods)
        + " |",
        "|---|---|---|" + "---|" * len(methods),
    ]
    for result in results:
        cells = []
        for method in methods:
            mark = ""
            if method != "persistence":
                low, high = result["forecast"][method]["gain_over_sparse_95"]
                mark = " +" if low > 0 else " -" if high < 0 else ""
            cells.append(f"{_relative(result, method):.3f}{mark}")
        lines.append(
            f"| {result['ratio']:.3g} | {result['spacing_minutes'] / 60:.3g} h | "
            f"{result['forecast']['sparse']['crps']:.1f} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "`+` / `-`: significantly better / worse than sparse (95% bootstrap "
        "interval over test paths).",
        "",
        "## Densified history CRPS at x8 (W/m^2)",
        "",
        "| P/dt | Coarse-trained | Fine-data | Interpolation (abs. error) |",
        "|---|---|---|---|",
    ]
    for result in results:
        row = result["densified_history"]["x8"]
        lines.append(
            f"| {result['ratio']:.3g} | {row['coarse-trained']:.1f} | "
            f"{row['fine-data']:.1f} | {row['interpolation']:.1f} |"
        )
    lines += ["", "## Wall-clock (minutes)", ""]
    for result in results:
        lines.append(f"- {result['label']}: {sum(result['timings'].values()) / 60:.1f}")
    (output / "results.md").write_text("\n".join(lines) + "\n")


FOCUS = ("learned_x2", "interp_x2", "fine_x2", "oracle_x2")


def plot_sweep(plt, results, path, shown):
    """CRPS relative to sparse and gain over sparse against P/dt."""
    ratios = np.array([result["ratio"] for result in results])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax in axes:
        ax.axvspan(ratios.min() / 1.1, 2, color="#e1e0d9", alpha=0.6, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(ratios, [f"{r:.3g}" for r in ratios])
        ax.minorticks_off()
        ax.set_xlim(ratios.min() / 1.1, ratios.max() * 1.1)
        ax.set_xlabel("24 h / coarse spacing (shaded: aliased)")
    ceiling = 2.0
    for method in shown:
        values = np.array([_relative(result, method) for result in results])
        axes[0].plot(
            ratios,
            np.minimum(values, ceiling),
            color=COLOURS[method],
            linewidth=1.8,
            label=LABELS[method],
        )
        _clip_markers(axes[0], ratios, values, ceiling, method)
    axes[0].axhline(1, color=COLOURS["sparse"], linewidth=2, label=LABELS["sparse"])
    axes[0].set_ylim(None, ceiling * 1.04)
    axes[0].set_ylabel("Forecast CRPS / sparse CRPS")
    axes[0].set_title("Forecast CRPS relative to sparse (lower is better)", loc="left")
    for index, method in enumerate(shown):
        offset = (1 + 0.1 / len(shown)) ** (index - (len(shown) - 1) / 2)
        scale = np.array([r["forecast"]["sparse"]["crps"] for r in results])
        gain = np.array([r["forecast"][method]["gain_over_sparse"] for r in results])
        interval = np.array(
            [r["forecast"][method]["gain_over_sparse_95"] for r in results]
        )
        axes[1].vlines(
            ratios * offset,
            interval[:, 0] / scale,
            interval[:, 1] / scale,
            color=COLOURS[method],
            linewidth=2,
        )
        axes[1].plot(
            ratios * offset,
            gain / scale,
            linestyle="none",
            marker=MARKERS.get(method.split("_x")[0], "o"),
            color=COLOURS[method],
            mec=SURFACE,
            mew=1,
            ms=6,
        )
    axes[1].axhline(0, color="#c3c2b7", linewidth=1)
    axes[1].set_ylabel("CRPS gain over sparse / sparse CRPS")
    axes[1].set_title("Gain over sparse (above 0 = better; 95% interval)", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=5, frameon=False)
    figure.tight_layout(rect=(0, 0.14 if len(shown) > 5 else 0.08, 1, 1))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    main()
