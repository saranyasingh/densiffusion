"""Visualize CSDI → Chronos-2 electricity/FX run artifacts.

    uv run --with matplotlib python scripts/plot_csdi_chronos2_results.py \
        --run electricity_csdi_chronos2_61856074

Writes into RUN/:
  table.png          forecast metrics table (best cells highlighted)
  bars.png           MAE / MASE / SQL bar chart
  imputation.png     CSDI vs linear imputation CRPS
  examples.png       truth / sparse / linear / CSDI for a few clients
                     (needs imputed.npz next to results.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

BACKGROUND = "#f7f3ea"
PANEL = "#fffdf8"
HEADER = "#ebe3d3"
BORDER = "#c9bfae"
TEXT = "#2f2f2f"
MUTED = "#6e6a63"
ACCENT = "#2f6f6a"
ACCENT2 = "#b45a3c"
ACCENT3 = "#6b7c3d"
BEST = "#dceeea"

# Prefer new zero-shot keys; fall back to legacy fine-tuned names from older runs.
ARM_LABELS = (
    ("zeroshot_csdi", "CSDI → Chronos (ZS)"),
    ("csdi_imputed", "CSDI → Chronos (FT)"),
    ("zeroshot_linear", "Linear → Chronos (ZS)"),
    ("linear_imputed", "Linear → Chronos (FT)"),
    ("zeroshot_sparse", "Sparse → Chronos (ZS)"),
    ("sparse", "Sparse → Chronos (FT)"),
    ("dense_oracle", "Dense oracle → Chronos (ZS)"),
    ("seasonal_naive", "Seasonal naive"),
    ("last_value", "Last value"),
    ("forecast_last_value", "Last value"),
)

METRICS = (
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("smape", "sMAPE"),
    ("mase", "MASE"),
    ("wql", "WQL"),
    ("sql", "SQL"),
)

BAR_COLORS = {
    "zeroshot_csdi": ACCENT,
    "csdi_imputed": ACCENT,
    "zeroshot_linear": ACCENT2,
    "linear_imputed": ACCENT2,
    "zeroshot_sparse": "#7a6f64",
    "sparse": "#7a6f64",
    "dense_oracle": ACCENT3,
    "seasonal_naive": "#8a857c",
    "last_value": "#b0a89c",
    "forecast_last_value": "#b0a89c",
}


def load_run(run: Path) -> tuple[dict, Path]:
    results = run / "results.json" if run.is_dir() else run
    if results.name != "results.json":
        raise SystemExit(f"expected results.json, got {results}")
    payload = json.loads(results.read_text())
    # Flatten legacy last-value location into forecast for plotting.
    forecast = dict(payload.get("forecast", {}))
    if "last_value" not in forecast and "forecast_last_value" in payload:
        forecast["last_value"] = payload["forecast_last_value"]
    payload = {**payload, "forecast": forecast}
    return payload, results.parent


def available_arms(forecast: dict) -> list[tuple[str, str]]:
    return [(key, label) for key, label in ARM_LABELS if key in forecast]


def fmt(value: float, key: str) -> str:
    if key in {"smape", "mase", "sql"}:
        return f"{value:.3f}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def style_axis(axis, title: str | None = None) -> None:
    axis.set_facecolor(PANEL)
    for spine in axis.spines.values():
        spine.set_color(BORDER)
    axis.tick_params(colors=TEXT, labelsize=9)
    if title:
        axis.set_title(title, color=TEXT, fontsize=12, pad=10, loc="left")


def plot_table(payload: dict, output: Path) -> None:
    forecast = payload["forecast"]
    arms = available_arms(forecast)
    if not arms:
        raise SystemExit("no forecast arms found")
    metric_keys = [
        key for key, _ in METRICS if any(key in forecast[a] for a, _ in arms)
    ]
    metric_labels = [label for key, label in METRICS if key in metric_keys]
    best = {
        key: min(forecast[a][key] for a, _ in arms if key in forecast[a])
        for key in metric_keys
    }

    cols = 1 + len(metric_keys)
    rows = len(arms)
    figure = Figure(figsize=(1.35 * cols + 2.2, 0.52 * (rows + 3)), dpi=160)
    figure.patch.set_facecolor(BACKGROUND)
    axis = figure.add_axes((0.02, 0.08, 0.96, 0.82))
    axis.set_axis_off()
    axis.set_xlim(0, cols)
    axis.set_ylim(0, rows + 2)

    def cell(x, y, text, *, weight="normal", color=TEXT, size=9):
        axis.text(
            x + 0.5,
            y + 0.5,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            fontweight=weight,
            linespacing=1.15,
        )

    axis.fill_between(
        [0, cols], rows + 1, rows + 2, color=HEADER, linewidth=0, zorder=0
    )
    cell(0, rows + 1, "Method", weight="bold", size=10)
    for index, label in enumerate(metric_labels):
        cell(1 + index, rows + 1, label, weight="bold", size=10)

    for row, (arm, label) in enumerate(arms):
        y = rows - row
        axis.fill_between(
            [0, cols],
            y,
            y + 1,
            color=PANEL if row % 2 == 0 else BACKGROUND,
            linewidth=0,
            zorder=0,
        )
        cell(0, y, label)
        for col, key in enumerate(metric_keys):
            if key not in forecast[arm]:
                cell(1 + col, y, "—", color=MUTED)
                continue
            value = forecast[arm][key]
            is_best = abs(value - best[key]) <= 1e-9 * max(1.0, abs(best[key]))
            if is_best:
                axis.fill_between(
                    [1 + col, 2 + col],
                    y,
                    y + 1,
                    color=BEST,
                    linewidth=0,
                    zorder=0,
                )
            cell(
                1 + col,
                y,
                fmt(value, key),
                weight="bold" if is_best else "normal",
                color=ACCENT if is_best else TEXT,
            )

    for x in range(cols + 1):
        axis.axvline(x, color=BORDER, linewidth=0.7)
    for y in range(rows + 3):
        axis.axhline(y, color=BORDER, linewidth=0.7)

    title = "Forecast skill (lower is better)"
    parts = []
    if "imputation" in payload:
        imp = payload["imputation"]
        parts.append(
            f"imput. CRPS  CSDI {imp['csdi_crps']:.3f} · "
            f"linear {imp['linear_crps']:.3f}"
        )
    if "n_origins" in payload:
        parts.append(f"{payload['n_origins']} origins")
    if parts:
        title = f"{title}   ·   " + "   ·   ".join(parts)
    axis.set_title(title, fontsize=11, color=TEXT, pad=12, loc="left")
    figure.savefig(output, dpi=160, facecolor=BACKGROUND)
    print(output)


def plot_bars(payload: dict, output: Path) -> None:
    forecast = payload["forecast"]
    arms = available_arms(forecast)
    # Prefer scaled metrics when present; always show MAE.
    metric_order = [
        key
        for key in ("mae", "mase", "sql", "wql", "smape")
        if any(key in forecast[a] for a, _ in arms)
    ]
    if not metric_order:
        raise SystemExit("no plottable metrics")
    metric_order = metric_order[:3]
    labels = [label for _, label in arms]
    x = np.arange(len(arms))
    width = 0.72

    figure = Figure(figsize=(10.5, 3.2 * len(metric_order)), dpi=160)
    figure.patch.set_facecolor(BACKGROUND)
    colors = [BAR_COLORS.get(arm, MUTED) for arm, _ in arms]
    for index, key in enumerate(metric_order):
        axis = figure.add_subplot(len(metric_order), 1, index + 1)
        style_axis(axis, dict(METRICS).get(key, key.upper()) + "  (lower better)")
        values = [forecast[arm].get(key, np.nan) for arm, _ in arms]
        bars = axis.bar(
            x, values, width=width, color=colors, edgecolor=BORDER, linewidth=0.6
        )
        best_idx = int(np.nanargmin(values))
        bars[best_idx].set_edgecolor(ACCENT)
        bars[best_idx].set_linewidth(1.8)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel(dict(METRICS).get(key, key), color=TEXT)
        axis.yaxis.grid(True, color=BORDER, linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    fmt(value, key),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=TEXT,
                )
    figure.tight_layout(pad=1.2)
    figure.savefig(output, dpi=160, facecolor=BACKGROUND)
    print(output)


def plot_imputation(payload: dict, output: Path) -> None:
    if "imputation" not in payload:
        return
    imp = payload["imputation"]
    figure = Figure(figsize=(6.2, 3.6), dpi=160)
    figure.patch.set_facecolor(BACKGROUND)
    axis = figure.add_subplot(1, 1, 1)
    style_axis(axis, "Imputation CRPS on hidden points  (lower better)")
    names = ["CSDI", "Linear"]
    values = [imp["csdi_crps"], imp["linear_crps"]]
    colors = [ACCENT, ACCENT2]
    bars = axis.bar(names, values, color=colors, edgecolor=BORDER, width=0.55)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=TEXT,
            fontweight="bold",
        )
    if values[0] < values[1]:
        reduction = 100 * (1 - values[0] / values[1])
        axis.text(
            0.98,
            0.95,
            f"CSDI −{reduction:.0f}% vs linear",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=ACCENT,
            fontweight="bold",
        )
    axis.set_ylabel("CRPS (normalized)", color=TEXT)
    axis.yaxis.grid(True, color=BORDER, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, facecolor=BACKGROUND)
    print(output)


def plot_examples(
    run_dir: Path, payload: dict, output: Path, *, n_clients: int = 3
) -> None:
    path = run_dir / "imputed.npz"
    if not path.exists():
        print(f"skip examples (missing {path})")
        return
    with np.load(path, allow_pickle=False) as data:
        target = data["target_norm"][0]
        imputed = data["imputed_norm"][0]
        linear = data["linear_norm"][0]
        observed = data["observed"][0]
        mean = data["mean"]
        std = data["std"]
    # Back to original units for readability.
    truth = target * std + mean
    csdi = imputed * std + mean
    lin = linear * std + mean
    channels = payload.get("channels") or [f"ch{i}" for i in range(truth.shape[1])]
    # Pick clients with large hidden-point |CSDI-linear| disagreement.
    hidden = ~observed
    disagreement = np.mean(np.abs(csdi - lin) * hidden, axis=0)
    order = np.argsort(-disagreement)[:n_clients]

    # Show a recent window that includes missing mass.
    t1 = truth.shape[0]
    width = min(240, t1)
    start = max(0, t1 - width)
    times = np.arange(start, t1)

    figure = Figure(figsize=(11, 2.8 * len(order)), dpi=160)
    figure.patch.set_facecolor(BACKGROUND)
    for row, channel in enumerate(order):
        axis = figure.add_subplot(len(order), 1, row + 1)
        style_axis(axis, f"{channels[channel]}")
        obs_t = times[observed[start:t1, channel]]
        axis.plot(
            times,
            truth[start:t1, channel],
            color="#1f1f1f",
            linewidth=1.1,
            label="truth",
            zorder=3,
        )
        axis.plot(
            times,
            lin[start:t1, channel],
            color=ACCENT2,
            linewidth=1.2,
            alpha=0.9,
            label="linear",
        )
        axis.plot(
            times,
            csdi[start:t1, channel],
            color=ACCENT,
            linewidth=1.4,
            label="CSDI",
        )
        axis.scatter(
            obs_t,
            truth[obs_t, channel],
            s=10,
            color="#111111",
            zorder=4,
            label="observed" if row == 0 else None,
        )
        axis.set_xlim(start, t1 - 1)
        axis.yaxis.grid(True, color=BORDER, linewidth=0.5, alpha=0.7)
        axis.set_axisbelow(True)
        if row == 0:
            axis.legend(frameon=False, fontsize=8, ncols=4, loc="upper right")
        if row == len(order) - 1:
            axis.set_xlabel("time index", color=TEXT)
    figure.tight_layout(pad=1.1)
    figure.savefig(output, dpi=160, facecolor=BACKGROUND)
    print(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run directory containing results.json (and optional imputed.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the run directory",
    )
    parser.add_argument("--n-clients", type=int, default=3)
    args = parser.parse_args(argv)
    payload, run_dir = load_run(args.run)
    out = args.output_dir or run_dir
    out.mkdir(parents=True, exist_ok=True)
    plot_table(payload, out / "table.png")
    plot_bars(payload, out / "bars.png")
    plot_imputation(payload, out / "imputation.png")
    plot_examples(run_dir, payload, out / "examples.png", n_clients=args.n_clients)


if __name__ == "__main__":
    main()
