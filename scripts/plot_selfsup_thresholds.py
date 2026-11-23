"""Plot 1D self-sup threshold curves from a sweep output root.

    uv run --with matplotlib python scripts/plot_selfsup_thresholds.py \
        --root outputs

Finds ``elec_selfsup_*`` run directories under ``--root``, groups by
(missing_mode, sweep), and writes:
  thresholds_<mode>_A.png   base-sparsity curves
  thresholds_<mode>_B.png   SSL-mask-rate curves
  thresholds_summary.json   extracted points
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

BACKGROUND = "#f7f3ea"
PANEL = "#fffdf8"
BORDER = "#c9bfae"
TEXT = "#2f2f2f"
MUTED = "#6e6a63"
ACCENT = "#2f6f6a"
ACCENT2 = "#b45a3c"
ACCENT3 = "#6b7c3d"
SPARSE = "#7a6f64"

RUN_RE = re.compile(
    r"elec_selfsup_(?P<mode>block|random)_"
    r"(?P<sweep>A|B)_"
    r"(?P<label>[^/]+?)(?:_\d+(?:_\d+)?)?$"
)

SERIES = (
    ("zeroshot_csdi", "Self-sup CSDI → Chronos", ACCENT),
    ("zeroshot_linear", "Linear → Chronos", ACCENT2),
    ("zeroshot_sparse", "Sparse → Chronos", SPARSE),
    ("dense_oracle", "Dense oracle", ACCENT3),
)


def parse_label_value(sweep: str, mode: str, label: str) -> float:
    if sweep == "A" and mode == "random" and label.startswith("keep"):
        return float(label[len("keep") :])
    if sweep == "A" and mode == "block" and label.startswith("obs"):
        obs = float(label[len("obs") :])
        return obs / 24.0  # observed fraction within the 24h period
    if sweep == "B" and label.startswith("ssl"):
        return float(label[len("ssl") :])
    raise ValueError(f"unrecognized label {label!r} for sweep {sweep}/{mode}")


def xlabel_for(sweep: str, mode: str) -> str:
    if sweep == "A" and mode == "random":
        return "Keep probability (base sparsity)"
    if sweep == "A" and mode == "block":
        return "Observe fraction within 24h period"
    return "SSL mask rate (of observed points)"


def discover_runs(root: Path) -> list[tuple[Path, dict, dict]]:
    found = []
    for results in sorted(root.rglob("results.json")):
        run = results.parent
        match = RUN_RE.search(run.name)
        if not match:
            continue
        payload = json.loads(results.read_text())
        if payload.get("protocol") not in {"selfsup", "supervised"}:
            # Parent selfsup dir; skip nested supervised unless named match.
            if "protocol" not in payload:
                continue
        meta = match.groupdict()
        # Prefer selfsup protocol at the matched directory.
        if payload.get("protocol") == "supervised" and run.name.startswith(
            "elec_selfsup_"
        ):
            continue
        found.append((run, meta, payload))
    return found


def load_supervised(run: Path) -> dict | None:
    path = run / "supervised" / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect_points(runs: list[tuple[Path, dict, dict]], mode: str, sweep: str):
    points = []
    for run, meta, payload in runs:
        if meta["mode"] != mode or meta["sweep"] != sweep:
            continue
        # Strip trailing job id from label if the regex kept it in dirname only.
        label = meta["label"]
        # dirname may be elec_selfsup_block_A_obs12_123_3 — regex non-greedy label.
        try:
            x = parse_label_value(sweep, mode, label)
        except ValueError:
            continue
        forecast = payload.get("forecast", {})
        imputation = payload.get("imputation", {})
        row = {
            "x": x,
            "label": label,
            "run": str(run),
            "observed_fraction": payload.get("observed_fraction"),
            "ssl_mask_rate": payload.get("ssl_mask_rate"),
            "csdi_crps": imputation.get("csdi_crps"),
            "linear_crps": imputation.get("linear_crps"),
            "ssl_crps": imputation.get("ssl_crps"),
            "forecast": {
                arm: forecast[arm]
                for arm, _, _ in SERIES
                if arm in forecast
            },
        }
        supervised = load_supervised(run)
        if supervised and "zeroshot_csdi" in supervised.get("forecast", {}):
            row["supervised_forecast"] = supervised["forecast"]["zeroshot_csdi"]
            row["supervised_crps"] = supervised.get("imputation", {}).get("csdi_crps")
        points.append(row)
    points.sort(key=lambda r: r["x"])
    return points


def threshold_hint(xs: list[float], ys: list[float]) -> float | None:
    """X at the largest positive jump in y (quality drop for error metrics)."""
    if len(xs) < 2:
        return None
    deltas = np.diff(ys)
    # For sparsity axis where left is sparser: look at largest increase in error
    # when moving toward sparser (decreasing x for keep/observe fraction).
    # Report the x of the worse side of the steepest rise in error.
    idx = int(np.argmax(np.abs(deltas)))
    return float(xs[idx + 1] if deltas[idx] > 0 else xs[idx])


def plot_sweep(points: list[dict], *, mode: str, sweep: str, output: Path) -> dict:
    if not points:
        return {}
    figure = Figure(figsize=(10.5, 4.2), dpi=160)
    figure.patch.set_facecolor(BACKGROUND)
    ax_f = figure.add_subplot(1, 2, 1)
    ax_i = figure.add_subplot(1, 2, 2)
    for axis in (ax_f, ax_i):
        axis.set_facecolor(PANEL)
        for spine in axis.spines.values():
            spine.set_color(BORDER)
        axis.tick_params(colors=TEXT, labelsize=9)
        axis.grid(True, color=BORDER, alpha=0.45, linewidth=0.6)

    xs = [p["x"] for p in points]
    metric = "mase" if any("mase" in p["forecast"].get("zeroshot_csdi", {}) for p in points) else "mae"

    for arm, label, color in SERIES:
        ys = [
            p["forecast"][arm][metric]
            for p in points
            if arm in p["forecast"] and metric in p["forecast"][arm]
        ]
        x_arm = [
            p["x"]
            for p in points
            if arm in p["forecast"] and metric in p["forecast"][arm]
        ]
        if len(ys) == len(x_arm) and ys:
            ax_f.plot(x_arm, ys, marker="o", color=color, label=label, linewidth=2)

    if any("supervised_forecast" in p for p in points):
        ys = [
            p["supervised_forecast"][metric]
            for p in points
            if "supervised_forecast" in p and metric in p["supervised_forecast"]
        ]
        x_s = [
            p["x"]
            for p in points
            if "supervised_forecast" in p and metric in p["supervised_forecast"]
        ]
        if ys:
            ax_f.plot(
                x_s,
                ys,
                marker="s",
                color="#3d5a80",
                linestyle="--",
                label="Supervised CSDI → Chronos",
                linewidth=2,
            )

    ax_f.set_xlabel(xlabel_for(sweep, mode), color=TEXT)
    ax_f.set_ylabel(metric.upper(), color=TEXT)
    ax_f.set_title(
        f"Forecast ({mode}, sweep {sweep})",
        color=TEXT,
        fontsize=12,
        loc="left",
    )
    ax_f.legend(frameon=False, fontsize=8)

    csdi = [p["csdi_crps"] for p in points if p["csdi_crps"] is not None]
    x_c = [p["x"] for p in points if p["csdi_crps"] is not None]
    linear = [p["linear_crps"] for p in points if p["linear_crps"] is not None]
    x_l = [p["x"] for p in points if p["linear_crps"] is not None]
    if csdi:
        ax_i.plot(x_c, csdi, marker="o", color=ACCENT, label="CSDI CRPS", linewidth=2)
    if linear:
        ax_i.plot(
            x_l, linear, marker="o", color=ACCENT2, label="Linear CRPS", linewidth=2
        )
    ssl = [p["ssl_crps"] for p in points if p.get("ssl_crps") is not None]
    x_ssl = [p["x"] for p in points if p.get("ssl_crps") is not None]
    if ssl:
        ax_i.plot(
            x_ssl, ssl, marker="^", color="#5c4d7a", label="SSL-holdout CRPS", linewidth=2
        )
    if any(p.get("supervised_crps") is not None for p in points):
        ys = [p["supervised_crps"] for p in points if p.get("supervised_crps") is not None]
        x_s = [p["x"] for p in points if p.get("supervised_crps") is not None]
        ax_i.plot(
            x_s,
            ys,
            marker="s",
            color="#3d5a80",
            linestyle="--",
            label="Supervised CSDI CRPS",
            linewidth=2,
        )
    ax_i.set_xlabel(xlabel_for(sweep, mode), color=TEXT)
    ax_i.set_ylabel("CRPS", color=TEXT)
    ax_i.set_title(
        f"Imputation ({mode}, sweep {sweep})",
        color=TEXT,
        fontsize=12,
        loc="left",
    )
    ax_i.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(output, facecolor=figure.get_facecolor())
    # Threshold hint from self-sup forecast curve.
    selfsup_ys = [
        p["forecast"]["zeroshot_csdi"][metric]
        for p in points
        if "zeroshot_csdi" in p["forecast"] and metric in p["forecast"]["zeroshot_csdi"]
    ]
    selfsup_xs = [
        p["x"]
        for p in points
        if "zeroshot_csdi" in p["forecast"] and metric in p["forecast"]["zeroshot_csdi"]
    ]
    return {
        "mode": mode,
        "sweep": sweep,
        "metric": metric,
        "points": points,
        "threshold_hint_x": threshold_hint(selfsup_xs, selfsup_ys),
        "figure": str(output),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write plots (default: --root)",
    )
    args = parser.parse_args(argv)
    out_dir = args.output_dir or args.root
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.root)
    if not runs:
        raise SystemExit(f"no elec_selfsup_* results under {args.root}")
    summary = {"runs_found": len(runs), "curves": []}
    for mode in ("block", "random"):
        for sweep in ("A", "B"):
            points = collect_points(runs, mode, sweep)
            if not points:
                continue
            path = out_dir / f"thresholds_{mode}_{sweep}.png"
            # Strip heavy run paths from JSON summary copies.
            info = plot_sweep(points, mode=mode, sweep=sweep, output=path)
            slim_points = []
            for p in info.get("points", []):
                slim = {k: v for k, v in p.items() if k != "run"}
                slim_points.append(slim)
            summary["curves"].append(
                {
                    "mode": mode,
                    "sweep": sweep,
                    "metric": info.get("metric"),
                    "threshold_hint_x": info.get("threshold_hint_x"),
                    "figure": info.get("figure"),
                    "n_points": len(slim_points),
                    "points": slim_points,
                }
            )
            print(f"wrote {path} ({len(points)} points)")
    summary_path = out_dir / "thresholds_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
