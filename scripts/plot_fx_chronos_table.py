"""Render FX Chronos-2 sparse vs imputed forecast scores as a table image.

    uv run --with matplotlib python scripts/plot_fx_chronos_table.py \
        --results outputs/fx_csdi_chronos2/results.json \
        --output outputs/fx_csdi_chronos2/table.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

BACKGROUND = "#f4efe4"
HEADER = "#ebe3d3"
BORDER = "#c9bfae"
TEXT = "#3a3a3a"

ARMS = (
    ("sparse", "Chronos-2 fine-tuned\n(sparse)"),
    ("csdi_imputed", "Chronos-2 fine-tuned\n(CSDI imputed)"),
    ("linear_imputed", "Chronos-2 fine-tuned\n(linear imputed)"),
    ("zeroshot_sparse", "Chronos-2 zero-shot\n(sparse)"),
    ("zeroshot_csdi", "Chronos-2 zero-shot\n(CSDI imputed)"),
)
METRICS = (("mae", "MAE"), ("rmse", "RMSE"), ("smape", "sMAPE"), ("wql", "WQL"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    forecast = payload["forecast"]
    rows = [arm for arm, _ in ARMS if arm in forecast]
    if not rows:
        raise SystemExit("no forecast arms found in results.json")

    figure = Figure(figsize=(1.6 * (1 + len(METRICS)), 0.55 * (2 + len(rows))), dpi=150)
    axis = figure.add_axes((0.02, 0.02, 0.96, 0.96))
    axis.set_axis_off()
    axis.set_xlim(0, 1 + len(METRICS))
    axis.set_ylim(0, 2 + len(rows))
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)

    def cell(x, y, text, *, header=False, weight="normal"):
        axis.text(
            x + 0.5,
            y + 0.5,
            text,
            ha="center",
            va="center",
            fontsize=9 if not header else 10,
            color=TEXT,
            fontweight=weight,
            linespacing=1.2,
        )

    cell(0, 1 + len(rows), "Method", header=True, weight="bold")
    for index, (_, label) in enumerate(METRICS):
        cell(1 + index, 1 + len(rows), label, header=True, weight="bold")
    axis.fill_between(
        [0, 1 + len(METRICS)],
        1 + len(rows),
        2 + len(rows),
        color=HEADER,
        linewidth=0,
        zorder=0,
    )

    for row, arm in enumerate(rows):
        y = len(rows) - row
        label = dict(ARMS)[arm]
        cell(0, y, label)
        for col, (key, _) in enumerate(METRICS):
            cell(1 + col, y, f"{forecast[arm][key]:.4f}")

    for x in range(1 + len(METRICS) + 1):
        axis.axvline(x, color=BORDER, linewidth=0.8)
    for y in range(2 + len(rows) + 1):
        axis.axhline(y, color=BORDER, linewidth=0.8)

    title = "FX Chronos-2: sparse vs imputed"
    if "imputation" in payload:
        title += (
            f"  |  imput. CRPS CSDI {payload['imputation']['csdi_crps']:.4f} "
            f"/ linear {payload['imputation']['linear_crps']:.4f}"
        )
    axis.set_title(title, fontsize=11, color=TEXT, pad=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor=BACKGROUND)
    print(args.output)


if __name__ == "__main__":
    main()
