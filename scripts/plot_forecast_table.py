"""Render TimeAutoDiff forecast CRPS scores as a table image.

    uv run --with matplotlib python scripts/plot_forecast_table.py

Reads results.json from paired observed-grid and imputed-grid runs of
`forecast_timeautodiff.py`, one pair per dataset. Each forecaster's
"vs best" compares it with the exact GP forecast scored on the same points.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

BACKGROUND = "#f4efe4"
HEADER = "#ebe3d3"
BORDER = "#c9bfae"
TEXT = "#3a3a3a"
KERNELS = {
    "exponential": ("Exponential", "Ornstein–Uhlenbeck (exponential-kernel GP)"),
    "brownian": ("Brownian", "Brownian motion"),
    "fractional_brownian": ("fBm", "Fractional Brownian motion"),
    "matern_3_2": ("Matérn 3/2", "Matérn 3/2 GP"),
    "matern_5_2": ("Matérn 5/2", "Matérn 5/2 GP"),
    "periodic": ("Periodic", "Periodic-kernel GP"),
}


def load(path: Path) -> dict:
    return json.loads((path / "results.json").read_text())


def describe_dataset(path: Path) -> tuple[str, str]:
    """Return a column label and a one-line description of a generated dataset."""
    with np.load(path, allow_pickle=False) as dataset:
        n_series, n_steps = dataset["values"].shape[:2]
        config = json.loads(str(dataset["config"]))
    params = dict(config["params"])
    kernel = params.pop("kernel", config["generator"])
    label, name = KERNELS.get(kernel, (kernel, kernel))
    if "hurst" in params:
        label = f"{label} H={params['hurst']:g}"
    details = "".join(
        f", {key} {value:g}" for key, value in params.items() if key != "mean"
    )
    return label, (
        f"{label}: {name}{details}; "
        f"{n_series} series × {n_steps} steps, dt = {config['dt']:g}"
    )


def describe_datasets(paths) -> tuple[list[str], str]:
    labels, lines = zip(*(describe_dataset(Path(path)) for path in paths), strict=True)
    return list(labels), "\n".join((*lines, "Every 2nd step observed."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--observed",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/timeautodiff_forecast_observed"),
            Path("outputs/brownian_forecast_observed"),
        ],
    )
    parser.add_argument(
        "--imputed",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/timeautodiff_forecast"),
            Path("outputs/brownian_forecast"),
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/forecast_table.png")
    )
    args = parser.parse_args()
    if len(args.observed) != len(args.imputed):
        parser.error("Pass one --imputed run per --observed run")

    datasets = []
    for observed, imputed in zip(
        map(load, args.observed), map(load, args.imputed), strict=True
    ):
        best = observed["gp_forecast_crps"]
        model = observed["timeautodiff_forecast_crps"]
        dense = imputed["timeautodiff_forecast_crps"]
        last = observed["last_value_crps"]
        datasets.append(
            [
                (best, None),
                (model, model / best - 1),
                (dense, dense / imputed["gp_forecast_crps"] - 1),
                (last, last / best - 1),
            ]
        )
    labels, subtitle = describe_datasets(load(path)["data"] for path in args.observed)
    render_table(
        [
            "Exact GP (best possible)",
            "TimeAutoDiff direct\n(sparse)",
            "TimeAutoDiff ×2\ndensified*",
            "Last observed value",
        ],
        list(zip(labels, datasets, strict=True)),
        candidates=(1, 2),
        heading="Forecaster",
        title="TimeAutoDiff forecasting",
        subtitle=subtitle,
        footnote=(
            "* Scored on all dense points and compared with the exact GP there.\n"
            "Other rows are scored on the observed times only."
        ),
        output=args.output,
    )


def format_gap(gap: float | None) -> str:
    if gap is None:
        return "—"
    return f"+{100 * gap:.0f}%" if gap > 0.1 else f"+{100 * gap:.1f}%"


def render_table(
    names, datasets, *, candidates, heading, title, subtitle, output, footnote=None
) -> None:
    """Draw one CRPS and "vs best" column pair per (label, [(score, gap)]) dataset.

    In each dataset, the row in `candidates` with the smallest gap is bolded.
    """
    rows = [
        [name]
        + [
            cell
            for _, scores in datasets
            for cell in (f"{scores[row][0]:.4f}", format_gap(scores[row][1]))
        ]
        for row, name in enumerate(names)
    ]
    columns = [heading] + [
        text for label, _ in datasets for text in (f"{label}\nCRPS", "vs\nbest")
    ]
    winners = [
        min(candidates, key=lambda row, scores=scores: scores[row][1])
        for _, scores in datasets
    ]
    draw_table(
        rows,
        columns,
        bold={
            (winner, column)
            for index, winner in enumerate(winners)
            for column in (1 + 2 * index, 2 + 2 * index)
        },
        widths=[0.58, 0.42] * len(datasets),
        title=title,
        subtitle=subtitle,
        footnote=footnote,
        output=output,
    )


def draw_table(
    rows, columns, *, bold, widths, title, subtitle, output, footnote=None
) -> None:
    """Draw text rows under bold column labels; `bold` holds (row, column) cells.

    The first column holds row names; `widths` are relative widths of the rest.
    """
    width = max(7.0, 2.8 + 2.9 * sum(widths))
    subtitle_lines = subtitle.count("\n") + 1
    margin = 0.6 if footnote else 0.15
    header = 0.55 + 0.2 * subtitle_lines
    height = 0.75 * (len(rows) + 1) + margin + header
    figure = Figure(figsize=(width, height), facecolor=BACKGROUND)
    left = 0.2 / width
    axes = figure.add_axes(
        (left, margin / height, 1 - 2 * left, 1 - (margin + header) / height)
    )
    figure.text(
        left, 1 - 0.15 / height, title, fontsize=15, weight="bold", color=TEXT, va="top"
    )
    figure.text(left, 1 - 0.5 / height, subtitle, fontsize=9.5, color=TEXT, va="top")
    axes.axis("off")
    first = 2.6 / (width - 0.4)
    table = axes.table(
        cellText=rows,
        colLabels=columns,
        colWidths=[first] + [(1 - first) * w / sum(widths) for w in widths],
        cellLoc="left",
        loc="upper center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(1.2)
        cell.set_facecolor(HEADER if row == 0 else BACKGROUND)
        text = cell.get_text()
        text.set_color(TEXT)
        text.set_fontsize(13)
        cell.PAD = 0.05
        if row == 0:
            text.set_weight("bold")
            text.set_horizontalalignment("center")
            cell._loc = "center"
        elif (row - 1, column) in bold:
            text.set_weight("bold")
    if footnote:
        figure.text(left, 0.15 / height, footnote, fontsize=9, color=TEXT, va="bottom")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor=BACKGROUND)
    print(output)


if __name__ == "__main__":
    main()
