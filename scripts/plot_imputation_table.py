"""Render TimeAutoDiff imputation CRPS scores as a table image.

    uv run --with matplotlib python scripts/plot_imputation_table.py

Reads the JSON written by `run_timeautodiff.py --output` and adds linear
interpolation, a deterministic baseline whose CRPS equals its mean absolute
error, scored on the same alternate-mask test windows.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from plot_forecast_table import describe_datasets, render_table

from densiffusion.conditioning import prepare_targets
from densiffusion.data import make_windows, split_data


def linear_interpolation_crps(results: dict) -> float:
    if results["mask"] != "alternate":
        raise ValueError("Linear interpolation baseline expects the alternate mask")
    with np.load(results["data"], allow_pickle=False) as dataset:
        values = dataset["values"]
    test = split_data(values, train_fraction=0.7, validation_fraction=0.15)[2]
    windows = make_windows(
        test,
        window_length=int(results["window_length"]),
        stride=int(results["stride"]),
    )
    target, hidden = prepare_targets(windows)
    positions = np.arange(target.shape[1])
    known = positions[::2]
    errors = [
        np.abs(np.interp(positions, known, series[::2, 0]) - series[:, 0])[mask[:, 0]]
        for series, mask in zip(target, hidden, strict=True)
    ]
    return float(np.concatenate(errors).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/timeautodiff_gp.json"),
            Path("outputs/brownian_timeautodiff_gp.json"),
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/imputation_table.png")
    )
    args = parser.parse_args()

    runs = [json.loads(path.read_text()) for path in args.results]
    datasets = []
    for results in runs:
        best = results["gp_posterior_crps"]
        model = results["timeautodiff_crps"]
        linear = linear_interpolation_crps(results)
        datasets.append(
            [(best, None), (model, model / best - 1), (linear, linear / best - 1)]
        )
    labels, subtitle = describe_datasets(results["data"] for results in runs)
    render_table(
        [
            "Exact GP posterior\n(best possible)",
            "TimeAutoDiff ×2\ndensified",
            "Linear interpolation",
        ],
        list(zip(labels, datasets, strict=True)),
        candidates=(1,),
        heading="Imputer",
        title="TimeAutoDiff imputation",
        subtitle=subtitle,
        output=args.output,
    )


if __name__ == "__main__":
    main()
