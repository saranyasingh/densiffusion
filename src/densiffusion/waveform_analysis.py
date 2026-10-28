"""Recompute original-grid waveform forecast scores and paired uncertainty.

Bootstrap units are independent trajectories, retaining all windows and forecast
draws belonging to each unit. Intervals condition on the fitted models and the
saved ensembles; variation across training seeds is reported separately.
"""

import json
from html import escape
from pathlib import Path

import numpy as np

from densiffusion.contracts import positive_int, validate_array, validate_times
from densiffusion.metrics import evaluate_forecast

METHODS = ("direct_csdi", "linear_densification", "densiffusion")
METRICS = ("crps", "mse", "coverage_90", "interval_width_90")
COMPARISONS = (
    ("densiffusion", "direct_csdi"),
    ("densiffusion", "linear_densification"),
    ("linear_densification", "direct_csdi"),
)
COLORS = ("#1864ab", "#e67700", "#087f5b")
LABELS = ("Direct CSDI", "Linear densification", "Densiffusion")


def forecast_score_arrays(samples, target):
    """Return each metric at each (window, original horizon, channel) entry.

    CRPS is the empirical-ensemble score, including its finite-ensemble term.
    Intervals use linearly interpolated empirical 5th and 95th percentiles.
    These definitions exactly match :func:`metrics.evaluate_forecast`.
    """
    target = validate_array(target, (None, None, None))
    samples = validate_array(samples, (len(target), None, *target.shape[1:]))
    count = samples.shape[1]
    errors = np.sort(samples - target[:, None], axis=1)
    weights = (2 * np.arange(1, count + 1) - count - 1) / count**2
    lower, upper = np.quantile(samples, [0.05, 0.95], axis=1)
    return {
        "crps": np.abs(errors).mean(axis=1)
        - np.sum(errors * weights[None, :, None, None], axis=1),
        "mse": np.square(samples.mean(axis=1) - target),
        "coverage_90": ((lower <= target) & (target <= upper)).astype(float),
        "interval_width_90": upper - lower,
    }


def summarize_forecasts(samples, target, trajectory_ids):
    """Compute aggregate, per-horizon, and per-independent-trajectory scores.

    Aggregate scores weight entries equally, following evaluate_forecast.
    Trajectory scores average all windows, horizons, and channels belonging to
    each trajectory; their mean weights independent trajectories equally.
    Both are reported explicitly if trajectory window counts differ.
    """
    point_scores = forecast_score_arrays(samples, target)
    ids = np.asarray(trajectory_ids)
    if ids.ndim != 1 or ids.shape[0] != len(target) or ids.dtype.kind not in "iuUS":
        raise ValueError("trajectory_ids must contain one integer or string per window")
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    per_trajectory = {}
    for metric, values in point_scores.items():
        window_scores = values.mean(axis=(1, 2))
        per_trajectory[metric] = (
            np.bincount(inverse, weights=window_scores) / counts
        ).tolist()
    return {
        "aggregate": evaluate_forecast(samples, target),
        "per_horizon": {
            metric: values.mean(axis=(0, 2)).tolist()
            for metric, values in point_scores.items()
        },
        "trajectory_ids": unique.tolist(),
        "windows_per_trajectory": counts.tolist(),
        "per_trajectory": per_trajectory,
        "trajectory_weighted": {
            metric: float(np.mean(values)) for metric, values in per_trajectory.items()
        },
    }


def paired_bootstrap(left, right, *, replicates=10000, seed=8675309, chunk_size=256):
    """Percentile CIs for mean paired differences, resampling trajectory rows.

    Arrays have shape (independent trajectories, ...). A single index resample
    is shared across every trailing coordinate, retaining metric dependence.
    The score arrays must already aggregate correlated windows per trajectory.
    """
    positive_int("replicates", replicates)
    positive_int("chunk_size", chunk_size)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim < 1 or left.size == 0:
        raise ValueError("paired scores must have identical nonempty shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired scores must be finite")
    differences = left - right
    rng = np.random.default_rng(seed)
    means = np.empty((replicates, *differences.shape[1:]))
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        indices = rng.integers(len(left), size=(stop - start, len(left)))
        means[start:stop] = differences[indices].mean(axis=1)
    return {
        "difference": differences.mean(axis=0).tolist(),
        "ci95": np.quantile(means, [0.025, 0.975], axis=0).tolist(),
        "replicates": replicates,
        "n_trajectories": len(left),
        "seed": seed,
        "method": "paired trajectory percentile bootstrap",
    }


def _descriptive(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "values": values.tolist(),
    }


def _protocol_status(output_dir, folders):
    """Expose partial configured evaluations without blocking generic archives."""
    config_path = output_dir / "config.json"
    observed = [int(folder.name.removeprefix("seed_")) for folder in folders]
    status = {
        "manifest_available": config_path.is_file(),
        "configured_seed_ids": None,
        "sample_seed_ids": observed,
        "missing_seed_ids": None,
        "unexpected_seed_ids": None,
        "evaluated_marker_present": (output_dir / "EVALUATED.json").is_file(),
        "status": "unverified_without_manifest",
        "scope": "Artifact completeness only; does not certify convergence or quality.",
    }
    if not status["manifest_available"]:
        return status
    manifest = json.loads(config_path.read_text())
    configured = manifest.get("experiment", {}).get("seeds")
    if not isinstance(configured, list) or not configured:
        status["status"] = "unverified_manifest_has_no_seed_list"
        return status
    expected = sorted(configured)
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    status.update(
        configured_seed_ids=expected,
        missing_seed_ids=missing,
        unexpected_seed_ids=unexpected,
        status=(
            "complete"
            if not missing and not unexpected and status["evaluated_marker_present"]
            else "incomplete"
        ),
    )
    return status


def analyze_run(output_dir, bootstrap_replicates=10000, seed=8675309):
    """Analyze seed_*/samples.npz, write analysis.json and standalone SVG plots.

    Required archive keys: history, target, history_times, future_times,
    trajectory_ids, direct_csdi, linear_densification, and densiffusion.
    Plots require only NumPy and the standard library.
    """
    positive_int("bootstrap_replicates", bootstrap_replicates)
    output_dir = Path(output_dir)
    folders = sorted(
        (
            folder
            for folder in output_dir.glob("seed_*")
            if (folder / "samples.npz").is_file()
        ),
        key=lambda folder: int(folder.name.removeprefix("seed_")),
    )
    if not folders:
        raise ValueError(f"No seed_*/samples.npz artifacts found in {output_dir}")
    result = {
        "analysis_version": 1,
        "protocol": _protocol_status(output_dir, folders),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "unit": "independent test trajectory; all its windows kept together",
            "estimand": "equal-trajectory mean metric difference, first minus second",
            "scope": "conditional on each seed's fitted models and saved draws",
            "excludes": "training variability and resampling forecast ensembles",
            "multiple_comparisons": (
                "95% intervals are unadjusted across comparisons and metrics."
            ),
        },
        "metric_definitions": {
            "crps": "empirical ensemble CRPS, lower is better",
            "mse": "ensemble-mean MSE, lower is better",
            "coverage_90": "marginal coverage of empirical central 90% intervals",
            "interval_width_90": "mean empirical 95th minus 5th percentile",
            "quantiles": "linear interpolation; finite scored ensemble size",
            "ensemble_monte_carlo": (
                "Densiffusion draws sharing a sampled dense history are dependent. "
                "Equal draw counts do not imply equal Monte Carlo precision or "
                "finite-ensemble CRPS bias. Bootstrap intervals retain the fixed "
                "saved ensembles."
            ),
        },
        "seeds": {},
    }
    for seed_index, folder in enumerate(folders):
        with np.load(folder / "samples.npz", allow_pickle=False) as archive:
            target = archive["target"]
            scores = {
                method: summarize_forecasts(
                    archive[method], target, archive["trajectory_ids"]
                )
                for method in METHODS
            }
            draw_counts = {method: archive[method].shape[1] for method in METHODS}
            if len(set(draw_counts.values())) != 1:
                raise ValueError(f"Unequal scored ensemble sizes in {folder}")
            entry = {
                "n_windows": len(target),
                "n_trajectories": len(scores[METHODS[0]]["trajectory_ids"]),
                "draws_per_method": draw_counts,
                "scores": scores,
                "paired_differences": {},
            }
        for comparison_index, (left, right) in enumerate(COMPARISONS):
            left_values, right_values = (
                np.asarray([scores[method]["per_trajectory"][m] for m in METRICS]).T
                for method in (left, right)
            )
            uncertainty = paired_bootstrap(
                left_values,
                right_values,
                replicates=bootstrap_replicates,
                seed=seed + 100 * seed_index + comparison_index,
            )
            entry["paired_differences"][f"{left}-minus-{right}"] = {
                metric: {
                    "difference": uncertainty["difference"][metric_index],
                    "ci95": [bound[metric_index] for bound in uncertainty["ci95"]],
                    "bootstrap_seed": uncertainty["seed"],
                }
                for metric_index, metric in enumerate(METRICS)
            }
        timings_path = folder / "timings.json"
        if timings_path.exists():
            entry["timings"] = json.loads(timings_path.read_text())
        entry["training"] = _training_summary(folder)
        result["seeds"][folder.name] = entry
    entries = list(result["seeds"].values())
    result["across_seed_descriptive"] = {
        "n_seeds": len(entries),
        "scope": "descriptive training/dataset seed variation; no pooled inference",
        "scores": {
            method: {
                metric: _descriptive(
                    [entry["scores"][method]["aggregate"][metric] for entry in entries]
                )
                for metric in METRICS
            }
            for method in METHODS
        },
        "paired_differences": {
            comparison: {
                metric: _descriptive(
                    [
                        entry["paired_differences"][comparison][metric]["difference"]
                        for entry in entries
                    ]
                )
                for metric in METRICS
            }
            for comparison in entries[0]["paired_differences"]
        },
    }
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    _forecast_plot(folders[0], plot_dir / "representative_forecasts.svg")
    _horizon_plot(result, plot_dir / "per_horizon.svg")
    result["plots"] = [
        "plots/representative_forecasts.svg",
        "plots/per_horizon.svg",
    ]
    if _validation_plot(result, plot_dir / "validation_crps.svg"):
        result["plots"].append("plots/validation_crps.svg")
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _training_summary(folder):
    """Report selection and endpoint behavior without certifying convergence."""
    path = folder / "training.jsonl"
    records = (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )
    summaries = {}
    for method in ("densifier", *METHODS):
        rows = [record for record in records if record["method"] == method]
        summary = {}
        if rows:
            summary.update(
                final_epoch=rows[-1]["epoch"],
                first_train_loss=rows[0]["train_loss"],
                last_train_loss=rows[-1]["train_loss"],
                minimum_train_loss=min(row["train_loss"] for row in rows),
            )
            selected = [row for row in rows if "selection_score" in row]
            if selected:
                best = min(selected, key=lambda row: row["selection_score"])
                summary.update(
                    selected_epoch=best["epoch"],
                    selection_metric=best["selection_metric"],
                    best_selection_score=best["selection_score"],
                    last_selection_score=selected[-1]["selection_score"],
                    selected_at_final_epoch=best["epoch"] == rows[-1]["epoch"],
                    final_minus_previous_selection_score=(
                        selected[-1]["selection_score"]
                        - selected[-2]["selection_score"]
                        if len(selected) > 1
                        else None
                    ),
                )
        validation_path = folder / f"{method}_validation.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text())
            summary["forecast_validation"] = validation
            if validation:
                best = min(validation, key=lambda row: row["crps"])
                summary["minimum_original_grid_validation_crps"] = best["crps"]
                summary["minimum_original_grid_validation_crps_epoch"] = best["epoch"]
        if summary:
            summaries[method] = summary
    return {
        "interpretation": (
            "Within-method training behavior only; denoising losses across target "
            "grids are not comparable forecast scores. Endpoint summaries do not "
            "establish convergence."
        ),
        "models": summaries,
    }


def _svg(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text {font-family:Arial,sans-serif;fill:#263238;"
        "font-size:12px}</style>",
        '<text x="32" y="30" style="font-size:20px;font-weight:bold">'
        f"{escape(title)}</text>",
    ]


def _panel(svg, box, title, xs, ys, *, ylim=None, xlabel="Time", ylabel="Value"):
    x, y, width, height = box
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = ylim if ylim is not None else (float(np.min(ys)), float(np.max(ys)))
    if xmax <= xmin:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    pad = max((ymax - ymin) * 0.08, 0.01)
    ymin, ymax = ymin - pad, ymax + pad
    left, top, inner_width, inner_height = x + 54, y + 28, width - 68, height - 78

    def transform(a, b):
        a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
        return (
            left + (a - xmin) / (xmax - xmin) * inner_width,
            top + (ymax - b) / (ymax - ymin) * inner_height,
        )

    svg.append(f'<text x="{x + 54}" y="{y + 17}">{escape(title)}</text>')
    for value in np.linspace(ymin, ymax, 5):
        _, py = transform(xmin, value)
        svg.extend(
            [
                f'<path d="M {left} {py} h {inner_width}" stroke="#e9ecef"/>',
                f'<text x="{left - 7}" y="{py + 4}" text-anchor="end">'
                f"{value:.2f}</text>",
            ]
        )
    for value in np.linspace(xmin, xmax, 5):
        px, _ = transform(value, ymin)
        svg.append(
            f'<text x="{px}" y="{top + inner_height + 19}" '
            f'text-anchor="middle">{value:.2f}</text>'
        )
    svg.append(
        f'<path d="M {left} {top} v {inner_height} h {inner_width}" '
        'stroke="#868e96" fill="none"/>'
    )
    svg.append(
        f'<text x="{left + inner_width / 2}" y="{y + height - 6}" '
        f'text-anchor="middle">{escape(xlabel)}</text>'
    )
    svg.append(
        f'<text transform="translate({x + 13},{top + inner_height / 2}) rotate(-90)" '
        f'text-anchor="middle">{escape(ylabel)}</text>'
    )
    return transform


def _line(svg, transform, xs, ys, color, *, dashed=False):
    px, py = transform(xs, ys)
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px, py, strict=True))
    dash = ' stroke-dasharray="5,3"' if dashed else ""
    svg.append(
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        f'stroke-width="1.8"{dash}/>'
    )


def _band(svg, transform, xs, lower, upper, color):
    px, py = transform(np.r_[xs, xs[::-1]], np.r_[lower, upper[::-1]])
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px, py, strict=True))
    svg.append(f'<polygon points="{points}" fill="{color}" fill-opacity="0.16"/>')


def _forecast_plot(folder, destination):
    with np.load(folder / "samples.npz", allow_pickle=False) as archive:
        history = validate_array(archive["history"], (None, None, 1))
        target = validate_array(archive["target"], (len(history), None, 1))
        history_times = validate_times(archive["history_times"], history.shape[:2])
        future_times = validate_times(archive["future_times"], target.shape[:2])
        ids = archive["trajectory_ids"]
        # Fixed ordering, never choose examples based on their forecast score.
        _, first_indices = np.unique(ids, return_index=True)
        rows = np.sort(first_indices)[:6]
        width, row_height = 1320, 235
        svg = _svg(
            width, len(rows) * row_height + 95, "Representative held-out forecasts"
        )
        svg.append(
            '<text x="32" y="53">First six distinct trajectories of the first seed; '
            "black: observed history; dashed black: truth; color: forecast mean; "
            "band: empirical central 90% interval.</text>"
        )
        for row_index, index in enumerate(rows):
            bands = {
                method: np.quantile(
                    archive[method][index, :, :, 0], [0.05, 0.95], axis=0
                )
                for method in METHODS
            }
            ylim_values = np.r_[
                history[index, :, 0],
                target[index, :, 0],
                *(bands[m].ravel() for m in METHODS),
            ]
            ylim = (float(ylim_values.min()), float(ylim_values.max()))
            xs = np.r_[history_times[index], future_times[index]]
            for column, (method, label, color) in enumerate(
                zip(METHODS, LABELS, COLORS, strict=True)
            ):
                transform = _panel(
                    svg,
                    (column * 440, 76 + row_index * row_height, 435, row_height),
                    f"{label} | trajectory {ids[index]}",
                    xs,
                    ylim_values,
                    ylim=ylim,
                )
                _band(svg, transform, future_times[index], *bands[method], color)
                _line(
                    svg,
                    transform,
                    history_times[index],
                    history[index, :, 0],
                    "#212529",
                )
                _line(
                    svg,
                    transform,
                    xs[-len(target[index]) - 1 :],
                    np.r_[history[index, -1, 0], target[index, :, 0]],
                    "#212529",
                    dashed=True,
                )
                _line(
                    svg,
                    transform,
                    future_times[index],
                    archive[method][index, :, :, 0].mean(axis=0),
                    color,
                )
        destination.write_text("\n".join([*svg, "</svg>"]) + "\n")


def _horizon_plot(result, destination):
    svg = _svg(1120, 760, "Original-grid performance by forecast horizon")
    svg.append(
        '<text x="32" y="53">Lines: mean across seeds; '
        "shading: seed minimum to maximum "
        "(descriptive range, not a confidence interval).</text>"
    )
    for index, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        svg.append(
            f'<text x="{32 + index * 290}" y="77" style="fill:{color}">{label}</text>'
        )
    for metric_index, metric in enumerate(METRICS):
        values = {
            method: np.asarray(
                [
                    entry["scores"][method]["per_horizon"][metric]
                    for entry in result["seeds"].values()
                ]
            )
            for method in METHODS
        }
        xs = np.arange(1, values[METHODS[0]].shape[1] + 1)
        all_values = np.concatenate([value.ravel() for value in values.values()])
        ylim = (
            (min(float(all_values.min()), 0.9), max(float(all_values.max()), 0.9))
            if metric == "coverage_90"
            else None
        )
        transform = _panel(
            svg,
            ((metric_index % 2) * 560, 95 + (metric_index // 2) * 325, 555, 320),
            metric,
            xs,
            all_values,
            ylim=ylim,
            xlabel="Original-grid forecast step",
            ylabel=metric,
        )
        if metric == "coverage_90":
            _line(svg, transform, xs, np.full(len(xs), 0.9), "#868e96", dashed=True)
        for method, color in zip(METHODS, COLORS, strict=True):
            _band(
                svg,
                transform,
                xs,
                values[method].min(axis=0),
                values[method].max(axis=0),
                color,
            )
            _line(svg, transform, xs, values[method].mean(axis=0), color)
    destination.write_text("\n".join([*svg, "</svg>"]) + "\n")


def _validation_plot(result, destination):
    entries = [
        (name, entry)
        for name, entry in result["seeds"].items()
        if any(
            model.get("forecast_validation")
            for model in entry["training"]["models"].values()
        )
    ]
    if not entries:
        return False
    svg = _svg(1120, 105 + 325 * len(entries), "Original-grid validation CRPS")
    svg.append(
        '<text x="32" y="53">Fixed validation ensemble evaluation at checkpoint '
        "candidates; minimum selects each method's checkpoint.</text>"
    )
    for index, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        svg.append(
            f'<text x="{32 + index * 290}" y="77" style="fill:{color}">{label}</text>'
        )
    for index, (seed_name, entry) in enumerate(entries):
        curves = {
            method: entry["training"]["models"]
            .get(method, {})
            .get("forecast_validation")
            for method in METHODS
        }
        all_rows = [row for rows in curves.values() if rows for row in rows]
        transform = _panel(
            svg,
            (0, 95 + index * 325, 1115, 320),
            seed_name,
            [row["epoch"] for row in all_rows],
            [row["crps"] for row in all_rows],
            xlabel="Training epoch",
            ylabel="Validation CRPS",
        )
        for method, color in zip(METHODS, COLORS, strict=True):
            rows = curves[method]
            if not rows:
                continue
            xs = np.asarray([row["epoch"] for row in rows])
            ys = np.asarray([row["crps"] for row in rows])
            _line(svg, transform, xs, ys, color)
            px, py = transform(xs, ys)
            for x, y in zip(px, py, strict=True):
                svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
    destination.write_text("\n".join([*svg, "</svg>"]) + "\n")
    return True
