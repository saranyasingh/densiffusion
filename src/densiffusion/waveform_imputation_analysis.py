"""Hidden-point analysis for fresh waveform imputation validation experiments.

This module reads only the supplied attempt directory. Validation controls the
predeclared development gate; optional resolution-transfer results are reported
separately and never enter that gate. Original sample archives are read only.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from densiffusion.contracts import (
    positive_int,
    validate_array,
    validate_mask,
    validate_times,
)
from densiffusion.waveform_analysis import (
    METRICS,
    _band,
    _descriptive,
    _line,
    _panel,
    _svg,
    forecast_score_arrays,
    paired_bootstrap,
)

METHODS = ("learned", "linear")
COLORS = {"learned": "#087f5b", "linear": "#e67700"}


def summarize_imputations(samples, target, mask, trajectory_ids=None):
    """Score hidden entries, grouping repeated windows by independent trajectory.

    ``aggregate`` weights all hidden entries equally. ``trajectory_weighted``
    weights independent trajectories equally, after averaging every hidden
    entry in all windows of each trajectory. The latter is the bootstrap/gate
    estimand. Both agree for one window and a common mask per trajectory.
    """
    target = validate_array(target, (None, None, None))
    samples = validate_array(samples, (len(target), None, *target.shape[1:]))
    mask = validate_mask(mask, target.shape)
    hidden_per_window = mask.sum(axis=(1, 2))
    if np.any(hidden_per_window == 0):
        raise ValueError("Each trajectory window must have hidden entries to score")
    if not (~mask).any(axis=(1, 2)).all():
        raise ValueError("Each trajectory window must retain observed entries")
    ids = (
        np.arange(len(target)) if trajectory_ids is None else np.asarray(trajectory_ids)
    )
    if ids.shape != (len(target),) or ids.dtype.kind not in "iuUS":
        raise ValueError("trajectory_ids must have one integer or string per window")
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    hidden_per_trajectory = np.bincount(inverse, weights=hidden_per_window)
    per_time_counts = mask.sum(axis=(0, 2))
    arrays = forecast_score_arrays(samples, target)
    per_trajectory = {}
    per_time = {}
    for metric, values in arrays.items():
        selected = np.where(mask, values, 0)
        per_trajectory[metric] = (
            np.bincount(inverse, weights=selected.sum(axis=(1, 2)))
            / hidden_per_trajectory
        ).tolist()
        totals = selected.sum(axis=(0, 2))
        per_time[metric] = [
            float(total / count) if count else None
            for total, count in zip(totals, per_time_counts, strict=True)
        ]
    return {
        "aggregate": {
            metric: float(values[mask].mean()) for metric, values in arrays.items()
        },
        "trajectory_weighted": {
            metric: float(np.mean(values)) for metric, values in per_trajectory.items()
        },
        "per_trajectory": per_trajectory,
        "trajectory_ids": unique.tolist(),
        "windows_per_trajectory": counts.tolist(),
        "hidden_entries_per_trajectory": hidden_per_trajectory.astype(int).tolist(),
        "per_time": per_time,
        "hidden_entries_per_time": per_time_counts.tolist(),
        "observations_preserved": bool(
            np.all(np.where(mask[:, None], True, samples == target[:, None]))
        ),
    }


def _analyze_archive(path, *, bootstrap_replicates, seed):
    with np.load(path, allow_pickle=False) as archive:
        target = validate_array(archive["target"], (None, None, 1))
        mask = validate_mask(archive["mask"], target.shape)
        times = validate_array(archive["times"], (target.shape[1],))
        validate_times(times, target.shape[:2])
        ids = archive["trajectory_ids"] if "trajectory_ids" in archive else None
        scores = {
            method: summarize_imputations(archive[method], target, mask, ids)
            for method in METHODS
        }
        for method in METHODS:
            if not scores[method]["observations_preserved"]:
                raise ValueError(f"{path}: {method} changed original observations")
        draw_counts = {method: int(archive[method].shape[1]) for method in METHODS}
        result = {
            "source": str(path.name),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_windows": len(target),
            "n_trajectories": len(scores["learned"]["trajectory_ids"]),
            "draws_per_method": draw_counts,
            "trajectory_id_source": "archive"
            if ids is not None
            else "independent row IDs",
            "times": times.tolist(),
            "scores": scores,
        }
    paired = paired_bootstrap(
        np.asarray(
            [scores["learned"]["per_trajectory"][metric] for metric in METRICS]
        ).T,
        np.asarray(
            [scores["linear"]["per_trajectory"][metric] for metric in METRICS]
        ).T,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    result["learned_minus_linear"] = {
        metric: {
            "difference": paired["difference"][index],
            "ci95": [bound[index] for bound in paired["ci95"]],
        }
        for index, metric in enumerate(METRICS)
    }
    result["bootstrap_seed"] = seed
    return result


def _across_seeds(entries):
    return {
        "n_seeds": len(entries),
        "scope": (
            "Descriptive variation across paired training/dataset seeds; "
            "no pooled inference."
        ),
        "scores": {
            method: {
                metric: _descriptive(
                    [
                        entry["scores"][method]["trajectory_weighted"][metric]
                        for entry in entries
                    ]
                )
                for metric in METRICS
            }
            for method in METHODS
        },
        "learned_minus_linear": {
            metric: _descriptive(
                [
                    entry["learned_minus_linear"][metric]["difference"]
                    for entry in entries
                ]
            )
            for metric in METRICS
        },
    }


def validation_gate(entries, *, minimum_relative_improvement=0.05, expected_seeds=None):
    """Apply the predeclared three-seed CRPS development gate, never transfer scores.

    Relative improvement is one minus the ratio of equal-seed mean learned CRPS
    to equal-seed mean linear CRPS, not the mean of per-seed ratios. Each seed's
    score equally weights independent trajectories. A zero linear score cannot
    pass the meaningful-relative-improvement criterion.
    """
    if (
        not np.isfinite(minimum_relative_improvement)
        or not 0 < minimum_relative_improvement < 1
    ):
        raise ValueError(
            "minimum_relative_improvement must lie strictly between 0 and 1"
        )
    values = list(entries.values())
    learned = [
        entry["scores"]["learned"]["trajectory_weighted"]["crps"] for entry in values
    ]
    linear = [
        entry["scores"]["linear"]["trajectory_weighted"]["crps"] for entry in values
    ]
    learned_mean = float(np.mean(learned)) if learned else None
    linear_mean = float(np.mean(linear)) if linear else None
    improvement = (
        1 - learned_mean / linear_mean if linear_mean and linear_mean > 0 else None
    )
    observed_seeds = sorted(int(name.removeprefix("seed_")) for name in entries)
    expected = sorted(expected_seeds) if expected_seeds is not None else None
    criteria = {
        "exactly_three_paired_seeds": len(values) == 3,
        "configured_seed_inventory_complete": expected is None
        or observed_seeds == expected,
        "learned_crps_lower_in_every_seed": bool(values)
        and all(left < right for left, right in zip(learned, linear, strict=True)),
        "mean_relative_crps_reduction_at_least_threshold": improvement is not None
        and improvement >= minimum_relative_improvement,
        "every_seed_ci95_upper_below_zero": bool(values)
        and all(
            entry["learned_minus_linear"]["crps"]["ci95"][1] < 0 for entry in values
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "minimum_relative_improvement": minimum_relative_improvement,
        "mean_learned_crps": learned_mean,
        "mean_linear_crps": linear_mean,
        "relative_crps_reduction": improvement,
        "observed_seeds": observed_seeds,
        "expected_seeds": expected,
        "scope": (
            "Predeclared validation-only development decision, "
            "not a confirmatory test claim."
        ),
        "ratio_definition": (
            "1 - equal-seed mean learned trajectory CRPS / "
            "equal-seed mean linear trajectory CRPS"
        ),
        "transfer_scores_used": False,
    }


def _training_review(folder):
    path = folder / "training.jsonl"
    if not path.is_file():
        return {"records": [], "selected_epoch": None}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    candidates = [row for row in rows if row.get("selection_score") is not None]
    return {
        "records": rows,
        "selected_epoch": min(candidates, key=lambda row: row["selection_score"])[
            "epoch"
        ]
        if candidates
        else None,
        "scope": (
            "Within-model training behavior; endpoint summaries do not "
            "certify convergence."
        ),
    }


def analyze_imputation_run(
    output_dir,
    bootstrap_replicates=10000,
    seed=8675309,
    minimum_relative_improvement=0.05,
):
    """Save hidden-only metrics, paired intervals, gate, text report, and SVGs.

    Read ``seed_*/validation_samples.npz``. Use :func:`analyze_transfer_run` for
    later transfer samples without modifying the frozen validation analysis.
    Archives contain target (N,T,1), boolean mask (True hidden), learned/linear
    ensembles (N,S,T,1), shared times (T,), and optional trajectory_ids (N,).
    Without IDs each row is assumed to be an independent trajectory. All source
    samples and observations remain unchanged; no model or generator is called.
    """
    positive_int("bootstrap_replicates", bootstrap_replicates)
    root = Path(output_dir)
    if (root / "DENSIFIERS_FROZEN.json").is_file():
        raise FileExistsError("Frozen validation analysis must not be overwritten")
    folders = sorted(
        [
            folder
            for folder in root.glob("seed_*")
            if (folder / "validation_samples.npz").is_file()
        ],
        key=lambda folder: int(folder.name.removeprefix("seed_")),
    )
    if not folders:
        raise ValueError(f"No seed_*/validation_samples.npz files in {root}")
    result = _analysis_metadata(bootstrap_replicates, seed)
    result.update(validation={"seeds": {}}, training={}, tasks=["validation"])
    for index, folder in enumerate(folders):
        result["validation"]["seeds"][folder.name] = _analyze_archive(
            folder / "validation_samples.npz",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 100 * index,
        )
        result["training"][folder.name] = _training_review(folder)
        result["training"][folder.name]["linear_validation_crps"] = result[
            "validation"
        ]["seeds"][folder.name]["scores"]["linear"]["aggregate"]["crps"]
    config_path = root / "config.json"
    configured_seeds = None
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        configured_seeds = config.get("experiment", config).get("seeds")
    result["validation_gate"] = validation_gate(
        result["validation"]["seeds"],
        minimum_relative_improvement=minimum_relative_improvement,
        expected_seeds=configured_seeds,
    )
    plots = root / "imputation_plots"
    plots.mkdir(exist_ok=True)
    _finish_task(root, result, "validation", "validation_samples.npz")
    training_plot = plots / "training_curves.svg"
    if _training_plot(result["training"], training_plot):
        result["plots"].append(str(training_plot.relative_to(root)))
    (root / "imputation_analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (root / "imputation_report.txt").write_text(_report_text(result))
    return result


def _analysis_metadata(bootstrap_replicates, seed):
    return {
        "analysis_version": 1,
        "metric_scope": "Hidden entries only; known observations excluded.",
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "base_seed": seed,
            "unit": "independent trajectory, retaining hidden entries and draws",
            "scope": (
                "Conditional on fitted models and saved ensembles; "
                "training-seed variation separate."
            ),
            "multiple_comparisons": (
                "Intervals are unadjusted across metrics, seeds, "
                "and validation attempts."
            ),
        },
        "interval_definition": (
            "Empirical 5th and 95th percentiles, with linear interpolation; "
            "finite draws affect coverage."
        ),
        "linear_baseline": (
            "Deterministic linear interpolation has zero-width intervals; "
            "its coverage is not a probabilistic calibration benchmark."
        ),
        "validation_reuse": (
            "Repeated candidate selection on the same validation set induces "
            "selection optimism; the gate is not a held-out test claim."
        ),
        "plots": [],
    }


def _finish_task(root, result, name, filename):
    entries = result[name]["seeds"]
    result[name]["across_seed_descriptive"] = _across_seeds(list(entries.values()))
    first = root / next(iter(entries)) / filename
    plots = root / "imputation_plots"
    plots.mkdir(exist_ok=True)
    reconstruction = plots / f"{name}_reconstructions.svg"
    curve = plots / f"{name}_per_time.svg"
    _reconstruction_plot(first, reconstruction, name)
    _score_curve_plot(entries, curve, name)
    result["plots"].extend(
        [str(path.relative_to(root)) for path in (reconstruction, curve)]
    )


def analyze_transfer_run(output_dir, bootstrap_replicates=10000, seed=8675309):
    """Analyze both transfer tasks without changing frozen validation artifacts.

    Reads transfer_samples.npz (history) and transfer_future_samples.npz
    (anchored future), writing transfer_analysis.json and transfer_report.txt.
    This reports metrics only; transfer gate decisions belong to the protocol.
    """
    positive_int("bootstrap_replicates", bootstrap_replicates)
    root = Path(output_dir)
    result = _analysis_metadata(bootstrap_replicates, seed)
    result.update(
        tasks=[],
        scope="Separate transfer tasks; no validation gate recomputed or modified.",
    )
    task_files = {
        "transfer_history": "transfer_samples.npz",
        "transfer_future": "transfer_future_samples.npz",
    }
    folders = sorted(
        root.glob("seed_*"),
        key=lambda folder: int(folder.name.removeprefix("seed_")),
    )
    for task_index, (name, filename) in enumerate(task_files.items()):
        entries = {
            folder.name: _analyze_archive(
                folder / filename,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + 100 * index + task_index + 1,
            )
            for index, folder in enumerate(folders)
            if (folder / filename).is_file()
        }
        if entries:
            result[name] = {"seeds": entries}
            result["tasks"].append(name)
            _finish_task(root, result, name, filename)
    if not result["tasks"]:
        raise ValueError(f"No transfer sample archives in {root}")
    (root / "transfer_analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (root / "transfer_report.txt").write_text(_report_text(result))
    return result


def _reconstruction_plot(source, destination, split):
    with np.load(source, allow_pickle=False) as archive:
        target = archive["target"]
        mask = validate_mask(archive["mask"], target.shape)
        times = archive["times"]
        ids = (
            archive["trajectory_ids"]
            if "trajectory_ids" in archive
            else np.arange(len(target))
        )
        _, first = np.unique(ids, return_index=True)
        rows = np.sort(first)[:6]
        ensembles = {method: archive[method] for method in METHODS}
    svg = _svg(
        1120,
        95 + 255 * len(rows),
        f"{split.capitalize()} imputation: first {len(rows)} trajectories",
    )
    svg.append(
        '<text x="32" y="53">Black dots: observed; dashed gray: target; '
        "color: reconstruction mean; band: empirical central 90% interval.</text>"
    )
    for row, index in enumerate(rows):
        bands = {
            method: np.quantile(ensembles[method][index, :, :, 0], [0.05, 0.95], axis=0)
            for method in METHODS
        }
        all_values = np.r_[
            target[index, :, 0], *(band.ravel() for band in bands.values())
        ]
        ylim = (float(all_values.min()), float(all_values.max()))
        for column, method in enumerate(METHODS):
            transform = _panel(
                svg,
                (column * 560, 75 + row * 255, 555, 255),
                f"{method.capitalize()} | trajectory {ids[index]}",
                times,
                all_values,
                ylim=ylim,
            )
            _band(svg, transform, times, *bands[method], COLORS[method])
            _line(svg, transform, times, target[index, :, 0], "#495057", dashed=True)
            _line(
                svg,
                transform,
                times,
                ensembles[method][index, :, :, 0].mean(axis=0),
                COLORS[method],
            )
            known = ~mask[index, :, 0]
            px, py = transform(times[known], target[index, known, 0])
            for x, y in zip(px, py, strict=True):
                svg.append(f'<circle cx="{x}" cy="{y}" r="2.4" fill="#212529"/>')
    destination.write_text("\n".join([*svg, "</svg>"]) + "\n")


def _score_curve_plot(entries, destination, split):
    values = list(entries.values())
    times = np.asarray(values[0]["times"])
    if not all(np.array_equal(times, entry["times"]) for entry in values):
        raise ValueError("Measurement grids differ across seeds")
    svg = _svg(
        1120, 760, f"{split.capitalize()} hidden-point scores by measurement time"
    )
    svg.append(
        '<text x="32" y="53">Green: learned; orange: linear. Lines: seed means; '
        "bands: descriptive seed range, not confidence intervals.</text>"
    )
    for index, metric in enumerate(METRICS):
        curves = {}
        for method in METHODS:
            raw = np.asarray(
                [entry["scores"][method]["per_time"][metric] for entry in values],
                dtype=float,
            )
            valid = np.isfinite(raw).any(axis=0)
            curves[method] = (
                times[valid],
                np.nanmean(raw[:, valid], axis=0),
                np.nanmin(raw[:, valid], axis=0),
                np.nanmax(raw[:, valid], axis=0),
            )
        all_y = np.concatenate([np.r_[curve[2], curve[3]] for curve in curves.values()])
        if metric == "coverage_90":
            all_y = np.r_[all_y, 0.9]
        transform = _panel(
            svg,
            ((index % 2) * 560, 80 + (index // 2) * 330, 555, 320),
            metric,
            times,
            all_y,
            ylabel=metric,
        )
        if metric == "coverage_90":
            _line(
                svg, transform, times, np.full(len(times), 0.9), "#868e96", dashed=True
            )
        for method, (xs, mean, lower, upper) in curves.items():
            _band(svg, transform, xs, lower, upper, COLORS[method])
            _line(svg, transform, xs, mean, COLORS[method])
    destination.write_text("\n".join([*svg, "</svg>"]) + "\n")


def _training_plot(training, destination):
    entries = [(name, review) for name, review in training.items() if review["records"]]
    if not entries:
        return False
    svg = _svg(
        1120, 115 + 310 * len(entries), "Imputation training and validation curves"
    )
    svg.append(
        '<text x="32" y="53">Left: train noise MSE (green), validation noise '
        "MSE (gray). Right: original-grid validation CRPS candidates.</text>"
    )
    svg.extend(
        [
            f'<path d="M 32 73 h 28" stroke="{COLORS["learned"]}" stroke-width="2"/>',
            '<text x="68" y="77">Learned validation CRPS</text>',
            f'<path d="M 280 73 h 28" stroke="{COLORS["linear"]}" '
            'stroke-width="2" stroke-dasharray="6 4"/>',
            '<text x="316" y="77">Fixed linear validation CRPS baseline</text>',
        ]
    )
    for index, (name, review) in enumerate(entries):
        rows = review["records"]
        curves = {
            key: [(row["epoch"], row[key]) for row in rows if row.get(key) is not None]
            for key in ("train_loss", "validation_loss", "selection_score")
        }
        for column, keys in enumerate(
            (("train_loss", "validation_loss"), ("selection_score",))
        ):
            all_points = [point for key in keys for point in curves[key]]
            if not all_points:
                continue
            baseline = review.get("linear_validation_crps") if column == 1 else None
            baseline_epochs = [
                min(row["epoch"] for row in rows),
                max(row["epoch"] for row in rows),
            ]
            if baseline is not None:
                all_points.extend((epoch, baseline) for epoch in baseline_epochs)
            xs, ys = np.asarray(all_points).T
            label = "noise MSE" if column == 0 else "validation CRPS"
            transform = _panel(
                svg,
                (column * 560, 95 + index * 310, 555, 305),
                name if baseline is None else f"{name} | linear CRPS {baseline:.4f}",
                xs,
                ys,
                xlabel="Epoch",
                ylabel=label,
            )
            if baseline is not None:
                _line(
                    svg,
                    transform,
                    baseline_epochs,
                    [baseline, baseline],
                    COLORS["linear"],
                    dashed=True,
                )
            for key in keys:
                if not curves[key]:
                    continue
                xs, ys = np.asarray(curves[key]).T
                color = "#495057" if key == "validation_loss" else COLORS["learned"]
                _line(svg, transform, xs, ys, color)
                if key == "selection_score":
                    px, py = transform(xs, ys)
                    for x, y in zip(px, py, strict=True):
                        svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
    destination.write_text("\n".join([*svg, "</svg>"]) + "\n")
    return True


def _report_text(result):
    lines = ["Fresh waveform imputation review", ""]
    if "validation_gate" in result:
        gate = result["validation_gate"]
        threshold = 100 * gate["minimum_relative_improvement"]
        lines.extend(
            [
                f"Validation gate: {'PASS' if gate['passed'] else 'FAIL'}.",
                "Criteria: exactly three paired seeds; learned lower in every seed;",
                f"ratio-of-means relative CRPS reduction >= {threshold:.1f}%;",
                "every paired bootstrap 95% upper CRPS difference below zero.",
                f"Criteria results: {json.dumps(gate['criteria'], sort_keys=True)}",
                "",
            ]
        )
    for split in result["tasks"]:
        lines.append(f"{split.capitalize()} (hidden entries only):")
        for method, scores in result[split]["across_seed_descriptive"][
            "scores"
        ].items():
            lines.append(
                f"  {method}: "
                + "; ".join(
                    f"{metric}={summary['mean']:.6f}, seed SD={summary['sample_std']}"
                    for metric, summary in scores.items()
                )
            )
        for seed_name, entry in result[split]["seeds"].items():
            difference = entry["learned_minus_linear"]["crps"]
            lines.append(
                f"  {seed_name} learned-linear CRPS={difference['difference']:+.6f}, "
                f"95% CI [{difference['ci95'][0]:+.6f}, "
                f"{difference['ci95'][1]:+.6f}]"
            )
        lines.append("")
    lines.extend(
        [
            "Bootstrap units are independent trajectories, conditional on models",
            "and saved ensembles; seed variation is described separately.",
            "Intervals are unadjusted across metrics/seeds/attempts. Finite-ensemble",
            "quantiles affect coverage; linear intervals have zero width.",
            "Repeated validation selection can be optimistic; this is no test claim.",
            "Transfer results do not enter the validation gate.",
            "All source observations and sample archives remain unchanged.",
            "Full scores and trajectories: imputation_analysis.json (validation)",
            "or transfer_analysis.json (transfer).",
            "Plots: " + ", ".join(result["plots"]),
        ]
    )
    return "\n".join(lines) + "\n"
