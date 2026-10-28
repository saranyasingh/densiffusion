"""Analyze saved noisy-future forecasts after history-only preprocessing.

No generator, mediator, model, or sampler is called. Paired uncertainty resamples
independent trajectories, retaining their windows and fixed forecast ensembles.
"""

import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np

from densiffusion.contracts import positive_int, validate_array, validate_times
from densiffusion.waveform_analysis import (
    METRICS,
    _band,
    _descriptive,
    _line,
    _svg,
    paired_bootstrap,
    summarize_forecasts,
)

METHODS = ("raw", "learned", "median")
COMPARISONS = (("learned", "raw"), ("learned", "median"), ("median", "raw"))
LABELS = {"raw": "Raw history", "learned": "Learned polishing", "median": "Median (7)"}
COLORS = {"raw": "#1864ab", "learned": "#087f5b", "median": "#e67700"}
MARKERS = ("MEDIATORS_FROZEN.json", "FROZEN.json", "EVALUATED.json")
METHOD_TIMINGS = (
    "preprocessing_seconds",
    "forecaster_training_seconds",
    "validation_sampling_seconds",
    "test_preprocessing_seconds",
    "test_inference_seconds",
)


def _json(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error


def _sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _ids(values, count, label):
    values = np.asarray(values)
    if values.shape != (count,) or values.dtype.kind not in "iuUS":
        raise ValueError(f"{label}: trajectory_ids need one integer/string per window")
    return values


def _archives(folder, config):
    """Check pairing against the canonical noisy test data before any scoring."""
    source = folder / "test.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Missing canonical test archive: {source}")
    expected = {
        key: config.get(key)
        for key in ("n_test", "context_length", "horizon", "samples")
    }
    for key, value in expected.items():
        if value is not None:
            positive_int(f"config.experiment.{key}", value)
    try:
        with np.load(source, allow_pickle=False) as data:
            history = validate_array(
                data["history"], (expected["n_test"], expected["context_length"], 1)
            )
            target = validate_array(
                data["target"], (len(history), expected["horizon"], 1)
            )
            ids = _ids(data["trajectory_ids"], len(history), source)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{source}: {error}") from error
    result = {
        "history": history,
        "target": target,
        "trajectory_ids": ids,
        "methods": {},
        "source_sha256": {"test.npz": _sha256(source)},
    }
    for method in METHODS:
        path = folder / f"{method}_samples.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired method archive: {path}")
        try:
            with np.load(path, allow_pickle=False) as saved:
                for key, reference in (
                    ("history", history),
                    ("target", target),
                    ("trajectory_ids", ids),
                ):
                    if not np.array_equal(saved[key], reference):
                        raise ValueError(f"{key} differs from canonical test.npz")
                processed = validate_array(saved["processed_history"], history.shape)
                if method == "raw" and not np.array_equal(processed, history):
                    raise ValueError("raw processed_history differs from raw history")
                samples = validate_array(
                    saved["samples"],
                    (len(target), expected["samples"], *target.shape[1:]),
                )
                ht = validate_times(saved["history_times"], history.shape[:2])
                ft = validate_times(saved["future_times"], target.shape[:2])
                if not (ft[:, 0] > ht[:, -1]).all():
                    raise ValueError("future_times must start after history_times")
                if "history_times" in result:
                    if not np.array_equal(
                        ht, result["history_times"]
                    ) or not np.array_equal(ft, result["future_times"]):
                        raise ValueError(
                            "measurement times differ across paired methods"
                        )
                else:
                    result.update(history_times=ht, future_times=ft)
                if "dt" in config:
                    dt = _number(config["dt"], "config.experiment.dt")
                    if dt == 0 or not np.allclose(
                        np.c_[ht, ft],
                        np.arange(history.shape[1] + target.shape[1]) * dt,
                        rtol=1e-12,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            "measurement times differ from configured dt grid"
                        )
                result["methods"][method] = {
                    "samples": samples,
                    "processed_history": processed,
                }
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error
        result["source_sha256"][path.name] = _sha256(path)
    return result


def _validation(path):
    rows = _json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty list of validation rows")
    epochs = []
    for row in rows:
        if not isinstance(row, dict) or "epoch" not in row or "crps" not in row:
            raise ValueError(f"{path}: each validation row needs epoch and crps")
        positive_int(f"{path}: epoch", row["epoch"])
        epochs.append(row["epoch"])
        _number(row["crps"], f"{path}: crps")
        if "sampling_seconds" in row:
            _number(row["sampling_seconds"], f"{path}: sampling_seconds")
        if "scores" in row:
            if not isinstance(row["scores"], dict) or "crps" not in row["scores"]:
                raise ValueError(f"{path}: validation scores must contain crps")
            value = _number(row["scores"]["crps"], f"{path}: scores.crps")
            if not np.isclose(value, row["crps"], rtol=1e-9):
                raise ValueError(f"{path}: validation crps and scores.crps differ")
    if epochs != sorted(set(epochs)):
        raise ValueError(f"{path}: validation epochs must be unique and increasing")
    best = min(rows, key=lambda row: row["crps"])
    return {
        "rows": rows,
        "selected_epoch_from_validation": best["epoch"],
        "minimum_validation_crps": best["crps"],
        "selected_at_final_candidate": best["epoch"] == rows[-1]["epoch"],
        "scope": (
            "Recorded validation selection; endpoint selection "
            "does not prove convergence."
        ),
    }


def _costs(path, extra_path=None):
    recorded = _json(path)
    if not isinstance(recorded, dict):
        raise ValueError(f"{path}: expected a timings object")
    sources = [path.name]
    if extra_path is not None:
        extra = _json(extra_path)
        if not isinstance(extra, dict):
            raise ValueError(f"{extra_path}: expected a timings object")
        if any(
            key in recorded and recorded[key] != value for key, value in extra.items()
        ):
            raise ValueError(f"{extra_path}: conflicts with frozen training timings")
        recorded = {**recorded, **extra}
        sources.append(extra_path.name)
    fit_key = "mediator_training_seconds"
    if fit_key not in recorded:
        raise ValueError(f"{path}: missing {fit_key}")
    costs = {fit_key: _number(recorded[fit_key], f"{path}: {fit_key}")}
    for key in METHOD_TIMINGS:
        if not isinstance(recorded.get(key), dict):
            raise ValueError(f"{path}: missing method timing object {key}")
        costs[key] = {}
        for method in METHODS:
            if method not in recorded[key]:
                raise ValueError(f"{path}: missing {key}.{method}")
            costs[key][method] = _number(
                recorded[key][method], f"{path}: {key}.{method}"
            )
    for method in METHODS:
        if (
            costs["validation_sampling_seconds"][method]
            > costs["forecaster_training_seconds"][method]
        ):
            raise ValueError(
                f"{path}: {method} validation subset exceeds training time"
            )
    costs["method_total_seconds"] = {
        method: sum(
            costs[key][method]
            for key in METHOD_TIMINGS
            if key != "validation_sampling_seconds"
        )
        + (costs[fit_key] if method == "learned" else 0)
        for method in METHODS
    }
    return {"source_files": sources, "recorded": recorded, "seconds": costs}


def _auxiliary(folder, scores):
    result = {
        "missing_artifacts": [],
        "validation": {},
        "costs": None,
        "saved_scores_verified": False,
    }
    path = folder / "scores.json"
    if path.is_file():
        saved = _json(path)
        if not isinstance(saved, dict):
            raise ValueError(f"{path}: expected method-to-metric objects")
        for method in METHODS:
            if not isinstance(saved.get(method), dict):
                raise ValueError(f"{path}: missing metric object for {method}")
            for metric in METRICS:
                try:
                    value = _number(saved[method][metric], f"{path}: {method}.{metric}")
                except KeyError as error:
                    raise ValueError(f"{path}: missing {method}.{metric}") from error
                if not np.isclose(
                    value, scores[method]["aggregate"][metric], rtol=1e-9, atol=1e-12
                ):
                    raise ValueError(
                        f"{path}: {method}.{metric} disagrees with "
                        "saved-sample recomputation"
                    )
        result["saved_scores_verified"] = True
    else:
        result["missing_artifacts"].append(path.name)
    for method in METHODS:
        path = folder / f"{method}_validation.json"
        if path.is_file():
            result["validation"][method] = _validation(path)
        else:
            result["missing_artifacts"].append(path.name)
    combined = folder / "combined_timings.json"
    path, evaluation = folder / "timings.json", folder / "evaluation_timings.json"
    if combined.is_file():
        result["costs"] = _costs(combined)
    elif path.is_file():
        result["costs"] = _costs(path, evaluation if evaluation.is_file() else None)
    else:
        result["missing_artifacts"].append(
            "combined_timings.json (or timings.json + evaluation_timings.json)"
        )
    return result


def _across_seeds(entries):
    values = list(entries.values())
    result = {
        "n_seeds": len(values),
        "scope": (
            "Descriptive paired seed variation; no pooled trajectory "
            "or training-seed inference."
        ),
        "scores": {
            method: {
                metric: _descriptive(
                    [
                        entry["scores"][method]["trajectory_weighted"][metric]
                        for entry in values
                    ]
                )
                for metric in METRICS
            }
            for method in METHODS
        },
        "paired_differences": {
            f"{left}-minus-{right}": {
                metric: _descriptive(
                    [
                        entry["paired_differences"][f"{left}-minus-{right}"][metric][
                            "difference"
                        ]
                        for entry in values
                    ]
                )
                for metric in METRICS
            }
            for left, right in COMPARISONS
        },
        "per_horizon": {},
    }
    for method in METHODS:
        result["per_horizon"][method] = {}
        for metric in METRICS:
            arrays = np.asarray(
                [entry["scores"][method]["per_horizon"][metric] for entry in values]
            )
            result["per_horizon"][method][metric] = [
                _descriptive(column) for column in arrays.T
            ]
    return result


def analyze_polishing_run(output_dir, bootstrap_replicates=10000, seed=8675309):
    """Recompute saved noisy-future scores, paired intervals, costs, and SVGs.

    Required per seed: test.npz and raw/learned/median_samples.npz. Auxiliary
    validation, scores, and timing files are checked when present and explicitly
    listed when missing. Configured seed/sample dimensions must be complete.
    Existing source archives and markers are never modified.
    """
    positive_int("bootstrap_replicates", bootstrap_replicates)
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    root = Path(output_dir)
    manifest = _json(root / "config.json") if (root / "config.json").is_file() else None
    if manifest is not None and not isinstance(manifest, dict):
        raise ValueError("config.json must contain an object")
    config = manifest.get("experiment", {}) if manifest is not None else {}
    if not isinstance(config, dict):
        raise ValueError("config.json experiment must contain an object")
    folders = [path for path in root.glob("seed_*") if path.is_dir()]
    try:
        folders.sort(key=lambda path: int(path.name.removeprefix("seed_")))
        observed = [int(path.name.removeprefix("seed_")) for path in folders]
    except ValueError as error:
        raise ValueError("Seed directories must be named seed_<integer>") from error
    if not folders:
        raise ValueError(f"No seed_* artifact directories in {root}")
    expected = config.get("seeds")
    if expected is not None:
        if (
            not isinstance(expected, list)
            or not expected
            or any(type(value) is not int for value in expected)
            or len(set(expected)) != len(expected)
        ):
            raise ValueError(
                "config.experiment.seeds must be a nonempty list of distinct integers"
            )
        if sorted(expected) != observed:
            raise ValueError(
                "Configured seed inventory differs: "
                f"missing={sorted(set(expected) - set(observed))}, "
                f"unexpected={sorted(set(observed) - set(expected))}"
            )
    result = {
        "analysis_version": 1,
        "methods": list(METHODS),
        "protocol": {
            "manifest_available": manifest is not None,
            "configured_seed_ids": sorted(expected) if expected is not None else None,
            "sample_seed_ids": observed,
            "markers": {name: (root / name).is_file() for name in MARKERS},
            "scope": (
                "Artifact completeness only; not an independent "
                "provenance or convergence certificate."
            ),
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "base_seed": seed,
            "unit": "independent trajectory with all its windows and saved draws",
            "estimand": "equal-trajectory mean difference, first method minus second",
            "scope": (
                "conditional on fitted models, preprocessing, and fixed saved "
                "ensembles; no resampling draws or training"
            ),
            "multiple_comparisons": (
                "95% intervals are unadjusted across methods and metrics."
            ),
        },
        "metric_definitions": {
            "target": (
                "Saved noisy future observations; no latent or clean signals "
                "are read or scored."
            ),
            "crps": "Empirical CRPS, matching evaluate_forecast; lower is better.",
            "mse": "Ensemble-mean squared error against noisy future observations.",
            "coverage_90": "Empirical 5th-to-95th percentile marginal coverage.",
            "interval_width_90": (
                "Mean empirical 95th minus 5th percentile, linearly interpolated."
            ),
            "finite_ensembles": (
                "64 or configured finite draws affect quantiles and empirical "
                "CRPS; paired intervals condition on those draws."
            ),
        },
        "cost_definitions": {
            "preprocessing_seconds": "Training plus validation history preprocessing.",
            "forecaster_training_seconds": (
                "Includes validation sampling; validation_sampling_seconds "
                "is a subset, not an additional cost."
            ),
            "method_total_seconds": (
                "Preprocessing + forecaster training + test preprocessing + "
                "test inference; learned additionally includes mediator fit once."
            ),
            "scope": (
                "Worker elapsed seconds; separate phase wall times may overlap "
                "and must not be added to worker totals."
            ),
        },
        "seeds": {},
        "plots": [],
        "phase_wall_seconds": {},
    }
    first_data = None
    horizon = None
    for index, folder in enumerate(folders):
        data = _archives(folder, config)
        if horizon is not None and horizon != data["target"].shape[1]:
            raise ValueError("Forecast horizon differs across seeds")
        horizon = data["target"].shape[1]
        scores = {
            method: summarize_forecasts(
                data["methods"][method]["samples"],
                data["target"],
                data["trajectory_ids"],
            )
            for method in METHODS
        }
        entry = {
            "n_windows": len(data["target"]),
            "n_trajectories": len(scores["raw"]["trajectory_ids"]),
            "context_length": data["history"].shape[1],
            "horizon": horizon,
            "draws_per_method": {
                method: data["methods"][method]["samples"].shape[1]
                for method in METHODS
            },
            "source_sha256": data["source_sha256"],
            "scores": scores,
            "paired_differences": {},
        }
        for comparison, (left, right) in enumerate(COMPARISONS):
            paired = paired_bootstrap(
                np.asarray(
                    [scores[left]["per_trajectory"][metric] for metric in METRICS]
                ).T,
                np.asarray(
                    [scores[right]["per_trajectory"][metric] for metric in METRICS]
                ).T,
                replicates=bootstrap_replicates,
                seed=seed + 100 * index + comparison,
            )
            entry["paired_differences"][f"{left}-minus-{right}"] = {
                metric: {
                    "difference": paired["difference"][i],
                    "ci95": [bound[i] for bound in paired["ci95"]],
                }
                for i, metric in enumerate(METRICS)
            }
            entry["paired_differences"][f"{left}-minus-{right}"].update(
                bootstrap_seed=paired["seed"], n_trajectories=paired["n_trajectories"]
            )
        entry.update(_auxiliary(folder, scores))
        result["seeds"][folder.name] = entry
        if first_data is None:
            first_data = data
    result["across_seed_descriptive"] = _across_seeds(result["seeds"])
    cost_rows = [
        entry["costs"]["seconds"]
        for entry in result["seeds"].values()
        if entry["costs"] is not None
    ]
    result["costs_across_seed_descriptive"] = {
        "n_available_seeds": len(cost_rows),
        "seconds": {},
    }
    if cost_rows:
        result["costs_across_seed_descriptive"]["seconds"] = {
            key: {
                method: _descriptive([row[key][method] for row in cost_rows])
                for method in METHODS
            }
            if isinstance(cost_rows[0][key], dict)
            else _descriptive([row[key] for row in cost_rows])
            for key in cost_rows[0]
        }
    for name in (
        "MEDIATORS_FROZEN.json",
        "TRAINED.json",
        "FROZEN.json",
        "EVALUATED.json",
    ):
        if (root / name).is_file():
            marker = _json(root / name)
            if "wall_seconds" in marker:
                result["phase_wall_seconds"][name] = _number(
                    marker["wall_seconds"], f"{name}: wall_seconds"
                )
    complete = all(result["protocol"]["markers"].values()) and not any(
        entry["missing_artifacts"] for entry in result["seeds"].values()
    )
    result["protocol"]["status"] = (
        "complete"
        if complete and expected is not None
        else "incomplete"
        if manifest is not None
        else "unverified_without_manifest"
    )
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    _forecast_plot(first_data, folders[0].name, plots / "representative_forecasts.svg")
    _horizon_plot(result, plots / "per_horizon.svg")
    result["plots"] = ["plots/representative_forecasts.svg", "plots/per_horizon.svg"]
    if _validation_plot(result, plots / "validation_crps.svg"):
        result["plots"].append("plots/validation_crps.svg")
    (root / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (root / "analysis.txt").write_text(_report(result))
    return result


def _panel(svg, box, title, xs, ys, *, ylabel="Value", xlabel="Time", ylim=None):
    """Small SVG axis with significant-digit labels for small positive scores."""
    x, y, width, height = box
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = ylim if ylim is not None else (float(np.min(ys)), float(np.max(ys)))
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    pad = max((ymax - ymin) * 0.08, abs(ymax) * 0.01, 1e-6)
    ymin, ymax = ymin - pad, ymax + pad
    if ylabel in METRICS:
        ymin = max(ymin, 0)
        if ylabel == "coverage_90":
            ymax = min(ymax, 1)
    left, top, iw, ih = x + 66, y + 30, width - 84, height - 82

    def transform(a, b):
        a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
        return left + (a - xmin) / (xmax - xmin) * iw, top + (ymax - b) / (
            ymax - ymin
        ) * ih

    svg.append(f'<text x="{left}" y="{y + 18}">{escape(title)}</text>')
    for tick in np.linspace(ymin, ymax, 5):
        _, py = transform(xmin, tick)
        svg.extend(
            [
                f'<path d="M {left} {py} h {iw}" stroke="#e9ecef"/>',
                f'<text x="{left - 8}" y="{py + 4}" '
                f'text-anchor="end">{tick:.3g}</text>',
            ]
        )
    for tick in np.linspace(xmin, xmax, 5):
        px, _ = transform(tick, ymin)
        svg.append(
            f'<text x="{px}" y="{top + ih + 20}" text-anchor="middle">{tick:.4g}</text>'
        )
    svg.extend(
        [
            f'<path d="M {left} {top} v {ih} h {iw}" stroke="#868e96" fill="none"/>',
            f'<text x="{left + iw / 2}" y="{y + height - 7}" '
            f'text-anchor="middle">{escape(xlabel)}</text>',
            f'<text transform="translate({x + 15},{top + ih / 2}) rotate(-90)" '
            f'text-anchor="middle">{escape(ylabel)}</text>',
        ]
    )
    return transform


def _legend(svg, y=77):
    for index, method in enumerate(METHODS):
        x = 32 + 300 * index
        svg.append(
            f'<path d="M {x} {y - 4} h 22" stroke="{COLORS[method]}" stroke-width="2"/>'
        )
        svg.append(f'<text x="{x + 30}" y="{y}">{LABELS[method]}</text>')


def _forecast_plot(data, seed_name, destination):
    ids, history, target = data["trajectory_ids"], data["history"], data["target"]
    _, first = np.unique(ids, return_index=True)
    rows = np.sort(first)[:6]
    svg = _svg(
        1440,
        115 + 255 * len(rows),
        f"History polishing forecasts | {seed_name} | first {len(rows)} trajectories",
    )
    svg.extend(
        [
            '<text x="32" y="53">Gray: raw noisy history; colored history: '
            "actual supplied input; dashed black: noisy future target.</text>",
            '<text x="32" y="75">Colored future: forecast mean; shading: empirical '
            "central 90% interval. Vertical divider: forecast boundary. "
            "No clean signals.</text>",
        ]
    )
    for row, index in enumerate(rows):
        ht, ft = data["history_times"][index], data["future_times"][index]
        bands = {
            method: np.quantile(
                data["methods"][method]["samples"][index, :, :, 0], [0.05, 0.95], axis=0
            )
            for method in METHODS
        }
        all_y = np.r_[
            history[index, :, 0],
            target[index, :, 0],
            *(data["methods"][m]["processed_history"][index, :, 0] for m in METHODS),
            *(b.ravel() for b in bands.values()),
        ]
        for column, method in enumerate(METHODS):
            transform = _panel(
                svg,
                (column * 480, 95 + row * 255, 475, 250),
                f"{LABELS[method]} | trajectory {ids[index]}",
                np.r_[ht, ft],
                all_y,
            )
            _band(svg, transform, ft, *bands[method], COLORS[method])
            _line(svg, transform, ht, history[index, :, 0], "#adb5bd")
            _line(
                svg,
                transform,
                ht,
                data["methods"][method]["processed_history"][index, :, 0],
                COLORS[method],
            )
            _line(svg, transform, ft, target[index, :, 0], "#212529", dashed=True)
            _line(
                svg,
                transform,
                ft,
                data["methods"][method]["samples"][index, :, :, 0].mean(axis=0),
                COLORS[method],
            )
            boundary = (ht[-1] + ft[0]) / 2
            _line(
                svg,
                transform,
                [boundary, boundary],
                [all_y.min(), all_y.max()],
                "#868e96",
                dashed=True,
            )
    destination.write_text("\n".join([*svg, "</svg>"]) + "\n")


def _horizon_plot(result, destination):
    svg = _svg(1200, 800, "Noisy-future performance by forecast horizon")
    svg.append(
        '<text x="32" y="53">Lines: equal-seed means; shading: descriptive seed '
        "range, not confidence intervals.</text>"
    )
    _legend(svg)
    for index, metric in enumerate(METRICS):
        values = {
            method: np.asarray(
                [
                    entry["scores"][method]["per_horizon"][metric]
                    for entry in result["seeds"].values()
                ]
            )
            for method in METHODS
        }
        xs = np.arange(1, values["raw"].shape[1] + 1)
        all_y = np.concatenate([array.ravel() for array in values.values()])
        if metric == "coverage_90":
            all_y = np.r_[all_y, 0.9]
        transform = _panel(
            svg,
            ((index % 2) * 600, 95 + (index // 2) * 340, 595, 335),
            metric,
            xs,
            all_y,
            ylabel=metric,
            xlabel="Forecast step",
        )
        if metric == "coverage_90":
            _line(svg, transform, xs, np.full(len(xs), 0.9), "#868e96", dashed=True)
        for method in METHODS:
            _band(
                svg,
                transform,
                xs,
                values[method].min(axis=0),
                values[method].max(axis=0),
                COLORS[method],
            )
            _line(svg, transform, xs, values[method].mean(axis=0), COLORS[method])
    destination.write_text("\n".join([*svg, "</svg>"]) + "\n")


def _validation_plot(result, destination):
    entries = [
        (name, entry["validation"])
        for name, entry in result["seeds"].items()
        if entry["validation"]
    ]
    if not entries:
        return False
    svg = _svg(
        1200,
        110 + 310 * len(entries),
        "Noisy-future validation CRPS checkpoint candidates",
    )
    svg.append(
        '<text x="32" y="53">Each method selects its minimum recorded validation '
        "CRPS; curves do not certify convergence.</text>"
    )
    _legend(svg)
    for index, (name, validation) in enumerate(entries):
        all_rows = [row for method in validation.values() for row in method["rows"]]
        transform = _panel(
            svg,
            (0, 95 + index * 310, 1195, 305),
            name,
            [row["epoch"] for row in all_rows],
            [row["crps"] for row in all_rows],
            ylabel="crps",
            xlabel="Epoch",
        )
        for method, review in validation.items():
            xs, ys = np.asarray(
                [(row["epoch"], row["crps"]) for row in review["rows"]]
            ).T
            _line(svg, transform, xs, ys, COLORS[method])
            px, py = transform(xs, ys)
            for x, y in zip(px, py, strict=True):
                svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{COLORS[method]}"/>')
    destination.write_text("\n".join([*svg, "</svg>"]) + "\n")
    return True


def _report(result):
    lines = [
        "History polishing forecast analysis",
        f"Protocol completeness: {result['protocol']['status']}",
        "Targets: saved noisy future observations; no clean signals.",
        "",
        "Equal-seed mean scores (seed SD and range are descriptive):",
    ]
    for method, scores in result["across_seed_descriptive"]["scores"].items():
        lines.append(method)
        for metric, summary in scores.items():
            lines.append(
                f"  {metric}: {summary['mean']:.6f}; SD={summary['sample_std']}; "
                f"range=[{summary['minimum']:.6f}, {summary['maximum']:.6f}]"
            )
    lines.extend(["", "Per-seed paired CRPS differences (first minus second):"])
    for name, entry in result["seeds"].items():
        for comparison, metrics in entry["paired_differences"].items():
            score = metrics["crps"]
            lines.append(
                f"  {name} {comparison}: {score['difference']:+.6f}, "
                f"95% CI [{score['ci95'][0]:+.6f}, {score['ci95'][1]:+.6f}]"
            )
        if entry["missing_artifacts"]:
            lines.append(
                f"  {name} MISSING AUXILIARY ARTIFACTS: "
                + ", ".join(entry["missing_artifacts"])
            )
    lines.extend(["", "Costs (mean worker seconds across available seeds):"])
    for category, summary in result["costs_across_seed_descriptive"]["seconds"].items():
        if "mean" in summary:
            lines.append(f"  {category}: {summary['mean']:.3f}")
        else:
            lines.append(
                f"  {category}: "
                + ", ".join(
                    f"{method}={summary[method]['mean']:.3f}" for method in METHODS
                )
            )
    lines.extend(
        [
            "Validation sampling is already included in forecaster training totals.",
            "Paired trajectory intervals condition on models, preprocessing, "
            "and saved draws; they are unadjusted across comparisons/metrics. "
            "Seed variation is separate.",
            "Finite ensembles affect quantiles and CRPS. Validation checkpoint "
            "selection and completed budgets do not establish convergence.",
            "Full per-horizon, per-trajectory, cost, and paired results: analysis.json",
            "Plots: " + ", ".join(result["plots"]),
            "",
        ]
    )
    return "\n".join(lines)
