"""Audit saved validation traces and benchmark work without accessing test arrays."""

import argparse
import hashlib
import json
import platform
import shutil
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.metrics import evaluate_forecast
from densiffusion.resource_allocation import (
    budget_table,
    count_csdi_work,
    count_mediator_work,
    fitting_ledger,
)
from densiffusion.waveform_imputation import _snapshot

POLISHING = ("raw", "learned", "median")
DENSIFICATION = ("direct_csdi", "linear_densification", "densiffusion")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protected_hashes(roots):
    return {
        str(path): digest(path)
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def benchmark(architecture, length, device, training):
    """Native kernels, synchronized blocks, fixed hardware and randomized tensors."""
    torch.manual_seed(89123)
    model = CSDIDenoiser(**architecture).to(device).train(training)
    batch = 64 if training else 256
    values = torch.randn(batch, length, 1, device=device)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, : length * 3 // 4] = False
    steps = torch.zeros(batch, device=device, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    def step():
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(values, steps, observed=values, mask=mask, dt=1)
            if training:
                prediction.square().mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

    for _ in range(3):
        step()
    sync()
    seconds = []
    for _ in range(5):
        sync()
        start = perf_counter()
        for _ in range(3):
            step()
        sync()
        seconds.append((perf_counter() - start) / 3)
    return {
        "batch": batch,
        "seconds_per_batch": seconds,
        "median_seconds_per_batch": float(np.median(seconds)),
        "scope": "native forward/backward/Adam/clip" if training else "native forward",
        "warmup_steps": 3,
        "steps_per_repetition": 3,
    }


def polishing_traces(root, profiles):
    manifest = read(root / "config.json")
    config = manifest["experiment"]
    forward = profiles["large_40"]["forward"]
    training = profiles["large_40"]["training"]
    traces, summaries, checks = {}, {}, []
    for folder in sorted(root.glob("seed_*")):
        timings = read(folder / "timings.json")
        mediator = read(folder / "mediator/summary.json")
        setup_training, setup_validation = 0, 0
        for candidate in mediator["candidates"]:
            cc = candidate["config"]
            work = profiles[f"mediator_{cc['hidden_channels']}"]
            setup_training += cc["epochs"] * config["n_train"] * 32 * work["training"]
            setup_validation += (
                len(cc["score_epochs"]) * config["n_validation"] * 32 * work["forward"]
            )
        selected = next(
            c for c in mediator["candidates"] if c["name"] == mediator["selected"]
        )
        selected_work = profiles[f"mediator_{selected['config']['hidden_channels']}"]
        preprocessing = (
            (config["n_train"] + config["n_validation"]) * 32 * selected_work["forward"]
        )
        setup = setup_training + setup_validation + preprocessing
        traces[folder.name], summaries[folder.name] = {}, {}
        for method in POLISHING:
            epochs = {
                row["epoch"]: row
                for row in map(
                    json.loads,
                    (folder / f"{method}_training.jsonl").read_text().splitlines(),
                )
            }
            validation = read(folder / f"{method}_validation.json")
            rows = []
            for index, row in enumerate(validation, 1):
                epoch = row["epoch"]
                assert row["crps"] == epochs[epoch]["selection_score"]
                with np.load(folder / row["samples"], allow_pickle=False) as saved:
                    rescored = evaluate_forecast(saved["samples"], saved["target"])
                for metric, value in rescored.items():
                    if not np.isclose(
                        value, row["scores"][metric], rtol=1e-12, atol=1e-12
                    ):
                        raise AssertionError(
                            f"Metric changed: {folder}/{method}/{epoch}/{metric}"
                        )
                checks.append(
                    {
                        "path": str(folder / row["samples"]),
                        "all_metrics_recomputed": True,
                    }
                )
                ledger = fitting_ledger(
                    epochs=epoch,
                    n_train=config["n_train"],
                    n_validation=config["n_validation"],
                    forward=forward,
                    training=training,
                    noise_validation_count=sum(
                        "validation_loss" in r for e, r in epochs.items() if e <= epoch
                    ),
                    noise_validation_steps=50,
                    forecast_validation_count=index,
                    forecast_samples=config["samples"],
                    diffusion_steps=50,
                    setup=setup if method == "learned" else 0,
                )
                seconds = (
                    epochs[epoch]["elapsed_seconds"]
                    + timings["preprocessing_seconds"][method]
                )
                if method == "learned":
                    seconds += timings["mediator_training_seconds"]
                rows.append(
                    {
                        **row,
                        "work": ledger["total"],
                        "ledger": ledger,
                        "seconds": seconds,
                    }
                )
            traces[folder.name][method] = rows
            summaries[folder.name][method] = {
                "completed_training_work": rows[-1]["ledger"],
                "completed_training_seconds": timings["forecaster_training_seconds"][
                    method
                ]
                + timings["preprocessing_seconds"][method]
                + (timings["mediator_training_seconds"] if method == "learned" else 0),
                "validation_sampling_seconds": timings["validation_sampling_seconds"][
                    method
                ],
                "deployment_work_per_64_draw_forecast": 64 * 50 * forward
                + (32 * selected_work["forward"] if method == "learned" else 0),
            }
        summaries[folder.name]["mediator"] = {
            "search_training_work": setup_training,
            "search_validation_work": setup_validation,
            "preprocessing_work": preprocessing,
            "total_work": setup,
            "total_seconds": timings["mediator_training_seconds"],
            "equivalent_forecaster_optimization_epochs": setup
            / (config["n_train"] * training),
        }
    return traces, summaries, checks


def densification_traces(root, profiles):
    config = read(root / "config.json")["experiment"]
    p33, p65, p49, p17 = (profiles[f"small_{length}"] for length in (33, 65, 49, 17))
    q_ledger = fitting_ledger(
        epochs=config["densifier_epochs"],
        n_train=config["n_train"],
        n_validation=config["n_validation"],
        forward=p33["forward"],
        training=p33["training"],
        noise_validation_count=8,
        noise_validation_steps=1,
        forecast_validation_count=0,
        forecast_samples=64,
        diffusion_steps=50,
    )
    augmentation = (
        (config["n_train"] + config["n_validation"])
        * 50
        * (p49["forward"] + p17["forward"])
    )
    q_validation_sampling = config["n_validation"] * 8 * 50 * p49["forward"]
    traces, summaries = {}, {}
    for folder in sorted(root.glob("seed_*")):
        timings = read(folder / "timings.json")
        logs = list(
            map(json.loads, (folder / "training.jsonl").read_text().splitlines())
        )
        traces[folder.name], summaries[folder.name] = {}, {}
        for method in DENSIFICATION:
            epochs = {r["epoch"]: r for r in logs if r["method"] == method}
            profile = p33 if method == "direct_csdi" else p65
            rows = []
            for index, row in enumerate(read(folder / f"{method}_validation.json"), 1):
                epoch = row["epoch"]
                assert row["crps"] == epochs[epoch]["selection_score"]
                ledger = fitting_ledger(
                    epochs=epoch,
                    n_train=config["n_train"],
                    n_validation=config["n_validation"],
                    forward=profile["forward"],
                    training=profile["training"],
                    noise_validation_count=index,
                    noise_validation_steps=1,
                    forecast_validation_count=index,
                    forecast_samples=64,
                    diffusion_steps=50,
                    setup=q_ledger["total"] + augmentation
                    if method == "densiffusion"
                    else 0,
                    extra_per_forecast_validation=q_validation_sampling
                    if method == "densiffusion"
                    else 0,
                )
                seconds = epochs[epoch]["elapsed_seconds"]
                if method == "densiffusion":
                    seconds += timings["densifier_training_seconds"]
                rows.append(
                    {
                        **row,
                        "work": ledger["total"],
                        "ledger": ledger,
                        "seconds": seconds,
                    }
                )
            traces[folder.name][method] = rows
            summaries[folder.name][method] = {
                "completed_training_work": rows[-1]["ledger"],
                "completed_training_seconds": timings["methods"][method][
                    "total_training_seconds"
                ],
                "recorded_costs": timings["methods"][method],
                "deployment_work_per_64_draw_forecast": 64 * 50 * profile["forward"]
                + (8 * 50 * p49["forward"] if method == "densiffusion" else 0),
            }
        summaries[folder.name]["densifier"] = {
            "training_work": q_ledger,
            "augmentation_work": augmentation,
        }
    return traces, summaries


def frontiers(traces, methods):
    work_reference = max(rows[methods[0]][-1]["work"] for rows in traces.values())
    result = {"reference_raw_full_training_work": work_reference}
    for key in ("work", "seconds"):
        budgets = sorted(
            {
                row[key]
                for seed in traces.values()
                for method in methods
                for row in seed[method]
            }
        )
        result[key] = budget_table(traces, methods, budgets, key)
    result["work_landmarks"] = budget_table(
        traces,
        methods,
        [work_reference * x for x in (0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3)],
        "work",
    )
    result["wall_landmarks"] = budget_table(
        traces,
        methods,
        [100, 200, 300, 400, 500, 600, 800, 1000],
        "seconds",
    )
    return result


def plot_frontiers(result, output):
    colors = ("#2563eb", "#dc2626", "#15803d")
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1000" '
        'viewBox="0 0 1400 1000">',
        '<rect width="1400" height="1000" fill="white"/>',
    ]

    def text(x, y, content, size=17, color="#172033"):
        svg.append(
            f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" '
            f'font-size="{size}" fill="{color}">{escape(str(content))}</text>'
        )

    text(40, 42, "Saved-validation performance under a common resource ceiling", 27)
    text(
        40,
        74,
        "Descriptive checkpoint frontiers; all earlier selection work charged. "
        "No new test or optimality claim.",
        18,
    )
    for row_index, (experiment, methods, lower, upper) in enumerate(
        (
            ("polishing", POLISHING, 0.25, 0.42),
            ("densification", DENSIFICATION, 0.1, 0.43),
        )
    ):
        for col, key in enumerate(("work", "seconds")):
            left, top, width, height = 90 + col * 690, 190 + row_index * 420, 570, 275
            table = result[experiment]["frontiers"][key]
            reference = (
                result[experiment]["frontiers"]["reference_raw_full_training_work"]
                if key == "work"
                else 1
            )
            xmax = max(r["budget"] / reference for r in table)

            def point(
                x,
                y,
                left=left,
                width=width,
                xmax=xmax,
                top=top,
                height=height,
                upper=upper,
                lower=lower,
            ):
                return left + width * x / xmax, top + height * (upper - y) / (
                    upper - lower
                )

            text(
                left,
                top - 70,
                f"{experiment.title()} | "
                + (
                    "matrix-work budget" if key == "work" else "recorded elapsed budget"
                ),
                21,
            )
            for i, method in enumerate(methods):
                text(left + i * 190, top - 36, method, 15, colors[i])
            for value in np.linspace(lower, upper, 5):
                _, y = point(0, value)
                svg.append(
                    f'<path d="M {left} {y} H {left + width}" stroke="#dce2eb"/>'
                )
                text(left - 55, y + 5, f"{value:.3f}", 14)
            for value in np.linspace(0, xmax, 5):
                x, _ = point(value, lower)
                text(
                    x - 15,
                    top + height + 25,
                    f"{value:.1f}" if key == "work" else f"{value:.0f}",
                    14,
                )
            for i, method in enumerate(methods):
                points = []
                previous = None
                for entry in table:
                    score = entry["methods"][method]["mean_crps"]
                    if score is None or score > upper:
                        continue
                    x, y = point(entry["budget"] / reference, score)
                    if previous is not None:
                        points.append((x, previous[1]))
                    points.append((x, y))
                    previous = (x, y)
                if points:
                    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                    svg.append(
                        f'<polyline points="{coordinates}" fill="none" '
                        f'stroke="{colors[i]}" stroke-width="2.8"/>'
                    )
            text(
                left + 70,
                top + height + 55,
                "Budget / full raw-training matrix work"
                if key == "work"
                else "Budget per seed (seconds)",
                17,
            )
    text(
        40,
        972,
        "Means require all three seeds. Polishing epoch-10 scores (1.96-3.76) "
        "lie above this detail view; full data are in audit.json.",
        16,
    )
    svg.append("</svg>")
    (output / "budget_frontiers.svg").write_text("\n".join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--polishing",
        type=Path,
        default=Path("outputs/history-polishing-20261027/primary"),
    )
    parser.add_argument(
        "--densification", type=Path, default=Path("outputs/waveform-primary-20261027")
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    protected = protected_hashes(
        [
            args.polishing.parent,
            args.densification,
            Path("outputs/waveform-large-20261027"),
        ]
    )
    write(output / "protected_artifacts.json", protected)
    write(
        output / "protocol.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "scope": (
                "Retrospective validation-only allocation audit plus "
                "synthetic-tensor work/timing benchmarks"
            ),
            "new_model_fitting": False,
            "test_forecasts_read_for_selection": False,
            "matrix_work_definition": (
                "Counted dense matrix multiply/add FLOPs, one multiply plus one "
                "addition equals two; math SDPA exposes attention"
            ),
            "exclusions": [
                "activation, softmax, normalization, loss, RNG, reductions, "
                "Adam, clipping, data movement, I/O",
                "median comparisons and NumPy interpolation",
                "researcher time and earlier experiment search",
            ],
            "cost_scope": (
                "Each method's complete in-run search, preprocessing, optimization, "
                "noise validation, and all preceding forecast validations"
            ),
            "deployment": (
                "Separate per-forecast 64-draw work, not inside fit/selection ceiling"
            ),
            "opportunities": (
                "Historical schedules retained; equal resource ceilings can "
                "afford different candidate counts"
            ),
            "limits": (
                "Not a prospectively matched trial; unused budget, missing "
                "checkpoints, fixed old LR schedules, and selected validation "
                "optimism remain"
            ),
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "threads": 1,
            },
        },
    )
    _snapshot(output)
    small = read(args.densification / "config.json")["architecture"]
    large = read(args.polishing / "config.json")["architecture"]
    profiles = {}
    for name, architecture, lengths in (
        ("small", small, (17, 33, 49, 65)),
        ("large", large, (33, 40, 65, 79)),
    ):
        for length in lengths:
            profile = count_csdi_work(architecture, length)
            if profile != count_csdi_work(architecture, length, batch_size=4):
                raise AssertionError("Profile is not linear in batch size")
            profiles[f"{name}_{length}"] = profile
    for width in (32, 64):
        profiles[f"mediator_{width}"] = count_mediator_work(width)
    write(output / "work_profiles.json", profiles)
    print("Work profiles complete", flush=True)
    timings = {}
    for name, architecture in (("small", small), ("large", large)):
        for device in ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",):
            for length in (33, 65):
                for training in (False, True):
                    key = (
                        f"{name}_{device}_{length}_{'train' if training else 'forward'}"
                    )
                    timings[key] = benchmark(architecture, length, device, training)
    write(output / "kernel_benchmarks.json", timings)
    print("Native-kernel timing complete", flush=True)
    traces, summaries, checks = polishing_traces(args.polishing, profiles)
    result = {
        "polishing": {
            "traces": traces,
            "costs": summaries,
            "frontiers": frontiers(traces, POLISHING),
        }
    }
    traces, summaries = densification_traces(args.densification, profiles)
    result["densification"] = {
        "traces": traces,
        "costs": summaries,
        "frontiers": frontiers(traces, DENSIFICATION),
    }
    write(output / "audit.json", result)
    plot_frontiers(result, output)
    if protected != protected_hashes(
        [
            args.polishing.parent,
            args.densification,
            Path("outputs/waveform-large-20261027"),
        ]
    ):
        raise AssertionError("An existing artifact changed")
    write(
        output / "verification.json",
        {
            "protected_files_unchanged": len(protected),
            "work_profile_batch_linearity": True,
            "polishing_validation_ensembles_recomputed": checks,
            "waveform_validation_scope": (
                "Scores checked against training logs; old runner saved "
                "no checkpoint validation ensembles"
            ),
            "new_test_data_generated": False,
            "existing_test_ensembles_loaded": False,
        },
    )
    for experiment in result.values():
        for seed in experiment["traces"].values():
            for rows in seed.values():
                for row in rows:
                    assert row["work"] == sum(
                        v for k, v in row["ledger"].items() if k != "total"
                    )
    shutil.copy(__file__, output / "audit_compute_allocation.py")
    print(f"Audit saved: {output}", flush=True)


if __name__ == "__main__":
    main()
