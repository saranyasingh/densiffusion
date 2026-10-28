"""Independent budget, provenance, preprocessing, and saved-score verification."""

import argparse
import importlib.util
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from densiffusion.generators.noisy_periodic import generate_noisy_periodic

_spec = importlib.util.spec_from_file_location(
    "_polishing_audit_helpers", Path(__file__).with_name("verify_polishing_forecast.py")
)
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
read, require, equal = shared.read, shared.require, shared.equal
METHODS = ("raw", "learned", "median")


def matrix_counts(architecture):
    """Derive counted operations independently of allocator/profiling helpers."""
    a = architecture
    require(a["n_channels"] == 1, "Only univariate formula supported")
    h, r, d, length = (
        a["hidden_channels"],
        a["residual_layers"],
        a["diffusion_embedding_dim"],
        40,
    )
    side = a["time_embedding_dim"] + a["feature_embedding_dim"] + 1
    forward = (
        4 * r * h * length**2
        + (
            6 * h
            + 2 * h**2
            + r * (16 * h**2 + 4 * h * a["feedforward_dim"] + 4 * side * h)
        )
        * length
        + 4 * d**2
        + 2 * r * d * h
    )
    training = 3 * forward - 2 * d**2 - 4 * length * h
    return forward, training


def independent_scores(samples, target):
    # Pairwise definition, rather than the sorted-ensemble formula used in training.
    total = 0.0
    for start in range(0, len(target), 64):
        block = samples[start : start + 64]
        truth = target[start : start + 64]
        score = np.abs(block - truth[:, None]).mean(axis=1)
        score -= 0.5 * np.abs(block[:, :, None] - block[:, None, :]).mean(axis=(1, 2))
        total += score.sum()
    low, high = np.quantile(samples, [0.05, 0.95], axis=1)
    return {
        "crps": float(total / target.size),
        "mse": float(np.mean((samples.mean(axis=1) - target) ** 2)),
        "coverage_90": float(np.mean((low <= target) & (target <= high))),
        "interval_width_90": float(np.mean(high - low)),
    }


def verify(output, training_only=False):
    root = Path(output)
    torch.set_num_threads(1)
    frozen = read(root / "FROZEN.json")
    shared.verify_hashes(root, frozen["files"], "Frozen training artifact")
    require(
        frozen["test_data_generated"] is False
        and frozen["all_methods_and_seeds_selected"] is True,
        "Invalid global freeze",
    )
    manifest = read(root / "config.json")
    c = manifest["experiment"]
    profiles = manifest["work_profiles"]
    forward, training = matrix_counts(manifest["architecture"])
    require(
        profiles["forecaster"]["forward"] == forward
        and profiles["forecaster"]["training"] == training,
        "Forecaster operation counts differ",
    )
    require(
        manifest["architecture"] == manifest["recovered_gp_recipe"]["architecture"],
        "GP architecture differs",
    )
    require(
        manifest["recovered_gp_recipe"]["gp_weights_reused"] is False,
        "GP weights reused",
    )
    if not c["smoke"]:
        require(
            c["namespace"] == 2026102801 and c["seeds"] == [29101, 29102, 29103],
            "Scientific seed namespace differs",
        )
        require(
            (
                c["n_train"],
                c["n_validation"],
                c["n_test"],
                c["base_epochs"],
                c["samples"],
                c["diffusion_steps"],
            )
            == (256, 256, 2048, 1200, 64, 50),
            "Scientific sizes differ",
        )
        require(
            c["validation_epochs_prefix"] == [10, 50, 100, 200, 400, 800],
            "Scientific candidate quota differs",
        )
    med_freeze, budgets = (
        read(root / "MEDIATORS_FROZEN.json"),
        read(root / "BUDGETS_FROZEN.json"),
    )
    shared.verify_hashes(root, med_freeze["files"], "Mediator")
    shared.verify_hashes(root, budgets["files"], "Allocation")
    require(
        budgets["configuration_sha256"] == shared.sha256(root / "config.json"),
        "Configuration changed after allocations",
    )
    prefix = c["validation_epochs_prefix"]
    quota = len(prefix) + 1
    ntrain, nval = c["n_train"], c["n_validation"]
    noise_steps = c["diffusion_steps"]

    def cost(epochs, setup=0):
        return (
            setup
            + epochs * ntrain * training
            + math.ceil(epochs / c["validation_interval"])
            * nval
            * noise_steps
            * forward
            + quota * nval * c["samples"] * noise_steps * forward
        )

    ceiling = cost(c["base_epochs"])
    result = {"seeds": {}, "ceiling": ceiling, "training_only": training_only}
    for index, seed in enumerate(c["seeds"]):
        folder = root / f"seed_{seed}"
        seeds = shared.seed_plan(seed, c["namespace"])
        require(
            seeds == read(folder / "seeds.json") == manifest["seeds"][str(seed)],
            "Private RNG plan differs",
        )
        require(
            manifest["method_order"][str(seed)]
            == list(METHODS[index % 3 :] + METHODS[: index % 3]),
            "Rotated method order differs",
        )
        with np.load(folder / "train_validation.npz", allow_pickle=False) as archive:
            original = dict(archive)
        for split, number in (("train", ntrain), ("validation", nval)):
            expected = generate_noisy_periodic(
                number, rng=np.random.default_rng(seeds[f"{split}_data"])
            )
            equal(original[split], expected, f"{split} data")
        mediator, summary = shared.verify_mediator(
            folder / "mediator",
            original["train"][:, :32],
            original["validation"][:, :32],
            seeds["mediator_training"],
            "cpu",
            c["smoke"],
        )
        selected = next(
            row for row in summary["candidates"] if row["name"] == summary["selected"]
        )
        width = selected["config"]["hidden_channels"]
        med_epochs = 5 if c["smoke"] else 200
        med_validations = len({1, 5, *range(10, med_epochs + 1, 10), med_epochs})
        setup = 0
        for w in (32, 64):
            f = 2 * (12 * w + w * w + w)
            t = 3 * f - 24 * w
            require(
                profiles[f"mediator_{w}"]["forward"] == f
                and profiles[f"mediator_{w}"]["training"] == t,
                "Mediator profile differs",
            )
            setup += 2 * (
                med_epochs * ntrain * 32 * t + med_validations * nval * 32 * f
            )
        setup += (ntrain + nval) * 32 * 2 * (12 * width + width * width + width)
        plan = read(folder / "budget_plan.json")
        require(
            plan["ceiling"] == ceiling and plan["auxiliary"]["total"] == setup,
            "Budget or auxiliary charges differ",
        )
        with np.load(
            folder / "processed_train_validation.npz", allow_pickle=False
        ) as archive:
            processed = dict(archive)
        for split in ("train", "validation"):
            raw = original[split][:, :32]
            equal(processed[f"raw_{split}_history"], raw, "Raw history")
            equal(
                processed[f"learned_{split}_history"],
                shared.learned_history(mediator, raw),
                "Blind-spot history",
            )
            equal(
                processed[f"median_{split}_history"],
                shared.median_history(raw),
                "Median history",
            )
            equal(
                processed[f"{split}_future"],
                original[split][:, 32:],
                "Unchanged noisy future",
            )
        audit = read(folder / "training_audit.json")
        initial = shared.initial_forecaster(
            manifest["architecture"], seeds["forecaster_initialization"]
        )
        initial_hash = shared.forecaster_state_hash(initial)
        result["seeds"][str(seed)] = {}
        for method in METHODS:
            entry, allocation = audit["methods"][method], plan["methods"][method]
            epochs = allocation["epochs"]
            overhead = setup if method == "learned" else 0
            spent = cost(epochs, overhead)
            require(
                spent == allocation["work"]["total"]
                and spent <= ceiling < cost(epochs + 1, overhead),
                "Allocation is not maximal within budget",
            )
            require(
                allocation["unused_work"] == ceiling - spent, "Unused budget differs"
            )
            if not c["smoke"]:
                require(
                    (ceiling - spent) / ceiling < 0.0001,
                    "Budget mismatch exceeds 0.01 percent",
                )
            expected_training = {**manifest["forecaster_training"], "epochs": epochs}
            checkpoint = torch.load(
                folder / f"{method}.pt", map_location="cpu", weights_only=True
            )
            require(
                entry["initialization_sha256"] == initial_hash
                and entry["parameter_count"]
                == sum(p.numel() for p in initial.parameters()),
                "Initial weights or architecture differ",
            )
            require(
                checkpoint["config"] == expected_training == entry["training_config"],
                "Training recipe differs",
            )
            require(
                checkpoint["seed"] == seeds["forecaster_training"],
                "Training stream differs",
            )
            require(
                shared.sha256(folder / f"{method}.pt") == entry["checkpoint_sha256"],
                "Selected checkpoint changed",
            )
            logs = [
                json.loads(line)
                for line in (folder / f"{method}_training.jsonl")
                .read_text()
                .splitlines()
            ]
            require(
                [r["epoch"] for r in logs] == list(range(1, epochs + 1)),
                "Epoch inventory incomplete",
            )
            monitored = [r["epoch"] for r in logs if "validation_loss" in r]
            require(
                monitored
                == sorted(
                    {
                        *range(
                            c["validation_interval"],
                            epochs + 1,
                            c["validation_interval"],
                        ),
                        epochs,
                    }
                ),
                "All-step validation cadence differs",
            )
            for row in logs:
                decays = sum(
                    row["epoch"] > max(1, int(f * epochs))
                    for f in expected_training["lr_decay_fractions"]
                )
                expected_lr = (
                    expected_training["learning_rate"]
                    * expected_training["lr_decay_factor"] ** decays
                )
                require(
                    np.isclose(row["learning_rate"], expected_lr, rtol=1e-14),
                    "Budget-relative LR decay differs",
                )
            counts = {
                "optimizer_updates": epochs * (ntrain // c["batch_size"]),
                "train_examples": epochs * ntrain,
                "eval_examples": nval
                * noise_steps
                * (len(monitored) + quota * c["samples"]),
            }
            require(
                entry["measured_counts"] == counts,
                "Actual forward/backward work differs",
            )
            rows = read(folder / f"{method}_validation.json")
            require(
                [r["epoch"] for r in rows]
                == [*prefix, epochs]
                == allocation["validation_epochs"],
                "Forecast selection quota differs",
            )
            for row in rows:
                metrics = shared.check_ensemble(
                    folder / row["samples"],
                    original["validation"][:, :32],
                    processed[f"{method}_validation_history"],
                    original["validation"][:, 32:],
                    original["validation_trajectory_ids"],
                    c["samples"],
                )
                for key, value in metrics.items():
                    require(
                        np.isclose(value, row["scores"][key], rtol=1e-12),
                        "Validation metric differs",
                    )
                require(
                    row["crps"] == logs[row["epoch"] - 1]["selection_score"],
                    "Validation selection score differs",
                )
            best = min(rows, key=lambda row: row["crps"])
            require(
                checkpoint["epoch"] == entry["selected_epoch"] == best["epoch"],
                "Validation checkpoint selection differs",
            )
            for split in ("train", "validation"):
                require(
                    entry[f"{split}_history_sha256"]
                    == shared.raw_hash(processed[f"{method}_{split}_history"]),
                    "Conditioning hash differs",
                )
                require(
                    entry[f"{split}_future_sha256"]
                    == shared.raw_hash(original[split][:, 32:]),
                    "Future target hash differs",
                )
            require(
                shared.timestamp(med_freeze["frozen_utc"])
                < shared.timestamp(entry["started_utc"]),
                "Mediator not frozen before forecasting",
            )
            require(
                shared.timestamp(budgets["frozen_utc"])
                < shared.timestamp(entry["started_utc"]),
                "Budget not frozen before forecasting",
            )
            require(
                shared.timestamp(entry["completed_utc"])
                <= shared.timestamp(frozen["frozen_utc"]),
                "Training occurred after global freeze",
            )
            result["seeds"][str(seed)][method] = {
                "epochs": epochs,
                "charged_work": spent,
                "unused_fraction": (ceiling - spent) / ceiling,
                "selected_epoch": best["epoch"],
                "validation_crps": best["crps"],
            }
        if not training_only:
            require((root / "EVALUATED.json").is_file(), "Incomplete test evaluation")
            started = read(root / "EVALUATION_STARTED.json")
            require(
                shared.timestamp(started["started_utc"])
                > shared.timestamp(frozen["frozen_utc"]),
                "Test began before freeze",
            )
            with np.load(folder / "test.npz", allow_pickle=False) as archive:
                test = dict(archive)
            expected = generate_noisy_periodic(
                c["n_test"], rng=np.random.default_rng(seeds["test_data"])
            )
            equal(test["history"], expected[:, :32], "Fresh test history")
            equal(test["target"], expected[:, 32:], "Fresh test noisy future")
            equal(
                test["trajectory_ids"],
                np.arange(c["n_test"]) + ntrain + nval,
                "Test IDs",
            )
            histories = {
                "raw": test["history"],
                "learned": shared.learned_history(mediator, test["history"]),
                "median": shared.median_history(test["history"]),
            }
            scores = read(folder / "scores.json")
            inference = read(folder / "inference_audit.json")
            for method in METHODS:
                path = folder / f"{method}_samples.npz"
                shared.check_ensemble(
                    path,
                    test["history"],
                    histories[method],
                    test["target"],
                    test["trajectory_ids"],
                    c["samples"],
                )
                with np.load(path, allow_pickle=False) as archive:
                    rescored = independent_scores(archive["samples"], test["target"])
                for metric, value in rescored.items():
                    require(
                        np.isclose(
                            value, scores[method][metric], rtol=1e-12, atol=1e-12
                        ),
                        "Independent test metric differs",
                    )
                expected_hash = shared.raw_hash(histories[method])
                require(
                    inference[method]["observed_argument_sha256"]
                    == inference[method]["dense_history_argument_sha256"]
                    == expected_hash,
                    "Inference has raw side channel",
                )
                require(
                    inference[method]["future_conditioning"] is False
                    and inference[method]["raw_side_channel"] is False,
                    "Future/raw conditioning differs",
                )
    result.update(
        verified_utc=datetime.now(UTC).isoformat(),
        passed=True,
        independent_work_formula=True,
        no_new_diffusion_sampling=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    result = verify(args.output, args.training_only)
    name = "training_verification.json" if args.training_only else "verification.json"
    (args.output / name).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Verified {args.output}: {name}", flush=True)


if __name__ == "__main__":
    main()
