"""Audit saved polishing forecasts without fitting or diffusion sampling."""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.generators.noisy_periodic import generate_noisy_periodic
from densiffusion.history_polishing import BlindSpotMediator, load_mediator
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel

METHODS = ("raw", "learned", "median")
METRICS = ("crps", "mse", "coverage_90", "interval_width_90")
OFFSETS = (-3, -2, -1, 1, 2, 3)
SEED_NAMES = (
    "train_data",
    "validation_data",
    "test_data",
    "mediator_training",
    "forecaster_initialization",
    "forecaster_training",
    "validation_sampling",
    "test_sampling",
)


def require(condition, description):
    if not condition:
        raise ValueError(description)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def mediator_state_hash(state):
    digest = hashlib.sha256()
    for name, values in sorted(state.items()):
        digest.update(name.encode())
        digest.update(array_hash(values.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def forecaster_state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def equal(actual, expected, description):
    require(np.array_equal(actual, expected), f"{description} differs")


def verify_hashes(root, hashes, description):
    for name, expected in hashes.items():
        require(sha256(root / name) == expected, f"{description} changed: {name}")


def neighbor_features(history):
    """Independent fixed-slot reconstruction; no center or future is available."""
    require(history.ndim == 3 and history.shape[1:] == (32, 1), "History must be 32x1")
    features = np.zeros((len(history), 32, 12), dtype=np.float64)
    for center in range(32):
        for slot, offset in enumerate(OFFSETS):
            neighbor = center + offset
            if 0 <= neighbor < 32:
                features[:, center, slot] = history[:, neighbor, 0]
                features[:, center, slot + 6] = 1.0
    return features


def learned_history(model, history):
    features = neighbor_features(history)
    parameter = next(model.parameters())
    model.eval()
    with torch.inference_mode():
        return (
            model(
                torch.as_tensor(
                    features, dtype=parameter.dtype, device=parameter.device
                )
            )
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def median_history(history):
    result = np.empty_like(history)
    for center in range(32):
        neighbors = np.clip(np.arange(center - 3, center + 4), 0, 31)
        result[:, center] = np.median(history[:, neighbors], axis=1)
    return result


def verify_mediator(directory, train, validation, seed, device, smoke):
    """Recompute candidate/epoch choice from noisy-center predictions only."""
    summary = read(directory / "summary.json")
    require(
        summary["seed"] == seed
        and summary["smoke"] is smoke
        and summary["selection_metric"] == "noisy_center_validation_mse"
        and summary["selection_ties"] == "first_in_epoch_then_candidate_order",
        "Mediator selection protocol differs",
    )
    require(
        summary["train_shape"] == list(train.shape)
        and summary["validation_shape"] == list(validation.shape)
        and summary["train_history_sha256"] == array_hash(train)
        and summary["validation_history_sha256"] == array_hash(validation)
        and summary["validation_centers"] == validation.size
        and summary["center_excluded_from_features"] is True
        and summary["clean_values_or_future_observations_accessed"] is False,
        "Mediator inputs are not the noisy histories alone",
    )
    epochs = 5 if smoke else 200
    score_epochs = sorted({1, 5, *range(10, epochs + 1, 10), epochs})
    candidates = summary["candidates"]
    choices = [(width, loss) for width in (32, 64) for loss in ("mse", "huber")]
    require(len(candidates) == len(choices), "Mediator candidate inventory differs")
    streams = np.random.SeedSequence(seed).spawn(4)
    for index, (candidate, (width, loss)) in enumerate(
        zip(candidates, choices, strict=True)
    ):
        name = f"candidate_{index:02d}_{width}_{loss}"
        require(candidate["name"] == name, "Mediator candidate ordering differs")
        folder = directory / name
        initialization, shuffle = streams[index].spawn(2)
        expected_config = {
            "hidden_channels": width,
            "hidden_layers": 2,
            "activation": "relu",
            "loss": loss,
            "huber_delta": 0.2,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": epochs,
            "score_epochs": score_epochs,
            "offsets": list(OFFSETS),
            "feature_order": "six_neighbor_values_then_six_availability_indicators",
            "history_length": 32,
            "init_seed": int(initialization.generate_state(1)[0]),
            "shuffle_seed": int(shuffle.generate_state(1)[0]),
            "device": device,
            "smoke": smoke,
        }
        require(
            read(folder / "config.json") == candidate["config"] == expected_config,
            f"{name}: mediator recipe or private RNG streams differ",
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(expected_config["init_seed"])
            initial = BlindSpotMediator(width)
        initial_hash = mediator_state_hash(initial.state_dict())
        logs = [
            json.loads(line)
            for line in (folder / "training.jsonl").read_text().splitlines()
        ]
        updates_per_epoch = math.ceil(train.size / 256)
        require(
            [row["epoch"] for row in logs] == list(range(1, epochs + 1)),
            f"{name}: mediator epoch inventory differs",
        )
        require(
            all(
                row["optimizer_updates"] == row["epoch"] * updates_per_epoch
                for row in logs
            ),
            f"{name}: mediator optimizer budget differs",
        )
        selection = read(folder / "validation.json")
        require(
            selection
            == [
                {"epoch": row["epoch"], "validation_mse": row["validation_mse"]}
                for row in logs
                if "validation_mse" in row
            ],
            f"{name}: mediator selection logs differ",
        )
        require(
            [row["epoch"] for row in selection] == score_epochs,
            f"{name}: mediator validation cadence differs",
        )
        best = min(selection, key=lambda row: row["validation_mse"])
        checkpoint = torch.load(
            folder / "best.pt", map_location="cpu", weights_only=True
        )
        model = load_mediator(folder / "best.pt", device=device)
        predictions = np.load(
            folder / "best_validation_predictions.npy", allow_pickle=False
        )
        equal(
            predictions,
            learned_history(model, validation),
            f"{name}: selected predictions",
        )
        score = float(np.mean((predictions - validation) ** 2))
        require(
            score
            == best["validation_mse"]
            == candidate["validation_mse"]
            == checkpoint["validation_mse"],
            f"{name}: noisy-center score does not reproduce",
        )
        require(
            best["epoch"] == candidate["selected_epoch"] == checkpoint["epoch"]
            and checkpoint["optimizer_updates"] == best["epoch"] * updates_per_epoch
            and candidate["optimizer_updates"] == epochs * updates_per_epoch,
            f"{name}: mediator checkpoint is not the first validation argmin",
        )
        require(
            checkpoint["config"] == expected_config
            and checkpoint["initial_sha256"]
            == candidate["initial_sha256"]
            == initial_hash
            and candidate["selected_sha256"] == mediator_state_hash(checkpoint["model"])
            and candidate["checkpoint_sha256"] == sha256(folder / "best.pt"),
            f"{name}: mediator checkpoint provenance differs",
        )
    best = min(candidates, key=lambda row: row["validation_mse"])
    require(
        summary["selected"] == best["name"]
        and summary["selected_epoch"] == best["selected_epoch"]
        and summary["validation_mse"] == best["validation_mse"]
        and sha256(directory / "selected.pt")
        == best["checkpoint_sha256"]
        == summary["selected_checkpoint_sha256"],
        "Mediator candidate selection or byte-identical selected checkpoint differs",
    )
    model = load_mediator(directory / "selected.pt", device=device)
    with np.load(directory / "selected_predictions.npz", allow_pickle=False) as saved:
        for split, history in (("train", train), ("validation", validation)):
            expected = learned_history(model, history)
            equal(
                saved[f"{split}_predictions"],
                expected,
                f"Mediator selected {split} history",
            )
            require(
                array_hash(expected) == summary[f"{split}_predictions_sha256"],
                f"Mediator selected {split} hash differs",
            )
    return model, summary


def raw_hash(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Timestamps must include a timezone")
    return result


def seed_plan(seed, namespace):
    return {
        name: int(stream.generate_state(1)[0])
        for name, stream in zip(
            SEED_NAMES,
            np.random.SeedSequence([seed, namespace]).spawn(len(SEED_NAMES)),
            strict=True,
        )
    }


def initial_forecaster(architecture, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ImputationModel(CSDIDenoiser(**architecture))


def check_ensemble(path, raw, processed, target, ids, samples):
    with np.load(path, allow_pickle=False) as saved:
        require(
            set(saved.files)
            == {
                "samples",
                "history",
                "processed_history",
                "target",
                "trajectory_ids",
                "history_times",
                "future_times",
            },
            f"{path.name}: unexpected ensemble fields",
        )
        equal(saved["history"], raw, f"{path.name}: raw history")
        equal(saved["processed_history"], processed, f"{path.name}: processed history")
        equal(saved["target"], target, f"{path.name}: unchanged noisy targets")
        equal(saved["trajectory_ids"], ids, f"{path.name}: trajectory IDs")
        equal(
            saved["history_times"],
            np.arange(32, dtype=float),
            f"{path.name}: history times",
        )
        equal(
            saved["future_times"],
            np.arange(32, 40, dtype=float),
            f"{path.name}: future times",
        )
        require(
            saved["samples"].shape == (len(raw), samples, 8, 1),
            f"{path.name}: ensemble shape",
        )
        require(
            np.isfinite(saved["samples"]).all(), f"{path.name}: nonfinite forecasts"
        )
        return evaluate_forecast(saved["samples"], target)


def verify_recipe(manifest):
    config, recipe = manifest["experiment"], manifest["recovered_gp_recipe"]
    source = Path(recipe["source"])
    verify_hashes(source, recipe["source_sha256"], "Recovered GP recipe")
    plan, selection = read(source / "plan.json"), read(source / "selection.json")
    name = selection["selected"]
    require(
        name == recipe["selected"] == "reference_size",
        "Wrong larger GP architecture recipe",
    )
    require(recipe["gp_weights_reused"] is False, "GP weights must not be reused")
    selected = plan["candidates"][name]
    require(
        manifest["architecture"] == recipe["architecture"] == selected["architecture"],
        "Forecaster architecture differs from recovered recipe",
    )
    require(
        recipe["training"] == selected["training"], "Recovered optimizer recipe differs"
    )
    expected = {
        **recipe["training"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "validation_interval": config["validation_interval"],
        "diffusion_steps": config["diffusion_steps"],
        "device": config["device"],
    }
    require(
        manifest["forecaster_training"] == expected
        and expected["validation_steps"] == "all",
        "Paired forecaster optimizer or all-step validation recipe differs",
    )
    require(
        manifest["methods"] == list(METHODS)
        and manifest["normalization"] == "none"
        and manifest["future_targets"] == "unchanged noisy observations"
        and manifest["history_only_preprocessing"] is True
        and manifest["no_clean_signal_exposed"] is True
        and manifest["no_densification"] is True,
        "History-only noisy-target protocol differs",
    )
    require(
        config["context_length"] == 32 and config["horizon"] == 8 and config["dt"] == 1,
        "Forecast grid differs",
    )
    schedule = config["validation_epochs"]
    require(
        schedule
        and all(
            type(epoch) is int
            and 0 < epoch <= config["epochs"]
            and epoch % config["validation_interval"] == 0
            for epoch in schedule
        )
        and all(
            left < right for left, right in zip(schedule, schedule[1:], strict=False)
        ),
        "Explicit validation schedule is invalid",
    )
    if config["label"] == "scientific":
        fixed = {
            "seeds": [28101, 28102, 28103],
            "namespace": 2026102705,
            "n_train": 256,
            "n_validation": 256,
            "n_test": 2048,
            "epochs": 1200,
            "batch_size": 64,
            "validation_interval": 10,
            "validation_epochs": [10, 50, 100, 200, 400, 800, 1200],
            "samples": 64,
            "diffusion_steps": 50,
        }
        require(
            all(config[name] == value for name, value in fixed.items()),
            "Scientific paired budget or opportunity inventory differs",
        )
    else:
        require(
            config["label"] == "smoke" and config["namespace"] != 2026102705,
            "Smoke must have a distinct namespace",
        )


def required_training_inventory(output, config):
    result = {
        "config.json",
        "source.tar.gz",
        "git_head.txt",
        "git_diff.patch",
        "MEDIATORS_FROZEN.json",
        "TRAINED.json",
    }
    candidates = sorted({*config["validation_epochs"], config["epochs"]})
    mediator_files = set()
    for seed in config["seeds"]:
        prefix = f"seed_{seed}"
        result.update(
            f"{prefix}/{name}"
            for name in (
                "seeds.json",
                "train_validation.npz",
                "processed_train_validation.npz",
                "preprocessing_audit.json",
                "training_audit.json",
                "timings.json",
            )
        )
        mediator_names = {"selected.pt", "selected_predictions.npz", "summary.json"}
        for index, (width, loss) in enumerate(
            (w, loss) for w in (32, 64) for loss in ("mse", "huber")
        ):
            name = f"candidate_{index:02d}_{width}_{loss}"
            mediator_names.update(
                f"{name}/{filename}"
                for filename in (
                    "config.json",
                    "training.jsonl",
                    "validation.json",
                    "best.pt",
                    "best_validation_predictions.npy",
                )
            )
        mediator_files.update(f"{prefix}/mediator/{name}" for name in mediator_names)
        actual_mediators = {
            str(p.relative_to(output))
            for p in (output / prefix / "mediator").rglob("*")
            if p.is_file()
        }
        require(
            actual_mediators
            == {f"{prefix}/mediator/{name}" for name in mediator_names},
            f"{prefix}: mediator artifact inventory differs",
        )
        for method in METHODS:
            result.update(
                f"{prefix}/{method}{suffix}"
                for suffix in (".pt", "_validation.json", "_training.jsonl")
            )
            names = {
                f"{method}_validation_epoch_{epoch:04d}.npz" for epoch in candidates
            }
            result.update(f"{prefix}/{name}" for name in names)
            require(
                {
                    p.name
                    for p in (output / prefix).glob(f"{method}_validation_epoch_*.npz")
                }
                == names,
                f"{prefix}/{method}: candidate ensemble inventory differs",
            )
    return result | mediator_files, mediator_files


def verify_training_seed(output, manifest, seed, mediator_frozen, trained):
    config, training = manifest["experiment"], manifest["forecaster_training"]
    folder = output / f"seed_{seed}"
    seeds = seed_plan(seed, config["namespace"])
    require(
        seeds == manifest["seeds"][str(seed)] == read(folder / "seeds.json"),
        f"{folder.name}: private stream allocation differs",
    )
    original = {}
    with np.load(folder / "train_validation.npz", allow_pickle=False) as saved:
        require(
            set(saved.files)
            == {
                "train",
                "validation",
                "train_trajectory_ids",
                "validation_trajectory_ids",
            },
            f"{folder.name}: unexpected original-data fields",
        )
        for split, offset in (("train", 0), ("validation", config["n_train"])):
            expected = generate_noisy_periodic(
                config[f"n_{split}"], rng=seeds[f"{split}_data"]
            )
            equal(
                saved[split],
                expected,
                f"{folder.name}/{split}: noisy observed-generator replay",
            )
            equal(
                saved[f"{split}_trajectory_ids"],
                np.arange(len(expected)) + offset,
                f"{folder.name}/{split}: trajectory IDs",
            )
            original[split] = saved[split]
    mediator, summary = verify_mediator(
        folder / "mediator",
        original["train"][:, :32],
        original["validation"][:, :32],
        seeds["mediator_training"],
        "cpu",
        config["label"] == "smoke",
    )
    mediator_hash = forecaster_state_hash(mediator)
    preaudit = read(folder / "preprocessing_audit.json")
    require(
        preaudit["mediator_sha256"] == mediator_hash
        and preaudit["mediator_selection"] == summary
        and preaudit["normalization"] == "none"
        and preaudit["densification"] is False,
        f"{folder.name}: mediator preprocessing protocol differs",
    )
    processed = {}
    with np.load(
        folder / "processed_train_validation.npz", allow_pickle=False
    ) as saved:
        expected_names = {
            f"{method}_{split}_history"
            for method in METHODS
            for split in ("train", "validation")
        }
        expected_names.update(("train_future", "validation_future"))
        require(
            set(saved.files) == expected_names,
            f"{folder.name}: unexpected processed-data fields",
        )
        for split in ("train", "validation"):
            raw, future = original[split][:, :32], original[split][:, 32:]
            expected = {
                "raw": raw,
                "learned": learned_history(mediator, raw),
                "median": median_history(raw),
            }
            equal(
                saved[f"{split}_future"], future, f"{folder.name}/{split}: noisy future"
            )
            require(
                preaudit[f"raw_{split}_history_sha256"] == raw_hash(raw)
                and preaudit[f"{split}_future_sha256"] == raw_hash(future),
                f"{folder.name}/{split}: original history/target hash differs",
            )
            for method in METHODS:
                name = f"{method}_{split}_history"
                equal(
                    saved[name],
                    expected[method],
                    f"{folder.name}/{name}: independent preprocessing",
                )
                processed[name] = saved[name]
        require(
            preaudit["processed_sha256"]
            == {name: raw_hash(saved[name]) for name in saved.files},
            f"{folder.name}: processed-data hashes differ",
        )
    initial = initial_forecaster(
        manifest["architecture"], seeds["forecaster_initialization"]
    )
    initialization_hash = forecaster_state_hash(initial)
    parameter_count = sum(p.numel() for p in initial.parameters())
    if config["label"] == "scientific":
        require(
            parameter_count == 413521,
            "Scientific forecaster is not the larger CSDI model",
        )
    audit = read(folder / "training_audit.json")
    require(
        audit["future_targets_unchanged"] is True
        and audit["factor"] == 1
        and set(audit["methods"]) == set(METHODS),
        f"{folder.name}: training audit protocol differs",
    )
    updates = config["epochs"] * math.ceil(config["n_train"] / config["batch_size"])
    candidates = sorted({*config["validation_epochs"], config["epochs"]})
    noise_epochs = [
        epoch
        for epoch in range(1, config["epochs"] + 1)
        if epoch % config["validation_interval"] == 0 or epoch == config["epochs"]
    ]
    selected = {}
    for method in METHODS:
        evidence = audit["methods"][method]
        require(
            evidence["initialization_sha256"] == initialization_hash
            and evidence["parameter_count"] == parameter_count
            and evidence["training_seed"] == seeds["forecaster_training"],
            f"{folder.name}/{method}: paired initialization or optimizer seed differs",
        )
        require(
            evidence["measured_optimizer_updates"] == updates
            and evidence["epochs_completed"] == config["epochs"],
            f"{folder.name}/{method}: measured optimization budget differs",
        )
        require(
            timestamp(mediator_frozen["frozen_utc"])
            <= timestamp(evidence["started_utc"])
            <= timestamp(evidence["completed_utc"])
            <= timestamp(trained["completed_utc"]),
            f"{folder.name}/{method}: mediator-before-forecast chronology differs",
        )
        for split in ("train", "validation"):
            history, future = (
                processed[f"{method}_{split}_history"],
                original[split][:, 32:],
            )
            require(
                evidence[f"{split}_history_sha256"] == raw_hash(history)
                and evidence[f"{split}_future_sha256"] == raw_hash(future)
                and evidence[f"combined_{split}_sha256"]
                == raw_hash(np.concatenate((history, future), axis=1)),
                f"{folder.name}/{method}: processed history/noisy future inputs differ",
            )
        logs = [
            json.loads(line)
            for line in (folder / f"{method}_training.jsonl").read_text().splitlines()
        ]
        require(
            [row["epoch"] for row in logs] == list(range(1, config["epochs"] + 1))
            and [row["epoch"] for row in logs if "validation_loss" in row]
            == noise_epochs
            and [row["epoch"] for row in logs if "selection_score" in row]
            == candidates,
            f"{folder.name}/{method}: training/selection cadence differs",
        )
        for row in logs:
            decays = sum(
                max(1, int(fraction * config["epochs"])) < row["epoch"]
                for fraction in training["lr_decay_fractions"]
            )
            lr = training["learning_rate"] * training["lr_decay_factor"] ** decays
            require(
                np.isclose(row["learning_rate"], lr, rtol=1e-14, atol=0),
                f"{folder.name}/{method}: learning-rate schedule differs",
            )
            require(
                all(
                    np.isfinite(row[key])
                    for key in ("train_loss", "learning_rate", "elapsed_seconds")
                ),
                f"{folder.name}/{method}: nonfinite training log",
            )
        rows = read(folder / f"{method}_validation.json")
        require(
            [row["epoch"] for row in rows]
            == candidates
            == evidence["validation_candidate_epochs"],
            f"{folder.name}/{method}: selection opportunity inventory differs",
        )
        for row in rows:
            name = f"{method}_validation_epoch_{row['epoch']:04d}.npz"
            require(
                row["samples"] == name,
                f"{folder.name}/{method}: candidate filename differs",
            )
            actual = check_ensemble(
                folder / name,
                original["validation"][:, :32],
                processed[f"{method}_validation_history"],
                original["validation"][:, 32:],
                np.arange(config["n_validation"]) + config["n_train"],
                config["samples"],
            )
            require(
                row["scores"] == actual
                and row["crps"]
                == actual["crps"]
                == logs[row["epoch"] - 1]["selection_score"],
                f"{folder.name}/{method}: candidate metric recomputation failed",
            )
        best = min(rows, key=lambda row: row["crps"])
        checkpoint = torch.load(
            folder / f"{method}.pt", map_location="cpu", weights_only=True
        )
        require(
            best["epoch"] == checkpoint["epoch"] == evidence["selected_epoch"]
            and best["crps"]
            == checkpoint["selection_score"]
            == evidence["selected_validation_crps"]
            and checkpoint["validation_loss"]
            == logs[best["epoch"] - 1]["validation_loss"],
            f"{folder.name}/{method}: checkpoint is not first candidate CRPS argmin",
        )
        require(
            json.loads(json.dumps(checkpoint["config"])) == training
            and checkpoint["seed"] == seeds["forecaster_training"]
            and evidence["checkpoint_sha256"] == sha256(folder / f"{method}.pt"),
            f"{folder.name}/{method}: checkpoint configuration/provenance differs",
        )
        require(
            checkpoint["stage"] == "forecaster"
            and checkpoint["factor"] == 1
            and checkpoint["observed_dt"] == checkpoint["target_dt"] == 1
            and checkpoint["context_length"] == checkpoint["dense_context_length"] == 32
            and checkpoint["horizon"] == checkpoint["dense_horizon"] == 8
            and checkpoint["n_augmentations"] == 1
            and checkpoint["augmentation"]["densifier_bypassed"] is True,
            f"{folder.name}/{method}: future-only original-grid objective differs",
        )
        initial.load_state_dict(checkpoint["model_state_dict"], strict=True)
        require(
            all(
                torch.isfinite(value).all()
                for value in checkpoint["model_state_dict"].values()
            ),
            f"{folder.name}/{method}: nonfinite checkpoint weights",
        )
        selected[method] = {"epoch": best["epoch"], "crps": best["crps"]}
    return {
        "selected_checkpoints": selected,
        "measured_updates_per_method": updates,
        "forecaster_parameters": parameter_count,
        "mediator_selected": summary["selected"],
    }


def verify_test_seed(output, manifest, seed):
    config = manifest["experiment"]
    folder = output / f"seed_{seed}"
    seeds = seed_plan(seed, config["namespace"])
    observed = generate_noisy_periodic(config["n_test"], rng=seeds["test_data"])
    raw, target = observed[:, :32], observed[:, 32:]
    ids = np.arange(config["n_test"]) + config["n_train"] + config["n_validation"]
    with np.load(folder / "test.npz", allow_pickle=False) as saved:
        require(
            set(saved.files) == {"history", "target", "trajectory_ids"},
            "Unexpected fresh-test fields",
        )
        equal(saved["history"], raw, f"{folder.name}: fresh noisy test history")
        equal(saved["target"], target, f"{folder.name}: fresh noisy test target")
        equal(saved["trajectory_ids"], ids, f"{folder.name}: fresh test trajectory IDs")
    model = load_mediator(folder / "mediator/selected.pt", device="cpu")
    expected = {
        "raw": raw,
        "learned": learned_history(model, raw),
        "median": median_history(raw),
    }
    scores, audit = read(folder / "scores.json"), read(folder / "inference_audit.json")
    require(
        set(scores) == set(audit) == set(METHODS),
        f"{folder.name}: final method inventory differs",
    )
    for method in METHODS:
        values = expected[method]
        actual = check_ensemble(
            folder / f"{method}_samples.npz",
            raw,
            values,
            target,
            ids,
            config["samples"],
        )
        require(
            scores[method] == actual,
            f"{folder.name}/{method}: final metric recomputation failed",
        )
        row = audit[method]
        require(
            row["conditioning_sha256"]
            == row["observed_argument_sha256"]
            == row["dense_history_argument_sha256"]
            == raw_hash(values)
            and row["conditioning_shape"] == list(values.shape)
            and row["raw_side_channel"] is False
            and row["future_conditioning"] is False
            and row["samples"] == config["samples"]
            and row["horizon"] == 8
            and row["dt"] == 1
            and row["sampling_seed"] == seeds["test_sampling"],
            f"{folder.name}/{method}: wrong inference history or side channel",
        )
    training_times, evaluation_times = (
        read(folder / "timings.json"),
        read(folder / "evaluation_timings.json"),
    )
    require(
        read(folder / "combined_timings.json")
        == {**training_times, **evaluation_times},
        f"{folder.name}: training/inference cost combination differs",
    )
    return scores


def verify_run(output):
    """Replay noisy observations and deterministic mediation to check evidence."""
    output = Path(output)
    # Completion and chronology are established before any generator is called.
    frozen = read(output / "FROZEN.json")
    mediator_frozen = read(output / "MEDIATORS_FROZEN.json")
    trained = read(output / "TRAINED.json")
    started = read(output / "EVALUATION_STARTED.json")
    evaluated = read(output / "EVALUATED.json")
    verify_hashes(output, frozen["files"], "Frozen training evidence")
    manifest = read(output / "config.json")
    config = manifest["experiment"]
    verify_recipe(manifest)
    required, mediator_files = required_training_inventory(output, config)
    require(
        set(frozen["files"]) == required, "Frozen training-artifact inventory differs"
    )
    require(
        set(mediator_frozen["files"]) == mediator_files,
        "Frozen mediator-artifact inventory differs",
    )
    verify_hashes(output, mediator_frozen["files"], "Frozen mediators")
    require(
        mediator_frozen["configuration_sha256"] == sha256(output / "config.json")
        and mediator_frozen["before_any_forecaster_training"] is True
        and frozen["test_data_generated"] is False
        and frozen["all_methods_and_seeds_selected"] is True
        and evaluated["frozen_artifacts_verified"] is True,
        "Freeze markers do not establish the intended ordering",
    )
    require(
        timestamp(manifest["created_utc"])
        <= timestamp(mediator_frozen["frozen_utc"])
        <= timestamp(trained["completed_utc"])
        <= timestamp(frozen["frozen_utc"])
        <= timestamp(started["started_utc"])
        <= timestamp(evaluated["finished_utc"]),
        "Freeze/evaluation chronology differs",
    )
    require(
        set(manifest["seeds"]) == {str(seed) for seed in config["seeds"]}
        and len(set(config["seeds"])) == len(config["seeds"]),
        "Seed inventory differs",
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["threads"])
        reports = {
            str(seed): verify_training_seed(
                output, manifest, seed, mediator_frozen, trained
            )
            for seed in config["seeds"]
        }
        # Inspect every seed's selection evidence before replaying any fresh test.
        for seed in config["seeds"]:
            reports[str(seed)]["test_scores"] = verify_test_seed(output, manifest, seed)
    finally:
        torch.set_num_threads(previous_threads)
    verify_hashes(output, frozen["files"], "Training evidence after verification")
    report = {
        "status": "passed",
        "verified_utc": datetime.now(UTC).isoformat(),
        "label": config["label"],
        "forecast_training_or_sampling_performed": False,
        "clean_signal_replayed_or_used": False,
        "noisy_observations_replayed": True,
        "deterministic_mediator_inference_recomputed": True,
        "seeds": reports,
    }
    (output / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify_run(args.output)
    print(
        f"Verified {len(result['seeds'])} polishing forecast seeds ({result['label']})"
    )
    print(f"Saved {args.output / 'verification.json'}")


if __name__ == "__main__":
    main()
