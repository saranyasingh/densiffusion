"""Verify completed waveform artifacts without training or resampling forecasts."""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from densiffusion.generators.waveform import generate_waveforms
from densiffusion.metrics import evaluate_forecast

METHODS = ("direct_csdi", "linear_densification", "densiffusion")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_array(actual, expected, description):
    require(np.array_equal(actual, expected), f"{description} differs")


def audit_call(call, source, factor, n_samples, dt, description):
    require(call["input_shape"] == list(source.shape), f"{description}: input shape")
    require(
        call["input_sha256"] == hashlib.sha256(source.tobytes()).hexdigest(),
        f"{description}: conditioning input hash",
    )
    require(
        call["output_shape"]
        == [len(source), n_samples, (source.shape[1] - 1) * factor + 1, 1],
        f"{description}: output shape",
    )
    require(call["observations_preserved"] is True, f"{description}: observations")
    require(call["dt"] == dt, f"{description}: spacing")
    require(
        np.isfinite(call["seconds"]) and call["seconds"] >= 0, f"{description}: timing"
    )


def augmentation_inputs(train, validation, config):
    """Independently reconstruct ordered past-only and anchored-future inputs."""
    context = config["context_length"]
    augmentations = config["n_augmentations"]
    batch_size = config["batch_size"]
    for values in (train, validation):
        history, future = values[:, :context], values[:, context:]
        for start in range(0, len(values) * augmentations, batch_size):
            stop = min(start + batch_size, len(values) * augmentations)
            rows = np.arange(start, stop) // augmentations
            yield history[rows]
            yield np.concatenate((history[rows, -1:], future[rows]), axis=1)


def validation_epochs(epochs, interval):
    return [
        epoch
        for epoch in range(1, epochs + 1)
        if epoch % interval == 0 or epoch == epochs
    ]


def verify_training(directory, config, manifest, seeds):
    records = [
        json.loads(line)
        for line in (directory / "training.jsonl").read_text().splitlines()
        if line
    ]
    require(
        {row["method"] for row in records} == {*METHODS, "densifier"},
        f"{directory.name}: unexpected/missing training methods",
    )
    selected = {}
    for method in ("densifier", *METHODS):
        densifier = method == "densifier"
        budget = config["densifier_epochs" if densifier else "epochs"]
        interval = config[
            "densifier_validation_interval" if densifier else "validation_interval"
        ]
        rows = [row for row in records if row["method"] == method]
        prefix = f"{directory.name}/{method}"
        require(
            [row["epoch"] for row in rows] == list(range(1, budget + 1)),
            f"{prefix}: incomplete, duplicated, or unordered epoch logs",
        )
        require(
            all(np.isfinite(row["train_loss"]) for row in rows),
            f"{prefix}: nonfinite training loss",
        )
        expected_epochs = validation_epochs(budget, interval)
        selection_rows = [row for row in rows if "selection_score" in row]
        require(
            [row["epoch"] for row in selection_rows] == expected_epochs,
            f"{prefix}: validation epochs",
        )
        require(
            all(np.isfinite(row["selection_score"]) for row in selection_rows),
            f"{prefix}: nonfinite selection scores",
        )
        checkpoint = torch.load(
            directory / ("densifier.pt" if densifier else f"{method}_forecaster.pt"),
            map_location="cpu",
            weights_only=True,
        )
        training_name = "densifier_training" if densifier else "forecaster_training"
        require(
            json.loads(json.dumps(checkpoint["config"])) == manifest[training_name],
            f"{prefix}: checkpoint training configuration",
        )
        require(
            checkpoint["seed"] == seeds[training_name],
            f"{prefix}: checkpoint training seed",
        )
        if densifier:
            candidates = selection_rows
        else:
            candidates = read_json(directory / f"{method}_validation.json")
            require(
                [row["epoch"] for row in candidates] == expected_epochs,
                f"{prefix}: forecast validation epochs",
            )
            for score, record in zip(candidates, selection_rows, strict=True):
                require(
                    score["crps"] == record["selection_score"],
                    f"{prefix}: CRPS disagrees with selection log",
                )
            require(
                "forecast_crps" in checkpoint["selection_metric"],
                f"{prefix}: selected by a different metric",
            )
        key = "selection_score" if densifier else "crps"
        best = min(candidates, key=lambda row: row[key])
        require(
            checkpoint["epoch"] == best["epoch"],
            f"{prefix}: selected epoch is not first validation argmin",
        )
        require(
            checkpoint["selection_score"] == best[key],
            f"{prefix}: checkpoint selection score differs",
        )
        require(
            all(
                torch.isfinite(tensor).all().item()
                for tensor in checkpoint["model_state_dict"].values()
            ),
            f"{prefix}: nonfinite checkpoint",
        )
        selected[method] = {"epoch": best["epoch"], "selection_score": best[key]}
    pairing = read_json(directory / "pairing_audit.json")
    hashes = pairing["initial_state_sha256"]
    require(
        set(hashes) == set(METHODS) and len(set(hashes.values())) == 1,
        f"{directory.name}: forecaster initializations differ",
    )
    updates = config["epochs"] * math.ceil(
        config["n_train"] * config["n_augmentations"] / config["batch_size"]
    )
    require(
        pairing["updates_per_forecaster"] == updates,
        f"{directory.name}: forecaster update budget",
    )
    require(
        pairing["shared_training_seed"] == seeds["forecaster_training"],
        f"{directory.name}: paired training seed",
    )
    require(
        pairing["all_original_observations_preserved"] is True,
        f"{directory.name}: observation preservation audit failed",
    )
    return selected, updates


def verify_run(output):
    output = Path(output)
    # Require completed training AND evaluation before replaying any observations.
    frozen = read_json(output / "FROZEN.json")
    read_json(output / "EVALUATED.json")
    require(
        file_hash(output / "config.json") == frozen["config_sha256"],
        "Frozen configuration hash differs",
    )
    manifest = read_json(output / "config.json")
    config = manifest["experiment"]
    expected_files = {
        f"seed_{seed}/{name}"
        for seed in config["seeds"]
        for name in ("densifier.pt", *(f"{method}_forecaster.pt" for method in METHODS))
    }
    require(
        set(frozen["checkpoint_sha256"]) == expected_files,
        "Frozen checkpoint inventory differs from configured seeds and methods",
    )
    for name, expected in frozen["checkpoint_sha256"].items():
        require(
            file_hash(output / name) == expected, f"Frozen checkpoint differs: {name}"
        )
    for seed in config["seeds"]:
        read_json(output / f"seed_{seed}" / "TRAINED.json")
    aggregate_scores = read_json(output / "scores.json")
    require(
        set(aggregate_scores) == {str(seed) for seed in config["seeds"]},
        "Aggregate score seed inventory differs",
    )
    draws = config["n_histories"] * config["n_futures"]
    require(
        config["label"] == "smoke" or draws >= 64,
        "Scientific run has fewer than 64 scored draws",
    )
    report = {
        "status": "passed",
        "verified_utc": datetime.now(UTC).isoformat(),
        "config_sha256": frozen["config_sha256"],
        "draws_per_trajectory": draws,
        "forecast_inference_or_training_performed": False,
        "original_grid_data_replayed": True,
        "observation_preservation_evidence": (
            "Saved runtime checks and matching conditioning-input hashes; "
            "dense augmentation outputs are not resampled."
        ),
        "seeds": {},
    }
    for seed in config["seeds"]:
        directory = output / f"seed_{seed}"
        seeds = read_json(directory / "seeds.json")
        require(
            seeds == manifest["seeds"][str(seed)],
            f"{directory.name}: seed allocation differs",
        )
        original = {
            split: generate_waveforms(
                config[f"n_{split}"],
                config["context_length"] + config["horizon"],
                config["dt"],
                rng=seeds[f"{split}_data"],
            )
            for split in ("train", "validation", "test")
        }
        with np.load(directory / "train_validation.npz", allow_pickle=False) as data:
            for split in ("train", "validation"):
                equal_array(data[split], original[split], f"{directory.name}/{split}")
            ids = [data["train_trajectory_ids"], data["validation_trajectory_ids"]]
        scores = read_json(directory / "scores.json")
        require(
            scores == aggregate_scores[str(seed)],
            f"{directory.name}: aggregate and seed metrics differ",
        )
        context, horizon = config["context_length"], config["horizon"]
        with np.load(directory / "samples.npz", allow_pickle=False) as data:
            history, target = (
                original["test"][:, :context],
                original["test"][:, context:],
            )
            equal_array(data["history"], history, f"{directory.name}/test history")
            equal_array(data["target"], target, f"{directory.name}/test target")
            equal_array(
                data["history_times"],
                np.arange(context) * config["dt"],
                f"{directory.name}/history times",
            )
            equal_array(
                data["future_times"],
                np.arange(context, context + horizon) * config["dt"],
                f"{directory.name}/future times",
            )
            ids.append(data["trajectory_ids"])
            offset = 0
            for split, trajectory_ids in zip(
                ("train", "validation", "test"), ids, strict=True
            ):
                count = config[f"n_{split}"]
                equal_array(
                    trajectory_ids,
                    np.arange(offset, offset + count),
                    f"{directory.name}/{split} trajectory IDs",
                )
                offset += count
            require(
                len(np.unique(np.concatenate(ids))) == offset,
                f"{directory.name}: trajectory IDs overlap",
            )
            expected_shape = (config["n_test"], draws, horizon, 1)
            for method in METHODS:
                samples = data[method]
                require(
                    samples.shape == expected_shape,
                    f"{directory.name}/{method}: scored sample shape",
                )
                require(
                    np.isfinite(samples).all(),
                    f"{directory.name}/{method}: nonfinite forecasts",
                )
                require(
                    evaluate_forecast(samples, target) == scores[method],
                    f"{directory.name}/{method}: exact metric recomputation failed",
                )
        selected, updates = verify_training(directory, config, manifest, seeds)
        inputs = list(
            augmentation_inputs(original["train"], original["validation"], config)
        )
        inference_audits = []
        for method in METHODS:
            calls = read_json(directory / f"{method}_augmentation_audit.json")
            expected_inputs = [] if method == "direct_csdi" else inputs
            require(
                len(calls) == len(expected_inputs),
                f"{directory.name}/{method}: augmentation call count",
            )
            for index, (call, source) in enumerate(
                zip(calls, expected_inputs, strict=True)
            ):
                audit_call(
                    call,
                    source,
                    config["factor"],
                    1,
                    config["dt"],
                    f"{directory.name}/{method}/augmentation[{index}]",
                )
            path = directory / f"{method}_inference_audit.json"
            if path.exists():
                inference_calls = read_json(path)
                require(
                    len(inference_calls) == (0 if method == "direct_csdi" else 1),
                    f"{directory.name}/{method}: inference densification call count",
                )
                for call in inference_calls:
                    audit_call(
                        call,
                        history,
                        config["factor"],
                        config["n_histories"],
                        config["dt"],
                        f"{directory.name}/{method}/inference",
                    )
                inference_audits.append(method)
        report["seeds"][str(seed)] = {
            "samples_sha256": file_hash(directory / "samples.npz"),
            "trajectory_counts": {
                split: len(values) for split, values in original.items()
            },
            "selected_checkpoints": selected,
            "updates_per_forecaster": updates,
            "matching_augmentation_calls_per_dense_method": len(inputs),
            "inference_audits_verified": inference_audits,
            "exact_recomputed_scores": scores,
        }
    (output / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    report = verify_run(args.output)
    print(f"Verified {len(report['seeds'])} seeds")
    print(f"Saved {args.output / 'verification.json'}")


if __name__ == "__main__":
    main()
