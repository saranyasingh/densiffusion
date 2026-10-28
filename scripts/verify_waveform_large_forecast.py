"""Audit completed conditional forecasts without training or forecast inference."""

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.generators.waveform import generate_waveforms
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel

METHODS = ("direct_csdi", "linear_densification", "densiffusion")
METRICS = ("crps", "mse", "coverage_90", "interval_width_90")
SEED_NAMES = (
    "train_data",
    "validation_data",
    "test_data",
    "densifier_initialization",
    "densifier_training",
    "forecaster_initialization",
    "forecaster_training",
    "validation_sampling",
    "test_sampling",
    "diagnostic_sampling",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hashes(root, hashes, description):
    for name, expected in hashes.items():
        require(sha256(root / name) == expected, f"{description} changed: {name}")


def equal(actual, expected, description):
    require(np.array_equal(actual, expected), f"{description} differs")


def seed_plans(seed, namespace):
    original = {
        name: int(stream.generate_state(1)[0])
        for name, stream in zip(
            SEED_NAMES, np.random.SeedSequence(seed).spawn(len(SEED_NAMES)), strict=True
        )
    }
    fresh = dict(original)
    for name, stream in zip(
        ("test_data", "test_sampling"),
        np.random.SeedSequence([seed, namespace]).spawn(2),
        strict=True,
    ):
        fresh[name] = int(stream.generate_state(1)[0])
        require(
            fresh[name] != original[name],
            "Fresh test seed collides with original stream",
        )
    return original, fresh


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def initial_model(architecture, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ImputationModel(CSDIDenoiser(**architecture))


def imputation_verifier():
    path = Path(__file__).with_name("verify_waveform_imputation.py")
    spec = importlib.util.spec_from_file_location("source_imputation_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source(manifest):
    provenance = manifest["densifier_import"]
    source = Path(provenance["source"])
    verify_hashes(source, provenance["source_sha256"], "Imported densifier source")
    source_manifest = read(source / "config.json")
    frozen = read(source / "DENSIFIERS_FROZEN.json")
    analysis = read(source / "imputation_analysis.json")
    transfer = read(source / "TRANSFER.json")
    require(
        sha256(source / "config.json") == frozen["config_sha256"]
        and sha256(source / "imputation_analysis.json") == frozen["analysis_sha256"],
        "Imported source freeze metadata changed",
    )
    verify_hashes(source, frozen["checkpoint_sha256"], "Source frozen densifier")
    require(
        frozen["validation_gate"] == analysis["validation_gate"]
        and analysis["validation_gate"]["passed"] is True
        and transfer["passed"] is True,
        "Both source validation and transfer gates must pass",
    )
    require(
        source_manifest["recovered_gp_recipe"]["gp_weights_reused"] is False,
        "Imported source declares GP weight reuse",
    )
    require(
        manifest["architecture"]
        == provenance["architecture"]
        == source_manifest["architecture"]
        == source_manifest["recovered_gp_recipe"]["architecture"],
        "Forecaster/imported densifier architectures differ",
    )
    require(
        manifest["densifier_training"] == source_manifest["training"],
        "Imported densifier training recipe differs",
    )
    require(
        provenance["new_training_seconds"] == 0,
        "Imported densifier reports new training",
    )
    for previous in provenance["prior_development_sources"]:
        verify_hashes(
            Path(previous["source"]),
            previous["source_sha256"],
            "Prior development source",
        )
    if manifest["experiment"]["label"] == "smoke" and source_manifest["experiment"].get(
        "label"
    ) in (None, "smoke"):
        scope = "Controlled smoke fixture: source gate booleans and hashes verified"
    else:
        helper = imputation_verifier()
        helper.verify_recipe(source_manifest)
        trajectories = {}
        for seed in source_manifest["experiment"]["seeds"]:
            _, trajectories[seed] = helper.verify_seed(source, source_manifest, seed)
        validation = helper.verify_gate(source_manifest, analysis, trajectories)
        transfer_check = helper.verify_transfer(source, source_manifest, frozen)
        require(
            validation["passed"] and transfer_check["passed"],
            "Source gates do not reproduce from saved samples",
        )
        scope = "Source validation bootstrap and transfer tasks recomputed"
    return source, source_manifest, scope


def check_ensemble(path, history, target, ids, config):
    with np.load(path, allow_pickle=False) as saved:
        equal(saved["history"], history, f"{path.name}/history")
        equal(saved["target"], target, f"{path.name}/target")
        equal(saved["trajectory_ids"], ids, f"{path.name}/trajectory IDs")
        equal(
            saved["history_times"],
            np.arange(config["context_length"]) * config["dt"],
            f"{path.name}/history times",
        )
        equal(
            saved["future_times"],
            np.arange(
                config["context_length"], config["context_length"] + config["horizon"]
            )
            * config["dt"],
            f"{path.name}/future times",
        )
        samples = saved["samples"]
        require(
            samples.shape
            == (
                len(target),
                config["n_histories"] * config["n_futures"],
                config["horizon"],
                1,
            ),
            f"{path.name}: forecast shape",
        )
        require(np.isfinite(samples).all(), f"{path.name}: nonfinite forecasts")
        return evaluate_forecast(samples, target)


def expected_augmentations(train, validation, config):
    for data in (train, validation):
        context, copies = config["context_length"], config["n_augmentations"]
        history, future = data[:, :context], data[:, context:]
        for start in range(0, len(data) * copies, config["batch_size"]):
            rows = (
                np.arange(start, min(start + config["batch_size"], len(data) * copies))
                // copies
            )
            yield history[rows]
            yield np.concatenate((history[rows, -1:], future[rows]), axis=1)


def verify_call(call, values, config, n_samples, description):
    require(
        call["input_shape"] == list(values.shape)
        and call["input_sha256"] == hashlib.sha256(values.tobytes()).hexdigest(),
        f"{description}: conditioning differs from authorized original observations",
    )
    require(
        call["output_shape"]
        == [len(values), n_samples, (values.shape[1] - 1) * config["factor"] + 1, 1],
        f"{description}: dense shape",
    )
    require(
        call["dt"] == config["dt"] and call["observations_preserved"] is True,
        f"{description}: spacing or observation preservation differs",
    )


def verify_seed(output, source, source_manifest, manifest, seed):
    config, provenance = manifest["experiment"], manifest["densifier_import"]
    directory = output / f"seed_{seed}"
    original, fresh = seed_plans(seed, config["test_namespace"])
    require(
        original
        == manifest["original_seed_plans"][str(seed)]
        == read(source / directory.name / "seeds.json"),
        f"{directory.name}: original seed allocation changed",
    )
    require(
        fresh == manifest["seeds"][str(seed)] == read(directory / "seeds.json"),
        f"{directory.name}: fresh test namespace or paired streams differ",
    )
    for name in ("train_validation.npz", "densifier.pt"):
        require(
            sha256(directory / name)
            == sha256(source / directory.name / name)
            == provenance["source_sha256"][f"{directory.name}/{name}"],
            f"{directory.name}/{name}: import is not byte-identical",
        )
    imported = provenance["checkpoints"][str(seed)]
    require(
        imported["source"] == f"{directory.name}/densifier.pt"
        and imported["sha256"] == sha256(directory / "densifier.pt"),
        f"{directory.name}: densifier checkpoint provenance differs",
    )
    q = torch.load(directory / "densifier.pt", map_location="cpu", weights_only=True)
    summary = read(source / directory.name / "summary.json")
    require(
        json.loads(json.dumps(q["config"]))
        == imported["config"]
        == source_manifest["training"]
        and q["epoch"] == imported["selected_epoch"] == summary["selected_epoch"],
        f"{directory.name}: imported densifier schedule or epoch differs",
    )
    require(
        imported["parameter_count"] == summary["parameter_count"],
        f"{directory.name}: imported parameter count differs",
    )
    with np.load(directory / "train_validation.npz", allow_pickle=False) as saved:
        train, validation = saved["train"], saved["validation"]
        train_ids, validation_ids = (
            saved["train_trajectory_ids"],
            saved["validation_trajectory_ids"],
        )
    length = config["context_length"] + config["horizon"]
    require(
        train.shape == (config["n_train"], length, 1)
        and validation.shape == (config["n_validation"], length, 1),
        f"{directory.name}: original-grid data shape",
    )
    require(
        len(set(train_ids)) == len(train)
        and len(set(validation_ids)) == len(validation)
        and not set(train_ids) & set(validation_ids),
        f"{directory.name}: trajectory IDs overlap",
    )
    if config["label"] != "smoke":
        for split, values in (("train", train), ("validation", validation)):
            equal(
                values,
                generate_waveforms(
                    len(values), length, config["dt"], rng=original[f"{split}_data"]
                ),
                f"{directory.name}/{split} original-grid replay",
            )
    context = config["context_length"]
    history, target = validation[:, :context], validation[:, context:]
    model = initial_model(manifest["architecture"], fresh["forecaster_initialization"])
    initial_hash = state_hash(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    audit = read(directory / "pairing_audit.json")
    require(
        audit["initial_state_sha256"] == {method: initial_hash for method in METHODS},
        f"{directory.name}: paired fresh initializations differ",
    )
    require(
        audit["forecaster_parameter_count"]
        == audit["additional_densifier_parameters"]
        == parameter_count
        == imported["parameter_count"],
        f"{directory.name}: architecture parameter counts",
    )
    expected_updates = config["epochs"] * math.ceil(
        config["n_train"] * config["n_augmentations"] / config["batch_size"]
    )
    require(
        audit["updates_per_forecaster"] == expected_updates
        and audit["measured_updates_by_method"]
        == {method: expected_updates for method in METHODS}
        and audit["completed_epochs_by_method"]
        == {method: config["epochs"] for method in METHODS},
        f"{directory.name}: measured optimizer budgets differ",
    )
    require(
        audit["shared_training_seed"] == fresh["forecaster_training"]
        and audit["densifier_weights_unchanged"] is True
        and audit["densifier_import_sha256"] == imported["sha256"]
        and audit["all_original_observations_preserved"] is True,
        f"{directory.name}: pairing or frozen densifier audit differs",
    )
    logs = [
        json.loads(line)
        for line in (directory / "training.jsonl").read_text().splitlines()
    ]
    require(
        {row["method"] for row in logs} == set(METHODS),
        f"{directory.name}: unexpected densifier retraining or missing method logs",
    )
    score_epochs = [
        epoch
        for epoch in range(1, config["epochs"] + 1)
        if epoch % config["validation_score_interval"] == 0 or epoch == config["epochs"]
    ]
    noise_epochs = [
        epoch
        for epoch in range(1, config["epochs"] + 1)
        if epoch % config["validation_interval"] == 0 or epoch == config["epochs"]
    ]
    expected_inputs = list(expected_augmentations(train, validation, config))
    selected = {}
    for method in METHODS:
        method_logs = [row for row in logs if row["method"] == method]
        require(
            [row["epoch"] for row in method_logs]
            == list(range(1, config["epochs"] + 1))
            and [row["epoch"] for row in method_logs if "validation_loss" in row]
            == noise_epochs
            and [row["epoch"] for row in method_logs if "selection_score" in row]
            == score_epochs,
            f"{directory.name}/{method}: training/validation cadence differs",
        )
        rows = read(directory / f"{method}_validation.json")
        require(
            [row["epoch"] for row in rows] == score_epochs,
            f"{directory.name}/{method}: missing candidate selection opportunity",
        )
        for row in rows:
            name = f"{method}_validation_epoch_{row['epoch']:04d}.npz"
            require(
                row["samples"] == name
                and sha256(directory / name) == row["samples_sha256"],
                f"{directory.name}/{method}: candidate ensemble hash differs",
            )
            scores = check_ensemble(
                directory / name, history, target, validation_ids, config
            )
            require(
                scores == {key: row[key] for key in METRICS},
                f"{directory.name}/{method}: candidate metric recomputation failed",
            )
            require(
                scores["crps"] == method_logs[row["epoch"] - 1]["selection_score"],
                f"{directory.name}/{method}: logged selection score differs",
            )
        best = min(rows, key=lambda row: row["crps"])
        checkpoint = torch.load(
            directory / f"{method}_forecaster.pt", map_location="cpu", weights_only=True
        )
        require(
            checkpoint["epoch"] == best["epoch"]
            and checkpoint["selection_score"] == best["crps"],
            f"{directory.name}/{method}: checkpoint differs from first CRPS argmin",
        )
        require(
            json.loads(json.dumps(checkpoint["config"]))
            == manifest["forecaster_training"]
            and checkpoint["seed"] == fresh["forecaster_training"]
            and "forecast_crps" in checkpoint["selection_metric"],
            f"{directory.name}/{method}: training configuration or selection metric",
        )
        factor = 1 if method == "direct_csdi" else config["factor"]
        require(
            checkpoint["factor"] == factor
            and checkpoint["context_length"] == context
            and checkpoint["horizon"] == config["horizon"]
            and checkpoint["target_dt"] == config["dt"] / factor,
            f"{directory.name}/{method}: physical history/horizon geometry",
        )
        augmentation = checkpoint["augmentation"]
        require(
            augmentation["history_conditioning"] == "history_only"
            and augmentation["future_conditioning"] == "last_history_point_and_future"
            and augmentation["independent_history_future_draws"] is True
            and augmentation["cached"] is True,
            f"{directory.name}/{method}: independent augmentation construction differs",
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        require(
            all(
                torch.isfinite(value).all().item()
                for value in checkpoint["model_state_dict"].values()
            ),
            f"{directory.name}/{method}: nonfinite checkpoint",
        )
        calls = read(directory / f"{method}_augmentation_audit.json")
        inputs = [] if factor == 1 else expected_inputs
        require(
            len(calls) == len(inputs),
            f"{directory.name}/{method}: augmentation call count",
        )
        for index, (call, values) in enumerate(zip(calls, inputs, strict=True)):
            verify_call(
                call,
                values,
                config,
                1,
                f"{directory.name}/{method}/augmentation[{index}]",
            )
        selected[method] = {
            "epoch": best["epoch"],
            "crps": best["crps"],
            "candidate_sha256": best["samples_sha256"],
        }
    timings = read(directory / "timings.json")
    selected_cost = provenance["selected_attempt_training_seconds_by_seed"][str(seed)]
    require(
        selected_cost == summary["training_seconds"]
        and timings["densifier_training_seconds"] == 0
        and timings["imported_densifier_training_seconds"] == selected_cost,
        f"{directory.name}: imported q training-cost attribution",
    )
    prior_cost = sum(
        read(Path(previous["source"]) / directory.name / "summary.json")[
            "training_seconds"
        ]
        for previous in provenance["prior_development_sources"]
    )
    require(
        prior_cost
        == provenance["prior_development_training_seconds_by_seed"][str(seed)]
        == timings["prior_densifier_development_seconds"],
        f"{directory.name}: prior development cost",
    )
    for method in METHODS:
        row = timings["methods"][method]
        expected_cost = selected_cost if method == "densiffusion" else 0
        require(
            row["imported_densifier_training_seconds"] == expected_cost
            and row["total_training_seconds"]
            == row["new_training_seconds"] + expected_cost,
            f"{directory.name}/{method}: total training cost",
        )
    # No test generation is reachable until the global markers and source gates
    # were checked by verify_run, and this seed's training evidence is validated.
    test = generate_waveforms(
        config["n_test"], length, config["dt"], rng=fresh["test_data"]
    )
    test_history, test_target = test[:, :context], test[:, context:]
    scores = read(directory / "scores.json")
    with np.load(directory / "samples.npz", allow_pickle=False) as saved:
        equal(saved["history"], test_history, f"{directory.name}/fresh test history")
        equal(saved["target"], test_target, f"{directory.name}/fresh test target")
        test_ids = np.arange(
            config["n_train"] + config["n_validation"],
            config["n_train"] + config["n_validation"] + config["n_test"],
        )
        equal(saved["trajectory_ids"], test_ids, f"{directory.name}/fresh test IDs")
        require(
            not set(test_ids) & (set(train_ids) | set(validation_ids)),
            f"{directory.name}: test IDs overlap original splits",
        )
        equal(
            saved["history_times"],
            np.arange(context) * config["dt"],
            f"{directory.name}/test history times",
        )
        equal(
            saved["future_times"],
            np.arange(context, length) * config["dt"],
            f"{directory.name}/test future times",
        )
        for method in METHODS:
            samples = saved[method]
            require(
                samples.shape
                == (
                    config["n_test"],
                    config["n_histories"] * config["n_futures"],
                    config["horizon"],
                    1,
                )
                and np.isfinite(samples).all(),
                f"{directory.name}/{method}: final sample shape/finite check",
            )
            require(
                evaluate_forecast(samples, test_target) == scores[method],
                f"{directory.name}/{method}: final metric recomputation failed",
            )
    for method in METHODS:
        calls = read(directory / f"{method}_inference_audit.json")
        require(
            len(calls) == (0 if method == "direct_csdi" else 1),
            f"{directory.name}/{method}: inference densification call count",
        )
        for call in calls:
            verify_call(
                call,
                test_history,
                config,
                config["n_histories"],
                f"{directory.name}/{method}/test history",
            )
    return {
        "selected_checkpoints": selected,
        "measured_updates_per_forecaster": expected_updates,
        "parameter_count": parameter_count,
        "imported_q_sha256": imported["sha256"],
        "samples_sha256": sha256(directory / "samples.npz"),
        "exact_test_scores": scores,
        "fresh_test_data_seed": fresh["test_data"],
    }


def verify_run(output):
    output = Path(output)
    frozen, evaluated = read(output / "FROZEN.json"), read(output / "EVALUATED.json")
    require(
        sha256(output / "config.json") == frozen["config_sha256"],
        "Frozen configuration changed",
    )
    manifest = read(output / "config.json")
    config = manifest["experiment"]
    expected_checkpoints = {
        f"seed_{seed}/{name}"
        for seed in config["seeds"]
        for name in ("densifier.pt", *(f"{method}_forecaster.pt" for method in METHODS))
    }
    require(
        set(frozen["checkpoint_sha256"]) == expected_checkpoints,
        "Frozen checkpoint inventory differs",
    )
    require(
        set(frozen["train_validation_sha256"])
        == {f"seed_{seed}/train_validation.npz" for seed in config["seeds"]},
        "Frozen original-data inventory differs",
    )
    candidate_epochs = [
        epoch
        for epoch in range(1, config["epochs"] + 1)
        if epoch % config["validation_score_interval"] == 0 or epoch == config["epochs"]
    ]
    expected_evidence = set()
    for seed in config["seeds"]:
        folder = f"seed_{seed}"
        expected_evidence.update(
            f"{folder}/{name}" for name in ("training.jsonl", "pairing_audit.json")
        )
        for method in METHODS:
            expected_evidence.update(
                (
                    f"{folder}/{method}_validation.json",
                    f"{folder}/{method}_augmentation_audit.json",
                )
            )
            expected_evidence.update(
                f"{folder}/{method}_validation_epoch_{epoch:04d}.npz"
                for epoch in candidate_epochs
            )
    require(
        set(frozen["selection_evidence_sha256"]) == expected_evidence,
        "Frozen selection-evidence inventory differs",
    )
    for field in (
        "checkpoint_sha256",
        "train_validation_sha256",
        "selection_evidence_sha256",
    ):
        verify_hashes(output, frozen[field], f"Frozen {field}")
    for seed in config["seeds"]:
        read(output / f"seed_{seed}" / "TRAINED.json")
    require(
        datetime.fromisoformat(frozen["frozen_utc"])
        <= datetime.fromisoformat(evaluated["finished_utc"]),
        "Test evaluation completion predates global freeze",
    )
    require(
        manifest["fine_grid_truth_used_for_training_or_selection"] is False
        and manifest["transfer_diagnostics_completed_before_forecast_training"] is True
        and manifest["test_data_generated_before_all_models_frozen"] is False
        and manifest["joint_fine_truth_assumption"] is False
        and manifest["fresh_forecast_test_required"] is True
        and manifest["normalization"] == "none",
        "Forecast conditioning/selection scope metadata differs",
    )
    if config["label"] != "smoke":
        scientific_settings = {
            "epochs": 600,
            "validation_interval": 10,
            "validation_score_interval": 150,
            "n_train": 1024,
            "n_validation": 256,
            "n_test": 1024,
            "context_length": 25,
            "horizon": 8,
            "dt": 0.1,
            "factor": 2,
            "batch_size": 64,
            "n_histories": 8,
            "n_futures": 8,
            "n_augmentations": 1,
            "diffusion_steps": 50,
            "test_namespace": 2026102702,
        }
        require(
            all(
                config[name] == expected
                for name, expected in scientific_settings.items()
            )
            and len(config["seeds"]) == 3,
            "Scientific comparison budget, geometry, seed count, or namespace differs",
        )
    else:
        require(
            config["test_namespace"] != 2026102702,
            "Smoke consumed the scientific test namespace",
        )
    source, source_manifest, source_scope = verify_source(manifest)
    expected_training = {
        **source_manifest["recovered_gp_recipe"]["training"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "device": config["device"],
        "validation_steps": "all",
        "validation_interval": config["validation_interval"],
        "diffusion_steps": config["diffusion_steps"],
    }
    require(
        manifest["forecaster_training"] == expected_training,
        "Paired forecaster optimizer or diffusion recipe differs",
    )
    reports = {
        str(seed): verify_seed(output, source, source_manifest, manifest, seed)
        for seed in config["seeds"]
    }
    aggregate = read(output / "scores.json")
    require(
        aggregate
        == {seed: report["exact_test_scores"] for seed, report in reports.items()},
        "Aggregate final scores differ from seed artifacts",
    )
    # Source verification helpers are read-only; no new source artifacts allowed.
    verify_hashes(
        source,
        manifest["densifier_import"]["source_sha256"],
        "Source after verification",
    )
    result = {
        "status": "passed",
        "verified_utc": datetime.now(UTC).isoformat(),
        "config_sha256": frozen["config_sha256"],
        "source_gate_verification": source_scope,
        "forecast_training_or_inference_performed": False,
        "fresh_test_streams_replayed": True,
        "seeds": reports,
        "preservation_evidence": (
            "Byte-identical q imports, frozen hashes, saved runtime checks, "
            "and matching original-conditioning input hashes; "
            "dense outputs were not resampled."
        ),
    }
    (output / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    report = verify_run(args.output)
    print(f"Verified {len(report['seeds'])} conditional forecast seeds")
    print(f"Saved {args.output / 'verification.json'}")


if __name__ == "__main__":
    main()
