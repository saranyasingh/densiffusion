"""Independently verify frozen waveform imputation and optional transfer artifacts."""

import argparse
import hashlib
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def require_equal(actual, expected, label):
    require(np.array_equal(actual, expected), f"{label}: array differs")


def epochs_at_interval(epochs, interval):
    return [
        epoch
        for epoch in range(1, epochs + 1)
        if epoch % interval == 0 or epoch == epochs
    ]


def expected_seed_plan(seed):
    names = (
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
    streams = np.random.SeedSequence(seed).spawn(len(names))
    return {
        name: int(stream.generate_state(1)[0])
        for name, stream in zip(names, streams, strict=True)
    }


def verify_recipe(manifest):
    recipe = manifest["recovered_gp_recipe"]
    source = Path(recipe["source"])
    for name in ("plan.json", "selection.json"):
        require(
            sha256(source / name) == recipe["source_sha256"][name],
            f"Recovered GP recipe source changed: {name}",
        )
    plan, selection = (
        read_json(source / "plan.json"),
        read_json(source / "selection.json"),
    )
    require(
        recipe["selected"] == selection["selected"] == "reference_size",
        "Recovered recipe is not the selected larger GP architecture",
    )
    selected = plan["candidates"][selection["selected"]]
    require(
        recipe["architecture"] == manifest["architecture"] == selected["architecture"],
        "Waveform architecture differs from selected GP recipe",
    )
    require(
        recipe["training"] == selected["training"],
        "Recovered GP training recipe differs",
    )
    require(recipe["gp_weights_reused"] is False, "GP weight reuse declared")
    config = manifest["experiment"]
    expected = {
        **selected["training"],
        "epochs": config["epochs"],
        "device": config["device"],
    }
    if config["label"] == "smoke":
        expected.update(validation_interval=1, diffusion_steps=2)
    require(
        manifest["training"] == expected,
        "Training settings differ beyond declared epoch/device/smoke overrides",
    )
    require(
        manifest["training"]["validation_steps"] == "all",
        "Noise-MSE monitoring must average all diffusion steps",
    )
    require(
        type(config["score_interval"]) is int
        and config["score_interval"] > 0
        and config["score_interval"] % manifest["training"]["validation_interval"] == 0,
        "CRPS selection interval must be a positive multiple of the noise-MSE interval",
    )
    if config["label"] != "smoke":
        require(
            manifest["training"]["validation_interval"] == 10,
            "Scientific noise-MSE monitoring must occur every 10 epochs",
        )
        require(
            config["samples"] >= 64, "Scientific scoring requires at least 64 draws"
        )
    require(
        manifest["normalization"] == "none" and manifest["fine_truth_used"] is False,
        "Training normalization or fine-truth declaration differs",
    )


def verify_training_cadence(logs, config, training, label):
    require(
        [row["epoch"] for row in logs] == list(range(1, config["epochs"] + 1)),
        f"{label}: incomplete or duplicated epoch logs",
    )
    require(
        all(np.isfinite(row["train_loss"]) for row in logs),
        f"{label}: nonfinite training loss",
    )
    require(
        [row["epoch"] for row in logs if "validation_loss" in row]
        == epochs_at_interval(config["epochs"], training["validation_interval"]),
        f"{label}: all-step noise-MSE log cadence differs",
    )
    require(
        [row["epoch"] for row in logs if "selection_score" in row]
        == epochs_at_interval(config["epochs"], config["score_interval"]),
        f"{label}: validation-CRPS selection cadence differs",
    )


def inspect_samples(path, target, draws, dt):
    """Check complete sampled arrays, including every unmasked observation."""
    label = str(path)
    with np.load(path, allow_pickle=False) as saved:
        require_equal(saved["target"], target, f"{label}/target")
        expected_mask = np.broadcast_to(
            (np.arange(target.shape[1]) % 2 == 1)[None, :, None], target.shape
        )
        require(saved["mask"].dtype == np.bool_, f"{label}: mask must be boolean")
        require_equal(saved["mask"], expected_mask, f"{label}/mask")
        require_equal(saved["times"], np.arange(target.shape[1]) * dt, f"{label}/times")
        require_equal(saved["trajectory_ids"], np.arange(len(target)), f"{label}/IDs")
        baseline = target.copy()
        baseline[:, 1::2] = (target[:, :-1:2] + target[:, 2::2]) / 2
        scores, arrays = {}, {}
        expected_shape = (len(target), draws, target.shape[1], 1)
        for method in ("learned", "linear"):
            samples = saved[method]
            require(samples.shape == expected_shape, f"{label}/{method}: sample shape")
            require(np.isfinite(samples).all(), f"{label}/{method}: nonfinite samples")
            require_equal(
                samples[:, :, ::2],
                np.broadcast_to(target[:, None, ::2], samples[:, :, ::2].shape),
                f"{label}/{method}/preserved observations",
            )
            if method == "linear":
                require_equal(
                    samples,
                    np.repeat(baseline[:, None], draws, axis=1),
                    f"{label}/linear interpolation",
                )
            scores[method] = evaluate_forecast(samples[:, :, 1::2], target[:, 1::2])
            arrays[method] = samples[:, :, 1::2].copy()
    return scores, arrays


def trajectory_crps(samples, target):
    errors = np.sort(samples - target[:, None], axis=1)
    count = samples.shape[1]
    weights = (2 * np.arange(1, count + 1) - count - 1) / count**2
    point_scores = np.abs(errors).mean(axis=1) - np.sum(
        errors * weights[None, :, None, None], axis=1
    )
    return point_scores.mean(axis=(1, 2))


def bootstrap_difference(left, right, replicates, seed):
    difference = left - right
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 256):
        stop = min(start + 256, replicates)
        rows = rng.integers(len(left), size=(stop - start, len(left)))
        means[start:stop] = difference[rows].mean(axis=1)
    return float(difference.mean()), np.quantile(means, [0.025, 0.975])


def verify_seed(output, manifest, seed):
    config, training = manifest["experiment"], manifest["training"]
    directory = output / f"seed_{seed}"
    source = Path(manifest["data_source"]) / directory.name / "train_validation.npz"
    copied = directory / "train_validation.npz"
    require(
        sha256(source) == sha256(copied) == manifest["data_sha256"][str(seed)],
        f"{directory.name}: original train/validation archive changed",
    )
    with np.load(copied, allow_pickle=False) as saved:
        train, validation = saved["train"], saved["validation"]
        require(
            train.shape[1:] == validation.shape[1:] == (33, 1),
            f"{directory.name}: training/validation geometry must be 33 points",
        )
        require(
            np.isfinite(train).all() and np.isfinite(validation).all(),
            f"{directory.name}: nonfinite observations",
        )
        train_ids, validation_ids = (
            saved["train_trajectory_ids"],
            saved["validation_trajectory_ids"],
        )
        require(
            len(np.unique(train_ids)) == len(train)
            and len(np.unique(validation_ids)) == len(validation)
            and not set(train_ids) & set(validation_ids),
            f"{directory.name}: overlapping IDs",
        )
    if config["label"] == "smoke":
        train, validation = train[:16], validation[:8]
    else:
        require(
            train.shape == (1024, 33, 1) and validation.shape == (256, 33, 1),
            f"{directory.name}: original split sizes changed",
        )
    seeds, summary = (
        read_json(directory / "seeds.json"),
        read_json(directory / "summary.json"),
    )
    require(
        seeds == expected_seed_plan(seed),
        f"{directory.name}: original paired seed allocation changed",
    )
    original_seeds = source.parent / "seeds.json"
    if original_seeds.exists():
        require(
            seeds == read_json(original_seeds),
            f"{directory.name}: original source seed streams changed",
        )
    # Independently recreate the initialization; never read a GP weight file.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seeds["densifier_initialization"])
        initial = ImputationModel(CSDIDenoiser(**manifest["architecture"]))
    require(
        model_hash(initial) == summary["initialization_sha256"],
        f"{directory.name}: model was not the recorded fresh random initialization",
    )
    require(
        sum(p.numel() for p in initial.parameters()) == summary["parameter_count"],
        f"{directory.name}: parameter count differs",
    )
    require(
        summary["gp_weights_reused"] is False
        and summary["observations_preserved"] is True,
        f"{directory.name}: GP reuse or preservation audit failed",
    )
    expected_updates = config["epochs"] * math.ceil(len(train) / training["batch_size"])
    require(
        summary["optimizer_updates"] == expected_updates,
        f"{directory.name}: optimizer budget differs",
    )
    logs = [
        json.loads(line)
        for line in (directory / "training.jsonl").read_text().splitlines()
    ]
    verify_training_cadence(logs, config, training, directory.name)
    score_epochs = epochs_at_interval(config["epochs"], config["score_interval"])
    candidates = read_json(directory / "validation.json")
    require(
        [row["epoch"] for row in candidates] == score_epochs,
        f"{directory.name}: missing candidate CRPS evaluation",
    )
    candidate_arrays = {}
    for row in candidates:
        path = directory / row["samples"]
        expected_filename = f"validation_epoch_{row['epoch']:04d}.npz"
        require(
            row["samples"] == expected_filename, f"{directory.name}: candidate filename"
        )
        scores, arrays = inspect_samples(path, validation, config["samples"], 0.1)
        require(
            scores["learned"] == row["scores"] and scores["linear"] == row["linear"],
            f"{directory.name}/{path.name}: exact metric recomputation failed",
        )
        require(
            logs[row["epoch"] - 1]["selection_score"] == scores["learned"]["crps"],
            f"{directory.name}/{path.name}: CRPS disagrees with selection log",
        )
        candidate_arrays[row["epoch"]] = arrays
    selected = min(candidates, key=lambda row: row["scores"]["crps"])
    require(
        summary["selected_validation"] == selected
        and summary["selected_epoch"] == selected["epoch"],
        f"{directory.name}: summary selection is not validation CRPS argmin",
    )
    require(
        sha256(directory / "validation_samples.npz")
        == sha256(directory / selected["samples"]),
        f"{directory.name}: selected samples differ from selected candidate",
    )
    checkpoint = torch.load(
        directory / "densifier.pt", map_location="cpu", weights_only=True
    )
    require(
        checkpoint["epoch"] == selected["epoch"]
        and checkpoint["selection_score"] == selected["scores"]["crps"],
        f"{directory.name}: checkpoint is not first validation CRPS argmin",
    )
    require(
        "validation_interpolation_crps" in checkpoint["selection_metric"],
        f"{directory.name}: checkpoint selected by a different metric",
    )
    require(
        json.loads(json.dumps(checkpoint["config"])) == training
        and checkpoint["seed"] == seeds["densifier_training"],
        f"{directory.name}: checkpoint training configuration or seed differs",
    )
    require(
        checkpoint["validation_loss"] == summary["validation_noise_mse"],
        f"{directory.name}: selected noise-MSE differs",
    )
    require(
        checkpoint["stage"] == "densifier"
        and checkpoint["factor"] == 2
        and checkpoint["target_dt"] == 0.1,
        f"{directory.name}: checkpoint training geometry differs",
    )
    initial.load_state_dict(checkpoint["model_state_dict"], strict=True)
    require(
        all(
            torch.isfinite(value).all().item()
            for value in checkpoint["model_state_dict"].values()
        ),
        f"{directory.name}: nonfinite checkpoint",
    )
    per_trajectory = {
        method: trajectory_crps(samples, validation[:, 1::2])
        for method, samples in candidate_arrays[selected["epoch"]].items()
    }
    return {
        "selected_epoch": selected["epoch"],
        "candidate_epochs": score_epochs,
        "train_trajectories": len(train),
        "validation_trajectories": len(validation),
        "parameter_count": summary["parameter_count"],
        "optimizer_updates": expected_updates,
        "exact_scores": {"learned": selected["scores"], "linear": selected["linear"]},
        "data_sha256": sha256(copied),
        "selected_samples_sha256": sha256(directory / "validation_samples.npz"),
    }, per_trajectory


def verify_gate(manifest, analysis, trajectories):
    configured = manifest["experiment"]["seeds"]
    observed = sorted(trajectories)
    require(
        set(analysis["validation"]["seeds"]) == {f"seed_{seed}" for seed in observed},
        "Validation analysis seed inventory differs",
    )
    bootstrap = analysis["bootstrap"]
    learned, linear, upper = [], [], []
    for index, seed in enumerate(observed):
        entry = analysis["validation"]["seeds"][f"seed_{seed}"]
        sampling_seed = bootstrap["base_seed"] + 100 * index
        require(
            entry["bootstrap_seed"] == sampling_seed,
            "Bootstrap seed allocation differs",
        )
        left, right = trajectories[seed]["learned"], trajectories[seed]["linear"]
        mean_difference, ci = bootstrap_difference(
            left, right, bootstrap["replicates"], sampling_seed
        )
        stored = entry["learned_minus_linear"]["crps"]
        require(
            np.isclose(stored["difference"], mean_difference, rtol=0, atol=1e-14)
            and np.allclose(stored["ci95"], ci, rtol=0, atol=1e-14),
            f"seed_{seed}: paired trajectory bootstrap differs",
        )
        for method in ("learned", "linear"):
            stored_values = entry["scores"][method]["per_trajectory"]["crps"]
            require(
                np.allclose(
                    stored_values, trajectories[seed][method], rtol=0, atol=1e-14
                ),
                f"seed_{seed}/{method}: per-trajectory CRPS differs",
            )
        learned.append(float(left.mean()))
        linear.append(float(right.mean()))
        upper.append(float(ci[1]))
    relative = 1 - float(np.mean(learned)) / float(np.mean(linear))
    criteria = {
        "exactly_three_paired_seeds": len(observed)
        == manifest["gate"]["seeds_required"],
        "configured_seed_inventory_complete": set(observed) == set(configured),
        "learned_crps_lower_in_every_seed": all(
            a < b for a, b in zip(learned, linear, strict=True)
        ),
        "mean_relative_crps_reduction_at_least_threshold": relative
        >= manifest["gate"]["minimum_relative_mean_crps_reduction"],
        "every_seed_ci95_upper_below_zero": all(value < 0 for value in upper),
    }
    gate = analysis["validation_gate"]
    require(
        gate["minimum_relative_improvement"]
        == manifest["gate"]["minimum_relative_mean_crps_reduction"]
        and gate["observed_seeds"] == observed
        and gate["expected_seeds"] == sorted(configured),
        "Validation gate threshold or seed inventory differs",
    )
    require(
        np.isclose(gate["mean_learned_crps"], np.mean(learned), rtol=0, atol=1e-14)
        and np.isclose(gate["mean_linear_crps"], np.mean(linear), rtol=0, atol=1e-14)
        and gate["transfer_scores_used"] is False,
        "Validation gate mean scores or selection source differ",
    )
    require(
        gate["criteria"] == criteria and gate["passed"] == all(criteria.values()),
        "Validation gate does not reproduce from saved paired trajectories",
    )
    require(
        np.isclose(gate["relative_crps_reduction"], relative, rtol=0, atol=1e-14),
        "Mean relative validation improvement differs",
    )
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "relative_crps_reduction": relative,
    }


def verify_transfer(output, manifest, frozen):
    path = output / "TRANSFER.json"
    files = list(output.glob("seed_*/transfer*.npz"))
    if not path.exists():
        require(not files, "Transfer samples exist without a completed TRANSFER.json")
        return {"status": "not_run"}
    require(
        frozen["validation_gate"]["passed"] is True,
        "Transfer truth was generated despite a failed validation gate",
    )
    report = read_json(path)
    config, gate = manifest["experiment"], manifest["transfer_gate"]
    require(
        report["gate"] == gate
        and report["diagnostic_only"] is True
        and report["no_tuning_after_fine_truth"] is True,
        "Transfer diagnostic protocol differs",
    )
    require(
        set(report["seeds"]) == {str(seed) for seed in config["seeds"]},
        "Transfer seed inventory differs",
    )
    require(
        set(files)
        == {
            output / f"seed_{seed}" / name
            for seed in config["seeds"]
            for name in ("transfer_samples.npz", "transfer_future_samples.npz")
        },
        "Transfer sample inventory differs",
    )
    passed = True
    for seed in config["seeds"]:
        directory = output / f"seed_{seed}"
        streams = np.random.SeedSequence([seed, config["diagnostic_namespace"]]).spawn(
            3
        )
        data_seed, history_seed, future_seed = [
            int(s.generate_state(1)[0]) for s in streams
        ]
        # Fine data replay is reachable only after freeze, gate, and completion checks.
        fine = generate_waveforms(config["transfer_examples"], 65, 0.05, rng=data_seed)
        result = read_json(directory / "transfer_scores.json")
        require(
            result == report["seeds"][str(seed)] and result["data_seed"] == data_seed,
            f"seed_{seed}: transfer seed metadata differs",
        )
        require(
            set(result["tasks"]) == {"history", "anchored_future"},
            f"seed_{seed}: transfer requires history and anchored-future tasks",
        )
        for name, target, sampling_seed, filename in (
            ("history", fine[:, :49], history_seed, "transfer_samples.npz"),
            (
                "anchored_future",
                fine[:, 48:],
                future_seed,
                "transfer_future_samples.npz",
            ),
        ):
            scores, _ = inspect_samples(
                directory / filename, target, config["samples"], 0.05
            )
            saved = result["tasks"][name]
            require(
                saved["learned"] == scores["learned"]
                and saved["linear"] == scores["linear"],
                f"seed_{seed}/{name}: transfer metric recomputation failed",
            )
            require(
                saved["sampling_seed"] == sampling_seed
                and saved["observations_preserved"] is True,
                f"seed_{seed}/{name}: transfer sampling protocol differs",
            )
            crps_ratio = scores["learned"]["crps"] / scores["linear"]["crps"]
            mse_ratio = scores["learned"]["mse"] / scores["linear"]["mse"]
            require(
                saved["crps_ratio"] == crps_ratio and saved["mse_ratio"] == mse_ratio,
                f"seed_{seed}/{name}: transfer ratios differ",
            )
            passed &= (
                crps_ratio <= gate["maximum_crps_ratio_to_linear"]
                and mse_ratio <= gate["maximum_mse_ratio_to_linear"]
            )
    require(report["passed"] == passed, "Transfer gate does not reproduce")
    return {
        "status": "verified",
        "passed": passed,
        "tasks": {"history_points": 49, "anchored_future_points": 17},
    }


def verify_run(output):
    output = Path(output)
    read_json(output / "TRAINED.json")
    frozen = read_json(output / "DENSIFIERS_FROZEN.json")
    require(
        sha256(output / "config.json") == frozen["config_sha256"],
        "Frozen config changed",
    )
    require(
        sha256(output / "imputation_analysis.json") == frozen["analysis_sha256"],
        "Frozen analysis changed",
    )
    manifest, analysis = (
        read_json(output / "config.json"),
        read_json(output / "imputation_analysis.json"),
    )
    seeds = manifest["experiment"]["seeds"]
    require(
        set(frozen["checkpoint_sha256"])
        == {f"seed_{seed}/densifier.pt" for seed in seeds},
        "Frozen checkpoint inventory differs",
    )
    for name, expected in frozen["checkpoint_sha256"].items():
        require(sha256(output / name) == expected, f"Frozen checkpoint changed: {name}")
    require(
        frozen["validation_gate"] == analysis["validation_gate"],
        "Frozen validation gate differs from frozen analysis",
    )
    require(
        frozen["fine_grid_truth_generated"] is False,
        "Fine-grid truth existed before the densifier freeze",
    )
    verify_recipe(manifest)
    reports, trajectories = {}, {}
    for seed in seeds:
        reports[str(seed)], trajectories[seed] = verify_seed(output, manifest, seed)
    gate = verify_gate(manifest, analysis, trajectories)
    transfer = verify_transfer(output, manifest, frozen)
    result = {
        "status": "passed",
        "verified_utc": datetime.now(UTC).isoformat(),
        "config_sha256": frozen["config_sha256"],
        "analysis_sha256": frozen["analysis_sha256"],
        "gp_weights_loaded_by_verifier": False,
        "inference_or_training_performed": False,
        "fresh_random_initializations_reproduced": True,
        "seeds": reports,
        "validation_gate": gate,
        "transfer": transfer,
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
    print(f"Verified {len(report['seeds'])} frozen waveform densifiers")
    print(f"Saved {args.output / 'verification.json'}")


if __name__ == "__main__":
    main()
