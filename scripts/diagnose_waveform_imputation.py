"""Bounded train/validation conditioning check after a failed imputation gate.

Example:
  .venv/bin/python scripts/diagnose_waveform_imputation.py --attempt ATTEMPT_DIR

Reads only that attempt's original training/validation observations and selected
densifier checkpoints. Uses the first 32 rows of each split and 64 draws, without
training, selecting checkpoints, reading test data, or generating fine truth.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.models import ImputationModel
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig
from densiffusion.waveform_experiment import model_hash, timed, write_json
from densiffusion.waveform_imputation import hidden_scores, linear_samples, sha256

N_EXAMPLES = 32
N_SAMPLES = 64


def verify_failed_gate(attempt):
    """Refuse running/incomplete, passed, or already transferred experiments."""
    if not (attempt / "TRAINED.json").is_file():
        raise ValueError("All training must finish before this diagnostic")
    analysis = json.loads((attempt / "imputation_analysis.json").read_text())
    if analysis["validation_gate"]["passed"] is not False:
        raise ValueError("This bounded diagnostic is restricted to failed gates")
    if (attempt / "TRANSFER.json").exists() or any(
        attempt.glob("seed_*/transfer*.npz")
    ):
        raise ValueError("This diagnostic must precede all fine-grid transfer checks")
    manifest = json.loads((attempt / "config.json").read_text())
    if manifest["recovered_gp_recipe"]["gp_weights_reused"] is not False:
        raise ValueError("Require fresh waveform initialization without GP weights")
    return manifest


def sample_conditions(sampler, observed, *, seed, device):
    samples, seconds = timed(
        sampler.sample,
        observed.copy(),
        factor=2,
        n_samples=N_SAMPLES,
        dt=0.2,
        rng=np.random.default_rng(seed),
        device=device,
    )
    expected = np.broadcast_to(observed[:, None], samples[:, :, ::2].shape)
    if not np.array_equal(samples[:, :, ::2], expected):
        raise AssertionError("Sampler changed supplied conditioning observations")
    if not np.isfinite(samples).all():
        raise FloatingPointError("Nonfinite diagnostic samples")
    return samples, seconds


def diagnose_seed(arguments):
    attempt, destination, seed, manifest = arguments
    verify_failed_gate(attempt)
    config = manifest["experiment"]
    torch.set_num_threads(config["threads"])
    source = attempt / f"seed_{seed}"
    folder = destination / source.name
    folder.mkdir()
    data_path = source / "train_validation.npz"
    if sha256(data_path) != manifest["data_sha256"][str(seed)]:
        raise ValueError("Original training/validation arrays changed")
    seeds = json.loads((source / "seeds.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    torch.manual_seed(seeds["densifier_initialization"])
    model = ImputationModel(CSDIDenoiser(**manifest["architecture"]))
    if model_hash(model) != summary["initialization_sha256"]:
        raise AssertionError("Recorded initialization differs from fresh paired seed")
    if summary["gp_weights_reused"] is not False:
        raise AssertionError("Summary reports GP weight reuse")
    state = torch.load(source / "densifier.pt", map_location="cpu", weights_only=True)
    if state["epoch"] != summary["selected_epoch"]:
        raise AssertionError("Selected checkpoint and summary differ")
    if state["seed"] != seeds["densifier_training"]:
        raise AssertionError("Checkpoint training seed differs from paired seed plan")
    model.load_state_dict(state["model_state_dict"])
    model.to(config["device"]).requires_grad_(False).eval()
    before_model = model_hash(model)
    sampling = replace(
        TrainingConfig(**state["config"]),
        device=config["device"],
        batch_size=config["sample_batch_size"],
    )
    sampler = DDPMSampler(model, sampling)
    result = {
        "checkpoint_epoch": state["epoch"],
        "checkpoint_sha256": sha256(source / "densifier.pt"),
        "fresh_initialization_verified": True,
        "gp_weights_reused": False,
        "splits": {},
    }
    with np.load(data_path, allow_pickle=False) as data:
        for split_index, split in enumerate(("train", "validation")):
            target = data[split][:N_EXAMPLES].copy()
            if target.shape != (N_EXAMPLES, 33, 1):
                raise ValueError("Diagnostic requires 32 original 33-point rows")
            ids = data[f"{split}_trajectory_ids"][:N_EXAMPLES]
            permutation = np.roll(np.arange(N_EXAMPLES), 1)
            observed = target[:, ::2]
            permuted_observed = observed[permutation]
            sampling_seed = int(
                np.random.SeedSequence(
                    [seeds["validation_sampling"], 2026102703, split_index]
                ).generate_state(1)[0]
            )
            correct, correct_seconds = sample_conditions(
                sampler, observed, seed=sampling_seed, device=config["device"]
            )
            permuted, permuted_seconds = sample_conditions(
                sampler,
                permuted_observed,
                seed=sampling_seed,
                device=config["device"],
            )
            # Both calls use identical Gaussian-noise streams. Only observations
            # change, and all scores still use each row's own hidden targets.
            linear_correct = linear_samples(target, N_SAMPLES)
            linear_permuted = linear_samples(target[permutation], N_SAMPLES)
            ensembles = {
                "learned_correct": correct,
                "learned_permuted": permuted,
                "linear_correct": linear_correct,
                "linear_permuted": linear_permuted,
            }
            scores = {
                name: hidden_scores(samples, target)
                for name, samples in ensembles.items()
            }
            mean_change = (permuted - correct)[:, :, 1::2].mean(axis=1)
            row = {
                "n_examples": N_EXAMPLES,
                "n_samples": N_SAMPLES,
                "n_hidden": 16,
                "sampling_seed": sampling_seed,
                "same_noise_for_both_conditioning_cases": True,
                "supplied_observations_preserved": True,
                "scores": scores,
                "learned_permuted_minus_correct": {
                    metric: scores["learned_permuted"][metric]
                    - scores["learned_correct"][metric]
                    for metric in scores["learned_correct"]
                },
                "ensemble_mean_conditioning_response_mae": float(
                    np.abs(mean_change).mean()
                ),
                "ensemble_mean_conditioning_response_rms": float(
                    np.sqrt(np.square(mean_change).mean())
                ),
                "paired_sample_conditioning_response_rms": float(
                    np.sqrt(np.square((permuted - correct)[:, :, 1::2]).mean())
                ),
                "sampling_seconds": {
                    "correct": correct_seconds,
                    "permuted": permuted_seconds,
                },
            }
            path = folder / f"{split}_samples.npz"
            np.savez_compressed(
                path,
                target=target,
                observed=observed,
                permuted_observed=permuted_observed,
                trajectory_ids=ids,
                permutation=permutation,
                permuted_trajectory_ids=ids[permutation],
                times=np.arange(33) * 0.1,
                hidden_indices=np.arange(1, 33, 2),
                **ensembles,
            )
            with np.load(path, allow_pickle=False) as saved:
                for name in ensembles:
                    if hidden_scores(saved[name], saved["target"]) != scores[name]:
                        raise AssertionError("Saved diagnostic scores do not reproduce")
            row["scores_recomputed_from_saved_samples"] = True
            result["splits"][split] = row
    if model_hash(model) != before_model or model.training:
        raise AssertionError("Diagnostics changed model state or evaluation mode")
    write_json(folder / "scores.json", result)
    print(
        json.dumps(
            {
                "seed": seed,
                "crps": {
                    split: {
                        name: score["crps"] for name, score in row["scores"].items()
                    }
                    for split, row in result["splits"].items()
                },
            }
        ),
        flush=True,
    )
    return str(seed), result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    attempt = args.attempt.resolve()
    manifest = verify_failed_gate(attempt)
    destination = (
        args.output.resolve()
        if args.output
        else attempt / "diagnostics" / "conditioning"
    )
    protected = {
        str(path.relative_to(attempt)): sha256(path)
        for path in attempt.rglob("*")
        if path.is_file()
    }
    destination.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    jobs = [
        (attempt, destination, seed, manifest)
        for seed in manifest["experiment"]["seeds"]
    ]
    if args.workers > 1 and manifest["experiment"]["device"] == "cpu":
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)), mp_context=get_context("spawn")
        ) as executor:
            rows = list(executor.map(diagnose_seed, jobs))
    else:
        rows = [diagnose_seed(job) for job in jobs]
    verify_failed_gate(attempt)
    if any(sha256(attempt / name) != value for name, value in protected.items()):
        raise AssertionError("Existing attempt artifacts changed during diagnostics")
    result = {
        "diagnostic": "bounded training/validation conditioning sensitivity",
        "scope": (
            "Descriptive diagnostic of the selected densifier after a failed "
            "validation gate. First 32 train and first 32 validation trajectories; "
            "only alternate hidden original-grid points scored. This diagnostic "
            "does not select checkpoints or establish a forecasting claim."
        ),
        "permutation": "one-row cyclic shift within each selected split; no fixed rows",
        "matching_noise": "same private Torch RNG seed for correct/permuted conditions",
        "test_data_accessed": False,
        "fine_truth_generated_or_accessed": False,
        "models_trained_or_changed": False,
        "existing_artifacts_unchanged": True,
        "protected_sha256": protected,
        "script_sha256": sha256(Path(__file__)),
        "seeds": dict(rows),
        "wall_seconds": perf_counter() - start,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    write_json(destination / "scores.json", result)
    return result


if __name__ == "__main__":
    main()
