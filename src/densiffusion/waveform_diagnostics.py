"""Post-freeze interpolation diagnostics, separate from forecasting selection.

Fine-grid truth is generated only after both the model freeze and confirmatory
test evaluation have completed. The trained-resolution task withholds original
observations; the transfer task uses new independently noisy fine-grid targets.
Neither task assumes that independent dense histories and future augmentations
are draws from a joint fine-resolution trajectory.
"""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.contracts import positive_int, validate_array
from densiffusion.generators.waveform import (
    draw_waveform_parameters,
    evaluate_waveforms,
)
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig
from densiffusion.waveform_experiment import LinearDensifier, timed, write_json


def _verify_frozen(output):
    for name in ("FROZEN.json", "EVALUATED.json"):
        if not (output / name).is_file():
            raise FileNotFoundError(f"Diagnostics require completed {name}")
    frozen = json.loads((output / "FROZEN.json").read_text())
    json.loads((output / "EVALUATED.json").read_text())
    if (
        hashlib.sha256((output / "config.json").read_bytes()).hexdigest()
        != frozen["config_sha256"]
    ):
        raise ValueError("Configuration changed after freezing")
    for name, expected in frozen["checkpoint_sha256"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Checkpoint changed after freezing")
    return frozen


def _load_densifier(path, manifest):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = manifest["experiment"]
    model = ImputationModel(CSDIDenoiser(**manifest["architecture"]))
    model.load_state_dict(saved["model_state_dict"])
    model.to(config["device"]).requires_grad_(False).eval()
    sampling = replace(
        TrainingConfig(**saved["config"]),
        device=config["device"],
        batch_size=config["sample_batch_size"],
    )
    return DDPMSampler(model, sampling)


def _replay_test(config, seeds, history, target):
    """Replay all original draws before evaluating any finer-resolution times."""
    rng = np.random.default_rng(seeds["test_data"])
    parameters = draw_waveform_parameters(config["n_test"], rng=rng)
    times = np.arange(config["context_length"] + config["horizon"]) * config["dt"]
    signal = evaluate_waveforms(parameters, times)
    originals = signal + rng.normal(0.0, 0.02, size=signal.shape)
    expected = np.concatenate((history, target), axis=1)
    if not np.array_equal(originals, expected):
        raise ValueError(
            "Saved test observations do not match the test-data RNG replay"
        )
    return parameters, signal


def _score_task(observed, noisy_truth, signal_truth, q, *, dt, seed, n_samples, device):
    arrays = {
        "observed": observed,
        "noisy_truth": noisy_truth,
        "noiseless_signal_truth": signal_truth,
        "hidden_indices": np.arange(1, noisy_truth.shape[1], 2),
        "times": np.arange(noisy_truth.shape[1]) * dt / 2,
    }
    scores = {}
    timings = {}
    for method, sampler in (("learned", q), ("linear", LinearDensifier())):
        samples, seconds = timed(
            sampler.sample,
            observed.copy(),
            factor=2,
            n_samples=n_samples,
            dt=dt,
            rng=np.random.default_rng(seed),
            device=device,
        )
        samples = validate_array(
            samples, (len(observed), n_samples, noisy_truth.shape[1], 1)
        )
        expected = np.broadcast_to(observed[:, None], samples[:, :, ::2].shape)
        if not np.array_equal(samples[:, :, ::2], expected):
            raise AssertionError("Densifier changed observed values in diagnostics")
        hidden = samples[:, :, 1::2].copy()
        arrays[f"{method}_hidden_samples"] = hidden
        scores[method] = {
            "noisy_observations": evaluate_forecast(hidden, noisy_truth[:, 1::2]),
            "noiseless_signal": evaluate_forecast(hidden, signal_truth[:, 1::2]),
        }
        timings[method] = {
            "inference_seconds": seconds,
            "inference_seconds_per_trajectory": seconds / len(observed),
        }
    differences = {
        truth: {
            metric: scores["learned"][truth][metric] - scores["linear"][truth][metric]
            for metric in scores["learned"][truth]
        }
        for truth in ("noisy_observations", "noiseless_signal")
    }
    return arrays, {
        "input_dt": dt,
        "output_dt": dt / 2,
        "n_observed": observed.shape[1],
        "n_hidden": noisy_truth[:, 1::2].shape[1],
        "n_samples": n_samples,
        "sampling_seed": seed,
        "observations_preserved": True,
        "scores": scores,
        "learned_minus_linear": differences,
        "timings": timings,
    }


def run_diagnostics(output_dir, n_examples=256, n_samples=64):
    """Save post-freeze interpolation samples, truth, metrics, and timings.

    Uses the first min(n_examples, n_test) test trajectories of each seed without
    selecting examples by their scores. Refuses existing diagnostics or altered
    frozen inputs. Configurations, checkpoints, and forecast artifacts are read
    only. No returned or written quantity is used to select or train a model.
    """
    positive_int("n_examples", n_examples)
    positive_int("n_samples", n_samples)
    output = Path(output_dir)
    frozen = _verify_frozen(output)
    destination = output / "diagnostics"
    if destination.exists():
        raise FileExistsError(f"Diagnostics already exist: {destination}")
    manifest = json.loads((output / "config.json").read_text())
    config = manifest["experiment"]
    if config["context_length"] < 3 or config["context_length"] % 2 != 1:
        raise ValueError("Diagnostics require an odd history length of at least three")
    torch.set_num_threads(config["threads"])
    count = min(n_examples, config["n_test"])
    destination.mkdir()
    start = perf_counter()
    result = {
        "diagnostic_version": 1,
        "created_after_freeze_and_test_evaluation": True,
        "config_sha256": frozen["config_sha256"],
        "n_examples_per_seed": count,
        "n_samples": n_samples,
        "subset": "first test trajectories in saved order, chosen without scores",
        "noise_std": 0.02,
        "interpretation": [
            "Post-freeze diagnostic only; never used for training or selection.",
            "trained_resolution withholds alternate original noisy observations.",
            "transferred_resolution adds independent noise at new fine timestamps.",
            "Noiseless signal scores are separate from noisy observation scores.",
            "Linear interpolation repeats one point estimate, so its interval width "
            "is zero and its coverage is not a probabilistic calibration baseline.",
            "Differences between resolutions also reflect different gap sizes; "
            "these diagnostics do not by themselves isolate resolution transfer.",
            "These history-only interpolation diagnostics do not establish that "
            "independently densified histories and futures form joint fine truth.",
        ],
        "seeds": {},
    }
    for seed in config["seeds"]:
        source = output / f"seed_{seed}"
        folder = destination / source.name
        folder.mkdir()
        seeds = manifest["seeds"][str(seed)]
        with np.load(source / "samples.npz", allow_pickle=False) as saved:
            history = saved["history"]
            target = saved["target"]
            ids = saved["trajectory_ids"][:count]
        parameters, original_signal = _replay_test(config, seeds, history, target)
        streams = np.random.SeedSequence(seeds["diagnostic_sampling"]).spawn(3)
        noise_seed, trained_seed, transferred_seed = (
            int(stream.generate_state(1)[0]) for stream in streams
        )
        fine_times = np.arange(2 * config["context_length"] - 1) * config["dt"] / 2
        fine_signal = evaluate_waveforms(parameters, fine_times)[:count]
        fine_truth = fine_signal.copy()
        fine_truth[:, ::2] = history[:count]
        fine_truth[:, 1::2] += np.random.default_rng(noise_seed).normal(
            0.0, 0.02, size=fine_truth[:, 1::2].shape
        )
        q = _load_densifier(source / "densifier.pt", manifest)
        tasks = (
            (
                "trained_resolution",
                history[:count, ::2],
                history[:count],
                original_signal[:count, : config["context_length"]],
                2 * config["dt"],
                trained_seed,
            ),
            (
                "transferred_resolution",
                history[:count],
                fine_truth,
                fine_signal,
                config["dt"],
                transferred_seed,
            ),
        )
        entry = {"original_test_rng_reproduced_exactly": True, "tasks": {}}
        for name, observed, truth, signal, dt, sampling_seed in tasks:
            arrays, scores = _score_task(
                observed,
                truth,
                signal,
                q,
                dt=dt,
                seed=sampling_seed,
                n_samples=n_samples,
                device=config["device"],
            )
            np.savez_compressed(folder / f"{name}.npz", trajectory_ids=ids, **arrays)
            entry["tasks"][name] = scores
        entry["fine_observation_noise_seed"] = noise_seed
        result["seeds"][source.name] = entry
        write_json(folder / "scores.json", entry)
    result["total_seconds"] = perf_counter() - start
    result["finished_utc"] = datetime.now(UTC).isoformat()
    # Recheck that diagnostics have not altered frozen inputs.
    _verify_frozen(output)
    write_json(destination / "scores.json", result)
    return result
