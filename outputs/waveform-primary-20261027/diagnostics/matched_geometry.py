"""Post-freeze full-window imputation with the densifier's training geometry.

Run from the repository environment:
  .venv/bin/python outputs/waveform-primary-20261027/diagnostics/matched_geometry.py

This diagnostic conditions on alternating observations across the entire test
span 0..3.2, including observed times inside the forecasting horizon. Its samples
are full-window imputations only. No forecasting conditioning histories, model
weights, hyperparameters, or checkpoint choices are constructed or changed.
"""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.contracts import positive_int
from densiffusion.metrics import evaluate_forecast
from densiffusion.waveform_diagnostics import (
    _load_densifier,
    _replay_test,
    _score_task,
    _verify_frozen,
)
from densiffusion.waveform_experiment import write_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_seed(arguments):
    output, manifest, seed, n_examples, n_samples = arguments
    _verify_frozen(output)
    config = manifest["experiment"]
    torch.set_num_threads(config["threads"])
    source = output / f"seed_{seed}"
    destination = output / "diagnostics" / "matched_geometry" / source.name
    destination.mkdir()
    with np.load(source / "samples.npz", allow_pickle=False) as archive:
        history = archive["history"]
        target = archive["target"]
        ids = archive["trajectory_ids"][:n_examples]
    seeds = manifest["seeds"][str(seed)]
    _, signal = _replay_test(config, seeds, history, target)
    originals = np.concatenate((history, target), axis=1)[:n_examples]
    if originals.shape[1] != 33:
        raise ValueError("This matched-geometry diagnostic requires 33-point windows")
    sampling_seed = int(
        np.random.SeedSequence([seeds["diagnostic_sampling"], 330017]).generate_state(
            1
        )[0]
    )
    q = _load_densifier(source / "densifier.pt", manifest)
    arrays, scores = _score_task(
        originals[:, ::2],
        originals,
        signal[:n_examples],
        q,
        dt=2 * config["dt"],
        seed=sampling_seed,
        n_samples=n_samples,
        device=config["device"],
    )
    np.savez_compressed(destination / "samples.npz", trajectory_ids=ids, **arrays)
    with np.load(destination / "samples.npz", allow_pickle=False) as archive:
        for method in ("learned", "linear"):
            for label, key in (
                ("noisy_observations", "noisy_truth"),
                ("noiseless_signal", "noiseless_signal_truth"),
            ):
                actual = evaluate_forecast(
                    archive[f"{method}_hidden_samples"], archive[key][:, 1::2]
                )
                if actual != scores["scores"][method][label]:
                    raise AssertionError("Saved sample metrics did not reproduce")
    scores.update(
        n_examples=len(originals),
        original_test_rng_reproduced_exactly=True,
        scores_recomputed_from_saved_samples=True,
        artifact_sha256=sha256(destination / "samples.npz"),
    )
    write_json(destination / "scores.json", scores)
    _verify_frozen(output)
    print(
        f"seed={seed} learned_crps="
        f"{scores['scores']['learned']['noisy_observations']['crps']:.9f} "
        f"linear_crps={scores['scores']['linear']['noisy_observations']['crps']:.9f}",
        flush=True,
    )
    return str(seed), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    for name in ("examples", "samples", "workers"):
        positive_int(name, getattr(args, name))
    output = args.output.resolve()
    frozen = _verify_frozen(output)
    manifest = json.loads((output / "config.json").read_text())
    config = manifest["experiment"]
    count = min(args.examples, config["n_test"])
    protected = [output / name for name in frozen["checkpoint_sha256"]]
    protected.extend(
        output / name for name in ("config.json", "FROZEN.json", "EVALUATED.json")
    )
    protected.extend(output / f"seed_{seed}/samples.npz" for seed in config["seeds"])
    protected.extend(
        path for path in (output / "diagnostics").rglob("*") if path.is_file()
    )
    before = {str(path.relative_to(output)): sha256(path) for path in protected}
    destination = output / "diagnostics" / "matched_geometry"
    destination.mkdir(exist_ok=False)
    start = perf_counter()
    result = {
        "diagnostic": "matched training geometry full-window imputation",
        "started_utc": datetime.now(UTC).isoformat(),
        "post_freeze_and_confirmatory_test_evaluation": True,
        "forecast_claim": False,
        "scope": (
            "Conditions on original indices 0,2,...,32 across the full test span; "
            "scores original indices 1,3,...,31. Future observations within that "
            "span are permitted solely for this completed-experiment imputation "
            "diagnostic. No forecasting histories or models are constructed."
        ),
        "geometry": {
            "full_length": 33,
            "observed_length": 17,
            "hidden_length": 16,
            "input_dt": 2 * config["dt"],
            "output_dt": config["dt"],
        },
        "n_examples_per_seed": count,
        "subset": "first test trajectories in saved order, chosen without scores",
        "n_samples": args.samples,
        "workers": args.workers,
        "script_sha256": sha256(Path(__file__).resolve()),
        "configuration_sha256": frozen["config_sha256"],
        "new_fine_grid_truth_generated": False,
        "notes": [
            "Noisy targets are existing original-grid test observations.",
            "Noise-free targets are separately labeled replayed test signals.",
            "Linear ensembles repeat one interpolation and have zero interval width.",
            "Training geometry is matched in length, alternating mask, and spacing.",
        ],
    }
    jobs = [(output, manifest, seed, count, args.samples) for seed in config["seeds"]]
    if args.workers == 1:
        rows = [run_seed(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            rows = list(executor.map(run_seed, jobs))
    result["seeds"] = dict(rows)
    result["across_seed_mean"] = {
        method: {
            truth: {
                metric: float(
                    np.mean([row[1]["scores"][method][truth][metric] for row in rows])
                )
                for metric in ("crps", "mse", "coverage_90", "interval_width_90")
            }
            for truth in ("noisy_observations", "noiseless_signal")
        }
        for method in ("learned", "linear")
    }
    result["elapsed_seconds"] = perf_counter() - start
    result["finished_utc"] = datetime.now(UTC).isoformat()
    _verify_frozen(output)
    for name, expected in before.items():
        if sha256(output / name) != expected:
            raise AssertionError(f"Existing artifact changed: {name}")
    result["verification"] = {
        "frozen_inputs_unchanged": True,
        "all_protected_existing_artifacts_unchanged": True,
        "protected_sha256": before,
        "scores_recomputed_from_saved_samples": True,
    }
    write_json(destination / "scores.json", result)
    print(json.dumps(result["across_seed_mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
