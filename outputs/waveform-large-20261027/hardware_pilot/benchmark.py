"""Bounded CPU/MPS timing only: random weights and Gaussian conditioning inputs."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.models import ImputationModel
from densiffusion.samplers import DDPMForecastSampler
from densiffusion.training import TrainingConfig

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
RECIPE = ROOT / "outputs/csdi-gp-better/plan.json"
INITIALIZATION_SEED = 860401
INPUT_SEED = 860402
SAMPLING_SEED = 860403
CASES = ((33, 25, 8, 0.1), (65, 49, 16, 0.05))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def worker(device):
    torch.set_num_threads(1)
    recipe = json.loads(RECIPE.read_text())["candidates"]["reference_size"]
    training = replace(TrainingConfig(**recipe["training"]), device=device, batch_size=256)
    report = {"device": device, "torch_threads": 1, "worker_count": 1,
              "architecture": recipe["architecture"], "training_config": asdict(training),
              "cases": [], "status": "running"}
    write_json(OUTPUT / f"{device}.json", report)
    if device == "mps" and not torch.backends.mps.is_available():
        report["status"] = "unavailable"
        write_json(OUTPUT / f"{device}.json", report)
        return
    torch.manual_seed(INITIALIZATION_SEED)
    model = ImputationModel(CSDIDenoiser(**recipe["architecture"])).eval()
    report["initial_weights_sha256"] = state_hash(model)
    report["parameter_count"] = sum(p.numel() for p in model.parameters())
    model.to(device).requires_grad_(False)
    sampler = DDPMForecastSampler(model, training)
    for sequence_length, context, horizon, dt in CASES:
        dense_history = np.random.default_rng(INPUT_SEED + sequence_length).normal(
            size=(4, context, 1)
        )
        observed = dense_history.copy() if context == 25 else dense_history[:, ::2].copy()
        timings = {}
        for phase in ("warmup", "timed"):
            synchronize(device)
            start = perf_counter()
            samples = sampler.sample(
                observed, dense_history, horizon=horizon, n_samples=64, dt=dt,
                rng=np.random.default_rng(SAMPLING_SEED),
            )
            synchronize(device)
            timings[f"{phase}_seconds"] = perf_counter() - start
            if not np.isfinite(samples).all():
                raise FloatingPointError("Nonfinite random-model timing output")
        report["cases"].append({
            "sequence_length": sequence_length, "context_length": context,
            "hidden_suffix_length": horizon, "dt": dt,
            "n_input_histories": 4, "draws_per_history": 64,
            "effective_sample_batch": 256, "diffusion_steps": 50,
            "input_sha256": hashlib.sha256(dense_history.tobytes()).hexdigest(),
            "output_shape": list(samples.shape), "finite_outputs": True,
            **timings, "timed_seconds_per_draw": timings["timed_seconds"] / 256,
        })
        write_json(OUTPUT / f"{device}.json", report)
        print(json.dumps({"device": device, **report["cases"][-1]}), flush=True)
    report["status"] = "complete"
    write_json(OUTPUT / f"{device}.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"))
    args = parser.parse_args()
    if args.device:
        worker(args.device)
        return
    result = {
        "purpose": "Hardware timing only; no training, quality scoring, or model selection",
        "started_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "numpy": str(np.__version__),
        "torch": str(torch.__version__), "mps_available": torch.backends.mps.is_available(),
        "initialization_seed": INITIALIZATION_SEED, "input_seed": INPUT_SEED,
        "sampling_seed": SAMPLING_SEED, "recipe_sha256": hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data": "Independent Gaussian histories only; no waveform observations, latents, or truth",
        "weights": "Identical fresh random initialization on each device; no checkpoints loaded",
        "warmup": "One full 50-step sample batch of256 for each device and sequence length",
        "measurement": "One synchronized timed50-step sample batch of256 per case, inclusive of sampler transfers/checks",
        "concurrency_caveat": "Three separate CPU training workers were active during this pilot; timings are not isolated device comparisons. MPS would use one worker.",
        "exclusions": "No history-densifier cost, full1024trajectory end-to-end run, convergence, quality, or cross-device numerical-equivalence claim",
        "maximum_total_subprocess_seconds": 190, "devices": {},
    }
    write_json(OUTPUT / "timings.json", result)
    start = perf_counter()
    for device in ("cpu", "mps"):
        remaining = 190 - (perf_counter() - start)
        if remaining <= 0:
            result["devices"][device] = {"status": "skipped_time_budget"}
            continue
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__)), "--device", device],
                cwd=ROOT, capture_output=True, text=True, timeout=min(95, remaining),
            )
            (OUTPUT / f"{device}.log").write_text(completed.stdout + completed.stderr)
            path = OUTPUT / f"{device}.json"
            entry = json.loads(path.read_text()) if path.exists() else {}
            entry["process_returncode"] = completed.returncode
            if completed.returncode != 0:
                entry["status"] = "failed"
            result["devices"][device] = entry
        except subprocess.TimeoutExpired as error:
            path = OUTPUT / f"{device}.json"
            entry = json.loads(path.read_text()) if path.exists() else {}
            entry["status"] = "timed_out"
            result["devices"][device] = entry
            text = (error.stdout or b"") + (error.stderr or b"")
            (OUTPUT / f"{device}.log").write_bytes(text)
        write_json(OUTPUT / "timings.json", result)
        print(json.dumps({"device": device, "result": result["devices"][device]}), flush=True)
    result["elapsed_wall_seconds"] = perf_counter() - start
    result["finished_utc"] = datetime.now(UTC).isoformat()
    if all(result["devices"].get(device, {}).get("status") == "complete" for device in ("cpu", "mps")):
        cpu, mps = result["devices"]["cpu"], result["devices"]["mps"]
        result["identical_initial_weight_hashes"] = cpu["initial_weights_sha256"] == mps["initial_weights_sha256"]
        result["cpu_seconds_divided_by_mps_seconds"] = {
            str(left["sequence_length"]): left["timed_seconds"] / right["timed_seconds"]
            for left, right in zip(cpu["cases"], mps["cases"], strict=True)
        }
    write_json(OUTPUT / "timings.json", result)


if __name__ == "__main__":
    main()
