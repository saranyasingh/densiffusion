"""MPS sampling batch timing; disposable random weights and Gaussian histories."""

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

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
RECIPE = ROOT / "outputs/csdi-gp-better/plan.json"
INITIALIZATION_SEED, INPUT_SEED, SAMPLING_SEED = 860421, 860422, 860423
BATCHES = (256, 1024, 2048)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def memory():
    return {"current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory()}


def worker(batch):
    torch.set_num_threads(1)
    torch.mps.set_per_process_memory_fraction(0.1)
    recipe = json.loads(RECIPE.read_text())["candidates"]["reference_size"]
    config = replace(TrainingConfig(**recipe["training"]), device="mps", batch_size=batch)
    torch.manual_seed(INITIALIZATION_SEED)
    model = ImputationModel(CSDIDenoiser(**recipe["architecture"])).eval()
    initial_hash = state_hash(model)
    model.to("mps").requires_grad_(False)
    sampler = DDPMForecastSampler(model, config)
    history = np.random.default_rng(INPUT_SEED).normal(size=(32, 49, 1))[:batch // 64]
    observed = history[:, ::2].copy()
    result = {
        "status": "running", "device": "mps", "worker_count": 1,
        "torch_threads": 1, "sample_batch_size": batch,
        "training_batch_size_unchanged": recipe["training"]["batch_size"],
        "draws_per_history": 64, "n_histories": len(history),
        "sequence_length": 65, "context_length": 49, "hidden_suffix_length": 16,
        "diffusion_steps": 50, "dt": 0.05,
        "architecture": recipe["architecture"], "sampling_config": asdict(config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_weights_sha256": initial_hash,
        "input_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
        "shared_first_four_histories_sha256": hashlib.sha256(history[:4].tobytes()).hexdigest(),
        "memory_fraction_cap": 0.1,
        "recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
        "memory_before_sampling": memory(),
    }
    write_json(OUTPUT / f"batch_{batch}.json", result)
    for phase in ("warmup", "timed"):
        torch.mps.synchronize()
        start = perf_counter()
        samples = sampler.sample(observed, history, horizon=16, n_samples=64, dt=0.05,
                                 rng=np.random.default_rng(SAMPLING_SEED))
        torch.mps.synchronize()
        result[f"{phase}_seconds"] = perf_counter() - start
        result[f"memory_after_{phase}"] = memory()
        if not np.isfinite(samples).all():
            raise FloatingPointError("Nonfinite timing output")
    result.update(
        status="complete", output_shape=list(samples.shape), finite_outputs=True,
        seconds_per_draw=result["timed_seconds"] / batch,
        draws_per_second=batch / result["timed_seconds"],
    )
    write_json(OUTPUT / f"batch_{batch}.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=BATCHES)
    args = parser.parse_args()
    if args.batch:
        worker(args.batch)
        return
    result = {
        "purpose": "Sampling throughput only; no waveform data, quality scoring, training, checkpoint loading, or scientific selection",
        "started_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
        "mps_available": torch.backends.mps.is_available(),
        "recipe_sha256": hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initialization_seed": INITIALIZATION_SEED, "input_seed": INPUT_SEED,
        "sampling_seed": SAMPLING_SEED,
        "design": "Identical fresh random weights, nested Gaussian histories, 64 draws per history. Separate processes isolate allocator state. One full warmup then one timed 50-step batch per size.",
        "memory_scope": "Current tensor and driver-allocation snapshots after synchronization, not measured execution peaks. MPS has no exposed peak-allocation counter in this Torch build. Each child capped to 10% of MPS recommended memory.",
        "concurrency_caveat": "Three CPU training workers remain active; timings are not isolated. Future MPS execution would use one worker.",
        "exclusions": "No history-densifier cost, end-to-end forecast timing, quality or numerical-equivalence claim. Training batch remains 64 and forecast draws remain 64.",
        "maximum_subprocess_seconds": 170, "batches": {},
    }
    write_json(OUTPUT / "timings.json", result)
    if not result["mps_available"]:
        return
    start = perf_counter()
    for batch in BATCHES:
        remaining = 170 - (perf_counter() - start)
        if remaining <= 0:
            result["batches"][str(batch)] = {"status": "skipped_time_budget"}
            continue
        try:
            done = subprocess.run([sys.executable, str(Path(__file__)), "--batch", str(batch)],
                                  cwd=ROOT, capture_output=True, text=True,
                                  timeout=min(60, remaining))
            (OUTPUT / f"batch_{batch}.log").write_text(done.stdout + done.stderr)
            path = OUTPUT / f"batch_{batch}.json"
            entry = json.loads(path.read_text()) if path.exists() else {}
            entry["process_returncode"] = done.returncode
            if done.returncode != 0:
                entry["status"] = "failed"
            result["batches"][str(batch)] = entry
        except subprocess.TimeoutExpired as error:
            path = OUTPUT / f"batch_{batch}.json"
            entry = json.loads(path.read_text()) if path.exists() else {}
            entry["status"] = "timed_out"
            result["batches"][str(batch)] = entry
            (OUTPUT / f"batch_{batch}.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
        write_json(OUTPUT / "timings.json", result)
        print(json.dumps({"batch": batch, "result": result["batches"][str(batch)]}), flush=True)
    complete = [row for row in result["batches"].values() if row.get("status") == "complete"]
    if complete:
        baseline = result["batches"]["256"].get("seconds_per_draw")
        result["shared_initialization_and_history_prefix_hashes"] = (
            len({row["initial_weights_sha256"] for row in complete}) == 1
            and len({row["shared_first_four_histories_sha256"] for row in complete}) == 1
        )
        if baseline:
            result["throughput_relative_to_batch_256"] = {
                str(row["sample_batch_size"]): baseline / row["seconds_per_draw"]
                for row in complete
            }
    result["elapsed_wall_seconds"] = perf_counter() - start
    result["finished_utc"] = datetime.now(UTC).isoformat()
    write_json(OUTPUT / "timings.json", result)


if __name__ == "__main__":
    main()
