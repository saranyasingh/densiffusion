"""Bounded training-throughput pilot using Gaussian arrays and disposable weights."""

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
from densiffusion.objectives import noise_mse
from densiffusion.training import TrainingConfig, _loss

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
RECIPE = ROOT / "outputs/csdi-gp-better/plan.json"
INITIALIZATION_SEED = 860411
INPUT_SEED = 860412
TRAINING_SEED = 860413
WARMUP_UPDATES = 3
TIMED_UPDATES = 16
BATCH_SIZE = 64
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
    training = replace(TrainingConfig(**recipe["training"]), device=device)
    report = {"device": device, "torch_threads": 1, "worker_count": 1,
              "architecture": recipe["architecture"], "training_config": asdict(training),
              "cases": [], "status": "running"}
    write_json(OUTPUT / f"{device}.json", report)
    if device == "mps" and not torch.backends.mps.is_available():
        report["status"] = "unavailable"
        write_json(OUTPUT / f"{device}.json", report)
        return
    for length, context, horizon, dt in CASES:
        torch.manual_seed(INITIALIZATION_SEED)
        model = ImputationModel(CSDIDenoiser(**recipe["architecture"])).train()
        initial_hash = state_hash(model)
        model.to(device)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=training.learning_rate,
                                     weight_decay=training.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[max(1, int(f * training.epochs))
                                   for f in training.lr_decay_fractions],
            gamma=training.lr_decay_factor,
        )
        alpha_bar = (1 - training.betas(dtype=torch.float32, device=device)).cumprod(0)
        data = np.random.default_rng(INPUT_SEED + length).normal(
            size=(BATCH_SIZE * TIMED_UPDATES, length, 1)
        )
        masks = np.broadcast_to((np.arange(length) >= context)[None, :, None],
                                data.shape).copy()
        noise_seed = int(np.random.SeedSequence(TRAINING_SEED).generate_state(3, dtype=np.uint64)[1])
        noise = torch.Generator().manual_seed(noise_seed)
        torch.manual_seed(TRAINING_SEED)
        updates = 0

        def update(index):
            nonlocal updates
            rows = slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
            target = torch.as_tensor(data[rows], dtype=torch.float32, device=device)
            mask = torch.as_tensor(masks[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, target, mask, alpha_bar, noise, dt, noise_mse)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite disposable timing loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, training.max_grad_norm,
                                          error_if_nonfinite=True)
            optimizer.step()
            # Match the real per-minibatch logging synchronization; do not save
            # the value or interpret random-array losses as model quality.
            loss.item()
            updates += 1

        synchronize(device)
        warmup_start = perf_counter()
        for index in range(WARMUP_UPDATES):
            update(index)
        synchronize(device)
        warmup_seconds = perf_counter() - warmup_start
        learning_rate_before = optimizer.param_groups[0]["lr"]
        synchronize(device)
        start = perf_counter()
        for index in range(TIMED_UPDATES):
            update(index)
        # One epoch contains sixteen training minibatches, as in the target task.
        scheduler.step()
        synchronize(device)
        seconds = perf_counter() - start
        final_hash = state_hash(model)
        row = {
            "sequence_length": length, "context_length": context,
            "hidden_suffix_length": horizon, "dt": dt, "batch_size": BATCH_SIZE,
            "diffusion_steps": training.diffusion_steps,
            "warmup_updates": WARMUP_UPDATES, "warmup_seconds": warmup_seconds,
            "timed_updates": TIMED_UPDATES, "total_updates_executed": updates,
            "timed_seconds": seconds, "updates_per_second": TIMED_UPDATES / seconds,
            "seconds_per_update": seconds / TIMED_UPDATES,
            "timed_examples": TIMED_UPDATES * BATCH_SIZE,
            "examples_per_second": TIMED_UPDATES * BATCH_SIZE / seconds,
            "learning_rate_before": learning_rate_before,
            "learning_rate_after": optimizer.param_groups[0]["lr"],
            "scheduler_steps": 1, "dropout_enabled": model.training,
            "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
            "initial_weights_sha256": initial_hash,
            "disposable_weights_changed": initial_hash != final_hash,
            "all_training_steps_finite": True,
            "parameter_count": sum(p.numel() for p in parameters),
        }
        report["cases"].append(row)
        write_json(OUTPUT / f"{device}.json", report)
        print(json.dumps({"device": device, **row}), flush=True)
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
        "purpose": "Timing only; disposable updates on random Gaussian arrays, no quality assessment or scientific model selection",
        "started_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "numpy": str(np.__version__),
        "torch": str(torch.__version__), "mps_available": torch.backends.mps.is_available(),
        "initialization_seed": INITIALIZATION_SEED, "input_seed": INPUT_SEED,
        "training_seed": TRAINING_SEED,
        "recipe_sha256": hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_code_sha256": hashlib.sha256((ROOT / "src/densiffusion/training.py").read_bytes()).hexdigest(),
        "data": "Independent Gaussian full arrays, prefix observed and suffix hidden; no waveform data, latents, or truth",
        "weights": "Identical fresh initialization per device and shape; disposable weights are neither saved nor transferred to experiments",
        "measurement": "3 warmup updates, then16 batch64 updates including CPU input transfer, repository _loss, random diffusion step/noise, dropout, backward, gradient clipping, Adam, loss.item synchronization, and one epoch scheduler step",
        "concurrency_caveat": "Three separate CPU training workers remained active; these are nonisolated throughput measurements. MPS would use one worker.",
        "exclusions": "No validation or sampling cost, convergence/quality claim, checkpoints, or cross-device numerical-equivalence test",
        "maximum_subprocess_seconds": 190, "devices": {},
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
            (OUTPUT / f"{device}.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
        write_json(OUTPUT / "timings.json", result)
        print(json.dumps({"device": device, "result": result["devices"][device]}), flush=True)
    result["elapsed_wall_seconds"] = perf_counter() - start
    result["finished_utc"] = datetime.now(UTC).isoformat()
    if all(result["devices"].get(device, {}).get("status") == "complete" for device in ("cpu", "mps")):
        cpu, mps = result["devices"]["cpu"], result["devices"]["mps"]
        result["identical_initial_weight_and_input_hashes"] = all(
            left["initial_weights_sha256"] == right["initial_weights_sha256"]
            and left["input_sha256"] == right["input_sha256"]
            and left["mask_sha256"] == right["mask_sha256"]
            for left, right in zip(cpu["cases"], mps["cases"], strict=True)
        )
        result["cpu_seconds_divided_by_mps_seconds"] = {
            str(left["sequence_length"]): left["timed_seconds"] / right["timed_seconds"]
            for left, right in zip(cpu["cases"], mps["cases"], strict=True)
        }
    write_json(OUTPUT / "timings.json", result)


if __name__ == "__main__":
    main()
