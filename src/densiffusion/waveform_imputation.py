"""Validation-gated, larger CSDI waveform imputation experiments.

The original train/validation arrays are copied, never regenerated or altered.
No forecast test data are accessed. Fine-grid simulator values exist only in
the separate post-freeze transfer diagnostic, which cannot revise weights.
"""

import hashlib
import json
import shutil
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier
from densiffusion.waveform_experiment import model_hash, seed_plan, timed, write_json


@dataclass(frozen=True)
class ImputationConfig:
    seeds: tuple[int, ...] = (27101, 27102, 27103)
    epochs: int = 200
    score_interval: int = 50
    samples: int = 64
    sample_batch_size: int = 256
    device: str = "cpu"
    threads: int = 1
    workers: int = 3
    transfer_examples: int = 256
    diagnostic_namespace: int = 2026102701
    label: str = "gp_recipe"

    def __post_init__(self):
        for name in (
            "epochs",
            "score_interval",
            "samples",
            "sample_batch_size",
            "threads",
            "workers",
            "transfer_examples",
            "diagnostic_namespace",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Require distinct seeds")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("Unknown device")
        if self.label != "smoke" and self.samples < 64:
            raise ValueError("Scientific imputation runs require at least 64 draws")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover_recipe(recipe_dir):
    """Read the selected GP recipe, not any GP weights or test samples."""
    recipe_dir = Path(recipe_dir)
    plan = json.loads((recipe_dir / "plan.json").read_text())
    selection = json.loads((recipe_dir / "selection.json").read_text())
    selected = selection["selected"]
    recipe = plan["candidates"][selected]
    return {
        "source": str(recipe_dir.resolve()),
        "selected": selected,
        "architecture": recipe["architecture"],
        "training": recipe["training"],
        "source_sha256": {
            name: sha256(recipe_dir / name) for name in ("plan.json", "selection.json")
        },
        "gp_weights_reused": False,
        "source_selection": plan["selection"],
    }


def hidden_scores(samples, target):
    return evaluate_forecast(samples[:, :, 1::2], target[:, 1::2])


def linear_samples(target, draws):
    values = target.copy()
    values[:, 1::2] = (values[:, :-1:2] + values[:, 2::2]) / 2
    return np.repeat(values[:, None], draws, axis=1)


def sample_imputations(model, training, values, config, seed, *, target_dt=0.1):
    sampler = DDPMSampler(
        model,
        replace(training, batch_size=config.sample_batch_size, device=config.device),
    )
    samples = sampler.sample(
        values[:, ::2].copy(),
        factor=2,
        n_samples=config.samples,
        dt=2 * target_dt,
        rng=np.random.default_rng(seed),
    )
    expected = np.broadcast_to(values[:, None, ::2], samples[:, :, ::2].shape)
    if not np.array_equal(samples[:, :, ::2], expected):
        raise AssertionError("Original observations changed")
    if not np.isfinite(samples).all():
        raise FloatingPointError("Nonfinite imputation")
    return samples


def save_samples(path, values, samples, config, *, target_dt=0.1):
    mask = np.broadcast_to(
        (np.arange(values.shape[1]) % 2 == 1)[None, :, None], values.shape
    ).copy()
    np.savez_compressed(
        path,
        target=values,
        mask=mask,
        learned=samples,
        linear=linear_samples(values, config.samples),
        times=np.arange(values.shape[1]) * target_dt,
        trajectory_ids=np.arange(len(values)),
    )


def _training_config(manifest):
    return TrainingConfig(**manifest["training"])


def _new_model(architecture, seed):
    torch.manual_seed(seed)
    return ImputationModel(CSDIDenoiser(**architecture))


def _train_seed(manifest, seed, directory):
    directory = Path(directory)
    directory.mkdir()
    config = ImputationConfig(**manifest["experiment"])
    torch.set_num_threads(config.threads)
    source = Path(manifest["data_source"]) / f"seed_{seed}" / "train_validation.npz"
    if sha256(source) != manifest["data_sha256"][str(seed)]:
        raise ValueError("Source training/validation data changed")
    shutil.copyfile(source, directory / "train_validation.npz")
    with np.load(source, allow_pickle=False) as arrays:
        train, validation = arrays["train"], arrays["validation"]
    if config.label == "smoke":
        train, validation = train[:16], validation[:8]
    elif train.shape != (1024, 33, 1) or validation.shape != (256, 33, 1):
        raise ValueError("Expected unchanged 1024/256 original 33-point trajectories")
    seeds = seed_plan(seed)
    write_json(directory / "seeds.json", seeds)
    training = _training_config(manifest)
    model = _new_model(manifest["architecture"], seeds["densifier_initialization"])
    initial_hash = model_hash(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    validation_rows = []
    best = float("inf")
    start = perf_counter()
    linear = linear_samples(validation, config.samples)
    linear_metrics = hidden_scores(linear, validation)

    def validation_interpolation_crps(current_model, epoch):
        nonlocal best
        samples, seconds = timed(
            sample_imputations,
            current_model,
            training,
            validation,
            config,
            seeds["validation_sampling"],
            device=config.device,
        )
        scores = hidden_scores(samples, validation)
        filename = f"validation_epoch_{epoch:04d}.npz"
        save_samples(directory / filename, validation, samples, config)
        row = {
            "epoch": epoch,
            "scores": scores,
            "linear": linear_metrics,
            "sampling_seconds": seconds,
            "samples": filename,
        }
        validation_rows.append(row)
        write_json(directory / "validation.json", validation_rows)
        if scores["crps"] < best:
            best = scores["crps"]
            shutil.copyfile(directory / filename, directory / "validation_samples.npz")
        return scores["crps"]

    def record_epoch(record):
        with (directory / "training.jsonl").open("a") as stream:
            stream.write(
                json.dumps({**record, "elapsed_seconds": perf_counter() - start}) + "\n"
            )

    train_densifier(
        model,
        train,
        validation,
        config=training,
        dt=0.1,
        seed=seeds["densifier_training"],
        checkpoint=directory / "densifier.pt",
        validation_score=validation_interpolation_crps,
        validation_score_interval=config.score_interval,
        epoch_callback=record_epoch,
    )
    total_seconds = perf_counter() - start
    state = torch.load(
        directory / "densifier.pt", map_location="cpu", weights_only=True
    )
    selected = min(validation_rows, key=lambda row: row["scores"]["crps"])
    if state["epoch"] != selected["epoch"]:
        raise AssertionError("Checkpoint selection differs from sampled CRPS argmin")
    write_json(
        directory / "summary.json",
        {
            "seed": seed,
            "parameter_count": parameter_count,
            "initialization_sha256": initial_hash,
            "selected_epoch": state["epoch"],
            "validation_noise_mse": state["validation_loss"],
            "selected_validation": selected,
            "optimizer_updates": training.epochs
            * int(np.ceil(len(train) / training.batch_size)),
            "training_seconds": total_seconds,
            "validation_sampling_seconds": sum(
                row["sampling_seconds"] for row in validation_rows
            ),
            "observations_preserved": True,
            "gp_weights_reused": False,
        },
    )


def _snapshot(output):
    root = Path(__file__).resolve().parents[2]
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for folder in ("src", "scripts", "tests", "configs", "docs"):
            for path in (root / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(path, arcname=path.relative_to(root))
        for name in ("pyproject.toml", "uv.lock", "README.md"):
            archive.add(root / name, arcname=name)
    for args, filename in (
        (["diff"], "git_diff.patch"),
        (["rev-parse", "HEAD"], "git_head.txt"),
    ):
        (output / filename).write_bytes(
            subprocess.check_output(["git", *args], cwd=root)
        )


def _run_workers(function, manifest, output):
    config = ImputationConfig(**manifest["experiment"])
    jobs = [(manifest, seed, output / f"seed_{seed}") for seed in config.seeds]
    if config.workers > 1 and len(jobs) > 1 and config.device == "cpu":
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(jobs)), mp_context=get_context("spawn")
        ) as executor:
            futures = [executor.submit(function, *args) for args in jobs]
            return [future.result() for future in futures]
    return [function(*args) for args in jobs]


def train_run(output, data_source, recipe_source, config=None):
    config = config or ImputationConfig()
    output, data_source = Path(output), Path(data_source)
    output.mkdir(parents=True, exist_ok=False)
    recipe = recover_recipe(recipe_source)
    training = replace(
        TrainingConfig(**recipe["training"]),
        epochs=config.epochs,
        device=config.device,
    )
    if config.label == "smoke":
        training = replace(training, validation_interval=1, diffusion_steps=2)
    manifest = {
        "experiment": asdict(config),
        "architecture": recipe["architecture"],
        "training": asdict(training),
        "recovered_gp_recipe": recipe,
        "data_source": str(data_source.resolve()),
        "data_sha256": {
            str(seed): sha256(data_source / f"seed_{seed}" / "train_validation.npz")
            for seed in config.seeds
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "selection": (
            "all original-grid validation trajectories; hidden-point empirical CRPS"
        ),
        "normalization": "none",
        "fine_truth_used": False,
        "gate": {
            "seeds_required": 3,
            "minimum_relative_mean_crps_reduction": 0.05,
            "every_seed_lower_crps": True,
            "every_seed_paired_95pct_upper_below_zero": True,
        },
        "transfer_gate": {
            "maximum_crps_ratio_to_linear": 1.25,
            "maximum_mse_ratio_to_linear": 1.5,
            "required_tasks": ["history", "anchored_future"],
            "required_each_seed": True,
            "no_tuning_after_fine_truth": True,
        },
    }
    write_json(output / "config.json", manifest)
    _snapshot(output)
    start = perf_counter()
    _run_workers(_train_seed, manifest, output)
    write_json(
        output / "TRAINED.json",
        {
            "finished_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": perf_counter() - start,
        },
    )
    return manifest


def freeze_run(output):
    """Freeze all selected densifiers after validation analysis, before fine truth."""
    output = Path(output)
    if (output / "DENSIFIERS_FROZEN.json").exists():
        raise FileExistsError("Densifiers already frozen")
    if not (output / "TRAINED.json").is_file():
        raise ValueError("All seeds must finish before freezing")
    analysis = json.loads((output / "imputation_analysis.json").read_text())
    manifest = json.loads((output / "config.json").read_text())
    frozen = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(output / "config.json"),
        "analysis_sha256": sha256(output / "imputation_analysis.json"),
        "checkpoint_sha256": {
            f"seed_{seed}/densifier.pt": sha256(output / f"seed_{seed}/densifier.pt")
            for seed in manifest["experiment"]["seeds"]
        },
        "validation_gate": analysis["validation_gate"],
        "fine_grid_truth_generated": False,
    }
    write_json(output / "DENSIFIERS_FROZEN.json", frozen)
    return frozen


def verify_frozen(output):
    output = Path(output)
    frozen = json.loads((output / "DENSIFIERS_FROZEN.json").read_text())
    if sha256(output / "config.json") != frozen["config_sha256"]:
        raise ValueError("Frozen config changed")
    if sha256(output / "imputation_analysis.json") != frozen["analysis_sha256"]:
        raise ValueError("Frozen validation analysis changed")
    for name, expected in frozen["checkpoint_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError("Frozen checkpoint changed")
    return frozen


def _transfer_seed(manifest, seed, directory):
    from densiffusion.generators.waveform import generate_waveforms

    config = ImputationConfig(**manifest["experiment"])
    torch.set_num_threads(config.threads)
    streams = np.random.SeedSequence([seed, config.diagnostic_namespace]).spawn(3)
    data_seed, history_seed, future_seed = [
        int(s.generate_state(1)[0]) for s in streams
    ]
    # New paths and new observation noise; no earlier forecasting test data.
    fine = generate_waveforms(
        config.transfer_examples, 65, 0.05, rng=np.random.default_rng(data_seed)
    )
    model = _new_model(manifest["architecture"], seed)
    state = torch.load(
        directory / "densifier.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(config.device).requires_grad_(False).eval()
    training = TrainingConfig(**state["config"])
    result = {"data_seed": data_seed, "tasks": {}}
    for name, target, sampling_seed in (
        ("history", fine[:, :49], history_seed),
        ("anchored_future", fine[:, 48:], future_seed),
    ):
        samples, seconds = timed(
            sample_imputations,
            model,
            training,
            target,
            config,
            sampling_seed,
            target_dt=0.05,
            device=config.device,
        )
        filename = (
            "transfer_samples.npz"
            if name == "history"
            else "transfer_future_samples.npz"
        )
        save_samples(directory / filename, target, samples, config, target_dt=0.05)
        learned = hidden_scores(samples, target)
        linear = hidden_scores(linear_samples(target, config.samples), target)
        result["tasks"][name] = {
            "learned": learned,
            "linear": linear,
            "sampling_seconds": seconds,
            "sampling_seed": sampling_seed,
            "observations_preserved": True,
            "crps_ratio": learned["crps"] / linear["crps"],
            "mse_ratio": learned["mse"] / linear["mse"],
        }
    write_json(directory / "transfer_scores.json", result)
    return result


def transfer_run(output):
    output = Path(output)
    frozen = verify_frozen(output)
    if not frozen["validation_gate"]["passed"]:
        raise ValueError("Transfer requires the predeclared validation gate to pass")
    if any(output.glob("seed_*/transfer*.npz")):
        raise FileExistsError("Transfer diagnostics already started")
    manifest = json.loads((output / "config.json").read_text())
    start = perf_counter()
    results = _run_workers(_transfer_seed, manifest, output)
    gate = manifest["transfer_gate"]
    passed = all(
        task["crps_ratio"] <= gate["maximum_crps_ratio_to_linear"]
        and task["mse_ratio"] <= gate["maximum_mse_ratio_to_linear"]
        for result in results
        for task in result["tasks"].values()
    )
    verify_frozen(output)
    report = {
        "passed": passed,
        "gate": gate,
        "diagnostic_only": True,
        "no_tuning_after_fine_truth": True,
        "seeds": dict(
            zip(map(str, manifest["experiment"]["seeds"]), results, strict=True)
        ),
        "wall_seconds": perf_counter() - start,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "TRANSFER.json", report)
    return report
