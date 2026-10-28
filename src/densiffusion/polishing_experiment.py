"""Paired noisy-target forecasts with raw, blind-spot, or median histories."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.generators.noisy_periodic import generate_noisy_periodic
from densiffusion.history_polishing import (
    load_mediator,
    median_history,
    polish_history,
    train_mediator,
)
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel
from densiffusion.samplers import DDPMForecastSampler
from densiffusion.training import TrainingConfig, train_forecaster
from densiffusion.waveform_experiment import model_hash, write_json
from densiffusion.waveform_imputation import _snapshot, sha256

METHODS = ("raw", "learned", "median")
SEED_NAMES = (
    "train_data",
    "validation_data",
    "test_data",
    "mediator_training",
    "forecaster_initialization",
    "forecaster_training",
    "validation_sampling",
    "test_sampling",
)


@dataclass(frozen=True)
class PolishingConfig:
    seeds: tuple[int, ...] = (28101, 28102, 28103)
    namespace: int = 2026102705
    n_train: int = 256
    n_validation: int = 256
    n_test: int = 2048
    context_length: int = 32
    horizon: int = 8
    dt: float = 1.0
    epochs: int = 1200
    batch_size: int = 64
    validation_interval: int = 10
    validation_epochs: tuple[int, ...] = (10, 50, 100, 200, 400, 800, 1200)
    samples: int = 64
    sample_batch_size: int = 256
    diffusion_steps: int = 50
    device: str = "mps"
    threads: int = 1
    label: str = "scientific"

    def __post_init__(self):
        for name in (
            "namespace",
            "n_train",
            "n_validation",
            "n_test",
            "context_length",
            "horizon",
            "epochs",
            "batch_size",
            "validation_interval",
            "samples",
            "sample_batch_size",
            "diffusion_steps",
            "threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Require distinct seeds")
        if any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("Invalid seed")
        if self.context_length != 32 or self.horizon != 8 or self.dt != 1:
            raise ValueError("Require the original 32+8 integer-time grid")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("Unknown device")
        if (
            not self.validation_epochs
            or any(
                type(e) is not int
                or e < 1
                or e > self.epochs
                or e % self.validation_interval
                for e in self.validation_epochs
            )
            or any(
                a >= b
                for a, b in zip(
                    self.validation_epochs, self.validation_epochs[1:], strict=False
                )
            )
        ):
            raise ValueError("Invalid explicit CRPS schedule")
        if self.label == "smoke":
            if self.namespace == 2026102705:
                raise ValueError("Smoke requires an independent namespace")
        elif self.label != "scientific":
            raise ValueError("Unknown run label")
        else:
            fixed = {
                "n_train": 256,
                "n_validation": 256,
                "n_test": 2048,
                "epochs": 1200,
                "batch_size": 64,
                "validation_interval": 10,
                "samples": 64,
                "diffusion_steps": 50,
                "namespace": 2026102705,
            }
            if len(self.seeds) != 3 or any(
                getattr(self, k) != v for k, v in fixed.items()
            ):
                raise ValueError("Scientific comparison budget changed")
            if tuple(self.validation_epochs) != (10, 50, 100, 200, 400, 800, 1200):
                raise ValueError("Scientific validation opportunities changed")


def read(path):
    return json.loads(Path(path).read_text())


def array_hash(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def seed_plan(seed, namespace):
    return {
        name: int(stream.generate_state(1)[0])
        for name, stream in zip(
            SEED_NAMES,
            np.random.SeedSequence([seed, namespace]).spawn(len(SEED_NAMES)),
            strict=True,
        )
    }


def hashes(root):
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def verify_hashes(root, expected):
    for name, digest in expected.items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _timed(function, *args, device="cpu", **kwargs):
    _sync(device)
    start = perf_counter()
    result = function(*args, **kwargs)
    _sync(device)
    return result, perf_counter() - start


def recovered_recipe(source):
    source = Path(source)
    plan, selection = read(source / "plan.json"), read(source / "selection.json")
    name = selection["selected"]
    recipe = plan["candidates"][name]
    return {
        "source": str(source.resolve()),
        "selected": name,
        **recipe,
        "source_sha256": {
            name: sha256(source / name) for name in ("plan.json", "selection.json")
        },
        "gp_weights_reused": False,
    }


def _new_model(architecture, seed):
    torch.manual_seed(seed)
    return ImputationModel(CSDIDenoiser(**architecture))


def _preprocess(history, mediator):
    if history.ndim != 3 or history.shape[1:] != (32, 1):
        raise ValueError("Preprocessing accepts available 32-point histories only")
    results, seconds = {}, {}
    for method, function in (
        ("raw", lambda x: x.copy()),
        ("learned", lambda x: polish_history(mediator, x)),
        ("median", median_history),
    ):
        results[method], seconds[method] = _timed(function, history)
        if (
            results[method].shape != history.shape
            or not np.isfinite(results[method]).all()
        ):
            raise ValueError("Invalid history representation")
    return results, seconds


def _sample(model, history, manifest, seed):
    config = PolishingConfig(**manifest["experiment"])
    sampling = replace(
        TrainingConfig(**manifest["forecaster_training"]),
        batch_size=config.sample_batch_size,
    )
    # Both arguments are the processed history: there is no raw-history side channel.
    return DDPMForecastSampler(model, sampling).sample(
        history,
        history,
        horizon=config.horizon,
        n_samples=config.samples,
        dt=config.dt,
        rng=np.random.default_rng(seed),
    )


def _save_ensemble(path, samples, raw_history, processed_history, target, ids):
    np.savez_compressed(
        path,
        samples=samples,
        history=raw_history,
        processed_history=processed_history,
        target=target,
        trajectory_ids=ids,
        history_times=np.arange(32, dtype=float),
        future_times=np.arange(32, 40, dtype=float),
    )


def _prepare_mediator(manifest, seed, output):
    config = PolishingConfig(**manifest["experiment"])
    directory = output / f"seed_{seed}"
    directory.mkdir()
    seeds = manifest["seeds"][str(seed)]
    write_json(directory / "seeds.json", seeds)
    train = generate_noisy_periodic(
        config.n_train, rng=np.random.default_rng(seeds["train_data"])
    )
    validation = generate_noisy_periodic(
        config.n_validation, rng=np.random.default_rng(seeds["validation_data"])
    )
    train_ids = np.arange(config.n_train, dtype=np.int64)
    validation_ids = np.arange(config.n_validation, dtype=np.int64) + config.n_train
    np.savez_compressed(
        directory / "train_validation.npz",
        train=train,
        validation=validation,
        train_trajectory_ids=train_ids,
        validation_trajectory_ids=validation_ids,
    )
    # Only history slices cross the mediator API boundary, never future values.
    (mediator, summary), fit_seconds = _timed(
        train_mediator,
        train[:, :32].copy(),
        validation[:, :32].copy(),
        directory / "mediator",
        seeds["mediator_training"],
        device="cpu",
        smoke=config.label == "smoke",
    )
    before = model_hash(mediator)
    train_histories, train_seconds = _preprocess(train[:, :32], mediator)
    validation_histories, validation_seconds = _preprocess(validation[:, :32], mediator)
    if model_hash(mediator) != before:
        raise AssertionError("Preprocessing changed mediator weights")
    arrays = {f"{m}_train_history": train_histories[m] for m in METHODS}
    arrays.update({f"{m}_validation_history": validation_histories[m] for m in METHODS})
    arrays.update(train_future=train[:, 32:], validation_future=validation[:, 32:])
    np.savez_compressed(directory / "processed_train_validation.npz", **arrays)
    write_json(
        directory / "preprocessing_audit.json",
        {
            "history_shape": [config.n_train, 32, 1],
            "validation_history_shape": [config.n_validation, 32, 1],
            "input_scope": "available history only; no clean or future features",
            "normalization": "none",
            "densification": False,
            "mediator_sha256": before,
            "raw_train_history_sha256": array_hash(train[:, :32]),
            "raw_validation_history_sha256": array_hash(validation[:, :32]),
            "train_future_sha256": array_hash(train[:, 32:]),
            "validation_future_sha256": array_hash(validation[:, 32:]),
            "processed_sha256": {k: array_hash(v) for k, v in arrays.items()},
            "mediator_selection": summary,
        },
    )
    write_json(
        directory / "timings.json",
        {
            "mediator_training_seconds": fit_seconds,
            "preprocessing_seconds": {
                m: train_seconds[m] + validation_seconds[m] for m in METHODS
            },
            "forecaster_training_seconds": {},
            "validation_sampling_seconds": {},
            "test_preprocessing_seconds": {},
            "test_inference_seconds": {},
        },
    )


def _train_forecasters(manifest, seed, output):
    config = PolishingConfig(**manifest["experiment"])
    directory = output / f"seed_{seed}"
    seeds = manifest["seeds"][str(seed)]
    training = TrainingConfig(**manifest["forecaster_training"])
    with np.load(directory / "train_validation.npz", allow_pickle=False) as f:
        original = {k: f[k] for k in f.files}
    with np.load(directory / "processed_train_validation.npz", allow_pickle=False) as f:
        processed = {k: f[k] for k in f.files}
    timings = read(directory / "timings.json")
    audit = {"methods": {}, "future_targets_unchanged": True, "factor": 1}
    for method in METHODS:
        verify_hashes(output, read(output / "MEDIATORS_FROZEN.json")["files"])
        model = _new_model(manifest["architecture"], seeds["forecaster_initialization"])
        initial_hash = model_hash(model)
        count = sum(p.numel() for p in model.parameters())
        model.to(config.device)
        train = (processed[f"{method}_train_history"], original["train"][:, 32:].copy())
        validation = (
            processed[f"{method}_validation_history"],
            original["validation"][:, 32:].copy(),
        )
        if not np.array_equal(
            train[1], processed["train_future"]
        ) or not np.array_equal(validation[1], processed["validation_future"]):
            raise AssertionError("Future targets changed")
        rows, backward_calls = [], [0]
        handle = next(model.parameters()).register_post_accumulate_grad_hook(
            lambda parameter, counter=backward_calls: counter.__setitem__(
                0, counter[0] + 1
            )
        )

        def score(current, epoch, *, method=method, validation=validation, rows=rows):
            samples, seconds = _timed(
                _sample,
                current,
                validation[0],
                manifest,
                seeds["validation_sampling"],
                device=config.device,
            )
            metrics = evaluate_forecast(samples, validation[1])
            filename = f"{method}_validation_epoch_{epoch:04d}.npz"
            _save_ensemble(
                directory / filename,
                samples,
                original["validation"][:, :32],
                validation[0],
                validation[1],
                original["validation_trajectory_ids"],
            )
            rows.append(
                {
                    "epoch": epoch,
                    "crps": metrics["crps"],
                    "scores": metrics,
                    "sampling_seconds": seconds,
                    "samples": filename,
                }
            )
            write_json(directory / f"{method}_validation.json", rows)
            return metrics["crps"]

        start = perf_counter()
        started_utc = datetime.now(UTC).isoformat()

        def record(row, *, method=method, start=start):
            with (directory / f"{method}_training.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({**row, "elapsed_seconds": perf_counter() - start})
                    + "\n"
                )

        try:
            _, seconds = _timed(
                train_forecaster,
                model,
                None,
                train,
                validation,
                config=training,
                factor=1,
                dt=1.0,
                seed=seeds["forecaster_training"],
                checkpoint=directory / f"{method}.pt",
                n_augmentations=1,
                validation_score=score,
                validation_score_epochs=tuple(config.validation_epochs),
                epoch_callback=record,
                device=config.device,
            )
        finally:
            handle.remove()
        selected = min(rows, key=lambda x: x["crps"])
        checkpoint = torch.load(
            directory / f"{method}.pt", map_location="cpu", weights_only=True
        )
        expected_updates = config.epochs * int(
            np.ceil(config.n_train / config.batch_size)
        )
        if (
            backward_calls[0] != expected_updates
            or checkpoint["epoch"] != selected["epoch"]
        ):
            raise AssertionError("Optimization budget or selection differs")
        timings["forecaster_training_seconds"][method] = seconds
        timings["validation_sampling_seconds"][method] = sum(
            row["sampling_seconds"] for row in rows
        )
        audit["methods"][method] = {
            "started_utc": started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "initialization_sha256": initial_hash,
            "parameter_count": count,
            "measured_optimizer_updates": backward_calls[0],
            "epochs_completed": config.epochs,
            "selected_epoch": selected["epoch"],
            "selected_validation_crps": selected["crps"],
            "validation_candidate_epochs": [row["epoch"] for row in rows],
            "training_seed": seeds["forecaster_training"],
            "train_history_sha256": array_hash(train[0]),
            "train_future_sha256": array_hash(train[1]),
            "combined_train_sha256": array_hash(np.concatenate(train, axis=1)),
            "validation_history_sha256": array_hash(validation[0]),
            "validation_future_sha256": array_hash(validation[1]),
            "combined_validation_sha256": array_hash(
                np.concatenate(validation, axis=1)
            ),
            "checkpoint_sha256": sha256(directory / f"{method}.pt"),
        }
        write_json(directory / "training_audit.json", audit)
        write_json(directory / "timings.json", timings)
        del model
        if config.device == "mps":
            torch.mps.empty_cache()
    if len({m["initialization_sha256"] for m in audit["methods"].values()}) != 1:
        raise AssertionError("Forecasters did not receive paired initializations")


def train_run(output, config=None, recipe_source=Path("outputs/csdi-gp-better")):
    config = config or PolishingConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    recipe = recovered_recipe(recipe_source)
    training = replace(
        TrainingConfig(**recipe["training"]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_interval=config.validation_interval,
        diffusion_steps=config.diffusion_steps,
        device=config.device,
    )
    manifest = {
        "experiment": asdict(config),
        "architecture": recipe["architecture"],
        "recovered_gp_recipe": recipe,
        "forecaster_training": asdict(training),
        "seeds": {
            str(seed): seed_plan(seed, config.namespace) for seed in config.seeds
        },
        "methods": METHODS,
        "normalization": "none",
        "future_targets": "unchanged noisy observations",
        "history_only_preprocessing": True,
        "no_clean_signal_exposed": True,
        "no_densification": True,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "config.json", manifest)
    _snapshot(output)
    start = perf_counter()
    for seed in config.seeds:
        _prepare_mediator(manifest, seed, output)
    mediator_seconds = perf_counter() - start
    mediator_files = {
        str(p.relative_to(output)): sha256(p)
        for seed in config.seeds
        for p in (output / f"seed_{seed}" / "mediator").rglob("*")
        if p.is_file()
    }
    write_json(
        output / "MEDIATORS_FROZEN.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "files": mediator_files,
            "configuration_sha256": sha256(output / "config.json"),
            "before_any_forecaster_training": True,
            "wall_seconds": mediator_seconds,
        },
    )
    forecaster_start = perf_counter()
    for seed in config.seeds:
        _train_forecasters(manifest, seed, output)
    verify_hashes(output, mediator_files)
    write_json(
        output / "TRAINED.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "mediator_wall_seconds": mediator_seconds,
            "forecaster_wall_seconds": perf_counter() - forecaster_start,
            "wall_seconds": perf_counter() - start,
        },
    )
    write_json(
        output / "FROZEN.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "files": hashes(output),
            "test_data_generated": False,
            "all_methods_and_seeds_selected": True,
        },
    )
    return manifest


def verify_frozen(output):
    output = Path(output)
    frozen = read(output / "FROZEN.json")
    verify_hashes(output, frozen["files"])
    return frozen


def evaluate_run(output):
    output = Path(output)
    verify_frozen(output)
    if (output / "EVALUATION_STARTED.json").exists() or any(
        output.glob("seed_*/test.npz")
    ):
        raise FileExistsError(
            "Test evaluation already started; existing evidence is preserved"
        )
    manifest = read(output / "config.json")
    config = PolishingConfig(**manifest["experiment"])
    torch.set_num_threads(config.threads)
    write_json(
        output / "EVALUATION_STARTED.json",
        {"started_utc": datetime.now(UTC).isoformat()},
    )
    start = perf_counter()
    for seed in config.seeds:
        directory = output / f"seed_{seed}"
        seeds = manifest["seeds"][str(seed)]
        observations = generate_noisy_periodic(
            config.n_test, rng=np.random.default_rng(seeds["test_data"])
        )
        history, target = observations[:, :32], observations[:, 32:]
        ids = (
            np.arange(config.n_test, dtype=np.int64)
            + config.n_train
            + config.n_validation
        )
        np.savez_compressed(
            directory / "test.npz", history=history, target=target, trajectory_ids=ids
        )
        mediator = load_mediator(directory / "mediator/selected.pt", device="cpu")
        processed, preprocessing_seconds = _preprocess(history, mediator)
        timings = read(directory / "timings.json")
        # Keep frozen training timings immutable; save evaluation costs separately.
        evaluation_timings = {
            "test_preprocessing_seconds": preprocessing_seconds,
            "test_inference_seconds": {},
        }
        scores, inference_audit = {}, {}
        for method in METHODS:
            checkpoint = torch.load(
                directory / f"{method}.pt", map_location="cpu", weights_only=True
            )
            model = _new_model(manifest["architecture"], 0)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(config.device).eval()
            samples, seconds = _timed(
                _sample,
                model,
                processed[method],
                manifest,
                seeds["test_sampling"],
                device=config.device,
            )
            _save_ensemble(
                directory / f"{method}_samples.npz",
                samples,
                history,
                processed[method],
                target,
                ids,
            )
            scores[method] = evaluate_forecast(samples, target)
            evaluation_timings["test_inference_seconds"][method] = seconds
            inference_audit[method] = {
                "conditioning_sha256": array_hash(processed[method]),
                "observed_argument_sha256": array_hash(processed[method]),
                "dense_history_argument_sha256": array_hash(processed[method]),
                "conditioning_shape": list(processed[method].shape),
                "raw_side_channel": False,
                "future_conditioning": False,
                "samples": config.samples,
                "horizon": 8,
                "dt": 1,
                "sampling_seed": seeds["test_sampling"],
            }
            write_json(directory / "scores.json", scores)
            write_json(directory / "evaluation_timings.json", evaluation_timings)
            write_json(directory / "inference_audit.json", inference_audit)
            del model
            if config.device == "mps":
                torch.mps.empty_cache()
        write_json(
            directory / "combined_timings.json", {**timings, **evaluation_timings}
        )
    verify_frozen(output)
    write_json(
        output / "EVALUATED.json",
        {
            "finished_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": perf_counter() - start,
            "frozen_artifacts_verified": True,
        },
    )
