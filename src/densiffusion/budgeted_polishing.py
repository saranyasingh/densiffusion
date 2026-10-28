"""Prospective paired polishing forecasts with a charged matrix-work ceiling.

Reuses the established data, mediator, forecaster, sampler, and metric APIs.
Only allocation and orchestration differ: auxiliary search and preprocessing
consume part of the same total work ceiling as forecaster fitting/selection.
"""

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.generators.noisy_periodic import generate_noisy_periodic
from densiffusion.history_polishing import load_mediator, train_mediator
from densiffusion.metrics import evaluate_forecast
from densiffusion.polishing_experiment import (
    METHODS,
    _new_model,
    _preprocess,
    _save_ensemble,
    _timed,
    array_hash,
    hashes,
    read,
    recovered_recipe,
    seed_plan,
    verify_frozen,
    verify_hashes,
)
from densiffusion.resource_allocation import (
    analytical_csdi_forward_work,
    count_csdi_work,
    count_mediator_work,
    fitting_ledger,
)
from densiffusion.samplers import DDPMForecastSampler
from densiffusion.training import TrainingConfig, train_forecaster
from densiffusion.waveform_experiment import model_hash, write_json
from densiffusion.waveform_imputation import _snapshot, sha256


@dataclass(frozen=True)
class BudgetConfig:
    seeds: tuple[int, ...] = (29101, 29102, 29103)
    namespace: int = 2026102801
    n_train: int = 256
    n_validation: int = 256
    n_test: int = 2048
    context_length: int = 32
    horizon: int = 8
    dt: float = 1.0
    base_epochs: int = 1200
    batch_size: int = 64
    validation_interval: int = 10
    validation_epochs_prefix: tuple[int, ...] = (10, 50, 100, 200, 400, 800)
    samples: int = 64
    sample_batch_size: int = 1024
    diffusion_steps: int = 50
    device: str = "mps"
    threads: int = 1
    smoke: bool = False

    def __post_init__(self):
        for key in (
            "namespace",
            "n_train",
            "n_validation",
            "n_test",
            "base_epochs",
            "batch_size",
            "validation_interval",
            "samples",
            "sample_batch_size",
            "diffusion_steps",
            "threads",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Distinct paired seeds required")
        if any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("Invalid seed")
        if self.context_length != 32 or self.horizon != 8 or self.dt != 1:
            raise ValueError("Original 32+8 integer-time geometry required")
        if self.device not in ("cpu", "mps", "cuda") or type(self.smoke) is not bool:
            raise ValueError("Invalid device or smoke flag")
        prefix = tuple(self.validation_epochs_prefix)
        if (
            not prefix
            or prefix != tuple(sorted(set(prefix)))
            or any(
                type(e) is not int
                or e < 1
                or e >= self.base_epochs
                or e % self.validation_interval
                for e in prefix
            )
        ):
            raise ValueError("Invalid common validation schedule")
        if self.n_train % self.batch_size:
            raise ValueError("Full batches required for a simple matched ledger")
        if not self.smoke:
            expected = {
                "namespace": 2026102801,
                "n_train": 256,
                "n_validation": 256,
                "n_test": 2048,
                "base_epochs": 1200,
                "batch_size": 64,
                "validation_interval": 10,
                "samples": 64,
                "diffusion_steps": 50,
            }
            if len(self.seeds) != 3 or any(
                getattr(self, k) != v for k, v in expected.items()
            ):
                raise ValueError("Scientific design differs from fixed protocol")
            if prefix != (10, 50, 100, 200, 400, 800):
                raise ValueError("Scientific validation opportunities changed")
        elif self.namespace == 2026102801:
            raise ValueError("Smoke must use an independent namespace")


def smoke_config():
    return BudgetConfig(
        seeds=(29199,),
        namespace=2026102899,
        n_train=16,
        n_validation=8,
        n_test=8,
        base_epochs=20,
        batch_size=4,
        validation_epochs_prefix=(10,),
        samples=4,
        sample_batch_size=32,
        diffusion_steps=2,
        device="cpu",
        smoke=True,
    )


def auxiliary_work(config, profiles, selected_width):
    if selected_width not in (32, 64):
        raise ValueError("Unexpected selected mediator width")
    epochs = 5 if config.smoke else 200
    validation_epochs = sorted({1, 5, *range(10, epochs + 1, 10), epochs})
    # Two losses for each width; every candidate is fitted to its complete budget.
    training = sum(
        2 * epochs * config.n_train * 32 * profiles[f"mediator_{width}"]["training"]
        for width in (32, 64)
    )
    validation = sum(
        2
        * len(validation_epochs)
        * config.n_validation
        * 32
        * profiles[f"mediator_{width}"]["forward"]
        for width in (32, 64)
    )
    preprocessing = (
        (config.n_train + config.n_validation)
        * 32
        * profiles[f"mediator_{selected_width}"]["forward"]
    )
    return {
        "candidate_training": training,
        "candidate_validation": validation,
        "train_validation_preprocessing": preprocessing,
        "total": training + validation + preprocessing,
    }


def ledger(config, profile, epochs, setup=0):
    if type(epochs) is not int or epochs <= max(config.validation_epochs_prefix):
        raise ValueError("Budget must allow every common early candidate")
    return fitting_ledger(
        epochs=epochs,
        n_train=config.n_train,
        n_validation=config.n_validation,
        forward=profile["forward"],
        training=profile["training"],
        noise_validation_count=math.ceil(epochs / config.validation_interval),
        noise_validation_steps=config.diffusion_steps,
        forecast_validation_count=len(config.validation_epochs_prefix) + 1,
        forecast_samples=config.samples,
        diffusion_steps=config.diffusion_steps,
        setup=setup,
    )


def allocate(config, profiles, selected_width):
    profile = profiles["forecaster"]
    ceiling = ledger(config, profile, config.base_epochs)["total"]
    auxiliary = auxiliary_work(config, profiles, selected_width)
    result = {"ceiling": ceiling, "auxiliary": auxiliary, "methods": {}}
    for method in METHODS:
        setup = auxiliary["total"] if method == "learned" else 0
        affordable = [
            e
            for e in range(
                max(config.validation_epochs_prefix) + 1, config.base_epochs + 1
            )
            if ledger(config, profile, e, setup)["total"] <= ceiling
        ]
        if not affordable:
            raise ValueError("Auxiliary cost leaves too little forecasting budget")
        epochs = max(affordable)
        charged = ledger(config, profile, epochs, setup)
        unused = ceiling - charged["total"]
        if not config.smoke and unused / ceiling > 1e-4:
            raise ValueError("Scientific budget slack exceeds 0.01 percent")
        result["methods"][method] = {
            "epochs": epochs,
            "optimizer_updates": epochs * (config.n_train // config.batch_size),
            "validation_epochs": [*config.validation_epochs_prefix, epochs],
            "noise_validation_count": math.ceil(epochs / config.validation_interval),
            "work": charged,
            "unused_work": unused,
            "unused_fraction": unused / ceiling,
        }
    return result


def _sample(model, history, manifest, training, seed):
    config = BudgetConfig(**manifest["experiment"])
    return DDPMForecastSampler(
        model, replace(training, batch_size=config.sample_batch_size)
    ).sample(
        history,
        history,
        horizon=8,
        n_samples=config.samples,
        dt=1.0,
        rng=np.random.default_rng(seed),
    )


def _prepare(manifest, seed, output):
    config = BudgetConfig(**manifest["experiment"])
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
    np.savez_compressed(
        directory / "train_validation.npz",
        train=train,
        validation=validation,
        train_trajectory_ids=np.arange(config.n_train),
        validation_trajectory_ids=np.arange(config.n_validation) + config.n_train,
    )
    (mediator, summary), fitting_seconds = _timed(
        train_mediator,
        train[:, :32].copy(),
        validation[:, :32].copy(),
        directory / "mediator",
        seeds["mediator_training"],
        device="cpu",
        smoke=config.smoke,
    )
    before = model_hash(mediator)
    train_histories, train_seconds = _preprocess(train[:, :32].copy(), mediator)
    val_histories, val_seconds = _preprocess(validation[:, :32].copy(), mediator)
    if before != model_hash(mediator):
        raise AssertionError("Preprocessing changed the selected mediator")
    arrays = {f"{m}_train_history": train_histories[m] for m in METHODS}
    arrays.update({f"{m}_validation_history": val_histories[m] for m in METHODS})
    arrays.update(train_future=train[:, 32:], validation_future=validation[:, 32:])
    np.savez_compressed(directory / "processed_train_validation.npz", **arrays)
    selected = next(
        c for c in summary["candidates"] if c["name"] == summary["selected"]
    )
    plan = allocate(
        config, manifest["work_profiles"], selected["config"]["hidden_channels"]
    )
    write_json(directory / "budget_plan.json", plan)
    write_json(
        directory / "preprocessing_audit.json",
        {
            "mediator_sha256": before,
            "history_only": True,
            "future_targets_unchanged": True,
            "normalization": "none",
            "factor": 1,
            "processed_sha256": {k: array_hash(v) for k, v in arrays.items()},
            "raw_train_history_sha256": array_hash(train[:, :32]),
            "raw_validation_history_sha256": array_hash(validation[:, :32]),
        },
    )
    write_json(
        directory / "timings.json",
        {
            "mediator_training_seconds": fitting_seconds,
            "preprocessing_seconds": {
                m: train_seconds[m] + val_seconds[m] for m in METHODS
            },
            "forecaster_training_seconds": {},
            "validation_sampling_seconds": {},
            "test_preprocessing_seconds": {},
            "test_inference_seconds": {},
        },
    )
    print(
        f"seed={seed} mediator={summary['selected']} budget_epochs="
        f"{ {m: p['epochs'] for m, p in plan['methods'].items()} }",
        flush=True,
    )


def _fit(manifest, seed, output):
    config = BudgetConfig(**manifest["experiment"])
    directory = output / f"seed_{seed}"
    seeds = manifest["seeds"][str(seed)]
    with np.load(directory / "train_validation.npz", allow_pickle=False) as archive:
        original = dict(archive)
    with np.load(
        directory / "processed_train_validation.npz", allow_pickle=False
    ) as archive:
        processed = dict(archive)
    plan = read(directory / "budget_plan.json")
    timings = read(directory / "timings.json")
    audit = {"methods": {}, "future_targets_unchanged": True, "factor": 1}
    order = manifest["method_order"][str(seed)]
    for method in order:
        verify_hashes(output, read(output / "MEDIATORS_FROZEN.json")["files"])
        verify_hashes(output, read(output / "BUDGETS_FROZEN.json")["files"])
        allocation = plan["methods"][method]
        training = replace(
            TrainingConfig(**manifest["forecaster_training"]),
            epochs=allocation["epochs"],
        )
        model = _new_model(manifest["architecture"], seeds["forecaster_initialization"])
        initial_hash = model_hash(model)
        model.to(config.device)
        train = (processed[f"{method}_train_history"], original["train"][:, 32:].copy())
        validation = (
            processed[f"{method}_validation_history"],
            original["validation"][:, 32:].copy(),
        )
        for name, pair in (("train", train), ("validation", validation)):
            if not np.array_equal(pair[1], processed[f"{name}_future"]):
                raise AssertionError("Noisy future targets changed")
        counts = {"optimizer_updates": 0, "train_examples": 0, "eval_examples": 0}

        def count_forward(module, inputs, counter=counts):
            counter["train_examples" if module.training else "eval_examples"] += len(
                inputs[0]
            )

        forward_handle = model.register_forward_pre_hook(count_forward)
        backward_handle = next(model.parameters()).register_post_accumulate_grad_hook(
            lambda parameter, counter=counts: counter.__setitem__(
                "optimizer_updates", counter["optimizer_updates"] + 1
            )
        )
        rows = []

        def score(
            current,
            epoch,
            method=method,
            training=training,
            validation=validation,
            rows=rows,
        ):
            samples, seconds = _timed(
                _sample,
                current,
                validation[0],
                manifest,
                training,
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
            print(
                f"seed={seed} method={method} epoch={epoch} validation_crps="
                f"{metrics['crps']:.6f} sample_seconds={seconds:.2f}",
                flush=True,
            )
            return metrics["crps"]

        start = perf_counter()
        started_utc = datetime.now(UTC).isoformat()

        def record(row, method=method, start=start):
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
                dt=1,
                seed=seeds["forecaster_training"],
                checkpoint=directory / f"{method}.pt",
                n_augmentations=1,
                validation_score=score,
                validation_score_epochs=tuple(config.validation_epochs_prefix),
                epoch_callback=record,
                device=config.device,
            )
        finally:
            forward_handle.remove()
            backward_handle.remove()
        expected = {
            "optimizer_updates": allocation["optimizer_updates"],
            "train_examples": allocation["epochs"] * config.n_train,
            "eval_examples": config.n_validation
            * config.diffusion_steps
            * (
                allocation["noise_validation_count"]
                + config.samples * len(allocation["validation_epochs"])
            ),
        }
        if (
            counts != expected
            or [r["epoch"] for r in rows] != allocation["validation_epochs"]
        ):
            raise AssertionError(
                f"Measured computation differs: {counts} vs {expected}"
            )
        best = min(rows, key=lambda r: r["crps"])
        checkpoint = torch.load(
            directory / f"{method}.pt", map_location="cpu", weights_only=True
        )
        if checkpoint["epoch"] != best["epoch"]:
            raise AssertionError("Checkpoint did not minimize validation CRPS")
        timings["forecaster_training_seconds"][method] = seconds
        timings["validation_sampling_seconds"][method] = sum(
            r["sampling_seconds"] for r in rows
        )
        audit["methods"][method] = {
            "initialization_sha256": initial_hash,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "measured_counts": counts,
            "epochs_completed": allocation["epochs"],
            "selected_epoch": best["epoch"],
            "selected_validation_crps": best["crps"],
            "training_config": asdict(training),
            "validation_candidate_epochs": allocation["validation_epochs"],
            "training_seed": seeds["forecaster_training"],
            "started_utc": started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "train_history_sha256": array_hash(train[0]),
            "train_future_sha256": array_hash(train[1]),
            "validation_history_sha256": array_hash(validation[0]),
            "validation_future_sha256": array_hash(validation[1]),
            "checkpoint_sha256": sha256(directory / f"{method}.pt"),
        }
        write_json(directory / "timings.json", timings)
        write_json(directory / "training_audit.json", audit)
        del model
        if config.device == "mps":
            torch.mps.empty_cache()
    if len({r["initialization_sha256"] for r in audit["methods"].values()}) != 1:
        raise AssertionError("Paired initialization differs")


def train_run(output, config=None, recipe_source=Path("outputs/csdi-gp-better")):
    config = config or BudgetConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    recipe = recovered_recipe(recipe_source)
    profile = count_csdi_work(recipe["architecture"], 40)
    if profile != count_csdi_work(recipe["architecture"], 40, batch_size=4):
        raise AssertionError("Work profile depends on batch size")
    if profile["forward"] != analytical_csdi_forward_work(recipe["architecture"], 40):
        raise AssertionError("Counted attention work disagrees with matrix formula")
    profiles = {
        "forecaster": profile,
        **{f"mediator_{w}": count_mediator_work(w) for w in (32, 64)},
    }
    training = replace(
        TrainingConfig(**recipe["training"]),
        epochs=config.base_epochs,
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
        "work_profiles": profiles,
        "methods": list(METHODS),
        "seeds": {str(s): seed_plan(s, config.namespace) for s in config.seeds},
        "method_order": {
            str(s): list(METHODS[i % 3 :] + METHODS[: i % 3])
            for i, s in enumerate(config.seeds)
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "budget_definition": (
            "full in-run dense matrix FLOPs; multiply plus add counts two"
        ),
        "budget_tolerance": "less than 0.01% unused; maximal feasible whole epoch",
        "excluded_from_primary_work": (
            "nonmatrix operations, optimizer, median comparisons, I/O and "
            "data movement; wall time separately"
        ),
        "search": (
            "four fixed mediator candidates; all charged; same seven forecast "
            "opportunities; no further tuning"
        ),
        "new_weights": True,
        "fresh_all_data_splits": True,
        "future_targets": "unchanged noisy observations",
        "normalization": "none",
        "history_only_preprocessing": True,
        "no_clean_signal_exposed": True,
        "no_densification": True,
    }
    write_json(output / "config.json", manifest)
    _snapshot(output)
    start = perf_counter()
    for seed in config.seeds:
        _prepare(manifest, seed, output)
    mediator_seconds = perf_counter() - start
    mediator_files = {
        str(p.relative_to(output)): sha256(p)
        for seed in config.seeds
        for p in (output / f"seed_{seed}/mediator").rglob("*")
        if p.is_file()
    }
    write_json(
        output / "MEDIATORS_FROZEN.json",
        {
            "files": mediator_files,
            "frozen_utc": datetime.now(UTC).isoformat(),
            "before_any_forecaster_training": True,
            "wall_seconds": mediator_seconds,
        },
    )
    write_json(
        output / "BUDGETS_FROZEN.json",
        {
            "files": {
                str(p.relative_to(output)): sha256(p)
                for p in output.glob("seed_*/budget_plan.json")
            },
            "frozen_utc": datetime.now(UTC).isoformat(),
            "before_any_forecaster_training": True,
            "configuration_sha256": sha256(output / "config.json"),
        },
    )
    forecaster_start = perf_counter()
    for seed in config.seeds:
        _fit(manifest, seed, output)
    write_json(
        output / "TRAINED.json",
        {
            "mediator_wall_seconds": mediator_seconds,
            "forecaster_wall_seconds": perf_counter() - forecaster_start,
            "wall_seconds": perf_counter() - start,
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )
    write_json(
        output / "FROZEN.json",
        {
            "files": hashes(output),
            "frozen_utc": datetime.now(UTC).isoformat(),
            "test_data_generated": False,
            "all_methods_and_seeds_selected": True,
        },
    )
    return manifest


def evaluate_run(output):
    output = Path(output)
    verify_frozen(output)
    if (output / "EVALUATION_STARTED.json").exists() or any(
        output.glob("seed_*/test.npz")
    ):
        raise FileExistsError("Existing test evidence is preserved")
    manifest = read(output / "config.json")
    config = BudgetConfig(**manifest["experiment"])
    torch.set_num_threads(config.threads)
    write_json(
        output / "EVALUATION_STARTED.json",
        {"started_utc": datetime.now(UTC).isoformat()},
    )
    start = perf_counter()
    for seed in config.seeds:
        directory = output / f"seed_{seed}"
        seeds = manifest["seeds"][str(seed)]
        values = generate_noisy_periodic(
            config.n_test, rng=np.random.default_rng(seeds["test_data"])
        )
        history, target = values[:, :32], values[:, 32:]
        ids = np.arange(config.n_test) + config.n_train + config.n_validation
        np.savez_compressed(
            directory / "test.npz", history=history, target=target, trajectory_ids=ids
        )
        mediator = load_mediator(directory / "mediator/selected.pt", device="cpu")
        processed, preprocessing = _preprocess(history, mediator)
        timing = {
            "test_preprocessing_seconds": preprocessing,
            "test_inference_seconds": {},
        }
        scores, audit = {}, {}
        for method in manifest["method_order"][str(seed)]:
            checkpoint = torch.load(
                directory / f"{method}.pt", map_location="cpu", weights_only=True
            )
            model = _new_model(manifest["architecture"], 0)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(config.device).eval()
            training = TrainingConfig(**checkpoint["config"])
            samples, seconds = _timed(
                _sample,
                model,
                processed[method],
                manifest,
                training,
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
            timing["test_inference_seconds"][method] = seconds
            audit[method] = {
                "observed_argument_sha256": array_hash(processed[method]),
                "dense_history_argument_sha256": array_hash(processed[method]),
                "raw_side_channel": False,
                "future_conditioning": False,
                "samples": config.samples,
                "horizon": 8,
                "sampling_seed": seeds["test_sampling"],
            }
            write_json(directory / "scores.json", scores)
            write_json(directory / "evaluation_timings.json", timing)
            write_json(directory / "inference_audit.json", audit)
            print(
                f"test seed={seed} method={method} "
                f"crps={scores[method]['crps']:.6f} seconds={seconds:.2f}",
                flush=True,
            )
            del model
            if config.device == "mps":
                torch.mps.empty_cache()
        write_json(
            directory / "combined_timings.json",
            {**read(directory / "timings.json"), **timing},
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
