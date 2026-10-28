"""Fresh forecasting comparison conditional on both larger-densifier gates.

The selected waveform densifier and original train/validation arrays are
imported unchanged. Forecasters are initialized independently of densifiers.
Fresh test trajectories are generated only after every forecaster is frozen.
"""

import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.generators.waveform import generate_waveforms
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel
from densiffusion.pipeline import forecast
from densiffusion.samplers import DDPMForecastSampler, DDPMSampler
from densiffusion.training import TrainingConfig, train_forecaster
from densiffusion.waveform_experiment import (
    METHODS,
    AuditedDensifier,
    LinearDensifier,
    model_hash,
    seed_plan,
    timed,
    write_json,
)
from densiffusion.waveform_imputation import _snapshot, sha256, verify_frozen


@dataclass(frozen=True)
class LargeForecastConfig:
    seeds: tuple[int, ...] = (27101, 27102, 27103)
    epochs: int = 600
    validation_interval: int = 10
    validation_score_interval: int = 150
    n_train: int = 1024
    n_validation: int = 256
    n_test: int = 1024
    context_length: int = 25
    horizon: int = 8
    dt: float = 0.1
    factor: int = 2
    batch_size: int = 64
    sample_batch_size: int = 256
    n_histories: int = 8
    n_futures: int = 8
    n_augmentations: int = 1
    diffusion_steps: int = 50
    device: str = "cpu"
    threads: int = 1
    workers: int = 3
    test_namespace: int = 2026102702
    label: str = "large_forecast"

    def __post_init__(self):
        for name in (
            "epochs",
            "validation_interval",
            "validation_score_interval",
            "n_train",
            "n_validation",
            "n_test",
            "context_length",
            "horizon",
            "factor",
            "batch_size",
            "sample_batch_size",
            "n_histories",
            "n_futures",
            "n_augmentations",
            "diffusion_steps",
            "threads",
            "workers",
            "test_namespace",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Require distinct paired seeds")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("Seeds must be integers in [0, 2**32)")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.factor != 2 or self.context_length < 2:
            raise ValueError("Require factor=2 and at least two history observations")
        if self.n_augmentations != 1:
            raise ValueError("This comparison requires one cached augmentation per row")
        if self.validation_score_interval % self.validation_interval:
            raise ValueError("Score interval must be a multiple of validation interval")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("Unknown device")
        if self.label == "smoke" and self.test_namespace == 2026102702:
            raise ValueError("Smoke runs require a separate test_namespace")
        if self.label != "smoke":
            required = {
                "epochs": 600,
                "validation_interval": 10,
                "validation_score_interval": 150,
                "n_train": 1024,
                "n_validation": 256,
                "n_test": 1024,
                "context_length": 25,
                "horizon": 8,
                "dt": 0.1,
                "batch_size": 64,
                "n_histories": 8,
                "n_futures": 8,
                "n_augmentations": 1,
                "diffusion_steps": 50,
                "test_namespace": 2026102702,
            }
            if any(getattr(self, name) != value for name, value in required.items()):
                raise ValueError(
                    "Scientific runs must retain the declared comparison budget"
                )
            if len(self.seeds) != 3:
                raise ValueError("Scientific runs require three paired seeds")


def _read(path):
    return json.loads(Path(path).read_text())


def _hashes(directory):
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _verify_hashes(directory, hashes):
    for name, expected in hashes.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Protected source changed: {directory / name}")


def verify_imputation_source(source):
    """Require frozen selected q weights and both completed, passed gates."""
    source = Path(source)
    frozen = verify_frozen(source)
    if frozen["validation_gate"]["passed"] is not True:
        raise ValueError("Forecasting requires a passed validation imputation gate")
    transfer = _read(source / "TRANSFER.json")
    if transfer["passed"] is not True:
        raise ValueError("Forecasting requires a passed resolution-transfer gate")
    manifest = _read(source / "config.json")
    if manifest["architecture"] != manifest["recovered_gp_recipe"]["architecture"]:
        raise ValueError("Imputation architecture differs from the recovered GP recipe")
    if manifest["recovered_gp_recipe"]["gp_weights_reused"] is not False:
        raise ValueError("Require a freshly trained waveform densifier")
    return manifest


def fresh_seed_plan(seed, namespace=2026102702):
    original = seed_plan(seed)
    streams = np.random.SeedSequence([seed, namespace]).spawn(2)
    fresh = dict(original)
    for name, stream in zip(("test_data", "test_sampling"), streams, strict=True):
        fresh[name] = int(stream.generate_state(1)[0])
        if fresh[name] == original[name]:
            raise ValueError("Fresh test stream collided with its original stream")
    return fresh


def _new_model(architecture, seed):
    torch.manual_seed(seed)
    return ImputationModel(CSDIDenoiser(**architecture))


def _load_model(path, architecture, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = _new_model(architecture, 0)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval(), state


def _load_q(directory, manifest, config):
    model, state = _load_model(
        directory / "densifier.pt", manifest["architecture"], config.device
    )
    model.requires_grad_(False)
    sampling = replace(
        TrainingConfig(**state["config"]),
        device=config.device,
        batch_size=config.sample_batch_size,
    )
    return model, DDPMSampler(model, sampling)


def _sample_forecasts(model, densifier, history, manifest, config, method, seed):
    sampling = replace(
        TrainingConfig(**manifest["forecaster_training"]),
        batch_size=config.sample_batch_size,
    )
    return forecast(
        history,
        densifier,
        DDPMForecastSampler(model, sampling),
        horizon=config.horizon,
        factor=1 if method == "direct_csdi" else config.factor,
        n_histories=config.n_histories,
        n_futures=config.n_futures,
        dt=config.dt,
        seed=seed,
    )


def _train_seed(manifest, seed, directory):
    config = LargeForecastConfig(**manifest["experiment"])
    directory = Path(directory)
    directory.mkdir()
    torch.set_num_threads(config.threads)
    provenance = manifest["densifier_import"]
    source = Path(provenance["source"]) / directory.name
    for name in ("densifier.pt", "train_validation.npz"):
        expected = provenance["source_sha256"][f"{directory.name}/{name}"]
        if sha256(source / name) != expected:
            raise ValueError("Densifier import source changed")
        shutil.copyfile(source / name, directory / name)
        if sha256(directory / name) != expected:
            raise AssertionError("Densifier/data import was not byte-identical")
    with np.load(directory / "train_validation.npz", allow_pickle=False) as data:
        train_values, validation_values = data["train"], data["validation"]
        validation_ids = data["validation_trajectory_ids"]
    expected_length = config.context_length + config.horizon
    if train_values.shape != (
        config.n_train,
        expected_length,
        1,
    ) or validation_values.shape != (config.n_validation, expected_length, 1):
        raise ValueError("Imported original-grid arrays do not match configuration")
    train = (
        train_values[:, : config.context_length],
        train_values[:, config.context_length :],
    )
    validation = (
        validation_values[:, : config.context_length],
        validation_values[:, config.context_length :],
    )
    seeds = manifest["seeds"][str(seed)]
    write_json(directory / "seeds.json", seeds)
    q_model, q = _load_q(directory, manifest, config)
    q_hash = model_hash(q_model)
    training = TrainingConfig(**manifest["forecaster_training"])
    q_cost = provenance["selected_attempt_training_seconds_by_seed"][str(seed)]
    prior_cost = provenance["prior_development_training_seconds_by_seed"][str(seed)]
    timings = {
        "densifier_training_seconds": 0.0,
        "imported_densifier_training_seconds": q_cost,
        "prior_densifier_development_seconds": prior_cost,
        "methods": {},
    }
    initializations, updates, completed_epochs = {}, {}, {}
    expected_updates = config.epochs * math.ceil(config.n_train / config.batch_size)
    for method in METHODS:
        model = _new_model(
            manifest["architecture"], seeds["forecaster_initialization"]
        ).to(config.device)
        initializations[method] = model_hash(model)
        sampler = q if method == "densiffusion" else LinearDensifier()
        audited = AuditedDensifier(sampler, config.device)
        validation_rows, epoch_rows = [], []
        update_count = [0]

        def count_backward(parameter, count=update_count):
            count[0] += 1

        handle = next(model.parameters()).register_post_accumulate_grad_hook(
            count_backward
        )

        def forecast_crps(
            candidate, epoch, method=method, sampler=sampler, rows=validation_rows
        ):
            samples, seconds = timed(
                _sample_forecasts,
                candidate,
                sampler,
                validation[0],
                manifest,
                config,
                method,
                seeds["validation_sampling"],
                device=config.device,
            )
            scores = evaluate_forecast(samples, validation[1])
            filename = f"{method}_validation_epoch_{epoch:04d}.npz"
            path = directory / filename
            np.savez_compressed(
                path,
                history=validation[0],
                target=validation[1],
                samples=samples,
                trajectory_ids=validation_ids,
                history_times=np.arange(config.context_length) * config.dt,
                future_times=np.arange(
                    config.context_length, config.context_length + config.horizon
                )
                * config.dt,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "seconds": seconds,
                    "samples": filename,
                    "samples_sha256": sha256(path),
                    **scores,
                }
            )
            write_json(directory / f"{method}_validation.json", rows)
            return scores["crps"]

        start = perf_counter()

        def record_epoch(record, method=method, rows=epoch_rows, started=start):
            rows.append(record.copy())
            with (directory / "training.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "method": method,
                            **record,
                            "elapsed_seconds": perf_counter() - started,
                        }
                    )
                    + "\n"
                )

        try:
            _, seconds = timed(
                train_forecaster,
                model,
                audited,
                train,
                validation,
                config=training,
                factor=1 if method == "direct_csdi" else config.factor,
                dt=config.dt,
                n_augmentations=config.n_augmentations,
                seed=seeds["forecaster_training"],
                checkpoint=directory / f"{method}_forecaster.pt",
                validation_score=forecast_crps,
                validation_score_interval=config.validation_score_interval,
                epoch_callback=record_epoch,
                device=config.device,
            )
        finally:
            handle.remove()
        updates[method] = update_count[0]
        completed_epochs[method] = len(epoch_rows)
        if (
            updates[method] != expected_updates
            or completed_epochs[method] != config.epochs
        ):
            raise AssertionError(
                "Forecaster optimization-update budget was not completed"
            )
        selected = torch.load(
            directory / f"{method}_forecaster.pt", map_location="cpu", weights_only=True
        )
        if (
            selected["epoch"]
            != min(validation_rows, key=lambda row: row["crps"])["epoch"]
        ):
            raise AssertionError(
                "Checkpoint does not minimize validation forecast CRPS"
            )
        augmentation_seconds = sum(row["seconds"] for row in audited.calls)
        validation_seconds = sum(row["seconds"] for row in validation_rows)
        timings["methods"][method] = {
            "augmentation_and_forecaster_training_seconds": seconds,
            "augmentation_sampling_seconds": augmentation_seconds,
            "forecast_validation_seconds": validation_seconds,
            "forecaster_fit_seconds": seconds
            - augmentation_seconds
            - validation_seconds,
            "new_training_seconds": seconds,
            "total_training_seconds": seconds
            + (q_cost if method == "densiffusion" else 0),
            "imported_densifier_training_seconds": q_cost
            if method == "densiffusion"
            else 0,
        }
        write_json(directory / f"{method}_augmentation_audit.json", audited.calls)
        write_json(directory / "timings.json", timings)
        if model_hash(q_model) != q_hash or any(
            p.requires_grad for p in q_model.parameters()
        ):
            raise AssertionError(
                "Imported densifier changed during forecasting training"
            )
    if len(set(initializations.values())) != 1:
        raise AssertionError("Paired forecasting initializations differ")
    write_json(
        directory / "pairing_audit.json",
        {
            "initial_state_sha256": initializations,
            "forecaster_parameter_count": sum(p.numel() for p in model.parameters()),
            "additional_densifier_parameters": sum(
                p.numel() for p in q_model.parameters()
            ),
            "updates_per_forecaster": expected_updates,
            "measured_updates_by_method": updates,
            "completed_epochs_by_method": completed_epochs,
            "update_counter": (
                "first model parameter post-accumulate-grad hook, "
                "one per successful optimizer step"
            ),
            "shared_training_seed": seeds["forecaster_training"],
            "example_order": (
                "same permutation stream, batch size and "
                "one augmentation per trajectory"
            ),
            "diffusion_noise": (
                "same seed; differing grid shapes prevent complete noise-entry pairing"
            ),
            "all_original_observations_preserved": True,
            "densifier_weights_unchanged": True,
            "densifier_import_sha256": sha256(directory / "densifier.pt"),
        },
    )
    write_json(
        directory / "TRAINED.json", {"finished_utc": datetime.now(UTC).isoformat()}
    )


def _workers(function, manifest, output):
    config = LargeForecastConfig(**manifest["experiment"])
    jobs = [(manifest, seed, output / f"seed_{seed}") for seed in config.seeds]
    if config.workers > 1 and config.device == "cpu" and len(jobs) > 1:
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(jobs)), mp_context=get_context("spawn")
        ) as executor:
            futures = [executor.submit(function, *job) for job in jobs]
            return [future.result() for future in futures]
    return [function(*job) for job in jobs]


def _import_provenance(source, source_manifest, config, development_sources):
    prior = {str(seed): 0.0 for seed in config.seeds}
    development = []
    for directory in dict.fromkeys(Path(p).resolve() for p in development_sources):
        if directory == source:
            continue
        if not (directory / "TRAINED.json").is_file():
            raise ValueError("Prior development cost requires completed training")
        hashes = {
            "config.json": sha256(directory / "config.json"),
            "TRAINED.json": sha256(directory / "TRAINED.json"),
        }
        for seed in config.seeds:
            name = f"seed_{seed}/summary.json"
            prior[str(seed)] += _read(directory / name)["training_seconds"]
            hashes[name] = sha256(directory / name)
        development.append({"source": str(directory), "source_sha256": hashes})
    checkpoints, selected_cost = {}, {}
    for seed in config.seeds:
        name = f"seed_{seed}/densifier.pt"
        state = torch.load(source / name, map_location="cpu", weights_only=True)
        summary = _read(source / f"seed_{seed}/summary.json")
        if state["epoch"] != summary["selected_epoch"]:
            raise ValueError("Imported checkpoint differs from selected q summary")
        checkpoints[str(seed)] = {
            "source": name,
            "sha256": sha256(source / name),
            "config": state["config"],
            "selected_epoch": state["epoch"],
            "parameter_count": summary["parameter_count"],
        }
        selected_cost[str(seed)] = summary["training_seconds"]
    return {
        "source": str(source),
        "source_sha256": _hashes(source),
        "architecture": source_manifest["architecture"],
        "checkpoints": checkpoints,
        "new_training_seconds": 0.0,
        "selected_attempt_training_seconds_by_seed": selected_cost,
        "prior_development_training_seconds_by_seed": prior,
        "prior_development_sources": development,
        "cost_scope": (
            "Selected attempt training includes validation and all its epochs; "
            "earlier supplied attempts are reported separately."
        ),
    }


def _verify_import(manifest):
    imported = manifest["densifier_import"]
    source = Path(imported["source"])
    verify_imputation_source(source)
    _verify_hashes(source, imported["source_sha256"])
    for previous in imported["prior_development_sources"]:
        _verify_hashes(Path(previous["source"]), previous["source_sha256"])


def train_run(output, imputation_source, config=None, *, development_sources=()):
    """Import gated q and train all paired forecasters, without accessing test data."""
    config = config or LargeForecastConfig()
    output, source = Path(output).resolve(), Path(imputation_source).resolve()
    source_manifest = verify_imputation_source(source)
    if not set(config.seeds) <= set(source_manifest["experiment"]["seeds"]):
        raise ValueError("Every forecasting seed requires a gated source densifier")
    if output == source or source in output.parents:
        raise ValueError(
            "Forecast output must be separate from the immutable imputation source"
        )
    imported = _import_provenance(source, source_manifest, config, development_sources)
    original_plans = {
        str(seed): _read(source / f"seed_{seed}/seeds.json") for seed in config.seeds
    }
    if any(original_plans[str(seed)] != seed_plan(seed) for seed in config.seeds):
        raise ValueError("Source seeds differ from the original paired seed plan")
    training = replace(
        TrainingConfig(**source_manifest["recovered_gp_recipe"]["training"]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        device=config.device,
        validation_steps="all",
        validation_interval=config.validation_interval,
        diffusion_steps=config.diffusion_steps,
    )
    manifest = {
        "experiment": asdict(config),
        "architecture": source_manifest["architecture"],
        "forecaster_training": asdict(training),
        "densifier_training": source_manifest["training"],
        "densifier_import": imported,
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": {
            str(seed): fresh_seed_plan(seed, config.test_namespace)
            for seed in config.seeds
        },
        "original_seed_plans": original_plans,
        "fresh_test_streams": (
            "SeedSequence([paired_seed, test_namespace]).spawn(2): "
            "test_data then test_sampling"
        ),
        "model_selection": (
            "Original-grid validation forecast CRPS at score intervals plus final "
            "epoch; fixed seeds and ensemble count matching test evaluation"
        ),
        "normalization": "none",
        "fine_grid_truth_used_for_training_or_selection": False,
        "transfer_diagnostics_completed_before_forecast_training": True,
        "test_data_generated_before_all_models_frozen": False,
        "augmentation": (
            "Imported frozen q; independent history-only and anchored-future draws, "
            "cached once"
        ),
        "joint_fine_truth_assumption": False,
        "fresh_forecast_test_required": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", manifest)
    _snapshot(output)
    _verify_import(manifest)
    started = perf_counter()
    _workers(_train_seed, manifest, output)
    _verify_import(manifest)
    hashes = {
        str(p.relative_to(output)): sha256(p)
        for p in sorted(output.glob("seed_*/*.pt"))
    }
    selection_evidence = {
        str(path.relative_to(output)): sha256(path)
        for pattern in (
            "seed_*/*_validation_epoch_*.npz",
            "seed_*/*_validation.json",
            "seed_*/training.jsonl",
            "seed_*/pairing_audit.json",
            "seed_*/*_augmentation_audit.json",
        )
        for path in sorted(output.glob(pattern))
    }
    write_json(
        output / "FROZEN.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "checkpoint_sha256": hashes,
            "config_sha256": sha256(output / "config.json"),
            "test_evaluated": False,
            "training_wall_seconds": perf_counter() - started,
            "selection_evidence_sha256": selection_evidence,
            "train_validation_sha256": {
                f"seed_{seed}/train_validation.npz": sha256(
                    output / f"seed_{seed}/train_validation.npz"
                )
                for seed in config.seeds
            },
        },
    )
    return manifest


def _evaluate_seed(manifest, seed, directory):
    config = LargeForecastConfig(**manifest["experiment"])
    torch.set_num_threads(config.threads)
    seeds = manifest["seeds"][str(seed)]
    # This function is dispatched only after global freeze verification.
    values = generate_waveforms(
        config.n_test,
        config.context_length + config.horizon,
        config.dt,
        rng=np.random.default_rng(seeds["test_data"]),
    )
    history, target = (
        values[:, : config.context_length],
        values[:, config.context_length :],
    )
    _, q = _load_q(directory, manifest, config)
    arrays = {
        "history": history,
        "target": target,
        "history_times": np.arange(config.context_length) * config.dt,
        "future_times": np.arange(
            config.context_length, config.context_length + config.horizon
        )
        * config.dt,
        "trajectory_ids": np.arange(
            config.n_train + config.n_validation,
            config.n_train + config.n_validation + config.n_test,
        ),
    }
    timings = _read(directory / "timings.json")
    scores = {}
    for method in METHODS:
        model, _ = _load_model(
            directory / f"{method}_forecaster.pt",
            manifest["architecture"],
            config.device,
        )
        sampler = q if method == "densiffusion" else LinearDensifier()
        audited = AuditedDensifier(sampler, config.device)
        samples, seconds = timed(
            _sample_forecasts,
            model,
            audited,
            history,
            manifest,
            config,
            method,
            seeds["test_sampling"],
            device=config.device,
        )
        arrays[method] = samples
        scores[method] = evaluate_forecast(samples, target)
        timings["methods"][method].update(
            forecast_seconds=seconds,
            forecast_seconds_per_trajectory=seconds / config.n_test,
            history_densification_seconds=sum(row["seconds"] for row in audited.calls),
        )
        write_json(directory / f"{method}_inference_audit.json", audited.calls)
    np.savez_compressed(directory / "samples.npz", **arrays)
    write_json(directory / "scores.json", scores)
    write_json(directory / "timings.json", timings)
    return scores


def evaluate_run(output):
    """Evaluate frozen weights once on the pre-recorded fresh test streams."""
    output = Path(output).resolve()
    frozen = _read(output / "FROZEN.json")
    if sha256(output / "config.json") != frozen["config_sha256"]:
        raise ValueError("Frozen forecasting configuration changed")
    _verify_hashes(output, frozen["checkpoint_sha256"])
    _verify_hashes(output, frozen["train_validation_sha256"])
    _verify_hashes(output, frozen["selection_evidence_sha256"])
    manifest = _read(output / "config.json")
    _verify_import(manifest)
    seeds = manifest["experiment"]["seeds"]
    if (output / "EVALUATED.json").exists() or any(
        (output / f"seed_{seed}/samples.npz").exists() for seed in seeds
    ):
        raise FileExistsError(
            "Fresh test evaluation already started; refusing overwrite"
        )
    if not all((output / f"seed_{seed}/TRAINED.json").is_file() for seed in seeds):
        raise ValueError("All seed training must finish before fresh test generation")
    results = _workers(_evaluate_seed, manifest, output)
    _verify_hashes(output, frozen["checkpoint_sha256"])
    _verify_hashes(output, frozen["train_validation_sha256"])
    _verify_hashes(output, frozen["selection_evidence_sha256"])
    _verify_import(manifest)
    scores = dict(zip(map(str, seeds), results, strict=True))
    write_json(output / "scores.json", scores)
    write_json(
        output / "EVALUATED.json", {"finished_utc": datetime.now(UTC).isoformat()}
    )
    return scores
