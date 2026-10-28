"""Paired, validation-selected waveform forecasting experiment.

Training and confirmatory evaluation are separate phases. Test observations are
generated only after every seed and method has finished checkpoint selection.
"""

import hashlib
import json
import logging
import math
import platform
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
from densiffusion.conditioning import prepare_observations
from densiffusion.generators.waveform import generate_waveforms
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel
from densiffusion.pipeline import forecast
from densiffusion.samplers import DDPMForecastSampler, DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier, train_forecaster

METHODS = ("direct_csdi", "linear_densification", "densiffusion")
ARCHITECTURE = {
    "n_channels": 1,
    "hidden_channels": 16,
    "residual_layers": 2,
    "attention_heads": 2,
    "diffusion_embedding_dim": 32,
    "time_embedding_dim": 32,
    "feature_embedding_dim": 8,
    "feedforward_dim": 32,
    "dropout": 0.1,
}


@dataclass(frozen=True)
class WaveformConfig:
    seeds: tuple[int, ...] = (27101, 27102, 27103)
    n_train: int = 1024
    n_validation: int = 256
    n_test: int = 1024
    context_length: int = 25
    horizon: int = 8
    dt: float = 0.1
    factor: int = 2
    epochs: int = 600
    densifier_epochs: int = 400
    validation_interval: int = 100
    densifier_validation_interval: int = 50
    batch_size: int = 64
    sample_batch_size: int = 256
    diffusion_steps: int = 50
    n_histories: int = 8
    n_futures: int = 8
    n_augmentations: int = 1
    device: str = "cpu"
    threads: int = 1
    workers: int = 3
    label: str = "primary"

    def __post_init__(self):
        for name in (
            "n_train",
            "n_validation",
            "n_test",
            "context_length",
            "horizon",
            "epochs",
            "densifier_epochs",
            "validation_interval",
            "densifier_validation_interval",
            "batch_size",
            "sample_batch_size",
            "diffusion_steps",
            "n_histories",
            "n_futures",
            "n_augmentations",
            "threads",
            "workers",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.context_length < 2 or self.factor != 2:
            raise ValueError("Require context_length >= 2 and factor=2")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Require distinct experimental seeds")
        if any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("seeds must be integers in [0, 2**32)")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps, or cuda")
        if self.label != "smoke" and self.n_histories * self.n_futures < 64:
            raise ValueError("Scientific runs require at least 64 forecast draws")


class LinearDensifier:
    """Same deterministic linear construction as the existing GP experiment."""

    def sample(self, observed, *, factor, n_samples, dt, rng):
        values, hidden, times = prepare_observations(observed, factor=factor, dt=dt)
        for row in range(len(values)):
            for channel in range(values.shape[2]):
                known = ~hidden[row, :, channel]
                values[row, :, channel] = np.interp(
                    times[row], times[row, known], values[row, known, channel]
                )
        return np.repeat(values[:, None], n_samples, axis=1)


class AuditedDensifier:
    """Record augmentation cost and verify observation preservation on each call."""

    def __init__(self, sampler, device):
        self.sampler = sampler
        self.device = device
        self.calls = []
        if hasattr(sampler, "model"):
            self.model = sampler.model

    def sample(self, observed, **kwargs):
        result, seconds = timed(
            self.sampler.sample, observed, device=self.device, **kwargs
        )
        expected = np.broadcast_to(
            observed[:, None], result[:, :, :: kwargs["factor"]].shape
        )
        if not np.array_equal(result[:, :, :: kwargs["factor"]], expected):
            raise AssertionError("Densifier changed observed values")
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite augmentation")
        self.calls.append(
            {
                "input_shape": list(observed.shape),
                "output_shape": list(result.shape),
                "input_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
                "dt": kwargs["dt"],
                "seconds": seconds,
                "observations_preserved": True,
            }
        )
        return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(function, *args, device, **kwargs):
    synchronize(device)
    start = perf_counter()
    result = function(*args, **kwargs)
    synchronize(device)
    return result, perf_counter() - start


def seed_plan(seed):
    names = (
        "train_data",
        "validation_data",
        "test_data",
        "densifier_initialization",
        "densifier_training",
        "forecaster_initialization",
        "forecaster_training",
        "validation_sampling",
        "test_sampling",
        "diagnostic_sampling",
    )
    return dict(
        zip(
            names,
            [
                int(x.generate_state(1)[0])
                for x in np.random.SeedSequence(seed).spawn(len(names))
            ],
            strict=True,
        )
    )


def training_config(config, *, densifier=False):
    return TrainingConfig(
        epochs=config.densifier_epochs if densifier else config.epochs,
        batch_size=config.batch_size,
        learning_rate=1e-3,
        diffusion_steps=config.diffusion_steps,
        beta_start=1e-4,
        beta_end=0.5,
        device=config.device,
        schedule="quadratic",
        weight_decay=1e-6,
        lr_decay_fractions=(0.75, 0.9),
        lr_decay_factor=0.1,
        validation_interval=(
            config.densifier_validation_interval
            if densifier
            else config.validation_interval
        ),
    )


def new_model(seed):
    torch.manual_seed(seed)
    return ImputationModel(CSDIDenoiser(**ARCHITECTURE))


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def dataset(config, seed, split):
    return generate_waveforms(
        getattr(config, f"n_{split}"),
        config.context_length + config.horizon,
        config.dt,
        rng=np.random.default_rng(seed_plan(seed)[f"{split}_data"]),
    )


def split_pair(values, config):
    return values[:, : config.context_length], values[:, config.context_length :]


def sample_forecasts(model, densifier, history, config, method, seed):
    # A larger sampling batch changes neither model weights nor draw count.
    sample_config = replace(
        training_config(config), batch_size=config.sample_batch_size
    )
    return forecast(
        history,
        densifier,
        DDPMForecastSampler(model, sample_config),
        horizon=config.horizon,
        factor=1 if method == "direct_csdi" else config.factor,
        n_histories=config.n_histories,
        n_futures=config.n_futures,
        dt=config.dt,
        seed=seed,
    )


def _epoch_writer(path, method, start):
    def callback(record):
        row = {"method": method, **record, "elapsed_seconds": perf_counter() - start}
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    return callback


def train_seed(config, seed, directory):
    torch.set_num_threads(config.threads)
    directory.mkdir()
    seeds = seed_plan(seed)
    train_values = dataset(config, seed, "train")
    validation_values = dataset(config, seed, "validation")
    train, validation = (
        split_pair(train_values, config),
        split_pair(validation_values, config),
    )
    np.savez_compressed(
        directory / "train_validation.npz",
        train=train_values,
        validation=validation_values,
        train_trajectory_ids=np.arange(config.n_train),
        validation_trajectory_ids=np.arange(
            config.n_train, config.n_train + config.n_validation
        ),
    )
    write_json(directory / "seeds.json", seeds)
    q_config = training_config(config, densifier=True)
    p_config = training_config(config)
    q_model = new_model(seeds["densifier_initialization"])
    start = perf_counter()
    _, q_seconds = timed(
        train_densifier,
        q_model,
        train_values,
        validation_values,
        config=q_config,
        dt=config.dt,
        seed=seeds["densifier_training"],
        checkpoint=directory / "densifier.pt",
        device=config.device,
        epoch_callback=_epoch_writer(directory / "training.jsonl", "densifier", start),
    )
    q_model.requires_grad_(False).eval()
    q = DDPMSampler(q_model, replace(q_config, batch_size=config.sample_batch_size))
    timings = {"densifier_training_seconds": q_seconds, "methods": {}}
    initializations = {}
    augmentation_checks = []
    for method in METHODS:
        logging.info("seed=%d method=%s starting forecaster", seed, method)
        model = new_model(seeds["forecaster_initialization"])
        initializations[method] = model_hash(model)
        sampler = q if method == "densiffusion" else LinearDensifier()
        audited = AuditedDensifier(sampler, config.device)
        validation_costs = []

        def forecast_crps(
            candidate,
            epoch,
            sampler=sampler,
            method=method,
            validation_costs=validation_costs,
        ):
            samples, seconds = timed(
                sample_forecasts,
                candidate,
                sampler,
                validation[0],
                config,
                method,
                seeds["validation_sampling"],
                device=config.device,
            )
            scores = evaluate_forecast(samples, validation[1])
            validation_costs.append({"epoch": epoch, "seconds": seconds, **scores})
            logging.info(
                "seed=%d method=%s epoch=%d validation_crps=%.6f",
                seed,
                method,
                epoch,
                scores["crps"],
            )
            write_json(directory / f"{method}_validation.json", validation_costs)
            return scores["crps"]

        start = perf_counter()
        _, seconds = timed(
            train_forecaster,
            model,
            audited,
            train,
            validation,
            config=p_config,
            factor=1 if method == "direct_csdi" else config.factor,
            dt=config.dt,
            n_augmentations=config.n_augmentations,
            seed=seeds["forecaster_training"],
            checkpoint=directory / f"{method}_forecaster.pt",
            device=config.device,
            validation_score=forecast_crps,
            epoch_callback=_epoch_writer(directory / "training.jsonl", method, start),
        )
        write_json(directory / f"{method}_augmentation_audit.json", audited.calls)
        augmentation_checks.extend(
            call["observations_preserved"] for call in audited.calls
        )
        augmentation_seconds = sum(call["seconds"] for call in audited.calls)
        validation_seconds = sum(row["seconds"] for row in validation_costs)
        timings["methods"][method] = {
            "augmentation_and_forecaster_training_seconds": seconds,
            "augmentation_sampling_seconds": augmentation_seconds,
            "forecast_validation_seconds": validation_seconds,
            "forecaster_fit_seconds": seconds
            - augmentation_seconds
            - validation_seconds,
            "total_training_seconds": seconds
            + (q_seconds if method == "densiffusion" else 0),
        }
        write_json(directory / "timings.json", timings)
    if len(set(initializations.values())) != 1:
        raise AssertionError("Forecaster initialization must be identical")
    write_json(
        directory / "pairing_audit.json",
        {
            "initial_state_sha256": initializations,
            "forecaster_parameter_count": sum(p.numel() for p in model.parameters()),
            "additional_densifier_parameters": sum(
                p.numel() for p in q_model.parameters()
            ),
            "updates_per_forecaster": config.epochs
            * math.ceil(config.n_train * config.n_augmentations / config.batch_size),
            "shared_training_seed": seeds["forecaster_training"],
            "example_order": (
                "same NumPy permutation stream, same example and augmentation counts"
            ),
            "diffusion_noise": (
                "same seed; shapes differ across grids, so noise entries "
                "are not fully paired"
            ),
            "all_original_observations_preserved": all(augmentation_checks),
        },
    )
    write_json(
        directory / "TRAINED.json", {"finished_utc": datetime.now(UTC).isoformat()}
    )


def _load_model(path, config):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = new_model(0)
    model.load_state_dict(saved["model_state_dict"])
    return model.to(config.device).eval()


def evaluate_seed(config, seed, directory):
    torch.set_num_threads(config.threads)
    if (directory / "samples.npz").exists():
        raise FileExistsError("Confirmatory samples already exist")
    values = dataset(config, seed, "test")
    history, target = split_pair(values, config)
    seeds = seed_plan(seed)
    q_model = _load_model(directory / "densifier.pt", config).requires_grad_(False)
    q = DDPMSampler(
        q_model,
        replace(
            training_config(config, densifier=True), batch_size=config.sample_batch_size
        ),
    )
    timings = json.loads((directory / "timings.json").read_text())
    scores = {}
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
    for method in METHODS:
        model = _load_model(directory / f"{method}_forecaster.pt", config)
        sampler = AuditedDensifier(
            q if method == "densiffusion" else LinearDensifier(), config.device
        )
        samples, seconds = timed(
            sample_forecasts,
            model,
            sampler,
            history,
            config,
            method,
            seeds["test_sampling"],
            device=config.device,
        )
        arrays[method] = samples
        scores[method] = evaluate_forecast(samples, target)
        timings["methods"][method]["forecast_seconds"] = seconds
        history_seconds = sum(call["seconds"] for call in sampler.calls)
        timings["methods"][method]["inference_history_densification_seconds"] = (
            history_seconds
        )
        timings["methods"][method]["inference_future_and_overhead_seconds"] = (
            seconds - history_seconds
        )
        write_json(directory / f"{method}_inference_audit.json", sampler.calls)
        timings["methods"][method]["forecast_seconds_per_trajectory"] = (
            seconds / config.n_test
        )
    np.savez_compressed(directory / "samples.npz", **arrays)
    write_json(directory / "scores.json", scores)
    write_json(directory / "timings.json", timings)
    return scores


def train_run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.threads)
    manifest = {
        "experiment": asdict(config),
        "architecture": ARCHITECTURE,
        "forecaster_training": asdict(training_config(config)),
        "densifier_training": asdict(training_config(config, densifier=True)),
        "created_utc": datetime.now(UTC).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "platform": platform.platform(),
        },
        "normalization": "none",
        "fine_grid_truth_before_freeze": False,
        "test_data_generated_before_all_models_frozen": False,
        "model_selection": (
            "original-grid validation forecast CRPS, fixed ensemble seeds; "
            "draw count equals test draw count"
        ),
        "densifier_selection": "original-grid alternating-mask validation noise MSE",
        "augmentation": (
            "existing independent history-only and anchored-future draws; cached once"
        ),
        "resolution_transfer": (
            "densifier learns 0.2 to 0.1; applies at 0.1 to 0.05 for primary settings"
        ),
        "joint_fine_truth_assumption": False,
        "budget_rationale": (
            "primary: 600 forecaster epochs and 400 densifier epochs, based on "
            "prior 400-epoch repository runs; assess convergence from curves"
        ),
        "seeds": {str(seed): seed_plan(seed) for seed in config.seeds},
    }
    write_json(output / "config.json", manifest)
    root = Path(__file__).resolve().parents[2]
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for folder in ("src", "scripts", "tests", "configs", "docs"):
            for path in (root / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.add(path, arcname=path.relative_to(root))
        for name in ("pyproject.toml", "uv.lock", "README.md"):
            archive.add(root / name, arcname=name)
    (output / "git_diff.patch").write_text(
        subprocess.check_output(["git", "diff"], cwd=root, text=True)
    )
    (output / "git_head.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
    )
    if config.workers > 1 and len(config.seeds) > 1 and config.device == "cpu":
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(config.seeds)),
            mp_context=get_context("spawn"),
        ) as executor:
            jobs = [
                executor.submit(train_seed, config, seed, output / f"seed_{seed}")
                for seed in config.seeds
            ]
            for job in jobs:
                job.result()
    else:
        for seed in config.seeds:
            logging.info("starting paired seed %d", seed)
            train_seed(config, seed, output / f"seed_{seed}")
    hashes = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.glob("seed_*/*.pt"))
    }
    write_json(
        output / "FROZEN.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "checkpoint_sha256": hashes,
            "config_sha256": hashlib.sha256(
                (output / "config.json").read_bytes()
            ).hexdigest(),
            "test_evaluated": False,
        },
    )


def evaluate_run(output):
    output = Path(output)
    frozen = json.loads((output / "FROZEN.json").read_text())
    if (
        hashlib.sha256((output / "config.json").read_bytes()).hexdigest()
        != frozen["config_sha256"]
    ):
        raise ValueError("Configuration changed after freezing")
    for name, expected in frozen["checkpoint_sha256"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Checkpoint changed after freezing")
    manifest = json.loads((output / "config.json").read_text())
    config = WaveformConfig(
        **{**manifest["experiment"], "seeds": tuple(manifest["experiment"]["seeds"])}
    )
    if manifest["architecture"] != ARCHITECTURE:
        raise ValueError("Architecture differs from frozen configuration")
    for name, densifier in (
        ("forecaster_training", False),
        ("densifier_training", True),
    ):
        current = json.loads(
            json.dumps(asdict(training_config(config, densifier=densifier)))
        )
        if current != manifest[name]:
            raise ValueError(
                "Training/sampling schedule differs from frozen configuration"
            )
    if manifest["seeds"] != {str(seed): seed_plan(seed) for seed in config.seeds}:
        raise ValueError("Seed allocation differs from frozen configuration")
    torch.set_num_threads(config.threads)
    if any((output / f"seed_{seed}" / "samples.npz").exists() for seed in config.seeds):
        raise FileExistsError("Test evaluation already started; refusing to overwrite")
    if config.workers > 1 and len(config.seeds) > 1 and config.device == "cpu":
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(config.seeds)),
            mp_context=get_context("spawn"),
        ) as executor:
            jobs = {
                seed: executor.submit(
                    evaluate_seed, config, seed, output / f"seed_{seed}"
                )
                for seed in config.seeds
            }
            scores = {str(seed): job.result() for seed, job in jobs.items()}
    else:
        scores = {
            str(seed): evaluate_seed(config, seed, output / f"seed_{seed}")
            for seed in config.seeds
        }
    write_json(output / "scores.json", scores)
    write_json(
        output / "EVALUATED.json", {"finished_utc": datetime.now(UTC).isoformat()}
    )
    return scores
