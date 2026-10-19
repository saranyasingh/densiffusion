"""Compare two stronger CSDI training plans using validation-only selection.

uv run python scripts/compare_csdi_training.py --output outputs/csdi-gp-better

Requires the original example's artifacts at --baseline. Train, validation, and
fresh test paths use independent random streams on the same dt=0.1 grid. Test
generation and evaluation happen only after selection.json has been written.
"""

import argparse
import hashlib
import json
import logging
import platform
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.conditioning import prepare_targets
from densiffusion.data import make_windows
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import generate
from densiffusion.metrics import evaluate
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

GP = dict(kernel="exponential", mean=0.0, variance=1.0, lengthscale=1.0)
DT, WINDOW, STRIDE, PATH_LENGTH = 0.1, 17, 16, 33
DATA_SEEDS = dict(train=22001, validation=22002, test=22003)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def partition(name, n_series):
    """Create independent paths within a single partition, then window them."""
    values = generate(
        n_series=n_series,
        n_steps=PATH_LENGTH,
        dt=DT,
        rng=np.random.default_rng(DATA_SEEDS[name]),
        **GP,
    )
    return make_windows(
        values,
        times=np.arange(PATH_LENGTH) * DT,
        window_length=WINDOW,
        stride=STRIDE,
    )


def sample_model(model, config, target, times, *, draws, seed):
    target, mask = prepare_targets(target)
    samples = impute(
        target,
        DDPMSampler(model, config),
        factor=1,
        mask=mask,
        times=times,
        n_samples=draws,
        dt=DT,
        seed=seed,
    )
    return samples, evaluate(samples, target, mask=mask)["crps"]


def model_from_checkpoint(path, architecture, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ImputationModel(CSDIDenoiser(**architecture)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


def evaluate_comparison(models, target, times, *, draws, seed, output):
    target, mask = prepare_targets(target)
    ensembles, scores = {}, {}
    for name, (model, config) in models.items():
        samples, score = sample_model(
            model, config, target, times, draws=draws, seed=seed
        )
        ensembles[name], scores[name] = samples, score

    interpolated = target.copy()
    interpolated[:, 1::2] = (target[:, :-1:2] + target[:, 2::2]) / 2
    ensembles["linear_interpolation"] = np.repeat(interpolated[:, None], draws, axis=1)
    ensembles["exact_gp_posterior"] = impute(
        target,
        GPPosteriorSampler(partial(gp_posterior, **GP)),
        factor=1,
        mask=mask,
        times=times,
        n_samples=draws,
        dt=DT,
        seed=seed + 1,
    )
    for name, samples in ensembles.items():
        assert np.isfinite(samples).all()
        assert np.where(mask[:, None], True, samples == target[:, None]).all()
        scores[name] = evaluate(samples, target, mask=mask)["crps"]
    np.savez_compressed(output, target=target, times=times, mask=mask, **ensembles)
    return scores, ensembles


def paired_improvement(ensembles, target, *, seed):
    """Bootstrap whole independent trajectories (two adjacent windows each)."""
    _, mask = prepare_targets(target)
    differences = []
    for start in range(0, len(target), 2):
        part = slice(start, start + 2)
        old = evaluate(ensembles["original_csdi"][part], target[part], mask=mask[part])
        new = evaluate(ensembles["improved_csdi"][part], target[part], mask=mask[part])
        differences.append(old["crps"] - new["crps"])
    differences = np.array(differences)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(differences), size=(10000, len(differences)))
    interval = np.quantile(differences[indices].mean(1), [0.025, 0.975])
    return {
        "original_minus_improved": float(differences.mean()),
        "paired_trajectory_bootstrap_95pct": interval.tolist(),
        "independent_test_trajectories": len(differences),
        "note": (
            "Conditional on these trained models and sampled ensembles; "
            "not training-seed uncertainty."
        ),
    }


def run(args):
    baseline_manifest = json.loads((args.baseline / "config.json").read_text())
    if baseline_manifest["target_dt"] != DT or baseline_manifest["gp"] != GP:
        raise ValueError("Baseline must use the same GP and target spacing")
    if not (args.baseline / "model.pt").is_file():
        raise FileNotFoundError("Run the original example first")
    torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.output / "training.log"),
            logging.StreamHandler(),
        ],
    )
    common = dict(
        batch_size=64,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=0.5,
        schedule="quadratic",
        weight_decay=1e-6,
        lr_decay_fractions=(0.75, 0.9),
        validation_steps="all",
        validation_interval=10,
        device=args.device,
    )
    small = baseline_manifest["architecture"]
    large = dict(
        n_channels=1,
        hidden_channels=64,
        residual_layers=4,
        attention_heads=8,
        diffusion_embedding_dim=128,
        time_embedding_dim=128,
        feature_embedding_dim=16,
        feedforward_dim=64,
        dropout=0.1,
    )
    candidates = {
        "small_longer": (small, TrainingConfig(epochs=args.small_epochs, **common)),
        "reference_size": (large, TrainingConfig(epochs=args.large_epochs, **common)),
    }
    code_paths = [
        Path(__file__),
        Path("src/densiffusion/architectures/csdi.py"),
        Path("src/densiffusion/training.py"),
        Path("src/densiffusion/samplers.py"),
    ]
    plan = {
        "task": "held_out_0.2_to_0.1_refinement; no finer ground truth",
        "gp": GP,
        "dt": DT,
        "data_seeds": DATA_SEEDS,
        "paths": {
            "train": args.train_series,
            "validation": args.validation_series,
            "test": args.test_series,
        },
        "path_length": PATH_LENGTH,
        "window_length": WINDOW,
        "stride": STRIDE,
        "normalization": "none; population mean 0 and variance 1",
        "seed": args.seed,
        "draws": args.samples,
        "threads": args.threads,
        "selection": (
            "best all-step validation-MSE checkpoint per candidate, "
            "then lowest validation ensemble CRPS"
        ),
        "baseline": str(args.baseline.resolve()),
        "candidates": {
            name: {"architecture": arch, "training": asdict(config)}
            for name, (arch, config) in candidates.items()
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths
        },
    }
    write_json(args.output / "plan.json", plan)
    train, _ = partition("train", args.train_series)
    validation, val_times = partition("validation", args.validation_series)
    results = {}
    for name, (architecture, config) in candidates.items():
        logging.info("Starting candidate %s", name)
        folder = args.output / name
        folder.mkdir()
        torch.manual_seed(args.seed)
        model = ImputationModel(CSDIDenoiser(**architecture))
        started = time.perf_counter()
        train_densifier(
            model,
            train,
            validation,
            config=config,
            dt=DT,
            seed=args.seed,
            checkpoint=folder / "model.pt",
        )
        train_seconds = time.perf_counter() - started
        samples, score = sample_model(
            model,
            config,
            validation,
            val_times,
            draws=args.samples,
            seed=31001,
        )
        state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        results[name] = {
            "validation_crps": score,
            "selected_epoch": state["epoch"],
            "validation_noise_mse": state["validation_loss"],
            "training_seconds": train_seconds,
            "optimizer_updates": config.epochs
            * int(np.ceil(len(train) / config.batch_size)),
        }
        np.savez_compressed(folder / "validation_samples.npz", samples=samples)
        write_json(folder / "result.json", results[name])
        logging.info("Candidate %s: %s", name, results[name])

    selected = min(results, key=lambda name: results[name]["validation_crps"])
    selection = {"selected": selected, "candidates": results, "test_evaluated": False}
    # Freeze and persist the decision before generating/reading any test targets.
    write_json(args.output / "selection.json", selection)
    logging.info("Frozen selection: %s. Starting test evaluation.", selected)
    architecture, config = candidates[selected]
    improved, _ = model_from_checkpoint(
        args.output / selected / "model.pt", architecture, args.device
    )
    original, original_state = model_from_checkpoint(
        args.baseline / "model.pt", small, args.device
    )
    original_config = replace(
        TrainingConfig(**original_state["config"]), batch_size=64, device=args.device
    )
    models = {
        "original_csdi": (original, original_config),
        "improved_csdi": (improved, config),
    }

    test, test_times = partition("test", args.test_series)
    fresh_scores, ensembles = evaluate_comparison(
        models,
        test,
        test_times,
        draws=args.samples,
        seed=31002,
        output=args.output / "fresh_test_samples.npz",
    )
    uncertainty = paired_improvement(ensembles, test, seed=31003)
    with np.load(args.baseline / "samples.npz") as previous:
        legacy_target, legacy_times = previous["target"], previous["times"]
        legacy_draws = previous["csdi"].shape[1]
    legacy_scores, _ = evaluate_comparison(
        {"improved_csdi": (improved, config)},
        legacy_target,
        legacy_times,
        draws=legacy_draws,
        seed=31004,
        output=args.output / "original_test_samples.npz",
    )
    historical = json.loads((args.baseline / "scores.json").read_text())
    report = {
        "selected": selected,
        "fresh_test_crps": fresh_scores,
        "fresh_test_improvement": uncertainty,
        "original_test_crps": legacy_scores,
        "original_test_historical_crps": historical,
        "note": (
            "Candidates selected using validation only; original test has been "
            "examined previously. Fresh test is the primary comparison."
        ),
    }
    write_json(args.output / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/csdi-gp-better"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("outputs/csdi-gp-verified")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-epochs", type=int, default=400)
    parser.add_argument("--large-epochs", type=int, default=200)
    parser.add_argument("--train-series", type=int, default=512)
    parser.add_argument("--validation-series", type=int, default=64)
    parser.add_argument("--test-series", type=int, default=128)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
