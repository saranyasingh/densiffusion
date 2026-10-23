"""Small original-grid forecasting experiment for the Densiffusion pipeline.

Run from the repository root with:

    uv run python scripts/train_csdi_forecast.py --output outputs/csdi-forecast

Independent GP trajectories are split before windowing. Ground truth is generated
only at the original spacing dt. Fine-grid targets are synthetic augmentations
from a frozen CSDI densifier or a deterministic interpolation baseline.
"""

import argparse
import json
import logging
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.conditioning import prepare_observations
from densiffusion.data import make_forecast_windows, split_data
from densiffusion.gp.sampling import generate
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel
from densiffusion.pipeline import forecast
from densiffusion.samplers import DDPMForecastSampler, DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier, train_forecaster

GP_PARAMS = {"kernel": "exponential", "mean": 0.0, "variance": 1.0, "lengthscale": 1.0}
SPLIT_FRACTIONS = {"train_fraction": 0.7, "validation_fraction": 0.15}
METHODS = ("densiffusion", "direct_csdi", "linear_densification")


class LinearDensifier:
    """Deterministic interpolation, repeated as a point-mass ensemble."""

    def sample(self, observed, *, factor, n_samples, dt, rng, mask=None, times=None):
        values, hidden, grid = prepare_observations(
            observed, factor=factor, dt=dt, mask=mask, times=times
        )
        for row in range(len(values)):
            for channel in range(values.shape[2]):
                known = ~hidden[row, :, channel]
                values[row, :, channel] = np.interp(
                    grid[row], grid[row, known], values[row, known, channel]
                )
        return np.repeat(values[:, None], n_samples, axis=1)


def make_dataset(args):
    """Return original-grid windows and auditable, disjoint trajectory IDs."""
    values = generate(
        n_series=args.n_series,
        n_steps=args.n_steps,
        dt=args.dt,
        rng=np.random.default_rng(args.seed),
        **GP_PARAMS,
    )
    parts = split_data(values, axis="series", **SPLIT_FRACTIONS)
    windows = [
        make_forecast_windows(
            part,
            context_length=args.context_length,
            horizon=args.horizon,
            stride=args.stride,
        )
        for part in parts
    ]
    boundaries = np.cumsum([0, *(len(part) for part in parts)])
    starts = list(
        range(0, args.n_steps - args.context_length - args.horizon + 1, args.stride)
    )
    split_info = {
        name: {
            "trajectory_ids": list(range(int(start), int(stop))),
            "n_windows": len(history),
            "window_trajectory_ids": [
                trajectory
                for trajectory in range(int(start), int(stop))
                for _ in starts
            ],
            "window_start_indices": starts * int(stop - start),
        }
        for name, start, stop, (history, _) in zip(
            ("train", "validation", "test"),
            boundaries[:-1],
            boundaries[1:],
            windows,
            strict=True,
        )
    }
    return windows, split_info


def _timed(function, *args, device, **kwargs):
    if device == "mps":
        torch.mps.synchronize()
    start = perf_counter()
    result = function(*args, **kwargs)
    if device == "mps":
        torch.mps.synchronize()
    return result, perf_counter() - start


def run_example(args):
    for name in (
        "threads",
        "n_histories",
        "n_futures",
        "n_augmentations",
        "factor",
        "context_length",
        "horizon",
        "stride",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.context_length < 2:
        raise ValueError("context_length must be at least 2 for history densification")
    if args.n_steps < args.context_length + args.horizon:
        raise ValueError("n_steps must contain a complete history and future")
    if not np.isfinite(args.dt) or args.dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not 0 <= args.seed < 2**63 - 100:
        raise ValueError("seed must be in [0, 2**63 - 100)")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")

    torch.set_num_threads(args.threads)
    common_training = {
        "batch_size": args.batch_size,
        "diffusion_steps": args.diffusion_steps,
        "beta_start": 1e-4,
        "beta_end": args.beta_end,
        "device": args.device,
    }
    densifier_config = TrainingConfig(epochs=args.densifier_epochs, **common_training)
    forecast_config = TrainingConfig(epochs=args.epochs, **common_training)
    architecture = {
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
    windows, split_info = make_dataset(args)
    train, validation, (test_history, target) = windows
    seeds = {
        "data": args.seed,
        "densifier_initialization": args.seed + 1,
        "densifier_training": args.seed + 2,
        "forecaster_initialization": args.seed + 3,
        "forecaster_training": args.seed + 4,
        "test_sampling": {
            method: args.seed + 10 + index for index, method in enumerate(METHODS)
        },
    }
    manifest = {
        "task": "original_grid_forecasting_with_synthetic_densification",
        "original_dt": args.dt,
        "fine_dt": args.dt / args.factor,
        "factor": args.factor,
        "densifier_training": {
            "observed_dt": 2 * args.dt,
            "target_dt": args.dt,
            "mask": "alternating interior points; even-length windows are cropped",
            "config": asdict(densifier_config),
        },
        "forecaster_training": asdict(forecast_config),
        "architecture": architecture,
        "context_length": args.context_length,
        "horizon": args.horizon,
        "stride": args.stride,
        "n_augmentations": args.n_augmentations,
        "n_histories": args.n_histories,
        "n_futures": args.n_futures,
        "draws_per_method": args.n_histories * args.n_futures,
        "dataset": {"n_series": args.n_series, "n_steps": args.n_steps},
        "gp": GP_PARAMS,
        "split_axis": "series_before_windowing",
        "splits": split_info,
        "normalization": "none (known zero-mean, unit-variance prior)",
        "finer_ground_truth_generated": False,
        "test_targets_used_for_training_or_selection": False,
        "augmentation": (
            "fixed synthetic draws; history from past only; future from last "
            "observed history value plus future, dropping the shared anchor"
        ),
        "conditioning": "all dense-history values, including original observations",
        "comparison": (
            "same forecaster architecture, epochs, augmentations and scored draw "
            "count; sequence lengths and compute differ; no equal-FLOP claim"
        ),
        "seeds": seeds,
        "threads": args.threads,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def new_model(seed):
        torch.manual_seed(seed)
        return ImputationModel(CSDIDenoiser(**architecture))

    densifier_model = new_model(seeds["densifier_initialization"])
    _, densifier_seconds = _timed(
        train_densifier,
        densifier_model,
        np.concatenate(train, axis=1),
        np.concatenate(validation, axis=1),
        config=densifier_config,
        dt=args.dt,
        seed=seeds["densifier_training"],
        checkpoint=args.output / "densifier.pt",
        device=args.device,
    )
    densifier_model.requires_grad_(False).eval()
    learned_densifier = DDPMSampler(densifier_model, densifier_config)
    linear_densifier = LinearDensifier()
    samplers = {
        "densiffusion": learned_densifier,
        "direct_csdi": linear_densifier,
        "linear_densification": linear_densifier,
    }
    trained = {}
    timings = {"densifier_training_seconds": densifier_seconds, "methods": {}}
    # Every training and checkpoint-selection step precedes test sampling.
    for method in METHODS:
        model = new_model(seeds["forecaster_initialization"])
        factor = 1 if method == "direct_csdi" else args.factor
        _, seconds = _timed(
            train_forecaster,
            model,
            samplers[method],
            train,
            validation,
            config=forecast_config,
            factor=factor,
            dt=args.dt,
            seed=seeds["forecaster_training"],
            checkpoint=args.output / f"{method}_forecaster.pt",
            n_augmentations=args.n_augmentations,
            device=args.device,
        )
        trained[method] = DDPMForecastSampler(model, forecast_config)
        timings["methods"][method] = {
            "augmentation_and_forecaster_training_seconds": seconds,
            "total_training_seconds": seconds
            + (densifier_seconds if method == "densiffusion" else 0),
        }

    ensembles, scores = {}, {}
    for method in METHODS:
        direct = method == "direct_csdi"
        samples, seconds = _timed(
            forecast,
            test_history,
            samplers[method],
            trained[method],
            horizon=args.horizon,
            factor=1 if direct else args.factor,
            n_histories=1 if direct else args.n_histories,
            n_futures=args.n_histories * args.n_futures if direct else args.n_futures,
            dt=args.dt,
            seed=seeds["test_sampling"][method],
            device=args.device,
        )
        ensembles[method] = samples
        scores[method] = evaluate_forecast(samples, target)
        timings["methods"][method]["forecast_seconds"] = seconds

    starts = np.asarray(split_info["test"]["window_start_indices"])
    with (args.output / "samples.npz").open("xb") as output:
        np.savez_compressed(
            output,
            history=test_history,
            target=target,
            history_times=(starts[:, None] + np.arange(args.context_length)) * args.dt,
            future_times=(
                starts[:, None] + args.context_length + np.arange(args.horizon)
            )
            * args.dt,
            **ensembles,
        )
    (args.output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    (args.output / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    print("Original-grid held-out forecasting scores:")
    for method, result in scores.items():
        print(
            f"  {method:22s} CRPS={result['crps']:.6f} "
            f"MSE={result['mse']:.6f} coverage90={result['coverage_90']:.3f} "
            f"width90={result['interval_width_90']:.6f}"
        )
    print(
        f"Saved configuration, four checkpoints, exact samples and costs: {args.output}"
    )
    print(
        "Smoke experiment; these results do not establish convergence or improvement."
    )
    return scores


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/csdi-forecast"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--densifier-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--n-histories", type=int, default=2)
    parser.add_argument("--n-futures", type=int, default=4)
    parser.add_argument("--n-augmentations", type=int, default=1)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--n-series", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--context-length", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.1)
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_example(parse_args())


if __name__ == "__main__":
    main()
