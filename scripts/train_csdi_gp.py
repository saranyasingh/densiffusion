"""Small held-out 2*dt -> dt CSDI experiment on independent GP trajectories.

Run from the repository root with:

    uv run python scripts/train_csdi_gp.py --output outputs/csdi-gp

This is a training/sampling smoke experiment, not a scale-transfer evaluation.
It never generates ground truth finer than dt and performs no normalization.
"""

import argparse
import json
import logging
import platform
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.conditioning import prepare_targets
from densiffusion.data import make_windows, split_data
from densiffusion.experiments import run_experiment
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import generate
from densiffusion.metrics import evaluate
from densiffusion.models import ImputationModel
from densiffusion.objectives import noise_mse
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig

GP_PARAMS = {"kernel": "exponential", "mean": 0.0, "variance": 1.0, "lengthscale": 1.0}
SPLIT_FRACTIONS = {"train_fraction": 0.7, "validation_fraction": 0.15}


class _RecordingDDPMSampler(DDPMSampler):
    """Retain exactly the draws scored by the shared experiment runner."""

    def __init__(self, model, config, batches):
        super().__init__(model, config)
        self.batches = batches

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        self.batches.append(samples)
        return samples


def make_dataset(args):
    """Split independent trajectories first, then window each split separately."""
    times = np.arange(args.n_steps) * args.dt
    values = generate(
        n_series=args.n_series,
        n_steps=args.n_steps,
        dt=args.dt,
        rng=np.random.default_rng(args.seed),
        **GP_PARAMS,
    )
    splits = split_data(values, times=times, axis="series", **SPLIT_FRACTIONS)
    windows = [
        make_windows(
            part,
            times=part_times,
            window_length=args.window_length,
            stride=args.stride,
        )
        for part, part_times in splits
    ]
    # Contiguous IDs identify the independent paths represented in each split.
    boundaries = np.cumsum([0, *(len(part) for part, _ in splits)])
    split_info = {
        name: {
            "trajectory_ids": list(range(int(start), int(stop))),
            "n_windows": len(window_values),
        }
        for name, start, stop, (window_values, _) in zip(
            ("train", "validation", "test"),
            boundaries[:-1],
            boundaries[1:],
            windows,
            strict=True,
        )
    }
    return windows, split_info


def linear_interpolation(target, mask, times, n_samples):
    """Repeat the deterministic interpolant as a point-mass ensemble."""
    interpolated = np.where(mask, 0.0, target)
    for series in range(len(target)):
        known = ~mask[series, :, 0]
        interpolated[series, :, 0] = np.interp(
            times[series], times[series, known], target[series, known, 0]
        )
    return np.repeat(interpolated[:, None], n_samples, axis=1)


def run_example(args):
    if args.window_length < 3 or args.window_length % 2 != 1:
        raise ValueError("window_length must be odd and at least 3")
    if args.n_steps < args.window_length:
        raise ValueError("n_steps must be at least window_length")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if not np.isfinite(args.dt) or args.dt <= 0:
        raise ValueError("dt must be finite and positive")
    if args.n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not 0 <= args.seed < 2**63 - 1:
        raise ValueError("seed must be in [0, 2**63 - 1)")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)  # Separate seed for network initialization.
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps,
        beta_start=1e-4,
        beta_end=args.beta_end,
        device=args.device,
    )
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
    model = ImputationModel(CSDIDenoiser(**architecture))
    windows, split_info = make_dataset(args)
    (train, _), (validation, _), (test, test_times) = windows
    test, mask = prepare_targets(test)
    terminal_alpha_bar = float(
        np.prod(
            1 - np.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
        )
    )
    manifest = {
        "task": "held_out_refinement_at_training_resolution",
        "target_dt": args.dt,
        "observed_dt": 2 * args.dt,
        "scale_transfer_evaluated": False,
        "normalization": "none (known zero-mean, unit-variance prior)",
        "gp": GP_PARAMS,
        "dataset": {"n_series": args.n_series, "n_steps": args.n_steps},
        "window_length": args.window_length,
        "stride": args.stride,
        "split_axis": "series_before_windowing",
        "splits": split_info,
        "hidden_mask": "odd interior indices; True means hidden",
        "architecture": architecture,
        "training": asdict(config),
        "schedule": "linear",
        "terminal_alpha_bar": terminal_alpha_bar,
        "objective": "noise_mse",
        "sampler": "DDPMSampler (fixed posterior variance)",
        "n_samples": args.n_samples,
        "seed": args.seed,
        "gp_posterior_seed": args.seed + 1,
        "threads": args.threads,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }
    # A new directory protects every output, including the selected checkpoint.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Training resolution: {2 * args.dt:g} -> {args.dt:g}; "
        f"terminal alpha_bar={terminal_alpha_bar:.3g}",
        flush=True,
    )
    sample_batches = []
    csdi_scores = run_experiment(
        model,
        train,
        validation,
        test,
        sampler_factory=lambda model, config: _RecordingDDPMSampler(
            model, config, sample_batches
        ),
        config=config,
        objective=noise_mse,
        n_samples=args.n_samples,
        dt=args.dt,
        seed=args.seed,
        checkpoint=args.output / "model.pt",
        test_times=test_times,
    )
    csdi_samples = np.concatenate(sample_batches)
    interpolation_samples = linear_interpolation(test, mask, test_times, args.n_samples)
    gp_samples = impute(
        test,
        GPPosteriorSampler(partial(gp_posterior, **GP_PARAMS)),
        factor=1,
        mask=mask,
        times=test_times,
        n_samples=args.n_samples,
        dt=args.dt,
        seed=args.seed + 1,
    )
    scores = {
        "csdi": csdi_scores,
        "linear_interpolation": evaluate(interpolation_samples, test, mask=mask),
        "exact_gp_posterior": evaluate(gp_samples, test, mask=mask),
    }
    for samples in (csdi_samples, interpolation_samples, gp_samples):
        if not np.isfinite(samples).all():
            raise FloatingPointError("Non-finite held-out samples")
        if not np.where(mask[:, None], True, samples == test[:, None]).all():
            raise AssertionError("Observed values were changed")
    with (args.output / "samples.npz").open("xb") as output:
        np.savez_compressed(
            output,
            target=test,
            times=test_times,
            mask=mask,
            csdi=csdi_samples,
            linear_interpolation=interpolation_samples,
            exact_gp_posterior=gp_samples,
        )
    (args.output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print("Held-out hidden-point ensemble CRPS (lower is better):")
    for name, result in scores.items():
        print(f"  {name:22s} {result['crps']:.6f}")
    print(f"Saved config, selected checkpoint, scores, and samples to {args.output}")
    print("Short smoke run only; these scores do not establish model convergence.")
    return scores


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/csdi-gp"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--n-series", type=int, default=48)
    parser.add_argument("--n-steps", type=int, default=33)
    parser.add_argument("--window-length", type=int, default=17)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.1)
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_example(parse_args())


if __name__ == "__main__":
    main()
