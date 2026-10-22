"""Train MTSCI on Brownian motion and compare its CRPS with two baselines.

Run from the repository root:

    uv run --extra research python scripts/run_mtsci_brownian.py --device mps

Same setup as scripts/run_sssd_brownian.py: each series is one window of 65
points of standard Brownian motion at spacing dt = 0.01, every other point is
hidden (2 dt -> dt), and CRPS is averaged over hidden test points. The exact
GP posterior and linear interpolation are scored on the same test windows and
masks. MTSCI is trained twice from the same initialisation: with the
reference intra-consistency weight of 5, and with the weight set to zero (plain
noise MSE), to isolate the effect of the consistency term.
"""

import argparse
import json
import logging
import time
from dataclasses import replace

import torch

from densiffusion.architectures.mtsci import MTSCIDenoiser, MTSCIObjective
from densiffusion.experiments import run_experiment
from densiffusion.models import ImputationModel
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig

from run_sssd_brownian import DT, N_SAMPLES, baseline_scores, brownian_splits


def train_and_score(weight: float, config: TrainingConfig, splits, seed: int):
    (train, _), (validation, _), (test, _) = splits
    torch.manual_seed(seed)
    denoiser = MTSCIDenoiser(1)

    def sampler_factory(model, config):
        # Larger sampling batches only change speed, not the draws' distribution.
        return DDPMSampler(model, replace(config, batch_size=1024))

    start = time.perf_counter()
    scores = run_experiment(
        ImputationModel(denoiser),
        train,
        validation,
        test,
        sampler_factory=sampler_factory,
        objective=MTSCIObjective(denoiser, consistency_weight=weight),
        config=config,
        n_samples=N_SAMPLES,
        dt=DT,
        seed=seed,
    )
    return {"crps": scores["crps"], "seconds": time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-series", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=float, nargs="+", default=[5.0, 0.0])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    splits = brownian_splits(args.n_series, args.seed)
    test, test_times = splits[2]
    results = baseline_scores(test, test_times, args.seed)
    print(json.dumps(results, indent=2))

    # Reference point-missingness schedule: 50 quadratic steps up to 0.2.
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=64,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=0.2,
        schedule="quadratic",
        device=args.device,
    )
    for weight in args.weights:
        results[f"mtsci_consistency_{weight:g}"] = train_and_score(
            weight, config, splits, args.seed
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
