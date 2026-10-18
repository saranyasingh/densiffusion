"""Train SSSD on Brownian motion and compare its CRPS with two baselines.

Run from the repository root (about 15 minutes on an Apple M-series GPU):

    uv run --extra research python scripts/run_sssd_brownian.py --device mps

Each series is one window of 65 points of standard Brownian motion at spacing
dt = 0.01. Every other point is hidden and imputed from its neighbours
(2 dt -> dt), and CRPS is averaged over hidden test points. Baselines on the
same test windows and masks:

- exact GP posterior: the best achievable distribution, sigma / sqrt(pi) in
  expectation, where sigma = sqrt(dt / 2) is the midpoint's posterior std;
- linear interpolation: one deterministic draw, so its CRPS is the mean
  absolute error, sigma * sqrt(2 / pi) in expectation.
"""

import argparse
import json
import logging
import math
import time
from dataclasses import replace
from functools import partial

import numpy as np
import torch

from densiffusion.architectures.sssd import SSSDDenoiser
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.conditioning import prepare_targets
from densiffusion.data import split_data
from densiffusion.experiments import run_experiment
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import sample_gp
from densiffusion.metrics import evaluate
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig

DT = 0.01
LENGTH = 65
N_SAMPLES = 100


def brownian_splits(n_series: int, seed: int):
    """Return ((values, times) for train, validation, test), split by series."""
    times = np.arange(LENGTH) * DT
    paths = sample_gp(times, "brownian", rng=seed, n_samples=n_series, variance=1.0)
    values = paths[:, :, None]
    all_times = np.broadcast_to(times, paths.shape).copy()
    return split_data(
        values, times=all_times, train_fraction=0.8, validation_fraction=0.1
    )


def baseline_scores(test: np.ndarray, test_times: np.ndarray, seed: int) -> dict:
    target, mask = prepare_targets(test)
    sampler = GPPosteriorSampler(partial(gp_posterior, kernel="brownian", variance=1))
    draws = impute(
        target[:, ::2],
        sampler,
        factor=2,
        n_samples=N_SAMPLES,
        dt=2 * DT,
        seed=seed,
        times=test_times[:, ::2],
    )
    interpolated = target.copy()
    interpolated[:, 1::2] = (target[:, :-2:2] + target[:, 2::2]) / 2
    sigma = math.sqrt(DT / 2)
    return {
        "exact_gp_posterior": evaluate(draws, target, mask=mask)["crps"],
        "linear_interpolation": evaluate(interpolated[:, None], target, mask=mask)[
            "crps"
        ],
        "theory_exact_gp_posterior": sigma / math.sqrt(math.pi),
        "theory_linear_interpolation": sigma * math.sqrt(2 / math.pi),
    }


def train_and_score(model, config: TrainingConfig, splits, seed: int) -> dict:
    (train, _), (validation, _), (test, _) = splits

    def sampler_factory(model, config):
        # Larger sampling batches only change speed, not the draws' distribution.
        return DDPMSampler(model, replace(config, batch_size=1024))

    start = time.perf_counter()
    scores = run_experiment(
        model,
        train,
        validation,
        test,
        sampler_factory=sampler_factory,
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    splits = brownian_splits(args.n_series, args.seed)
    test, test_times = splits[2]
    results = baseline_scores(test, test_times, args.seed)
    print(json.dumps(results, indent=2))

    torch.manual_seed(args.seed)
    model = ImputationModel(SSSDDenoiser(1))
    # SSSD paper schedule: 200 linear steps from 1e-4 to 0.02.
    config = TrainingConfig(
        epochs=args.epochs,
        diffusion_steps=200,
        beta_start=1e-4,
        beta_end=0.02,
        device=args.device,
    )
    results["sssd"] = train_and_score(model, config, splits, args.seed)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
