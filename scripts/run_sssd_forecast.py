"""Does forecasting from a densified past help? SSSD on fBM and Brownian motion.

Run from the repository root (roughly 45 minutes per process on an Apple GPU):

    uv run --extra research python scripts/run_sssd_forecast.py --device mps

Each series has CONTEXT past points and HORIZON future points at spacing DT,
starting at t = 0. Training only ever sees that spacing. For each process:

1. Train an SSSD imputer to fill alternate points (2 dt -> dt).
2. Train three SSSD forecasters with identical settings: on the original past
   (direct), on a past densified once (dt / 2), and on a past densified twice
   (dt / 4). Their training windows (past and future) are densified by the
   imputer, keeping every measured value.
3. At test time, densify only the past, draw N_HISTORIES completions and
   N_FUTURES forecasts per completion, and score the forecasts at the original
   future times with CRPS.

References: the exact GP forecast (the best achievable CRPS), the last observed
value, and a calibration check comparing the imputer's dt / 2 and dt / 4 fills
of the test past with the exact GP posterior. Test series are also simulated
on the dt / 4 grid; those finer values are used only for that check.
"""

import argparse
import json
import logging
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.sssd import SSSDDenoiser
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.contracts import Array, Densifier, Mask
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import sample_gp
from densiffusion.metrics import evaluate
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

DT = 0.01
CONTEXT = 32
HORIZON = 8
LEVELS = (0, 1, 2)
N_HISTORIES = 10
N_FUTURES = 10
PROCESSES = {
    "fbm": ("fractional_brownian", {"variance": 1.0, "hurst": 0.7}),
    "brownian": ("brownian", {"variance": 1.0}),
}


def future_mask(context: int, horizon: int, factor: int) -> Mask:
    """(time,) mask on a grid refined `factor` times; True after the past."""
    mask = np.zeros((context + horizon - 1) * factor + 1, dtype=bool)
    mask[(context - 1) * factor + 1 :] = True
    return mask


def densify(
    values: Array,
    sampler: Densifier,
    *,
    levels: int,
    n_samples: int,
    dt: float,
    seed: int,
) -> Array:
    """Halve the spacing `levels` times; return (batch, n_samples, time, channels).

    The first level draws n_samples completions; later levels refine each one
    once. dt is the spacing of `values`. Measured values are kept exactly.
    """
    batch, _, channels = values.shape
    dense = impute(values, sampler, factor=2, n_samples=n_samples, dt=dt, seed=seed)
    for level in range(1, levels):
        flat = dense.reshape(batch * n_samples, -1, channels)
        spacing = dt / 2**level
        dense = impute(flat, sampler, factor=2, dt=spacing, seed=seed + level)
        dense = dense.reshape(batch, n_samples, -1, channels)
    return dense


def dense_training_windows(
    values: Array, sampler: Densifier, *, levels: int, dt: float, seed: int
) -> Array:
    """Densify whole training windows (past and future) once each."""
    dense = densify(values, sampler, levels=levels, n_samples=1, dt=dt, seed=seed)
    return dense[:, 0]


def forecast(
    past: Array,
    densifier: Densifier,
    forecaster: Densifier,
    *,
    horizon: int,
    levels: int,
    n_histories: int,
    n_futures: int,
    dt: float,
    seed: int,
) -> Array:
    """Forecast from the past alone; return (batch, draws, horizon, channels).

    Only `past` reaches the densifier and forecaster, so future values cannot
    leak. With levels > 0 the past is densified n_histories times and each
    completion gets n_futures forecasts; draws are pooled and read back at the
    original future times.
    """
    batch, context, channels = past.shape
    factor = 2**levels
    if levels:
        histories = densify(
            past, densifier, levels=levels, n_samples=n_histories, dt=dt, seed=seed
        )
    else:
        histories = past[:, None]
    count = histories.shape[1]
    mask = future_mask(context, horizon, factor)
    window = np.zeros((batch * count, mask.size, channels))
    window[:, : histories.shape[2]] = histories.reshape(batch * count, -1, channels)
    hidden = np.broadcast_to(mask[None, :, None], window.shape)
    draws = impute(
        window,
        forecaster,
        factor=1,
        n_samples=n_futures,
        dt=dt / factor,
        seed=seed + levels + 1,
        mask=hidden,
    )
    draws = draws.reshape(batch, count * n_futures, mask.size, channels)
    return draws[:, :, ::factor][:, :, context:].copy()


def simulate(process: str, n_series: int, *, step: float, seed: int) -> Array:
    """(n_series, time, 1) paths from t = 0 at spacing `step`."""
    kernel, params = PROCESSES[process]
    length = round((CONTEXT + HORIZON - 1) * DT / step) + 1
    times = np.arange(length) * step
    return sample_gp(times, kernel, rng=seed, n_samples=n_series, **params)[..., None]


def gp_sampler(process: str) -> GPPosteriorSampler:
    kernel, params = PROCESSES[process]
    return GPPosteriorSampler(partial(gp_posterior, kernel=kernel, **params))


def crps_at_new_points(samples: Array, target: Array, factor: int) -> float:
    """CRPS at the points added by densification (not on the original grid)."""
    mask = np.ones(target.shape, dtype=bool)
    mask[:, ::factor] = False
    return evaluate(samples, target, mask=mask)["crps"]


def run(process: str, args: argparse.Namespace, output: Path) -> dict:
    start = time.perf_counter()
    rng = np.random.SeedSequence(args.seed).spawn(3)
    seeds = [int(s.generate_state(1)[0]) for s in rng]
    n_test = args.n_series // 10
    train = simulate(process, args.n_series * 8 // 10, step=DT, seed=seeds[0])
    validation = simulate(process, n_test, step=DT, seed=seeds[1])
    test_fine = simulate(process, n_test, step=DT / 4, seed=seeds[2])
    test = test_fine[:, ::4]
    past, future = test[:, :CONTEXT], test[:, CONTEXT:]
    times = np.broadcast_to(np.arange(CONTEXT + HORIZON) * DT, test.shape[:2])

    config = TrainingConfig(
        epochs=args.epochs,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=0.2,
        device=args.device,
    )

    def sampler_factory(model):
        # Larger sampling batches only change speed, not the draws' distribution.
        return DDPMSampler(model, replace(config, batch_size=1024))

    def fit(name, train_values, validation_values, *, spacing, mask=None):
        torch.manual_seed(args.seed)
        model = ImputationModel(SSSDDenoiser(1))
        train_densifier(
            model,
            train_values,
            validation_values,
            config=config,
            dt=spacing,
            seed=args.seed,
            checkpoint=output / f"{process}_{name}.pt",
            train_mask=mask,
            validation_mask=mask,
        )
        return sampler_factory(model)

    results = {}
    logging.info("[%s] training imputer", process)
    imputer = fit("imputer", train, validation, spacing=DT)

    oracle = impute(
        test,
        gp_sampler(process),
        factor=1,
        n_samples=N_HISTORIES * N_FUTURES,
        dt=DT,
        seed=args.seed,
        mask=np.broadcast_to(
            future_mask(CONTEXT, HORIZON, 1)[None, :, None], test.shape
        ),
        times=times,
    )
    results["exact_gp_forecast"] = evaluate(oracle[:, :, CONTEXT:], future)["crps"]
    last = np.repeat(past[:, None, -1:], HORIZON, axis=2)
    results["last_observed_value"] = evaluate(last, future)["crps"]

    for levels in LEVELS[1:]:
        factor = 2**levels
        target = test_fine[:, : (CONTEXT - 1) * 4 + 1 : 4 // factor]
        fills = densify(past, imputer, levels=levels, n_samples=100, dt=DT, seed=1)
        exact = impute(
            past,
            gp_sampler(process),
            factor=factor,
            n_samples=100,
            dt=DT,
            seed=1,
            times=times[:, :CONTEXT],
        )
        results[f"calibration_x{factor}"] = {
            "sssd": crps_at_new_points(fills, target, factor),
            "exact_gp": crps_at_new_points(exact, target, factor),
        }
        logging.info("[%s] %s", process, results[f"calibration_x{factor}"])

    for levels in LEVELS:
        factor = 2**levels
        name = "direct" if levels == 0 else f"densified_x{factor}"
        logging.info("[%s] training forecaster %s", process, name)
        if levels:
            train_values, validation_values = (
                dense_training_windows(
                    values, imputer, levels=levels, dt=DT, seed=args.seed + i
                )
                for i, values in enumerate((train, validation))
            )
        else:
            train_values, validation_values = train, validation
        forecaster = fit(
            name,
            train_values,
            validation_values,
            spacing=DT / factor,
            mask=future_mask(CONTEXT, HORIZON, factor),
        )
        samples = forecast(
            past,
            imputer,
            forecaster,
            horizon=HORIZON,
            levels=levels,
            n_histories=N_HISTORIES if levels else 1,
            n_futures=N_FUTURES if levels else N_HISTORIES * N_FUTURES,
            dt=DT,
            seed=args.seed,
        )
        results[f"sssd_{name}"] = evaluate(samples, future)["crps"]
        logging.info("[%s] %s CRPS %.5f", process, name, results[f"sssd_{name}"])

    results["seconds"] = time.perf_counter() - start
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--process", choices=[*PROCESSES, "both"], default="both")
    parser.add_argument("--n-series", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/sssd_forecast"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    processes = list(PROCESSES) if args.process == "both" else [args.process]
    for process in processes:
        results = run(process, args, args.output)
        path = args.output / f"{process}_results.json"
        path.write_text(json.dumps(results, indent=2))
        print(json.dumps({process: results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
