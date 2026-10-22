"""Forecast with the TimeAutoDiff imputer after imputing an entire dataset.

Run from the repository root, for example:

    uv run --extra research python scripts/forecast_timeautodiff.py data/gp.npz \
        --output outputs/timeautodiff_forecast

Each series is observed at every other step (spacing 2*dt). The script

1. trains the TimeAutoDiff imputer on alternate-mask windows, exactly as
   `run_timeautodiff.py` does, or loads it from OUTPUT/imputer.pt;
2. imputes every series over its whole length on the dt grid and saves the
   dense dataset to OUTPUT/imputed.npz;
3. fine-tunes a copy of the imputer to fill the last `horizon` points of
   `context + horizon` windows of the imputed train and validation series
   (OUTPUT/forecaster.pt), since alternate-mask training never hides a
   contiguous future block;
4. backtests on the test series: imputes their history only, forecasts the
   held-out tail, and scores CRPS against the true values, the exact GP
   forecast from the same coarse observations, and the last observed value;
5. forecasts `horizon` steps past the end of every imputed series and saves
   the draws to OUTPUT/forecast.npz.

Existing checkpoints are reused, so later runs skip training.
"""

import argparse
import json
import logging
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.timeautodiff import (
    TimeAutoDiffDenoiser,
    TimeAutoDiffEncoder,
)
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.data import make_windows, split_data
from densiffusion.gp.posterior import gp_posterior
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

GP_GENERATOR = "densiffusion.gp.sampling:generate"
SPLITS = dict(train_fraction=0.7, validation_fraction=0.15)


def build_model(channels, args):
    return ImputationModel(
        TimeAutoDiffDenoiser(
            channels=channels,
            hidden=args.hidden,
            layers=args.layers,
            diffusion_steps=args.diffusion_steps,
        ),
        encoder=TimeAutoDiffEncoder(
            channels=channels, hidden=args.hidden, layers=args.layers
        ),
    )


def fit(model, checkpoint, train, validation, *, config, dt, seed, mask=None):
    """Load the best weights from checkpoint, or train and save them there."""
    if checkpoint.exists():
        logging.info("Loading %s", checkpoint)
        state = torch.load(checkpoint, map_location=config.device)
        model.to(config.device).load_state_dict(state["model_state_dict"])
        return model.eval()
    return train_densifier(
        model,
        train,
        validation,
        config=config,
        dt=dt,
        seed=seed,
        checkpoint=checkpoint,
        train_mask=mask,
        validation_mask=mask,
    )


def impute_alternate(coarse, sampler, *, window, n_samples, dt, seed):
    """Fill the odd indices of odd-length series, window by window.

    Returns (batch, n_samples, time, channels). Windows start at even indices
    and share observed endpoints, so each hidden point is drawn exactly once.
    """
    length = coarse.shape[1]
    if length % 2 == 0 or window % 2 == 0 or length < window:
        raise ValueError("Need odd window and odd series length >= window")
    starts = list(range(0, length - window + 1, window - 1))
    if starts[-1] + window < length:
        starts.append(length - window)
    mask = np.arange(window) % 2 == 1
    dense = np.repeat(coarse[:, None], n_samples, axis=1)
    filled = 0
    for index, start in enumerate(starts):
        samples = impute(
            coarse[:, start : start + window],
            sampler,
            factor=1,
            mask=mask,
            n_samples=n_samples,
            dt=dt,
            seed=seed + index,
        )
        dense[:, :, filled : start + window] = samples[:, :, filled - start :]
        filled = start + window
    return dense


def forecast_tail(history, sampler, *, context, horizon, n_samples, dt, seed):
    """Return (batch, n_samples, horizon, channels) draws after each history."""
    batch, _, channels = history.shape
    window = np.concatenate(
        (history[:, -context:], np.zeros((batch, horizon, channels))), axis=1
    )
    samples = impute(
        window,
        sampler,
        factor=1,
        mask=np.arange(context + horizon) >= context,
        n_samples=n_samples,
        dt=dt,
        seed=seed,
    )
    return samples[:, :, context:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help=".npz from `densiffusion generate`")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-length", type=int, default=65)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--context", type=int, default=33)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--forecast-stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--forecaster-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--forecaster-learning-rate", type=float, default=5e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n-histories", type=int, default=10)
    parser.add_argument("--n-futures", type=int, default=10)
    parser.add_argument("--n-future-samples", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.horizon % 2:
        parser.error("--horizon must be even so test histories end on observations")

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.output / "training.log"),
            logging.StreamHandler(),
        ],
    )
    with np.load(args.data, allow_pickle=False) as dataset:
        values = dataset["values"]
        generation = json.loads(str(dataset["config"]))
    dt = float(generation["dt"])
    length = values.shape[1] - (1 - values.shape[1] % 2)
    values = values[:, :length]
    channels = values.shape[2]
    hidden = np.zeros(values.shape, dtype=bool)
    hidden[:, 1::2] = True
    coarse = np.where(hidden, 0.0, values)
    train, validation, test = split_data(values, **SPLITS)
    test_coarse = split_data(coarse, **SPLITS)[2]
    seeds = [int(s) for s in np.random.SeedSequence(args.seed).generate_state(6)]

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=args.device,
    )
    sample_config = replace(config, batch_size=args.sample_batch_size)

    torch.manual_seed(args.seed)
    imputer = fit(
        build_model(channels, args),
        args.output / "imputer.pt",
        *(
            make_windows(split, window_length=args.window_length, stride=args.stride)
            for split in (train, validation)
        ),
        config=config,
        dt=dt,
        seed=args.seed,
    )
    imputer_sampler = DDPMSampler(imputer, sample_config)

    logging.info("Imputing %d series of length %d", len(values), length)
    imputed = impute_alternate(
        coarse,
        imputer_sampler,
        window=args.window_length,
        n_samples=1,
        dt=dt,
        seed=seeds[0],
    )[:, 0]
    np.savez_compressed(
        args.output / "imputed.npz",
        values=imputed,
        times=np.arange(length) * dt,
        observed=~hidden[0, :, 0],
    )

    window = args.context + args.horizon
    forecast_mask = np.arange(window) >= args.context
    imputed_train, imputed_validation, _ = split_data(imputed, **SPLITS)
    forecaster = build_model(channels, args)
    forecaster.load_state_dict(imputer.state_dict())
    forecaster = fit(
        forecaster,
        args.output / "forecaster.pt",
        *(
            make_windows(split, window_length=window, stride=args.forecast_stride)
            for split in (imputed_train, imputed_validation)
        ),
        config=replace(
            config,
            epochs=args.forecaster_epochs,
            learning_rate=args.forecaster_learning_rate,
        ),
        dt=dt,
        seed=args.seed,
        mask=forecast_mask,
    )
    forecaster_sampler = DDPMSampler(forecaster, sample_config)

    history_length = length - args.horizon
    logging.info(
        "Backtesting %d test series: history %d, horizon %d",
        len(test),
        history_length,
        args.horizon,
    )
    histories = impute_alternate(
        test_coarse[:, :history_length],
        imputer_sampler,
        window=args.window_length,
        n_samples=args.n_histories,
        dt=dt,
        seed=seeds[1],
    )
    futures = forecast_tail(
        histories.reshape(-1, history_length, channels),
        forecaster_sampler,
        context=args.context,
        horizon=args.horizon,
        n_samples=args.n_futures,
        dt=dt,
        seed=seeds[2],
    ).reshape(len(test), -1, args.horizon, channels)
    truth = test[:, history_length:]
    last_value = np.broadcast_to(
        test[:, None, history_length - 1 : history_length],
        (len(test), 1, *truth.shape[1:]),
    )
    scores = {
        "timeautodiff_forecast_crps": crps(futures, truth),
        "last_value_crps": crps(last_value, truth),
    }
    if generation["generator"] == GP_GENERATOR:
        gp_mask = hidden[: len(test)].copy()
        gp_mask[:, history_length:] = True
        gp_samples = impute(
            test,
            GPPosteriorSampler(partial(gp_posterior, **generation["params"])),
            factor=1,
            mask=gp_mask,
            n_samples=futures.shape[1],
            dt=dt,
            seed=seeds[3],
        )
        scores["gp_forecast_crps"] = crps(gp_samples[:, :, history_length:], truth)
    logging.info("%s", scores)

    logging.info("Forecasting %d steps past the end of every series", args.horizon)
    forecast = forecast_tail(
        imputed,
        forecaster_sampler,
        context=args.context,
        horizon=args.horizon,
        n_samples=args.n_future_samples,
        dt=dt,
        seed=seeds[4],
    )
    np.savez_compressed(
        args.output / "forecast.npz",
        samples=forecast,
        times=(length + np.arange(args.horizon)) * dt,
    )
    (args.output / "results.json").write_text(
        json.dumps({**scores, **vars(args)}, indent=2, default=str) + "\n"
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
