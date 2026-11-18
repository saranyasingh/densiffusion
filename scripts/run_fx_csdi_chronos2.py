"""Strongest FX pipeline: CSDI imputation then Chronos-2 sparse vs imputed forecast.

Paper: Chronos-2 (https://arxiv.org/abs/2610.41956), code via chronos-forecasting.
CSDI uses this repo's reference-size multivariate denoiser with the quadratic
schedule / LR decay settings from docs/csdi.md.

    # one-time data prep
    uv run python scripts/prepare_fx.py --output data/fx.npz

    # full GPU run
    uv run --extra research --extra chronos python scripts/run_fx_csdi_chronos2.py \
        --data data/fx.npz --output outputs/fx_csdi_chronos2 --device cuda

    # short smoke check
    uv run --extra research --extra chronos python scripts/run_fx_csdi_chronos2.py \
        --data data/fx.npz --output outputs/fx_csdi_chronos2_quick --quick --device cpu

Stages (checkpoints are reused when present):

1. Temporal train/val/test split of the multivariate FX series
2. Regular sparsification (keep every --sparse-step day)
3. Train reference-size CSDI on train windows; validate on val windows
4. Impute train/val/test histories with a multi-draw median ensemble
5. Fine-tune Chronos-2 (LoRA by default) on sparse and on CSDI-imputed train
6. Rolling backtest: Chronos on sparse vs imputed vs linear-fill, plus zero-shot
7. Score MAE / RMSE / sMAPE / WQL on the shared coarse forecast grid
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.data import make_windows, split_data
from densiffusion.fx import (
    FX_START,
    expand_mask,
    extract_quantile_matrix,
    fit_normalizer,
    linear_fill,
    load_fx_dataset,
    point_metrics,
    sparse_observation_mask,
    to_chronos_frame,
    weighted_quantile_loss,
)
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

LOGGER = logging.getLogger("fx_csdi_chronos2")
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SPLITS = dict(train_fraction=0.7, validation_fraction=0.15)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def build_csdi(n_channels: int, args) -> ImputationModel:
    return ImputationModel(
        CSDIDenoiser(
            n_channels=n_channels,
            hidden_channels=args.hidden_channels,
            residual_layers=args.residual_layers,
            attention_heads=args.attention_heads,
            diffusion_embedding_dim=args.diffusion_embedding_dim,
            time_embedding_dim=args.time_embedding_dim,
            feature_embedding_dim=args.feature_embedding_dim,
            feedforward_dim=args.feedforward_dim,
            dropout=args.dropout,
        )
    )


def fit_csdi(model, checkpoint, train, validation, mask, *, config, seed):
    if checkpoint.exists():
        LOGGER.info("Loading CSDI checkpoint %s", checkpoint)
        state = torch.load(checkpoint, map_location=config.device, weights_only=False)
        model.to(config.device).load_state_dict(state["model_state_dict"])
        return model.eval()
    return train_densifier(
        model,
        train,
        validation,
        config=config,
        dt=1.0,
        seed=seed,
        checkpoint=checkpoint,
        train_mask=np.broadcast_to(mask, train.shape),
        validation_mask=np.broadcast_to(mask, validation.shape),
    )


def impute_series(
    values: np.ndarray,
    sampler: DDPMSampler,
    *,
    hidden_time: np.ndarray,
    window: int,
    stride: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Fill hidden entries of (1, T, C) with overlapping-window CSDI draws.

    Returns ``(n_samples, T, C)``. Observed positions are preserved exactly.
    Overlapping hidden predictions are averaged across windows.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.shape[0] != 1:
        raise ValueError("impute_series expects a single multivariate series")
    length, channels = series.shape[1], series.shape[2]
    if length < window:
        raise ValueError("series shorter than imputation window")
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] + window < length:
        starts.append(length - window)
    full_hidden = expand_mask(hidden_time, batch=1, channels=channels)
    masked = np.where(full_hidden, 0.0, series)
    total = np.zeros((n_samples, length, channels), dtype=np.float64)
    weight = np.zeros((length,), dtype=np.float64)
    for index, start in enumerate(starts):
        local_hidden = hidden_time[start : start + window]
        local_mask = expand_mask(local_hidden, batch=1, channels=channels)
        draws = impute(
            masked[:, start : start + window],
            sampler,
            factor=1,
            mask=local_mask,
            n_samples=n_samples,
            dt=1.0,
            seed=seed + index,
        )[0]
        stop = start + window
        # Average only at positions that were hidden in this window.
        total[:, start:stop] += draws * local_hidden[None, :, None]
        weight[start:stop] += local_hidden.astype(np.float64)
    # Observed entries: copy through exactly. Hidden entries: average draws.
    result = np.repeat(series, n_samples, axis=0)
    hidden = hidden_time.astype(bool)
    if hidden.any():
        if (weight[hidden] == 0).any():
            raise RuntimeError("imputation windows failed to cover every hidden point")
        scale = weight[hidden][None, :, None]
        result[:, hidden] = total[:, hidden] / scale
    return result


def aggregate_imputations(draws: np.ndarray, how: str) -> np.ndarray:
    if how == "mean":
        return draws.mean(axis=0, keepdims=True)
    if how == "median":
        return np.median(draws, axis=0, keepdims=True)
    raise ValueError("aggregate must be mean or median")


def load_chronos(model_id: str, device: str):
    from chronos import Chronos2Pipeline

    device_map = device if device in {"cpu", "mps"} else device
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)


def finetune_chronos(
    pipeline,
    frame,
    *,
    currencies,
    prediction_length: int,
    output_dir: Path,
    args,
):
    from chronos.chronos2 import preprocess

    checkpoint = output_dir / "finetuned-ckpt"
    if checkpoint.exists() and (checkpoint / "config.json").exists():
        LOGGER.info("Loading fine-tuned Chronos from %s", checkpoint)
        return type(pipeline).from_pretrained(checkpoint, device_map=args.device)

    train_inputs = preprocess.from_data_frame(
        frame,
        target_columns=list(currencies),
        prediction_length=prediction_length,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Fine-tuning Chronos-2 (%s) for %d steps -> %s",
        args.finetune_mode,
        args.chronos_steps,
        output_dir,
    )
    return pipeline.fit(
        inputs=train_inputs,
        prediction_length=prediction_length,
        finetune_mode=args.finetune_mode,
        learning_rate=args.chronos_lr,
        num_steps=args.chronos_steps,
        batch_size=args.chronos_batch_size,
        output_dir=output_dir,
        finetuned_ckpt_name="finetuned-ckpt",
        logging_steps=max(1, args.chronos_steps // 10),
    )


def chronos_forecast(
    pipeline,
    context_values: np.ndarray,
    *,
    currencies,
    prediction_length: int,
    step: int,
    start: np.datetime64,
    quantile_levels=QUANTILES,
) -> dict[float, np.ndarray]:
    frame = to_chronos_frame(
        context_values,
        currencies=currencies,
        start=start,
        step=step,
    )
    pred = pipeline.predict_df(
        frame,
        prediction_length=prediction_length,
        quantile_levels=list(quantile_levels),
        id_column="item_id",
        timestamp_column="timestamp",
        target=list(currencies),
        validate_inputs=True,
    )
    return extract_quantile_matrix(
        pred,
        currencies=currencies,
        quantile_levels=quantile_levels,
        prediction_length=prediction_length,
    )


def score_forecast(truth: np.ndarray, quantiles: dict[float, np.ndarray]) -> dict:
    median = quantiles[0.5]
    return {
        **point_metrics(truth, median),
        "wql": weighted_quantile_loss(truth, quantiles),
    }


def run(args) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(args.output / "training.log"),
            logging.StreamHandler(),
        ],
    )
    started = time.time()
    torch.manual_seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")

    values, _times, meta = load_fx_dataset(args.data)
    currencies = meta["currencies"]
    n_channels = values.shape[2]
    # Keep an odd length so alternate-style windows have shared endpoints.
    usable = values.shape[1] - (1 - values.shape[1] % 2)
    values = values[:, :usable]
    hidden_time = sparse_observation_mask(usable, step=args.sparse_step, offset=0)
    observed = ~expand_mask(hidden_time, batch=1, channels=n_channels)

    train_v, val_v, _test_v = split_data(values, axis="time", **SPLITS)
    n_train, n_val = train_v.shape[1], val_v.shape[1]

    normalizer = fit_normalizer(train_v, observed=observed[:, :n_train])
    train_n = normalizer.transform(train_v)
    val_n = normalizer.transform(val_v)
    full_n = normalizer.transform(values)

    window_mask = sparse_observation_mask(
        args.window_length, step=args.sparse_step, offset=0
    )
    train_windows = make_windows(
        train_n, window_length=args.window_length, stride=args.stride
    )
    val_windows = make_windows(
        val_n, window_length=args.window_length, stride=args.stride
    )
    LOGGER.info(
        "FX %s: T=%d C=%d; windows train/val=%d/%d; sparse_step=%d",
        currencies,
        usable,
        n_channels,
        len(train_windows),
        len(val_windows),
        args.sparse_step,
    )

    csdi_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.schedule,
        weight_decay=args.weight_decay,
        lr_decay_fractions=tuple(args.lr_decay_fractions),
        lr_decay_factor=args.lr_decay_factor,
        validation_steps=args.validation_steps,
        validation_interval=args.validation_interval,
        device=args.device,
    )
    model = build_csdi(n_channels, args)
    model = fit_csdi(
        model,
        args.output / "csdi.pt",
        train_windows,
        val_windows,
        expand_mask(window_mask, batch=1, channels=n_channels)[0],
        config=csdi_config,
        seed=args.seed,
    )
    sampler = DDPMSampler(
        model, replace(csdi_config, batch_size=args.sample_batch_size)
    )

    imputed_path = args.output / "imputed.npz"
    if imputed_path.exists():
        LOGGER.info("Loading imputations from %s", imputed_path)
        with np.load(imputed_path, allow_pickle=False) as payload:
            imputed_norm = payload["imputed_norm"]
            linear_norm = payload["linear_norm"]
            imputation_crps = float(payload["imputation_crps"])
            linear_crps = float(payload["linear_crps"])
    else:
        LOGGER.info("Imputing full series with %d CSDI draws", args.n_samples)
        draws = impute_series(
            full_n,
            sampler,
            hidden_time=hidden_time,
            window=args.window_length,
            stride=args.impute_stride,
            n_samples=args.n_samples,
            seed=args.seed + 17,
        )
        imputed_norm = aggregate_imputations(draws, args.aggregate)
        sparse_full = np.where(observed, full_n, 0.0)
        linear_norm = linear_fill(sparse_full, ~observed)
        eval_mask = ~observed
        imputation_crps = crps(draws[None], full_n, mask=eval_mask)
        linear_crps = crps(linear_norm[:, None], full_n, mask=eval_mask)
        np.savez_compressed(
            imputed_path,
            imputed_norm=imputed_norm,
            linear_norm=linear_norm,
            draws_norm=draws,
            target_norm=full_n,
            observed=observed,
            imputation_crps=np.asarray(imputation_crps),
            linear_crps=np.asarray(linear_crps),
            mean=normalizer.mean,
            std=normalizer.std,
        )
        LOGGER.info(
            "Imputation CRPS: CSDI=%.6f linear=%.6f", imputation_crps, linear_crps
        )

    imputed = normalizer.inverse(imputed_norm)
    linear_imputed = normalizer.inverse(linear_norm)

    # Coarse grids for the sparse Chronos arm (observed indices only).
    observed_index = np.where(~hidden_time)[0]
    sparse_series = values[:, observed_index]
    dense_truth = values

    # Chronos horizons counted in coarse steps; calendar length matches.
    if args.horizon % args.sparse_step:
        raise ValueError("--horizon must be divisible by --sparse-step")
    if args.context % args.sparse_step:
        raise ValueError("--context must be divisible by --sparse-step")
    coarse_horizon = args.horizon // args.sparse_step
    coarse_context = args.context // args.sparse_step

    train_obs_end = int((~hidden_time[:n_train]).sum())
    sparse_train = sparse_series[:, :train_obs_end]
    imputed_train = imputed[:, :n_train]
    linear_train = linear_imputed[:, :n_train]

    pipeline = load_chronos(args.chronos_model, args.device)
    arms = {
        "sparse": {
            "train_frame": to_chronos_frame(
                sparse_train,
                currencies=currencies,
                start=FX_START,
                step=args.sparse_step,
            ),
            "series": sparse_series,
            "step": args.sparse_step,
            "context": coarse_context,
            "horizon": coarse_horizon,
            "prediction_length": coarse_horizon,
            "finetune_dir": args.output / "chronos_sparse",
        },
        "csdi_imputed": {
            "train_frame": to_chronos_frame(
                imputed_train,
                currencies=currencies,
                start=FX_START,
                step=1,
            ),
            "series": imputed,
            "step": 1,
            "context": args.context,
            "horizon": args.horizon,
            "prediction_length": args.horizon,
            "finetune_dir": args.output / "chronos_csdi",
        },
        "linear_imputed": {
            "train_frame": to_chronos_frame(
                linear_train,
                currencies=currencies,
                start=FX_START,
                step=1,
            ),
            "series": linear_imputed,
            "step": 1,
            "context": args.context,
            "horizon": args.horizon,
            "prediction_length": args.horizon,
            "finetune_dir": args.output / "chronos_linear",
        },
    }

    finetuned = {}
    for name, arm in arms.items():
        if args.skip_finetune:
            finetuned[name] = pipeline
        else:
            finetuned[name] = finetune_chronos(
                pipeline,
                arm["train_frame"],
                currencies=currencies,
                prediction_length=arm["prediction_length"],
                output_dir=arm["finetune_dir"],
                args=args,
            )

    # Rolling backtest on the test split, scored on the coarse observation grid.
    test_start = n_train + n_val
    first_origin = test_start + args.context
    last_origin = usable - args.horizon
    origins = list(range(first_origin, last_origin + 1, args.forecast_stride))
    if not origins:
        raise ValueError("No forecast origins in test region; reduce context/horizon")
    if origins[-1] != last_origin:
        origins.append(last_origin)
    if args.max_origins is not None:
        origins = origins[: args.max_origins]

    LOGGER.info("Backtesting %d origins on test region", len(origins))
    arm_scores = {name: [] for name in (*arms, "zeroshot_sparse", "zeroshot_csdi")}
    last_value_scores = []

    for origin in origins:
        # Shared coarse evaluation indices inside the forecast horizon.
        future_idx = np.arange(origin, origin + args.horizon)
        score_idx = future_idx[future_idx % args.sparse_step == 0]
        if len(score_idx) != coarse_horizon:
            # Keep exact coarse-horizon count aligned with sparse grid phase.
            score_idx = observed_index[
                (observed_index >= origin) & (observed_index < origin + args.horizon)
            ]
        truth = dense_truth[0, score_idx]
        last_value = np.broadcast_to(dense_truth[0, origin - 1][None, :], truth.shape)
        last_value_scores.append(point_metrics(truth, last_value))

        for name, arm in arms.items():
            step = arm["step"]
            if step == 1:
                context_vals = arm["series"][:, origin - arm["context"] : origin]
                start = FX_START + np.timedelta64(origin - arm["context"], "D")
                quantiles = chronos_forecast(
                    finetuned[name],
                    context_vals,
                    currencies=currencies,
                    prediction_length=arm["horizon"],
                    step=1,
                    start=start,
                )
                # Restrict dense forecasts to the shared coarse score grid.
                offset = score_idx - origin
                quantiles = {q: matrix[offset] for q, matrix in quantiles.items()}
            else:
                # Context on the coarse grid ending at the last observed point < origin.
                coarse_end = int(np.searchsorted(observed_index, origin, side="left"))
                coarse_start = coarse_end - arm["context"]
                if coarse_start < 0:
                    continue
                context_vals = arm["series"][:, coarse_start:coarse_end]
                start = FX_START + np.timedelta64(
                    int(observed_index[coarse_start]), "D"
                )
                quantiles = chronos_forecast(
                    finetuned[name],
                    context_vals,
                    currencies=currencies,
                    prediction_length=arm["horizon"],
                    step=step,
                    start=start,
                )
            arm_scores[name].append(score_forecast(truth, quantiles))

        # Zero-shot baselines on sparse and CSDI-imputed contexts.
        for zs_name, arm_name in (
            ("zeroshot_sparse", "sparse"),
            ("zeroshot_csdi", "csdi_imputed"),
        ):
            arm = arms[arm_name]
            step = arm["step"]
            if step == 1:
                context_vals = arm["series"][:, origin - arm["context"] : origin]
                start = FX_START + np.timedelta64(origin - arm["context"], "D")
            else:
                coarse_end = int(np.searchsorted(observed_index, origin, side="left"))
                coarse_start = coarse_end - arm["context"]
                if coarse_start < 0:
                    continue
                context_vals = arm["series"][:, coarse_start:coarse_end]
                start = FX_START + np.timedelta64(
                    int(observed_index[coarse_start]), "D"
                )
            quantiles = chronos_forecast(
                pipeline,
                context_vals,
                currencies=currencies,
                prediction_length=arm["horizon"],
                step=step,
                start=start,
            )
            if step == 1:
                offset = score_idx - origin
                quantiles = {q: matrix[offset] for q, matrix in quantiles.items()}
            arm_scores[zs_name].append(score_forecast(truth, quantiles))

    def summarize(scores: list[dict]) -> dict[str, float]:
        if not scores:
            return {}
        keys = scores[0].keys()
        return {key: float(np.mean([row[key] for row in scores])) for key in keys}

    results = {
        "imputation": {
            "csdi_crps": float(imputation_crps),
            "linear_crps": float(linear_crps),
            "csdi_beats_linear": bool(imputation_crps < linear_crps),
        },
        "forecast": {
            name: summarize(rows) for name, rows in arm_scores.items() if rows
        },
        "forecast_last_value": summarize(last_value_scores),
        "n_origins": len(origins),
        "currencies": currencies,
        "meta": meta,
        "elapsed_sec": time.time() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "args": vars(args),
    }
    # Convenience deltas: imputed vs sparse fine-tuned.
    if "csdi_imputed" in results["forecast"] and "sparse" in results["forecast"]:
        results["deltas_csdi_minus_sparse"] = {
            key: results["forecast"]["csdi_imputed"][key]
            - results["forecast"]["sparse"][key]
            for key in results["forecast"]["sparse"]
        }
    write_json(args.output / "results.json", results)
    LOGGER.info("Results:\n%s", json.dumps(results["forecast"], indent=2))
    LOGGER.info("Imputation CRPS CSDI=%.6f linear=%.6f", imputation_crps, linear_crps)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/fx.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse-step", type=int, default=2)
    parser.add_argument("--window-length", type=int, default=65)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--impute-stride", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--forecast-stride", type=int, default=32)
    parser.add_argument("--max-origins", type=int, default=None)
    # CSDI — reference-size defaults from docs/csdi.md stronger training.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument(
        "--schedule", choices=("linear", "quadratic"), default="quadratic"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--lr-decay-fractions",
        type=float,
        nargs="+",
        default=[0.75, 0.9],
    )
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--validation-steps", choices=("sampled", "all"), default="all")
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--diffusion-embedding-dim", type=int, default=128)
    parser.add_argument("--time-embedding-dim", type=int, default=128)
    parser.add_argument("--feature-embedding-dim", type=int, default=16)
    parser.add_argument("--feedforward-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Chronos-2
    parser.add_argument("--chronos-model", default="amazon/chronos-2")
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--chronos-steps", type=int, default=1000)
    parser.add_argument("--chronos-lr", type=float, default=1e-4)
    parser.add_argument("--chronos-batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Evaluate zero-shot Chronos only (still labels arms with base weights)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny CSDI + Chronos-2-small smoke settings",
    )
    args = parser.parse_args(argv)
    if args.window_length < 3:
        parser.error("--window-length must be at least 3")
    if (args.window_length - 1) % args.sparse_step:
        parser.error("--window-length must satisfy (L-1) % sparse_step == 0")
    if args.stride % args.sparse_step or args.impute_stride % args.sparse_step:
        parser.error("--stride and --impute-stride must be multiples of --sparse-step")
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.n_samples = min(args.n_samples, 2)
        args.window_length = min(args.window_length, 33)
        args.stride = min(args.stride, 32)
        args.impute_stride = min(args.impute_stride, 32)
        args.context = min(args.context, 64)
        args.horizon = min(args.horizon, 16)
        args.forecast_stride = max(args.horizon, args.forecast_stride)
        args.max_origins = 2 if args.max_origins is None else args.max_origins
        args.chronos_model = "autogluon/chronos-2-small"
        args.chronos_steps = min(args.chronos_steps, 20)
        args.hidden_channels = 32
        args.residual_layers = 2
        args.attention_heads = 4
        args.diffusion_embedding_dim = 64
        args.time_embedding_dim = 64
        args.feedforward_dim = 64
        args.batch_size = min(args.batch_size, 8)
        # Avoid MultiStepLR collapsing the LR on a 2-epoch smoke run.
        args.lr_decay_fractions = []
        args.validation_steps = "sampled"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
