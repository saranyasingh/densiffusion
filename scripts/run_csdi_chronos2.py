"""CSDI imputation then Chronos-2 sparse vs imputed forecast (multivariate TS).

Paper: Chronos-2 (https://arxiv.org/abs/2610.41956), code via chronos-forecasting.
CSDI uses this repo's reference-size multivariate denoiser with the quadratic
schedule / LR decay settings from docs/csdi.md.

Default preset is **electricity** with **block missing** (12h gaps) so linear
fill must bridge diurnal peaks. Chronos evaluation is **zero-shot only** for
now; LoRA fine-tuning is gated by ``ENABLE_CHRONOS_FINETUNE`` below.

    uv run python scripts/prepare_electricity.py --output data/electricity.npz

    uv run --extra research --extra chronos python scripts/run_csdi_chronos2.py \
        --data data/electricity.npz --output outputs/electricity_csdi_chronos2 \
        --device cuda

Stages (checkpoints are reused when present):

1. Temporal train/val/test split
2. Sparsify with grid / block / random missingness
3. Train CSDI; impute with a multi-draw median ensemble
4. Zero-shot Chronos on sparse (NaN) / CSDI / linear / dense-oracle contexts
5. Score MAE/RMSE/sMAPE/MASE/WQL/SQL vs seasonal naive on the dense horizon
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
    build_observation_mask,
    expand_mask,
    extract_quantile_matrix,
    fit_normalizer,
    forecast_metrics,
    linear_fill,
    load_multivariate_dataset,
    seasonal_naive_forecast,
    seasonal_scale,
    to_chronos_frame,
)
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

LOGGER = logging.getLogger("csdi_chronos2")
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SPLITS = dict(train_fraction=0.7, validation_fraction=0.15)

# Flip to True to restore LoRA/full Chronos fine-tuning of the context arms.
ENABLE_CHRONOS_FINETUNE = False

# Electricity default: 12h hidden blocks so linear fill cannot track diurnal peaks.
PRESETS = {
    "electricity": dict(
        missing_mode="block",
        sparse_step=4,
        block_length=12,
        observe_length=12,
        keep_probability=0.25,
        window_length=97,
        stride=48,
        impute_stride=48,
        context=168,
        horizon=24,
        forecast_stride=24,
        season=24,
    ),
    "fx": dict(
        missing_mode="grid",
        sparse_step=2,
        block_length=7,
        observe_length=7,
        keep_probability=0.25,
        window_length=65,
        stride=32,
        impute_stride=32,
        context=128,
        horizon=32,
        forecast_stride=32,
        season=7,
    ),
}


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


def fit_csdi(
    model,
    checkpoint,
    train,
    validation,
    train_mask,
    *,
    config,
    seed,
    validation_masks=None,
):
    if checkpoint.exists():
        LOGGER.info("Loading CSDI checkpoint %s", checkpoint)
        state = torch.load(checkpoint, map_location=config.device, weights_only=False)
        model.to(config.device).load_state_dict(state["model_state_dict"])
        return model.eval()
    if validation_masks is None:
        validation_masks = np.broadcast_to(train_mask, validation.shape).copy()
    return train_densifier(
        model,
        train,
        validation,
        config=config,
        dt=1.0,
        seed=seed,
        checkpoint=checkpoint,
        train_mask=train_mask,
        validation_mask=validation_masks,
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


def time_unit(freq: str) -> str:
    return "h" if freq in {"h", "H"} else "D"


def finetune_chronos(
    pipeline,
    frame,
    *,
    channels,
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
        target_columns=list(channels),
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
    channels,
    prediction_length: int,
    step: int,
    start: np.datetime64,
    freq: str,
    item_id: str,
    quantile_levels=QUANTILES,
    validate_inputs: bool = True,
) -> dict[float, np.ndarray]:
    frame = to_chronos_frame(
        context_values,
        channels=channels,
        start=start,
        freq=freq,
        item_id=item_id,
        step=step,
    )
    pred = pipeline.predict_df(
        frame,
        prediction_length=prediction_length,
        quantile_levels=list(quantile_levels),
        id_column="item_id",
        timestamp_column="timestamp",
        target=list(channels),
        validate_inputs=validate_inputs,
    )
    return extract_quantile_matrix(
        pred,
        currencies=channels,
        quantile_levels=quantile_levels,
        prediction_length=prediction_length,
    )


def score_forecast(
    truth: np.ndarray,
    quantiles: dict[float, np.ndarray] | None,
    *,
    scale: float,
    point: np.ndarray | None = None,
) -> dict:
    if quantiles is not None:
        point = quantiles[0.5]
    if point is None:
        raise ValueError("point forecast or quantile forecasts required")
    return forecast_metrics(
        truth, point, scale=scale, quantile_forecasts=quantiles
    )


def windowed_masks(hidden_time: np.ndarray, *, window: int, stride: int, channels: int):
    """Sliding windows of a length-T hidden mask, shaped like value windows."""
    mask = expand_mask(hidden_time, batch=1, channels=channels).astype(np.float64)
    return make_windows(mask, window_length=window, stride=stride) > 0.5


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

    values, _times, meta = load_multivariate_dataset(args.data)
    channels = meta["channels"]
    series_start = np.datetime64(meta["start"])
    series_freq = meta.get("freq", "D")
    item_id = meta.get("item_id", "series")
    n_channels = values.shape[2]
    if args.missing_mode == "grid":
        remainder = (values.shape[1] - 1) % args.sparse_step
        usable = values.shape[1] - remainder
    else:
        usable = values.shape[1]
    values = values[:, :usable]
    hidden_time = build_observation_mask(
        usable,
        mode=args.missing_mode,
        sparse_step=args.sparse_step,
        block_length=args.block_length,
        observe_length=args.observe_length,
        keep_probability=args.keep_probability,
        seed=args.seed,
    )
    observed = ~expand_mask(hidden_time, batch=1, channels=n_channels)

    train_v, val_v, _test_v = split_data(values, axis="time", **SPLITS)
    n_train, n_val = train_v.shape[1], val_v.shape[1]

    normalizer = fit_normalizer(train_v, observed=observed[:, :n_train])
    train_n = normalizer.transform(train_v)
    val_n = normalizer.transform(val_v)
    full_n = normalizer.transform(values)

    train_windows = make_windows(
        train_n, window_length=args.window_length, stride=args.stride
    )
    val_windows = make_windows(
        val_n, window_length=args.window_length, stride=args.stride
    )
    train_masks = windowed_masks(
        hidden_time[:n_train],
        window=args.window_length,
        stride=args.stride,
        channels=n_channels,
    )
    val_masks = windowed_masks(
        hidden_time[n_train : n_train + n_val],
        window=args.window_length,
        stride=args.stride,
        channels=n_channels,
    )
    # Drop windows that are fully observed or fully hidden.
    train_ok = np.array(
        [mask.any() and not mask.all() for mask in train_masks], dtype=bool
    )
    val_ok = np.array(
        [mask.any() and not mask.all() for mask in val_masks], dtype=bool
    )
    if not train_ok.any() or not val_ok.any():
        raise ValueError("No valid imputation windows under the current missing mask")
    train_windows, train_masks = train_windows[train_ok], train_masks[train_ok]
    val_windows, val_masks = val_windows[val_ok], val_masks[val_ok]
    LOGGER.info(
        "%s mode=%s: T=%d C=%d; windows train/val=%d/%d; freq=%s season=%d",
        meta.get("dataset", "series"),
        args.missing_mode,
        usable,
        n_channels,
        len(train_windows),
        len(val_windows),
        series_freq,
        args.season,
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
        train_masks,
        validation_masks=val_masks,
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
    dense_truth = values
    # Sparse Chronos context: dense timestamps with NaNs at missing times.
    sparse_nan = values.astype(np.float64, copy=True)
    sparse_nan[0, hidden_time] = np.nan

    scale = seasonal_scale(train_v, season=args.season)
    unit = time_unit(series_freq)
    pipeline = load_chronos(args.chronos_model, args.device)
    frame_kwargs = dict(
        channels=channels, start=series_start, freq=series_freq, item_id=item_id
    )
    # Zero-shot context arms (all on the native dense grid).
    context_arms = {
        "zeroshot_sparse": sparse_nan,
        "zeroshot_csdi": imputed,
        "zeroshot_linear": linear_imputed,
        "dense_oracle": dense_truth,
    }

    # Optional fine-tuning path — off unless ENABLE_CHRONOS_FINETUNE is True.
    finetuned_arms = {}
    if ENABLE_CHRONOS_FINETUNE and not args.skip_finetune:
        LOGGER.info("ENABLE_CHRONOS_FINETUNE=True: fine-tuning Chronos arms")
        train_frames = {
            "sparse": to_chronos_frame(
                sparse_nan[:, :n_train], step=1, **frame_kwargs
            ),
            "csdi_imputed": to_chronos_frame(
                imputed[:, :n_train], step=1, **frame_kwargs
            ),
            "linear_imputed": to_chronos_frame(
                linear_imputed[:, :n_train], step=1, **frame_kwargs
            ),
        }
        for name, frame in train_frames.items():
            finetuned_arms[name] = {
                "pipeline": finetune_chronos(
                    pipeline,
                    frame,
                    channels=channels,
                    prediction_length=args.horizon,
                    output_dir=args.output / f"chronos_{name}",
                    args=args,
                ),
                "series": {
                    "sparse": sparse_nan,
                    "csdi_imputed": imputed,
                    "linear_imputed": linear_imputed,
                }[name],
            }
    else:
        LOGGER.info(
            "Chronos fine-tuning disabled "
            "(ENABLE_CHRONOS_FINETUNE=%s); zero-shot only",
            ENABLE_CHRONOS_FINETUNE,
        )

    test_start = n_train + n_val
    first_origin = test_start + max(args.context, args.season)
    last_origin = usable - args.horizon
    origins = list(range(first_origin, last_origin + 1, args.forecast_stride))
    if not origins:
        raise ValueError("No forecast origins in test region; reduce context/horizon")
    if origins[-1] != last_origin:
        origins.append(last_origin)
    if args.max_origins is not None:
        origins = origins[: args.max_origins]

    LOGGER.info(
        "Backtesting %d origins on dense horizon=%d", len(origins), args.horizon
    )
    score_names = [
        *context_arms,
        "seasonal_naive",
        "last_value",
        *{f"finetuned_{name}" for name in finetuned_arms},
    ]
    arm_scores = {name: [] for name in score_names}

    for origin in origins:
        truth = dense_truth[0, origin : origin + args.horizon]
        history = dense_truth[0, :origin]
        forecast_kwargs = dict(channels=channels, freq=series_freq, item_id=item_id)
        start = series_start + np.timedelta64(origin - args.context, unit)

        naive = seasonal_naive_forecast(
            history, horizon=args.horizon, season=args.season
        )
        arm_scores["seasonal_naive"].append(
            score_forecast(truth, None, scale=scale, point=naive)
        )
        last_value = np.broadcast_to(history[-1][None, :], truth.shape)
        arm_scores["last_value"].append(
            score_forecast(truth, None, scale=scale, point=last_value)
        )

        for name, series in context_arms.items():
            context_vals = series[:, origin - args.context : origin]
            # Sparse contexts contain NaNs; Chronos-2 masks them when validation is off.
            allow_nan = name == "zeroshot_sparse"
            quantiles = chronos_forecast(
                pipeline,
                context_vals,
                prediction_length=args.horizon,
                step=1,
                start=start,
                validate_inputs=not allow_nan,
                **forecast_kwargs,
            )
            arm_scores[name].append(score_forecast(truth, quantiles, scale=scale))

        for name, arm in finetuned_arms.items():
            context_vals = arm["series"][:, origin - args.context : origin]
            quantiles = chronos_forecast(
                arm["pipeline"],
                context_vals,
                prediction_length=args.horizon,
                step=1,
                start=start,
                **forecast_kwargs,
            )
            arm_scores[f"finetuned_{name}"].append(
                score_forecast(truth, quantiles, scale=scale)
            )

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
        "seasonal_scale": scale,
        "n_origins": len(origins),
        "channels": channels,
        "meta": meta,
        "enable_chronos_finetune": ENABLE_CHRONOS_FINETUNE,
        "elapsed_sec": time.time() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "args": vars(args),
    }
    if (
        "zeroshot_csdi" in results["forecast"]
        and "zeroshot_sparse" in results["forecast"]
    ):
        results["deltas_csdi_minus_sparse"] = {
            key: results["forecast"]["zeroshot_csdi"][key]
            - results["forecast"]["zeroshot_sparse"][key]
            for key in results["forecast"]["zeroshot_sparse"]
        }
    if (
        "zeroshot_csdi" in results["forecast"]
        and "zeroshot_linear" in results["forecast"]
    ):
        results["deltas_csdi_minus_linear"] = {
            key: results["forecast"]["zeroshot_csdi"][key]
            - results["forecast"]["zeroshot_linear"][key]
            for key in results["forecast"]["zeroshot_linear"]
        }
    write_json(args.output / "results.json", results)
    LOGGER.info("Results:\n%s", json.dumps(results["forecast"], indent=2))
    LOGGER.info("Imputation CRPS CSDI=%.6f linear=%.6f", imputation_crps, linear_crps)
    try:
        import sys

        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from plot_csdi_chronos2_results import main as plot_main

        plot_main(["--run", str(args.output)])
        LOGGER.info("Wrote visualizations under %s", args.output)
    except Exception as exc:  # plotting is best-effort
        LOGGER.warning("Skipping result plots (%s)", exc)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="electricity",
        help="Default window/sparsity/horizon settings (overridden by explicit flags)",
    )
    parser.add_argument("--data", type=Path, default=Path("data/electricity.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--missing-mode",
        choices=("grid", "block", "random"),
        default=None,
        help="grid=every sparse-step; block=contiguous gaps; random=Bernoulli keep",
    )
    parser.add_argument("--sparse-step", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--observe-length", type=int, default=None)
    parser.add_argument("--keep-probability", type=float, default=None)
    parser.add_argument("--season", type=int, default=None, help="MASE/SQL season")
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--impute-stride", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--forecast-stride", type=int, default=None)
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
        help="Ignored while ENABLE_CHRONOS_FINETUNE is False; kept for later use",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny CSDI + Chronos-2-small smoke settings",
    )
    args = parser.parse_args(argv)
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.window_length < 3:
        parser.error("--window-length must be at least 3")
    if args.season < 1:
        parser.error("--season must be positive")
    if args.missing_mode == "grid":
        if (args.window_length - 1) % args.sparse_step:
            parser.error("--window-length must satisfy (L-1) % sparse_step == 0")
        if args.stride % args.sparse_step or args.impute_stride % args.sparse_step:
            parser.error(
                "--stride and --impute-stride must be multiples of --sparse-step"
            )
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.n_samples = min(args.n_samples, 2)
        args.window_length = min(args.window_length, 49)
        args.stride = min(args.stride, 24)
        args.impute_stride = args.stride
        args.context = min(args.context, 48)
        args.horizon = min(args.horizon, 24)
        args.forecast_stride = max(args.horizon, args.forecast_stride)
        args.max_origins = 2 if args.max_origins is None else args.max_origins
        args.block_length = min(args.block_length, 6)
        args.observe_length = min(args.observe_length, 6)
        args.chronos_model = "autogluon/chronos-2-small"
        args.chronos_steps = min(args.chronos_steps, 20)
        args.hidden_channels = 32
        args.residual_layers = 2
        args.attention_heads = 4
        args.diffusion_embedding_dim = 64
        args.time_embedding_dim = 64
        args.feedforward_dim = 64
        args.batch_size = min(args.batch_size, 8)
        args.lr_decay_fractions = []
        args.validation_steps = "sampled"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
