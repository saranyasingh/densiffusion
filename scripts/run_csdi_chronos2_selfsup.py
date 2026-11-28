"""Self-supervised CSDI on sparse electricity, then Chronos-2 forecast.

Simulates a sparse observation set, further-masks a fraction of those
observations as SSL targets (never using naturally missing values as
training targets), densifies the base-missing times, and zero-shot
Chronos-2 on sparse / CSDI / linear / dense-oracle contexts.

    uv run --extra research --extra chronos python \
        scripts/run_csdi_chronos2_selfsup.py \
        --data data/electricity.npz \
        --output outputs/elec_selfsup_block \
        --missing-mode block --ssl-mask-rate 0.25 --device cuda

Pass ``--supervised-baseline`` to also train the supervised densifier under
the same base mask into ``output/supervised/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
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
    fit_normalizer,
    linear_fill,
    load_multivariate_dataset,
    sample_ssl_target_mask,
    seasonal_naive_forecast,
    seasonal_scale,
    to_chronos_frame,
)
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

# Reuse Chronos / impute helpers from the supervised runner.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from run_csdi_chronos2 import (  # noqa: E402
    ENABLE_CHRONOS_FINETUNE,
    QUANTILES,
    SPLITS,
    aggregate_imputations,
    chronos_forecast,
    finetune_chronos,
    impute_series,
    load_chronos,
    score_forecast,
    step_delta,
    windowed_masks,
    write_json,
)

LOGGER = logging.getLogger("csdi_chronos2_selfsup")

PRESETS = {
    "electricity": dict(
        missing_mode="block",
        sparse_step=4,
        block_length=12,
        observe_length=12,
        keep_probability=0.5,
        window_length=97,
        stride=48,
        impute_stride=48,
        context=168,
        horizon=24,
        forecast_stride=24,
        season=24,
        ssl_mask_rate=0.25,
        natural_missing=False,
    ),
    # Intel Lab temps @ 1 min: natural ~50% missing, SSL rate from random Sweep B.
    "intel_lab": dict(
        missing_mode="random",  # unused when natural_missing=True
        sparse_step=2,
        block_length=12,
        observe_length=12,
        keep_probability=0.5,
        window_length=97,
        stride=48,
        impute_stride=48,
        context=180,  # 3h
        horizon=60,  # 1h
        forecast_stride=60,
        season=1440,  # 1 day of minutes
        ssl_mask_rate=0.25,
        natural_missing=True,
    ),
}


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


def windowed_available(
    available: np.ndarray, *, window: int, stride: int, channels: int | None = None
) -> np.ndarray:
    """Sliding windows of availability (True = observed).

    ``available`` is length-T (broadcast to all channels) or ``(1, T, C)``.
    """
    available = np.asarray(available)
    if available.ndim == 1:
        if channels is None:
            raise ValueError("channels required for 1-D availability")
        mask = np.broadcast_to(
            available.astype(np.float64)[None, :, None],
            (1, len(available), channels),
        ).copy()
    elif available.ndim == 3:
        mask = available.astype(np.float64)
    else:
        raise ValueError("available must be 1-D or (1, T, C)")
    return make_windows(mask, window_length=window, stride=stride) > 0.5


def fit_csdi_selfsup(
    model,
    checkpoint,
    train,
    validation,
    train_mask,
    validation_mask,
    train_availability,
    validation_availability,
    *,
    config,
    seed,
):
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
        train_mask=train_mask,
        validation_mask=validation_mask,
        train_availability=train_availability,
        validation_availability=validation_availability,
    )


def fit_csdi_supervised(
    model,
    checkpoint,
    train,
    validation,
    train_mask,
    validation_mask,
    *,
    config,
    seed,
):
    if checkpoint.exists():
        LOGGER.info("Loading supervised CSDI checkpoint %s", checkpoint)
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
        train_mask=train_mask,
        validation_mask=validation_mask,
    )


def filter_ssl_windows(values, availability, targets):
    """Keep windows with SSL targets and leftover conditioning observations."""
    ok = np.array(
        [
            tgt.any() and (avail & ~tgt).any()
            for avail, tgt in zip(availability, targets, strict=True)
        ],
        dtype=bool,
    )
    if not ok.any():
        raise ValueError(
            "No valid SSL windows under the current base mask / ssl-mask-rate"
        )
    return values[ok], availability[ok], targets[ok]


def run_imputation_and_forecast(
    *,
    args,
    output: Path,
    values,
    full_n,
    train_v,
    n_train,
    n_val,
    hidden_time,
    observed,
    normalizer,
    channels,
    series_start,
    series_freq,
    item_id,
    meta,
    usable,
    sampler: DDPMSampler,
    protocol: str,
    ssl_eval_mask=None,
    natural_missing: bool = False,
):
    """Impute base-missing times, backtest Chronos, write results.json.

    ``hidden_time`` is length-T (synthetic) or ``(1, T, C)`` (natural missing).
    With ``natural_missing``, skip dense-oracle CRPS / forecast arms and score
    Chronos only on horizon cells that were actually observed.
    """
    imputed_path = output / "imputed.npz"
    if imputed_path.exists():
        LOGGER.info("Loading imputations from %s", imputed_path)
        with np.load(imputed_path, allow_pickle=False) as payload:
            imputed_norm = payload["imputed_norm"]
            linear_norm = payload["linear_norm"]
            imputation_crps = (
                float(payload["imputation_crps"])
                if "imputation_crps" in payload.files
                and np.isfinite(payload["imputation_crps"])
                else None
            )
            linear_crps = (
                float(payload["linear_crps"])
                if "linear_crps" in payload.files
                and np.isfinite(payload["linear_crps"])
                else None
            )
            ssl_crps = (
                float(payload["ssl_crps"]) if "ssl_crps" in payload.files else None
            )
    else:
        LOGGER.info("Imputing full series with %d CSDI draws (%s)", args.n_samples, protocol)
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
        imputation_crps = None
        linear_crps = None
        if not natural_missing:
            eval_mask = ~observed
            imputation_crps = crps(draws[None], full_n, mask=eval_mask)
            linear_crps = crps(linear_norm[:, None], full_n, mask=eval_mask)
        ssl_crps = None
        if ssl_eval_mask is not None and np.asarray(ssl_eval_mask).any():
            ssl_targets = np.asarray(ssl_eval_mask, dtype=bool)
            if ssl_targets.ndim == 1:
                ssl_hidden = (
                    expand_mask(hidden_time, batch=1, channels=full_n.shape[2])
                    if np.asarray(hidden_time).ndim == 1
                    else np.asarray(hidden_time, dtype=bool)
                ) | expand_mask(ssl_targets, batch=1, channels=full_n.shape[2])
                ssl_score_mask = expand_mask(
                    ssl_targets, batch=1, channels=full_n.shape[2]
                )
            else:
                base_hidden = (
                    expand_mask(hidden_time, batch=1, channels=full_n.shape[2])
                    if np.asarray(hidden_time).ndim == 1
                    else np.asarray(hidden_time, dtype=bool)
                )
                ssl_hidden = base_hidden | ssl_targets
                ssl_score_mask = ssl_targets
            ssl_draws = impute_series(
                full_n,
                sampler,
                hidden_time=ssl_hidden,
                window=args.window_length,
                stride=args.impute_stride,
                n_samples=args.n_samples,
                seed=args.seed + 31,
            )
            ssl_crps = crps(ssl_draws[None], full_n, mask=ssl_score_mask)
        save_kwargs = dict(
            imputed_norm=imputed_norm,
            linear_norm=linear_norm,
            draws_norm=draws,
            target_norm=full_n,
            observed=observed,
            mean=normalizer.mean,
            std=normalizer.std,
        )
        if imputation_crps is not None:
            save_kwargs["imputation_crps"] = np.asarray(imputation_crps)
            save_kwargs["linear_crps"] = np.asarray(linear_crps)
        if ssl_crps is not None:
            save_kwargs["ssl_crps"] = np.asarray(ssl_crps)
        np.savez_compressed(imputed_path, **save_kwargs)
        LOGGER.info(
            "Imputation CRPS: CSDI=%s linear=%s ssl=%s",
            f"{imputation_crps:.6f}" if imputation_crps is not None else "n/a",
            f"{linear_crps:.6f}" if linear_crps is not None else "n/a",
            f"{ssl_crps:.6f}" if ssl_crps is not None else "n/a",
        )

    imputed = normalizer.inverse(imputed_norm)
    linear_imputed = normalizer.inverse(linear_norm)
    dense_truth = values
    sparse_nan = values.astype(np.float64, copy=True)
    sparse_nan = np.where(observed, sparse_nan, np.nan)

    # MASE scale: use linear-filled train so sparse zeros do not dominate.
    train_filled = linear_fill(
        np.where(observed[:, :n_train], train_v, 0.0), ~observed[:, :n_train]
    )
    scale = seasonal_scale(train_filled, season=args.season)
    delta = step_delta(series_freq)
    pipeline = load_chronos(args.chronos_model, args.device)
    frame_kwargs = dict(
        channels=channels, start=series_start, freq=series_freq, item_id=item_id
    )
    context_arms = {
        "zeroshot_sparse": sparse_nan,
        "zeroshot_csdi": imputed,
        "zeroshot_linear": linear_imputed,
    }
    if not natural_missing:
        context_arms["dense_oracle"] = dense_truth

    finetuned_arms = {}
    if ENABLE_CHRONOS_FINETUNE and not args.skip_finetune:
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
                    output_dir=output / f"chronos_{name}",
                    args=args,
                ),
                "series": {
                    "sparse": sparse_nan,
                    "csdi_imputed": imputed,
                    "linear_imputed": linear_imputed,
                }[name],
            }

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
        "Backtesting %d origins on horizon=%d (%s, natural_missing=%s)",
        len(origins),
        args.horizon,
        protocol,
        natural_missing,
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
        horizon_mask = observed[0, origin : origin + args.horizon]
        if natural_missing and not horizon_mask.any():
            continue
        score_mask = horizon_mask if natural_missing else None
        # Baselines use linear-filled history when values are naturally sparse.
        history = linear_imputed[0, :origin] if natural_missing else dense_truth[0, :origin]
        forecast_kwargs = dict(channels=channels, freq=series_freq, item_id=item_id)
        start = series_start + (origin - args.context) * delta

        naive = seasonal_naive_forecast(
            history, horizon=args.horizon, season=args.season
        )
        arm_scores["seasonal_naive"].append(
            score_forecast(truth, None, scale=scale, point=naive, mask=score_mask)
        )
        last_value = np.broadcast_to(history[-1][None, :], truth.shape)
        arm_scores["last_value"].append(
            score_forecast(
                truth, None, scale=scale, point=last_value, mask=score_mask
            )
        )

        for name, series in context_arms.items():
            context_vals = series[:, origin - args.context : origin]
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
            arm_scores[name].append(
                score_forecast(truth, quantiles, scale=scale, mask=score_mask)
            )

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
                score_forecast(truth, quantiles, scale=scale, mask=score_mask)
            )

    def summarize(scores: list[dict]) -> dict[str, float]:
        if not scores:
            return {}
        keys = scores[0].keys()
        return {key: float(np.mean([row[key] for row in scores])) for key in keys}

    observed_fraction = float(observed.mean())
    imputation = {}
    if imputation_crps is not None and linear_crps is not None:
        imputation = {
            "csdi_crps": float(imputation_crps),
            "linear_crps": float(linear_crps),
            "csdi_beats_linear": bool(imputation_crps < linear_crps),
        }
    if ssl_crps is not None:
        imputation["ssl_crps"] = float(ssl_crps)
    results = {
        "protocol": protocol,
        "ssl_mask_rate": float(args.ssl_mask_rate),
        "observed_fraction": observed_fraction,
        "natural_missing": bool(natural_missing),
        "imputation": imputation,
        "forecast": {
            name: summarize(rows) for name, rows in arm_scores.items() if rows
        },
        "seasonal_scale": scale,
        "n_origins": len(origins),
        "channels": channels,
        "meta": {k: v for k, v in meta.items() if k != "observed"},
        "enable_chronos_finetune": ENABLE_CHRONOS_FINETUNE,
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
    return results


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
    natural_missing = bool(getattr(args, "natural_missing", False))
    if not natural_missing and args.missing_mode not in {"block", "random"}:
        raise ValueError("self-sup runner supports missing-mode block or random")

    values, _times, meta = load_multivariate_dataset(args.data)
    channels = meta["channels"]
    series_start = np.datetime64(meta["start"])
    series_freq = meta.get("freq", "D")
    item_id = meta.get("item_id", "series")
    n_channels = values.shape[2]
    usable = values.shape[1]
    values = values[:, :usable]
    if natural_missing or meta.get("natural_missing") or "observed" in meta:
        natural_missing = True
        args.natural_missing = True
        if "observed" not in meta:
            raise ValueError(
                "natural-missing datasets must include an observed mask in the NPZ"
            )
        observed = np.asarray(meta["observed"], dtype=bool)[:, :usable]
        if observed.shape != values.shape:
            raise ValueError("observed mask shape mismatch")
        hidden_time = ~observed  # (1, T, C) per-channel natural gaps
    else:
        hidden_1d = build_observation_mask(
            usable,
            mode=args.missing_mode,
            sparse_step=args.sparse_step,
            block_length=args.block_length,
            observe_length=args.observe_length,
            keep_probability=args.keep_probability,
            seed=args.seed,
        )
        hidden_time = hidden_1d
        observed = ~expand_mask(hidden_1d, batch=1, channels=n_channels)

    train_v, val_v, _test_v = split_data(values, axis="time", **SPLITS)
    n_train, n_val = train_v.shape[1], val_v.shape[1]

    normalizer = fit_normalizer(train_v, observed=observed[:, :n_train])
    train_n = np.where(
        observed[:, :n_train], normalizer.transform(train_v), 0.0
    )
    val_n = np.where(
        observed[:, n_train : n_train + n_val],
        normalizer.transform(val_v),
        0.0,
    )
    full_n = np.where(observed, normalizer.transform(values), 0.0)

    train_windows = make_windows(
        train_n, window_length=args.window_length, stride=args.stride
    )
    val_windows = make_windows(
        val_n, window_length=args.window_length, stride=args.stride
    )
    if natural_missing:
        train_available = windowed_available(
            observed[:, :n_train],
            window=args.window_length,
            stride=args.stride,
        )
        val_available = windowed_available(
            observed[:, n_train : n_train + n_val],
            window=args.window_length,
            stride=args.stride,
        )
    else:
        available_time = ~hidden_time
        train_available = windowed_available(
            available_time[:n_train],
            window=args.window_length,
            stride=args.stride,
            channels=n_channels,
        )
        val_available = windowed_available(
            available_time[n_train : n_train + n_val],
            window=args.window_length,
            stride=args.stride,
            channels=n_channels,
        )
    train_ssl = sample_ssl_target_mask(
        train_available, rate=args.ssl_mask_rate, seed=args.seed
    )
    val_ssl = sample_ssl_target_mask(
        val_available, rate=args.ssl_mask_rate, seed=args.seed + 1
    )
    train_windows, train_available, train_ssl = filter_ssl_windows(
        train_windows, train_available, train_ssl
    )
    val_windows, val_available, val_ssl = filter_ssl_windows(
        val_windows, val_available, val_ssl
    )
    # Supervised masks only for synthetic missingness (dense GT exists).
    train_sup_masks = train_sup_windows = val_sup_masks = val_sup_windows = None
    train_sup_ok = val_sup_ok = np.array([], dtype=bool)
    if not natural_missing:
        train_sup_masks = windowed_masks(
            hidden_time[:n_train],
            window=args.window_length,
            stride=args.stride,
            channels=n_channels,
        )
        val_sup_masks = windowed_masks(
            hidden_time[n_train : n_train + n_val],
            window=args.window_length,
            stride=args.stride,
            channels=n_channels,
        )
        train_sup_ok = np.array(
            [mask.any() and not mask.all() for mask in train_sup_masks], dtype=bool
        )
        val_sup_ok = np.array(
            [mask.any() and not mask.all() for mask in val_sup_masks], dtype=bool
        )
        train_sup_windows = make_windows(
            train_n, window_length=args.window_length, stride=args.stride
        )[train_sup_ok]
        val_sup_windows = make_windows(
            val_n, window_length=args.window_length, stride=args.stride
        )[val_sup_ok]
        train_sup_masks = train_sup_masks[train_sup_ok]
        val_sup_masks = val_sup_masks[val_sup_ok]

    LOGGER.info(
        "%s selfsup natural=%s mode=%s ssl_rate=%.3f: T=%d C=%d; "
        "windows train/val=%d/%d; observed_fraction=%.3f",
        meta.get("dataset", "series"),
        natural_missing,
        args.missing_mode,
        args.ssl_mask_rate,
        usable,
        n_channels,
        len(train_windows),
        len(val_windows),
        float(observed.mean()),
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

    # Held-out SSL eval mask on available entries only.
    ssl_eval = sample_ssl_target_mask(
        observed, rate=args.ssl_mask_rate, seed=args.seed + 99
    )

    model = build_csdi(n_channels, args)
    model = fit_csdi_selfsup(
        model,
        args.output / "csdi.pt",
        train_windows,
        val_windows,
        train_ssl,
        val_ssl,
        train_available,
        val_available,
        config=csdi_config,
        seed=args.seed,
    )
    sampler = DDPMSampler(
        model, replace(csdi_config, batch_size=args.sample_batch_size)
    )

    shared = dict(
        args=args,
        values=values,
        full_n=full_n,
        train_v=train_v,
        n_train=n_train,
        n_val=n_val,
        hidden_time=hidden_time,
        observed=observed,
        normalizer=normalizer,
        channels=channels,
        series_start=series_start,
        series_freq=series_freq,
        item_id=item_id,
        meta=meta,
        usable=usable,
        ssl_eval_mask=ssl_eval,
        natural_missing=natural_missing,
    )
    results = run_imputation_and_forecast(
        output=args.output,
        sampler=sampler,
        protocol="selfsup",
        **shared,
    )
    results["elapsed_sec"] = time.time() - started
    write_json(args.output / "results.json", results)
    LOGGER.info("Self-sup results:\n%s", json.dumps(results["forecast"], indent=2))

    if args.supervised_baseline:
        if natural_missing:
            raise ValueError(
                "--supervised-baseline requires dense ground truth; "
                "not available with natural missingness"
            )
        sup_dir = args.output / "supervised"
        sup_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Training supervised baseline under the same base mask")
        sup_model = build_csdi(n_channels, args)
        if not train_sup_ok.any() or not val_sup_ok.any():
            raise ValueError("No valid supervised windows under the current missing mask")
        # Rebuild windows aligned with filter above (already filtered).
        sup_model = fit_csdi_supervised(
            sup_model,
            sup_dir / "csdi.pt",
            train_sup_windows,
            val_sup_windows,
            train_sup_masks,
            val_sup_masks,
            config=csdi_config,
            seed=args.seed,
        )
        sup_sampler = DDPMSampler(
            sup_model, replace(csdi_config, batch_size=args.sample_batch_size)
        )
        sup_results = run_imputation_and_forecast(
            output=sup_dir,
            sampler=sup_sampler,
            protocol="supervised",
            ssl_eval_mask=None,
            **{
                k: v
                for k, v in shared.items()
                if k not in {"ssl_eval_mask", "natural_missing"}
            },
            natural_missing=False,
        )
        # Drop unused ssl from supervised shared call — already excluded.
        write_json(sup_dir / "results.json", sup_results)
        results["supervised_baseline"] = {
            "imputation": sup_results["imputation"],
            "forecast": sup_results["forecast"],
            "deltas_csdi_minus_sparse": sup_results.get("deltas_csdi_minus_sparse"),
        }
        write_json(args.output / "results.json", results)
        LOGGER.info("Supervised baseline written under %s", sup_dir)

    try:
        from plot_csdi_chronos2_results import main as plot_main

        plot_main(["--run", str(args.output)])
        if args.supervised_baseline:
            plot_main(["--run", str(args.output / "supervised")])
        LOGGER.info("Wrote visualizations under %s", args.output)
    except Exception as exc:  # plotting is best-effort
        LOGGER.warning("Skipping result plots (%s)", exc)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=tuple(PRESETS), default="electricity")
    parser.add_argument("--data", type=Path, default=Path("data/electricity.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--missing-mode",
        choices=("block", "random"),
        default=None,
        help="Synthetic base sparsity (ignored with --natural-missing / intel_lab)",
    )
    parser.add_argument(
        "--natural-missing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use observed mask from the NPZ instead of synthetic missingness",
    )
    parser.add_argument("--sparse-step", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--observe-length", type=int, default=None)
    parser.add_argument("--keep-probability", type=float, default=None)
    parser.add_argument(
        "--ssl-mask-rate",
        type=float,
        default=None,
        help="Fraction of observed points held out as SSL training targets",
    )
    parser.add_argument(
        "--supervised-baseline",
        action="store_true",
        help="Also train/evaluate supervised densification under the same base mask",
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--impute-stride", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--forecast-stride", type=int, default=None)
    parser.add_argument("--max-origins", type=int, default=None)
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
        "--lr-decay-fractions", type=float, nargs="+", default=[0.75, 0.9]
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
    parser.add_argument("--chronos-model", default="amazon/chronos-2")
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--chronos-steps", type=int, default=1000)
    parser.add_argument("--chronos-lr", type=float, default=1e-4)
    parser.add_argument("--chronos-batch-size", type=int, default=32)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny CSDI + Chronos-2-small smoke settings",
    )
    args = parser.parse_args(argv)
    if args.preset == "intel_lab" and args.data == Path("data/electricity.npz"):
        args.data = Path("data/intel_lab.npz")
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.natural_missing is None:
        args.natural_missing = bool(PRESETS[args.preset].get("natural_missing", False))
    if args.window_length < 3:
        parser.error("--window-length must be at least 3")
    if args.season < 1:
        parser.error("--season must be positive")
    if not 0.0 < args.ssl_mask_rate < 1.0:
        parser.error("--ssl-mask-rate must be in (0, 1)")
    if (
        not args.natural_missing
        and args.missing_mode == "random"
        and not 0.0 < args.keep_probability < 1.0
    ):
        parser.error("--keep-probability must be in (0, 1)")
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
