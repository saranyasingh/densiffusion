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

With ``--protocol selfsup``, CSDI trains only on observed values, as in
``run_csdi_chronos2_selfsup.py``: some observed time steps are hidden as targets
(by default 1-2 extra hours per window, so training inputs are as sparse as test
inputs) and the base-missing values are neither conditioned on nor scored.

With ``--per-draw-forecasts K``, Chronos also forecasts from each of the first K
CSDI draws separately and pools the forecasts (``zeroshot_csdi_draws``), so
uncertainty about the gaps carries into the forecast instead of being collapsed
by the median. Every arm then also gets CRPS, from samples of its quantile
forecast. ``--stitch block`` keeps each draw a single CSDI sample (see
``impute_series``).
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
    random_block_observation_mask,
    sample_ssl_target_mask,
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
# Wider levels for CRPS: each quantile forecast is sampled at QUANTILE_SAMPLES
# evenly spaced probabilities, flat beyond the 1% and 99% levels.
FINE_QUANTILES = (0.01, 0.05, *QUANTILES, 0.95, 0.99)
QUANTILE_SAMPLES = 20
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
        min_gap=2,
        max_gap=12,
        min_observed=2,
        max_observed=12,
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
        min_gap=1,
        max_gap=7,
        min_observed=1,
        max_observed=7,
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
    train_availability=None,
    validation_availability=None,
):
    if checkpoint.exists():
        LOGGER.info("Loading CSDI checkpoint %s", checkpoint)
        state = torch.load(checkpoint, map_location=config.device, weights_only=False)
        model.to(config.device).load_state_dict(state["model_state_dict"])
        return model.eval()
    if validation_masks is None:
        validation_masks = np.broadcast_to(train_mask, validation.shape).copy()
    availability = {}
    if train_availability is not None:
        # Self-supervised: unavailable values are neither inputs nor targets.
        availability = dict(
            train_availability=train_availability,
            validation_availability=validation_availability,
        )
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
        **availability,
    )


def hidden_runs(hidden: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of each contiguous run of True in a 1-D mask."""
    padded = np.concatenate([[0], np.asarray(hidden, dtype=np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def impute_series(
    values: np.ndarray,
    sampler: DDPMSampler,
    *,
    hidden_time: np.ndarray,
    window: int,
    stride: int,
    n_samples: int,
    seed: int,
    stitch: str = "average",
) -> np.ndarray:
    """Fill hidden entries of (1, T, C) with overlapping-window CSDI draws.

    Returns ``(n_samples, T, C)``. Observed positions are preserved exactly.
    With ``stitch="average"``, overlapping hidden predictions are averaged
    across windows; this blends independent draws, so each returned draw is
    smoother than a CSDI sample. With ``stitch="block"``, each contiguous
    hidden run is taken whole from the window that contains it most centrally,
    so every draw stays a single CSDI sample (runs that no window contains are
    still averaged).
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
    hidden = np.asarray(hidden_time, dtype=bool)
    owned_runs: dict[int, list[tuple[int, int]]] = {}
    owned = np.zeros(length, dtype=bool)
    if stitch == "block":
        for run_start, run_stop in hidden_runs(hidden):
            inside = [
                i
                for i, s in enumerate(starts)
                if s <= run_start and run_stop <= s + window
            ]
            if inside:
                centre = (run_start + run_stop) / 2
                best = min(inside, key=lambda i: abs(starts[i] + window / 2 - centre))
                owned_runs.setdefault(best, []).append((run_start, run_stop))
                owned[run_start:run_stop] = True
    elif stitch != "average":
        raise ValueError("stitch must be average or block")
    result = np.repeat(series, n_samples, axis=0)
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
        for run_start, run_stop in owned_runs.get(index, ()):
            local = slice(run_start - start, run_stop - start)
            result[:, run_start:run_stop] = draws[:, local]
        # Average only at positions that were hidden in this window and that
        # no single window owns.
        shared = np.asarray(local_hidden, dtype=bool) & ~owned[start:stop]
        total[:, start:stop] += draws * shared[None, :, None]
        weight[start:stop] += shared.astype(np.float64)
    # Observed entries: copy through exactly. Hidden entries: owned runs were
    # copied above; average the rest.
    averaged = hidden & ~owned
    if averaged.any():
        if (weight[averaged] == 0).any():
            raise RuntimeError("imputation windows failed to cover every hidden point")
        scale = weight[averaged][None, :, None]
        result[:, averaged] = total[:, averaged] / scale
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


def chronos_forecast_draws(
    pipeline,
    contexts: list[np.ndarray],
    *,
    channels,
    prediction_length: int,
    start: np.datetime64,
    freq: str,
    item_id: str,
    quantile_levels=QUANTILES,
) -> list[dict[float, np.ndarray]]:
    """Forecast each (1, T, C) context as its own item in one Chronos call.

    Each context is one whole CSDI draw across all channels, so Chronos sees
    every channel of the same draw together.
    """
    import pandas as pd

    names = [f"{item_id}-draw{k}" for k in range(len(contexts))]
    frame = pd.concat(
        [
            to_chronos_frame(
                context,
                channels=channels,
                start=start,
                freq=freq,
                item_id=name,
                step=1,
            )
            for context, name in zip(contexts, names, strict=True)
        ],
        ignore_index=True,
    )
    pred = pipeline.predict_df(
        frame,
        prediction_length=prediction_length,
        quantile_levels=list(quantile_levels),
        id_column="item_id",
        timestamp_column="timestamp",
        target=list(channels),
    )
    return [
        extract_quantile_matrix(
            pred[pred["item_id"] == name],
            currencies=channels,
            quantile_levels=quantile_levels,
            prediction_length=prediction_length,
        )
        for name in names
    ]


def quantile_samples(quantiles: dict[float, np.ndarray], n: int) -> np.ndarray:
    """``(n, horizon, channels)`` draws from a quantile forecast.

    Linearly interpolates the quantile function at probabilities
    ``(j + 0.5) / n``, flat beyond the outermost levels. Crossed quantiles are
    sorted first.
    """
    levels = np.array(sorted(quantiles))
    values = np.sort(np.stack([quantiles[q] for q in levels]), axis=0)
    probabilities = (np.arange(n) + 0.5) / n
    upper = np.clip(np.searchsorted(levels, probabilities), 1, len(levels) - 1)
    lower = upper - 1
    weight = (probabilities - levels[lower]) / (levels[upper] - levels[lower])
    weight = np.clip(weight, 0.0, 1.0)[:, None, None]
    return values[lower] * (1 - weight) + values[upper] * weight


def add_crps(scores: dict, truth: np.ndarray, samples: np.ndarray, *, scale: float):
    """Add CRPS of ``(n, horizon, channels)`` samples, raw and over ``scale``.

    For a point forecast (n = 1) CRPS equals the mean absolute error.
    """
    value = crps(samples[None], truth[None])
    return {**scores, "crps": value, "scaled_crps": value / scale}


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


def extra_run_targets(available, *, min_length, max_length, seed):
    """Hide one run of ``min_length``-``max_length`` observed steps per window.

    The run lies wholly inside observed data and away from the window's first
    and last steps, so the window stays only slightly sparser than before.
    """
    rng = np.random.default_rng(seed)
    observed = available.any(axis=-1)
    targets = np.zeros_like(available, dtype=bool)
    for index, row in enumerate(observed):
        length = int(rng.integers(min_length, max_length + 1))
        sums = np.convolve(row.astype(int), np.ones(length, dtype=int), "valid")
        starts = np.flatnonzero(sums == length)
        starts = starts[(starts > 0) & (starts + length < len(row))]
        if starts.size:
            start = int(rng.choice(starts))
            targets[index, start : start + length] = available[
                index, start : start + length
            ]
    return targets


def ssl_targets(
    available,
    *,
    kind,
    rate,
    min_gap,
    max_gap,
    seed,
    extra_min=1,
    extra_max=2,
):
    """Self-supervised targets: observed entries hidden for CSDI to fill in.

    ``kind="extra"`` hides one run of ``extra_min``-``extra_max`` observed
    steps per window (see ``extra_run_targets``), so training inputs are about
    as sparse as test inputs. ``kind="points"`` hides each available time step
    with probability ``rate``, as in ``run_csdi_chronos2_selfsup.py``.
    ``kind="gaps"`` hides runs of ``min_gap``-``max_gap`` steps, spaced so that
    about ``rate`` of each window is hidden. Targets are shared across channels
    and never include naturally missing values.
    """
    if kind == "extra":
        return extra_run_targets(
            available, min_length=extra_min, max_length=extra_max, seed=seed
        )
    if kind == "points":
        return sample_ssl_target_mask(available, rate=rate, seed=seed)
    if kind != "gaps":
        raise ValueError("kind must be extra, points or gaps")
    mean_gap = (min_gap + max_gap) / 2
    # Observed runs uniform on [1, max_observed] have mean gap * (1 - rate) / rate.
    max_observed = max(1, round(2 * mean_gap * (1 - rate) / rate - 1))
    rng = np.random.default_rng(seed)
    targets = np.zeros_like(available, dtype=bool)
    for index in range(len(available)):
        try:
            gaps = random_block_observation_mask(
                available.shape[1],
                min_gap=min_gap,
                max_gap=max_gap,
                min_observed=1,
                max_observed=max_observed,
                seed=int(rng.integers(2**32)),
            )
        except ValueError:  # no gap fell inside this window
            continue
        targets[index] = gaps[:, None] & available[index]
    return targets


def filter_ssl_windows(values, available, targets):
    """Keep windows with targets and observed values left to condition on."""
    keep = np.array(
        [t.any() and (a & ~t).any() for a, t in zip(available, targets, strict=True)],
        dtype=bool,
    )
    if not keep.any():
        raise ValueError("No windows with SSL targets under this mask and rate")
    return values[keep], available[keep], targets[keep]


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
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        min_observed=args.min_observed,
        max_observed=args.max_observed,
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
    window_kwargs = dict(
        window=args.window_length, stride=args.stride, channels=n_channels
    )
    train_hidden = hidden_time[:n_train]
    val_hidden = hidden_time[n_train : n_train + n_val]
    train_available = val_available = None
    if args.protocol == "selfsup":
        # Train only on observed values: hide some as targets; base-missing
        # values are neither conditioned on nor scored. Each window appears
        # --ssl-copies times with different targets, like redrawing them
        # every epoch (epochs are divided by the same factor below).
        copies = args.ssl_copies
        train_windows = np.repeat(train_windows, copies, axis=0)
        val_windows = np.repeat(val_windows, copies, axis=0)
        train_available = np.repeat(
            windowed_masks(~train_hidden, **window_kwargs), copies, axis=0
        )
        val_available = np.repeat(
            windowed_masks(~val_hidden, **window_kwargs), copies, axis=0
        )
        ssl = dict(
            kind=args.ssl_targets,
            rate=args.ssl_mask_rate,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            extra_min=args.ssl_extra_min,
            extra_max=args.ssl_extra_max,
        )
        train_windows, train_available, train_masks = filter_ssl_windows(
            train_windows,
            train_available,
            ssl_targets(train_available, seed=args.seed, **ssl),
        )
        val_windows, val_available, val_masks = filter_ssl_windows(
            val_windows,
            val_available,
            ssl_targets(val_available, seed=args.seed + 1, **ssl),
        )
    else:
        train_masks = windowed_masks(train_hidden, **window_kwargs)
        val_masks = windowed_masks(val_hidden, **window_kwargs)
        # Drop windows that are fully observed or fully hidden.
        train_ok = np.array(
            [mask.any() and not mask.all() for mask in train_masks], dtype=bool
        )
        val_ok = np.array(
            [mask.any() and not mask.all() for mask in val_masks], dtype=bool
        )
        if not train_ok.any() or not val_ok.any():
            raise ValueError(
                "No valid imputation windows under the current missing mask"
            )
        train_windows, train_masks = train_windows[train_ok], train_masks[train_ok]
        val_windows, val_masks = val_windows[val_ok], val_masks[val_ok]
    LOGGER.info(
        "%s mode=%s protocol=%s: T=%d C=%d; windows train/val=%d/%d; "
        "freq=%s season=%d",
        meta.get("dataset", "series"),
        args.missing_mode,
        args.protocol,
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
    if args.protocol == "selfsup" and args.ssl_copies > 1:
        # Same number of updates as without copies.
        csdi_config = replace(
            csdi_config,
            epochs=max(1, args.epochs // args.ssl_copies),
            validation_interval=max(1, args.validation_interval // args.ssl_copies),
        )
    protocol_tag = (
        "" if args.protocol == "supervised" else f"_selfsup_{args.ssl_targets}"
    )
    model = build_csdi(n_channels, args)
    model = fit_csdi(
        model,
        args.output / f"csdi{protocol_tag}.pt",
        train_windows,
        val_windows,
        train_masks,
        validation_masks=val_masks,
        config=csdi_config,
        seed=args.seed,
        train_availability=train_available,
        validation_availability=val_available,
    )
    sampler = DDPMSampler(
        model, replace(csdi_config, batch_size=args.sample_batch_size)
    )

    suffix = protocol_tag + ("" if args.stitch == "average" else f"_{args.stitch}")
    imputed_path = args.output / f"imputed{suffix}.npz"
    if imputed_path.exists():
        LOGGER.info("Loading imputations from %s", imputed_path)
        with np.load(imputed_path, allow_pickle=False) as payload:
            imputed_norm = payload["imputed_norm"]
            draws_norm = payload["draws_norm"]
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
            stitch=args.stitch,
        )
        draws_norm = draws
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

    # Per-draw arm: each of the first K draws, whole across channels.
    per_draw = args.per_draw_forecasts
    if per_draw > draws_norm.shape[0]:
        raise ValueError(
            f"--per-draw-forecasts {per_draw} exceeds the {draws_norm.shape[0]} "
            "saved CSDI draws"
        )
    draw_series = [normalizer.inverse(draws_norm[k : k + 1]) for k in range(per_draw)]
    levels = FINE_QUANTILES if per_draw else QUANTILES
    n_crps = per_draw * QUANTILE_SAMPLES

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
        *(["zeroshot_csdi_draws"] if per_draw else []),
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
        last_value = np.broadcast_to(history[-1][None, :], truth.shape)
        for name, point in (("seasonal_naive", naive), ("last_value", last_value)):
            scores = score_forecast(truth, None, scale=scale, point=point)
            if per_draw:
                scores = add_crps(scores, truth, point[None], scale=scale)
            arm_scores[name].append(scores)

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
                quantile_levels=levels,
                validate_inputs=not allow_nan,
                **forecast_kwargs,
            )
            # The original metrics always use the standard levels.
            standard = {q: quantiles[q] for q in QUANTILES}
            scores = score_forecast(truth, standard, scale=scale)
            if per_draw:
                samples = quantile_samples(quantiles, n_crps)
                scores = add_crps(scores, truth, samples, scale=scale)
            arm_scores[name].append(scores)

        if per_draw:
            forecasts = chronos_forecast_draws(
                pipeline,
                [series[:, origin - args.context : origin] for series in draw_series],
                prediction_length=args.horizon,
                start=start,
                quantile_levels=levels,
                **forecast_kwargs,
            )
            pooled = np.concatenate(
                [quantile_samples(f, QUANTILE_SAMPLES) for f in forecasts]
            )
            pooled_quantiles = {q: np.quantile(pooled, q, axis=0) for q in QUANTILES}
            scores = score_forecast(truth, pooled_quantiles, scale=scale)
            arm_scores["zeroshot_csdi_draws"].append(
                add_crps(scores, truth, pooled, scale=scale)
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
                score_forecast(truth, quantiles, scale=scale)
            )

    def summarize(scores: list[dict]) -> dict[str, float]:
        if not scores:
            return {}
        keys = scores[0].keys()
        return {key: float(np.mean([row[key] for row in scores])) for key in keys}

    results = {
        "protocol": args.protocol,
        "observed_fraction": float((~hidden_time).mean()),
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
    forecast = results["forecast"]
    for arm, baseline in (
        ("zeroshot_csdi_draws", "zeroshot_csdi"),
        ("zeroshot_csdi_draws", "zeroshot_linear"),
    ):
        if arm in forecast and baseline in forecast:
            results[f"deltas_{arm}_minus_{baseline}"] = {
                key: forecast[arm][key] - forecast[baseline][key]
                for key in forecast[baseline]
                if key in forecast[arm]
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
        choices=("grid", "block", "random", "random_block"),
        default=None,
        help="grid=every sparse-step; block=contiguous gaps at fixed times; "
        "random=Bernoulli keep; random_block=gaps of random length at random times",
    )
    parser.add_argument("--sparse-step", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--observe-length", type=int, default=None)
    parser.add_argument("--keep-probability", type=float, default=None)
    for name in ("min-gap", "max-gap", "min-observed", "max-observed"):
        parser.add_argument(
            f"--{name}",
            type=int,
            default=None,
            help="random_block: run lengths drawn uniformly between min and max",
        )
    parser.add_argument("--season", type=int, default=None, help="MASE/SQL season")
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--impute-stride", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--aggregate", choices=("mean", "median"), default="median")
    parser.add_argument(
        "--protocol",
        choices=("supervised", "selfsup"),
        default="supervised",
        help="supervised: CSDI learns from the true values in the gaps; selfsup: "
        "only from observed values, hiding some of them as targets",
    )
    parser.add_argument(
        "--ssl-targets",
        choices=("extra", "points", "gaps"),
        default="extra",
        help="selfsup targets: one extra run of --ssl-extra-min to --ssl-extra-max "
        "observed steps per window (default); single steps at --ssl-mask-rate (as "
        "in the selfsup script); or runs like the real gaps at --ssl-mask-rate",
    )
    parser.add_argument("--ssl-extra-min", type=int, default=1)
    parser.add_argument("--ssl-extra-max", type=int, default=2)
    parser.add_argument(
        "--ssl-copies",
        type=int,
        default=8,
        help="selfsup: copies of each window with different targets; epochs are "
        "divided by this, so training costs the same",
    )
    parser.add_argument(
        "--ssl-mask-rate",
        type=float,
        default=0.25,
        help="selfsup: fraction of observed time steps hidden as targets",
    )
    parser.add_argument(
        "--per-draw-forecasts",
        type=int,
        default=0,
        help="Also forecast from each of the first K CSDI draws and pool the "
        "forecasts (zeroshot_csdi_draws); adds CRPS for every arm",
    )
    parser.add_argument(
        "--stitch",
        choices=("average", "block"),
        default="average",
        help="Combine overlapping imputation windows by averaging (default) or "
        "by taking each gap whole from one window, so draws stay CSDI samples",
    )
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
    if not 0.0 < args.ssl_mask_rate < 1.0:
        parser.error("--ssl-mask-rate must be in (0, 1)")
    if not 1 <= args.ssl_extra_min <= args.ssl_extra_max:
        parser.error("need 1 <= --ssl-extra-min <= --ssl-extra-max")
    if args.ssl_copies < 1:
        parser.error("--ssl-copies must be positive")
    if args.per_draw_forecasts < 0:
        parser.error("--per-draw-forecasts must be non-negative")
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.n_samples = min(args.n_samples, 2)
        args.per_draw_forecasts = min(args.per_draw_forecasts, args.n_samples)
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
    if args.per_draw_forecasts > args.n_samples:
        parser.error("--per-draw-forecasts cannot exceed --n-samples")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
