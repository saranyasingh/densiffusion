"""Helpers for sparse FX imputation and Chronos-2 forecasting experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from densiffusion.contracts import Array, positive_int, validate_array

FX_URL = (
    "https://github.com/laiguokun/multivariate-time-series-data/raw/master/"
    "exchange_rate/exchange_rate.txt.gz"
)
FX_CURRENCIES = (
    "AUD",
    "GBP",
    "CAD",
    "CHF",
    "CNY",
    "JPY",
    "NZD",
    "SGD",
)
FX_START = np.datetime64("1990-01-01")

ELECTRICITY_URL = (
    "https://github.com/laiguokun/multivariate-time-series-data/raw/master/"
    "electricity/electricity.txt.gz"
)
ELECTRICITY_START = np.datetime64("2012-01-01T00")
ELECTRICITY_N_CLIENTS = 321


@dataclass(frozen=True)
class ChannelNormalizer:
    """Per-channel z-score fit on finite training observations only."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: Array) -> Array:
        values = validate_array(values, (None, None, len(self.mean)))
        return ((values - self.mean) / self.std).astype(np.float64, copy=False)

    def inverse(self, values: Array) -> Array:
        values = validate_array(values, (None, None, len(self.mean)))
        return (values * self.std + self.mean).astype(np.float64, copy=False)


def fit_normalizer(
    values: Array, *, observed: Array | None = None
) -> ChannelNormalizer:
    """Fit channel mean/std on observed entries; ``observed`` is True where known."""
    values = validate_array(values, (None, None, None))
    if observed is None:
        mask = np.isfinite(values)
    else:
        observed = np.asarray(observed, dtype=bool)
        if observed.shape != values.shape:
            raise ValueError("observed must match values shape")
        mask = observed & np.isfinite(values)
    mean = np.zeros(values.shape[-1], dtype=np.float64)
    std = np.ones(values.shape[-1], dtype=np.float64)
    for channel in range(values.shape[-1]):
        points = values[..., channel][mask[..., channel]]
        if points.size == 0:
            raise ValueError(f"channel {channel} has no observed training points")
        mean[channel] = float(points.mean())
        scale = float(points.std())
        std[channel] = scale if scale > 1e-8 else 1.0
    return ChannelNormalizer(mean=mean, std=std)


def sparse_observation_mask(
    length: int,
    *,
    step: int = 2,
    offset: int = 0,
) -> np.ndarray:
    """Return True at hidden positions for a regular observation grid.

    Observed indices are ``offset, offset+step, ...``. Endpoints are forced
    observed so every window has boundary conditioning.
    """
    positive_int("length", length)
    positive_int("step", step)
    if not 0 <= offset < step:
        raise ValueError("offset must satisfy 0 <= offset < step")
    hidden = np.ones(length, dtype=bool)
    hidden[offset::step] = False
    hidden[0] = False
    hidden[-1] = False
    if hidden.all() or not hidden.any():
        raise ValueError("sparse mask must leave both observed and hidden points")
    return hidden


def expand_mask(hidden_time: np.ndarray, *, batch: int, channels: int) -> np.ndarray:
    """Broadcast a length-T hidden mask to (batch, time, channels)."""
    hidden_time = np.asarray(hidden_time, dtype=bool)
    if hidden_time.ndim != 1:
        raise ValueError("hidden_time must be 1-D")
    return np.broadcast_to(
        hidden_time[None, :, None], (batch, len(hidden_time), channels)
    ).copy()


def linear_fill(values: Array, hidden: Array) -> Array:
    """Channel-wise linear interpolation of hidden entries; endpoints must be known."""
    values = validate_array(values, (None, None, None))
    hidden = np.asarray(hidden, dtype=bool)
    if hidden.shape != values.shape:
        raise ValueError("hidden must match values shape")
    filled = values.copy()
    times = np.arange(values.shape[1], dtype=np.float64)
    for series in range(values.shape[0]):
        for channel in range(values.shape[2]):
            known = ~hidden[series, :, channel]
            if not known.any():
                raise ValueError("each series/channel needs at least one observation")
            filled[series, :, channel] = np.interp(
                times, times[known], values[series, known, channel]
            )
    return filled


def load_multivariate_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load prepare_*.py output: values (1, T, C), times (T,), metadata.

    Channel names are read from ``channels`` or legacy ``currencies``.
    """
    with np.load(path, allow_pickle=False) as dataset:
        values = validate_array(dataset["values"], (None, None, None))
        times = np.asarray(dataset["times"], dtype=np.float64)
        if "channels" in dataset.files:
            channels = [str(name) for name in dataset["channels"].tolist()]
        elif "currencies" in dataset.files:
            channels = [str(name) for name in dataset["currencies"].tolist()]
        else:
            raise KeyError("dataset must contain 'channels' or 'currencies'")
        config = json.loads(str(dataset["config"]))
    if times.shape != (values.shape[1],):
        raise ValueError("times must have shape (n_steps,)")
    if len(channels) != values.shape[2]:
        raise ValueError("channel names length must match channel count")
    config = {
        **config,
        "channels": channels,
        "currencies": channels,  # back-compat for FX call sites
        "freq": config.get("freq", "D"),
        "start": config.get("start", str(FX_START)),
        "item_id": config.get("item_id", "series"),
    }
    return values, times, config


def load_fx_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load ``prepare_fx.py`` output: values (1, T, C), times (T,), metadata."""
    return load_multivariate_dataset(path)


def to_chronos_frame(
    values: Array,
    *,
    channels: Sequence[str] | None = None,
    currencies: Sequence[str] | None = None,
    start: np.datetime64 | str = FX_START,
    freq: str = "D",
    item_id: str = "series",
    step: int = 1,
):
    """Build a Chronos-2 DataFrame for one multivariate series.

    ``values`` is ``(1, T, C)`` or ``(T, C)`` on a regular grid. ``freq`` is the
    native spacing of one step (``D`` or ``h``); ``step>1`` uses ``f"{step}{freq}"``.
    """
    import pandas as pd

    names = list(channels if channels is not None else currencies or ())
    if not names:
        raise ValueError("channels or currencies must be provided")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("to_chronos_frame expects a single multivariate series")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            "values must have shape (time, channels) or (1, time, channels)"
        )
    if len(names) != array.shape[1]:
        raise ValueError("channel names length must match channel count")
    positive_int("step", step)
    if freq not in {"D", "h", "H"}:
        raise ValueError("freq must be 'D' or 'h'")
    unit = "h" if freq in {"h", "H"} else "D"
    timestamps = pd.date_range(
        start=pd.Timestamp(start),
        periods=array.shape[0],
        freq=unit if step == 1 else f"{step}{unit}",
    )
    frame = pd.DataFrame({"item_id": item_id, "timestamp": timestamps})
    for name, column in zip(names, array.T, strict=True):
        frame[name] = column
    return frame


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball loss averaged over all elements."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    error = y_true - y_pred
    return float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())


def weighted_quantile_loss(
    y_true: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> float:
    """Mean pinball loss across the supplied quantile forecasts."""
    if not quantile_forecasts:
        raise ValueError("quantile_forecasts must be non-empty")
    return float(
        np.mean(
            [
                quantile_loss(y_true, forecast, quantile)
                for quantile, forecast in quantile_forecasts.items()
            ]
        )
    )


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / sMAPE for matched forecast arrays."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    absolute = np.abs(y_true - y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = np.where(denom > 0, 2.0 * absolute / denom, 0.0)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
        "smape": float(smape.mean()),
    }


def extract_quantile_matrix(
    pred_df,
    *,
    currencies: Sequence[str],
    quantile_levels: Sequence[float],
    prediction_length: int,
) -> dict[float, np.ndarray]:
    """Reshape Chronos ``predict_df`` output to ``{q: (horizon, channels)}``."""
    ordered = pred_df.sort_values("timestamp")
    if len(ordered) % prediction_length != 0:
        # Multivariate predict_df returns one row per timestamp with all targets,
        # or stacked target rows; handle both.
        pass
    if "target_name" not in ordered.columns:
        raise KeyError(
            "Chronos predict_df output must include target_name "
            "(multivariate long format)"
        )
    matrices: dict[float, np.ndarray] = {}
    for quantile in quantile_levels:
        # Chronos-2 writes quantile columns as str(level), e.g. "0.1".
        column = str(quantile)
        if column not in ordered.columns and quantile == 0.5:
            column = "predictions"
        if column not in ordered.columns:
            raise KeyError(f"missing quantile column {column!r} in Chronos output")
        pieces = []
        for name in currencies:
            part = ordered.loc[ordered["target_name"] == name, column].to_numpy(
                dtype=np.float64
            )
            if len(part) != prediction_length:
                raise ValueError(
                    f"expected {prediction_length} forecasts for {name}, "
                    f"got {len(part)}"
                )
            pieces.append(part)
        matrices[float(quantile)] = np.stack(pieces, axis=1)
    return matrices
