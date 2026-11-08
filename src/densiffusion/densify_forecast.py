"""Densify-diffuse-downsample experiments with any of the repository's denoisers.

Every network is an `ImputationModel` trained with `noise_mse` and sampled with
`DDPMSampler`, so PriSTI, CSDI, SSSD and TimeAutoDiff are interchangeable.
`small_model` builds each at about 45k parameters. Arrays here are
(windows, time) on a regular grid; the channel axis is added only when calling
the shared trainer and sampler.

Imputer: a model trained on windows of 2 * CHUNK - 1 points with alternate
points hidden (2*s -> s refinement). Windows can come from several strides of
the observed data, and each window is normalised by the mean and RMS increment
of its observed points, so a model trained at coarse spacings can be applied
at finer ones. `densify` doubles the resolution repeatedly, refining
non-overlapping chunks of CHUNK observed points independently.

Forecaster: a model trained with the last points of each window hidden. A
window holds `history` coarse points and `horizon` coarse future points,
optionally refined by `factor`; it is normalised by its coarse history points
only, so sparse and densified forecasters share one normalisation. `forecast`
returns samples at the coarse future times, i.e. after downsampling.
"""

import dataclasses

import numpy as np

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.architectures.pristi import PriSTIDenoiser, PriSTIEncoder
from densiffusion.architectures.sssd import SSSDDenoiser
from densiffusion.architectures.timeautodiff import (
    TimeAutoDiffDenoiser,
    TimeAutoDiffEncoder,
)
from densiffusion.models import ImputationModel
from densiffusion.objectives import noise_mse
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

CHUNK = 5
SMALL = {"width": 32, "heads": 4, "feedforward": 32, "time_dim": 32, "channel_dim": 8}
SMALL_CSDI = {
    "hidden_channels": 32,
    "residual_layers": 2,
    "attention_heads": 4,
    "diffusion_embedding_dim": 32,
    "time_embedding_dim": 32,
    "feature_embedding_dim": 8,
    "feedforward_dim": 32,
}
SMALL_SSSD = {"width": 24, "layers": 2, "state_size": 32, "step_dim": 32}
SMALL_TIMEAUTODIFF = {"hidden": 24, "layers": 2, "embedding": 32, "terms": 4}


def small_pristi(*, layers: int = 2, step_dim: int = 32, **sizes) -> ImputationModel:
    """A reduced PriSTI: 44k parameters, about 8x fewer than the defaults."""
    sizes = {**SMALL, **sizes}
    return ImputationModel(
        PriSTIDenoiser(layers=layers, step_dim=step_dim, **sizes),
        encoder=PriSTIEncoder(**sizes),
    )


def small_csdi(**sizes) -> ImputationModel:
    """A reduced univariate CSDI: 45k parameters, two residual layers of width 32."""
    return ImputationModel(CSDIDenoiser(1, **{**SMALL_CSDI, **sizes}))


def small_sssd(**sizes) -> ImputationModel:
    """A reduced univariate SSSD: 46k parameters, two S4 blocks of width 24."""
    return ImputationModel(SSSDDenoiser(1, **{**SMALL_SSSD, **sizes}))


def small_timeautodiff(*, diffusion_steps: int = 50, **sizes) -> ImputationModel:
    """A reduced TimeAutoDiff: 48k parameters, two recurrent layers of width 24.

    diffusion_steps scales the step encoding and should match the schedule.
    """
    sizes = {**SMALL_TIMEAUTODIFF, **sizes}
    embedding, terms = sizes.pop("embedding"), sizes.pop("terms")
    return ImputationModel(
        TimeAutoDiffDenoiser(diffusion_steps=diffusion_steps, **sizes),
        encoder=TimeAutoDiffEncoder(embedding=embedding, terms=terms, **sizes),
    )


SMALL_MODELS = {
    "pristi": small_pristi,
    "csdi": small_csdi,
    "sssd": small_sssd,
    "timeautodiff": small_timeautodiff,
}
MODEL_LABELS = {
    "pristi": "PriSTI",
    "csdi": "CSDI",
    "sssd": "SSSD",
    "timeautodiff": "TimeAutoDiff",
}


def small_model(name: str, **sizes) -> ImputationModel:
    """One of the reduced models in SMALL_MODELS, with optional size overrides."""
    if name not in SMALL_MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(SMALL_MODELS)}")
    return SMALL_MODELS[name](**sizes)


def window_scale(values: np.ndarray, reference: np.ndarray):
    """Mean and RMS increment of each row over the reference positions."""
    points = values[:, reference]
    location = points.mean(axis=1)
    scale = np.sqrt((np.diff(points, axis=1) ** 2).mean(axis=1))
    return location, np.where(scale > 1e-12, scale, 1.0)


def normalise(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    location, scale = window_scale(values, reference)
    return (values - location[:, None]) / scale[:, None]


def imputation_windows(paths: np.ndarray, strides=(1, 2, 4)) -> np.ndarray:
    """Non-overlapping imputer windows from paths subsampled at each stride."""
    length = 2 * CHUNK - 1
    windows = []
    for stride in strides:
        series = paths[:, ::stride]
        for start in range(0, series.shape[1] - length + 1, length - 1):
            windows.append(series[:, start : start + length])
    if not windows:
        raise ValueError("Paths are too short for any imputer window")
    return np.concatenate(windows)


def train_imputer(
    model, train, validation, config: TrainingConfig, *, seed: int, checkpoint=None
):
    """Train alternate-point refinement on per-window normalised windows."""
    observed = np.arange(train.shape[1]) % 2 == 0
    return train_densifier(
        model,
        normalise(train, observed)[..., None],
        normalise(validation, observed)[..., None],
        config=config,
        objective=noise_mse,
        dt=1.0,
        seed=seed,
        checkpoint=checkpoint,
    )


def refine(model, config, series, *, n_samples: int, seed: int) -> np.ndarray:
    """Double the resolution of (windows, n) series; returns (windows, draws, 2n-1).

    Chunks of CHUNK observed points are normalised and refined independently,
    so a chunk boundary at a multiple of CHUNK - 1 never mixes information.
    """
    windows, length = series.shape
    if (length - 1) % (CHUNK - 1):
        raise ValueError(f"Series length minus one must be a multiple of {CHUNK - 1}")
    starts = range(0, length - 1, CHUNK - 1)
    chunks = np.stack([series[:, s : s + CHUNK] for s in starts], axis=1)
    chunks = chunks.reshape(-1, CHUNK)
    location, scale = window_scale(chunks, np.ones(CHUNK, dtype=bool))
    samples = impute(
        ((chunks - location[:, None]) / scale[:, None])[..., None],
        DDPMSampler(model, config),
        factor=2,
        n_samples=n_samples,
        dt=2.0,
        seed=seed,
    )[..., 0]
    samples = samples * scale[:, None, None] + location[:, None, None]
    samples = samples.reshape(windows, len(starts), n_samples, 2 * CHUNK - 1)
    fine = np.empty((windows, n_samples, 2 * length - 1))
    fine[:, :, ::2] = series[:, None]
    for index, start in enumerate(starts):
        stop = 2 * start + 2 * CHUNK - 2
        fine[:, :, 2 * start + 1 : stop : 2] = samples[:, index, :, 1::2]
    return fine


def densify(model, config, series, *, levels: int, n_samples: int, seed: int):
    """Refine `levels` times; returns [(windows, draws, length)] for each level.

    The first refinement draws n_samples realisations; each realisation is
    then refined once per level, so level k holds 2**k-times resolution.
    `model` is one imputer for every level or a sequence of one per level.
    """
    models = list(model) if isinstance(model, (list, tuple)) else [model] * levels
    if len(models) != levels:
        raise ValueError("Pass one imputer, or one per level")
    rng = np.random.default_rng(seed)
    windows = len(series)
    current, results = series, []
    for level, imputer in enumerate(models):
        draws = n_samples if level == 0 else 1
        fine = refine(
            imputer, config, current, n_samples=draws, seed=int(rng.integers(2**62))
        )
        current = fine.reshape(-1, fine.shape[-1])
        results.append(current.reshape(windows, n_samples, -1))
    return results


def interpolate(series: np.ndarray, factor: int) -> np.ndarray:
    """Linear interpolation of (windows, n) series at factor-times resolution."""
    coarse = np.arange(series.shape[1]) * factor
    fine = np.arange(coarse[-1] + 1)
    return np.stack([np.interp(fine, coarse, row) for row in series])


def forecast_layout(history: int, horizon: int, factor: int = 1, joint=False):
    """Hidden future and coarse-history reference masks for a refined window.

    With joint=True the history points between coarse observations are hidden
    too, so one model fills in the history and forecasts the future together.
    """
    length = (history + horizon - 1) * factor + 1
    hidden = np.zeros(length, dtype=bool)
    hidden[(history - 1) * factor + 1 :] = True
    if joint:
        hidden[: (history - 1) * factor + 1] = (
            np.arange((history - 1) * factor + 1) % factor != 0
        )
    reference = np.zeros(length, dtype=bool)
    reference[: (history - 1) * factor + 1 : factor] = True
    return hidden, reference


def train_forecaster(
    model,
    train,
    validation,
    config: TrainingConfig,
    *,
    history: int,
    horizon: int,
    factor: int = 1,
    seed: int,
    checkpoint=None,
    joint: bool = False,
):
    """Train future prediction on (windows, time) windows at factor resolution.

    With joint=True the loss also covers the hidden history points (see
    forecast_layout); train on true fine windows.
    """
    hidden, reference = forecast_layout(history, horizon, factor, joint)
    if train.shape[1] != hidden.size:
        raise ValueError("Windows do not match history, horizon and factor")
    return train_densifier(
        model,
        normalise(train, reference)[..., None],
        normalise(validation, reference)[..., None],
        config=config,
        objective=noise_mse,
        dt=1.0,
        seed=seed,
        checkpoint=checkpoint,
        train_mask=hidden,
        validation_mask=hidden,
    )


def forecast(
    model,
    config,
    histories,
    *,
    horizon: int,
    factor: int = 1,
    n_samples: int,
    seed: int,
    downsample: bool = True,
    joint: bool = False,
):
    """Sample futures for (windows, history) series; returns coarse future values.

    Histories are at factor resolution, or coarse with joint=True, where the
    model fills in the history and samples the future together. Output is
    (windows, n_samples, horizon) at the coarse future times, or with
    downsample=False (windows, n_samples, horizon * factor) at every future point.
    """
    windows = len(histories)
    if joint:
        history = histories.shape[1]
        length = (history - 1) * factor + 1
    else:
        length = histories.shape[1]
        history = (length - 1) // factor + 1
    hidden, reference = forecast_layout(history, horizon, factor, joint)
    values = np.zeros((windows, hidden.size))
    if joint:
        values[:, :length:factor] = histories
    else:
        values[:, :length] = histories
    location, scale = window_scale(values, reference)
    samples = impute(
        ((values - location[:, None]) / scale[:, None])[..., None],
        DDPMSampler(model, config),
        factor=1,
        mask=hidden,
        n_samples=n_samples,
        dt=1.0,
        seed=seed,
    )[..., 0]
    samples = samples * scale[:, None, None] + location[:, None, None]
    if not downsample:
        return samples[:, :, length:]
    return samples[:, :, length - 1 + factor :: factor]


def sampling_config(config: TrainingConfig, batch_size: int = 512) -> TrainingConfig:
    """The training schedule with a larger batch for faster sampling."""
    return dataclasses.replace(config, batch_size=batch_size)
