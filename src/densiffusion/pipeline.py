import numpy as np

from densiffusion.conditioning import prepare_observations
from densiffusion.contracts import (
    Array,
    Densifier,
    Forecaster,
    Mask,
    positive_int,
    validate_array,
)


def impute(
    observed: Array,
    sampler: Densifier,
    *,
    factor: int = 2,
    n_samples: int = 1,
    dt: float = 1.0,
    seed: int = 0,
    mask: Mask | None = None,
    times: Array | None = None,
) -> Array:
    """Refine a grid or fill mask=True entries at factor=1, preserving observations."""
    positive_int("n_samples", n_samples)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    conditioning, hidden, target_times = prepare_observations(
        observed, factor=factor, dt=dt, mask=mask, times=times
    )
    options = {}
    if mask is not None:
        options["mask"] = hidden.copy()
    if times is not None:
        options["times"] = target_times[:, ::factor].copy()
    samples = sampler.sample(
        conditioning[:, ::factor].copy(),
        factor=factor,
        n_samples=n_samples,
        dt=dt,
        rng=np.random.default_rng(seed),
        **options,
    )
    samples = validate_array(
        samples, (conditioning.shape[0], n_samples, *conditioning.shape[1:])
    )
    if not np.where(hidden[:, None], True, samples == conditioning[:, None]).all():
        raise ValueError("Densifier must preserve every observed value")
    return samples.copy()


def forecast(
    observed: Array,
    densifier: Densifier,
    forecaster: Forecaster,
    *,
    horizon: int,
    factor: int = 2,
    n_histories: int = 1,
    n_futures: int = 1,
    dt: float = 1.0,
    seed: int = 0,
) -> Array:
    """Return (batch, n_histories * n_futures, horizon, channels) at spacing dt."""
    raise NotImplementedError("Forecast sampling is not implemented yet.")
