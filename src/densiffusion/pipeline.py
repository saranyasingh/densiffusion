import numpy as np

from densiffusion.contracts import (
    Array,
    Densifier,
    Forecaster,
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
) -> Array:
    """Fill intermediate points at spacing dt / factor, preserving observations."""
    positive_int("factor", factor)
    positive_int("n_samples", n_samples)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    observed = validate_array(observed, (None, None, None))
    batch, steps, channels = observed.shape
    samples = sampler.sample(
        observed.copy(),
        factor=factor,
        n_samples=n_samples,
        dt=dt,
        rng=np.random.default_rng(seed),
    )
    samples = validate_array(
        samples, (batch, n_samples, (steps - 1) * factor + 1, channels)
    )
    if not np.all(samples[:, :, ::factor, :] == observed[:, None, :, :]):
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
