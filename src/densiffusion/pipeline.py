import numpy as np

from densiffusion.contracts import (
    Array,
    Densifier,
    Forecaster,
    positive_int,
    validate_array,
)


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
    for name, value in (
        ("horizon", horizon),
        ("factor", factor),
        ("n_histories", n_histories),
        ("n_futures", n_futures),
    ):
        positive_int(name, value)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    observed = validate_array(observed, (None, None, None))
    batch, steps, channels = observed.shape
    history_seed, future_seed = np.random.SeedSequence(seed).spawn(2)
    histories = densifier.sample(
        observed.copy(),
        factor=factor,
        n_samples=n_histories,
        dt=dt,
        rng=np.random.default_rng(history_seed),
    )
    histories = validate_array(
        histories, (batch, n_histories, (steps - 1) * factor + 1, channels)
    )
    if not np.all(histories[:, :, ::factor, :] == observed[:, None, :, :]):
        raise ValueError("Densifier must preserve every observed value")

    rng = np.random.default_rng(future_seed)
    samples = []
    for history in np.moveaxis(histories, 1, 0):
        futures = forecaster.sample(
            observed.copy(),
            history.copy(),
            horizon=horizon * factor,
            n_samples=n_futures,
            dt=dt / factor,
            rng=rng,
        )
        futures = validate_array(
            futures, (batch, n_futures, horizon * factor, channels)
        )
        samples.append(futures[:, :, factor - 1 :: factor, :].copy())
    return np.concatenate(samples, axis=1)
