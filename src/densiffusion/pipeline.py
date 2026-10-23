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
    """Marginalize sampled dense histories and return forecasts at spacing dt.

    Returns (batch, n_histories * n_futures, horizon, channels), grouping each
    history's future draws together. The frozen densifier sees only observed
    history. The forecaster receives the original history and one sampled dense
    history, and forecasts at spacing dt / factor before coarse-grid selection.
    At factor=1 the observed history is used directly without a densifier call.
    """
    observed = validate_array(observed, (None, None, None))
    positive_int("horizon", horizon)
    positive_int("factor", factor)
    positive_int("n_histories", n_histories)
    positive_int("n_futures", n_futures)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    fine_dt = dt / factor
    if not np.isfinite(fine_dt) or fine_dt <= 0:
        raise ValueError("dt / factor must be finite and positive")

    history_seed, future_seed = np.random.SeedSequence(seed).spawn(2)
    if factor == 1:
        histories = np.repeat(observed[:, None], n_histories, axis=1)
    else:
        histories = impute(
            observed,
            densifier,
            factor=factor,
            n_samples=n_histories,
            dt=dt,
            seed=int(history_seed.generate_state(1, dtype=np.uint64)[0]),
        )
    batch, context_length, channels = observed.shape
    dense_length = (context_length - 1) * factor + 1
    futures = forecaster.sample(
        np.repeat(observed, n_histories, axis=0),
        histories.reshape(batch * n_histories, dense_length, channels).copy(),
        horizon=horizon * factor,
        n_samples=n_futures,
        dt=fine_dt,
        rng=np.random.default_rng(future_seed),
    )
    futures = validate_array(
        futures, (batch * n_histories, n_futures, horizon * factor, channels)
    )
    # Fine forecasts start one fine step after the history endpoint; the first
    # requested coarse forecast is therefore at index factor - 1, not zero.
    return (
        futures[:, :, factor - 1 :: factor]
        .reshape(batch, n_histories * n_futures, horizon, channels)
        .copy()
    )
