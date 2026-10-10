import numpy as np

from densiffusion.contracts import (
    Array,
    Mask,
    positive_int,
    validate_array,
    validate_mask,
    validate_times,
)


def prepare_targets(values: Array, mask: Mask | None = None) -> tuple[Array, Mask]:
    """Keep complete targets and select entries for training or evaluation."""
    values = validate_array(values, (None, None, None))
    if mask is None:
        if values.shape[1] < 3:
            raise ValueError("Default imputation requires at least three time points")
        values = values[:, : 2 * ((values.shape[1] - 1) // 2) + 1]
        mask = np.zeros(values.shape, dtype=bool)
        mask[:, 1::2] = True
    else:
        mask = validate_mask(mask, values.shape)
    counts = mask.sum(axis=(1, 2))
    if (counts == 0).any() or (counts == values.shape[1] * values.shape[2]).any():
        raise ValueError("Each target must have both hidden and observed entries")
    return values, mask


def prepare_observations(
    observed: Array,
    *,
    factor: int,
    dt: float,
    mask: Mask | None = None,
    times: Array | None = None,
) -> tuple[Array, Mask, Array]:
    """Return zero-filled conditioning, hidden entries, and target grid times."""
    positive_int("factor", factor)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    observed = np.asarray(observed)
    if observed.ndim != 3 or 0 in observed.shape:
        raise ValueError("observed must have shape (batch, time, channels)")
    batch, length, channels = observed.shape
    if mask is None:
        observed = validate_array(observed, (batch, length, channels))
        shape = (batch, (length - 1) * factor + 1, channels)
        conditioning = np.zeros(shape)
        conditioning[:, ::factor] = observed
        hidden = np.ones(shape, dtype=bool)
        hidden[:, ::factor] = False
    else:
        if factor != 1:
            raise ValueError("factor must be 1 when an explicit mask is supplied")
        hidden = validate_mask(mask, observed.shape)
        conditioning = validate_array(np.where(hidden, 0, observed), observed.shape)
        if hidden.all(axis=(1, 2)).any():
            raise ValueError("Each series must have at least one observed entry")
    input_times = validate_times(
        np.arange(length) * dt if times is None else times, (batch, length)
    )
    positions = np.arange(conditioning.shape[1]) / factor
    left = positions.astype(int)
    right = np.minimum(left + 1, length - 1)
    target_times = input_times[:, left] + (positions - left) * (
        input_times[:, right] - input_times[:, left]
    )
    target_times = validate_times(target_times, conditioning.shape[:2])
    return conditioning, hidden, target_times
