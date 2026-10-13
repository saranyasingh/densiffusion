import numpy as np

from densiffusion.contracts import Array, Mask, validate_array, validate_mask


def evaluate(
    samples: Array, target: Array, *, mask: Mask | None = None
) -> dict[str, float]:
    """Shared imputation scores; mask=True selects entries to evaluate."""
    return {"crps": crps(samples, target, mask=mask)}


def crps(samples: Array, target: Array, *, mask: Mask | None = None) -> float:
    """Mean empirical-ensemble CRPS for (batch, samples, time, channels) draws."""
    target = validate_array(target, (None, None, None))
    samples = validate_array(samples, (target.shape[0], None, *target.shape[1:]))
    if mask is not None:
        mask = validate_mask(mask, target.shape)
        if not mask.any():
            raise ValueError("mask must select entries")
    errors = np.sort(samples - target[:, None], axis=1)
    count = samples.shape[1]
    weights = (2 * np.arange(1, count + 1) - count - 1) / count**2
    scores = np.abs(errors).mean(axis=1) - np.sum(
        errors * weights[None, :, None, None], axis=1
    )
    return float(scores[mask].mean() if mask is not None else scores.mean())
