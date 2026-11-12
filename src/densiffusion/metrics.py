import numpy as np

from densiffusion.contracts import Array, Mask, validate_array, validate_mask


def evaluate(
    samples: Array, target: Array, *, mask: Mask | None = None
) -> dict[str, float]:
    """Shared imputation scores; mask=True selects entries to evaluate."""
    return {"crps": crps(samples, target, mask=mask)}


def crps(
    samples: Array, target: Array, *, mask: Mask | None = None, fair: bool = False
) -> float:
    """Mean empirical-ensemble CRPS for (batch, samples, time, channels) draws.

    The default divides the pairwise spread term by n**2, the usual ensemble
    estimator, which overstates the CRPS of a finite ensemble. ``fair=True``
    divides by n * (n - 1) instead, the unbiased estimate of the CRPS of the
    distribution the draws come from; it needs at least two draws.
    """
    target = validate_array(target, (None, None, None))
    samples = validate_array(samples, (target.shape[0], None, *target.shape[1:]))
    if mask is not None:
        mask = validate_mask(mask, target.shape)
        if not mask.any():
            raise ValueError("mask must select entries")
    errors = np.sort(samples - target[:, None], axis=1)
    count = samples.shape[1]
    if fair and count < 2:
        raise ValueError("fair CRPS requires at least two samples")
    pairs = count * (count - 1) if fair else count**2
    weights = (2 * np.arange(1, count + 1) - count - 1) / pairs
    scores = np.abs(errors).mean(axis=1) - np.sum(
        errors * weights[None, :, None, None], axis=1
    )
    return float(scores[mask].mean() if mask is not None else scores.mean())
