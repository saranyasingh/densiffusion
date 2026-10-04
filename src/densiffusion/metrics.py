from densiffusion.contracts import Array


def evaluate(
    samples: Array, target: Array, *, interval: float = 0.9
) -> dict[str, float]:
    """Evaluation hook for MSE, CRPS, and interval statistics."""
    raise NotImplementedError("Forecast evaluation is not implemented yet.")
