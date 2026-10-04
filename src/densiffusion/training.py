from densiffusion.contracts import Array, Densifier, Forecaster


def train_densifier(
    model: Densifier,
    train: Array,
    validation: Array,
    *,
    dt: float = 1.0,
    seed: int = 0,
) -> Densifier:
    """Train the densifier by hiding alternate observations."""
    raise NotImplementedError("Densifier training is not implemented yet.")


def train_forecaster(
    model: Forecaster,
    densifier: Densifier,
    train: tuple[Array, Array],
    validation: tuple[Array, Array],
    *,
    factor: int = 2,
    dt: float = 1.0,
    seed: int = 0,
) -> Forecaster:
    """Train on (history, future) windows with the trained densifier fixed."""
    raise NotImplementedError("Forecaster training is not implemented yet.")
