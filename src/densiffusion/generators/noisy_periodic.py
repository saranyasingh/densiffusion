"""Independent periodic observations with Gaussian noise and sparse outliers.

Only noisy observations are public: no signal, latent parameter, or outlier
label is returned. All parameters stay fixed within each trajectory.
"""

import numpy as np

from densiffusion.contracts import Array, positive_int


def generate_noisy_periodic(
    n_trajectories: int,
    n_steps: int = 40,
    *,
    rng: np.random.Generator | int,
) -> Array:
    """Return (trajectories, integer times, 1) noisy observations.

    Draw order is amplitude, period, phase, ordinary Gaussian noise, Bernoulli
    outlier indicators, and outlier Gaussian noise. Each draw is independent.
    The latter Gaussian is drawn at every point, including inactive outliers.
    """
    positive_int("n_trajectories", n_trajectories)
    positive_int("n_steps", n_steps)
    generator = np.random.default_rng(rng)
    amplitude = generator.uniform(0.7, 1.3, (n_trajectories, 1))
    period = generator.uniform(24.0, 40.0, (n_trajectories, 1))
    phase = generator.uniform(0.0, 2 * np.pi, (n_trajectories, 1))
    angle = 2 * np.pi * np.arange(n_steps)[None, :] / period + phase
    signal = amplitude * (np.sin(angle) + 0.2 * np.sin(2 * angle + 0.4))
    ordinary = generator.normal(0.0, 0.15, signal.shape)
    active = generator.binomial(1, 0.1, signal.shape)
    outliers = generator.normal(0.0, 2.0, signal.shape)
    return (signal + ordinary + active * outliers)[:, :, None]


def generate(
    n_series: int,
    n_steps: int = 40,
    dt: float = 1.0,
    *,
    rng: np.random.Generator | int,
) -> Array:
    """GenerationConfig-compatible entry point, restricted to integer times."""
    if dt != 1.0:
        raise ValueError("Noisy periodic observations require integer times (dt=1)")
    return generate_noisy_periodic(n_series, n_steps, rng=rng)
