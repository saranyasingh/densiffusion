"""Independent noisy waveforms with trajectory-specific, time-fixed latents.

The public generator returns only observations. Parameter helpers exist for
generator checks and separately labelled diagnostics after model selection;
latent parameters and fine-grid signal evaluations are not model inputs.
"""

from dataclasses import dataclass

import numpy as np

from densiffusion.contracts import Array, positive_int, validate_array


@dataclass(frozen=True)
class WaveformParameters:
    """One amplitude, frequency, phase, and harmonic phase per trajectory."""

    amplitude: Array
    frequency: Array
    phase: Array
    harmonic_phase: Array

    def __post_init__(self) -> None:
        amplitude = validate_array(self.amplitude, (None,))
        for name in ("amplitude", "frequency", "phase", "harmonic_phase"):
            value = validate_array(getattr(self, name), amplitude.shape).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def draw_waveform_parameters(
    n_series: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> WaveformParameters:
    """Draw independent A~U(.7,1.3), f~U(.6,.9), phi,psi~U(0,2*pi)."""
    positive_int("n_series", n_series)
    generator = np.random.default_rng(rng)
    parameters = generator.uniform(
        low=[0.7, 0.6, 0.0, 0.0],
        high=[1.3, 0.9, 2 * np.pi, 2 * np.pi],
        size=(n_series, 4),
    )
    return WaveformParameters(*(parameters[:, column] for column in range(4)))


def evaluate_waveforms(parameters: WaveformParameters, times: Array) -> Array:
    """Return the noise-free (series, time, 1) signal at the supplied times.

    This evaluator reuses the same latents at every time, including when
    separately evaluating a finer grid. It does not simulate observation noise.
    """
    times = validate_array(times, (None,))
    theta = (
        2 * np.pi * parameters.frequency[:, None] * times[None, :]
        + parameters.phase[:, None]
    )
    values = parameters.amplitude[:, None] * (
        np.sin(theta) + 0.3 * np.sin(2 * theta + parameters.harmonic_phase[:, None])
    )
    return values[:, :, None]


def generate_waveforms(
    n_series: int,
    n_steps: int = 33,
    dt: float = 0.1,
    *,
    rng: np.random.Generator | int | None = None,
    noise_std: float = 0.02,
) -> Array:
    """Return independent noisy trajectories, with shape (n_series,n_steps,1).

    Times are ``arange(n_steps) * dt``. Each trajectory keeps its amplitude,
    frequency, and phases fixed; observation noise is independently drawn at
    every time for every trajectory. A seed or NumPy Generator is accepted.
    No latent parameters or simulated fine-grid truth are returned.
    """
    positive_int("n_series", n_series)
    positive_int("n_steps", n_steps)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    generator = np.random.default_rng(rng)
    parameters = draw_waveform_parameters(n_series, rng=generator)
    values = evaluate_waveforms(parameters, np.arange(n_steps) * dt)
    noise = generator.normal(0.0, noise_std, size=values.shape)
    return values + noise


# Entry point for the repository's GenerationConfig module:function interface.
generate = generate_waveforms
