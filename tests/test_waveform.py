import numpy as np
import pytest

from densiffusion.data import GenerationConfig, generate
from densiffusion.generators.waveform import (
    WaveformParameters,
    draw_waveform_parameters,
    evaluate_waveforms,
    generate_waveforms,
)


def test_default_grid_shape_and_reproducibility():
    values = generate_waveforms(7, rng=91)
    assert isinstance(values, np.ndarray)
    assert values.shape == (7, 33, 1)
    assert values.dtype == np.float64
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values, generate_waveforms(7, rng=91))
    np.testing.assert_array_equal(
        values, generate_waveforms(7, rng=np.random.default_rng(91))
    )
    assert not np.array_equal(values, generate_waveforms(7, rng=92))


def test_generator_plugin_and_single_point_trajectory():
    config = GenerationConfig(
        generator="densiffusion.generators.waveform:generate",
        n_series=3,
        n_steps=1,
        dt=0.25,
        seed=34,
        params={"noise_std": 0.01},
    )
    np.testing.assert_array_equal(
        generate(config),
        generate_waveforms(3, 1, 0.25, rng=34, noise_std=0.01),
    )


def test_supplied_generator_advances_without_changing_global_random_state():
    rng = np.random.default_rng(5)
    first = generate_waveforms(4, rng=rng)
    second = generate_waveforms(4, rng=rng)
    assert not np.array_equal(first, second)
    np.random.seed(13)
    expected = np.random.random(5)
    np.random.seed(13)
    generate_waveforms(4, rng=5)
    np.testing.assert_array_equal(np.random.random(5), expected)


def test_latents_are_fixed_over_time_and_harmonic_phase_is_inside_sine():
    # Fixed nontrivial phases expose both harmonic-frequency and phase mistakes.
    parameters = WaveformParameters(
        amplitude=np.array([0.8, 1.2]),
        frequency=np.array([0.65, 0.85]),
        phase=np.array([0.2, 1.1]),
        harmonic_phase=np.array([0.7, 2.3]),
    )
    times = np.arange(33) * 0.1
    values = evaluate_waveforms(parameters, times)
    for row, (amplitude, frequency, phase, harmonic_phase) in enumerate(
        [(0.8, 0.65, 0.2, 0.7), (1.2, 0.85, 1.1, 2.3)]
    ):
        expected = [
            amplitude
            * (
                np.sin(2 * np.pi * frequency * time + phase)
                + 0.3
                * np.sin(4 * np.pi * frequency * time + 2 * phase + harmonic_phase)
            )
            for time in times
        ]
        np.testing.assert_allclose(values[row, :, 0], expected, atol=1e-14)
    fine = evaluate_waveforms(parameters, np.arange(65) * 0.05)
    np.testing.assert_array_equal(fine[:, ::2], values)


def test_noiseless_generator_uses_one_parameter_draw_per_trajectory():
    parameters = draw_waveform_parameters(12, rng=123)
    expected = evaluate_waveforms(parameters, np.arange(17) * 0.125)
    np.testing.assert_array_equal(
        generate_waveforms(12, 17, 0.125, rng=123, noise_std=0), expected
    )


def test_parameter_distributions_and_independence():
    parameters = draw_waveform_parameters(30_000, rng=991)
    draws = np.column_stack(
        [
            parameters.amplitude,
            parameters.frequency,
            parameters.phase,
            parameters.harmonic_phase,
        ]
    )
    low = np.array([0.7, 0.6, 0, 0])
    width = np.array([0.6, 0.3, 2 * np.pi, 2 * np.pi])
    uniform = (draws - low) / width
    assert np.all((uniform >= 0) & (uniform < 1))
    np.testing.assert_allclose(uniform.mean(axis=0), 0.5, atol=0.006)
    np.testing.assert_allclose(uniform.var(axis=0), 1 / 12, atol=0.002)
    np.testing.assert_allclose(np.corrcoef(uniform.T), np.eye(4), atol=0.025)
    # Also reject shared or strongly correlated parameters across trajectories.
    adjacent = np.corrcoef(uniform[:-1].T, uniform[1:].T)[:4, 4:]
    np.testing.assert_allclose(adjacent, 0, atol=0.025)


def test_observation_noise_is_independent_of_times_trajectories_and_latents():
    count = 30_000
    parameters = draw_waveform_parameters(count, rng=72)
    signal = evaluate_waveforms(parameters, np.arange(6) * 0.1)
    noise = generate_waveforms(count, 6, rng=72)[:, :, 0] - signal[:, :, 0]
    np.testing.assert_allclose(noise.mean(axis=0), 0, atol=0.0004)
    np.testing.assert_allclose(noise.std(axis=0), 0.02, atol=0.0003)
    np.testing.assert_allclose(np.corrcoef(noise.T), np.eye(6), atol=0.025)
    np.testing.assert_allclose(
        np.corrcoef(noise[:-1, 0], noise[1:, 0])[0, 1], 0, atol=0.025
    )
    np.testing.assert_allclose(
        np.corrcoef(noise[:, 0], parameters.amplitude)[0, 1], 0, atol=0.025
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"n_series": 0},
        {"n_series": True},
        {"n_series": 1.5},
        {"n_steps": 0},
        {"n_steps": 2.5},
        {"dt": 0},
        {"dt": -0.1},
        {"dt": np.inf},
        {"dt": np.nan},
        {"noise_std": -0.02},
        {"noise_std": np.inf},
        {"noise_std": np.nan},
    ],
)
def test_invalid_generator_settings_are_rejected(arguments):
    with pytest.raises(ValueError):
        generate_waveforms(**({"n_series": 2, "rng": 0} | arguments))


def test_diagnostic_parameter_and_time_shapes_are_validated():
    with pytest.raises(ValueError):
        WaveformParameters(np.ones(2), np.ones(3), np.zeros(2), np.zeros(2))
    parameters = draw_waveform_parameters(2, rng=1)
    for times in (np.empty(0), np.array([0, np.nan]), np.ones((2, 2))):
        with pytest.raises(ValueError):
            evaluate_waveforms(parameters, times)
