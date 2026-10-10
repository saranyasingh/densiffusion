import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.pipeline import impute  # noqa: E402
from densiffusion.samplers import DDPMSampler  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402


class Denoiser(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1, dtype=dtype))
        self.dropout = torch.nn.Dropout(0.5)
        self.calls = []

    def forward(self, noisy, steps, *, observed, mask, dt):
        assert not self.training and not self.dropout.training
        assert not torch.is_grad_enabled()
        self.calls.append(
            (noisy.clone(), steps.clone(), observed.clone(), mask.clone(), dt)
        )
        return self.dropout(self.scale * noisy)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("factor", [2, 3])
def test_sampling_preserves_conditions_rng_and_model_state(dtype, factor):
    model = Denoiser(dtype)
    model.dropout.eval()
    modes = [module.training for module in model.modules()]
    config = TrainingConfig(
        batch_size=2, diffusion_steps=3, beta_start=0.05, beta_end=0.2
    )
    sampler = DDPMSampler(model, config)
    observed = np.arange(12, dtype=float).reshape(2, 3, 2) / 7
    original = observed.copy()
    torch_state = torch.random.get_rng_state().clone()

    samples = impute(observed, sampler, factor=factor, n_samples=3, dt=0.5, seed=19)
    calls = model.calls.copy()
    assert samples.shape == (2, 3, 2 * factor + 1, 2)
    assert samples.dtype == np.float64
    np.testing.assert_array_equal(
        samples[:, :, ::factor], np.repeat(observed[:, None], 3, axis=1)
    )
    np.testing.assert_array_equal(observed, original)
    assert not np.array_equal(samples[:, 0, 1], samples[:, 1, 1])
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)

    assert len(calls) == 9
    indices = np.arange(6) // 3
    for index, (noisy, steps, conditioning, mask, dt) in enumerate(calls):
        start = index // 3 * config.batch_size
        expected = torch.as_tensor(observed[indices[start : start + 2]], dtype=dtype)
        torch.testing.assert_close(noisy[:, ::factor], expected)
        torch.testing.assert_close(noisy[~mask], conditioning[~mask])
        torch.testing.assert_close(steps, torch.full((2,), 2 - index % 3))
        assert mask[:, 1].all() and not mask[:, ::factor].any()
        assert (conditioning[mask] == 0).all()
        assert dt == 0.5 / factor

    same = impute(observed, sampler, factor=factor, n_samples=3, dt=0.5, seed=19)
    different = impute(observed, sampler, factor=factor, n_samples=3, dt=0.5, seed=20)
    np.testing.assert_array_equal(samples, same)
    assert not np.array_equal(samples, different)


@pytest.mark.parametrize("prediction_type", ["noise", "clean"])
def test_ddpm_matches_gaussian_posterior_and_recovers_clean_target(prediction_type):
    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
            self.register_buffer(
                "alpha_bar", torch.tensor([0.8, 0.48], dtype=torch.float64)
            )
            self.inputs = []

        def forward(self, noisy, steps, *, observed, mask, dt):
            self.inputs.append(noisy.clone())
            clean = torch.full_like(noisy, 3.0)
            if prediction_type == "noise":
                alpha = self.alpha_bar[steps, None, None]
                prediction = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            else:
                prediction = clean
            return prediction.masked_fill(~mask, float("nan"))

    model = Oracle()
    config = TrainingConfig(
        batch_size=8192, diffusion_steps=2, beta_start=0.2, beta_end=0.4
    )
    sampler = DDPMSampler(model, config, prediction_type=prediction_type)
    samples = impute(np.array([[[0.1], [0.7]]]), sampler, n_samples=8192, seed=42)
    initial, intermediate = (values[:, 1, 0].numpy() for values in model.inputs)

    assert initial.mean() == pytest.approx(0, abs=0.04)
    assert initial.var() == pytest.approx(1, rel=0.05)
    posterior_mean = (np.sqrt(0.8) * 0.4 * 3 + np.sqrt(0.6) * 0.2 * initial) / 0.52
    residual = intermediate - posterior_mean
    assert residual.mean() == pytest.approx(0, abs=0.015)
    assert residual.var() == pytest.approx(0.4 * 0.2 / 0.52, rel=0.05)
    np.testing.assert_allclose(samples[:, :, 1], 3, rtol=0, atol=1e-12)


@pytest.mark.parametrize("length, factor", [(3, 1), (1, 2)])
def test_no_hidden_points_skip_network_and_randomness(length, factor):
    observed = np.arange(2 * length, dtype=float).reshape(2, length, 1)
    rng = np.random.default_rng(1)
    initial = rng.bit_generator.state
    sampler = DDPMSampler(torch.nn.Identity(), TrainingConfig())
    samples = sampler.sample(observed, factor=factor, n_samples=3, dt=1, rng=rng)
    np.testing.assert_array_equal(samples, np.repeat(observed[:, None], 3, axis=1))
    assert not np.shares_memory(samples, observed)
    assert rng.bit_generator.state == initial


@pytest.mark.parametrize(
    "failure, error",
    [("shape", ValueError), ("non_tensor", ValueError), ("nan", FloatingPointError)],
)
def test_invalid_predictions_restore_model_modes(failure, error):
    class InvalidDenoiser(Denoiser):
        def forward(self, *args, **kwargs):
            prediction = super().forward(*args, **kwargs)
            if failure == "shape":
                return prediction[:, :1]
            if failure == "non_tensor":
                return None
            return prediction + float("nan")

    model = InvalidDenoiser()
    model.dropout.eval()
    modes = [module.training for module in model.modules()]
    with pytest.raises(error):
        impute(
            np.ones((1, 2, 1)), DDPMSampler(model, TrainingConfig(diffusion_steps=2))
        )
    assert [module.training for module in model.modules()] == modes


def test_observation_overflow_fails_before_network_call():
    model = Denoiser()
    with pytest.raises(ValueError, match="overflow"):
        impute(np.full((1, 2, 1), 1e300), DDPMSampler(model, TrainingConfig()))
    assert model.training and not model.calls


def test_unrepresentable_schedule_is_rejected():
    config = TrainingConfig(diffusion_steps=2, beta_start=1e-12, beta_end=1e-12)
    with pytest.raises(ValueError, match="schedule"):
        impute(np.ones((1, 2, 1)), DDPMSampler(Denoiser(), config))


def test_ddpm_explicit_masks_hide_inputs_and_preserve_each_observed_channel():
    model = Denoiser()
    config = TrainingConfig(
        batch_size=2, diffusion_steps=2, beta_start=0.1, beta_end=0.2
    )
    sampler = DDPMSampler(model, config)
    values = np.arange(16, dtype=float).reshape(2, 4, 2) / 7
    mask = np.array(
        [
            [[True, False], [False, True], [True, False], [False, True]],
            [[False, True], [True, True], [False, False], [True, False]],
        ]
    )
    kwargs = dict(
        factor=1,
        mask=mask,
        times=10 + np.arange(4) * 0.25,
        n_samples=3,
        dt=0.25,
        seed=42,
    )
    samples = impute(values, sampler, **kwargs)
    assert samples.shape == (2, 3, 4, 2)
    for draw in np.moveaxis(samples, 1, 0):
        np.testing.assert_array_equal(draw[~mask], values[~mask])
    np.testing.assert_array_equal(
        samples, impute(np.where(mask, np.nan, values), sampler, **kwargs)
    )
    for noisy, _, observed, batch_mask, dt in model.calls:
        assert (observed[batch_mask] == 0).all() and dt == 0.25
        torch.testing.assert_close(noisy[~batch_mask], observed[~batch_mask])


def test_ddpm_rejects_irregular_times_before_calling_model():
    model = Denoiser()
    with pytest.raises(ValueError, match="regular times"):
        impute(
            np.ones((1, 3, 1)),
            DDPMSampler(model, TrainingConfig()),
            times=np.array([10, 11, 13]),
        )
    assert not model.calls


@pytest.mark.parametrize("origin", [-1_700_000_000.0, 1_700_000_000.0])
@pytest.mark.parametrize("factor", [1, 2, 3])
def test_ddpm_regular_grid_accepts_large_time_origins(origin, factor):
    sampler = DDPMSampler(Denoiser(), TrainingConfig(diffusion_steps=2))
    observed = np.ones((1, 5, 1))
    mask = np.array([False, True, False, True, False]) if factor == 1 else None
    times = np.arange(5) * 0.1
    options = dict(factor=factor, dt=0.1, mask=mask, n_samples=2, seed=42)
    expected = impute(observed, sampler, times=times, **options)
    samples = impute(observed, sampler, times=times + origin, **options)
    np.testing.assert_array_equal(samples, expected)
    irregular = times + origin
    irregular[2] += 0.01
    with pytest.raises(ValueError, match="regular times"):
        impute(observed, sampler, times=irregular, **options)
