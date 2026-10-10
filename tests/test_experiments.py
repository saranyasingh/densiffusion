from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.experiments import run_experiment  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.objectives import noise_mse, reconstruction_mse  # noqa: E402
from densiffusion.samplers import DDPMSampler  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, observed, *, mask, dt):
        assert (observed[mask] == 0).all()
        return self.scale * observed.mean(dim=1, keepdim=True)


class DirectDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, noisy, steps, *, observed, mask, dt):
        return self.bias.expand_as(noisy)


class ConditionalDenoiser(DirectDenoiser):
    def forward(self, noisy, steps, *, observed, mask, dt, context):
        return self.bias.expand_as(noisy) + context


class ReconstructionSampler:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.calls = []
        self.samples = []

    def sample(self, observed, *, factor, n_samples, dt, rng):
        assert not self.model.training and not torch.is_grad_enabled()
        self.calls.append((observed.copy(), dt))
        batch, steps, channels = observed.shape
        dense = np.zeros((batch, (steps - 1) * factor + 1, channels))
        dense[:, ::factor] = observed
        mask = torch.ones(dense.shape, dtype=torch.bool)
        mask[:, ::factor] = False
        conditioning = torch.as_tensor(dense, dtype=torch.float32)
        prediction = self.model(
            conditioning,
            torch.zeros(batch, dtype=torch.long),
            observed=conditioning,
            mask=mask,
            dt=dt / factor,
        ).numpy()
        samples = np.repeat(prediction[:, None], n_samples, axis=1).astype(float)
        samples[:, :, ::factor] = observed[:, None]
        self.samples.append(samples)
        return samples


@pytest.mark.parametrize("encoded", [False, True])
def test_workflow_trains_both_paths_and_scores_hidden_test_points(encoded, tmp_path):
    model = ImputationModel(
        ConditionalDenoiser() if encoded else DirectDenoiser(),
        encoder=Encoder() if encoded else None,
    )
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    config = TrainingConfig(
        epochs=3, batch_size=2, diffusion_steps=4, learning_rate=0.1
    )
    samplers = []

    def sampler_factory(trained, settings):
        assert trained is model and settings is config
        sampler = ReconstructionSampler(trained, settings)
        samplers.append(sampler)
        return sampler

    train = np.ones((4, 6, 2))
    validation = np.ones((2, 6, 2))
    test = np.broadcast_to(np.arange(1, 6)[:, None, None], (5, 6, 2)).copy()
    test[:, 1::2] += 10
    original = test.copy()
    path = tmp_path / "model.pt"
    metrics = run_experiment(
        model,
        train,
        validation,
        test,
        sampler_factory=sampler_factory,
        config=config,
        objective=reconstruction_mse,
        n_samples=3,
        dt=0.25,
        seed=42,
        checkpoint=path,
    )
    sampler = samplers[0]
    assert [len(call[0]) for call in sampler.calls] == [2, 2, 1]
    assert all(call[1] == 0.5 for call in sampler.calls)
    np.testing.assert_array_equal(
        np.concatenate([call[0] for call in sampler.calls]), test[:, :5:2]
    )
    predictions = np.concatenate(sampler.samples)[:, 0, 1::2]
    expected = np.abs(predictions - test[:, 1:5:2]).mean()
    assert metrics == pytest.approx({"crps": expected})
    np.testing.assert_array_equal(test, original)
    np.testing.assert_array_equal(train, 1)
    for name, value in model.state_dict().items():
        assert not torch.equal(initial[name], value)
    restored = ImputationModel(
        ConditionalDenoiser() if encoded else DirectDenoiser(),
        encoder=Encoder() if encoded else None,
    )
    saved = torch.load(path, weights_only=True)
    restored.load_state_dict(saved["model_state_dict"])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    assert saved["objective"] == "reconstruction_mse"


def test_invalid_test_split_fails_before_training():
    model = ImputationModel(DirectDenoiser())
    initial = model.denoiser.bias.clone()
    data = np.zeros((2, 5, 1))
    with pytest.raises(ValueError, match="shape"):
        run_experiment(
            model,
            data,
            data,
            np.zeros((2, 4, 1)),
            sampler_factory=ReconstructionSampler,
        )
    torch.testing.assert_close(model.denoiser.bias, initial)


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize(
    "objective, prediction_type", [(noise_mse, "noise"), (reconstruction_mse, "clean")]
)
def test_ddpm_experiment_trains_and_evaluates_both_paths(
    encoded, objective, prediction_type
):
    model = ImputationModel(
        ConditionalDenoiser() if encoded else DirectDenoiser(),
        encoder=Encoder() if encoded else None,
    )
    values = np.random.default_rng(42).normal(size=(8, 5, 2))
    scores = run_experiment(
        model,
        values[:3],
        values[3:5],
        values[5:],
        sampler_factory=partial(DDPMSampler, prediction_type=prediction_type),
        objective=objective,
        config=TrainingConfig(
            epochs=2, batch_size=2, diffusion_steps=3, beta_start=0.05, beta_end=0.2
        ),
        n_samples=3,
        dt=0.25,
        seed=42,
    )
    assert set(scores) == {"crps"}
    assert np.isfinite(scores["crps"]) and scores["crps"] >= 0
    assert not model.training


@pytest.mark.parametrize("encoded", [False, True])
def test_masked_experiment_scores_hidden_entries_across_partial_batches(encoded):
    model = ImputationModel(
        ConditionalDenoiser() if encoded else DirectDenoiser(),
        encoder=Encoder() if encoded else None,
    )
    values = np.arange(24, dtype=float).reshape(3, 4, 2)
    mask = np.zeros(values.shape, dtype=bool)
    mask[0, -1, 0] = True
    mask[1, 1:3] = True
    mask[2, [0, 2, 3]] = True
    times = 10 + np.arange(3)[:, None] + np.arange(4) * 0.25
    calls = []

    class Sampler:
        def sample(self, observed, *, factor, n_samples, dt, rng, mask, times):
            assert not model.training and not torch.is_grad_enabled()
            assert factor == 1 and dt == 0.25
            assert (observed[mask] == 0).all()
            calls.append(times.copy())
            return np.repeat(observed[:, None], n_samples, axis=1)

    scores = run_experiment(
        model,
        values,
        values,
        values,
        train_mask=mask,
        validation_mask=mask,
        test_mask=mask,
        test_times=times,
        sampler_factory=lambda *args: Sampler(),
        config=TrainingConfig(epochs=1, batch_size=2, diffusion_steps=2),
        objective=reconstruction_mse,
        n_samples=3,
        dt=0.25,
    )
    assert scores == pytest.approx({"crps": np.abs(values[mask]).mean()})
    np.testing.assert_array_equal(np.concatenate(calls), times)


def test_invalid_test_mask_fails_before_training():
    model = ImputationModel(DirectDenoiser())
    initial = model.denoiser.bias.clone()
    values = np.zeros((2, 4, 1))
    with pytest.raises(ValueError, match="both hidden and observed"):
        run_experiment(
            model,
            values,
            values,
            values,
            test_mask=np.ones(4, dtype=bool),
            sampler_factory=DDPMSampler,
        )
    torch.testing.assert_close(model.denoiser.bias, initial)


@pytest.mark.parametrize("times", [[10, 11, 12, 14], [10, 12, 14, 16]])
def test_incompatible_test_times_fail_before_training(times):
    model = ImputationModel(DirectDenoiser())
    initial = model.denoiser.bias.clone()
    values = np.zeros((2, 4, 1))
    with pytest.raises(ValueError, match="regularly spaced by dt"):
        run_experiment(
            model,
            values,
            values,
            values,
            test_times=np.array(times),
            sampler_factory=DDPMSampler,
        )
    torch.testing.assert_close(model.denoiser.bias, initial)


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_experiment_accepts_regular_times_with_large_origins(explicit_mask):
    values = np.ones((2, 5, 1))
    mask = np.array([False, True, False, True, False]) if explicit_mask else None
    settings = TrainingConfig(epochs=1, diffusion_steps=2)
    scores = []
    for origin in (0.0, 1_700_000_000.0):
        scores.append(
            run_experiment(
                ImputationModel(DirectDenoiser()),
                values,
                values,
                values,
                sampler_factory=DDPMSampler,
                config=settings,
                test_mask=mask,
                test_times=origin + np.arange(5) * 0.1,
                dt=0.1,
                n_samples=2,
            )
        )
    assert scores[0] == scores[1]
