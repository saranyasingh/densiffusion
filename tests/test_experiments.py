import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.experiments import run_experiment  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.objectives import reconstruction_mse  # noqa: E402
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
