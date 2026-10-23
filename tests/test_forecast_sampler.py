import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.csdi import CSDIDenoiser  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.samplers import DDPMForecastSampler  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402


class RecordingDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))
        self.calls = []

    def forward(self, noisy, steps, *, observed, mask, dt):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append((noisy.clone(), observed.clone(), mask.clone(), dt))
        # A clean-prediction oracle makes future extraction exact.
        return (
            torch.arange(noisy.shape[1])
            .expand_as(noisy.transpose(1, 2))
            .transpose(1, 2)
            + self.bias * 0
        )


def test_forecast_suffix_is_hidden_and_history_remains_exact():
    model = RecordingDenoiser()
    sampler = DDPMForecastSampler(
        model,
        TrainingConfig(batch_size=3, diffusion_steps=2),
        prediction_type="clean",
    )
    history = np.arange(10, dtype=float).reshape(2, 5, 1) / 7
    original = history.copy()
    rng = np.random.default_rng(5)
    torch_rng = torch.random.get_rng_state().clone()
    result = sampler.sample(
        history[:, ::2], history, horizon=4, n_samples=2, dt=0.125, rng=rng
    )
    assert result.shape == (2, 2, 4, 1)
    np.testing.assert_array_equal(
        result, np.broadcast_to(np.arange(5, 9)[None, None, :, None], result.shape)
    )
    for noisy, observed, mask, dt in model.calls:
        assert dt == 0.125
        assert not mask[:, :5].any() and mask[:, 5:].all()
        assert (observed[:, 5:] == 0).all()
        torch.testing.assert_close(noisy[:, :5], observed[:, :5])
    assert model.training
    torch.testing.assert_close(torch_rng, torch.random.get_rng_state())
    np.testing.assert_array_equal(history, original)


@pytest.mark.parametrize(
    "observed,dense,message",
    [
        (np.ones((2, 3, 1)), np.ones((2, 4, 1)), "refine"),
        (np.ones((2, 3, 1)), np.zeros((2, 5, 1)), "preserve"),
        (np.ones((2, 1, 1)), np.ones((2, 2, 1)), "refine"),
        (np.ones((2, 3, 1)), np.ones((1, 5, 1)), "shape"),
    ],
)
def test_forecast_rejects_incompatible_histories(observed, dense, message):
    model = RecordingDenoiser()
    with pytest.raises(ValueError, match=message):
        DDPMForecastSampler(model, TrainingConfig()).sample(
            observed,
            dense,
            horizon=2,
            n_samples=1,
            dt=1.0,
            rng=np.random.default_rng(0),
        )
    assert not model.calls


@pytest.mark.parametrize("channels", [1, 2])
def test_csdi_forecast_draws_are_reproducible_and_condition_on_past(channels):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    network = CSDIDenoiser(
        channels,
        hidden_channels=8,
        residual_layers=1,
        attention_heads=2,
        diffusion_embedding_dim=8,
        time_embedding_dim=8,
        feature_embedding_dim=4,
        feedforward_dim=8,
        dropout=0,
    )
    # Avoid the default zero-output initialization hiding conditioning effects.
    torch.nn.init.normal_(network.output_projection2.weight)
    sampler = DDPMForecastSampler(
        ImputationModel(network),
        TrainingConfig(batch_size=4, diffusion_steps=3, beta_end=0.2),
    )
    history = np.ones((2, 5, channels))

    def sample(values, seed=4):
        return sampler.sample(
            values[:, ::2],
            values,
            horizon=4,
            n_samples=3,
            dt=0.1,
            rng=np.random.default_rng(seed),
        )

    first = sample(history)
    assert first.shape == (2, 3, 4, channels) and np.isfinite(first).all()
    np.testing.assert_array_equal(first, sample(history))
    assert not np.array_equal(first, sample(history, 5))
    assert not np.array_equal(first, sample(history + 3))


def test_single_point_history_forecasts_without_extra_history_points():
    model = RecordingDenoiser()
    sampler = DDPMForecastSampler(model, TrainingConfig(diffusion_steps=1))
    history = np.ones((2, 1, 1))
    result = sampler.sample(
        history, history, horizon=1, n_samples=2, dt=1, rng=np.random.default_rng(0)
    )
    assert result.shape == (2, 2, 1, 1)
