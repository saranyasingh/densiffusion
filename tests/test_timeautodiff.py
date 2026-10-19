import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.timeautodiff import (  # noqa: E402
    PositionalEncoding,
    TimeAutoDiffDenoiser,
    TimeAutoDiffEncoder,
    sine_cosine,
)
from densiffusion.experiments import run_experiment  # noqa: E402
from densiffusion.gp.sampling import generate  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.pipeline import impute  # noqa: E402
from densiffusion.samplers import DDPMSampler  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402


def make_model(channels=1, hidden=8, steps=5):
    torch.manual_seed(0)
    return ImputationModel(
        TimeAutoDiffDenoiser(
            channels=channels, hidden=hidden, layers=1, diffusion_steps=steps
        ),
        encoder=TimeAutoDiffEncoder(
            channels=channels, hidden=hidden, embedding=16, layers=1, terms=3
        ),
    )


def inputs(batch=3, length=9, channels=2):
    generator = torch.Generator().manual_seed(1)
    target = torch.randn(batch, length, channels, generator=generator)
    mask = torch.zeros(batch, length, channels, dtype=torch.bool)
    mask[:, 1::2] = True
    noisy = torch.where(mask, torch.randn(target.shape, generator=generator), target)
    steps = torch.tensor([0, 2, 4])[:batch]
    return noisy, steps, target.masked_fill(mask, 0), mask


def test_sine_cosine_matches_reference_layout():
    values = torch.tensor([[0.25, -0.5]])
    features = sine_cosine(values, 2)
    angles = math.pi * torch.tensor([0.25, 0.5, -0.5, -1.0])
    expected = torch.cat((angles.sin(), angles.cos()))[None]
    assert features.shape == (1, 8)
    torch.testing.assert_close(features, expected)


def test_positional_encoding_shape_and_validation():
    encoding = PositionalEncoding(7, max_value=100)
    assert encoding(torch.rand(2, 5, 1)).shape == (2, 5, 7)
    with pytest.raises(ValueError, match="max_value"):
        PositionalEncoding(8, max_value=0)


def test_model_predicts_values_shape_with_and_without_encoder():
    noisy, steps, observed, mask = inputs()
    model = make_model(channels=2)
    prediction = model(noisy, steps, observed=observed, mask=mask, dt=0.5)
    assert prediction.shape == noisy.shape
    assert torch.isfinite(prediction).all()
    denoiser = TimeAutoDiffDenoiser(channels=2, hidden=8, layers=1, diffusion_steps=5)
    alone = ImputationModel(denoiser)(noisy, steps, observed=observed, mask=mask, dt=1)
    assert alone.shape == noisy.shape


def test_encoder_ignores_values_at_hidden_entries():
    _, _, observed, mask = inputs()
    encoder = make_model(channels=2).encoder
    corrupted = torch.where(mask, torch.full_like(observed, 123.0), observed)
    torch.testing.assert_close(
        encoder(observed, mask=mask, dt=1.0), encoder(corrupted, mask=mask, dt=1.0)
    )


def test_context_is_bidirectional():
    _, _, observed, mask = inputs()
    encoder = make_model(channels=2).encoder
    changed = observed.clone()
    changed[:, -1] += 1.0
    first = encoder(observed, mask=mask, dt=1.0)[:, 0]
    assert not torch.allclose(first, encoder(changed, mask=mask, dt=1.0)[:, 0])


def test_rejects_wrong_channel_count():
    noisy, steps, observed, mask = inputs(channels=2)
    with pytest.raises(ValueError, match="channels|Expected"):
        make_model(channels=1)(noisy, steps, observed=observed, mask=mask, dt=1.0)


def test_trains_samples_and_scores_on_gp_data():
    values = generate(
        n_series=24,
        n_steps=9,
        dt=0.5,
        rng=np.random.default_rng(0),
        kernel="exponential",
        variance=1.0,
        lengthscale=2.0,
    )
    config = TrainingConfig(epochs=2, batch_size=8, diffusion_steps=5)
    model = make_model(steps=config.diffusion_steps)
    scores = run_experiment(
        model,
        values[:16],
        values[16:20],
        values[20:],
        sampler_factory=DDPMSampler,
        config=config,
        n_samples=4,
        dt=0.5,
        seed=0,
    )
    assert np.isfinite(scores["crps"]) and scores["crps"] > 0

    samples = impute(
        values[20:, ::2],
        DDPMSampler(model, config),
        factor=2,
        n_samples=3,
        dt=1.0,
        seed=1,
    )
    assert samples.shape == (4, 3, 9, 1)
    np.testing.assert_array_equal(
        samples[:, :, ::2], np.repeat(values[20:, None, ::2], 3, axis=1)
    )
