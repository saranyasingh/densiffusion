import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.pristi import (  # noqa: E402
    PriSTIContext,
    PriSTIDenoiser,
    PriSTIEncoder,
    _StepEmbedding,
    interpolate,
)
from densiffusion.experiments import run_experiment  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.samplers import DDPMSampler  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402

SIZES = dict(width=8, heads=2, feedforward=8, time_dim=8, channel_dim=4)


def networks(channels=1):
    torch.manual_seed(0)
    encoder = PriSTIEncoder(channels=channels, **SIZES).eval()
    denoiser = PriSTIDenoiser(layers=2, step_dim=8, **SIZES).eval()
    torch.nn.init.normal_(denoiser.output.weight)
    return encoder, denoiser


def inputs(batch=2, length=9, channels=1):
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(batch, length, channels, generator=generator)
    mask = torch.zeros(batch, length, channels, dtype=torch.bool)
    mask[:, 1::2] = True
    noisy = torch.where(mask, torch.randn(values.shape, generator=generator), values)
    steps = torch.arange(batch) % 5
    return values.masked_fill(mask, 0), mask, noisy, steps


def predict(encoder, denoiser, observed, mask, noisy, steps):
    with torch.no_grad():
        context = encoder(observed, mask=mask, dt=0.5)
        return denoiser(
            noisy, steps, observed=observed, mask=mask, dt=0.5, context=context
        )


def test_interpolation_matches_numpy_with_held_edges():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20, 12))
    observed = rng.random((20, 12)) < 0.4
    observed[:, 5] = True
    expected = [
        np.interp(np.arange(12), np.flatnonzero(seen), row[seen])
        for row, seen in zip(values, observed, strict=True)
    ]
    hidden_noise = np.where(observed, values, 1e6)
    result = interpolate(torch.tensor(hidden_noise), torch.tensor(observed))
    np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=1e-12)


def test_interpolation_recovers_lines_and_zeroes_unobserved_rows():
    line = torch.arange(7.0).expand(2, 7).clone()
    observed = torch.tensor(
        [[True, False, False, True, False, True, True], [False] * 7]
    )
    result = interpolate(line, observed)
    torch.testing.assert_close(result[0], line[0])
    assert (result[1] == 0).all()
    with pytest.raises(ValueError, match="boolean"):
        interpolate(line, observed.float())


def test_encoder_builds_context_and_ignores_hidden_values():
    encoder, _ = networks(channels=3)
    observed, mask, _, _ = inputs(channels=3)
    context = encoder(observed, mask=mask, dt=0.5)
    assert isinstance(context, PriSTIContext)
    assert context.interpolated.shape == (6, 9)
    assert context.prior.shape == (6, 9, 8)
    assert context.side.shape == (6, 9, 12)
    rows = context.interpolated.reshape(2, 3, 9).transpose(1, 2)
    assert torch.equal(rows[~mask], observed[~mask])
    changed = encoder(observed + 5 * mask, mask=mask, dt=0.5)
    for original, perturbed in zip(context, changed, strict=True):
        torch.testing.assert_close(original, perturbed)
    with pytest.raises(ValueError, match="channels|observations"):
        encoder(observed[..., :2], mask=mask[..., :2], dt=0.5)


def test_step_embedding_stays_finite_in_float16():
    torch.manual_seed(0)
    embedding = _StepEmbedding(8)
    steps = torch.tensor([0, 7, 49, 999])
    reference = embedding(steps)
    result = embedding.half()(steps)
    assert result.dtype == torch.float16 and torch.isfinite(result).all()
    torch.testing.assert_close(result.float(), reference, rtol=0, atol=1e-2)


def test_untrained_denoiser_predicts_zero_noise():
    torch.manual_seed(0)
    encoder = PriSTIEncoder(**SIZES)
    denoiser = PriSTIDenoiser(layers=2, step_dim=8, **SIZES)
    observed, mask, noisy, steps = inputs()
    prediction = predict(encoder, denoiser, observed, mask, noisy, steps)
    assert prediction.shape == noisy.shape
    assert (prediction == 0).all()


def test_prediction_depends_on_hidden_noise_and_observations_only():
    encoder, denoiser = networks()
    observed, mask, noisy, steps = inputs()
    prediction = predict(encoder, denoiser, observed, mask, noisy, steps)

    changed_observed_noise = torch.where(mask, noisy, noisy + 3)
    torch.testing.assert_close(
        predict(encoder, denoiser, observed, mask, changed_observed_noise, steps),
        prediction,
    )
    other_hidden = noisy.clone()
    other_hidden[0, 1] += 3
    changed = predict(encoder, denoiser, observed, mask, other_hidden, steps)
    assert not torch.allclose(changed[0, 3], prediction[0, 3])
    other_observation = observed.clone()
    other_observation[0, 4] += 3
    changed = predict(encoder, denoiser, other_observation, mask, noisy, steps)
    assert not torch.allclose(changed[0, 3], prediction[0, 3])


def test_noise_attention_takes_queries_and_keys_from_the_prior():
    encoder, denoiser = networks()
    observed, mask, noisy, steps = inputs()
    calls = []
    for layer in denoiser.layers:
        layer.temporal.attention.register_forward_pre_hook(
            lambda module, args: calls.append(args)
        )
    context = encoder(observed, mask=mask, dt=0.5)
    denoiser(noisy, steps, observed=observed, mask=mask, dt=0.5, context=context)
    assert len(calls) == len(denoiser.layers)
    for query, key, values in calls:
        assert query is context.prior and key is context.prior
        assert values is not context.prior


def test_batch_rows_and_channels_are_processed_independently():
    encoder, denoiser = networks(channels=2)
    observed, mask, noisy, steps = inputs(channels=2)
    prediction = predict(encoder, denoiser, observed, mask, noisy, steps)
    assert prediction.shape == (2, 9, 2)
    for row in range(2):
        window = slice(row, row + 1)
        alone = predict(
            encoder,
            denoiser,
            observed[window],
            mask[window],
            noisy[window],
            steps[window],
        )
        torch.testing.assert_close(alone, prediction[window])
    changed = noisy.clone()
    changed[0, 1, 0] += 3
    observed_changed = observed.clone()
    observed_changed[0, 2, 0] += 3
    result = predict(encoder, denoiser, observed_changed, mask, changed, steps)
    assert not torch.allclose(result[0, :, 0], prediction[0, :, 0])
    torch.testing.assert_close(result[0, :, 1], prediction[0, :, 1])
    torch.testing.assert_close(result[1], prediction[1])


def test_denoiser_rejects_mismatched_context():
    encoder, denoiser = networks()
    observed, mask, noisy, steps = inputs()
    context = encoder(observed, mask=mask, dt=0.5)
    with pytest.raises(ValueError, match="context"):
        denoiser(
            noisy[:1],
            steps[:1],
            observed=observed[:1],
            mask=mask[:1],
            dt=0.5,
            context=context,
        )


@pytest.mark.parametrize("channels", [1, 2])
def test_pristi_trains_and_samples_in_the_shared_workflow(channels):
    rng = np.random.default_rng(2)
    paths = rng.normal(scale=math.sqrt(0.1), size=(12, 9, channels)).cumsum(axis=1)
    torch.manual_seed(0)
    model = ImputationModel(
        PriSTIDenoiser(layers=2, step_dim=8, **SIZES),
        encoder=PriSTIEncoder(channels=channels, **SIZES),
    )
    config = TrainingConfig(epochs=2, batch_size=4, diffusion_steps=5, beta_end=0.2)
    scores = run_experiment(
        model,
        paths[:8],
        paths[8:10],
        paths[10:],
        sampler_factory=DDPMSampler,
        config=config,
        n_samples=3,
        dt=0.1,
        seed=3,
    )
    assert scores and all(np.isfinite(value) for value in scores.values())
