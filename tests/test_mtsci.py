import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.mtsci import (  # noqa: E402
    MTSCIDenoiser,
    MTSCIObjective,
    _DiffusionEmbedding,
    info_nce,
)
from densiffusion.experiments import run_experiment  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.objectives import DenoisingBatch, noise_mse  # noqa: E402
from densiffusion.samplers import DDPMSampler  # noqa: E402
from densiffusion.training import TrainingConfig, train_densifier  # noqa: E402

SIZES = dict(
    hidden_channels=8,
    residual_layers=2,
    attention_heads=2,
    diffusion_embedding_dim=8,
    time_embedding_dim=8,
    feature_embedding_dim=4,
    feedforward_dim=8,
)


def network(channels=1):
    torch.manual_seed(0)
    denoiser = MTSCIDenoiser(channels, **SIZES).eval()
    torch.nn.init.normal_(denoiser.output_projection2.weight)
    return denoiser


def inputs(batch=3, length=9, channels=1):
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(batch, length, channels, generator=generator)
    mask = torch.zeros(batch, length, channels, dtype=torch.bool)
    mask[:, 1::2] = True
    noisy = torch.where(mask, torch.randn(values.shape, generator=generator), values)
    steps = torch.arange(batch) % 5
    return values, mask, noisy, steps


def predict(denoiser, observed, mask, noisy, steps):
    with torch.no_grad():
        return denoiser(noisy, steps, observed=observed, mask=mask, dt=0.5)


def denoising_batch(values, mask, steps, seed=2):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(values.shape, generator=generator)
    alpha_bar = torch.linspace(0.2, 0.9, len(values))[:, None, None]
    noisy = alpha_bar.sqrt() * values + (1 - alpha_bar).sqrt() * noise
    return DenoisingBatch(
        clean=values,
        noisy=torch.where(mask, noisy, values),
        noise=noise,
        observed=values.masked_fill(mask, 0),
        mask=mask,
        steps=steps,
        alpha_bar=alpha_bar,
        dt=0.5,
    )


def test_untrained_denoiser_predicts_its_output_bias():
    torch.manual_seed(0)
    denoiser = MTSCIDenoiser(2, **SIZES)
    values, mask, noisy, steps = inputs(channels=2)
    prediction = predict(denoiser, values, mask, noisy, steps)
    assert prediction.shape == noisy.shape
    torch.testing.assert_close(
        prediction,
        denoiser.output_projection2.bias.expand_as(prediction),
    )


def test_prediction_ignores_hidden_observations_and_observed_noise():
    denoiser = network()
    values, mask, noisy, steps = inputs()
    observed = values.masked_fill(mask, 0)
    prediction = predict(denoiser, observed, mask, noisy, steps)
    torch.testing.assert_close(
        predict(denoiser, observed + 7 * mask, mask, noisy, steps), prediction
    )
    torch.testing.assert_close(
        predict(denoiser, observed, mask, torch.where(mask, noisy, noisy + 3), steps),
        prediction,
    )
    nan_hidden = observed.masked_fill(mask, float("nan"))
    torch.testing.assert_close(
        predict(denoiser, nan_hidden, mask, noisy, steps), prediction
    )


def test_hidden_points_and_observations_both_influence_predictions():
    denoiser = network()
    values, mask, noisy, steps = inputs()
    observed = values.masked_fill(mask, 0)
    prediction = predict(denoiser, observed, mask, noisy, steps)
    other_hidden = noisy.clone()
    other_hidden[0, 7] += 3
    changed = predict(denoiser, observed, mask, other_hidden, steps)
    assert not torch.allclose(changed[0, 1], prediction[0, 1])
    other_observation = observed.clone()
    other_observation[0, 8] += 3
    changed = predict(denoiser, other_observation, mask, noisy, steps)
    assert not torch.allclose(changed[0, 1], prediction[0, 1])
    changed = predict(denoiser, observed, mask, noisy, (steps + 1) % 5)
    assert not torch.allclose(changed, prediction)


def test_mask_enters_as_side_information():
    denoiser = network()
    values, mask, _, steps = inputs()
    # Same input plane, different conditioning indicator.
    other = mask.clone()
    other[:, 0] = True
    first = predict(denoiser, values.masked_fill(mask, 0), mask, values, steps)
    second = predict(denoiser, values.masked_fill(other, 0), other, values, steps)
    assert not torch.allclose(first[:, 1], second[:, 1])


def test_side_indicator_is_one_at_observed_entries():
    denoiser = network()
    values, mask, noisy, steps = inputs()
    seen = []
    layer = denoiser.residual_layers[0].side_projection
    layer.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    predict(denoiser, values.masked_fill(mask, 0), mask, noisy, steps)
    indicator = seen[0][..., -1]
    assert torch.equal(indicator, (~mask).float())


def test_feature_attention_mixes_channels_and_rows_stay_independent():
    denoiser = network(channels=2)
    values, mask, noisy, steps = inputs(channels=2)
    prediction = predict(denoiser, values, mask, noisy, steps)
    changed = values.clone()
    changed[0, 2, 1] += 3
    result = predict(denoiser, changed, mask, noisy, steps)
    assert not torch.allclose(result[0, :, 0], prediction[0, :, 0])
    torch.testing.assert_close(result[1:], prediction[1:])
    for row in range(len(values)):
        window = slice(row, row + 1)
        alone = predict(
            denoiser, values[window], mask[window], noisy[window], steps[window]
        )
        torch.testing.assert_close(alone, prediction[window])


def test_step_embedding_stays_finite_in_float16():
    torch.manual_seed(0)
    embedding = _DiffusionEmbedding(8)
    steps = torch.tensor([0, 7, 49, 999])
    reference = embedding(steps)
    result = embedding.half()(steps)
    assert result.dtype == torch.float16 and torch.isfinite(result).all()
    torch.testing.assert_close(result.float(), reference, rtol=0, atol=1e-2)


def test_info_nce_matches_definition():
    generator = torch.Generator().manual_seed(0)
    query, positive = torch.randn(2, 4, 6, generator=generator, dtype=torch.float64)
    loss = info_nce(query, positive, temperature=0.5)
    q = query / query.norm(dim=1, keepdim=True)
    p = positive / positive.norm(dim=1, keepdim=True)
    logits = q @ p.T / 0.5
    expected = -(logits.diag() - logits.logsumexp(dim=1)).mean()
    torch.testing.assert_close(loss, expected)
    aligned = info_nce(positive * 3, positive, temperature=0.01)
    assert aligned < 1e-3
    with pytest.raises(ValueError, match="temperature"):
        info_nce(query, positive, temperature=0)


def test_consistency_views_use_complementary_conditioning():
    denoiser = network()
    values, mask, _, steps = inputs()
    batch = denoising_batch(values, mask, steps)
    calls = []
    original = denoiser.encode

    def record(views, view_steps, *, conditioning):
        calls.append((views, view_steps, conditioning))
        return original(views, view_steps, conditioning=conditioning)

    denoiser.encode = record
    denoiser.consistency_loss(batch)
    (views, view_steps, conditioning) = calls[0]
    forward, reverse, negative = views.chunk(3)
    noisy = batch.alpha_bar.sqrt() * values + (1 - batch.alpha_bar).sqrt() * (
        batch.noise
    )
    torch.testing.assert_close(forward, batch.noisy)
    torch.testing.assert_close(reverse[mask], values[mask])
    torch.testing.assert_close(reverse[~mask], noisy[~mask])
    torch.testing.assert_close(negative, noisy)
    assert torch.equal(conditioning, torch.cat((~mask, mask, torch.zeros_like(mask))))
    assert torch.equal(view_steps, steps.repeat(3))


def test_objective_adds_consistency_only_in_training_mode():
    denoiser = network()
    values, mask, noisy, steps = inputs()
    batch = denoising_batch(values, mask, steps)
    prediction = torch.zeros_like(values)
    objective = MTSCIObjective(denoiser, consistency_weight=2.0)
    noise_only = noise_mse(prediction, batch)
    torch.testing.assert_close(objective(prediction, batch), noise_only)
    denoiser.train()
    torch.manual_seed(5)
    total = objective(prediction, batch)
    torch.manual_seed(5)
    consistency = denoiser.consistency_loss(batch)
    torch.testing.assert_close(total, noise_only + 2.0 * consistency)
    total.backward()
    assert denoiser.input_projection.weight.grad.abs().sum() > 0
    off = MTSCIObjective(denoiser, consistency_weight=0)
    torch.testing.assert_close(off(prediction, batch), noise_only)
    with pytest.raises(ValueError, match="consistency_weight"):
        MTSCIObjective(denoiser, consistency_weight=-1)


def test_rejects_invalid_inputs_and_sizes():
    denoiser = network(channels=2)
    values, mask, noisy, steps = inputs(channels=2)
    with pytest.raises(ValueError, match="n_channels"):
        denoiser(
            noisy[..., :1], steps, observed=values[..., :1], mask=mask[..., :1], dt=1
        )
    with pytest.raises(ValueError, match="boolean"):
        denoiser(noisy, steps, observed=values, mask=mask.float(), dt=1)
    with pytest.raises(ValueError, match="steps"):
        denoiser(noisy, steps.float(), observed=values, mask=mask, dt=1)
    with pytest.raises(ValueError, match="dt"):
        denoiser(noisy, steps, observed=values, mask=mask, dt=0)
    with pytest.raises(ValueError, match="divisible"):
        MTSCIDenoiser(1, hidden_channels=6, attention_heads=4)
    with pytest.raises(ValueError, match="dropout"):
        MTSCIDenoiser(1, dropout=1.0)


@pytest.mark.parametrize("channels", [1, 2])
def test_mtsci_trains_and_samples_in_the_shared_workflow(channels):
    rng = np.random.default_rng(2)
    paths = rng.normal(scale=math.sqrt(0.1), size=(12, 9, channels)).cumsum(axis=1)
    torch.manual_seed(0)
    denoiser = MTSCIDenoiser(channels, **SIZES)
    model = ImputationModel(denoiser)
    config = TrainingConfig(
        epochs=2, batch_size=4, diffusion_steps=5, beta_end=0.2, schedule="quadratic"
    )
    scores = run_experiment(
        model,
        paths[:8],
        paths[8:10],
        paths[10:],
        sampler_factory=DDPMSampler,
        objective=MTSCIObjective(denoiser),
        config=config,
        n_samples=3,
        dt=0.1,
        seed=3,
    )
    assert scores and all(np.isfinite(value) for value in scores.values())


def test_consistency_term_changes_training():
    rng = np.random.default_rng(4)
    paths = rng.normal(size=(16, 9, 1)).cumsum(axis=1)
    config = TrainingConfig(epochs=2, batch_size=8, diffusion_steps=5)
    weights = []
    for weight in (0.0, 5.0):
        torch.manual_seed(0)
        denoiser = MTSCIDenoiser(1, **SIZES)
        objective = MTSCIObjective(denoiser, consistency_weight=weight)
        train_densifier(
            ImputationModel(denoiser),
            paths[:12],
            paths[12:],
            config=config,
            objective=objective,
            seed=1,
        )
        weights.append(denoiser.input_projection.weight.detach().clone())
    assert not torch.allclose(*weights)
