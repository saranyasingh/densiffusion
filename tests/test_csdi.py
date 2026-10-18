import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.csdi import CSDIDenoiser  # noqa: E402
from densiffusion.experiments import run_experiment  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.objectives import DenoisingBatch, noise_mse  # noqa: E402
from densiffusion.pipeline import impute  # noqa: E402
from densiffusion.samplers import DDPMSampler  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def denoiser(channels=1):
    torch.manual_seed(17)
    return CSDIDenoiser(
        channels,
        hidden_channels=8,
        residual_layers=2,
        attention_heads=2,
        diffusion_embedding_dim=8,
        time_embedding_dim=8,
        feature_embedding_dim=4,
        feedforward_dim=16,
        dropout=0,
    )


def activate_output(network):
    # The reference zero-initializes this weight. A nonconstant output is
    # essential: otherwise leakage and embedding tests would pass vacuously.
    torch.nn.init.normal_(network.output_projection2.weight, std=0.2)


@pytest.mark.parametrize("channels,length", [(1, 7), (3, 7), (2, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_and_finite_gradients(channels, length, dtype):
    network = denoiser(channels).to(dtype)
    model = ImputationModel(network)
    clean = torch.randn(2, length, channels, dtype=dtype)
    mask = torch.ones_like(clean, dtype=torch.bool)
    mask[:, 0, 0] = False
    steps = torch.tensor([0, 3])
    alpha = torch.tensor([0.9, 0.2], dtype=dtype)[:, None, None]
    noise = torch.randn_like(clean)
    observed = clean.masked_fill(mask, 0)
    noisy = torch.where(mask, alpha.sqrt() * clean + (1 - alpha).sqrt() * noise, clean)
    batch = DenoisingBatch(
        clean=clean,
        noisy=noisy,
        noise=noise,
        observed=observed,
        mask=mask,
        steps=steps,
        alpha_bar=alpha,
        dt=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, steps, observed=observed, mask=mask, dt=0.1)
        assert prediction.shape == clean.shape and prediction.dtype == dtype
        assert torch.isfinite(prediction).all()
        loss = noise_mse(prediction, batch)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step()
    assert network.input_projection.weight.grad.abs().sum() > 0
    if length > 1:
        assert (
            network.residual_layers[0]
            .time_attention.self_attn.in_proj_weight.grad.abs()
            .sum()
            > 0
        )
    if channels > 1:
        assert (
            network.residual_layers[0]
            .feature_attention.self_attn.in_proj_weight.grad.abs()
            .sum()
            > 0
        )


def test_hidden_conditioning_cannot_leak_and_observations_inform_targets():
    network = denoiser(2).eval()
    activate_output(network)
    noisy = torch.randn(2, 5, 2, requires_grad=True)
    observed = torch.randn_like(noisy, requires_grad=True)
    mask = torch.zeros_like(noisy, dtype=torch.bool)
    mask[:, 1::2] = True
    mask[0, 2, 1] = True
    steps = torch.tensor([0, 4])
    prediction = network(noisy, steps, observed=observed, mask=mask, dt=0.2)
    contaminated = observed.detach().masked_fill(mask, float("nan"))
    unused_noisy = noisy.detach().masked_fill(~mask, float("nan"))
    same = network(unused_noisy, steps, observed=contaminated, mask=mask, dt=0.2)
    torch.testing.assert_close(prediction, same, rtol=0, atol=0)
    prediction[mask].square().sum().backward()
    assert (observed.grad[mask] == 0).all()
    assert (noisy.grad[~mask] == 0).all()
    assert observed.grad[~mask].abs().sum() > 0
    assert noisy.grad[mask].abs().sum() > 0
    changed = network(noisy, steps, observed=observed + 2, mask=mask, dt=0.2)
    assert not torch.allclose(prediction[mask], changed[mask])


def test_measurement_time_feature_mask_and_diffusion_embeddings_are_distinct():
    network = denoiser(2).eval()
    activate_output(network)
    records = []
    handle = network.residual_layers[0].register_forward_pre_hook(
        lambda module, args: records.append(tuple(x.detach().clone() for x in args))
    )
    values = torch.randn(2, 5, 2)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[:, 1::2] = True
    for steps, dt in [
        (torch.tensor([0, 3]), 0.2),
        (torch.tensor([1, 4]), 0.2),
        (torch.tensor([0, 3]), 0.1),
    ]:
        network(values, steps, observed=values, mask=mask, dt=dt)
    handle.remove()
    _, side, diffusion = records[0]
    torch.testing.assert_close(side, records[1][1])
    assert not torch.allclose(diffusion, records[1][2])
    torch.testing.assert_close(diffusion, records[2][2])
    assert not torch.allclose(side[..., :8], records[2][1][..., :8])
    torch.testing.assert_close(side[..., -1], (~mask).float())
    torch.testing.assert_close(side[0, 0, :, 8:-1], network.feature_embedding.weight)
    phase = torch.arange(5)[:, None] * 0.2 * 10000 ** (-torch.arange(0, 8, 2) / 8)
    expected = torch.stack((phase.sin(), phase.cos()), -1).flatten(1)
    torch.testing.assert_close(side[0, :, 0, :8], expected)

    # Official lookup table rows 0 and 3; no extra +1 for the paper's 1..T notation.
    phases = torch.tensor([0, 3])[:, None] * 10 ** (torch.arange(4) / 3 * 4)
    embedding = torch.cat((phases.sin(), phases.cos()), -1)
    expected = torch.nn.functional.silu(
        network.diffusion_embedding.projection2(
            torch.nn.functional.silu(network.diffusion_embedding.projection1(embedding))
        )
    )
    torch.testing.assert_close(diffusion, expected)


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_sampling_pipeline_preserves_observations_at_every_reverse_step(explicit_mask):
    model = ImputationModel(denoiser(2))
    activate_output(model.denoiser)
    config = TrainingConfig(batch_size=3, diffusion_steps=4, beta_end=0.5)
    values = np.random.default_rng(5).normal(size=(2, 5, 2))
    mask = np.zeros(values.shape, dtype=bool)
    mask[:, 1::2] = True
    if explicit_mask:
        mask[0, 2, 1] = True
        inputs = np.where(mask, np.nan, values)
        options = dict(factor=1, mask=mask, dt=0.1)
    else:
        inputs = values[:, ::2]
        options = dict(factor=2, dt=0.2)
    calls = []

    def check_inputs(module, args, kwargs):
        noisy, steps = args
        hidden, observed = kwargs["mask"], kwargs["observed"]
        assert (observed[hidden] == 0).all()
        torch.testing.assert_close(noisy[~hidden], observed[~hidden], rtol=0, atol=0)
        assert kwargs["dt"] == 0.1
        calls.append(steps.clone())

    handle = model.register_forward_pre_hook(check_inputs, with_kwargs=True)
    sampler = DDPMSampler(model, config)
    samples = impute(inputs, sampler, n_samples=2, seed=9, **options)
    handle.remove()
    assert samples.shape == (2, 2, 5, 2) and np.isfinite(samples).all()
    assert [int(s[0]) for s in calls] == [3, 2, 1, 0] * 2
    for draw in np.moveaxis(samples, 1, 0):
        np.testing.assert_array_equal(draw[~mask], values[~mask])
    np.testing.assert_array_equal(
        samples, impute(inputs, sampler, n_samples=2, seed=9, **options)
    )
    if explicit_mask:
        np.testing.assert_array_equal(
            samples, impute(values, sampler, n_samples=2, seed=9, **options)
        )


def test_shared_training_and_sampling_selects_reloadable_weights(tmp_path):
    model = ImputationModel(denoiser())
    values = np.random.default_rng(5).normal(size=(9, 7, 1))
    path = tmp_path / "csdi.pt"
    scores = run_experiment(
        model,
        values[:5],
        values[5:7],
        values[7:],
        sampler_factory=DDPMSampler,
        objective=noise_mse,
        config=TrainingConfig(epochs=2, batch_size=3, diffusion_steps=4, beta_end=0.5),
        n_samples=3,
        dt=0.1,
        seed=31,
        checkpoint=path,
    )
    assert np.isfinite(scores["crps"]) and scores["crps"] >= 0
    assert not model.training
    saved = torch.load(path, weights_only=True)
    restored = ImputationModel(denoiser())
    restored.load_state_dict(saved["model_state_dict"])
    assert saved["objective"] == "noise_mse" and saved["target_dt"] == 0.1
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])
    assert model.denoiser.output_projection2.weight.abs().sum() > 0


@pytest.mark.parametrize(
    "options",
    [
        {"n_channels": 0},
        {"residual_layers": 0},
        {"hidden_channels": 7},
        {"diffusion_embedding_dim": 2},
        {"diffusion_embedding_dim": 7},
        {"time_embedding_dim": 3},
        {"dropout": float("nan")},
    ],
)
def test_invalid_architecture_sizes(options):
    with pytest.raises(ValueError):
        CSDIDenoiser(**({"n_channels": 1} | options))
