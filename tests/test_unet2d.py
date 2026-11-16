import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.unet2d import UNet2DDenoiser  # noqa: E402
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


def denoiser(planes=1):
    torch.manual_seed(17)
    return UNet2DDenoiser(
        planes,
        hidden_channels=8,
        channel_multipliers=(1, 2),
        blocks_per_level=1,
        diffusion_embedding_dim=8,
        groups=4,
    )


def activate_output(network):
    # The final convolution starts at zero. A nonconstant output is essential:
    # otherwise leakage and embedding tests would pass vacuously.
    torch.nn.init.normal_(network.output_projection.weight, std=0.2)


@pytest.mark.parametrize("planes,rows,columns", [(1, 5, 7), (3, 6, 5), (2, 1, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_and_finite_gradients(planes, rows, columns, dtype):
    network = denoiser(planes).to(dtype)
    model = ImputationModel(network)
    clean = torch.randn(2, rows, columns * planes, dtype=dtype)
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
    assert network.middle[0].conv1.weight.grad.abs().sum() > 0
    assert network.step_embedding.projection1.weight.grad.abs().sum() > 0


def test_planes_are_the_adjacent_channels_of_each_pixel():
    network = denoiser(3).eval()
    records = []
    handle = network.input_projection.register_forward_pre_hook(
        lambda module, args: records.append(args[0].detach().clone())
    )
    values = torch.randn(2, 4, 15)
    mask = torch.rand(values.shape) < 0.5
    network(values, torch.tensor([0, 1]), observed=values, mask=mask, dt=1.0)
    handle.remove()
    (inputs,) = records
    assert inputs.shape == (2, 9, 4, 5)
    grid = values.reshape(2, 4, 5, 3).permute(0, 3, 1, 2)
    hidden = mask.reshape(2, 4, 5, 3).permute(0, 3, 1, 2)
    torch.testing.assert_close(inputs[:, :3], grid.masked_fill(hidden, 0))
    torch.testing.assert_close(inputs[:, 3:6], grid.masked_fill(~hidden, 0))
    torch.testing.assert_close(inputs[:, 6:], (~hidden).float())


def test_hidden_conditioning_cannot_leak_and_observations_inform_targets():
    network = denoiser(2).eval()
    activate_output(network)
    noisy = torch.randn(2, 5, 6, requires_grad=True)
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


def test_diffusion_step_changes_prediction_but_dt_does_not():
    network = denoiser().eval()
    activate_output(network)
    values = torch.randn(2, 5, 4)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[:, 1::2] = True
    steps = torch.tensor([0, 3])
    reference = network(values, steps, observed=values, mask=mask, dt=0.2)
    same = network(values, steps, observed=values, mask=mask, dt=0.1)
    torch.testing.assert_close(reference, same, rtol=0, atol=0)
    later = network(values, steps + 1, observed=values, mask=mask, dt=0.2)
    assert not torch.allclose(reference, later)


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_sampling_pipeline_preserves_observations_at_every_reverse_step(explicit_mask):
    model = ImputationModel(denoiser(2))
    activate_output(model.denoiser)
    config = TrainingConfig(batch_size=3, diffusion_steps=4, beta_end=0.5)
    values = np.random.default_rng(5).normal(size=(2, 5, 6))
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
        calls.append(steps.clone())

    handle = model.register_forward_pre_hook(check_inputs, with_kwargs=True)
    sampler = DDPMSampler(model, config)
    samples = impute(inputs, sampler, n_samples=2, seed=9, **options)
    handle.remove()
    assert samples.shape == (2, 2, 5, 6) and np.isfinite(samples).all()
    assert [int(s[0]) for s in calls] == [3, 2, 1, 0] * 2
    for draw in np.moveaxis(samples, 1, 0):
        np.testing.assert_array_equal(draw[~mask], values[~mask])
    np.testing.assert_array_equal(
        samples, impute(inputs, sampler, n_samples=2, seed=9, **options)
    )


def test_shared_training_and_sampling_selects_reloadable_weights(tmp_path):
    model = ImputationModel(denoiser())
    values = np.random.default_rng(5).normal(size=(9, 7, 5))
    path = tmp_path / "unet.pt"
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
    assert model.denoiser.output_projection.weight.abs().sum() > 0


@pytest.mark.parametrize(
    "options",
    [
        {"planes": 0},
        {"hidden_channels": 6},
        {"channel_multipliers": ()},
        {"channel_multipliers": (1, 0)},
        {"blocks_per_level": 0},
        {"diffusion_embedding_dim": 7},
        {"dropout": 1.0},
    ],
)
def test_invalid_architecture_sizes(options):
    with pytest.raises(ValueError):
        UNet2DDenoiser(**({"planes": 1, "hidden_channels": 8, "groups": 4} | options))


def test_invalid_forward_inputs():
    network = denoiser(2)
    values = torch.zeros(1, 3, 5)
    mask = torch.zeros_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="planes"):
        network(values, torch.tensor([0]), observed=values, mask=mask, dt=1.0)
    values = torch.zeros(1, 3, 4)
    mask = torch.zeros_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="boolean"):
        network(values, torch.tensor([0]), observed=values, mask=values, dt=1.0)
    with pytest.raises(ValueError, match="steps"):
        network(values, torch.tensor([-1]), observed=values, mask=mask, dt=1.0)
    with pytest.raises(ValueError, match="dt"):
        network(values, torch.tensor([0]), observed=values, mask=mask, dt=0.0)
