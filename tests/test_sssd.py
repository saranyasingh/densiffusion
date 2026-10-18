import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.sssd import (  # noqa: E402
    S4D,
    SSSDDenoiser,
    _StepEmbedding,
)
from densiffusion.experiments import run_experiment  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.samplers import DDPMSampler  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402

SIZES = dict(width=8, layers=2, state_size=4, step_dim=8)


def denoiser(channels=1):
    torch.manual_seed(0)
    network = SSSDDenoiser(channels, **SIZES).eval()
    torch.nn.init.normal_(network.output.weight)
    return network


def inputs(batch=2, length=9, channels=1):
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(batch, length, channels, generator=generator)
    mask = torch.zeros(batch, length, channels, dtype=torch.bool)
    mask[:, 1::2] = True
    noisy = torch.where(mask, torch.randn(values.shape, generator=generator), values)
    steps = torch.arange(batch) % 5
    return values.masked_fill(mask, 0), mask, noisy, steps


def predict(network, observed, mask, noisy, steps):
    with torch.no_grad():
        return network(noisy, steps, observed=observed, mask=mask, dt=0.5)


def test_s4d_kernel_is_the_impulse_response_of_its_recurrence():
    # S4D is a linear recurrence x[k] = Ab x[k-1] + Bb u[k], y[k] = 2 Re(C x[k]).
    # Its convolution kernel must equal the output after a unit impulse.
    torch.manual_seed(0)
    layer = S4D(3, state_size=4, bidirectional=False).double()
    kernel = layer.kernel
    A = torch.complex(-kernel.log_A_real.exp(), kernel.A_imag)
    dtA = A * kernel.log_dt.exp()[:, None]
    C = torch.view_as_complex(kernel.C)[0]
    state = torch.zeros_like(A)
    impulse_response = []
    for k in range(6):
        state = dtA.exp() * state + (dtA.exp() - 1) / A * (1.0 if k == 0 else 0.0)
        impulse_response.append(2 * (C * state).sum(-1).real)
    torch.testing.assert_close(
        kernel(6)[0], torch.stack(impulse_response, dim=-1), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("bidirectional", [False, True])
def test_s4d_convolution_matches_direct_sum(bidirectional):
    # The FFT convolution must equal the explicit sum, with no wrap-around.
    torch.manual_seed(0)
    layer = S4D(3, state_size=4, bidirectional=bidirectional).double()
    u = torch.randn(2, 3, 7, dtype=torch.float64)
    k = layer.kernel(7)
    expected = layer.D[:, None] * u
    for t in range(7):
        for s in range(7):
            if s <= t:
                expected[..., t] += k[0, :, t - s] * u[..., s]
            elif bidirectional:
                expected[..., t] += k[1, :, s - t - 1] * u[..., s]
    torch.testing.assert_close(layer.convolve(u), expected, rtol=0, atol=1e-12)


def test_step_embedding_uses_reference_sinusoids():
    embedding = _StepEmbedding(8, 8)
    steps = torch.tensor([0, 3])
    frequencies = 10000.0 ** (-torch.arange(4.0) / 3)
    angles = steps[:, None] * frequencies
    expected = torch.cat((angles.sin(), angles.cos()), dim=-1)
    torch.testing.assert_close(embedding.sinusoids(steps), expected)


def test_untrained_denoiser_predicts_zero_noise():
    torch.manual_seed(0)
    network = SSSDDenoiser(1, **SIZES)
    observed, mask, noisy, steps = inputs()
    prediction = predict(network, observed, mask, noisy, steps)
    assert prediction.shape == noisy.shape
    assert (prediction == 0).all()


def test_prediction_depends_on_hidden_noise_and_observations_only():
    network = denoiser()
    observed, mask, noisy, steps = inputs()
    prediction = predict(network, observed, mask, noisy, steps)

    changed_observed_noise = torch.where(mask, noisy, noisy + 3)
    torch.testing.assert_close(
        predict(network, observed, mask, changed_observed_noise, steps), prediction
    )
    # Hidden points on both sides of t=3 matter. t=7 is beyond the two-step
    # reach of the blocks' 3-tap convolutions, so only a backward S4 kernel
    # can carry its information to t=3.
    for other in (1, 7):
        other_hidden = noisy.clone()
        other_hidden[0, other] += 3
        changed = predict(network, observed, mask, other_hidden, steps)
        assert not torch.allclose(changed[0, 3], prediction[0, 3])
    other_observation = observed.clone()
    other_observation[0, 4] += 3
    changed = predict(network, other_observation, mask, noisy, steps)
    assert not torch.allclose(changed[0, 3], prediction[0, 3])


def test_conditioning_marks_observed_entries_with_one():
    # SSSD's reference code uses mask=1 for observed points; ours uses
    # mask=True for hidden points. The conditioning input must use SSSD's form.
    network = denoiser(channels=2)
    observed, mask, noisy, steps = inputs(channels=2)
    seen = []
    network.blocks[0].condition.register_forward_pre_hook(
        lambda module, args: seen.append(args[0])
    )
    predict(network, observed, mask, noisy, steps)
    (condition,) = seen
    torch.testing.assert_close(condition[:, :2], observed.transpose(1, 2))
    torch.testing.assert_close(condition[:, 2:], (~mask).transpose(1, 2).float())


def test_batch_rows_are_processed_independently():
    network = denoiser(channels=2)
    observed, mask, noisy, steps = inputs(channels=2)
    prediction = predict(network, observed, mask, noisy, steps)
    for row in range(2):
        window = slice(row, row + 1)
        alone = predict(
            network, observed[window], mask[window], noisy[window], steps[window]
        )
        torch.testing.assert_close(alone, prediction[window])


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)"
)
def test_gradients_on_apple_gpu_match_cpu():
    # PyTorch's MPS backend computes ReLU and tanh gradients incorrectly for
    # 3-D tensors with channels-last strides, which transposes can produce.
    network = denoiser()
    observed, mask, noisy, steps = inputs(batch=4, length=33)
    gradients = {}
    for device in ("cpu", "mps"):
        model = SSSDDenoiser(1, **SIZES).to(device)
        model.load_state_dict(network.state_dict())
        prediction = model(
            noisy.to(device),
            steps.to(device),
            observed=observed.to(device),
            mask=mask.to(device),
            dt=0.5,
        )
        prediction.square().mean().backward()
        gradients[device] = {
            name: parameter.grad.cpu()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
    for name, expected in gradients["cpu"].items():
        torch.testing.assert_close(
            gradients["mps"][name], expected, rtol=1e-3, atol=1e-5, msg=name
        )


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError, match="state_size"):
        SSSDDenoiser(1, **(SIZES | {"state_size": 3}))
    with pytest.raises(ValueError, match="layers"):
        SSSDDenoiser(1, **(SIZES | {"layers": 0}))
    network = denoiser(channels=2)
    observed, mask, noisy, steps = inputs(channels=1)
    with pytest.raises(ValueError, match="channels"):
        predict(network, observed, mask, noisy, steps)


@pytest.mark.parametrize("channels", [1, 2])
def test_sssd_trains_and_samples_in_the_shared_workflow(channels):
    rng = np.random.default_rng(2)
    paths = rng.normal(scale=math.sqrt(0.1), size=(12, 9, channels)).cumsum(axis=1)
    torch.manual_seed(0)
    model = ImputationModel(SSSDDenoiser(channels, **SIZES))
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
