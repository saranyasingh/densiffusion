from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.objectives import noise_mse  # noqa: E402
from densiffusion.training import TrainingConfig, train_forecaster  # noqa: E402


class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.visible_bias = torch.nn.Parameter(torch.tensor(100.0))

    def forward(self, noisy, steps, *, observed, mask, dt):
        return torch.where(mask, self.scale * noisy, self.visible_bias)


class RecordingDensifier:
    def __init__(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
        self.calls = []

    def sample(self, observed, *, factor, n_samples, dt, rng):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in self.model.modules())
        assert n_samples == 1
        positions = np.arange((observed.shape[1] - 1) * factor + 1) / factor
        left = positions.astype(int)
        right = np.minimum(left + 1, observed.shape[1] - 1)
        samples = observed[:, left] + (positions - left)[None, :, None] * (
            observed[:, right] - observed[:, left]
        )
        samples += rng.standard_normal(samples.shape)
        samples[:, ::factor] = observed
        self.calls.append((observed.copy(), samples.copy(), dt))
        return samples[:, None]


@pytest.fixture
def config():
    return TrainingConfig(epochs=3, batch_size=2, diffusion_steps=2)


@pytest.fixture
def windows():
    history = np.arange(12, dtype=float).reshape(4, 3, 1)
    future = 100 + np.arange(8, dtype=float).reshape(4, 2, 1)
    return history, future


def test_future_only_training_bridge_alignment_and_frozen_densifier(
    config, windows, tmp_path
):
    densifier = RecordingDensifier()
    densifier.model[1].eval()
    densifier.model[0].bias.requires_grad_(False)
    modes = [module.training for module in densifier.model.modules()]
    weights = {
        name: value.clone() for name, value in densifier.model.state_dict().items()
    }
    parameters = list(densifier.model.parameters())
    grad_flags = [parameter.requires_grad for parameter in parameters]
    model = Denoiser()
    seen = []
    factor = 3
    dense_context = (windows[0].shape[1] - 1) * factor + 1

    def objective(prediction, batch):
        assert batch.dt == pytest.approx(0.2)
        assert not batch.mask[:, :dense_context].any()
        assert batch.mask[:, dense_context:].all()
        assert (batch.observed[batch.mask] == 0).all()
        torch.testing.assert_close(batch.clean[~batch.mask], batch.noisy[~batch.mask])
        torch.testing.assert_close(
            batch.clean[~batch.mask], batch.observed[~batch.mask]
        )
        indices = (batch.clean[:, 0, 0] / 3).to(torch.long).numpy()
        np.testing.assert_array_equal(
            batch.clean[:, :dense_context:factor].numpy(), windows[0][indices]
        )
        np.testing.assert_array_equal(
            batch.clean[:, dense_context + factor - 1 :: factor].numpy(),
            windows[1][indices],
        )
        seen.append((torch.is_grad_enabled(), batch.clean.clone(), batch.noisy.clone()))
        return noise_mse(prediction, batch)

    checkpoint = tmp_path / "forecaster.pt"
    result = train_forecaster(
        model,
        densifier,
        windows,
        tuple(array[:2] for array in windows),
        factor=factor,
        n_augmentations=3,
        dt=0.6,
        seed=13,
        config=config,
        objective=objective,
        checkpoint=checkpoint,
    )
    assert result is model and not model.training
    assert model.visible_bias.item() == 100.0
    assert model.scale.item() != 0.0
    assert [module.training for module in densifier.model.modules()] == modes
    assert [parameter.requires_grad for parameter in parameters] == grad_flags
    assert all(parameter.grad is None for parameter in parameters)
    for name, value in densifier.model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    # Augmentations are batched and cached once, independent of epoch count.
    assert len(densifier.calls) == 2 * (6 + 3)
    for history_call, future_call in zip(
        densifier.calls[::2], densifier.calls[1::2], strict=True
    ):
        history, dense_history, spacing = history_call
        bridge, dense_future, _ = future_call
        assert len(history) <= config.batch_size
        assert spacing == 0.6
        np.testing.assert_array_equal(bridge[:, 0], history[:, -1])
        indices = (history[:, 0, 0] / 3).astype(int)
        np.testing.assert_array_equal(history, windows[0][indices])
        np.testing.assert_array_equal(bridge[:, 1:], windows[1][indices])
        np.testing.assert_array_equal(
            dense_future[:, factor::factor], windows[1][indices]
        )
        assert dense_history.shape[1] == dense_context
    validation_calls = [
        (clean, noisy) for training, clean, noisy in seen if not training
    ]
    for first, following in zip(
        validation_calls[:3], validation_calls[3:6], strict=True
    ):
        torch.testing.assert_close(first[0], following[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], following[1], rtol=0, atol=0)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["stage"] == "forecaster"
    assert saved["factor"] == factor
    assert saved["observed_dt"] == 0.6
    assert saved["target_dt"] == pytest.approx(0.2)
    assert saved["context_length"] == 3 and saved["horizon"] == 2
    assert saved["dense_context_length"] == 7 and saved["dense_horizon"] == 6
    assert saved["n_augmentations"] == 3
    assert saved["augmentation"]["history_conditioning"] == "history_only"
    assert saved["augmentation"]["cached"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["model_state_dict"][name])


def test_augmentation_and_training_reproduce_and_histories_never_see_future(
    config, windows
):
    runs = []
    for shift in (0, 0, 1000):
        densifier = RecordingDensifier()
        model = Denoiser()
        pair = windows[0], windows[1] + shift
        train_forecaster(
            model,
            densifier,
            pair,
            pair,
            config=config,
            n_augmentations=2,
            seed=17,
        )
        runs.append((model, densifier.calls))
    torch.testing.assert_close(runs[0][0].scale, runs[1][0].scale, rtol=0, atol=0)
    for first, repeated, changed_future in zip(
        runs[0][1][::2], runs[1][1][::2], runs[2][1][::2], strict=True
    ):
        np.testing.assert_array_equal(first[0], changed_future[0])
        np.testing.assert_array_equal(first[1], repeated[1])
        np.testing.assert_array_equal(first[1], changed_future[1])
    # Repeated augmentations of a window differ at sampled interior points.
    assert not np.array_equal(runs[0][1][0][1][0, 1::2], runs[0][1][0][1][1, 1::2])
    # Independent RNG streams are used even for identical conditioning windows.
    identical = (windows[0], windows[0][:, 1:])
    densifier = RecordingDensifier()
    train_forecaster(
        Denoiser(), densifier, identical, identical, config=config, seed=17
    )
    history_draw, future_draw = densifier.calls[:2]
    history_noise = history_draw[1][:, 1] - history_draw[0][:, :2].mean(axis=1)
    future_noise = future_draw[1][:, 1] - future_draw[0][:, :2].mean(axis=1)
    assert not np.array_equal(history_noise, future_noise)


def test_factor_one_bypasses_densifier(config, windows, tmp_path):
    class NeverSample:
        def sample(self, *args, **kwargs):
            raise AssertionError("factor=1 must bypass densification")

    calls = []

    def objective(prediction, batch):
        calls.append(batch.clean.clone())
        assert batch.dt == 0.5
        assert batch.clean.shape[1:] == (5, 1)
        assert not batch.mask[:, :3].any() and batch.mask[:, 3:].all()
        indices = (batch.clean[:, 0, 0] / 3).to(torch.long).numpy()
        np.testing.assert_array_equal(
            batch.clean.numpy(), np.concatenate(windows, axis=1)[indices]
        )
        return noise_mse(prediction, batch)

    path = tmp_path / "baseline.pt"
    train_forecaster(
        Denoiser(),
        NeverSample(),
        windows,
        windows,
        config=replace(config, epochs=1),
        factor=1,
        dt=0.5,
        objective=objective,
        checkpoint=path,
    )
    assert len(calls) == 4
    assert torch.load(path, weights_only=True)["augmentation"]["densifier_bypassed"]


def test_sampler_may_reuse_output_buffer(config, windows):
    class BufferDensifier(RecordingDensifier):
        def __init__(self):
            super().__init__()
            self.buffer = None

        def sample(self, observed, **kwargs):
            samples = super().sample(observed, **kwargs)
            if self.buffer is None:
                self.buffer = samples.copy()
            self.buffer[:] = samples
            return self.buffer

    def objective(prediction, batch):
        indices = (batch.clean[:, 0, 0] / 3).to(torch.long).numpy()
        np.testing.assert_array_equal(batch.clean[:, :5:2].numpy(), windows[0][indices])
        return noise_mse(prediction, batch)

    train_forecaster(
        Denoiser(),
        BufferDensifier(),
        windows,
        windows,
        config=replace(config, epochs=1),
        objective=objective,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"factor": 0},
        {"factor": True},
        {"dt": 0},
        {"dt": float("inf")},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**63},
        {"n_augmentations": 0},
        {"n_augmentations": True},
        {"dt": np.nextafter(0.0, 1.0), "factor": 2},
    ],
)
def test_invalid_options_rejected_before_sampling(config, windows, options):
    densifier = RecordingDensifier()
    with pytest.raises(ValueError):
        train_forecaster(
            Denoiser(), densifier, windows, windows, config=config, **options
        )
    assert not densifier.calls


@pytest.mark.parametrize(
    "invalid",
    [
        (),
        (np.zeros((2, 3, 1)),),
        (np.zeros((2, 3, 1)), np.zeros((3, 2, 1))),
        (np.zeros((2, 3, 1)), np.zeros((2, 2, 2))),
        (np.zeros((2, 4, 1)), np.zeros((2, 2, 1))),
        (np.zeros((2, 3, 1)), np.zeros((2, 0, 1))),
        (np.zeros((2, 3, 1)), np.full((2, 2, 1), np.nan)),
    ],
)
def test_invalid_validation_windows_rejected_before_sampling(config, windows, invalid):
    densifier = RecordingDensifier()
    with pytest.raises(ValueError):
        train_forecaster(Denoiser(), densifier, windows, invalid, config=config)
    assert not densifier.calls


def test_existing_checkpoint_rejected_before_sampling(config, windows, tmp_path):
    densifier = RecordingDensifier()
    path = tmp_path / "existing.pt"
    path.write_text("already saved")
    with pytest.raises(FileExistsError):
        train_forecaster(
            Denoiser(), densifier, windows, windows, config=config, checkpoint=path
        )
    assert not densifier.calls
    assert path.read_text() == "already saved"


@pytest.mark.parametrize("share_storage_only", [False, True])
@pytest.mark.parametrize("factor", [1, 2])
def test_shared_forecaster_parameters_are_rejected(
    config, windows, share_storage_only, factor
):
    model = Denoiser()
    densifier = RecordingDensifier()
    if share_storage_only:
        densifier.model[0].weight = torch.nn.Parameter(model.scale.view(1, 1))
    else:
        densifier.model = model
    with pytest.raises(ValueError, match="must not share parameters"):
        train_forecaster(
            model, densifier, windows, windows, config=config, factor=factor
        )
    assert not densifier.calls


@pytest.mark.parametrize("failure", ["exception", "observed", "shape", "nan"])
def test_bad_densifier_restores_modes_and_does_not_train(config, windows, failure):
    class BrokenDensifier(RecordingDensifier):
        def sample(self, observed, **kwargs):
            samples = super().sample(observed, **kwargs)
            if failure == "exception":
                raise RuntimeError("sampling failed")
            if failure == "observed":
                samples[:, :, 0] += 1
            elif failure == "shape":
                samples = samples[:, :, :-1]
            elif failure == "nan":
                samples[:] = np.nan
            return samples

    densifier = BrokenDensifier()
    densifier.model[1].eval()
    modes = [module.training for module in densifier.model.modules()]
    model = Denoiser()
    with pytest.raises((ValueError, RuntimeError)):
        train_forecaster(model, densifier, windows, windows, config=config)
    assert [module.training for module in densifier.model.modules()] == modes
    assert model.scale.item() == 0
