from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.training import (  # noqa: E402
    TrainingConfig,
    train_densifier,
)


class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.visible_bias = torch.nn.Parameter(torch.tensor(100.0))
        self.calls = []

    def forward(self, noisy, steps, **conditions):
        self.calls.append(
            (
                self.training,
                torch.is_grad_enabled(),
                noisy.detach().clone(),
                steps.clone(),
                {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in conditions.items()
                },
            )
        )
        prediction = self.scale * noisy
        return torch.where(conditions["mask"], prediction, self.visible_bias)


@pytest.fixture
def config():
    return TrainingConfig(
        epochs=3,
        batch_size=2,
        diffusion_steps=1,
        beta_start=0.5,
        beta_end=0.5,
        learning_rate=0.1,
    )


def test_densifier_masks_only_interior_points_and_learns(config, tmp_path):
    model = Denoiser()
    train = np.zeros((5, 6, 2))
    validation = np.zeros((3, 6, 2))
    path = tmp_path / "weights" / "densifier.pt"
    result = train_densifier(
        model,
        train,
        validation,
        config=replace(config, epochs=15),
        dt=0.25,
        seed=13,
        checkpoint=path,
    )
    assert result is model
    assert not model.training
    assert model.visible_bias.item() == 100.0
    saved = torch.load(path, weights_only=True)
    assert saved["validation_loss"] < 0.02
    assert saved["stage"] == "densifier"
    assert saved["target_dt"] == 0.25
    assert saved["config"]["epochs"] == 15
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["model_state_dict"][key])
    for training, grad_enabled, noisy, steps, conditions in model.calls:
        assert training == grad_enabled
        assert noisy.shape[1:] == (5, 2)
        assert steps.shape == (len(noisy),)
        assert conditions["dt"] == 0.25
        mask = conditions["mask"]
        assert mask[:, 1::2].all() and not mask[:, ::2].any()
        assert torch.equal(noisy[~mask], conditions["observed"][~mask])
        assert (conditions["observed"][mask] == 0).all()
    np.testing.assert_array_equal(train, 0)
    with pytest.raises(FileExistsError):
        train_densifier(model, train, validation, config=config, checkpoint=path)


def test_seeded_training_and_fixed_validation_noise(config):
    data = np.arange(25, dtype=float).reshape(5, 5, 1)
    first, second = Denoiser(), Denoiser()
    for model in (first, second):
        train_densifier(model, data, data[:3], config=config, seed=42)
    torch.testing.assert_close(first.scale, second.scale, rtol=0, atol=0)
    validation_calls = [call for call in first.calls if not call[0]]
    for previous, following in zip(
        validation_calls[:2], validation_calls[2:4], strict=True
    ):
        torch.testing.assert_close(previous[2], following[2], rtol=0, atol=0)
        torch.testing.assert_close(previous[3], following[3], rtol=0, atol=0)
    for _, _, noisy, _, conditions in first.calls:
        mask = conditions["mask"]
        assert (conditions["observed"][mask] == 0).all()
        torch.testing.assert_close(noisy[:, ::2], conditions["observed"][:, ::2])


def test_restores_best_epoch_instead_of_last(config, tmp_path):
    class WorseningValidation(Denoiser):
        def __init__(self):
            super().__init__()
            self.validation_states = []

        def forward(self, *args, **kwargs):
            prediction = super().forward(*args, **kwargs)
            if not self.training:
                self.validation_states.append(self.scale.detach().clone())
                prediction = prediction + 100 * (len(self.validation_states) - 1)
            return prediction

    model = WorseningValidation()
    path = tmp_path / "best.pt"
    data = np.zeros((2, 5, 1))
    train_densifier(model, data, data, config=config, checkpoint=path)
    assert not torch.equal(model.validation_states[0], model.validation_states[-1])
    torch.testing.assert_close(model.scale, model.validation_states[0])
    assert torch.load(path, weights_only=True)["epoch"] == 1


def test_validation_loss_weights_partial_batches(config, tmp_path):
    class KnownError(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, noisy, steps, *, observed, mask, dt):
            level = observed[:, :1]
            noise = (noisy - 0.5 * level) / np.sqrt(0.75)
            return noise + self.scale * level

    model = KnownError()
    train = np.ones((2, 5, 1))
    validation = np.broadcast_to(np.array([1.0, 2.0, 10.0])[:, None, None], (3, 5, 1))
    path = tmp_path / "weighted.pt"
    train_densifier(
        model,
        train,
        validation,
        checkpoint=path,
        config=replace(config, epochs=1, beta_start=0.75, beta_end=0.75),
    )
    saved = torch.load(path, weights_only=True)
    expected = (1 + 4 + 100) / 3 * model.scale.item() ** 2
    assert saved["validation_loss"] == pytest.approx(expected, rel=1e-6)


def test_failed_checkpoint_write_is_cleaned_up(config, tmp_path, monkeypatch):
    def fail_save(state, path):
        path.write_bytes(b"partial checkpoint")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", fail_save)
    data = np.zeros((2, 5, 1))
    with pytest.raises(OSError, match="interrupted write"):
        train_densifier(
            Denoiser(),
            data,
            data,
            config=config,
            checkpoint=tmp_path / "model.pt",
        )
    assert not list(tmp_path.iterdir())


def test_custom_objective_receives_targets_and_diffusion_context(config, tmp_path):
    calls = []

    def objective(prediction, batch):
        calls.append(torch.is_grad_enabled())
        assert batch.alpha_bar.shape == (len(prediction), 1, 1)
        assert batch.steps.shape == (len(prediction),)
        assert batch.dt == 0.25
        assert (batch.observed[batch.mask] == 0).all()
        expected = (
            batch.alpha_bar.sqrt() * batch.clean
            + (1 - batch.alpha_bar).sqrt() * batch.noise
        )
        torch.testing.assert_close(batch.noisy[batch.mask], expected[batch.mask])
        return (prediction[batch.mask] - batch.clean[batch.mask]).abs().mean()

    data = np.ones((3, 5, 1))
    model = Denoiser()
    path = tmp_path / "custom.pt"
    train_densifier(
        model,
        data,
        data[:2],
        config=config,
        objective=objective,
        dt=0.25,
        checkpoint=path,
    )
    assert any(calls) and not all(calls)
    assert model.scale.item() != 0.0
    assert model.visible_bias.item() == 100.0
    assert torch.load(path, weights_only=True)["objective"].endswith("objective")


@pytest.mark.parametrize(
    "loss, error",
    [
        (0.0, ValueError),
        (torch.ones(2), ValueError),
        (torch.tensor(float("nan")), FloatingPointError),
    ],
)
def test_invalid_objective_results_are_rejected(config, loss, error):
    data = np.zeros((2, 5, 1))
    with pytest.raises(error):
        train_densifier(
            Denoiser(),
            data,
            data,
            config=config,
            objective=lambda prediction, batch: loss,
        )


@pytest.mark.parametrize("wrong_shape", [False, True])
def test_invalid_predictions_are_rejected(config, wrong_shape):
    class InvalidDenoiser(Denoiser):
        def forward(self, noisy, steps, **kwargs):
            if wrong_shape:
                return self.scale * noisy[:, :1]
            return self.scale * noisy + float("nan")

    error = ValueError if wrong_shape else FloatingPointError
    data = np.zeros((2, 5, 1))
    with pytest.raises(error):
        train_densifier(InvalidDenoiser(), data, data, config=config)


@pytest.mark.parametrize(
    "updates",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"diffusion_steps": 0},
        {"learning_rate": np.inf},
        {"max_grad_norm": 0},
        {"beta_start": 0},
        {"beta_end": 1},
        {"beta_end": np.nan},
    ],
)
def test_invalid_training_settings(config, updates):
    with pytest.raises(ValueError):
        replace(config, **updates)
