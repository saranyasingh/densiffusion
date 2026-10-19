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


def test_custom_masks_follow_shuffled_targets_and_weight_hidden_entries(
    config, tmp_path
):
    values = np.broadcast_to(np.array([1.0, 2.0, 10.0])[:, None, None], (3, 4, 2))
    mask = np.array(
        [
            [[False, False], [True, False], [False, False], [False, False]],
            [[False, False], [False, False], [True, True], [False, False]],
            [[False, False], [True, True], [True, True], [True, True]],
        ]
    )
    model = Denoiser()
    path = tmp_path / "masked.pt"

    def objective(prediction, batch):
        indices = [
            np.flatnonzero(np.array([1.0, 2.0, 10.0]) == row[0, 0].item())[0]
            for row in batch.clean
        ]
        torch.testing.assert_close(batch.mask, torch.as_tensor(mask[indices]))
        assert batch.clean.shape[1] == 4
        assert (batch.observed[batch.mask] == 0).all()
        torch.testing.assert_close(batch.noisy[~batch.mask], batch.clean[~batch.mask])
        return batch.clean[batch.mask].mean() + 0 * prediction.mean()

    train_densifier(
        model,
        values,
        values,
        train_mask=mask,
        validation_mask=mask,
        config=replace(config, epochs=1),
        objective=objective,
        checkpoint=path,
    )
    saved = torch.load(path, weights_only=True)
    expected = values[mask].mean()
    assert saved["train_loss"] == pytest.approx(expected)
    assert saved["validation_loss"] == pytest.approx(expected)
    assert saved["factor"] is None


def test_invalid_validation_mask_fails_before_training(config):
    model = Denoiser()
    values = np.zeros((2, 4, 1))
    with pytest.raises(ValueError, match="both hidden and observed"):
        train_densifier(
            model,
            values,
            values,
            config=config,
            validation_mask=np.zeros(4, dtype=bool),
        )
    assert not model.calls and model.scale.item() == 0


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
        {"schedule": "unknown"},
        {"weight_decay": -1},
        {"weight_decay": np.nan},
        {"lr_decay_fractions": (0.9, 0.75)},
        {"lr_decay_fractions": (np.nan,)},
        {"lr_decay_factor": 0},
        {"validation_steps": "unknown"},
        {"validation_interval": 0},
    ],
)
def test_invalid_training_settings(config, updates):
    with pytest.raises(ValueError):
        replace(config, **updates)


def test_all_step_validation_is_fixed_complete_and_weights_hidden_entries(tmp_path):
    model = Denoiser()
    values = np.broadcast_to(np.array([1.0, 2.0, 4.0])[:, None, None], (3, 5, 1))
    mask = np.zeros_like(values, dtype=bool)
    mask[0, 1] = True
    mask[1, 1:3] = True
    mask[2, 1:4] = True
    calls = []
    config = TrainingConfig(
        epochs=3,
        batch_size=2,
        diffusion_steps=3,
        schedule="quadratic",
        beta_start=0.1,
        beta_end=0.4,
        validation_steps="all",
        validation_interval=2,
    )
    expected_alpha = np.cumprod(1 - np.linspace(np.sqrt(0.1), np.sqrt(0.4), 3) ** 2)

    def objective(prediction, batch):
        np.testing.assert_allclose(
            batch.alpha_bar[:, 0, 0].numpy(), expected_alpha[batch.steps], rtol=1e-6
        )
        if not torch.is_grad_enabled():
            calls.append((batch.steps.clone(), batch.noisy.clone()))
        return (
            batch.clean[batch.mask].square().mean()
            + batch.steps.float().mean()
            + 0 * prediction.mean()
        )

    path = tmp_path / "all_steps.pt"
    train_densifier(
        model,
        values,
        values,
        config=config,
        objective=objective,
        train_mask=mask,
        validation_mask=mask,
        checkpoint=path,
    )
    # Validate on epoch 2 and the final epoch 3: both batches cover every step.
    assert [int(steps[0]) for steps, _ in calls] == [0, 1, 2] * 4
    for first, second in zip(calls[:6], calls[6:], strict=True):
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    state = torch.load(path, weights_only=True)
    assert state["validation_loss"] == pytest.approx(np.square(values[mask]).mean() + 1)
    assert state["config"]["schedule"] == "quadratic"


def test_learning_rate_decays_change_actual_parameter_updates(tmp_path):
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, noisy, steps, **conditions):
            return self.value.expand_as(noisy)

    model = Constant()
    data = np.ones((2, 5, 1))
    path = tmp_path / "decayed.pt"
    train_densifier(
        model,
        data,
        data,
        config=TrainingConfig(
            epochs=4,
            batch_size=2,
            diffusion_steps=2,
            learning_rate=0.1,
            lr_decay_fractions=(0.5, 0.75),
            lr_decay_factor=0.1,
        ),
        objective=lambda prediction, batch: prediction[batch.mask].mean(),
        checkpoint=path,
    )
    assert model.value.item() == pytest.approx(1 - 0.1 - 0.1 - 0.01 - 0.001)
    assert torch.load(path, weights_only=True)["learning_rate"] == pytest.approx(0.001)


def test_weight_decay_regularizes_a_parameter_without_data_gradient(config):
    model = Denoiser()
    data = np.ones((2, 5, 1))
    train_densifier(model, data, data, config=replace(config, weight_decay=0.01))
    # visible_bias has zero data gradient because the objective scores hidden points.
    assert model.visible_bias.item() < 100
