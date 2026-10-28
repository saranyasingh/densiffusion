from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.training import (  # noqa: E402
    TrainingConfig,
    train_densifier,
    train_forecaster,
)


class ScalarDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(1.0))
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, noisy, steps, **kwargs):
        return torch.ones_like(noisy) * self.value


@pytest.fixture
def config():
    return TrainingConfig(epochs=3, batch_size=2, diffusion_steps=2, learning_rate=0.1)


def squared_prediction(prediction, batch):
    return prediction.square().mean()


def fit(stage, model, config, **kwargs):
    data = np.ones((2, 5, 1))
    if stage == "densifier":
        return train_densifier(model, data, data, config=config, **kwargs)
    pair = data[:, :3], data[:, 3:]
    return train_forecaster(model, None, pair, pair, factor=1, config=config, **kwargs)


@pytest.mark.parametrize("stage", ["densifier", "forecaster"])
def test_custom_score_selects_checkpoint_instead_of_denoising_loss(
    stage, config, tmp_path
):
    model = ScalarDenoiser()
    states = {}
    records = []

    def forecast_crps(current, epoch):
        assert current is model
        assert not torch.is_grad_enabled()
        assert not any(module.training for module in current.modules())
        states[epoch] = current.value.detach().clone()
        return float((epoch - 2) ** 2)

    checkpoint = tmp_path / "selected.pt"
    fit(
        stage,
        model,
        config,
        objective=squared_prediction,
        validation_score=forecast_crps,
        epoch_callback=records.append,
        checkpoint=checkpoint,
    )
    assert records[2]["validation_loss"] < records[1]["validation_loss"]
    torch.testing.assert_close(model.value, states[2], rtol=0, atol=0)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["epoch"] == 2
    assert saved["selection_score"] == 0
    assert saved["selection_metric"].endswith("forecast_crps")
    assert saved["validation_loss"] == records[1]["validation_loss"]
    assert not model.training


def test_callbacks_include_every_epoch_and_only_validated_metrics(config):
    model = ScalarDenoiser()
    records = []
    scored_epochs = []

    def score(current, epoch):
        scored_epochs.append(epoch)
        return -epoch

    fit(
        "densifier",
        model,
        replace(config, epochs=5, validation_interval=2),
        validation_score=score,
        epoch_callback=records.append,
    )
    assert [row["epoch"] for row in records] == [1, 2, 3, 4, 5]
    assert scored_epochs == [2, 4, 5]
    for row in records:
        assert {"epoch", "train_loss", "learning_rate"} <= row.keys()
        if row["epoch"] in scored_epochs:
            assert row["selection_score"] == -row["epoch"]
            assert {"validation_loss", "selection_metric"} <= row.keys()
        else:
            assert "validation_loss" not in row
            assert "selection_score" not in row


DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "score_options",
    [{}, {"validation_score_interval": 4}, {"validation_score_epochs": (2,)}],
)
def test_monitoring_does_not_perturb_training_rng_or_modes(
    config, device, score_options
):
    class StochasticDenoiser(ScalarDenoiser):
        def forward(self, noisy, steps, **kwargs):
            assert self.training == torch.is_grad_enabled()
            return self.value * self.dropout(noisy)

    runs = []
    for perturb in (False, True):
        model = StochasticDenoiser()
        records = []

        def score(current, epoch, perturb=perturb):
            assert not current.training and not current.dropout.training
            if perturb:
                torch.manual_seed(999)
                torch.randn(51)
                torch.randn(31, device=device)
                current.train()
                current.dropout.eval()
            return -epoch

        def callback(record, model=model, records=records, perturb=perturb):
            # Validation leaves the model in eval; training-only epochs do not.
            expected_training = "validation_loss" not in record
            assert model.training == expected_training
            assert model.dropout.training == expected_training
            records.append(record.copy())
            if perturb:
                torch.manual_seed(47)
                torch.randn(19, device=device)
                model.train(not expected_training)

        fit(
            "densifier",
            model,
            replace(config, epochs=5, validation_interval=2, device=device),
            validation_score=score,
            **score_options,
            epoch_callback=callback,
            seed=71,
        )
        runs.append((model.value.detach().cpu(), records, torch.get_rng_state()))
    torch.testing.assert_close(runs[0][0], runs[1][0], rtol=0, atol=0)
    assert runs[0][1] == runs[1][1]
    assert torch.equal(runs[0][2], runs[1][2])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_validation_score_is_rejected_without_checkpoint(
    config, tmp_path, score
):
    checkpoint = tmp_path / "invalid.pt"
    with pytest.raises(FloatingPointError, match="Non-finite validation_score"):
        fit(
            "densifier",
            ScalarDenoiser(),
            config,
            validation_score=lambda model, epoch: score,
            checkpoint=checkpoint,
        )
    assert not checkpoint.exists()


def test_score_exception_restores_mode_and_rng(config):
    model = ScalarDenoiser()
    before = {}

    def fail(current, epoch):
        before["rng"] = torch.get_rng_state()
        current.train()
        torch.randn(31)
        raise RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        fit("densifier", model, config, validation_score=fail)
    assert not any(module.training for module in model.modules())
    assert torch.equal(torch.get_rng_state(), before["rng"])


def test_default_checkpoint_still_uses_validation_loss(config, tmp_path):
    checkpoint = tmp_path / "default.pt"
    fit(
        "densifier",
        ScalarDenoiser(),
        config,
        objective=squared_prediction,
        checkpoint=checkpoint,
    )
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["epoch"] == config.epochs
    assert saved["selection_score"] == saved["validation_loss"]
    assert saved["selection_metric"] == "validation_loss"


@pytest.mark.parametrize("stage", ["densifier", "forecaster"])
def test_sparse_custom_selection_keeps_all_step_validation(stage, config, tmp_path):
    model = ScalarDenoiser()
    states = {}
    records = []
    validation_steps = []
    checkpoint = tmp_path / "sparse.pt"

    def objective(prediction, batch):
        if not torch.is_grad_enabled():
            validation_steps.append(batch.steps.tolist())
        return squared_prediction(prediction, batch)

    def score(current, epoch):
        states[epoch] = current.value.detach().clone()
        return 0.0 if epoch == 4 else 1.0

    def callback(record):
        records.append(record)
        if record["epoch"] < 4:
            assert not checkpoint.exists()
        else:
            # Better denoising losses at skipped/final epochs cannot replace it.
            assert torch.load(checkpoint, weights_only=True)["epoch"] == 4

    fit(
        stage,
        model,
        replace(config, epochs=7, validation_interval=2, validation_steps="all"),
        objective=objective,
        validation_score=score,
        validation_score_interval=4,
        epoch_callback=callback,
        checkpoint=checkpoint,
    )
    assert list(states) == [4, 7]
    assert [r["epoch"] for r in records if "validation_loss" in r] == [2, 4, 6, 7]
    assert [r["epoch"] for r in records if "selection_score" in r] == [4, 7]
    assert validation_steps == [[0, 0], [1, 1]] * 4
    assert records[5]["validation_loss"] < records[3]["validation_loss"]
    torch.testing.assert_close(model.value, states[4], rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["densifier", "forecaster"])
@pytest.mark.parametrize("interval", [0, -1, True, 1.5, 3])
def test_invalid_score_interval_is_rejected(stage, config, interval):
    with pytest.raises(ValueError, match="validation_score_interval"):
        fit(
            stage,
            ScalarDenoiser(),
            replace(config, validation_interval=2),
            validation_score=lambda model, epoch: 0.0,
            validation_score_interval=interval,
        )


def test_score_interval_requires_custom_score(config):
    with pytest.raises(ValueError, match="requires validation_score"):
        fit("densifier", ScalarDenoiser(), config, validation_score_interval=2)


def test_invalid_interval_rejected_before_forecaster_augmentation(config):
    class NeverSample:
        def sample(self, *args, **kwargs):
            raise AssertionError("Invalid interval must fail before sampling")

    data = np.ones((2, 5, 1))
    pair = data[:, :3], data[:, 3:]
    with pytest.raises(ValueError, match="multiple of validation_interval"):
        train_forecaster(
            ScalarDenoiser(),
            NeverSample(),
            pair,
            pair,
            config=replace(config, validation_interval=2),
            validation_score=lambda model, epoch: 0.0,
            validation_score_interval=3,
        )


@pytest.mark.parametrize("stage", ["densifier", "forecaster"])
def test_nonuniform_score_epochs_keep_noise_validation_and_checkpoint_selection(
    stage, config, tmp_path
):
    model = ScalarDenoiser()
    states, records, validation_steps = {}, [], []
    checkpoint = tmp_path / "nonuniform.pt"

    def objective(prediction, batch):
        if not torch.is_grad_enabled():
            validation_steps.append(batch.steps.tolist())
        return squared_prediction(prediction, batch)

    def score(current, epoch):
        states[epoch] = current.value.detach().clone()
        return 0.0 if epoch == 6 else 1.0

    def callback(record):
        records.append(record)
        epoch = record["epoch"]
        if epoch < 2:
            assert not checkpoint.exists()
        else:
            expected = 2 if epoch < 6 else 6
            assert torch.load(checkpoint, weights_only=True)["epoch"] == expected

    fit(
        stage,
        model,
        replace(config, epochs=11, validation_interval=2, validation_steps="all"),
        objective=objective,
        validation_score=score,
        validation_score_epochs=(2, 6, 8),
        epoch_callback=callback,
        checkpoint=checkpoint,
    )
    assert list(states) == [2, 6, 8, 11]
    assert [row["epoch"] for row in records] == list(range(1, 12))
    assert [row["epoch"] for row in records if "validation_loss" in row] == [
        2,
        4,
        6,
        8,
        10,
        11,
    ]
    assert [row["epoch"] for row in records if "selection_score" in row] == [
        2,
        6,
        8,
        11,
    ]
    assert validation_steps == [[0, 0], [1, 1]] * 6
    assert records[9]["validation_loss"] < records[5]["validation_loss"]
    torch.testing.assert_close(model.value, states[6], rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["densifier", "forecaster"])
@pytest.mark.parametrize("epochs, expected", [((), [6]), ((2, 6), [2, 6])])
def test_explicit_score_schedule_always_scores_final_once(
    stage, config, epochs, expected
):
    scored = []

    def score(model, epoch):
        scored.append(epoch)
        return -epoch

    fit(
        stage,
        ScalarDenoiser(),
        replace(config, epochs=6, validation_interval=2),
        validation_score=score,
        validation_score_epochs=epochs,
    )
    assert scored == expected


@pytest.mark.parametrize("stage", ["densifier", "forecaster"])
@pytest.mark.parametrize(
    "epochs", [(0,), (-2,), (True,), (2.0,), (2, 2), (4, 2), (8,), (3,), 2, "2", {2, 4}]
)
def test_invalid_explicit_score_schedule_is_rejected(stage, config, epochs):
    with pytest.raises(ValueError, match="validation_score_epochs"):
        fit(
            stage,
            ScalarDenoiser(),
            replace(config, epochs=6, validation_interval=2),
            validation_score=lambda model, epoch: 0.0,
            validation_score_epochs=epochs,
        )


@pytest.mark.parametrize("stage", ["densifier", "forecaster"])
def test_explicit_score_schedule_requires_score_and_excludes_interval(stage, config):
    with pytest.raises(ValueError, match="requires validation_score"):
        fit(stage, ScalarDenoiser(), config, validation_score_epochs=(2,))
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit(
            stage,
            ScalarDenoiser(),
            config,
            validation_score=lambda model, epoch: 0.0,
            validation_score_interval=2,
            validation_score_epochs=(2,),
        )


def test_invalid_explicit_schedule_rejected_before_forecaster_augmentation(config):
    class NeverSample:
        def sample(self, *args, **kwargs):
            raise AssertionError("Invalid schedule must fail before sampling")

    data = np.ones((2, 5, 1))
    pair = data[:, :3], data[:, 3:]
    with pytest.raises(ValueError, match="multiples of validation_interval"):
        train_forecaster(
            ScalarDenoiser(),
            NeverSample(),
            pair,
            pair,
            config=replace(config, validation_interval=2),
            validation_score=lambda model, epoch: 0.0,
            validation_score_epochs=(3,),
        )


def test_explicit_schedule_is_copied_before_callbacks(config):
    epochs, scored = [2, 6], []

    def score(model, epoch):
        epochs.clear()
        scored.append(epoch)
        return -epoch

    fit(
        "densifier",
        ScalarDenoiser(),
        replace(config, epochs=7, validation_interval=2),
        validation_score=score,
        validation_score_epochs=epochs,
    )
    assert scored == [2, 6, 7]
