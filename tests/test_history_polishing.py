import hashlib
import json

import numpy as np
import pytest
import torch

from densiffusion.data import GenerationConfig, generate
from densiffusion.generators import noisy_periodic
from densiffusion.history_polishing import (
    OFFSETS,
    BlindSpotMediator,
    load_mediator,
    median_history,
    neighbor_features,
    polish_history,
    raw_history,
    train_mediator,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _replay(n, steps, seed):
    rng = np.random.default_rng(seed)
    amplitude = rng.uniform(0.7, 1.3, (n, 1))
    period = rng.uniform(24, 40, (n, 1))
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    theta = 2 * np.pi * np.arange(steps)[None] / period + phase
    signal = amplitude * (np.sin(theta) + 0.2 * np.sin(2 * theta + 0.4))
    noise = rng.normal(0, 0.15, (n, steps))
    active = rng.binomial(1, 0.1, (n, steps))
    outliers = rng.normal(0, 2, (n, steps))
    return signal, noise, active, outliers, amplitude, period, phase


def test_generator_formula_replay_and_observations_only():
    signal, noise, active, outliers, *_ = _replay(11, 40, 172)
    actual = noisy_periodic.generate_noisy_periodic(11, rng=172)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (11, 40, 1)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual[:, :, 0], signal + noise + active * outliers)
    np.testing.assert_array_equal(
        actual,
        noisy_periodic.generate_noisy_periodic(11, rng=np.random.default_rng(172)),
    )
    assert not hasattr(noisy_periodic, "evaluate_signal")
    config = GenerationConfig(
        generator="densiffusion.generators.noisy_periodic:generate",
        n_series=11,
        n_steps=40,
        seed=172,
    )
    np.testing.assert_array_equal(generate(config), actual)
    with pytest.raises(ValueError, match="integer times"):
        noisy_periodic.generate(1, dt=0.5, rng=0)


def test_generator_independence_distributions_and_private_rng():
    signal, noise, active, outliers, amplitude, period, phase = _replay(30_000, 4, 318)
    actual = noisy_periodic.generate_noisy_periodic(30_000, 4, rng=318)[:, :, 0]
    np.testing.assert_array_equal(actual, signal + noise + active * outliers)
    standardized = np.column_stack(
        [
            (amplitude[:, 0] - 0.7) / 0.6,
            (period[:, 0] - 24) / 16,
            phase[:, 0] / (2 * np.pi),
        ]
    )
    np.testing.assert_allclose(standardized.mean(axis=0), 0.5, atol=0.006)
    np.testing.assert_allclose(standardized.var(axis=0), 1 / 12, atol=0.002)
    np.testing.assert_allclose(np.corrcoef(standardized.T), np.eye(3), atol=0.025)
    assert abs(noise.std() - 0.15) < 0.002
    assert abs(active.mean() - 0.1) < 0.004
    assert abs(outliers.std() - 2) < 0.02
    np.testing.assert_allclose(np.corrcoef((actual - signal).T), np.eye(4), atol=0.025)
    private = np.random.default_rng(9)
    first = noisy_periodic.generate_noisy_periodic(2, rng=private)
    assert not np.array_equal(
        first, noisy_periodic.generate_noisy_periodic(2, rng=private)
    )
    before = np.random.get_state()
    noisy_periodic.generate_noisy_periodic(2, rng=1)
    after = np.random.get_state()
    for left, right in zip(before, after, strict=True):
        np.testing.assert_equal(left, right)


@pytest.mark.parametrize(
    "n,steps", [(0, 40), (True, 40), (1, 0), (1, False), (1.0, 40)]
)
def test_generator_rejects_invalid_counts(n, steps):
    with pytest.raises(ValueError):
        noisy_periodic.generate_noisy_periodic(n, steps, rng=1)


def test_neighbor_fixed_slots_and_zero_availability_at_every_boundary():
    history = (np.arange(64).reshape(2, 32, 1) + 10).astype(float)
    features = neighbor_features(history)
    assert features.shape == (2, 32, 12)
    for center in range(32):
        for slot, offset in enumerate(OFFSETS):
            available = 0 <= center + offset < 32
            np.testing.assert_array_equal(
                features[:, center, slot + 6], float(available)
            )
            expected = history[:, center + offset, 0] if available else np.zeros(2)
            np.testing.assert_array_equal(features[:, center, slot], expected)
    zero_features = neighbor_features(np.zeros((1, 32, 1)))
    assert np.all(zero_features[:, :, :6] == 0)
    assert np.all(zero_features[:, 3:-3, 6:] == 1)


def test_every_center_excluded_from_features_and_model_predictions():
    history = np.random.default_rng(1).normal(size=(2, 32, 1))
    original = history.copy()
    features = neighbor_features(history)
    model = BlindSpotMediator()
    predictions = polish_history(model, history)
    for center in range(32):
        changed = history.copy()
        changed[:, center] += 10_000
        np.testing.assert_array_equal(
            neighbor_features(changed)[:, center], features[:, center]
        )
        np.testing.assert_array_equal(
            polish_history(model, changed)[:, center], predictions[:, center]
        )
    np.testing.assert_array_equal(history, original)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    np.testing.assert_array_equal(polish_history(model, history), 0)


def test_median_edges_use_history_padding_and_no_future():
    full = np.random.default_rng(4).normal(size=(2, 40, 1))
    history = full[:, :32]
    expected = np.empty_like(history)
    for center in range(32):
        indices = np.clip(np.arange(center - 3, center + 4), 0, 31)
        expected[:, center] = np.median(history[:, indices], axis=1)
    actual = median_history(history)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, -1], history[:, -1])
    full[:, 32:] = 1e15
    np.testing.assert_array_equal(median_history(full[:, :32]), actual)
    with pytest.raises(ValueError):
        median_history(full)
    copied = raw_history(history)
    np.testing.assert_array_equal(copied, history)
    assert not np.shares_memory(copied, history)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 40, 1)),
        np.ones((32, 1)),
        np.ones((2, 32, 2)),
        np.full((2, 32, 1), np.nan),
    ],
)
def test_polishers_reject_wrong_scope_or_nonfinite_values(bad):
    for function in (raw_history, median_history, neighbor_features):
        with pytest.raises(ValueError):
            function(bad)


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    train = noisy_periodic.generate_noisy_periodic(8, rng=7)[:, :32].copy()
    validation = noisy_periodic.generate_noisy_periodic(4, rng=8)[:, :32].copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting must not call any generator")

    monkeypatch.setattr(noisy_periodic, "generate_noisy_periodic", forbidden)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    output = tmp_path / "fit"
    model, summary = train_mediator(train, validation, output, seed=101, smoke=True)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    for left, right in zip(np.random.get_state(), numpy_before, strict=True):
        np.testing.assert_equal(left, right)
    return train, validation, output, model, summary


def test_fit_uses_noisy_history_targets_and_saves_exact_selected_predictions(fitted):
    train, validation, output, model, summary = fitted
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert summary["validation_centers"] == 4 * 32
    assert summary["center_excluded_from_features"]
    assert not summary["clean_values_or_future_observations_accessed"]
    assert len(summary["candidates"]) == 4
    assert (
        len({candidate["initial_sha256"] for candidate in summary["candidates"]}) == 4
    )
    for candidate in summary["candidates"]:
        directory = output / candidate["name"]
        config = json.loads((directory / "config.json").read_text())
        assert config["score_epochs"] == [1, 5]
        assert config["epochs"] == 5
        assert config["offsets"] == [-3, -2, -1, 1, 2, 3]
        assert config["batch_size"] == 256
        assert config["learning_rate"] == 0.001
        assert config["weight_decay"] == 0.0001
        log = [
            json.loads(line)
            for line in (directory / "training.jsonl").read_text().splitlines()
        ]
        assert [row["epoch"] for row in log] == [1, 2, 3, 4, 5]
        assert log[-1]["optimizer_updates"] == 5
        selection = json.loads((directory / "validation.json").read_text())
        winner = min(selection, key=lambda row: row["validation_mse"])
        assert candidate["selected_epoch"] == winner["epoch"]
        candidate_model = load_mediator(directory / "best.pt")
        predictions = np.load(directory / "best_validation_predictions.npy")
        np.testing.assert_array_equal(
            polish_history(candidate_model, validation), predictions
        )
        assert candidate["validation_mse"] == np.mean((predictions - validation) ** 2)
    selected = min(summary["candidates"], key=lambda row: row["validation_mse"])
    assert summary["selected"] == selected["name"]
    assert (output / "selected.pt").read_bytes() == (
        output / selected["name"] / "best.pt"
    ).read_bytes()
    assert (
        summary["selected_checkpoint_sha256"]
        == hashlib.sha256((output / "selected.pt").read_bytes()).hexdigest()
    )
    with np.load(output / "selected_predictions.npz") as saved:
        np.testing.assert_array_equal(
            saved["train_predictions"], polish_history(model, train)
        )
        np.testing.assert_array_equal(
            saved["validation_predictions"], polish_history(model, validation)
        )
        assert (
            np.mean((saved["validation_predictions"] - validation) ** 2)
            == summary["validation_mse"]
        )


def test_fit_reproducibility_and_no_overwrite(fitted, tmp_path):
    train, validation, output, model, summary = fitted
    second, second_summary = train_mediator(
        train, validation, tmp_path / "replay", 101, smoke=True
    )
    assert second_summary["selected"] == summary["selected"]
    assert second_summary["validation_mse"] == summary["validation_mse"]
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[key])
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        train_mediator(train, validation, output, 102, smoke=True)
    assert (output / "summary.json").read_bytes() == before


def test_fit_rejects_full_trajectory_before_creating_output(tmp_path):
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        train_mediator(
            np.zeros((2, 40, 1)), np.zeros((2, 32, 1)), output, 1, smoke=True
        )
    assert not output.exists()
