import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.experiments import run_forecast_experiment  # noqa: E402
from densiffusion.metrics import evaluate_forecast  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.objectives import reconstruction_mse  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402


class Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, noisy, steps, **kwargs):
        return self.bias.expand_as(noisy)


class LinearDensifier:
    def __init__(self):
        self.calls = []

    def sample(self, observed, *, factor, n_samples, dt, rng):
        assert not torch.is_grad_enabled()
        self.calls.append(observed.copy())
        positions = np.arange((observed.shape[1] - 1) * factor + 1) / factor
        dense = np.stack(
            [
                np.stack(
                    [
                        np.interp(positions, np.arange(len(path)), channel)
                        for channel in path.T
                    ],
                    axis=-1,
                )
                for path in observed
            ]
        )
        return np.repeat(dense[:, None], n_samples, axis=1)


@pytest.mark.parametrize("factor", [1, 2, 3])
def test_runner_scores_original_grid_only_and_weights_partial_batches(factor, tmp_path):
    model = ImputationModel(Denoiser())
    densifier = LinearDensifier()
    train = (np.ones((3, 3, 1)), np.ones((3, 2, 1)))
    validation = (np.ones((2, 3, 1)), np.ones((2, 2, 1)))
    history = np.broadcast_to(np.arange(1, 6)[:, None, None], (5, 3, 1)).copy()
    future = np.arange(10).reshape(5, 2, 1) + 100
    recorded = []
    original = (history.copy(), future.copy())

    class RecordingForecaster:
        def sample(self, observed, dense_history, *, horizon, n_samples, dt, rng):
            assert not model.training and not torch.is_grad_enabled()
            np.testing.assert_array_equal(dense_history[:, ::factor], observed)
            assert horizon == 2 * factor and dt == 0.3 / factor
            draws = rng.normal(size=(len(observed), n_samples, horizon, 1))
            draws += observed[:, None, -1:, :]
            recorded.append(draws[:, :, factor - 1 :: factor].reshape(-1, 6, 2, 1))
            return draws

    config = TrainingConfig(epochs=1, batch_size=2, diffusion_steps=2)
    result = run_forecast_experiment(
        model,
        densifier,
        train,
        validation,
        (history, future),
        sampler_factory=lambda trained, cfg: RecordingForecaster(),
        config=config,
        objective=reconstruction_mse,
        factor=factor,
        dt=0.3,
        n_histories=2,
        n_futures=3,
        checkpoint=tmp_path / "forecast.pt",
    )
    assert result == pytest.approx(evaluate_forecast(np.concatenate(recorded), future))
    assert [len(part) for part in recorded] == [2, 2, 1]
    # Test future values never enter any density call (train/validation are 1).
    assert all(np.max(call) <= 5 for call in densifier.calls)
    for actual, expected in zip((history, future), original, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert not model.training


@pytest.mark.parametrize(
    "test",
    [
        (np.zeros((2, 2, 1)), np.zeros((2, 2, 1))),
        (np.zeros((2, 3, 1)), np.zeros((1, 2, 1))),
        (np.zeros((2, 3, 1)), np.full((2, 2, 1), np.nan)),
        np.zeros((2, 3, 1)),
    ],
)
def test_runner_invalid_test_split_fails_before_training(test):
    model = ImputationModel(Denoiser())
    initial = model.denoiser.bias.clone()
    densifier = LinearDensifier()
    valid = (np.zeros((2, 3, 1)), np.zeros((2, 2, 1)))
    with pytest.raises(ValueError):
        run_forecast_experiment(
            model, densifier, valid, valid, test, sampler_factory=lambda *args: None
        )
    assert not densifier.calls
    torch.testing.assert_close(model.denoiser.bias, initial)


def test_forecast_metrics_mean_error_and_central_interval():
    samples = np.array([0, 10, 20], dtype=float).reshape(1, 3, 1, 1)
    target = np.full((1, 1, 1), 12.0)
    result = evaluate_forecast(samples, target)
    assert result == pytest.approx(
        {"crps": 22 / 3 - 40 / 9, "mse": 4, "coverage_90": 1, "interval_width_90": 18}
    )
    assert evaluate_forecast(samples, target + 10)["coverage_90"] == 0
    point = evaluate_forecast(target[:, None] + 2, target)
    assert point == {"crps": 2, "mse": 4, "coverage_90": 0, "interval_width_90": 0}
