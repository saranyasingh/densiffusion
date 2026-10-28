import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion import waveform_diagnostics as diagnostics  # noqa: E402
from densiffusion.architectures.csdi import CSDIDenoiser  # noqa: E402
from densiffusion.generators.waveform import (  # noqa: E402
    draw_waveform_parameters,
    evaluate_waveforms,
    generate_waveforms,
)
from densiffusion.metrics import evaluate_forecast  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402
from densiffusion.waveform_experiment import seed_plan  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen_run(tmp_path):
    seed = 17
    folder = tmp_path / f"seed_{seed}"
    folder.mkdir()
    config = {
        "seeds": [seed],
        "n_test": 5,
        "context_length": 25,
        "horizon": 8,
        "dt": 0.1,
        "device": "cpu",
        "threads": 1,
        "sample_batch_size": 16,
    }
    architecture = {
        "n_channels": 1,
        "hidden_channels": 8,
        "residual_layers": 1,
        "attention_heads": 2,
        "diffusion_embedding_dim": 8,
        "time_embedding_dim": 8,
        "feature_embedding_dim": 4,
        "feedforward_dim": 8,
        "dropout": 0.0,
    }
    seeds = seed_plan(seed)
    write_json(
        tmp_path / "config.json",
        {"experiment": config, "architecture": architecture, "seeds": {seed: seeds}},
    )
    model = ImputationModel(CSDIDenoiser(**architecture))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(TrainingConfig(diffusion_steps=2)),
        },
        folder / "densifier.pt",
    )
    values = generate_waveforms(5, rng=seeds["test_data"])
    np.savez_compressed(
        folder / "samples.npz",
        history=values[:, :25],
        target=values[:, 25:],
        trajectory_ids=np.arange(100, 105),
    )
    write_json(
        tmp_path / "FROZEN.json",
        {
            "config_sha256": hashlib.sha256(
                (tmp_path / "config.json").read_bytes()
            ).hexdigest(),
            "checkpoint_sha256": {
                "seed_17/densifier.pt": hashlib.sha256(
                    (folder / "densifier.pt").read_bytes()
                ).hexdigest()
            },
        },
    )
    write_json(tmp_path / "EVALUATED.json", {"finished_utc": "test"})
    return tmp_path


@pytest.mark.parametrize("available", [(), ("FROZEN.json",), ("EVALUATED.json",)])
def test_both_completion_markers_required_before_truth_access(
    tmp_path, monkeypatch, available
):
    for name in available:
        write_json(tmp_path / name, {})

    def forbidden(*args, **kwargs):
        raise AssertionError("Truth accessed before both completion markers")

    monkeypatch.setattr(diagnostics, "draw_waveform_parameters", forbidden)
    monkeypatch.setattr(diagnostics, "evaluate_waveforms", forbidden)
    with pytest.raises(FileNotFoundError, match="Diagnostics require completed"):
        diagnostics.run_diagnostics(tmp_path)
    assert not (tmp_path / "diagnostics").exists()


@pytest.mark.parametrize("changed", ["config.json", "seed_17/densifier.pt"])
def test_frozen_inputs_are_verified_before_truth_generation(frozen_run, changed):
    path = frozen_run / changed
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after freezing"):
        diagnostics.run_diagnostics(frozen_run)
    assert not (frozen_run / "diagnostics").exists()


def test_saved_diagnostics_recompute_and_leave_frozen_artifacts_unchanged(frozen_run):
    protected = {
        path: path.read_bytes() for path in frozen_run.rglob("*") if path.is_file()
    }
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = diagnostics.run_diagnostics(frozen_run, n_examples=3, n_samples=4)
    finally:
        torch.set_num_threads(previous_threads)
    assert result["created_after_freeze_and_test_evaluation"]
    assert result["n_examples_per_seed"] == 3
    assert result["total_seconds"] > 0
    entry = result["seeds"]["seed_17"]
    assert entry["original_test_rng_reproduced_exactly"]
    with np.load(frozen_run / "seed_17" / "samples.npz") as saved:
        original_history = saved["history"][:3]
    parameters = draw_waveform_parameters(
        5, rng=np.random.default_rng(seed_plan(17)["test_data"])
    )
    signal = evaluate_waveforms(parameters, np.arange(49) * 0.1 / 2)[:3]
    for name, hidden_count, dt in (
        ("trained_resolution", 12, 0.2),
        ("transferred_resolution", 24, 0.1),
    ):
        scores = entry["tasks"][name]
        assert scores["n_hidden"] == hidden_count
        assert scores["input_dt"] == dt
        assert scores["observations_preserved"]
        with np.load(frozen_run / "diagnostics" / "seed_17" / f"{name}.npz") as data:
            np.testing.assert_array_equal(data["trajectory_ids"], [100, 101, 102])
            if name == "trained_resolution":
                np.testing.assert_array_equal(data["noisy_truth"], original_history)
                np.testing.assert_array_equal(
                    data["observed"], original_history[:, ::2]
                )
            else:
                np.testing.assert_array_equal(
                    data["noisy_truth"][:, ::2], original_history
                )
                np.testing.assert_array_equal(data["noiseless_signal_truth"], signal)
                assert not np.array_equal(data["noisy_truth"][:, 1::2], signal[:, 1::2])
            for method in ("learned", "linear"):
                samples = data[f"{method}_hidden_samples"]
                assert samples.shape == (3, 4, hidden_count, 1)
                assert scores["timings"][method]["inference_seconds"] >= 0
                for label, key in (
                    ("noisy_observations", "noisy_truth"),
                    ("noiseless_signal", "noiseless_signal_truth"),
                ):
                    assert scores["scores"][method][label] == evaluate_forecast(
                        samples, data[key][:, 1::2]
                    )
    assert all(path.read_bytes() == contents for path, contents in protected.items())
    assert (
        json.loads((frozen_run / "diagnostics" / "scores.json").read_text()) == result
    )
    with pytest.raises(FileExistsError, match="Diagnostics already exist"):
        diagnostics.run_diagnostics(frozen_run)


def test_replayed_test_observations_must_match_before_fine_truth(
    frozen_run, monkeypatch
):
    path = frozen_run / "seed_17" / "samples.npz"
    with np.load(path) as archive:
        data = dict(archive)
    data["history"][0, 0, 0] += 1
    np.savez_compressed(path, **data)
    evaluated_sizes = []
    original_evaluate = diagnostics.evaluate_waveforms

    def recorded_evaluate(parameters, times):
        evaluated_sizes.append(len(times))
        return original_evaluate(parameters, times)

    monkeypatch.setattr(diagnostics, "evaluate_waveforms", recorded_evaluate)
    with pytest.raises(ValueError, match="do not match the test-data RNG replay"):
        diagnostics.run_diagnostics(frozen_run)
    # The original 33-point grid may be regenerated to verify identity; no finer grid.
    assert evaluated_sizes == [33]


def test_diagnostic_sampling_checks_observation_preservation():
    class BrokenSampler:
        def sample(self, observed, **kwargs):
            return np.zeros((len(observed), kwargs["n_samples"], 5, 1))

    with pytest.raises(AssertionError, match="changed observed values"):
        diagnostics._score_task(
            np.ones((2, 3, 1)),
            np.ones((2, 5, 1)),
            np.ones((2, 5, 1)),
            BrokenSampler(),
            dt=0.2,
            seed=7,
            n_samples=4,
            device="cpu",
        )
