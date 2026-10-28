import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion import polishing_experiment as experiment  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture(scope="module")
def verifier():
    path = Path(__file__).parents[1] / "scripts/verify_polishing_forecast.py"
    spec = importlib.util.spec_from_file_location("verify_polishing_forecast", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("polishing")
    recipe = base / "recipe"
    recipe.mkdir()
    architecture = {
        "n_channels": 1,
        "hidden_channels": 4,
        "residual_layers": 1,
        "attention_heads": 1,
        "diffusion_embedding_dim": 4,
        "time_embedding_dim": 4,
        "feature_embedding_dim": 2,
        "feedforward_dim": 8,
        "dropout": 0.1,
    }
    training = TrainingConfig(
        epochs=3,
        batch_size=2,
        diffusion_steps=2,
        schedule="quadratic",
        beta_end=0.5,
        weight_decay=1e-6,
        lr_decay_fractions=(0.75, 0.9),
        validation_steps="all",
        validation_interval=1,
    )
    write(
        recipe / "plan.json",
        {
            "candidates": {
                "reference_size": {
                    "architecture": architecture,
                    "training": asdict(training),
                }
            }
        },
    )
    write(recipe / "selection.json", {"selected": "reference_size"})
    config = experiment.PolishingConfig(
        seeds=(981, 982),
        namespace=2026102706,
        n_train=4,
        n_validation=3,
        n_test=5,
        epochs=3,
        batch_size=2,
        validation_interval=1,
        validation_epochs=(1, 3),
        samples=2,
        sample_batch_size=4,
        diffusion_steps=2,
        device="cpu",
        threads=1,
        label="smoke",
    )
    output = base / "run"
    previous_threads = torch.get_num_threads()
    monkeypatch = pytest.MonkeyPatch()
    real_generator = experiment.generate_noisy_periodic
    real_train = experiment.train_forecaster
    real_sample = experiment.DDPMForecastSampler.sample
    generated, trained = [], []

    def generator(n, *args, **kwargs):
        if n == config.n_test:
            assert (output / "FROZEN.json").is_file()
            assert (output / "EVALUATION_STARTED.json").is_file()
            for seed in config.seeds:
                assert set(
                    read(output / f"seed_{seed}/training_audit.json")["methods"]
                ) == set(experiment.METHODS)
        generated.append(n)
        return real_generator(n, *args, **kwargs)

    def train(model, densifier, train, validation, **kwargs):
        assert densifier is None and kwargs["factor"] == 1
        assert kwargs["n_augmentations"] == 1
        assert (output / "MEDIATORS_FROZEN.json").is_file()
        assert all(
            (output / f"seed_{seed}/mediator/selected.pt").is_file()
            for seed in config.seeds
        )
        matched = False
        for seed in config.seeds:
            with np.load(output / f"seed_{seed}/train_validation.npz") as original:
                if not np.array_equal(original["train"][:, 32:], train[1]):
                    continue
                matched = True
                np.testing.assert_array_equal(
                    original["validation"][:, 32:], validation[1]
                )
                with np.load(
                    output / f"seed_{seed}/processed_train_validation.npz"
                ) as processed:
                    assert any(
                        np.array_equal(train[0], processed[f"{method}_train_history"])
                        and np.array_equal(
                            validation[0], processed[f"{method}_validation_history"]
                        )
                        for method in experiment.METHODS
                    )
        assert matched
        trained.append(1)
        return real_train(model, densifier, train, validation, **kwargs)

    def sample(self, observed, dense_history, **kwargs):
        np.testing.assert_array_equal(observed, dense_history)
        assert observed.shape[1:] == (32, 1)
        assert kwargs["dt"] == 1 and kwargs["horizon"] == 8
        return real_sample(self, observed, dense_history, **kwargs)

    monkeypatch.setattr(experiment, "generate_noisy_periodic", generator)
    monkeypatch.setattr(experiment, "train_forecaster", train)
    monkeypatch.setattr(experiment.DDPMForecastSampler, "sample", sample)
    try:
        experiment.train_run(output, config, recipe_source=recipe)
        assert config.n_test not in generated
        assert len(trained) == len(config.seeds) * 3
        experiment.evaluate_run(output)
        assert generated.count(config.n_test) == len(config.seeds)
    finally:
        monkeypatch.undo()
        torch.set_num_threads(previous_threads)
    return output, config


@pytest.fixture
def completed_run(reference_run, tmp_path):
    reference, config = reference_run
    output = tmp_path / "copy"
    shutil.copytree(reference, output)
    return output, config


def update_frozen(output, path, verifier):
    frozen = read(output / "FROZEN.json")
    frozen["files"][str(path.relative_to(output))] = verifier.sha256(path)
    write(output / "FROZEN.json", frozen)


def test_independent_verifier_recomputes_complete_tiny_run(
    completed_run, verifier, monkeypatch
):
    output, config = completed_run
    protected = {
        path: path.read_bytes() for path in output.rglob("*") if path.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Verifier attempted diffusion sampling or fitting")

    monkeypatch.setattr(experiment, "train_forecaster", forbidden)
    monkeypatch.setattr(experiment, "train_mediator", forbidden)
    monkeypatch.setattr(experiment.DDPMForecastSampler, "sample", forbidden)
    report = verifier.verify_run(output)
    assert report["status"] == "passed"
    assert not report["forecast_training_or_sampling_performed"]
    assert not report["clean_signal_replayed_or_used"]
    assert report["deterministic_mediator_inference_recomputed"]
    assert report == read(output / "verification.json")
    assert set(report["seeds"]) == set(map(str, config.seeds))
    for seed in config.seeds:
        assert report["seeds"][str(seed)]["measured_updates_per_method"] == 6
    assert all(path.read_bytes() == contents for path, contents in protected.items())


@pytest.mark.parametrize(
    "marker",
    [
        "MEDIATORS_FROZEN.json",
        "FROZEN.json",
        "TRAINED.json",
        "EVALUATION_STARTED.json",
        "EVALUATED.json",
    ],
)
def test_completion_markers_required_before_observed_generator_replay(
    completed_run, verifier, monkeypatch, marker
):
    output, _ = completed_run
    (output / marker).unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete run caused generator replay")

    monkeypatch.setattr(verifier, "generate_noisy_periodic", forbidden)
    with pytest.raises(FileNotFoundError, match=marker):
        verifier.verify_run(output)


def test_omitted_candidate_cannot_escape_frozen_inventory(completed_run, verifier):
    output, _ = completed_run
    frozen = read(output / "FROZEN.json")
    missing = next(name for name in frozen["files"] if "_validation_epoch_" in name)
    del frozen["files"][missing]
    write(output / "FROZEN.json", frozen)
    with pytest.raises(ValueError, match="training-artifact inventory"):
        verifier.verify_run(output)


@pytest.mark.parametrize(
    "change",
    [
        "processed_history",
        "future",
        "updates",
        "learning_rate",
        "chronology",
        "candidate_score",
    ],
)
def test_semantic_training_corruption_rejected_even_when_file_hash_updated(
    completed_run, verifier, change
):
    output, config = completed_run
    folder = output / f"seed_{config.seeds[0]}"
    if change in ("processed_history", "future"):
        path = folder / "processed_train_validation.npz"
        with np.load(path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        key = (
            "learned_train_history" if change == "processed_history" else "train_future"
        )
        arrays[key][0, 0, 0] += 1
        np.savez_compressed(path, **arrays)
        expected = (
            "independent preprocessing"
            if change == "processed_history"
            else "noisy future"
        )
    elif change in ("updates", "chronology"):
        path = folder / "training_audit.json"
        audit = read(path)
        if change == "updates":
            audit["methods"]["raw"]["measured_optimizer_updates"] += 1
            expected = "optimization budget"
        else:
            audit["methods"]["raw"]["started_utc"] = "2000-01-01T00:00:00+00:00"
            expected = "chronology"
        write(path, audit)
    elif change == "learning_rate":
        path = folder / "raw_training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["learning_rate"] *= 2
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        expected = "learning-rate"
    else:
        path = folder / "raw_validation_epoch_0001.npz"
        with np.load(path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["samples"][0, 0, 0, 0] += 100
        np.savez_compressed(path, **arrays)
        expected = "candidate metric recomputation"
    update_frozen(output, path, verifier)
    with pytest.raises(ValueError, match=expected):
        verifier.verify_run(output)


@pytest.mark.parametrize(
    "change", ["processed_history", "future", "side_channel", "test_stream"]
)
def test_final_original_target_history_and_input_audits(
    completed_run, verifier, change
):
    output, config = completed_run
    folder = output / f"seed_{config.seeds[0]}"
    if change == "side_channel":
        path = folder / "inference_audit.json"
        audit = read(path)
        audit["learned"]["observed_argument_sha256"] = "0" * 64
        write(path, audit)
        expected = "side channel"
    else:
        path = folder / (
            "test.npz" if change == "test_stream" else "learned_samples.npz"
        )
        with np.load(path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        key = "processed_history" if change == "processed_history" else "target"
        arrays[key][0, 0, 0] += 1
        np.savez_compressed(path, **arrays)
        expected = {
            "processed_history": "processed history",
            "future": "unchanged noisy targets",
            "test_stream": "fresh noisy test target",
        }[change]
    with pytest.raises(ValueError, match=expected):
        verifier.verify_run(output)


def test_run_refuses_existing_training_or_evaluation_artifacts(reference_run):
    output, config = reference_run
    with pytest.raises(FileExistsError):
        experiment.train_run(output, config)
    with pytest.raises(FileExistsError, match="already started"):
        experiment.evaluate_run(output)


def test_declared_scientific_budget_and_nonuniform_opportunities():
    config = experiment.PolishingConfig()
    assert config.epochs * (config.n_train // config.batch_size) == 4800
    assert config.validation_epochs == (10, 50, 100, 200, 400, 800, 1200)
    assert config.n_test == 2048 and config.samples == 64
    with pytest.raises(ValueError, match="budget changed"):
        experiment.PolishingConfig(n_train=512)
    with pytest.raises(ValueError, match="independent namespace"):
        experiment.PolishingConfig(label="smoke")
