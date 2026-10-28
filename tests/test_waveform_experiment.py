import hashlib
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion import training  # noqa: E402
from densiffusion import waveform_experiment as experiment  # noqa: E402
from densiffusion.metrics import evaluate_forecast  # noqa: E402


def read_json(path):
    return json.loads(path.read_text())


def sha256(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Exercise real training and sampling, recording actual work."""
    output = tmp_path_factory.mktemp("waveform") / "run"
    config = experiment.WaveformConfig(
        seeds=(913, 914),
        n_train=8,
        n_validation=4,
        n_test=4,
        epochs=2,
        densifier_epochs=2,
        validation_interval=1,
        densifier_validation_interval=1,
        batch_size=4,
        sample_batch_size=8,
        diffusion_steps=2,
        n_histories=2,
        n_futures=2,
        label="smoke",
        workers=1,
    )
    calls = []
    starts = {}
    updates = Counter()
    orders = defaultdict(list)
    active = [None]
    original_dataset = experiment.dataset
    original_train = experiment.train_forecaster
    original_densifier_train = experiment.train_densifier
    original_loss = training._loss
    original_step = torch.optim.Adam.step
    original_sample = experiment.sample_forecasts

    def dataset(config, seed, split):
        frozen = (output / "FROZEN.json").exists()
        calls.append((seed, split, frozen))
        if split == "test":
            assert frozen
            assert all(
                (output / f"seed_{paired_seed}" / "TRAINED.json").is_file()
                for paired_seed in config.seeds
            )
        else:
            assert not frozen
        return original_dataset(config, seed, split)

    def train(model, densifier, train, validation, **kwargs):
        path = kwargs["checkpoint"]
        key = (
            int(path.parent.name.removeprefix("seed_")),
            path.stem.removesuffix("_forecaster"),
        )
        starts[key] = {
            "state_hash": experiment.model_hash(model),
            "config": kwargs["config"],
            "seed": kwargs["seed"],
            "train": train,
            "validation": validation,
        }
        active[0] = key
        try:
            return original_train(model, densifier, train, validation, **kwargs)
        finally:
            active[0] = None

    def train_densifier(*args, **kwargs):
        assert active[0] is None
        return original_densifier_train(*args, **kwargs)

    def loss(model, target, mask, alpha_bar, rng, dt, objective, **kwargs):
        if active[0] is not None and torch.is_grad_enabled():
            key = active[0]
            factor = 1 if key[1] == "direct_csdi" else config.factor
            context = (config.context_length - 1) * factor + 1
            assert target.shape[1] == context + factor * config.horizon
            assert not mask[:, :context].any()
            assert mask[:, context:].all()
            assert dt == config.dt / factor
            orders[key].append(target[:, 0, 0].detach().cpu().tolist())
        return original_loss(
            model, target, mask, alpha_bar, rng, dt, objective, **kwargs
        )

    def step(optimizer, *args, **kwargs):
        if active[0] is not None:
            updates[active[0]] += 1
        return original_step(optimizer, *args, **kwargs)

    def sample(model, densifier, history, config, method, seed):
        # The runner passes only past values into both validation and test forecasts.
        split = "test" if (output / "FROZEN.json").exists() else "validation"
        matching_seeds = [
            paired_seed
            for paired_seed in config.seeds
            if seed == experiment.seed_plan(paired_seed)[f"{split}_sampling"]
        ]
        assert len(matching_seeds) == 1
        expected = original_dataset(config, matching_seeds[0], split)
        np.testing.assert_array_equal(history, expected[:, : config.context_length])
        return original_sample(model, densifier, history, config, method, seed)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(experiment, "dataset", dataset)
        monkeypatch.setattr(experiment, "train_forecaster", train)
        monkeypatch.setattr(experiment, "train_densifier", train_densifier)
        monkeypatch.setattr(training, "_loss", loss)
        monkeypatch.setattr(torch.optim.Adam, "step", step)
        monkeypatch.setattr(experiment, "sample_forecasts", sample)
        experiment.train_run(config, output)
        calls_after_training = calls.copy()
        assert not any(output.glob("seed_*/samples.npz"))
        scores = experiment.evaluate_run(output)
    return {
        "output": output,
        "config": config,
        "calls": calls,
        "training_calls": calls_after_training,
        "starts": starts,
        "updates": updates,
        "orders": orders,
        "scores": scores,
    }


def test_all_paired_models_freeze_before_any_test_access(smoke_run):
    run = smoke_run
    config = run["config"]
    assert run["training_calls"] == [
        (seed, split, False)
        for seed in config.seeds
        for split in ("train", "validation")
    ]
    assert run["calls"][len(run["training_calls"]) :] == [
        (seed, "test", True) for seed in config.seeds
    ]
    frozen = read_json(run["output"] / "FROZEN.json")
    assert len(frozen["checkpoint_sha256"]) == 4 * len(config.seeds)
    assert (run["output"] / "EVALUATED.json").is_file()
    manifest = read_json(run["output"] / "config.json")
    assert manifest["normalization"] == "none"
    assert not manifest["fine_grid_truth_before_freeze"]
    assert not manifest["test_data_generated_before_all_models_frozen"]
    assert not manifest["joint_fine_truth_assumption"]
    assert (run["output"] / "source.tar.gz").stat().st_size > 0


def test_same_forecaster_initialization_actual_updates_order_and_optimizer(smoke_run):
    run = smoke_run
    config = run["config"]
    for seed in config.seeds:
        starts = [run["starts"][seed, method] for method in experiment.METHODS]
        assert len({start["state_hash"] for start in starts}) == 1
        assert len({start["seed"] for start in starts}) == 1
        assert all(start["config"] == starts[0]["config"] for start in starts)
        reference_order = run["orders"][seed, experiment.METHODS[0]]
        for method, start in zip(experiment.METHODS, starts, strict=True):
            assert run["updates"][seed, method] == 4
            assert run["orders"][seed, method] == reference_order
            for split in ("train", "validation"):
                for actual, expected in zip(
                    start[split], starts[0][split], strict=True
                ):
                    np.testing.assert_array_equal(actual, expected)
        audit = read_json(run["output"] / f"seed_{seed}" / "pairing_audit.json")
        assert audit["updates_per_forecaster"] == 4
        assert len(set(audit["initial_state_sha256"].values())) == 1
        assert audit["additional_densifier_parameters"] > 0


def test_training_history_augmentation_receives_only_past_observations(smoke_run):
    config = smoke_run["config"]
    for seed in config.seeds:
        directory = smoke_run["output"] / f"seed_{seed}"
        with np.load(directory / "train_validation.npz", allow_pickle=False) as values:
            expected_calls = []
            for split in ("train", "validation"):
                history, future = experiment.split_pair(values[split], config)
                for start in range(0, len(history), config.batch_size):
                    past = history[start : start + config.batch_size]
                    bridge = np.concatenate(
                        (past[:, -1:], future[start : start + config.batch_size]), 1
                    )
                    expected_calls.extend([(past, 49), (bridge, 17)])
            train_ids = values["train_trajectory_ids"]
            validation_ids = values["validation_trajectory_ids"]
            assert not set(train_ids) & set(validation_ids)
        for method in experiment.METHODS:
            audit = read_json(directory / f"{method}_augmentation_audit.json")
            if method == "direct_csdi":
                assert audit == []
                continue
            assert len(audit) == len(expected_calls)
            for actual, (source, dense_length) in zip(
                audit, expected_calls, strict=True
            ):
                assert actual["input_sha256"] == sha256(source)
                assert actual["input_shape"] == list(source.shape)
                assert actual["output_shape"] == [len(source), 1, dense_length, 1]
                assert actual["observations_preserved"]
                assert actual["dt"] == 0.1
                assert actual["seconds"] >= 0


def test_smoke_saved_scores_samples_times_and_validation_selected_weights(smoke_run):
    output, config = smoke_run["output"], smoke_run["config"]
    assert smoke_run["scores"] == read_json(output / "scores.json")
    for seed in config.seeds:
        directory = output / f"seed_{seed}"
        timings = read_json(directory / "timings.json")
        assert timings["densifier_training_seconds"] >= 0
        scores = read_json(directory / "scores.json")
        assert scores == smoke_run["scores"][str(seed)]
        with np.load(directory / "samples.npz", allow_pickle=False) as saved:
            assert saved["history"].shape == (4, 25, 1)
            assert saved["target"].shape == (4, 8, 1)
            np.testing.assert_allclose(saved["history_times"], np.arange(25) * 0.1)
            np.testing.assert_allclose(saved["future_times"], np.arange(25, 33) * 0.1)
            np.testing.assert_array_equal(saved["trajectory_ids"], np.arange(12, 16))
            for method in experiment.METHODS:
                assert saved[method].shape == (4, 4, 8, 1)
                assert np.isfinite(saved[method]).all()
                assert scores[method] == evaluate_forecast(
                    saved[method], saved["target"]
                )
                method_times = timings["methods"][method]
                assert all(seconds >= 0 for seconds in method_times.values())
                assert method_times["forecast_seconds_per_trajectory"] == pytest.approx(
                    method_times["forecast_seconds"] / config.n_test
                )
                validation = read_json(directory / f"{method}_validation.json")
                assert [row["epoch"] for row in validation] == [1, 2]
                checkpoint = torch.load(
                    directory / f"{method}_forecaster.pt", weights_only=True
                )
                assert checkpoint["selection_score"] == min(
                    row["crps"] for row in validation
                )
                assert "forecast_crps" in checkpoint["selection_metric"]
                assert checkpoint["context_length"] == 25
                assert checkpoint["horizon"] == 8


def test_existing_run_and_test_forecasts_are_immutable(smoke_run):
    output = smoke_run["output"]
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        experiment.train_run(smoke_run["config"], output)
    with pytest.raises(FileExistsError, match="already started"):
        experiment.evaluate_run(output)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert before == after


@pytest.mark.parametrize("change", ["config", "checkpoint"])
def test_post_freeze_changes_are_rejected_before_generating_test(
    smoke_run, tmp_path, monkeypatch, change
):
    output = tmp_path / "tampered"
    output.mkdir()
    config_bytes = (smoke_run["output"] / "config.json").read_bytes()
    frozen = read_json(smoke_run["output"] / "FROZEN.json")
    # One hash is sufficient to check the guard, without copying every checkpoint.
    name, checksum = next(iter(frozen["checkpoint_sha256"].items()))
    frozen["checkpoint_sha256"] = {name: checksum}
    (output / "FROZEN.json").write_text(json.dumps(frozen))
    (output / "config.json").write_bytes(
        config_bytes + (b" " if change == "config" else b"")
    )
    checkpoint = output / name
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"changed weights")

    def forbidden_test(*args, **kwargs):
        pytest.fail("Test observations were accessed before freeze integrity checks")

    monkeypatch.setattr(experiment, "dataset", forbidden_test)
    with pytest.raises(ValueError, match="changed after freezing"):
        experiment.evaluate_run(output)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("architecture", "Architecture differs"),
        ("forecaster_schedule", "Training/sampling schedule differs"),
        ("densifier_schedule", "Training/sampling schedule differs"),
        ("seed_allocation", "Seed allocation differs"),
    ],
)
def test_source_drift_is_rejected_before_test_access(
    smoke_run, tmp_path, monkeypatch, change, message
):
    # Reuse the tiny trained weights in an unevaluated directory. The freeze
    # manifest and checkpoint/config bytes stay intact; only live code changes.
    output = tmp_path / "source_drift"
    output.mkdir()
    frozen = read_json(smoke_run["output"] / "FROZEN.json")
    for name in ("FROZEN.json", "config.json", *frozen["checkpoint_sha256"]):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((smoke_run["output"] / name).read_bytes())

    def forbidden_test(*args, **kwargs):
        pytest.fail("Test observations were accessed after frozen code drifted")

    monkeypatch.setattr(experiment, "dataset", forbidden_test)
    if change == "architecture":
        monkeypatch.setattr(
            experiment,
            "ARCHITECTURE",
            {**experiment.ARCHITECTURE, "hidden_channels": 32},
        )
    elif change.endswith("schedule"):
        original_config = experiment.training_config

        def changed_config(config, *, densifier=False):
            current = original_config(config, densifier=densifier)
            if densifier == (change == "densifier_schedule"):
                return replace(current, beta_end=0.4)
            return current

        monkeypatch.setattr(experiment, "training_config", changed_config)
    else:
        original_plan = experiment.seed_plan

        def changed_plan(seed):
            plan = original_plan(seed)
            return {**plan, "test_data": plan["test_data"] + 1}

        monkeypatch.setattr(experiment, "seed_plan", changed_plan)
    with pytest.raises(ValueError, match=message):
        experiment.evaluate_run(output)
    assert not any(output.glob("seed_*/samples.npz"))
    assert not (output / "EVALUATED.json").exists()


def test_evaluation_requires_completed_freeze(tmp_path, monkeypatch):
    def forbidden_test(*args, **kwargs):
        pytest.fail("Test observations were generated without a completed freeze")

    monkeypatch.setattr(experiment, "dataset", forbidden_test)
    with pytest.raises(FileNotFoundError, match="FROZEN"):
        experiment.evaluate_run(tmp_path)


def test_scientific_configs_require_at_least_64_scored_draws():
    config = experiment.WaveformConfig()
    assert config.n_histories * config.n_futures == 64
    assert len(config.seeds) == 3
    assert (config.n_train, config.n_validation, config.n_test) == (1024, 256, 1024)
    with pytest.raises(ValueError, match="64"):
        replace(config, n_histories=4)
    assert replace(config, label="smoke", n_histories=2, n_futures=2)


@pytest.fixture(scope="module")
def verifier():
    path = Path(__file__).parents[1] / "scripts" / "verify_waveform_forecast.py"
    spec = importlib.util.spec_from_file_location("verify_waveform_forecast", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_artifact_verifier_replays_completed_smoke_without_inference(
    smoke_run, verifier, monkeypatch
):
    def forbidden_inference(*args, **kwargs):
        pytest.fail("Artifact verification performed new model inference")

    monkeypatch.setattr(experiment, "sample_forecasts", forbidden_inference)
    report = verifier.verify_run(smoke_run["output"])
    assert report["status"] == "passed"
    assert not report["forecast_inference_or_training_performed"]
    assert report == read_json(smoke_run["output"] / "verification.json")
    assert set(report["seeds"]) == {str(seed) for seed in smoke_run["config"].seeds}


@pytest.mark.parametrize("marker", ["FROZEN.json", "EVALUATED.json"])
def test_verifier_requires_completed_run_before_any_data_replay(
    tmp_path, monkeypatch, verifier, marker
):
    other_marker = "EVALUATED.json" if marker == "FROZEN.json" else "FROZEN.json"
    (tmp_path / other_marker).write_text("{}")

    def forbidden_replay(*args, **kwargs):
        pytest.fail("Verifier replayed test observations for an incomplete run")

    monkeypatch.setattr(verifier, "generate_waveforms", forbidden_replay)
    with pytest.raises(FileNotFoundError, match=marker):
        verifier.verify_run(tmp_path)
    assert not (tmp_path / "verification.json").exists()


@pytest.mark.parametrize("change", ["samples", "augmentation"])
def test_verifier_detects_changed_samples_and_future_leakage_audits(
    smoke_run, tmp_path, verifier, change
):
    output = tmp_path / "changed_artifact"
    shutil.copytree(
        smoke_run["output"], output, ignore=shutil.ignore_patterns("verification.json")
    )
    directory = output / f"seed_{smoke_run['config'].seeds[0]}"
    if change == "samples":
        path = directory / "samples.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["direct_csdi"][0, 0, 0, 0] = 1000
        np.savez_compressed(path, **arrays)
        expected = "metric recomputation failed"
    else:
        path = directory / "densiffusion_augmentation_audit.json"
        calls = read_json(path)
        calls[0]["input_sha256"] = "0" * 64
        path.write_text(json.dumps(calls))
        expected = "conditioning input hash"
    with pytest.raises(ValueError, match=expected):
        verifier.verify_run(output)
    assert not (output / "verification.json").exists()


def test_verifier_augmentation_batching_handles_repeated_trajectories(verifier):
    train = np.arange(15, dtype=float).reshape(3, 5, 1)
    validation = train[:1] + 100
    config = {"context_length": 3, "n_augmentations": 3, "batch_size": 4}
    calls = list(verifier.augmentation_inputs(train, validation, config))
    # Three draws per trajectory, with batches crossing repeated-trajectory blocks.
    expected_indices = [[0, 0, 0, 1], [1, 1, 2, 2], [2]]
    assert len(calls) == 8
    for batch, indices in enumerate(expected_indices):
        np.testing.assert_array_equal(calls[2 * batch], train[indices, :3])
        np.testing.assert_array_equal(calls[2 * batch + 1], train[indices, 2:])
    np.testing.assert_array_equal(calls[-2], np.repeat(validation[:, :3], 3, axis=0))
    np.testing.assert_array_equal(calls[-1], np.repeat(validation[:, 2:], 3, axis=0))
