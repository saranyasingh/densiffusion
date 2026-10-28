import hashlib
import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion import waveform_imputation as experiment  # noqa: E402
from densiffusion.generators import waveform  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402
from densiffusion.waveform_imputation_analysis import (  # noqa: E402
    analyze_imputation_run,
)

LARGE_ARCHITECTURE = {
    "n_channels": 1,
    "hidden_channels": 64,
    "residual_layers": 4,
    "attention_heads": 8,
    "diffusion_embedding_dim": 128,
    "time_embedding_dim": 128,
    "feature_embedding_dim": 16,
    "feedforward_dim": 64,
    "dropout": 0.1,
}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture(scope="module")
def verifier():
    path = Path(__file__).parents[1] / "scripts" / "verify_waveform_imputation.py"
    spec = importlib.util.spec_from_file_location("verify_waveform_imputation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("waveform_imputation")
    data_source, recipe_source, output = base / "original", base / "gp", base / "run"
    directory = data_source / "seed_123"
    directory.mkdir(parents=True)
    recipe_source.mkdir()
    train = waveform.generate_waveforms(8, rng=11)
    validation = waveform.generate_waveforms(4, rng=12)
    np.savez_compressed(
        directory / "train_validation.npz",
        train=train,
        validation=validation,
        train_trajectory_ids=np.arange(8),
        validation_trajectory_ids=np.arange(8, 12),
    )
    training = TrainingConfig(
        epochs=200,
        batch_size=64,
        diffusion_steps=50,
        beta_end=0.5,
        schedule="quadratic",
        weight_decay=1e-6,
        lr_decay_fractions=(0.75, 0.9),
        validation_steps="all",
        validation_interval=10,
    )
    write_json(
        recipe_source / "plan.json",
        {
            "selection": "selected GP recipe",
            "candidates": {
                "reference_size": {
                    "architecture": LARGE_ARCHITECTURE,
                    "training": asdict(training),
                }
            },
        },
    )
    write_json(recipe_source / "selection.json", {"selected": "reference_size"})
    (recipe_source / "gp_weights.pt").write_bytes(b"Forbidden GP checkpoint")
    config = experiment.ImputationConfig(
        seeds=(123,),
        epochs=2,
        score_interval=1,
        samples=4,
        sample_batch_size=16,
        threads=1,
        workers=1,
        transfer_examples=4,
        label="smoke",
    )
    load = torch.load

    def forbid_gp_load(path, *args, **kwargs):
        assert recipe_source not in Path(path).parents
        return load(path, *args, **kwargs)

    def forbid_generation(*args, **kwargs):
        pytest.fail("Imputation training/selection generated new or fine-grid data")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch, "load", forbid_gp_load)
        monkeypatch.setattr(waveform, "generate_waveforms", forbid_generation)
        experiment.train_run(output, data_source, recipe_source, config)
        analyze_imputation_run(output, bootstrap_replicates=128)
        experiment.freeze_run(output)
    return {
        "output": output,
        "data_source": data_source,
        "recipe_source": recipe_source,
        "config": config,
    }


def test_larger_recipe_smoke_preserves_data_and_uses_validation_only(
    smoke_run, verifier
):
    output = smoke_run["output"]
    report = verifier.verify_run(output)
    assert report["status"] == "passed"
    assert report["fresh_random_initializations_reproduced"]
    assert not report["gp_weights_loaded_by_verifier"]
    assert not report["inference_or_training_performed"]
    assert not report["validation_gate"][
        "passed"
    ]  # One seed cannot meet a 3-seed gate.
    assert report["transfer"] == {"status": "not_run"}
    assert report == read_json(output / "verification.json")
    manifest = read_json(output / "config.json")
    assert manifest["architecture"] == LARGE_ARCHITECTURE
    assert manifest["training"]["validation_steps"] == "all"
    assert (output / "seed_123" / "train_validation.npz").read_bytes() == (
        smoke_run["data_source"] / "seed_123" / "train_validation.npz"
    ).read_bytes()
    summary = read_json(output / "seed_123" / "summary.json")
    assert summary["optimizer_updates"] == 2
    assert summary["parameter_count"] > 17_273
    assert not any(output.glob("seed_*/transfer*.npz"))


def test_gate_failure_blocks_transfer_before_any_data_generation(
    smoke_run, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Transfer generated truth despite the failed validation gate")

    monkeypatch.setattr(waveform, "generate_waveforms", forbidden)
    with pytest.raises(ValueError, match="validation gate"):
        experiment.transfer_run(smoke_run["output"])


def test_original_runs_and_frozen_selections_are_immutable(smoke_run):
    output = smoke_run["output"]
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        experiment.train_run(
            output,
            smoke_run["data_source"],
            smoke_run["recipe_source"],
            smoke_run["config"],
        )
    with pytest.raises(FileExistsError):
        experiment.freeze_run(output)
    with pytest.raises(FileExistsError, match="Frozen"):
        analyze_imputation_run(output, bootstrap_replicates=32)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


@pytest.mark.parametrize(
    "change", ["mask", "observation", "metric", "selected", "logs"]
)
def test_verifier_rejects_bad_imputation_artifacts(
    smoke_run, verifier, tmp_path, change
):
    output = tmp_path / "altered"
    shutil.copytree(
        smoke_run["output"], output, ignore=shutil.ignore_patterns("verification.json")
    )
    directory = output / "seed_123"
    if change in ("mask", "observation"):
        path = directory / "validation_epoch_0001.npz"
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        if change == "mask":
            arrays["mask"][0, 0, 0] = True
            message = "mask"
        else:
            arrays["learned"][0, 0, 0, 0] += 1
            message = "preserved observations"
        np.savez_compressed(path, **arrays)
    elif change == "metric":
        path = directory / "validation.json"
        data = read_json(path)
        data[0]["scores"]["crps"] += 1
        write_json(path, data)
        message = "metric recomputation"
    elif change == "selected":
        (directory / "validation_samples.npz").write_bytes(b"wrong selected ensemble")
        message = "selected samples differ"
    else:
        path = directory / "training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0].pop("selection_score")
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        message = "selection cadence"
    with pytest.raises(ValueError, match=message):
        verifier.verify_run(output)
    assert not (output / "verification.json").exists()


def test_verifier_does_not_replay_incomplete_transfer(
    smoke_run, verifier, tmp_path, monkeypatch
):
    output = tmp_path / "incomplete_transfer"
    shutil.copytree(
        smoke_run["output"], output, ignore=shutil.ignore_patterns("verification.json")
    )
    (output / "seed_123" / "transfer_samples.npz").write_bytes(b"partial")

    def forbidden(*args, **kwargs):
        pytest.fail("Verifier generated fine truth for incomplete transfer diagnostics")

    monkeypatch.setattr(verifier, "generate_waveforms", forbidden)
    with pytest.raises(ValueError, match="completed TRANSFER"):
        verifier.verify_run(output)


def test_frozen_checkpoint_change_rejected_before_any_replay(
    smoke_run, verifier, tmp_path, monkeypatch
):
    output = tmp_path / "changed_checkpoint"
    shutil.copytree(
        smoke_run["output"], output, ignore=shutil.ignore_patterns("verification.json")
    )
    (output / "seed_123" / "densifier.pt").write_bytes(b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("Verifier generated data despite an invalid freeze")

    monkeypatch.setattr(verifier, "generate_waveforms", forbidden)
    with pytest.raises(ValueError, match="Frozen checkpoint changed"):
        verifier.verify_run(output)


def test_complete_transfer_replays_both_geometries_without_inference(
    smoke_run, verifier, tmp_path
):
    # Isolate transfer verification: a synthetic passing freeze authorizes this
    # test-only diagnostic, without overriding the real failed validation gate.
    output = tmp_path / "transfer"
    (output / "seed_123").mkdir(parents=True)
    manifest = read_json(smoke_run["output"] / "config.json")
    config = manifest["experiment"]
    streams = np.random.SeedSequence([123, config["diagnostic_namespace"]]).spawn(3)
    data_seed, history_seed, future_seed = [
        int(s.generate_state(1)[0]) for s in streams
    ]
    fine = waveform.generate_waveforms(
        config["transfer_examples"], 65, 0.05, rng=data_seed
    )
    result = {"data_seed": data_seed, "tasks": {}}
    for name, target, seed, filename in (
        ("history", fine[:, :49], history_seed, "transfer_samples.npz"),
        ("anchored_future", fine[:, 48:], future_seed, "transfer_future_samples.npz"),
    ):
        samples = experiment.linear_samples(target, config["samples"])
        experiment.save_samples(
            output / "seed_123" / filename,
            target,
            samples,
            smoke_run["config"],
            target_dt=0.05,
        )
        scores = experiment.hidden_scores(samples, target)
        result["tasks"][name] = {
            "learned": scores,
            "linear": scores,
            "sampling_seconds": 0.0,
            "sampling_seed": seed,
            "observations_preserved": True,
            "crps_ratio": 1.0,
            "mse_ratio": 1.0,
        }
    write_json(output / "seed_123" / "transfer_scores.json", result)
    write_json(
        output / "TRANSFER.json",
        {
            "passed": True,
            "gate": manifest["transfer_gate"],
            "diagnostic_only": True,
            "no_tuning_after_fine_truth": True,
            "seeds": {"123": result},
        },
    )
    verified = verifier.verify_transfer(
        output, manifest, {"validation_gate": {"passed": True}}
    )
    assert verified["passed"]
    assert verified["tasks"] == {"history_points": 49, "anchored_future_points": 17}
    with np.load(output / "seed_123" / "transfer_future_samples.npz") as saved:
        require_first_anchor = saved["target"][:, 0]
    np.testing.assert_array_equal(require_first_anchor, fine[:, 48])


def test_configured_150_epoch_crps_interval_and_actual_log_cadence(smoke_run, verifier):
    manifest = read_json(smoke_run["output"] / "config.json")
    manifest["experiment"].update(
        label="longer_attempt", epochs=600, score_interval=150, samples=64
    )
    manifest["training"] = {
        **manifest["recovered_gp_recipe"]["training"],
        "epochs": 600,
    }
    verifier.verify_recipe(manifest)
    config, training = manifest["experiment"], manifest["training"]
    records = []
    for epoch in range(1, 601):
        row = {"epoch": epoch, "train_loss": 1.0}
        if epoch % 10 == 0:
            row["validation_loss"] = 1.0
        if epoch % 150 == 0:
            row["selection_score"] = 1.0
        records.append(row)
    verifier.verify_training_cadence(records, config, training, "longer_attempt")
    assert [row["epoch"] for row in records if "selection_score" in row] == [
        150,
        300,
        450,
        600,
    ]
    # An unconfigured extra selection opportunity must invalidate the audit.
    records[49]["selection_score"] = 0.5
    with pytest.raises(ValueError, match="selection cadence"):
        verifier.verify_training_cadence(records, config, training, "longer_attempt")
    config["score_interval"] = 151
    with pytest.raises(ValueError, match="positive multiple"):
        verifier.verify_recipe(manifest)
