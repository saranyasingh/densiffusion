import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_waveform_large_forecast import gated_source  # noqa: E402, F401

from densiffusion import waveform_large_forecast as large  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture(scope="module")
def verifier():
    path = Path(__file__).parents[1] / "scripts" / "verify_waveform_large_forecast.py"
    spec = importlib.util.spec_from_file_location(
        "verify_waveform_large_forecast", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completed_run(gated_source, tmp_path):  # noqa: F811
    # Deliberately mocked source gates and a tiny architecture. This fixture
    # exercises artifact wiring; it is not an actual larger forecasting result.
    source, config = gated_source
    config = replace(config, epochs=2)
    output = tmp_path / "forecast"
    previous_threads = torch.get_num_threads()
    try:
        large.train_run(output, source, config)
        large.evaluate_run(output)
    finally:
        torch.set_num_threads(previous_threads)
    return output, source, config


def update_evidence_hash(output, path, verifier):
    frozen = read(output / "FROZEN.json")
    frozen["selection_evidence_sha256"][str(path.relative_to(output))] = (
        verifier.sha256(path)
    )
    write(output / "FROZEN.json", frozen)


def test_independent_verifier_checks_completed_controlled_fixture(
    completed_run, verifier
):
    output, source, config = completed_run
    source_before = {
        path: verifier.sha256(path) for path in source.rglob("*") if path.is_file()
    }
    report = verifier.verify_run(output)
    assert report["status"] == "passed"
    assert report == read(output / "verification.json")
    assert report["fresh_test_streams_replayed"]
    assert not report["forecast_training_or_inference_performed"]
    assert "Controlled smoke fixture" in report["source_gate_verification"]
    assert set(report["seeds"]) == set(map(str, config.seeds))
    for row in report["seeds"].values():
        assert row["measured_updates_per_forecaster"] == 4
        assert set(row["selected_checkpoints"]) == set(large.METHODS)
    assert source_before == {path: verifier.sha256(path) for path in source_before}


@pytest.mark.parametrize("marker", ["FROZEN.json", "EVALUATED.json"])
def test_missing_completion_markers_block_any_generator(
    completed_run, verifier, monkeypatch, marker
):
    output, _, _ = completed_run
    (output / marker).unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete run caused data replay")

    monkeypatch.setattr(verifier, "generate_waveforms", forbidden)
    with pytest.raises(FileNotFoundError, match=marker):
        verifier.verify_run(output)


def test_source_gate_is_checked_independently_of_provenance_hashes(
    completed_run, verifier
):
    output, source, _ = completed_run
    manifest = read(output / "config.json")
    write(source / "TRANSFER.json", {"passed": False})
    manifest["densifier_import"]["source_sha256"]["TRANSFER.json"] = verifier.sha256(
        source / "TRANSFER.json"
    )
    with pytest.raises(ValueError, match="Both source"):
        verifier.verify_source(manifest)


def test_candidate_metric_recomputed_even_with_matching_archive_hashes(
    completed_run, verifier, monkeypatch
):
    output, _, config = completed_run
    directory = output / f"seed_{config.seeds[0]}"
    rows_path = directory / "direct_csdi_validation.json"
    rows = read(rows_path)
    best = min(rows, key=lambda row: row["crps"])
    candidate = next(row for row in rows if row["epoch"] != best["epoch"])
    path = directory / candidate["samples"]
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["samples"][0, 0, 0, 0] = 1000
    np.savez_compressed(path, **arrays)
    candidate["samples_sha256"] = verifier.sha256(path)
    write(rows_path, rows)
    update_evidence_hash(output, path, verifier)
    update_evidence_hash(output, rows_path, verifier)

    def forbidden(*args, **kwargs):
        pytest.fail("Test data replay preceded candidate score verification")

    monkeypatch.setattr(verifier, "generate_waveforms", forbidden)
    with pytest.raises(ValueError, match="candidate metric recomputation"):
        verifier.verify_run(output)


def test_frozen_inventory_requires_every_validation_candidate(completed_run, verifier):
    output, _, _ = completed_run
    frozen = read(output / "FROZEN.json")
    name = next(
        key for key in frozen["selection_evidence_sha256"] if key.endswith(".npz")
    )
    del frozen["selection_evidence_sha256"][name]
    write(output / "FROZEN.json", frozen)
    with pytest.raises(ValueError, match="selection-evidence inventory"):
        verifier.verify_run(output)


@pytest.mark.parametrize("change", ["updates", "history", "test_stream", "q_weights"])
def test_pairing_leakage_stream_and_frozen_weight_guards(
    completed_run, verifier, change
):
    output, _, config = completed_run
    directory = output / f"seed_{config.seeds[0]}"
    if change == "updates":
        path = directory / "pairing_audit.json"
        audit = read(path)
        audit["measured_updates_by_method"]["direct_csdi"] += 1
        write(path, audit)
        update_evidence_hash(output, path, verifier)
        message = "optimizer budgets"
    elif change == "history":
        path = directory / "densiffusion_augmentation_audit.json"
        calls = read(path)
        calls[0]["input_sha256"] = "0" * 64
        write(path, calls)
        update_evidence_hash(output, path, verifier)
        message = "authorized original observations"
    elif change == "test_stream":
        path = directory / "seeds.json"
        seeds = read(path)
        original, _ = verifier.seed_plans(config.seeds[0], config.test_namespace)
        seeds["test_data"] = original["test_data"]
        write(path, seeds)
        message = "fresh test namespace"
    else:
        (directory / "densifier.pt").write_bytes(b"changed frozen q")
        message = "Frozen checkpoint_sha256"
    with pytest.raises(ValueError, match=message):
        verifier.verify_run(output)


def test_final_saved_targets_must_match_fresh_namespace_replay(completed_run, verifier):
    output, _, config = completed_run
    directory = output / f"seed_{config.seeds[0]}"
    path = directory / "samples.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["target"][0, 0, 0] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="fresh test target"):
        verifier.verify_run(output)


def test_selected_checkpoint_epoch_must_match_validation_argmin(
    completed_run, verifier
):
    output, _, config = completed_run
    directory = output / f"seed_{config.seeds[0]}"
    path = directory / "direct_csdi_forecaster.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["epoch"] = 3 - checkpoint["epoch"]
    torch.save(checkpoint, path)
    frozen = read(output / "FROZEN.json")
    frozen["checkpoint_sha256"][str(path.relative_to(output))] = verifier.sha256(path)
    write(output / "FROZEN.json", frozen)
    with pytest.raises(ValueError, match="first CRPS argmin"):
        verifier.verify_run(output)


def test_final_saved_forecasts_must_reproduce_reported_metrics(completed_run, verifier):
    output, _, config = completed_run
    path = output / f"seed_{config.seeds[0]}" / "samples.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["densiffusion"][0, 0, 0, 0] = 1000
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="final metric recomputation"):
        verifier.verify_run(output)
