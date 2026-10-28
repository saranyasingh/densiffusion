import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion import waveform_large_forecast as large  # noqa: E402
from densiffusion.architectures.csdi import CSDIDenoiser  # noqa: E402
from densiffusion.metrics import evaluate_forecast  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402
from densiffusion.waveform_experiment import seed_plan  # noqa: E402
from densiffusion.waveform_imputation import sha256  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def gated_source(tmp_path):
    """Tiny synthetic gate fixtures; no scientific experiment or GP weights."""
    source = tmp_path / "imputation"
    source.mkdir()
    seeds = (91, 92)
    architecture = {
        "n_channels": 1,
        "hidden_channels": 4,
        "residual_layers": 1,
        "attention_heads": 2,
        "diffusion_embedding_dim": 4,
        "time_embedding_dim": 4,
        "feature_embedding_dim": 2,
        "feedforward_dim": 4,
        "dropout": 0.1,
    }
    training = asdict(TrainingConfig(epochs=1, batch_size=2, diffusion_steps=2))
    manifest = {
        "experiment": {"seeds": list(seeds)},
        "architecture": architecture,
        "training": training,
        "recovered_gp_recipe": {
            "architecture": architecture,
            "training": training,
            "gp_weights_reused": False,
        },
    }
    write(source / "config.json", manifest)
    write(source / "imputation_analysis.json", {"validation_gate": {"passed": True}})
    checkpoints = {}
    for seed in seeds:
        folder = source / f"seed_{seed}"
        folder.mkdir()
        plans = seed_plan(seed)
        write(folder / "seeds.json", plans)
        torch.manual_seed(plans["densifier_initialization"])
        model = ImputationModel(CSDIDenoiser(**architecture))
        torch.save(
            {"config": training, "epoch": 1, "model_state_dict": model.state_dict()},
            folder / "densifier.pt",
        )
        rng = np.random.default_rng(seed)
        np.savez_compressed(
            folder / "train_validation.npz",
            train=rng.normal(size=(4, 5, 1)),
            validation=rng.normal(size=(2, 5, 1)),
            train_trajectory_ids=np.arange(4),
            validation_trajectory_ids=np.arange(4, 6),
        )
        write(
            folder / "summary.json",
            {
                "selected_epoch": 1,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "training_seconds": 12.5,
            },
        )
        checkpoints[f"seed_{seed}/densifier.pt"] = sha256(folder / "densifier.pt")
    write(source / "TRAINED.json", {})
    write(
        source / "DENSIFIERS_FROZEN.json",
        {
            "validation_gate": {"passed": True},
            "config_sha256": sha256(source / "config.json"),
            "analysis_sha256": sha256(source / "imputation_analysis.json"),
            "checkpoint_sha256": checkpoints,
        },
    )
    write(source / "TRANSFER.json", {"passed": True})
    config = large.LargeForecastConfig(
        seeds=seeds,
        epochs=1,
        validation_interval=1,
        validation_score_interval=1,
        n_train=4,
        n_validation=2,
        n_test=3,
        context_length=3,
        horizon=2,
        batch_size=2,
        sample_batch_size=4,
        n_histories=1,
        n_futures=2,
        diffusion_steps=3,
        workers=1,
        test_namespace=2026102704,
        label="smoke",
    )
    return source, config


@pytest.mark.parametrize("gate", ["validation", "transfer", "missing_freeze"])
def test_gates_block_before_any_training_or_test_access(
    gated_source, tmp_path, monkeypatch, gate
):
    source, config = gated_source
    if gate == "validation":
        frozen = read(source / "DENSIFIERS_FROZEN.json")
        frozen["validation_gate"]["passed"] = False
        write(source / "DENSIFIERS_FROZEN.json", frozen)
    elif gate == "transfer":
        write(source / "TRANSFER.json", {"passed": False})
    else:
        (source / "DENSIFIERS_FROZEN.json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Training or data generation occurred before gates")

    monkeypatch.setattr(large, "train_forecaster", forbidden)
    monkeypatch.setattr(large, "generate_waveforms", forbidden)
    output = tmp_path / "forecast"
    with pytest.raises((ValueError, FileNotFoundError)):
        large.train_run(output, source, config)
    assert not output.exists()


def test_exact_import_pairing_and_global_freeze_before_fresh_test(
    gated_source, tmp_path, monkeypatch
):
    source, config = gated_source
    output = tmp_path / "forecast"
    protected = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    generated = []
    original_generator = large.generate_waveforms

    def generator(*args, **kwargs):
        assert (output / "FROZEN.json").is_file()
        assert all((output / f"seed_{s}/TRAINED.json").is_file() for s in config.seeds)
        generated.append(1)
        return original_generator(*args, **kwargs)

    monkeypatch.setattr(large, "generate_waveforms", generator)
    previous_threads = torch.get_num_threads()
    try:
        manifest = large.train_run(output, source, config)
        assert not generated
        assert manifest["fine_grid_truth_used_for_training_or_selection"] is False
        assert (
            manifest["transfer_diagnostics_completed_before_forecast_training"] is True
        )
        assert "fine_grid_truth_before_freeze" not in manifest
        for seed in config.seeds:
            folder = output / f"seed_{seed}"
            assert (folder / "densifier.pt").read_bytes() == protected[
                source / folder.name / "densifier.pt"
            ]
            assert (folder / "train_validation.npz").read_bytes() == protected[
                source / folder.name / "train_validation.npz"
            ]
            audit = read(folder / "pairing_audit.json")
            assert set(audit["measured_updates_by_method"].values()) == {2}
            assert len(set(audit["initial_state_sha256"].values())) == 1
            assert audit["densifier_weights_unchanged"]
            model, sampler = large._load_q(folder, manifest, config)
            assert not any(p.requires_grad for p in model.parameters())
            assert sampler.config.diffusion_steps == 2
            assert manifest["forecaster_training"]["diffusion_steps"] == 3
            plans = manifest["seeds"][str(seed)]
            original = seed_plan(seed)
            for key in original:
                assert (plans[key] != original[key]) == (
                    key in ("test_data", "test_sampling")
                )
            timing = read(folder / "timings.json")
            assert timing["densifier_training_seconds"] == 0
            assert timing["imported_densifier_training_seconds"] == 12.5
            for method in large.METHODS:
                rows = read(folder / f"{method}_validation.json")
                assert len(rows) == 1
                candidate = folder / rows[0]["samples"]
                assert sha256(candidate) == rows[0]["samples_sha256"]
                with np.load(candidate) as samples:
                    actual = evaluate_forecast(samples["samples"], samples["target"])
                    assert all(actual[key] == rows[0][key] for key in actual)
                    np.testing.assert_array_equal(samples["trajectory_ids"], [4, 5])
            assert timing["methods"]["densiffusion"][
                "total_training_seconds"
            ] == pytest.approx(
                timing["methods"]["densiffusion"]["new_training_seconds"] + 12.5
            )
        scores = large.evaluate_run(output)
    finally:
        torch.set_num_threads(previous_threads)
    assert len(generated) == len(config.seeds)
    assert all(p.read_bytes() == value for p, value in protected.items())
    for seed in config.seeds:
        with np.load(output / f"seed_{seed}/samples.npz") as samples:
            for method in large.METHODS:
                assert samples[method].shape == (3, 2, 2, 1)
                assert scores[str(seed)][method] == evaluate_forecast(
                    samples[method], samples["target"]
                )
                inference = read(output / f"seed_{seed}/{method}_inference_audit.json")
                if method == "direct_csdi":
                    assert inference == []
                else:
                    assert len(inference) == 1
                    assert inference[0]["observations_preserved"]
                    assert inference[0]["input_shape"] == list(samples["history"].shape)
                    assert (
                        inference[0]["input_sha256"]
                        == hashlib.sha256(samples["history"].tobytes()).hexdigest()
                    )
                    assert inference[0]["dt"] == config.dt
    with pytest.raises(FileExistsError):
        large.evaluate_run(output)


def test_declared_scientific_budget_and_namespace():
    config = large.LargeForecastConfig()
    assert config.epochs * (config.n_train // config.batch_size) == 9600
    assert list(
        range(
            config.validation_score_interval,
            config.epochs + 1,
            config.validation_score_interval,
        )
    ) == [150, 300, 450, 600]
    assert config.n_histories * config.n_futures == 64
    with pytest.raises(ValueError, match="declared comparison budget"):
        large.LargeForecastConfig(epochs=200)
    with pytest.raises(ValueError, match="separate test_namespace"):
        large.LargeForecastConfig(label="smoke")
    for seed in config.seeds:
        fresh = large.fresh_seed_plan(seed)
        streams = np.random.SeedSequence([seed, 2026102702]).spawn(2)
        assert fresh["test_data"] == int(streams[0].generate_state(1)[0])
        assert fresh["test_sampling"] == int(streams[1].generate_state(1)[0])
