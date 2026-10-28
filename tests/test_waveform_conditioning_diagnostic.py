import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from densiffusion.architectures.csdi import CSDIDenoiser  # noqa: E402
from densiffusion.models import ImputationModel  # noqa: E402
from densiffusion.training import TrainingConfig  # noqa: E402
from densiffusion.waveform_experiment import model_hash, seed_plan  # noqa: E402
from densiffusion.waveform_imputation import sha256  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "conditioning_diagnostic",
    Path(__file__).resolve().parents[1] / "scripts/diagnose_waveform_imputation.py",
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_failed_gate_and_pretransfer_state_required(tmp_path):
    with pytest.raises(ValueError, match="All training must finish"):
        diagnostic.verify_failed_gate(tmp_path)
    write_json(tmp_path / "TRAINED.json", {})
    write_json(
        tmp_path / "imputation_analysis.json", {"validation_gate": {"passed": True}}
    )
    with pytest.raises(ValueError, match="restricted to failed gates"):
        diagnostic.verify_failed_gate(tmp_path)
    write_json(
        tmp_path / "imputation_analysis.json", {"validation_gate": {"passed": False}}
    )
    write_json(tmp_path / "TRANSFER.json", {})
    with pytest.raises(ValueError, match="precede all fine-grid"):
        diagnostic.verify_failed_gate(tmp_path)


def test_saved_conditioning_cases_keep_own_targets_and_match_permutation(tmp_path):
    seed = 17
    source = tmp_path / f"seed_{seed}"
    source.mkdir()
    destination = tmp_path / "diagnostic"
    destination.mkdir()
    architecture = {
        "n_channels": 1,
        "hidden_channels": 4,
        "residual_layers": 1,
        "attention_heads": 2,
        "diffusion_embedding_dim": 4,
        "time_embedding_dim": 4,
        "feature_embedding_dim": 2,
        "feedforward_dim": 4,
        "dropout": 0.0,
    }
    seeds = seed_plan(seed)
    torch.manual_seed(seeds["densifier_initialization"])
    model = ImputationModel(CSDIDenoiser(**architecture))
    initial_hash = model_hash(model)
    torch.save(
        {
            "epoch": 1,
            "seed": seeds["densifier_training"],
            "config": asdict(TrainingConfig(diffusion_steps=2)),
            "model_state_dict": model.state_dict(),
        },
        source / "densifier.pt",
    )
    values = np.arange(64 * 33, dtype=float).reshape(64, 33, 1) / 1000
    np.savez_compressed(
        source / "train_validation.npz",
        train=values[:32],
        validation=values[32:],
        train_trajectory_ids=np.arange(32),
        validation_trajectory_ids=np.arange(32, 64),
    )
    manifest = {
        "architecture": architecture,
        "experiment": {"device": "cpu", "threads": 1, "sample_batch_size": 256},
        "data_sha256": {str(seed): sha256(source / "train_validation.npz")},
        "recovered_gp_recipe": {"gp_weights_reused": False},
    }
    write_json(tmp_path / "config.json", manifest)
    write_json(tmp_path / "TRAINED.json", {})
    write_json(
        tmp_path / "imputation_analysis.json", {"validation_gate": {"passed": False}}
    )
    write_json(source / "seeds.json", seeds)
    write_json(
        source / "summary.json",
        {
            "selected_epoch": 1,
            "initialization_sha256": initial_hash,
            "gp_weights_reused": False,
        },
    )
    old_threads = torch.get_num_threads()
    try:
        _, report = diagnostic.diagnose_seed((tmp_path, destination, seed, manifest))
    finally:
        torch.set_num_threads(old_threads)
    for index, split in enumerate(("train", "validation")):
        with np.load(destination / source.name / f"{split}_samples.npz") as saved:
            np.testing.assert_array_equal(
                saved["target"], values[index * 32 : (index + 1) * 32]
            )
            np.testing.assert_array_equal(
                saved["permuted_observed"], saved["observed"][saved["permutation"]]
            )
            for name, observations in (
                ("learned_correct", saved["observed"]),
                ("learned_permuted", saved["permuted_observed"]),
            ):
                assert saved[name].shape == (32, 64, 33, 1)
                np.testing.assert_array_equal(
                    saved[name][:, :, ::2],
                    np.broadcast_to(observations[:, None], (32, 64, 17, 1)),
                )
            assert np.all(saved["permutation"] != np.arange(32))
        assert report["splits"][split]["scores_recomputed_from_saved_samples"]
