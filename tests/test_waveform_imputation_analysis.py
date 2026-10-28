import json
from xml.etree import ElementTree

import numpy as np
import pytest

from densiffusion.metrics import crps
from densiffusion.waveform_imputation_analysis import (
    analyze_imputation_run,
    analyze_transfer_run,
    summarize_imputations,
    validation_gate,
)


def test_hidden_only_metrics_match_hand_ensemble_and_existing_crps():
    target = np.array([[100.0, 0, 100, 2, 100], [100, 2, 100, 4, 100]])[..., None]
    samples = np.repeat(target[:, None], 2, axis=1)
    samples[0, :, 1, 0] = [-1, 1]
    samples[0, :, 3, 0] = [1, 3]
    samples[1, :, 1, 0] = [0, 0]
    samples[1, :, 3, 0] = [2, 2]
    mask = np.array([False, True, False, True, False])
    result = summarize_imputations(samples, target, mask)
    assert result["aggregate"] == pytest.approx(
        {"crps": 1.25, "mse": 2, "coverage_90": 0.5, "interval_width_90": 0.9}
    )
    assert result["aggregate"]["crps"] == crps(samples, target, mask=mask)
    assert result["per_trajectory"]["crps"] == [0.5, 2.0]
    assert result["per_time"]["crps"] == [None, 1.25, None, 1.25, None]
    assert result["observations_preserved"] is True
    # Known-point errors cannot enter hidden-point metrics, but are audited.
    samples[:, :, 0] += 1_000_000
    changed = summarize_imputations(samples, target, mask)
    assert changed["aggregate"] == result["aggregate"]
    assert changed["observations_preserved"] is False


def test_correlated_windows_group_all_hidden_entries_with_correct_denominator():
    target = np.zeros((3, 5, 1))
    mask = np.zeros_like(target, dtype=bool)
    mask[:, 1] = True
    mask[1:, 3] = True
    samples = np.where(mask[:, None], np.array([1, 3, 10])[:, None, None, None], 0)
    result = summarize_imputations(samples, target, mask, [7, 7, 8])
    assert result["trajectory_ids"] == [7, 8]
    assert result["windows_per_trajectory"] == [2, 1]
    assert result["hidden_entries_per_trajectory"] == [3, 2]
    assert result["per_trajectory"]["crps"] == pytest.approx([7 / 3, 10])
    assert result["aggregate"]["crps"] == pytest.approx(27 / 5)
    assert result["trajectory_weighted"]["crps"] == pytest.approx((7 / 3 + 10) / 2)


def _gate_entries(learned, linear, upper=-0.001):
    return {
        f"seed_{index + 1}": {
            "scores": {
                "learned": {"trajectory_weighted": {"crps": left}},
                "linear": {"trajectory_weighted": {"crps": right}},
            },
            "learned_minus_linear": {
                "crps": {"difference": left - right, "ci95": [-1, upper]}
            },
        }
        for index, (left, right) in enumerate(zip(learned, linear, strict=True))
    }


def test_gate_uses_ratio_of_seed_means_not_mean_of_seed_relative_improvements():
    entries = _gate_entries([0.8, 99, 99], [1, 100, 100])
    result = validation_gate(entries)
    assert result["relative_crps_reduction"] == pytest.approx(1 - 198.8 / 201)
    assert result["criteria"]["learned_crps_lower_in_every_seed"] is True
    assert result["criteria"]["every_seed_ci95_upper_below_zero"] is True
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("learned", "linear", "upper", "expected", "passed"),
    [
        ([0.5, 0.6, 0.7], [1, 1, 1], -0.1, [1, 2, 3], True),
        ([0.5, 0.6], [1, 1], -0.1, [1, 2], False),
        ([0.5, 0.6, 0.7], [1, 1, 1], 0, [1, 2, 3], False),
        ([0.5, 0.6, 1.1], [1, 1, 1], -0.1, [1, 2, 3], False),
        ([0.99, 0.99, 0.99], [1, 1, 1], -0.001, [1, 2, 3], False),
        ([0.5, 0.6, 0.7], [1, 1, 1], -0.1, [1, 2, 4], False),
        ([0, 0, 0], [0, 0, 0], -0.1, [1, 2, 3], False),
    ],
)
def test_gate_requires_all_predeclared_conditions(
    learned, linear, upper, expected, passed
):
    result = validation_gate(
        _gate_entries(learned, linear, upper), expected_seeds=expected
    )
    assert result["passed"] is passed
    assert result["transfer_scores_used"] is False


def _write_samples(path, *, learned_error=0.2, linear_error=1.0, origin=0):
    target = np.broadcast_to(np.array([1, 0, 2, 0, 3])[None, :, None], (4, 5, 1)).copy()
    mask = np.array([False, True, False, True, False])
    np.savez_compressed(
        path,
        target=target,
        mask=mask,
        times=origin + np.arange(5) * 0.05,
        trajectory_ids=np.array([10, 11, 12, 13]),
        learned=np.repeat(target[:, None], 4, axis=1)
        + mask[None, None, :, None] * learned_error,
        linear=target[:, None] + mask[None, None, :, None] * linear_error,
    )


def _write_attempt(root):
    (root / "config.json").write_text(json.dumps({"experiment": {"seeds": [1, 2, 3]}}))
    for seed in (1, 2, 3):
        folder = root / f"seed_{seed}"
        folder.mkdir()
        _write_samples(folder / "validation_samples.npz")
        records = [
            {"epoch": 1, "train_loss": 1.0, "validation_loss": 0.9},
            {
                "epoch": 2,
                "train_loss": 0.8,
                "validation_loss": 0.7,
                "selection_score": 0.3,
            },
            {
                "epoch": 3,
                "train_loss": 0.7,
                "validation_loss": 0.6,
                "selection_score": 0.2,
            },
        ]
        (folder / "training.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records)
        )


def test_analysis_recomputes_scores_groups_bootstrap_and_preserves_input_archives(
    tmp_path,
):
    _write_attempt(tmp_path)
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*.npz")}
    result = analyze_imputation_run(tmp_path, bootstrap_replicates=99, seed=41)
    assert result == json.loads((tmp_path / "imputation_analysis.json").read_text())
    assert result["validation_gate"]["passed"] is True
    assert result["validation_gate"]["relative_crps_reduction"] == pytest.approx(0.8)
    assert result["tasks"] == ["validation"]
    assert "transfer" not in result
    assert "transfer_history" not in result
    for name, entry in result["validation"]["seeds"].items():
        assert entry["scores"]["learned"]["aggregate"]["crps"] == pytest.approx(0.2)
        assert entry["learned_minus_linear"]["crps"]["difference"] == pytest.approx(
            -0.8
        )
        assert entry["learned_minus_linear"]["crps"]["ci95"] == pytest.approx(
            [-0.8, -0.8]
        )
        assert entry["bootstrap_seed"] == 41 + 100 * (int(name[-1]) - 1)
        assert result["training"][name]["selected_epoch"] == 3
    assert len(result["plots"]) == 3
    for plot in result["plots"]:
        tree = ElementTree.parse(tmp_path / plot)
        assert tree.getroot().tag == "{http://www.w3.org/2000/svg}svg"
    assert (tmp_path / "imputation_report.txt").read_text().startswith("Fresh waveform")
    assert all(path.read_bytes() == value for path, value in originals.items())


def test_separate_transfer_analysis_leaves_frozen_validation_analysis_unchanged(
    tmp_path,
):
    _write_attempt(tmp_path)
    validation = analyze_imputation_run(tmp_path, bootstrap_replicates=20)
    frozen_bytes = (tmp_path / "imputation_analysis.json").read_bytes()
    (tmp_path / "DENSIFIERS_FROZEN.json").write_text("{}")
    for seed in (1, 2, 3):
        folder = tmp_path / f"seed_{seed}"
        _write_samples(folder / "transfer_samples.npz", learned_error=2.0)
        _write_samples(
            folder / "transfer_future_samples.npz", learned_error=3.0, origin=2.4
        )
    transfer = analyze_transfer_run(tmp_path, bootstrap_replicates=20)
    assert transfer["tasks"] == ["transfer_history", "transfer_future"]
    assert "validation_gate" not in transfer
    assert validation["validation_gate"]["passed"] is True
    assert len(transfer["plots"]) == 4
    assert transfer["transfer_history"]["across_seed_descriptive"]["scores"]["learned"][
        "crps"
    ]["mean"] == pytest.approx(2.0)
    assert transfer["transfer_future"]["across_seed_descriptive"]["scores"]["learned"][
        "crps"
    ]["mean"] == pytest.approx(3.0)
    assert (tmp_path / "imputation_analysis.json").read_bytes() == frozen_bytes
    with pytest.raises(FileExistsError, match="Frozen"):
        analyze_imputation_run(tmp_path)


def test_analysis_rejects_changed_observations_and_no_samples(tmp_path):
    with pytest.raises(ValueError, match="No seed"):
        analyze_imputation_run(tmp_path)
    with pytest.raises(ValueError, match="No transfer"):
        analyze_transfer_run(tmp_path)
    _write_attempt(tmp_path)
    path = tmp_path / "seed_1" / "validation_samples.npz"
    with np.load(path) as archive:
        data = dict(archive)
    data["learned"][:, :, 0] += 1
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="changed original observations"):
        analyze_imputation_run(tmp_path, bootstrap_replicates=20)


@pytest.mark.parametrize("mask", [np.zeros(5, bool), np.ones(5, bool), np.ones(5)])
def test_metrics_reject_invalid_masks(mask):
    with pytest.raises(ValueError):
        summarize_imputations(np.zeros((2, 4, 5, 1)), np.zeros((2, 5, 1)), mask)
