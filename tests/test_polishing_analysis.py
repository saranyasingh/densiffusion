import json
from xml.etree import ElementTree

import numpy as np
import pytest

from densiffusion.metrics import evaluate_forecast
from densiffusion.polishing_analysis import (
    COMPARISONS,
    METHODS,
    analyze_polishing_run,
)


def _write_run(root, seeds=(3, 9), *, auxiliary=True, manifest=True):
    if manifest:
        (root / "config.json").write_text(
            json.dumps(
                {
                    "experiment": {
                        "seeds": list(seeds),
                        "n_test": 4,
                        "context_length": 4,
                        "horizon": 2,
                        "samples": 2,
                        "dt": 1,
                    }
                }
            )
        )
        for marker in ("MEDIATORS_FROZEN.json", "FROZEN.json", "EVALUATED.json"):
            (root / marker).write_text(json.dumps({"wall_seconds": 20.0}))
    for seed in seeds:
        folder = root / f"seed_{seed}"
        folder.mkdir()
        history = np.tile(np.array([1, -2, 3, -1])[None, :, None], (4, 1, 1))
        target = np.tile(np.array([0.0, 2.0])[None, :, None], (4, 1, 1))
        ids = np.array([20, 21, 22, 23])
        np.savez_compressed(
            folder / "test.npz", history=history, target=target, trajectory_ids=ids
        )
        all_scores = {}
        for method in METHODS:
            offsets = {"raw": [1, 1], "learned": [-1, 1], "median": [-0.5, 0.5]}[method]
            samples = (
                target[:, None]
                + np.array(offsets)[None, :, None, None]
                * np.array([1, 2])[None, None, :, None]
            )
            processed = history * {"raw": 1, "learned": 0.5, "median": 0.75}[method]
            np.savez_compressed(
                folder / f"{method}_samples.npz",
                history=history,
                processed_history=processed,
                target=target,
                trajectory_ids=ids,
                samples=samples,
                history_times=np.arange(4),
                future_times=np.arange(4, 6),
            )
            all_scores[method] = evaluate_forecast(samples, target)
            if auxiliary:
                (folder / f"{method}_validation.json").write_text(
                    json.dumps(
                        [
                            {"epoch": 10, "crps": 0.6, "sampling_seconds": 1.0},
                            {"epoch": 50, "crps": 0.4, "sampling_seconds": 2.0},
                            {"epoch": 100, "crps": 0.5, "sampling_seconds": 3.0},
                        ]
                    )
                )
        if auxiliary:
            (folder / "scores.json").write_text(json.dumps(all_scores))
            training = {
                "mediator_training_seconds": 3.0,
                "preprocessing_seconds": dict.fromkeys(METHODS, 1.0),
                "forecaster_training_seconds": dict.fromkeys(METHODS, 10.0),
                "validation_sampling_seconds": dict.fromkeys(METHODS, 6.0),
            }
            evaluation = {
                "test_preprocessing_seconds": dict.fromkeys(METHODS, 2.0),
                "test_inference_seconds": dict.fromkeys(METHODS, 4.0),
            }
            (folder / "timings.json").write_text(json.dumps(training))
            (folder / "evaluation_timings.json").write_text(json.dumps(evaluation))
            (folder / "combined_timings.json").write_text(
                json.dumps({**training, **evaluation})
            )


def _change_archive(path, update):
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    update(arrays)
    np.savez_compressed(path, **arrays)


def test_hand_scored_complete_run_pairs_metrics_and_preserves_sources(tmp_path):
    _write_run(tmp_path)
    original = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    result = analyze_polishing_run(tmp_path, bootstrap_replicates=53, seed=7)
    assert result == json.loads((tmp_path / "analysis.json").read_text())
    assert result["protocol"]["status"] == "complete"
    assert result["phase_wall_seconds"]["EVALUATED.json"] == 20.0
    for i, seed in enumerate((3, 9)):
        entry = result["seeds"][f"seed_{seed}"]
        learned = entry["scores"]["learned"]
        assert learned["aggregate"] == pytest.approx(
            {"crps": 0.75, "mse": 0, "coverage_90": 1, "interval_width_90": 2.7}
        )
        assert learned["per_horizon"]["crps"] == [0.5, 1.0]
        assert learned["per_trajectory"]["crps"] == [0.75] * 4
        assert entry["saved_scores_verified"] is True
        for j, (left, right) in enumerate(COMPARISONS):
            comparison = entry["paired_differences"][f"{left}-minus-{right}"]
            assert comparison["bootstrap_seed"] == 7 + 100 * i + j
            difference = {
                ("learned", "raw"): -0.75,
                ("learned", "median"): 0.375,
                ("median", "raw"): -1.125,
            }[(left, right)]
            assert comparison["crps"]["difference"] == difference
            assert comparison["crps"]["ci95"] == [difference, difference]
        assert entry["validation"]["raw"]["selected_epoch_from_validation"] == 50
        assert entry["validation"]["raw"]["selected_at_final_candidate"] is False
        costs = entry["costs"]
        assert costs["source_files"] == ["combined_timings.json"]
        assert costs["seconds"]["method_total_seconds"] == {
            "raw": 17,
            "learned": 20,
            "median": 17,
        }
    across = result["across_seed_descriptive"]
    assert across["n_seeds"] == 2
    assert across["scores"]["learned"]["crps"]["sample_std"] == 0
    assert across["per_horizon"]["learned"]["crps"][1]["values"] == [1.0, 1.0]
    assert all(path.read_bytes() == content for path, content in original.items())
    assert "already included" in (tmp_path / "analysis.txt").read_text()
    assert len(result["plots"]) == 3
    for name in result["plots"]:
        tree = ElementTree.parse(tmp_path / name)
        assert tree.getroot().tag == "{http://www.w3.org/2000/svg}svg"


def test_separate_frozen_and_evaluation_timings_merge_without_writes(tmp_path):
    _write_run(tmp_path, seeds=(3,))
    folder = tmp_path / "seed_3"
    (folder / "combined_timings.json").unlink()
    frozen = (folder / "timings.json").read_bytes()
    result = analyze_polishing_run(tmp_path, bootstrap_replicates=11)
    assert result["seeds"]["seed_3"]["costs"]["source_files"] == [
        "timings.json",
        "evaluation_timings.json",
    ]
    assert (folder / "timings.json").read_bytes() == frozen


def test_repeated_windows_are_one_bootstrap_unit(tmp_path):
    _write_run(tmp_path, seeds=(3,))
    folder = tmp_path / "seed_3"
    for path in folder.glob("*.npz"):
        _change_archive(
            path, lambda arrays: arrays.update(trajectory_ids=np.array([5, 5, 5, 9]))
        )
    result = analyze_polishing_run(tmp_path, bootstrap_replicates=11)
    entry = result["seeds"]["seed_3"]
    assert entry["n_windows"] == 4
    assert entry["n_trajectories"] == 2
    assert entry["scores"]["learned"]["windows_per_trajectory"] == [3, 1]
    assert entry["paired_differences"]["learned-minus-raw"]["n_trajectories"] == 2


@pytest.mark.parametrize(
    "field",
    ["history", "target", "trajectory_ids", "processed_history", "future_times"],
)
def test_rejects_misaligned_or_invalid_raw_archives(tmp_path, field):
    _write_run(tmp_path, seeds=(3,))

    def change(arrays):
        arrays[field] = arrays[field] + (1 if field != "future_times" else -5)

    _change_archive(tmp_path / "seed_3/raw_samples.npz", change)
    with pytest.raises(ValueError, match="raw_samples.npz"):
        analyze_polishing_run(tmp_path, bootstrap_replicates=5)
    assert not (tmp_path / "analysis.json").exists()


@pytest.mark.parametrize("invalid", ["nonfinite", "dimensions"])
def test_rejects_nonfinite_samples_and_bad_dimensions(tmp_path, invalid):
    _write_run(tmp_path, seeds=(3,))
    path = tmp_path / "seed_3/learned_samples.npz"
    if invalid == "nonfinite":
        _change_archive(path, lambda arrays: arrays["samples"].fill(np.nan))
        message = "finite real numbers"
    else:
        _change_archive(
            path, lambda arrays: arrays.update(samples=arrays["samples"][:, :1])
        )
        message = "Expected shape"
    with pytest.raises(ValueError, match=message):
        analyze_polishing_run(tmp_path, bootstrap_replicates=5)


def test_missing_seed_or_method_and_stale_saved_scores_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="No seed"):
        analyze_polishing_run(tmp_path)
    _write_run(tmp_path, seeds=(3,))
    config = json.loads((tmp_path / "config.json").read_text())
    config["experiment"]["seeds"].append(9)
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r"missing=\[9\]"):
        analyze_polishing_run(tmp_path, bootstrap_replicates=5)
    config["experiment"]["seeds"] = [3]
    (tmp_path / "config.json").write_text(json.dumps(config))
    path = tmp_path / "seed_3/scores.json"
    scores = json.loads(path.read_text())
    scores["raw"]["crps"] += 1
    path.write_text(json.dumps(scores))
    with pytest.raises(ValueError, match="raw.crps disagrees"):
        analyze_polishing_run(tmp_path, bootstrap_replicates=5)
    (tmp_path / "seed_3/median_samples.npz").unlink()
    with pytest.raises(FileNotFoundError, match="median_samples.npz"):
        analyze_polishing_run(tmp_path, bootstrap_replicates=5)


def test_optional_metadata_absence_is_reported_without_claiming_complete(tmp_path):
    _write_run(tmp_path, seeds=(3,), auxiliary=False)
    result = analyze_polishing_run(tmp_path, bootstrap_replicates=5)
    assert result["protocol"]["status"] == "incomplete"
    assert len(result["seeds"]["seed_3"]["missing_artifacts"]) == 5
    assert result["costs_across_seed_descriptive"]["n_available_seeds"] == 0
    assert len(result["plots"]) == 2
    (tmp_path / "config.json").unlink()
    result = analyze_polishing_run(tmp_path, bootstrap_replicates=5)
    assert result["protocol"]["status"] == "unverified_without_manifest"


def test_invalid_validation_cadence_and_negative_timings_rejected(tmp_path):
    _write_run(tmp_path, seeds=(3,))
    folder = tmp_path / "seed_3"
    path = folder / "raw_validation.json"
    saved = path.read_text()
    rows = json.loads(saved)
    rows[1]["epoch"] = rows[0]["epoch"]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unique and increasing"):
        analyze_polishing_run(tmp_path, bootstrap_replicates=5)
    path.write_text(saved)
    path = folder / "combined_timings.json"
    timings = json.loads(path.read_text())
    timings["test_inference_seconds"]["median"] = -1
    path.write_text(json.dumps(timings))
    with pytest.raises(ValueError, match="test_inference_seconds.median"):
        analyze_polishing_run(tmp_path, bootstrap_replicates=5)
