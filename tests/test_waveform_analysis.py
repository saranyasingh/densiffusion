import json
from xml.etree import ElementTree

import numpy as np
import pytest

from densiffusion.metrics import evaluate_forecast
from densiffusion.waveform_analysis import (
    METHODS,
    analyze_run,
    forecast_score_arrays,
    paired_bootstrap,
    summarize_forecasts,
)


def test_score_arrays_match_pairwise_crps_and_shared_metric_definitions():
    rng = np.random.default_rng(42)
    target = rng.normal(size=(4, 3, 2))
    samples = rng.normal(size=(4, 7, 3, 2))
    scores = forecast_score_arrays(samples, target)
    expected_crps = np.abs(samples - target[:, None]).mean(axis=1) - 0.5 * np.abs(
        samples[:, :, None] - samples[:, None, :]
    ).mean(axis=(1, 2))
    np.testing.assert_allclose(scores["crps"], expected_crps)
    assert {
        metric: float(value.mean()) for metric, value in scores.items()
    } == pytest.approx(evaluate_forecast(samples, target))


def test_hand_known_per_horizon_and_trajectory_scores_retain_correlated_windows():
    target = np.zeros((3, 2, 1))
    samples = np.array([[0.0, 2.0], [2.0, 4.0], [10.0, 12.0]])[:, None, :, None]
    summary = summarize_forecasts(samples, target, np.array([8, 8, 20]))
    assert summary["trajectory_ids"] == [8, 20]
    assert summary["windows_per_trajectory"] == [2, 1]
    assert summary["per_trajectory"]["crps"] == [2.0, 11.0]
    assert summary["per_trajectory"]["mse"] == [6.0, 122.0]
    assert summary["per_trajectory"]["coverage_90"] == [0.25, 0.0]
    assert summary["per_horizon"]["crps"] == [4.0, 6.0]
    assert summary["aggregate"]["crps"] == 5.0
    assert summary["trajectory_weighted"]["crps"] == 6.5
    assert summary["aggregate"]["interval_width_90"] == 0.0


def test_paired_bootstrap_resamples_differences_not_independent_methods():
    left = np.arange(1, 11) * 100.0
    right = left - 2.0
    result = paired_bootstrap(left, right, replicates=101, seed=3, chunk_size=13)
    assert result["difference"] == 2.0
    assert result["ci95"] == [2.0, 2.0]
    assert result["n_trajectories"] == 10
    assert result == paired_bootstrap(
        left, right, replicates=101, seed=3, chunk_size=101
    )


def test_bootstrap_multimetric_matches_explicit_paired_resampling():
    left = np.array([[1.0, 2.0], [4.0, 8.0], [8.0, 16.0]])
    right = np.ones((3, 2))
    indices = np.random.default_rng(9).integers(3, size=(57, 3))
    expected = np.quantile((left - right)[indices].mean(axis=1), [0.025, 0.975], axis=0)
    actual = paired_bootstrap(left, right, replicates=57, seed=9, chunk_size=5)
    np.testing.assert_allclose(actual["ci95"], expected)


def test_saved_sample_analysis_keeps_seed_variation_separate_and_writes_plots(tmp_path):
    for seed in (7, 8):
        folder = tmp_path / f"seed_{seed}"
        folder.mkdir()
        target = np.zeros((3, 2, 1))
        history = np.ones((3, 4, 1))
        np.savez_compressed(
            folder / "samples.npz",
            target=target,
            history=history,
            history_times=np.arange(4) * 0.1,
            future_times=np.arange(4, 6) * 0.1,
            trajectory_ids=np.array([100, 101, 102]),
            **{
                method: np.ones((3, 2, 2, 1)) * (index + seed - 7)
                for index, method in enumerate(METHODS)
            },
        )
        (folder / "timings.json").write_text(json.dumps({"training_seconds": 12.0}))
    result = analyze_run(tmp_path, bootstrap_replicates=30, seed=100)
    assert result == json.loads((tmp_path / "analysis.json").read_text())
    assert result["protocol"]["status"] == "unverified_without_manifest"
    assert result["protocol"]["sample_seed_ids"] == [7, 8]
    assert result["seeds"]["seed_7"]["timings"]["training_seconds"] == 12.0
    across = result["across_seed_descriptive"]
    assert across["n_seeds"] == 2
    assert across["scores"]["direct_csdi"]["crps"]["values"] == [0.0, 1.0]
    difference = across["paired_differences"]["densiffusion-minus-direct_csdi"]["crps"]
    assert difference["values"] == [2.0, 2.0]
    assert difference["sample_std"] == 0.0
    assert len(result["plots"]) == 2
    for plot in result["plots"]:
        root = ElementTree.parse(tmp_path / plot).getroot()
        assert root.tag.endswith("svg")
        assert root.findall(".//{http://www.w3.org/2000/svg}polygon")


def test_analysis_rejects_missing_samples_and_invalid_trajectory_ids(tmp_path):
    with pytest.raises(ValueError, match="No seed"):
        analyze_run(tmp_path)
    with pytest.raises(ValueError, match="trajectory_ids"):
        summarize_forecasts(np.zeros((2, 3, 4, 1)), np.zeros((2, 4, 1)), [1])


def test_training_summary_tracks_validation_selection_without_convergence_claim(
    tmp_path,
):
    from densiffusion.waveform_analysis import _training_summary, _validation_plot

    rows = [
        {
            "method": "direct_csdi",
            "epoch": epoch,
            "train_loss": loss,
            "selection_metric": "original_grid_crps",
            "selection_score": score,
        }
        for epoch, loss, score in ((1, 1.0, 0.7), (2, 0.5, 0.4), (3, 0.3, 0.5))
    ]
    (tmp_path / "training.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "direct_csdi_validation.json").write_text(
        json.dumps(
            [{"epoch": row["epoch"], "crps": row["selection_score"]} for row in rows]
        )
    )
    summary = _training_summary(tmp_path)
    model = summary["models"]["direct_csdi"]
    assert model["selected_epoch"] == 2
    assert model["selected_at_final_epoch"] is False
    assert model["final_minus_previous_selection_score"] == pytest.approx(0.1)
    assert model["minimum_original_grid_validation_crps_epoch"] == 2
    result = {"seeds": {"seed_7": {"training": summary}}}
    destination = tmp_path / "validation.svg"
    assert _validation_plot(result, destination)
    root = ElementTree.parse(destination).getroot()
    assert len(root.findall(".//{http://www.w3.org/2000/svg}circle")) == 3


@pytest.mark.parametrize(
    ("observed", "evaluated", "status", "missing", "unexpected"),
    [
        ([7], False, "incomplete", [8], []),
        ([7, 8], False, "incomplete", [], []),
        ([7, 8], True, "complete", [], []),
        ([7, 8, 9], True, "incomplete", [], [9]),
        ([7], True, "incomplete", [8], []),
    ],
)
def test_protocol_status_requires_configured_seeds_and_evaluation_marker(
    tmp_path, observed, evaluated, status, missing, unexpected
):
    from densiffusion.waveform_analysis import _protocol_status

    (tmp_path / "config.json").write_text(json.dumps({"experiment": {"seeds": [8, 7]}}))
    if evaluated:
        (tmp_path / "EVALUATED.json").write_text("{}")
    actual = _protocol_status(
        tmp_path, [tmp_path / f"seed_{seed}" for seed in observed]
    )
    assert actual["status"] == status
    assert actual["configured_seed_ids"] == [7, 8]
    assert actual["sample_seed_ids"] == observed
    assert actual["missing_seed_ids"] == missing
    assert actual["unexpected_seed_ids"] == unexpected
    assert actual["evaluated_marker_present"] is evaluated
