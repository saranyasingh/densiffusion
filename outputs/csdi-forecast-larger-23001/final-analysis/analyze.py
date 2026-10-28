"""Paired trajectory bootstrap and analytic GP audit of saved forecasts."""

import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from densiffusion.gp.posterior import gp_posterior
from densiffusion.metrics import evaluate_forecast

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
METHODS = ("densiffusion", "linear_densification", "direct_csdi")
PAIRS = ((0, 2), (1, 2), (0, 1))
REPLICATES = 50_000
SEED = 23031


def hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(RUN.iterdir())
        if path.is_file()
    }


def point_crps(samples, target):
    """Empirical CRPS at each window, horizon, channel (same estimator as repo)."""
    count = samples.shape[1]
    errors = np.sort(samples - target[:, None], axis=1)
    weights = (2 * np.arange(1, count + 1) - count - 1) / count**2
    return np.abs(errors).mean(axis=1) - np.sum(
        errors * weights[None, :, None, None], axis=1
    )


def write_json(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    before = hashes()
    archived = json.loads((RUN / "sha256.json").read_text())
    assert all(before[name] == digest for name, digest in archived.items())
    config = json.loads((RUN / "config.json").read_text())
    original_scores = json.loads((RUN / "assessment_scores.json").read_text())
    with np.load(RUN / "samples.npz", allow_pickle=False) as source:
        saved = {name: source[name] for name in source.files}
    with np.load(RUN / "baseline_samples.npz", allow_pickle=False) as source:
        gp = {name: source[name] for name in source.files}
    target, history = saved["target"], saved["history"]
    np.testing.assert_array_equal(gp["target"], target)
    np.testing.assert_array_equal(gp["history"], history)
    test = config["splits"]["test"]
    trajectory_ids = np.asarray(test["window_trajectory_ids"])
    unique_ids, counts = np.unique(trajectory_ids, return_counts=True)
    np.testing.assert_array_equal(unique_ids, test["trajectory_ids"])
    assert len(unique_ids) == 155 and len(trajectory_ids) == len(target) == 310
    np.testing.assert_array_equal(counts, np.full(155, 2))
    groups = [np.flatnonzero(trajectory_ids == key) for key in unique_ids]
    starts = np.asarray(test["window_start_indices"])
    for indices in groups:
        np.testing.assert_array_equal(starts[indices], [0, 12])

    window_crps = np.column_stack(
        [point_crps(saved[name], target).mean(axis=(1, 2)) for name in METHODS]
    )
    # Independent O(K^2) check of the sorted-sample CRPS formula on two windows.
    for index, name in enumerate(METHODS):
        draws = saved[name][:2]
        direct = np.abs(draws - target[:2, None]).mean(axis=1) - 0.5 * np.abs(
            draws[:, :, None] - draws[:, None, :]
        ).mean(axis=(1, 2))
        np.testing.assert_allclose(
            direct, point_crps(draws, target[:2]), rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            window_crps[:, index].mean(), original_scores[name]["crps"], atol=1e-14
        )
        assert evaluate_forecast(saved[name], target) == original_scores[name]
    cluster_crps = np.stack([window_crps[indices].mean(axis=0) for indices in groups])
    cluster_differences = np.column_stack(
        [cluster_crps[:, left] - cluster_crps[:, right] for left, right in PAIRS]
    )

    posterior_mean, covariance = gp_posterior(
        saved["history_times"][0],
        history[:, :, 0],
        saved["future_times"][0],
        **config["gp"],
    )
    mu = posterior_mean[..., None]
    np.testing.assert_array_equal(mu, gp["posterior_mean"])
    np.testing.assert_array_equal(covariance, gp["posterior_covariance"])
    sigma = np.sqrt(np.diag(covariance))[None, :, None]
    z = NormalDist().inv_cdf(0.95)
    lower, upper = mu - z * sigma, mu + z * sigma
    analytic_covered = (lower <= target) & (target <= upper)
    empirical_lower, empirical_upper = np.quantile(gp["exact_gp"], [0.05, 0.95], axis=1)
    empirical_covered = (empirical_lower <= target) & (target <= empirical_upper)
    cluster_coverage = np.stack(
        [
            [analytic_covered[indices].mean(), empirical_covered[indices].mean()]
            for indices in groups
        ]
    )

    # Use identical trajectory selections for every contrast and coverage estimate.
    rng = np.random.default_rng(SEED)
    bootstrap_differences = np.empty((REPLICATES, len(PAIRS)))
    bootstrap_coverage = np.empty((REPLICATES, 2))
    for start in range(0, REPLICATES, 2000):
        stop = min(start + 2000, REPLICATES)
        chosen = rng.integers(len(unique_ids), size=(stop - start, len(unique_ids)))
        bootstrap_differences[start:stop] = cluster_differences[chosen].mean(axis=1)
        bootstrap_coverage[start:stop] = cluster_coverage[chosen].mean(axis=1)
    comparisons = {}
    for index, (left, right) in enumerate(PAIRS):
        comparisons[f"{METHODS[left]} minus {METHODS[right]}"] = {
            "crps_difference": float(cluster_differences[:, index].mean()),
            "percentile_95_interval": np.quantile(
                bootstrap_differences[:, index], [0.025, 0.975]
            ).tolist(),
        }

    sample_mean = gp["exact_gp"].mean(axis=1)
    sampling_error = sample_mean - mu
    forecast_error = mu - target
    analytic_mse = float(np.square(forecast_error).mean())
    empirical_mse = float(np.square(sample_mean - target).mean())
    sampling_mse = float(np.square(sampling_error).mean())
    cross_term = float(2 * (sampling_error * forecast_error).mean())
    np.testing.assert_allclose(
        empirical_mse, analytic_mse + sampling_mse + cross_term, atol=1e-14
    )
    draws = gp["exact_gp"].shape[1]
    assert draws == 32
    analytic = {
        "coverage_90": float(analytic_covered.mean()),
        "interval_width_90": float((upper - lower).mean()),
        "mse_conditional_mean": analytic_mse,
        "coverage_trajectory_bootstrap_95_interval": np.quantile(
            bootstrap_coverage[:, 0], [0.025, 0.975]
        ).tolist(),
    }
    empirical = evaluate_forecast(gp["exact_gp"], target)
    assert empirical == original_scores["exact_gp"]
    empirical["coverage_trajectory_bootstrap_95_interval"] = np.quantile(
        bootstrap_coverage[:, 1], [0.025, 0.975]
    ).tolist()
    result = {
        "bootstrap": {
            "replicates": REPLICATES,
            "seed": SEED,
            "method": "paired nonparametric trajectory bootstrap; percentile intervals",
            "independent_units": 155,
            "windows_per_unit": 2,
            "estimand": "mean empirical 32-draw CRPS difference, left minus right",
            "scope": (
                "test-trajectory uncertainty conditional on this training run and "
                "the saved forecast draws; no retraining or ensemble resampling"
            ),
            "comparisons": comparisons,
        },
        "gp": {
            "parameters": config["gp"],
            "normal_95_percentile": z,
            "analytic": analytic,
            "empirical_32_draws": empirical,
            "mean_mse_decomposition": {
                "analytic_conditional_mean_mse": analytic_mse,
                "sample_mean_mse": empirical_mse,
                "realized_mse_difference": empirical_mse - analytic_mse,
                "sample_mean_error_relative_to_analytic_mean_mse": sampling_mse,
                "realized_cross_term": cross_term,
                "expected_sampling_mse_penalty": float(
                    np.diag(covariance).mean() / draws
                ),
                "identity": (
                    "sample_mean_mse = analytic_mean_mse + sampling_mse + cross_term"
                ),
                "expectation": (
                    "conditional on these histories and targets, independent posterior "
                    "sampling adds average posterior variance / 32 in expectation; "
                    "the cross term has zero expectation"
                ),
            },
        },
        "checks": {
            "two_windows_kept_together_by_recorded_trajectory_id": True,
            "original_crps_and_gp_scores_recomputed": True,
            "crps_formula_matches_pairwise_definition": True,
            "gp_posterior_recomputed_from_generator_parameters": True,
            "mse_decomposition_verified": True,
        },
    }
    assert before == hashes(), "An original run artifact changed"
    write_json("results.json", result)
    write_json("original_artifact_hashes.json", before)
    with (OUT / "bootstrap_samples.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            trajectory_ids=unique_ids,
            methods=np.array(METHODS),
            window_trajectory_ids=trajectory_ids,
            window_crps=window_crps,
            trajectory_crps=cluster_crps,
            trajectory_differences=cluster_differences,
            bootstrap_differences=bootstrap_differences,
            bootstrap_gp_coverage=bootstrap_coverage,
        )
    with (OUT / "gp_analytic.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            mean=mu,
            covariance=covariance,
            lower_05=lower,
            upper_95=upper,
            target=target,
            sample_mean_32=sample_mean,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
