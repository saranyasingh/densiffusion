"""Audit and score this fixed development run; never modifies training artifacts."""

import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.conditioning import prepare_observations
from densiffusion.gp.posterior import gp_posterior
from densiffusion.metrics import evaluate_forecast
from densiffusion.models import ImputationModel
from densiffusion.pipeline import forecast
from densiffusion.samplers import DDPMForecastSampler, DDPMSampler
from densiffusion.training import TrainingConfig

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
GP_SEED = 23021
N_DRAWS = 32


def save_json(name, value):
    with (RUN / name).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    started = perf_counter()
    config = json.loads((RUN / "config.json").read_text())
    scores = json.loads((RUN / "scores.json").read_text())
    torch.set_num_threads(config["threads"])
    spec = importlib.util.spec_from_file_location(
        "example", ROOT / "scripts/train_csdi_forecast.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    args = SimpleNamespace(
        **config["dataset"],
        dt=config["original_dt"],
        seed=config["seeds"]["data"],
        context_length=config["context_length"],
        horizon=config["horizon"],
        stride=config["stride"],
    )
    windows, splits = example.make_dataset(args)
    assert splits == config["splits"]
    ids = [
        set(splits[name]["trajectory_ids"]) for name in ("train", "validation", "test")
    ]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i))
    assert set.union(*ids) == set(range(config["dataset"]["n_series"]))
    with np.load(RUN / "samples.npz", allow_pickle=False) as source:
        saved = {key: source[key].copy() for key in source.files}
    history, target = saved["history"], saved["target"]
    np.testing.assert_array_equal(history, windows[2][0])
    np.testing.assert_array_equal(target, windows[2][1])
    expected = (len(target), N_DRAWS, *target.shape[1:])
    for method in example.METHODS:
        assert saved[method].shape == expected
        assert np.isfinite(saved[method]).all()
        assert evaluate_forecast(saved[method], target) == scores[method]

    # Independent analytic cross-check of the noiseless exponential-GP posterior.
    gp = config["gp"]
    assert gp["kernel"] == "exponential" and target.shape[2] == 1
    tau = saved["future_times"][0] - saved["history_times"][0, -1]
    np.testing.assert_allclose(
        saved["future_times"] - saved["history_times"][:, -1:],
        np.broadcast_to(tau, saved["future_times"].shape),
        atol=1e-14,
    )
    posterior_mean, covariance = gp_posterior(
        saved["history_times"][0], history[:, :, 0], saved["future_times"][0], **gp
    )
    decay = np.exp(-tau / gp["lengthscale"])
    analytic_mean = gp["mean"] + (history[:, -1, 0] - gp["mean"])[:, None] * decay
    analytic_cov = gp["variance"] * (
        np.exp(-np.abs(tau[:, None] - tau[None, :]) / gp["lengthscale"])
        - decay[:, None] * decay[None, :]
    )
    np.testing.assert_allclose(posterior_mean, analytic_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(covariance, analytic_cov, rtol=1e-12, atol=1e-12)
    before = perf_counter()
    rng = np.random.default_rng(GP_SEED)
    gp_samples = (
        posterior_mean[:, None]
        + rng.standard_normal(expected[:-1]) @ np.linalg.cholesky(covariance).T
    )[..., None]
    gp_seconds = perf_counter() - before
    persistence = np.broadcast_to(history[:, None, -1:, :], expected).copy()
    baselines = {"persistence": persistence, "exact_gp": gp_samples}
    for method, samples in baselines.items():
        assert samples.shape == expected and np.isfinite(samples).all()
        scores[method] = evaluate_forecast(samples, target)
    with (RUN / "baseline_samples.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            history=history,
            target=target,
            history_times=saved["history_times"],
            future_times=saved["future_times"],
            posterior_mean=posterior_mean[..., None],
            posterior_covariance=covariance,
            **baselines,
        )
    with np.load(RUN / "baseline_samples.npz", allow_pickle=False) as source:
        np.testing.assert_array_equal(source["target"], target)
        for method in baselines:
            assert evaluate_forecast(source[method], source["target"]) == scores[method]
    save_json(
        "baseline_config.json",
        {
            "gp": gp,
            "gp_seed": GP_SEED,
            "draws": N_DRAWS,
            "conditioning": (
                "noiseless observed history only; recorded generator parameters, "
                "no fitting"
            ),
            "persistence": (
                "32 copies of the last observed value at each forecast horizon"
            ),
            "gp_sampling_seconds": gp_seconds,
            "gp_posterior_checked_against_analytic_exponential_markov_formula": True,
        },
    )

    # Reload the actual selected weights; reproduce all scored forecasts while
    # checking exact preservation at every sampler output, including histories.
    checkpoint_info = {}

    def load_model(filename):
        checkpoint = torch.load(RUN / filename, map_location="cpu", weights_only=True)
        model = ImputationModel(CSDIDenoiser(**config["architecture"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        checkpoint_info[filename] = {
            key: checkpoint[key]
            for key in ("epoch", "train_loss", "validation_loss", "stage")
        }
        return model, TrainingConfig(**checkpoint["config"])

    q_model, q_config = load_model("densifier.pt")
    q_model.requires_grad_(False)
    q = DDPMSampler(q_model, q_config)
    linear = example.LinearDensifier()
    samplers = {
        "densiffusion": q,
        "direct_csdi": linear,
        "linear_densification": linear,
    }
    original_sample = DDPMSampler.sample
    audit = {"calls": 0, "draws": 0, "observed_values_checked": 0}
    dense_histories = {}

    def checked_sample(self, observed, **kwargs):
        original = observed.copy()
        result = original_sample(self, observed, **kwargs)
        np.testing.assert_array_equal(observed, original)
        conditioning, mask, _ = prepare_observations(
            observed,
            factor=kwargs["factor"],
            dt=kwargs["dt"],
            mask=kwargs.get("mask"),
            times=kwargs.get("times"),
        )
        known = np.broadcast_to(~mask[:, None], result.shape)
        np.testing.assert_array_equal(
            result[known], np.broadcast_to(conditioning[:, None], result.shape)[known]
        )
        assert np.isfinite(result).all()
        audit["calls"] += 1
        audit["draws"] += result.shape[0] * result.shape[1]
        audit["observed_values_checked"] += int(known.sum())
        if self is q:
            dense_histories["densiffusion"] = result.copy()
        return result

    before = perf_counter()
    with patch.object(DDPMSampler, "sample", checked_sample):
        for method in example.METHODS:
            model, training_config = load_model(f"{method}_forecaster.pt")
            direct = method == "direct_csdi"
            reproduced = forecast(
                history,
                samplers[method],
                DDPMForecastSampler(model, training_config),
                horizon=config["horizon"],
                factor=1 if direct else config["factor"],
                n_histories=1 if direct else config["n_histories"],
                n_futures=N_DRAWS if direct else config["n_futures"],
                dt=config["original_dt"],
                seed=config["seeds"]["test_sampling"][method],
            )
            np.testing.assert_array_equal(reproduced, saved[method])
            print(
                f"{method}: exact checkpoint replay, finite samples, "
                "and observation preservation passed",
                flush=True,
            )
    with (RUN / "verification_histories.npz").open("xb") as handle:
        np.savez_compressed(handle, history=history, **dense_histories)

    # Confirm 400 logged epochs per stage and validation-selected checkpoints.
    pattern = (
        r"epoch=(\d+) train_loss=([\d.eE+-]+) "
        r"validation_loss=([\d.eE+-]+) lr=([\d.eE+-]+)"
    )
    rows = re.findall(pattern, (RUN / "training.log").read_text())
    assert len(rows) == 1600
    training_history = {}
    names = ["densifier.pt", *(f"{method}_forecaster.pt" for method in example.METHODS)]
    for stage, name in enumerate(names):
        records = rows[400 * stage : 400 * (stage + 1)]
        assert [int(row[0]) for row in records] == list(range(1, 401))
        entries = [
            {
                "epoch": int(epoch),
                "train_loss": float(train),
                "validation_loss": float(validation),
                "learning_rate": float(lr),
            }
            for epoch, train, validation, lr in records
        ]
        best = checkpoint_info[name]
        assert np.isclose(
            best["validation_loss"],
            min(row["validation_loss"] for row in entries),
            rtol=1e-5,
        )
        training_history[name] = entries
    save_json("training_history.json", training_history)
    save_json("assessment_scores.json", scores)
    save_json(
        "verification.json",
        {
            "development_experiment": True,
            "split_trajectory_counts": [len(group) for group in ids],
            "split_window_counts": [len(pair[0]) for pair in windows],
            "test_shape": list(target.shape),
            "all_samples_finite": True,
            "scores_recomputed_exactly_from_saved_samples": True,
            "saved_data_reproduced_exactly": True,
            "all_forecasts_reproduced_exactly_from_saved_checkpoints": True,
            "sampler_observation_preservation": audit,
            "checkpoints": checkpoint_info,
            "replay_seconds": perf_counter() - before,
            "assessment_seconds": perf_counter() - started,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(RUN.iterdir())
        if path.is_file() and path.name not in ("sha256.json", "assessment.log")
    }
    save_json("sha256.json", hashes)
    for method, metrics in scores.items():
        print(method, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
