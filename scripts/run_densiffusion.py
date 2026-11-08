"""Benchmark densified vs sparse diffusion forecasting on GP paths.

`--model` picks the network used for every imputer and forecaster: PriSTI,
CSDI, SSSD or TimeAutoDiff, each at about 45k parameters (see
densiffusion.densify_forecast). Everything else is identical across models.

For each task, fine ground-truth paths are generated at spacing dt / 8 and
observed every 8th point (spacing dt). Everything is trained on the coarse
observations only:

1. an imputer trained at strides 1, 2 and 4 of the observations (2*dt -> dt,
   4*dt -> 2*dt, 8*dt -> 4*dt) and applied three times (x8);
2. forecasters from densified history to densified future at x2, x4 and x8,
   plus x8 with linear interpolation instead of the imputer;
3. a forecaster from the sparse history to the sparse future.

Forecasts are scored at the coarse future times against the exact GP forecast
(the best possible) and repeating the last observation.

The quasi-periodic tasks (qp_*) sweep the ratio of the period to the coarse
spacing and compare two data settings (see docs/pristi_experiments.md):

- coarse-only ("learned"): as above, but the imputer trains at stride 1 only,
  since thinning a wave aliases it;
- fine training data ("fine"): one imputer per refinement level, each trained
  on the true fine paths at exactly the spacing it refines (dt -> dt/2,
  dt/2 -> dt/4, dt/4 -> dt/8), and a forecaster trained on the true fine
  paths. Test histories are still coarse and are densified by those imputers.

"Oracle" forecasters share the fine-data forecaster but densify test histories
with the exact GP posterior: the best any densifier could do. Report each group
of tasks and each model into its own output folder. Run one task per process,
then combine:

    uv run --extra research python scripts/run_densiffusion.py run \
        --model csdi --task qp_2.5 --output outputs/quasiperiodic/csdi
    uv run --extra research --with matplotlib python \
        scripts/run_densiffusion.py report --output outputs/quasiperiodic/csdi

`compare` reads one such folder per model under --output and writes a
cross-model table, the checks of each quasi-periodic finding and a plot:

    uv run --extra research --with matplotlib python \
        scripts/run_densiffusion.py compare --output outputs/quasiperiodic
"""

import argparse
import dataclasses
import json
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from densiffusion.baselines import GPPosteriorSampler
from densiffusion.densify_forecast import (
    CHUNK,
    MODEL_LABELS,
    SMALL_MODELS,
    densify,
    forecast,
    forecast_layout,
    imputation_windows,
    interpolate,
    refine,
    sampling_config,
    small_model,
    train_forecaster,
    train_imputer,
)
from densiffusion.gp.kernels import matern_3_2
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import sample_gp
from densiffusion.pipeline import impute
from densiffusion.training import TrainingConfig

TASKS = {
    "brownian": ("Brownian motion", 0.04, {"kernel": "brownian", "variance": 1.0}),
    "fbm_h0.7": (
        "fBm, H = 0.7",
        0.04,
        {"kernel": "fractional_brownian", "variance": 1.0, "hurst": 0.7},
    ),
    "fbm_h0.3": (
        "fBm, H = 0.3",
        0.04,
        {"kernel": "fractional_brownian", "variance": 1.0, "hurst": 0.3},
    ),
    "matern52": (
        "Matérn 5/2",
        0.25,
        {"kernel": "matern_5_2", "variance": 1.0, "lengthscale": 1.0},
    ),
}
QUASI_DECAY = 12.0
for _ratio in (1.6, 2.5, 3, 4, 6, 8, 16):
    TASKS[f"qp_{_ratio:g}"] = (
        f"P/dt = {_ratio:g}",
        1.0,
        {
            "kernel": "quasi_periodic",
            "variance": 1.0,
            "period": float(_ratio),
            "decay": QUASI_DECAY,
        },
    )
LEVELS, FINE = 3, 8
COARSE_POINTS = 33
HISTORY, HORIZON = 9, 4
WINDOW_STARTS = (0, 20)
DATA_SEEDS = {"train": 22001, "validation": 22002, "test": 22003}
HISTORIES, FUTURES = 10, 10
DRAWS = HISTORIES * FUTURES
METHODS = (
    "persistence",
    "sparse",
    "learned_x2",
    "learned_x4",
    "learned_x8",
    "interp_x8",
)
QUASI_METHODS = (
    "persistence",
    "sparse",
    "learned_x2",
    "learned_x8",
    "interp_x8",
    "fine_x2",
    "fine_x8",
    "oracle_x2",
    "oracle_x8",
    "interp_x2",
    "fineinterp_x2",
    "sparse_shifted",
)
# Added after the first sweep; each draws its own seed (method_seed), so the
# earlier methods' samples are unchanged and --extend matches a fresh run.
ADDED_METHODS = ("interp_x2", "fineinterp_x2", "sparse_shifted")
LEVELS_TESTED = (-2, -1, 0, 1, 2, 3)


def quasi_periodic(t, s, *, variance, period, decay):
    """Cosine times Matern 3/2: one frequency with drifting amplitude and phase."""
    lag = np.subtract.outer(t, s)
    return np.cos(2 * np.pi * lag / period) * matern_3_2(
        t, s, variance=variance, lengthscale=decay
    )


def kernel_params(gp):
    """GP parameters with script-local kernel names resolved to functions."""
    local = {"quasi_periodic": quasi_periodic}
    return {**gp, "kernel": local.get(gp["kernel"], gp["kernel"])}


def task_methods(gp):
    return QUASI_METHODS if "period" in gp else METHODS


def imputer_strides(gp):
    """Coarse strides for the imputer; thinning would alias a periodic process."""
    return (1,) if "period" in gp else (1, 2, 4)


def method_seed(base, name):
    """Sampling seed for one method that does not depend on run order."""
    return int(np.random.default_rng([base, *name.encode()]).integers(2**62))


def phase_shifted_windows(paths):
    """Coarse forecaster windows at every offset of the fine grid.

    The fine paths seen at the coarse spacing: forecast_windows' coarse windows,
    plus the same windows shifted by 1 to FINE - 1 fine steps (one coarse step
    earlier where they would overrun the path). FINE times as many windows.
    """
    span = (HISTORY + HORIZON - 1) * FINE
    last = paths.shape[1] - 1 - span
    windows = []
    for start in WINDOW_STARTS:
        for offset in range(FINE):
            first = FINE * start + offset
            first = first if first <= last else first - FINE
            windows.append(paths[:, first : first + span + 1 : FINE])
    return np.concatenate(windows)


def level_windows(paths, level, rng):
    """Imputer windows refining the fine paths to spacing dt / 2**level.

    Subsampled to as many windows per path as stride 1 of the coarse paths
    gives, so each per-level imputer trains on as much data as the coarse one.
    """
    windows = imputation_windows(paths, strides=(FINE // 2**level,))
    count = len(paths) * ((COARSE_POINTS - 1) // (2 * CHUNK - 2))
    return windows[np.sort(rng.choice(len(windows), size=count, replace=False))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("run", "report", "compare"))
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--model", choices=list(SMALL_MODELS), default="pristi")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(SMALL_MODELS),
        default=list(SMALL_MODELS),
        help="Models to compare; each is read from --output/<model>",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-paths", type=int, default=256)
    parser.add_argument("--validation-paths", type=int, default=32)
    parser.add_argument("--test-paths", type=int, default=64)
    parser.add_argument("--imputer-epochs", type=int, default=60)
    parser.add_argument("--forecaster-epochs", type=int, default=150)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--device", default="cpu", help="Torch device, e.g. mps for the Apple GPU"
    )
    parser.add_argument(
        "--train-device",
        help="Device for training if not --device; SSSD training on mps can "
        "produce non-finite gradients, so train it on cpu and sample on mps",
    )
    parser.add_argument("--seed", type=int, default=42)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--rescore", action="store_true", help="Reuse checkpoints")
    reuse.add_argument(
        "--extend",
        action="store_true",
        help="Add methods missing from a finished run, reusing its results",
    )
    args = parser.parse_args()
    if args.mode == "run":
        torch.set_num_threads(args.threads)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        run_task(args)
    elif args.mode == "report":
        report(args)
    else:
        compare(args)


def config(epochs, device="cpu"):
    return TrainingConfig(
        epochs=epochs,
        batch_size=64,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=0.5,
        schedule="quadratic",
        weight_decay=1e-6,
        lr_decay_fractions=(0.75, 0.9),
        validation_steps="all",
        validation_interval=10,
        device=device,
    )


def generate(task, count, part):
    _, dt, gp = TASKS[task]
    params = kernel_params(gp)
    times = np.arange((COARSE_POINTS - 1) * FINE + 1) * dt / FINE
    paths = sample_gp(
        times, params.pop("kernel"), rng=DATA_SEEDS[part], n_samples=count, **params
    )
    return paths, times


def forecast_windows(fine, times):
    """Coarse windows, their fine truth, and absolute coarse times."""
    length = HISTORY + HORIZON - 1
    coarse = fine[:, ::FINE]
    windows = np.concatenate([coarse[:, s : s + length + 1] for s in WINDOW_STARTS])
    truth = np.concatenate(
        [fine[:, FINE * s : FINE * (s + length) + 1] for s in WINDOW_STARTS]
    )
    window_times = np.concatenate(
        [
            np.broadcast_to(times[::FINE][s : s + length + 1], (len(fine), length + 1))
            for s in WINDOW_STARTS
        ]
    )
    return windows, truth, window_times


def run_task(args):
    task = args.task
    label, dt, gp = TASKS[task]
    folder = args.output / task
    reuse = args.rescore or args.extend
    train_device = args.train_device or args.device
    folder.mkdir(parents=True, exist_ok=reuse)
    previous = folder / "results.json"
    previous = json.loads(previous.read_text()) if reuse else {}
    logger = logging.getLogger("densiffusion.training")
    timings = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timings[name] = now - clock
        clock = now
        print(f"[{task}] {name}: {timings[name] / 60:.1f} min", flush=True)

    fine = {
        part: generate(task, count, part)
        for part, count in (
            ("train", args.train_paths),
            ("validation", args.validation_paths),
            ("test", args.test_paths),
        )
    }
    coarse = {part: paths[:, ::FINE] for part, (paths, _) in fine.items()}
    posterior = partial(gp_posterior, mean=0.0, **kernel_params(gp))
    gp_sampler = GPPosteriorSampler(posterior)
    rng = np.random.default_rng(args.seed)

    def seed():
        return int(rng.integers(2**62))

    def fit(model, checkpoint, train):
        """Train, or load the saved best weights with --rescore (or --extend,
        if they exist)."""
        if args.rescore or (args.extend and checkpoint.exists()):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            model.to(args.device).eval()
            return None
        log = _EpochLog()
        logger.addHandler(log)
        try:
            train()
        finally:
            logger.removeHandler(log)
        model.to(args.device)
        return log.records

    windows = {
        part: forecast_windows(paths, times) for part, (paths, times) in fine.items()
    }
    train_windows, train_truth, _ = windows["train"]
    validation_windows, validation_truth, _ = windows["validation"]
    test_windows, test_truth, test_times = windows["test"]
    test_history = test_windows[:, :HISTORY]
    future = test_windows[:, HISTORY:]

    methods = task_methods(gp)
    if args.extend:
        results = {"model": args.model, **previous, "methods": methods}
        with np.load(folder / "samples.npz") as saved:
            stored = {key: saved[key] for key in saved.files}
        epochs = dict(previous["epochs"])
        methods_fine = "densified_fine" in stored
    else:
        imputer_config = config(args.imputer_epochs, train_device)
        sampler = sampling_config(imputer_config)
        torch.manual_seed(args.seed)
        imputer = small_model(args.model)
        strides = imputer_strides(gp)
        imputer_epochs = fit(
            imputer,
            folder / "imputer.pt",
            lambda: train_imputer(
                imputer,
                imputation_windows(coarse["train"], strides),
                imputation_windows(coarse["validation"], strides),
                imputer_config,
                seed=args.seed,
                checkpoint=folder / "imputer.pt",
            ),
        )
        lap("imputer training")

        results = {"task": task, "label": label, "dt": dt, "gp": gp}
        results["model"] = args.model
        results["device"] = args.device
        results["train_device"] = train_device
        results["methods"] = methods
        results["imputer_strides"] = strides
        results["scale_transfer"] = scale_transfer(
            imputer, sampler, fine["test"], gp_sampler, seed
        )
        lap("scale-transfer diagnostics")

        dense_train = densify(
            imputer, sampler, train_windows, levels=LEVELS, n_samples=1, seed=seed()
        )
        dense_validation = densify(
            imputer,
            sampler,
            validation_windows,
            levels=LEVELS,
            n_samples=1,
            seed=seed(),
        )
        dense_test = densify(
            imputer,
            sampler,
            test_history,
            levels=LEVELS,
            n_samples=HISTORIES,
            seed=seed(),
        )
        lap("densification")
        exact_test = [
            exact_history(
                gp_sampler, test_truth, test_times, 2**level, HISTORIES, seed()
            )
            for level in range(1, LEVELS + 1)
        ]
        results["densified_history"] = history_quality(
            dense_test, exact_test, test_history, test_truth
        )
        methods_fine = any(m.startswith("fine_") for m in methods)
        if methods_fine:
            fine_imputers = []
            for level in range(1, LEVELS + 1):
                pick = np.random.default_rng(args.seed + level)
                checkpoint = folder / f"imputer_fine_level{level}.pt"
                torch.manual_seed(args.seed)
                model = small_model(args.model)
                epochs_fine = fit(
                    model,
                    checkpoint,
                    partial(
                        train_imputer,
                        model,
                        level_windows(fine["train"][0], level, pick),
                        level_windows(fine["validation"][0], level, pick),
                        imputer_config,
                        seed=args.seed,
                        checkpoint=checkpoint,
                    ),
                )
                results.setdefault("fine_imputer_epochs", []).append(epochs_fine)
                fine_imputers.append(model)
            lap("fine-data imputer training")
            dense_test_fine = densify(
                fine_imputers,
                sampler,
                test_history,
                levels=LEVELS,
                n_samples=HISTORIES,
                seed=seed(),
            )
            results["densified_history_fine"] = history_quality(
                dense_test_fine, exact_test, test_history, test_truth
            )
            lap("fine-data densification")
        epochs = {"imputer": imputer_epochs}

    if args.extend:
        keys = ("gp", "persistence", *methods)
        samples = {key: stored[key] for key in keys if key in stored}
    else:
        hidden, _ = forecast_layout(HISTORY, HORIZON)
        gp_forecast = impute(
            test_windows[..., None],
            gp_sampler,
            factor=1,
            mask=hidden,
            times=test_times,
            n_samples=DRAWS,
            dt=dt,
            seed=seed(),
        )[:, :, HISTORY:, 0]
        persistence = np.repeat(test_history[:, -1:, None], HORIZON, axis=2)
        samples = {"gp": gp_forecast, "persistence": persistence}

    def variant(name):
        """(checkpoint name, factor, train, validation, test histories,
        histories per window). Fine, oracle and fineinterp variants share a
        forecaster trained on the true fine paths; only their test histories
        differ. sparse_shifted trains on the fine paths at every coarse phase."""
        if name == "sparse":
            return name, 1, train_windows, validation_windows, test_history, 1
        if name == "sparse_shifted":
            return (
                name,
                1,
                phase_shifted_windows(fine["train"][0]),
                phase_shifted_windows(fine["validation"][0]),
                test_history,
                1,
            )
        kind, factor = name.split("_x")
        factor = int(factor)
        level = factor.bit_length() - 1
        if kind == "learned":
            dense = dense_test[level - 1]
            return (
                name,
                factor,
                dense_train[level - 1][:, 0],
                dense_validation[level - 1][:, 0],
                dense.reshape(-1, dense.shape[-1]),
                HISTORIES,
            )
        if kind == "fineinterp":
            return (
                f"truefine_x{factor}",
                factor,
                train_truth[:, :: FINE // factor],
                validation_truth[:, :: FINE // factor],
                interpolate(test_history, factor),
                1,
            )
        if kind in ("fine", "oracle"):
            dense = (dense_test_fine if kind == "fine" else exact_test)[level - 1]
            return (
                f"truefine_x{factor}",
                factor,
                train_truth[:, :: FINE // factor],
                validation_truth[:, :: FINE // factor],
                dense.reshape(-1, dense.shape[-1]),
                HISTORIES,
            )
        return (
            name,
            factor,
            interpolate(train_windows, factor),
            interpolate(validation_windows, factor),
            interpolate(test_history, factor),
            1,
        )

    forecaster_config = config(args.forecaster_epochs, train_device)
    forecaster_sampler = sampling_config(forecaster_config)
    # FINE times the windows: train for the same number of updates as sparse.
    shifted_config = dataclasses.replace(
        forecaster_config,
        epochs=max(1, args.forecaster_epochs // FINE),
        validation_interval=1,
    )
    forecasters = {}
    for name in methods:
        if name == "persistence" or name in samples:
            continue
        key, factor, train, validation, histories, per_window = variant(name)
        if key not in forecasters:
            torch.manual_seed(args.seed)
            forecasters[key] = small_model(args.model)
            record = fit(
                forecasters[key],
                folder / f"{key}.pt",
                partial(
                    train_forecaster,
                    forecasters[key],
                    train,
                    validation,
                    shifted_config if key == "sparse_shifted" else forecaster_config,
                    history=HISTORY,
                    horizon=HORIZON,
                    factor=factor,
                    seed=args.seed,
                    checkpoint=folder / f"{key}.pt",
                ),
            )
            if not (args.extend and key in epochs):
                epochs[key] = record
        model = forecasters[key]
        draws = DRAWS // per_window
        predicted = forecast(
            model,
            forecaster_sampler,
            histories,
            horizon=HORIZON,
            factor=factor,
            n_samples=draws,
            seed=method_seed(args.seed, name) if name in ADDED_METHODS else seed(),
        )
        samples[name] = predicted.reshape(len(test_windows), DRAWS, HORIZON)
        lap(f"forecaster {name}")

    results["forecast"] = score_forecasts(samples, future, args.test_paths)
    if args.rescore:
        results["epochs"] = previous["epochs"]
        if "fine_imputer_epochs" in previous:
            results["fine_imputer_epochs"] = previous["fine_imputer_epochs"]
        results["timings"] = previous["timings"]
        results["rescore_timings"] = timings
    elif args.extend:
        results["epochs"] = epochs
        results["extend_timings"] = timings
    else:
        results["epochs"] = epochs
        results["timings"] = timings
    if args.extend:
        densified = {
            k: stored[k] for k in ("densified", "densified_fine") if k in stored
        }
    else:
        densified = {"densified": dense_test[-1]}
        if methods_fine:
            densified["densified_fine"] = dense_test_fine[-1]
    (folder / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(
        folder / "samples.npz",
        truth=test_truth,
        windows=test_windows,
        times=test_times,
        **densified,
        **samples,
    )
    print(f"[{task}] done", flush=True)


def scale_transfer(imputer, sampler, fine_test, gp_sampler, seed):
    """One-step refinement from true data at each spacing dt * 2**-level."""
    paths, times = fine_test
    rows = {}
    alternate = np.arange(2 * 5 - 1) % 2 == 1
    for level in LEVELS_TESTED:
        stride = FINE // 2**level if level >= 0 else FINE * 2**-level
        span = 8 * stride
        start = paths.shape[1] - 1 - span
        truth = paths[:, start : start + span + 1 : stride]
        window_times = np.broadcast_to(
            times[start : start + span + 1 : stride], truth.shape
        )
        learned = refine(imputer, sampler, truth[:, ::2], n_samples=DRAWS, seed=seed())
        exact = impute(
            truth[..., None],
            gp_sampler,
            factor=1,
            mask=alternate,
            times=window_times,
            n_samples=DRAWS,
            dt=float(window_times[0, 1] - window_times[0, 0]),
            seed=seed(),
        )[..., 0]
        midpoint = (truth[:, :-1:2] + truth[:, 2::2]) / 2
        rows[str(level)] = {
            "trained_scale": level <= 0,
            "model": float(ensemble_crps(learned[:, :, 1::2], truth[:, 1::2]).mean()),
            "gp": float(ensemble_crps(exact[:, :, 1::2], truth[:, 1::2]).mean()),
            "interpolation": float(np.abs(midpoint - truth[:, 1::2]).mean()),
        }
    return rows


def exact_history(gp_sampler, truth, times, factor, draws, seed):
    """Exact GP posterior draws of the x factor history given its coarse points."""
    length = (HISTORY - 1) * factor + 1
    fine_truth = truth[:, : (HISTORY - 1) * FINE + 1 : FINE // factor]
    fine_times = times[:, :1] + np.arange(length) * (times[0, 1] - times[0, 0]) / factor
    return impute(
        fine_truth[..., None],
        gp_sampler,
        factor=1,
        mask=np.arange(length) % factor != 0,
        times=fine_times,
        n_samples=draws,
        dt=float(fine_times[0, 1] - fine_times[0, 0]),
        seed=seed,
    )[..., 0]


def history_quality(dense_test, exact_test, history, truth):
    """CRPS of densified history points against the GP given coarse history."""
    rows = {}
    for level, (densified, exact) in enumerate(
        zip(dense_test, exact_test, strict=True), start=1
    ):
        factor = 2**level
        length = (HISTORY - 1) * factor + 1
        fine_truth = truth[:, : (HISTORY - 1) * FINE + 1 : FINE // factor]
        new = np.arange(length) % factor != 0
        interpolated = interpolate(history, factor)
        rows[f"x{factor}"] = {
            "model": float(
                ensemble_crps(densified[:, :, new], fine_truth[:, new]).mean()
            ),
            "gp": float(ensemble_crps(exact[:, :, new], fine_truth[:, new]).mean()),
            "interpolation": float(
                np.abs(interpolated[:, new] - fine_truth[:, new]).mean()
            ),
            "draws": int(densified.shape[1]),
        }
    return rows


def ensemble_crps(samples, truth):
    """Fair ensemble CRPS per entry for (windows, draws, time) samples.

    Unlike the plug-in estimator in metrics.crps, its expectation does not
    depend on the number of draws, so 4-draw, 16-draw and deterministic
    (one draw: absolute error) forecasts are directly comparable.
    """
    draws = samples.shape[1]
    errors = np.abs(samples - truth[:, None]).mean(axis=1)
    if draws == 1:
        return errors
    spread = np.abs(samples[:, :, None] - samples[:, None]).sum(axis=(1, 2))
    return errors - spread / (2 * draws * (draws - 1))


def score_forecasts(samples, future, n_paths):
    """CRPS, error and coverage per method, plus paired gains over sparse."""
    scores = {}
    per_window = {}
    for name, draws in samples.items():
        crps = ensemble_crps(draws, future)
        per_window[name] = crps.mean(axis=1)
        mean = draws.mean(axis=1)
        lower, upper = draws.min(axis=1), draws.max(axis=1)
        scores[name] = {
            "crps": float(crps.mean()),
            "crps_by_lead": crps.mean(axis=0).tolist(),
            "rmse": float(np.sqrt(((mean - future) ** 2).mean())),
            "range_coverage": (
                float(((future >= lower) & (future <= upper)).mean())
                if draws.shape[1] > 1
                else None
            ),
        }
    rng = np.random.default_rng(0)
    paths = np.arange(len(future)) % n_paths
    for name in samples:
        if name in ("sparse", "gp"):
            continue
        gain = per_window["sparse"] - per_window[name]
        by_path = np.array([gain[paths == path].mean() for path in range(n_paths)])
        resampled = rng.integers(n_paths, size=(4000, n_paths))
        interval = np.quantile(by_path[resampled].mean(axis=1), [0.025, 0.975])
        scores[name]["gain_over_sparse"] = float(gain.mean())
        scores[name]["gain_over_sparse_95"] = interval.tolist()
    return scores


class _EpochLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("epoch="):
            fields = dict(item.split("=") for item in message.split())
            self.records.append({key: float(value) for key, value in fields.items()})


INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOURS = {
    "gp": "#eb6834",
    "persistence": "#c3c2b7",
    "sparse": "#eda100",
    "learned_x2": "#86b6ef",
    "learned_x4": "#3987e5",
    "learned_x8": "#1c5cab",
    "interp_x8": "#1baf7a",
    "fine_x2": "#f2b3cb",
    "fine_x8": "#e87ba4",
    "oracle_x2": "#9085e9",
    "oracle_x8": "#6250d6",
    "interp_x2": "#8fd9b6",
    "fineinterp_x2": "#a3245a",
    "sparse_shifted": "#8a5d00",
    "model": "#2a78d6",
    "interpolation": "#1baf7a",
}
NAMES = {
    "gp": "Exact GP forecast (optimal)",
    "persistence": "Repeat last value",
    "sparse": "Sparse forecaster",
    "learned_x2": "Coarse-data x2",
    "learned_x4": "Coarse-data x4",
    "learned_x8": "Coarse-data x8",
    "interp_x8": "Interpolated x8",
    "fine_x2": "Fine-data x2",
    "fine_x8": "Fine-data x8",
    "oracle_x2": "Oracle x2",
    "oracle_x8": "Oracle x8",
    "interp_x2": "Interpolated x2",
    "fineinterp_x2": "Fine-data x2, interpolated history",
    "sparse_shifted": "Sparse, all phase shifts",
}
MARKERS = {
    "learned": "o",
    "fine": "^",
    "oracle": "s",
    "interp": "D",
    "fineinterp": "v",
    "sparse_shifted": "P",
}


def report(args):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _style(plt)
    results = {
        task: json.loads((args.output / task / "results.json").read_text())
        for task in TASKS
        if (args.output / task / "results.json").exists()
    }
    if not results:
        raise FileNotFoundError(f"No task results in {args.output}")
    methods = {tuple(result.get("methods", METHODS)) for result in results.values()}
    if len(methods) > 1:
        raise ValueError("Report tasks with different methods into separate folders")
    methods = methods.pop()
    model = _model_label(results)
    write_tables(results, methods, args.output, model)
    plot_forecast_crps(plt, results, methods, args.output)
    plot_gains(plt, results, methods, args.output)
    plot_by_lead(plt, results, methods, args.output)
    if all("period" in result["gp"] for result in results.values()):
        plot_sweep(plt, results, methods, args.output)
    plot_scale_transfer(plt, results, args.output, model)
    plot_examples(plt, results, args.output)
    print(f"Report written to {args.output}")


def _model_label(results):
    """Display name of the one model behind a folder of task results."""
    models = {result.get("model", "pristi") for result in results.values()}
    if len(models) > 1:
        raise ValueError("Report each model's results into a separate folder")
    return MODEL_LABELS[models.pop()]


def _learned(row):
    """The model's score in a diagnostic row (older PriSTI results: "pristi")."""
    return row["model"] if "model" in row else row["pristi"]


def write_tables(results, methods, output, model="PriSTI"):
    lines = [
        f"# Densified vs sparse {model} forecasting",
        "",
        f"Coarse spacing dt, x{FINE} ground truth; {HISTORY} history and {HORIZON} "
        f"future coarse points; {DRAWS} draws per window "
        f"({HISTORIES} densified histories x {FUTURES} futures).",
        "",
        "## Forecast CRPS relative to the exact GP forecast (1 = best possible)",
        "",
        "| Task | GP CRPS | " + " | ".join(NAMES[m] for m in methods) + " |",
        "|---|---|" + "---|" * len(methods),
    ]
    for result in results.values():
        forecast = result["forecast"]
        gp = forecast["gp"]["crps"]
        cells = [f"{forecast[m]['crps'] / gp:.3f}" for m in methods]
        lines.append(f"| {result['label']} | {gp:.4f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## CRPS gain over the sparse forecaster (positive = better than sparse)",
        "",
        "Mean per-window CRPS reduction with a 95% bootstrap interval over test paths.",
        "",
        "| Task | " + " | ".join(NAMES[m] for m in methods if m != "sparse") + " |",
        "|---|" + "---|" * (len(methods) - 1),
    ]
    for result in results.values():
        cells = []
        for method in methods:
            if method == "sparse":
                continue
            entry = result["forecast"][method]
            low, high = entry["gain_over_sparse_95"]
            cells.append(f"{entry['gain_over_sparse']:+.4f} [{low:+.4f}, {high:+.4f}]")
        lines.append(f"| {result['label']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Forecast RMSE of the ensemble mean and coverage of the draws' range",
        "",
        f"Coverage of [min, max] of the {DRAWS} draws; a calibrated forecaster "
        f"covers {(DRAWS - 1) / (DRAWS + 1):.0%} of outcomes.",
        "",
        "| Task | " + " | ".join(NAMES[m] for m in ("gp", *methods)) + " |",
        "|---|" + "---|" * (len(methods) + 1),
    ]
    for result in results.values():
        forecast = result["forecast"]
        cells = [
            f"{forecast[m]['rmse']:.4f} ({_percent(forecast[m]['range_coverage'])})"
            for m in ("gp", *methods)
        ]
        lines.append(f"| {result['label']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Imputer scale transfer: one-step CRPS relative to the exact GP",
        "",
        "Spacing of the refined grid relative to dt; the imputer was trained only "
        f"at the listed strides of dt. {model} / interpolation ratios.",
        "",
        "| Task | Strides | "
        + " | ".join(_spacing(level) for level in LEVELS_TESTED)
        + " |",
        "|---|---|" + "---|" * len(LEVELS_TESTED),
    ]
    for result in results.values():
        rows = result["scale_transfer"]
        cells = [
            f"{_learned(rows[str(level)]) / rows[str(level)]['gp']:.2f} / "
            f"{rows[str(level)]['interpolation'] / rows[str(level)]['gp']:.2f}"
            for level in LEVELS_TESTED
        ]
        strides = ", ".join(map(str, result.get("imputer_strides", (1, 2, 4))))
        lines.append(f"| {result['label']} | {strides} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Densified history quality: CRPS relative to the exact GP given the "
        "coarse history",
        "",
        f"{HISTORIES} densified draws per window, scored with the fair ensemble "
        f"CRPS, which is unbiased for any number of draws. {model} / interpolation "
        "ratios; fine-data rows use the per-level imputers trained on fine paths.",
        "",
        "| Task | Imputer | x2 | x4 | x8 |",
        "|---|---|---|---|---|",
    ]
    for result in results.values():
        for key, imputer in (
            ("densified_history", "coarse-trained"),
            ("densified_history_fine", "fine-data"),
        ):
            if key not in result:
                continue
            rows = result[key]
            cells = [
                f"{_learned(rows[k]) / rows[k]['gp']:.2f} / "
                f"{rows[k]['interpolation'] / rows[k]['gp']:.2f}"
                for k in ("x2", "x4", "x8")
            ]
            lines.append(
                f"| {result['label']} | {imputer} | " + " | ".join(cells) + " |"
            )
    lines += ["", "## Wall-clock (minutes)", ""]
    for result in results.values():
        total = sum(result["timings"].values()) / 60
        lines.append(f"- {result['label']}: {total:.1f}")
    (output / "results.md").write_text("\n".join(lines) + "\n")


def _percent(value):
    return "n/a" if value is None else f"{value:.0%}"


def _spacing(level):
    return f"{2.0**-level:g} dt" if level <= 0 else f"dt/{2**level}"


def plot_forecast_crps(plt, results, methods, output):
    labels = [result["label"] for result in results.values()]
    x = np.arange(len(labels))
    width = 0.9 / len(methods)
    figure, ax = plt.subplots(figsize=(11, 4.6))
    for index, method in enumerate(methods):
        ratios = [
            result["forecast"][method]["crps"] / result["forecast"]["gp"]["crps"]
            for result in results.values()
        ]
        offset = (index - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            ratios,
            width=width * 0.92,
            color=COLOURS[method],
            label=NAMES[method],
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, color=SECONDARY, fontsize=7)
    ax.axhline(1, color=COLOURS["gp"], linewidth=2, label=NAMES["gp"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Forecast CRPS / exact GP CRPS")
    ax.set_ylim(0.9 * min(ax.get_ylim()[1], 1), None)
    ax.legend(frameon=False, ncols=4, loc="upper left", fontsize=8)
    ax.set_title(
        "Forecast CRPS at coarse future points (lower is better; 1 = best possible)",
        loc="left",
    )
    figure.tight_layout()
    figure.savefig(output / "forecast_crps.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_gains(plt, results, methods, output):
    methods = [m for m in methods if m not in ("sparse", "persistence")]
    labels = [result["label"] for result in results.values()]
    figure, ax = plt.subplots(figsize=(9, 3.8))
    ax.axhline(0, color=AXIS, linewidth=1)
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * 0.8 / len(methods)
        for row, result in enumerate(results.values()):
            entry = result["forecast"][method]
            scale = result["forecast"]["gp"]["crps"]
            low, high = (value / scale for value in entry["gain_over_sparse_95"])
            ax.vlines(row + offset, low, high, color=COLOURS[method], linewidth=2.5)
            ax.plot(
                row + offset,
                entry["gain_over_sparse"] / scale,
                "o",
                color=COLOURS[method],
                ms=7,
                mec=SURFACE,
                mew=2,
                label=NAMES[method] if row == 0 else None,
            )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("CRPS gain over sparse / GP CRPS")
    ax.legend(frameon=False, ncols=4, loc="upper left", fontsize=8)
    ax.set_title(
        "Gain of densified forecasters over the sparse forecaster "
        "(above 0 = better; 95% interval over test paths)",
        loc="left",
    )
    figure.tight_layout()
    figure.savefig(output / "forecast_gain.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_by_lead(plt, results, methods, output):
    figure, axes = plt.subplots(
        1, len(results), figsize=(3.4 * len(results), 3.4), squeeze=False
    )
    leads = np.arange(1, HORIZON + 1)
    for ax, result in zip(axes[0], results.values(), strict=True):
        shown = (
            "gp",
            "persistence",
            "sparse",
            "learned_x8",
            "interp_x8",
            "fine_x8",
            "oracle_x8",
        )
        for method in (m for m in shown if m == "gp" or m in methods):
            ax.plot(
                leads,
                result["forecast"][method]["crps_by_lead"],
                "o-",
                color=COLOURS[method],
                linewidth=2,
                ms=5,
                mec=SURFACE,
                mew=1,
                label=NAMES[method],
            )
        ax.set_xticks(leads)
        ax.set_xlabel("Lead time (coarse steps)")
        ax.set_title(result["label"], loc="left")
    axes[0, 0].set_ylabel("CRPS")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=6, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output / "forecast_by_lead.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_sweep(plt, results, methods, output):
    """CRPS ratio and gain over sparse against the period-to-spacing ratio."""
    ordered = sorted(results.values(), key=_period_ratio)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    shown = [m for m in methods if m not in ("persistence", "sparse")]
    _sweep_axes(axes[0], axes[1], ordered, shown)
    axes[0].set_ylabel("Forecast CRPS / exact GP CRPS")
    axes[0].set_title("Forecast CRPS (lower is better; 1 = best possible)", loc="left")
    axes[1].set_ylabel("CRPS gain over sparse / GP CRPS")
    axes[1].set_title(
        "Gain over the sparse forecaster (above 0 = better; 95% interval)", loc="left"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=5, frameon=False)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(output / "sweep.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def _period_ratio(result):
    return result["gp"]["period"] / result["dt"]


def _sweep_axes(crps_ax, gain_ax, ordered, densified):
    """Draw one model's CRPS ratio and gain over sparse against P/dt."""
    ratios = np.array([_period_ratio(r) for r in ordered])
    for ax in (crps_ax, gain_ax):
        ax.axvspan(ratios.min() / 1.1, 2, color=GRID, alpha=0.6, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(ratios, [f"{r:g}" for r in ratios])
        ax.minorticks_off()
        ax.set_xlim(ratios.min() / 1.1, ratios.max() * 1.1)
        ax.set_xlabel("Period / coarse spacing (shaded: aliased)")
    ceiling, off_scale = 2.8, []
    for method in ("sparse", *densified):
        values = np.array(
            [
                r["forecast"][method]["crps"] / r["forecast"]["gp"]["crps"]
                for r in ordered
            ]
        )
        style = {"color": COLOURS[method], "linewidth": 2, "label": NAMES[method]}
        crps_ax.plot(ratios, np.minimum(values, ceiling), **style)
        off_scale += _clip_markers(crps_ax, ratios, values, ceiling, method)
    crps_ax.axhline(1, color=COLOURS["gp"], linewidth=2, label=NAMES["gp"])
    crps_ax.set_ylim(0.9, ceiling * 1.04)
    _off_scale_note(crps_ax, off_scale, ratios)
    gain_ax.axhline(0, color=AXIS, linewidth=1)
    floor, off_scale = -1.0, []
    for index, method in enumerate(densified):
        offset = 1.025 ** (index - (len(densified) - 1) / 2)
        scale = np.array([r["forecast"]["gp"]["crps"] for r in ordered])
        gain = np.array([r["forecast"][method]["gain_over_sparse"] for r in ordered])
        interval = np.array(
            [r["forecast"][method]["gain_over_sparse_95"] for r in ordered]
        )
        gain_ax.vlines(
            ratios * offset,
            np.maximum(interval[:, 0] / scale, floor),
            np.maximum(interval[:, 1] / scale, floor),
            color=COLOURS[method],
            linewidth=2,
        )
        off_scale += _clip_markers(
            gain_ax, ratios * offset, gain / scale, floor, method
        )
    gain_ax.set_ylim(floor * 1.06, None)
    _off_scale_note(gain_ax, off_scale, ratios)


def _clip_markers(ax, x, values, limit, method):
    """Method markers, with arrows at the axis edge for points beyond limit.

    Returns (method, x, value) for each clipped point.
    """
    beyond = values > limit if limit > 0 else values < limit
    marker = MARKERS.get(method.split("_x")[0], "o")
    style = {"color": COLOURS[method], "linestyle": "none", "mec": SURFACE, "mew": 1}
    ax.plot(x[~beyond], values[~beyond], marker=marker, ms=6, **style)
    ax.plot(
        x[beyond],
        np.full(beyond.sum(), limit),
        marker="^" if limit > 0 else "v",
        ms=8,
        **style,
    )
    return [(method, a, b) for a, b in zip(x[beyond], values[beyond], strict=True)]


def _off_scale_note(ax, clipped, ratios):
    """List clipped values once, in the panel's corner."""
    if not clipped:
        return
    nearest = [ratios[np.abs(np.log(ratios / a)).argmin()] for _, a, _ in clipped]
    lines = [
        f"{NAMES[method]}, P/dt = {ratio:g}: {value:.1f}"
        for (method, _, value), ratio in zip(clipped, nearest, strict=True)
    ]
    ax.text(
        0.99,
        0.02 if clipped[0][2] < 0 else 0.98,
        "Off scale (arrows):\n" + "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom" if clipped[0][2] < 0 else "top",
        color=SECONDARY,
        fontsize=7,
    )


def plot_scale_transfer(plt, results, output, model="PriSTI"):
    figure, axes = plt.subplots(
        1, len(results), figsize=(3.4 * len(results), 3.4), squeeze=False
    )
    levels = np.array(LEVELS_TESTED)
    for ax, result in zip(axes[0], results.values(), strict=True):
        rows = result["scale_transfer"]
        ax.axvspan(0.5, levels.max() + 0.5, color=GRID, alpha=0.5, linewidth=0)
        for key in ("model", "interpolation"):
            ratios = [
                (_learned(row) if key == "model" else row[key]) / row["gp"]
                for row in (rows[str(level)] for level in levels)
            ]
            ax.plot(
                levels,
                ratios,
                "o-",
                color=COLOURS[key],
                linewidth=2,
                ms=5,
                mec=SURFACE,
                mew=1,
                label=f"{model} imputer" if key == "model" else "Linear interpolation",
            )
        ax.axhline(1, color=COLOURS["gp"], linewidth=2, label="Exact GP")
        ax.set_xticks(levels, [_spacing(level) for level in levels], fontsize=7)
        ax.set_xlabel("Refined spacing (shaded: finer than trained)")
        ax.set_title(result["label"], loc="left")
    axes[0, 0].set_ylabel("One-step CRPS / exact GP CRPS")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output / "imputer_scale_transfer.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_examples(plt, results, output):
    figure, axes = plt.subplots(
        len(results), 1, figsize=(11, 3.2 * len(results)), squeeze=False
    )
    for ax, (task, result) in zip(axes[:, 0], results.items(), strict=True):
        data = np.load(output / task / "samples.npz")
        truth, window = data["truth"][0], data["windows"][0]
        fine_steps = np.arange(truth.size) / FINE
        ax.plot(fine_steps, truth, color=AXIS, linewidth=1, label="Fine ground truth")
        dense = data["densified"][0]
        dense_steps = np.arange(dense.shape[-1]) / 2**LEVELS
        for index, path in enumerate(dense[:2]):
            ax.plot(
                dense_steps,
                path,
                color=COLOURS["learned_x8"],
                linewidth=1,
                alpha=0.7,
                label="Densified x8 history (2 draws)" if index == 0 else None,
            )
        coarse_steps = np.arange(window.size)
        ax.plot(
            coarse_steps[:HISTORY],
            window[:HISTORY],
            "o",
            color=INK,
            ms=5,
            mec=SURFACE,
            mew=1,
            label="Observed history",
        )
        ax.plot(
            coarse_steps[HISTORY:],
            window[HISTORY:],
            "o",
            mfc=SURFACE,
            mec=INK,
            ms=6,
            mew=1.4,
            label="True future",
        )
        for offset, method in ((-0.18, "sparse"), (0.0, "learned_x8"), (0.18, "gp")):
            draws = data[method][0]
            low, mid, high = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
            x = coarse_steps[HISTORY:] + offset
            ax.vlines(x, low, high, color=COLOURS[method], linewidth=2.5)
            ax.plot(
                x,
                mid,
                "o",
                color=COLOURS[method],
                ms=4,
                mec=SURFACE,
                mew=1,
                label=f"{NAMES[method]}: 90% interval",
            )
        ax.set_title(f"{result['label']}: first test window (not selected)", loc="left")
        ax.set_xlabel("Time (coarse steps)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=4, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output / "forecast_examples.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


COMPARED = (
    "learned_x2",
    "interp_x2",
    "fine_x2",
    "fineinterp_x2",
    "sparse_shifted",
    "oracle_x2",
)


def compare(args):
    """Cross-model tables, finding checks and plot for the quasi-periodic sweep."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _style(plt)
    sweeps = {}
    for model in args.models:
        found = [
            {**json.loads(path.read_text()), "folder": str(path.parent)}
            for task in TASKS
            if task.startswith("qp_")
            and (path := args.output / model / task / "results.json").exists()
        ]
        if found:
            sweeps[model] = sorted(found, key=_period_ratio)
    if not sweeps:
        raise FileNotFoundError(f"No quasi-periodic results under {args.output}")
    write_comparison(sweeps, args.output)
    plot_comparison(plt, sweeps, args.output)
    print(f"Comparison written to {args.output}")


def _crps_ratio(result, method):
    forecast = result["forecast"]
    return forecast[method]["crps"] / forecast["gp"]["crps"]


def _significance(result, method):
    """+1 if significantly better than sparse, -1 if significantly worse, else 0."""
    low, high = result["forecast"][method]["gain_over_sparse_95"]
    return 1 if low > 0 else -1 if high < 0 else 0


def _verdict(passed, total):
    if total == 0:
        return "n/a"
    if passed == total:
        return "yes"
    return "no" if passed == 0 else f"partly ({passed}/{total})"


def finding_checks(sweep):
    """(finding, verdict, evidence) for each quasi-periodic finding from PriSTI."""
    at = {_period_ratio(result): result for result in sweep}
    barely = [at[p] for p in (2.5, 3, 4) if p in at]
    resolved = [at[p] for p in sorted(at) if p >= 6]
    sampled = [at[p] for p in sorted(at) if p >= 2.5]
    tag = {1: "", 0: " (ns)", -1: " (worse)"}
    rows = []

    def cut(result, method):
        sparse = _crps_ratio(result, "sparse")
        return (sparse - _crps_ratio(result, method)) / (sparse - 1)

    passed = sum(_significance(r, "fine_x2") == 1 for r in barely)
    evidence = ", ".join(
        f"{cut(r, 'fine_x2'):.0%}{tag[_significance(r, 'fine_x2')]}" for r in barely
    )
    rows.append(
        (
            "Fine-data x2 significantly beats sparse at P/dt 2.5, 3 and 4 "
            "(share of sparse's excess CRPS removed)",
            _verdict(passed, len(barely)),
            evidence,
        )
    )
    margins = [_crps_ratio(r, "interp_x8") - _crps_ratio(r, "fine_x8") for r in barely]
    rows.append(
        (
            "Fine-data x8 beats interpolated x8 at P/dt 2.5-4 (CRPS margin)",
            _verdict(sum(m > 0 for m in margins), len(margins)),
            ", ".join(f"{m:+.2f}" for m in margins),
        )
    )
    gaps = [_crps_ratio(r, "fine_x2") - _crps_ratio(r, "oracle_x2") for r in sampled]
    rows.append(
        (
            "Fine-data x2 is within 0.05 of the oracle for P/dt >= 2.5 (largest gap)",
            _verdict(sum(g <= 0.05 for g in gaps), len(gaps)),
            f"{max(gaps):.3f}" if gaps else "",
        )
    )
    cases = [(r, m) for m in ("learned_x2", "learned_x8") for r in barely[1:]]
    rows.append(
        (
            "Coarse-data x2 and x8 are significantly worse than sparse at P/dt 3 "
            "and 4 (CRPS / sparse CRPS)",
            _verdict(sum(_significance(r, m) == -1 for r, m in cases), len(cases)),
            ", ".join(
                f"{_crps_ratio(r, m) / _crps_ratio(r, 'sparse'):.1f}x" for r, m in cases
            ),
        )
    )
    densified = [m for m in QUASI_METHODS if "_x" in m]
    winners = [
        f"{NAMES[m]} at {_period_ratio(r):g}"
        for r in resolved
        for m in densified
        if m in r["forecast"] and _significance(r, m) == 1
    ]
    rows.append(
        (
            "No densifier significantly beats sparse at P/dt >= 6 (exceptions)",
            "yes" if resolved and not winners else "no" if winners else "n/a",
            ", ".join(winners) or "none",
        )
    )
    cases = [(r, m) for r in resolved for m in densified if m.endswith("_x8")]
    rows.append(
        (
            "Every x8 densifier is significantly worse than sparse at P/dt >= 6",
            _verdict(sum(_significance(r, m) == -1 for r, m in cases), len(cases)),
            f"{sum(_significance(r, m) == -1 for r, m in cases)} of {len(cases)}",
        )
    )
    wins = sum(_crps_ratio(r, "fine_x2") < _crps_ratio(r, "fine_x8") for r in sampled)
    rows.append(
        (
            "Fine-data x2 beats fine-data x8 for P/dt >= 2.5",
            _verdict(wins, len(sampled)),
            f"{wins} of {len(sampled)}",
        )
    )
    if 1.6 in at:
        aliased = at[1.6]
        best = min(("fine_x2", "fine_x8"), key=lambda m: _crps_ratio(aliased, m))
        helps = _significance(aliased, best) == 1
        gap = _crps_ratio(aliased, "interp_x8") - _crps_ratio(aliased, best)
        verdict = _verdict(helps + (gap <= 0.05), 2)
        evidence = (
            f"best fine {_crps_ratio(aliased, best):.3f}"
            f"{tag[_significance(aliased, best)]}, "
            f"interpolated {_crps_ratio(aliased, 'interp_x8'):.3f}"
        )
    else:
        verdict, evidence = "n/a", ""
    rows.append(
        (
            "At P/dt 1.6 fine data help, but interpolation is within 0.05 of the "
            "best fine-data forecaster",
            verdict,
            evidence,
        )
    )
    return rows


CONTROLS = (
    (
        "interp_x2",
        "Fine-data x2 beats interpolated x2 (same factor, coarse training data)",
    ),
    (
        "fineinterp_x2",
        "Fine-data x2 beats the same forecaster given interpolated histories "
        "(the densified test history matters)",
    ),
    (
        "sparse_shifted",
        "Fine-data x2 beats a sparse forecaster trained on the fine paths at "
        "every coarse phase (densifying beats using fine data directly)",
    ),
)


def paired_gain(samples, baseline, method):
    """Mean per-window CRPS reduction of method over baseline, with a 95%
    bootstrap interval over test paths (as in score_forecasts)."""
    future = samples["windows"][:, HISTORY:]
    n_paths = len(future) // len(WINDOW_STARTS)
    gain = ensemble_crps(samples[baseline], future).mean(axis=1)
    gain = gain - ensemble_crps(samples[method], future).mean(axis=1)
    paths = np.arange(len(gain)) % n_paths
    by_path = np.bincount(paths, gain) / np.bincount(paths)
    resampled = np.random.default_rng(0).integers(n_paths, size=(4000, n_paths))
    low, high = np.quantile(by_path[resampled].mean(axis=1), [0.025, 0.975])
    return float(gain.mean()), float(low), float(high)


def control_checks(sweep, ratios=(2.5, 3, 4)):
    """(control, verdict, evidence) comparing fine-data x2 with each control.

    Evidence is fine-data x2's CRPS gain over the control in units of the
    exact GP's CRPS, with its 95% interval, at each P/dt in ratios.
    """
    at = {_period_ratio(result): result for result in sweep}
    tasks = [at[p] for p in ratios if p in at]
    rows = []
    for control, finding in CONTROLS:
        usable = [r for r in tasks if control in r["forecast"] and "folder" in r]
        if not usable:
            rows.append((finding, "n/a", "not run"))
            continue
        cells, passed = [], 0
        for result in usable:
            with np.load(Path(result["folder"]) / "samples.npz") as samples:
                gain = paired_gain(samples, control, "fine_x2")
            scale = result["forecast"]["gp"]["crps"]
            mean, low, high = (value / scale for value in gain)
            passed += low > 0
            cells.append(f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]")
        rows.append((finding, _verdict(passed, len(usable)), "; ".join(cells)))
    return rows


def write_comparison(sweeps, output):
    labels = [MODEL_LABELS[model] for model in sweeps]
    checks = {model: finding_checks(sweep) for model, sweep in sweeps.items()}
    findings = [row[0] for row in next(iter(checks.values()))]
    header = ["| " + " | ".join(labels) + " |", "|---" * len(labels) + "|"]
    lines = [
        "# Quasi-periodic sweep across models",
        "",
        f"Every model gets identical data, windows, training schedule and {DRAWS} "
        f"draws per test window ({HISTORIES} densified histories x {FUTURES} "
        "futures); only the network changes, each at about 45k parameters. CRPS "
        "is relative to the exact GP forecast (1 = best possible).",
        "",
        "## Do the PriSTI findings hold?",
        "",
        "Verdicts use each model's own sparse forecaster and 95% bootstrap "
        "intervals over test paths; (ns) marks a difference that is not "
        "significant.",
        "",
        "| Finding " + header[0],
        "|---" + header[1],
    ]
    for index, finding in enumerate(findings):
        cells = []
        for model in sweeps:
            _, verdict, evidence = checks[model][index]
            cells.append(f"{verdict}: {evidence}" if evidence else verdict)
        lines.append(f"| {finding} | " + " | ".join(cells) + " |")
    controls = {model: control_checks(sweep) for model, sweep in sweeps.items()}
    lines += [
        "",
        "## Where does the fine-data gain come from?",
        "",
        "Fine-data x2 against three controls at P/dt = 2.5, 3 and 4: its CRPS "
        "gain over each control in units of the exact GP's CRPS (positive = "
        "fine-data x2 is better), with 95% bootstrap intervals over test paths. "
        "Yes means significantly better at every P/dt.",
        "",
        "| Control " + header[0],
        "|---" + header[1],
    ]
    for index, (_, finding) in enumerate(CONTROLS):
        cells = []
        for model in sweeps:
            _, verdict, evidence = controls[model][index]
            cells.append(f"{verdict}: {evidence}")
        lines.append(f"| {finding} | " + " | ".join(cells) + " |")
    ratios = sorted({_period_ratio(r) for sweep in sweeps.values() for r in sweep})
    lines += [
        "",
        "## Forecast CRPS relative to the exact GP, by method",
        "",
        "`+` / `-`: significantly better / worse than the same model's sparse "
        "forecaster.",
    ]
    for method in (m for m in QUASI_METHODS if m != "persistence"):
        lines += [
            "",
            f"### {NAMES[method]}",
            "",
            "| P/dt " + header[0],
            "|---" + header[1],
        ]
        for ratio in ratios:
            cells = []
            for sweep in sweeps.values():
                result = next((r for r in sweep if _period_ratio(r) == ratio), None)
                if result is None:
                    cells.append("n/a")
                    continue
                if method not in result["forecast"]:
                    cells.append("n/a")
                    continue
                mark = {1: " +", 0: "", -1: " -"}
                sign = 0 if method == "sparse" else _significance(result, method)
                cells.append(f"{_crps_ratio(result, method):.3f}{mark[sign]}")
            lines.append(f"| {ratio:g} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Densified x8 history CRPS relative to the exact GP given the coarse "
        "history",
        "",
        "Fine-data imputers / coarse-trained imputer.",
        "",
        "| P/dt " + header[0],
        "|---" + header[1],
    ]
    for ratio in ratios:
        cells = []
        for sweep in sweeps.values():
            result = next((r for r in sweep if _period_ratio(r) == ratio), None)
            if result is None or "densified_history_fine" not in result:
                cells.append("n/a")
                continue
            fine = result["densified_history_fine"]["x8"]
            coarse = result["densified_history"]["x8"]
            cells.append(
                f"{_learned(fine) / fine['gp']:.2f} / "
                f"{_learned(coarse) / coarse['gp']:.2f}"
            )
        lines.append(f"| {ratio:g} | " + " | ".join(cells) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def plot_comparison(plt, sweeps, output):
    """Each model's sweep in one column: CRPS ratio above, gain over sparse below."""
    figure, axes = plt.subplots(
        2, len(sweeps), figsize=(4.3 * len(sweeps), 8), sharey="row", squeeze=False
    )
    for column, (model, sweep) in enumerate(sweeps.items()):
        shown = [m for m in COMPARED if all(m in r["forecast"] for r in sweep)]
        _sweep_axes(axes[0, column], axes[1, column], sweep, shown)
        axes[0, column].set_title(f"{MODEL_LABELS[model]}: forecast CRPS", loc="left")
        axes[1, column].set_title(
            f"{MODEL_LABELS[model]}: gain over sparse", loc="left"
        )
    axes[0, 0].set_ylabel("Forecast CRPS / exact GP CRPS (1 = best)")
    axes[1, 0].set_ylabel("CRPS gain over sparse / GP CRPS (above 0 = better)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=7, frameon=False)
    figure.suptitle(
        "Quasi-periodic sweep by model (95% intervals over test paths)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(output / "comparison.png", dpi=150, facecolor=SURFACE)
    plt.close(figure)


def _style(plt):
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "legend.labelcolor": SECONDARY,
            "font.family": "sans-serif",
        }
    )


if __name__ == "__main__":
    main()
