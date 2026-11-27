"""Train on a site's daily PM2.5 era, forecast its later, naturally sparse era.

The data come from EPA monitors that sampled daily for years and then moved to
the national 1-in-3 or 1-in-6 day calendar. Nothing is removed by hand to make
the test set: every scored value is a real reading from the sparse era.

Task. In the sparse era a site reports on day F - 3 and next on day F. Using the
site's own readings before F and its neighbours' readings up to F - 1, predict
the reading on F. Because PM2.5 has little memory beyond a day, most of the
skill must come from filling the site's missing days F - 1 and F - 2 from its
neighbours: this is where densifying the history can pay off.

Every model is fitted on the site's daily era only (training years, then the
last 20% of the era for validation and model selection):

- persistence: the last reading, F - 3
- recent_mean: mean of the site's readings in the 30-day window
- ridge_direct: ridge regression on the inputs the sparse era provides (site at
  F - 3, F - 6, F - 9; neighbours at F - 1, F - 2, F - 3; season), trained to
  forecast directly without filling anything
- fill_then_ridge: fill F - 1 and F - 2 by regressing the site on its neighbours
  the same day, then apply a forecaster trained on complete daily histories
- csdi: CSDI over the site and its neighbours (one channel each), trained on
  daily-era windows hidden in exactly the sparse-era pattern; it fills the gaps
  and forecasts F jointly, giving a predictive distribution
- csdi_fill_then_ridge: CSDI's filled F - 1 and F - 2, fed (draw by draw) to the
  same complete-history forecaster as fill_then_ridge

Scores on the original scale: MAE, RMSE and CRPS (lower is better). Ridge
models get Gaussian predictive draws from their validation residuals, so CRPS
is comparable across methods. The same task is also scored in the daily era's
validation years, where the true F - 1 and F - 2 are known (dense_val), which
adds an oracle forecaster with complete history.

    uv run --extra research python scripts/run_aqs_forecast.py --site 19-045-0021
    uv run --extra research python scripts/run_aqs_forecast.py --top 8 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from densiffusion.aqs import era_slice
from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig

WINDOW = 30
GAP = 3  # 1-in-3 calendar: the site reports every third day
N_DRAWS = 50


@dataclass(frozen=True)
class Setup:
    site: str
    name: str
    neighbours: list[str]
    dense: tuple[int, int]
    sparse: tuple[int, int]


# --------------------------------------------------------------------------- data


def load(path: Path) -> tuple[dict, dict]:
    archive = np.load(path, allow_pickle=False)
    data = {key: archive[key] for key in archive.files if key != "config"}
    return data, json.loads(str(archive["config"]))


def select_sites(config: dict, *, top: int, min_sparse_years: int = 3) -> list[dict]:
    chosen = []
    for c in config["dense_then_sparse"]:
        if (
            c["gap_coverage"] >= 0.9
            and c["sparse"][1] - c["sparse"][0] + 1 >= min_sparse_years
            and len(c["neighbours"]) >= 2
        ):
            chosen.append(c)
    return chosen[:top]


def panel(data: dict, setup: Setup) -> tuple[np.ndarray, np.ndarray]:
    """(days, 1 + neighbours) values with the target in column 0, and dates."""
    column = {s: i for i, s in enumerate(data["sites"])}
    columns = [column[setup.site], *(column[s] for s in setup.neighbours)]
    return data["values"][:, columns].astype(np.float64), data["dates"]


class Scaler:
    """log1p then per-channel standardisation, fitted on training days only."""

    def __init__(self, values: np.ndarray):
        logged = np.log1p(np.clip(values, 0, None))
        self.mean = np.nanmean(logged, axis=0)
        self.std = np.nanstd(logged, axis=0) + 1e-6

    def forward(self, values):
        return (np.log1p(np.clip(values, 0, None)) - self.mean) / self.std

    def inverse(self, z, channel=0):
        return np.expm1(z * self.std[channel] + self.mean[channel])


def forecast_days(available: np.ndarray, days: range, *, sparse: bool) -> np.ndarray:
    """Days F with a target reading on F and F - 3 (and, if sparse, none between)."""
    target = available[:, 0]
    chosen = []
    for f in days:
        if f < WINDOW or not (target[f] and target[f - GAP]):
            continue
        if sparse and (target[f - 1] or target[f - 2]):
            continue
        chosen.append(f)
    return np.array(chosen, dtype=int)


def windows(z: np.ndarray, available: np.ndarray, ends: np.ndarray, *, natural: bool):
    """Windows ending on each forecast day F, with the conditioning mask.

    hidden: what the model must produce. The target is hidden on F and, unless
    natural (sparse era, where the gaps are real), on every day not on F's
    1-in-3 phase. Neighbours are hidden on F (they are the future too).
    Unavailable entries are hidden as well but never scored or trained on.
    """
    index = ends[:, None] + np.arange(-WINDOW + 1, 1)
    values = np.nan_to_num(z[index], nan=0.0)
    avail = available[index]
    hidden = ~avail
    hidden[:, -1, :] = True
    if not natural:
        off_phase = (np.arange(WINDOW) - (WINDOW - 1)) % GAP != 0
        hidden[:, off_phase, 0] = True
    return values, avail, hidden


# ------------------------------------------------------------------------- ridge


def ridge_fit(x, y, alpha):
    x1 = np.c_[x, np.ones(len(x))]
    penalty = alpha * np.eye(x1.shape[1])
    penalty[-1, -1] = 0
    return np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y)


def ridge_predict(w, x):
    return np.c_[x, np.ones(len(x))] @ w


def lagged(z, available, days, lags, channels):
    """Features z[f - lag, c] with missing values as 0 plus a missing indicator."""
    columns = []
    for c in channels:
        for lag in lags:
            have = available[days - lag, c]
            columns.append(np.where(have, np.nan_to_num(z[days - lag, c]), 0.0))
            columns.append(have.astype(float))
    return np.stack(columns, axis=1)


def season(dates, days):
    angle = (
        2 * np.pi * (dates[days].astype("datetime64[D]").astype(int) % 365.25) / 365.25
    )
    return np.c_[np.sin(angle), np.cos(angle)]


def fit_ridge(features, z, train_days, val_days):
    """Choose the ridge penalty on validation; return weights and residual sd."""
    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        w = ridge_fit(features(train_days), z[train_days, 0], alpha)
        error = ridge_predict(w, features(val_days)) - z[val_days, 0]
        score = np.mean(error**2)
        if best is None or score < best[0]:
            best = (score, w, float(np.std(error)))
    return best[1], best[2]


# -------------------------------------------------------------------------- csdi


def train_csdi(values, avail, hidden, val, config, args, rng):
    """Masked denoising where only hidden entries that really exist are scored.

    The shared trainer assumes every hidden entry has a true value, which is
    not so here (neighbours miss days), so this loop adds that one condition.
    During training a fraction of the observed neighbour values is also hidden
    and scored, so the network learns the neighbours' joint behaviour too.
    """
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = ImputationModel(
        CSDIDenoiser(
            values.shape[-1], hidden_channels=args.hidden, residual_layers=args.layers
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-6
    )
    milestones = [int(0.75 * config.epochs), int(0.9 * config.epochs)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    alpha_bar = (1 - config.betas(dtype=torch.float32, device=device)).cumprod(0)

    def as_tensors(v, a, h):
        return (
            torch.as_tensor(v, dtype=torch.float32, device=device),
            torch.as_tensor(a, device=device),
            torch.as_tensor(h, device=device),
        )

    def loss(v, a, h, generator, extra_rate):
        if extra_rate:
            extra = (
                torch.rand(h.shape, generator=generator, device="cpu").to(device)
                < extra_rate
            )
            extra[..., 0] = False
            h = h | (extra & a)
        steps = torch.randint(len(alpha_bar), (len(v),), generator=generator).to(device)
        noise = torch.randn(v.shape, generator=generator).to(device)
        ab = alpha_bar[steps][:, None, None]
        noisy = torch.where(h, ab.sqrt() * v + (1 - ab).sqrt() * noise, v)
        prediction = model(noisy, steps, observed=v.masked_fill(h, 0), mask=h, dt=1.0)
        scored = h & a
        return (prediction - noise)[scored].square().mean()

    train = as_tensors(values, avail, hidden)
    validation = as_tensors(*val)
    best, best_state, generator = np.inf, None, torch.Generator().manual_seed(args.seed)
    for epoch in range(config.epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(values)), device=device)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            value = loss(
                *(t[batch] for t in train), generator, args.neighbour_mask_rate
            )
            optimizer.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            check = torch.Generator().manual_seed(12345)
            score = float(
                np.mean(
                    [
                        loss(*(t[i : i + 256] for t in validation), check, 0.0).item()
                        for i in range(0, len(validation[0]), 256)
                    ]
                )
            )
        if score < best:
            best, best_state = (
                score,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
            )
        if epoch % max(1, config.epochs // 10) == 0 or epoch == config.epochs - 1:
            logging.info(
                "epoch %d validation noise mse %.4f (best %.4f)", epoch, score, best
            )
    model.load_state_dict(best_state)
    return model, best


def sample_csdi(model, values, hidden, config, args, rng):
    sampler = DDPMSampler(model, replace(config, batch_size=args.sample_batch_size))
    return sampler.sample(
        values, factor=1, n_samples=N_DRAWS, dt=1.0, rng=rng, mask=hidden
    )  # (windows, draws, time, channels)


# ------------------------------------------------------------------------ scores


def scores(draws_or_point, truth):
    """MAE/RMSE of the median (or point) and CRPS of the draws, original scale."""
    if draws_or_point.ndim == 1:
        point, draws = draws_or_point, draws_or_point[:, None]
    else:
        point, draws = np.median(draws_or_point, axis=1), draws_or_point
    error = point - truth
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "crps": crps(draws[:, :, None, None], truth[:, None, None]),
    }


def gaussian_draws(mean_z, sd, scaler, rng):
    z = mean_z[:, None] + sd * rng.standard_normal((len(mean_z), N_DRAWS))
    return scaler.inverse(z)


# -------------------------------------------------------------------------- main


def run_site(data, setup: Setup, args) -> dict:
    rng = np.random.default_rng(args.seed)
    raw, dates = panel(data, setup)
    available = np.isfinite(raw)
    dense = era_slice(dates, setup.dense)
    sparse = era_slice(dates, setup.sparse)
    split = dense.start + int(0.8 * (dense.stop - dense.start))
    train_range, val_range = range(dense.start, split), range(split, dense.stop)
    assert split < dense.stop <= sparse.start, (
        "validation and test must follow training"
    )

    scaler = Scaler(raw[dense.start : split])  # training days only
    z = scaler.forward(raw)
    train_days = forecast_days(available, train_range, sparse=False)
    val_days = forecast_days(available, val_range, sparse=False)
    test_days = forecast_days(available, range(sparse.start, sparse.stop), sparse=True)
    logging.info(
        "%s %s: %d neighbours, train %d / val %d / sparse-era test %d forecasts",
        setup.site,
        setup.name,
        len(setup.neighbours),
        len(train_days),
        len(val_days),
        len(test_days),
    )
    if min(len(train_days), len(val_days), len(test_days)) < 50:
        raise ValueError("Too few forecast days for this site")
    neighbours = range(1, raw.shape[1])

    # The sparse-era inputs: site at F-3, F-6, F-9; neighbours at F-1..F-3.
    def direct_features(days):
        return np.c_[
            lagged(z, available, days, (3, 6, 9), [0]),
            lagged(z, available, days, (1, 2, 3), neighbours),
            season(dates, days),
        ]

    # Complete history: site at F-1, F-2, F-3, neighbours at F-1 (filled or true).
    def dense_features(days, site_lags=None):
        site = lagged(z, available, days, (1, 2, 3), [0])
        if site_lags is not None:  # replace F-1, F-2 values with filled ones
            site[:, 0], site[:, 2] = site_lags[:, 0], site_lags[:, 1]
            site[:, 1], site[:, 3] = 1.0, 1.0
        return np.c_[
            site, lagged(z, available, days, (1,), neighbours), season(dates, days)
        ]

    def fill_features(days, lag):
        return np.c_[
            lagged(z, available, days, (lag,), neighbours), season(dates, days - lag)
        ]

    results = {
        "setup": asdict(setup),
        "counts": {
            "train": len(train_days),
            "val": len(val_days),
            "test": len(test_days),
        },
    }
    w_direct, sd_direct = fit_ridge(direct_features, z, train_days, val_days)
    w_dense, sd_dense = fit_ridge(dense_features, z, train_days, val_days)
    # Same-day spatial fill: site(t) from neighbours(t), fitted on training days.
    w_fill = ridge_fit(fill_features(train_days, 0), z[train_days, 0], 1.0)

    def spatial_fill(days):
        return np.c_[
            ridge_predict(w_fill, fill_features(days, 1)),
            ridge_predict(w_fill, fill_features(days, 2)),
        ]

    # Composite residual sd for fill_then_ridge from validation (honest spread).
    val_fill_pred = ridge_predict(
        w_dense, dense_features(val_days, spatial_fill(val_days))
    )
    sd_fill = float(np.std(val_fill_pred - z[val_days, 0]))

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        diffusion_steps=args.diffusion_steps,
        beta_start=1e-4,
        beta_end=0.5,
        schedule="quadratic",
        device=args.device,
    )
    start = time.time()
    model, val_loss = train_csdi(
        *windows(z, available, train_days, natural=False),
        windows(z, available, val_days, natural=False),
        config,
        args,
        rng,
    )
    results["csdi_train_minutes"] = round((time.time() - start) / 60, 2)
    results["csdi_val_noise_mse"] = val_loss

    for label, days, natural in (
        ("sparse_test", test_days, True),
        ("dense_val", val_days, False),
    ):
        truth = raw[days, 0]
        values, avail, hidden = windows(z, available, days, natural=natural)
        assert hidden[:, -1, :].all(), "the forecast day is never conditioning"
        draws_z = sample_csdi(model, values, hidden, config, args, rng)
        arm = {}
        arm["persistence"] = scores(raw[days - GAP, 0], truth)
        site_window = np.where(
            avail[..., 0] & ~hidden[..., 0], scaler.inverse(values[..., 0]), np.nan
        )
        arm["recent_mean"] = scores(np.nanmean(site_window, axis=1), truth)
        arm["ridge_direct"] = scores(
            gaussian_draws(
                ridge_predict(w_direct, direct_features(days)), sd_direct, scaler, rng
            ),
            truth,
        )
        arm["fill_then_ridge"] = scores(
            gaussian_draws(
                ridge_predict(w_dense, dense_features(days, spatial_fill(days))),
                sd_fill,
                scaler,
                rng,
            ),
            truth,
        )
        arm["csdi"] = scores(scaler.inverse(draws_z[:, :, -1, 0]), truth)
        per_draw = np.stack(
            [
                ridge_predict(w_dense, dense_features(days, draws_z[:, d, [-2, -3], 0]))
                for d in range(N_DRAWS)
            ],
            axis=1,
        )
        noise = sd_dense * rng.standard_normal(per_draw.shape)
        arm["csdi_fill_then_ridge"] = scores(scaler.inverse(per_draw + noise), truth)
        if not natural:  # oracle: the true F-1 and F-2 are known in the daily era
            arm["oracle_complete_history"] = scores(
                gaussian_draws(
                    ridge_predict(w_dense, dense_features(days)), sd_dense, scaler, rng
                ),
                truth,
            )
        results[label] = arm
        logging.info(
            "%s %s", label, json.dumps({k: round(v["crps"], 3) for k, v in arm.items()})
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/aqs_pm25.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/aqs_forecast"))
    parser.add_argument("--site", action="append", help="site id(s); default --top")
    parser.add_argument(
        "--top", type=int, default=8, help="best-covered candidate sites"
    )
    parser.add_argument("--skip", type=int, default=0, help="skip this many candidates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--sample-batch-size", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--neighbour-mask-rate", type=float, default=0.1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output / "training.log"),
        ],
    )
    data, config = load(args.data)
    candidates = (
        [c for c in config["dense_then_sparse"] if c["site"] in args.site]
        if args.site
        else select_sites(config, top=args.skip + args.top)[args.skip :]
    )
    all_results = {"args": {k: str(v) for k, v in vars(args).items()}, "sites": {}}
    for c in candidates:
        setup = Setup(
            c["site"], c["name"], c["neighbours"], tuple(c["dense"]), tuple(c["sparse"])
        )
        try:
            all_results["sites"][setup.site] = run_site(data, setup, args)
        except ValueError as error:
            logging.warning("skipping %s: %s", setup.site, error)
        (args.output / "results.json").write_text(json.dumps(all_results, indent=2))
    logging.info("wrote %s", args.output / "results.json")


if __name__ == "__main__":
    main()
