"""Small, gappy datasets: impute with diffusion, then forecast with a GRU.

Run from the repository root (about an hour per data set and seed on an Apple GPU):

    uv run --extra research python scripts/run_impute_then_forecast.py \
        --data periodic --missing random --device mps --seeds 0,1,2

On Modal, one container per data set and kind of gap, each running every seed:

    modal run --env Densiffusion scripts/modal_run.py \
        --script run_impute_then_forecast.py --timeout-minutes 120 --args \
        "--data periodic --missing random --seeds 0,1,2 --gru-device cuda; ..."

Every window has CONTEXT past and HORIZON future points one step apart, and part
of it is missing: --missing random hides each point with probability --rate;
--missing stride hides every other point (the densification setting). Only N
gappy windows are available (each N in --sizes), 80% for training and 20% for
early stopping. Larger sizes contain the smaller ones.

1. An SSSD imputer learns from those gappy windows alone, as in CSDI: each step
   hides a random share of the observed points and learns to restore them.
   Truly missing points are hidden from it too but never scored.
2. One GRU forecaster per arm, identical apart from its input. It reads the past
   as (value, observed indicator), outputs a Gaussian for every future step, and
   is trained with CRPS on observed future points only.
   - masked: missing values set to zero (the network handles gaps itself)
   - linear: missing values linearly interpolated
   - diffusion: missing values filled by one imputer draw per window
   - diffusion_aug: AUGMENT imputer draws per training window (augmentation)
   - complete: nothing missing anywhere (a ceiling, not a method)
   - grud: GRU-D, which reads the gappy past directly and learns how to decay
     missing inputs toward the last seen value and the mean (no filling)
   Diffusion arms forecast from N_HISTORIES imputer draws of each test past and
   pool N_FUTURES draws from each. One arm has no GRU:
   - sssd_forecast: an SSSD model trained like the imputer but scored on the
     observed future points, which forecasts straight from the gappy past
   --arms picks a subset; the imputer is trained only for the diffusion arms.
3. Forecasts are scored with CRPS against the true future values that training
   could observe: all of them for random gaps, every other one for stride.

Data: a locally periodic GP (period PERIOD steps; references are its exact
posterior fills and forecast) or hourly UCI electricity loads from
scripts/prepare_electricity.py (reference: the same hours one day earlier).
Imputer fills of the test past are also scored at the missing points.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from densiffusion.architectures.sssd import SSSDDenoiser
from densiffusion.baselines import GPPosteriorSampler
from densiffusion.contracts import Array, Densifier, Mask
from densiffusion.gp.kernels import matern_5_2, periodic
from densiffusion.gp.posterior import gp_posterior
from densiffusion.gp.sampling import sample_gp
from densiffusion.metrics import crps
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig

CONTEXT = 96
HORIZON = 24
LENGTH = CONTEXT + HORIZON
ARMS = (
    "masked",
    "linear",
    "diffusion",
    "diffusion_aug",
    "complete",
    "grud",
    "sssd_forecast",
)
AUGMENT = 8
N_HISTORIES = 10
N_FUTURES = 10
TEST_WINDOWS = 500
POOL = 4096  # gappy windows any run can draw its training set from
PERIOD = 12.0
DECAY = 72.0  # how quickly the periodic pattern drifts, in steps
NUGGET = 0.01  # independent noise variance on each value
ELECTRICITY = Path("data/electricity_hourly.npz")
BATCH = 64
EVAL_EVERY = 50
PATIENCE = 20  # evaluations without improvement before stopping


def locally_periodic(t: Array, s: Array) -> Array:
    """A periodic pattern that drifts slowly, plus a little independent noise."""
    pattern = periodic(t, s, variance=1.0, lengthscale=1.0, period=PERIOD)
    drift = matern_5_2(t, s, variance=1.0, lengthscale=DECAY)
    return pattern * drift + NUGGET * np.equal.outer(t, s)


def periodic_windows(count: int, *, seed: int) -> Array:
    times = np.arange(LENGTH, dtype=float)
    return sample_gp(times, locally_periodic, rng=seed, n_samples=count)[..., None]


def electricity_windows(path: Path, *, seed: int) -> tuple[Array, Array]:
    """(train pool, test) windows of clients' scaled hourly loads.

    The first 80% of hours are for training and the rest for testing, windows
    start at midnight, and each client is scaled by its training-period mean
    and standard deviation.
    """
    if not path.exists():
        prepare = Path(__file__).with_name("prepare_electricity.py")
        subprocess.run(
            [sys.executable, str(prepare), "--output", str(path)], check=True
        )
    with np.load(path, allow_pickle=False) as data:
        loads = data["values"]
    split = 24 * (int(0.8 * loads.shape[1]) // 24)
    train = loads[:, :split]
    scaled = (loads - train.mean(axis=1, keepdims=True)) / train.std(
        axis=1, keepdims=True
    )
    rng = np.random.default_rng(seed)

    def windows(first: int, last: int, count: int) -> Array:
        starts = np.arange(first, last - LENGTH + 1, 24)
        pairs = [(c, s) for c in range(len(scaled)) for s in starts]
        chosen = rng.choice(len(pairs), size=count, replace=False)
        return np.stack(
            [scaled[c, s : s + LENGTH] for c, s in (pairs[i] for i in chosen)]
        )[..., None]

    return windows(0, split, POOL), windows(split, loads.shape[1], TEST_WINDOWS)


def missing_mask(shape: tuple[int, ...], kind: str, rate: float, rng) -> Mask:
    """True where a value is missing; the last past point is always observed."""
    if kind == "stride":
        offsets = CONTEXT - 1 - np.arange(shape[1])
        hidden = np.broadcast_to((offsets % 2 == 1)[None, :, None], shape).copy()
    else:
        hidden = rng.random(shape) < rate
    hidden[:, CONTEXT - 1] = False
    return hidden


def scored_future(kind: str) -> Mask:
    """(HORIZON,) future steps that training data can observe."""
    if kind == "stride":
        return (np.arange(CONTEXT, LENGTH) - (CONTEXT - 1)) % 2 == 0
    return np.ones(HORIZON, dtype=bool)


def linear_fill(values: Array, hidden: Mask) -> Array:
    """Interpolate missing values linearly; hold the nearest value at the ends."""
    filled = values.copy()
    steps = np.arange(values.shape[1])
    for row, gaps in zip(filled, hidden, strict=True):
        seen = ~gaps[:, 0]
        row[:, 0] = np.interp(steps, steps[seen], row[seen, 0])
    return filled


def hide_some(available: Mask, rng: np.random.Generator) -> Mask:
    """Hide a random share (10-90%) of each window's observed points.

    Every window with two or more observed points keeps at least one and hides
    at least one.
    """
    scores = rng.random(available.shape)
    rates = rng.uniform(0.1, 0.9, size=(len(available), 1, 1))
    extra = available & (scores < rates)
    for row in np.flatnonzero(~extra.any(axis=(1, 2))):
        candidates = np.where(available[row], scores[row], np.inf)
        extra[row].flat[np.argmin(candidates)] = True
    for row in np.flatnonzero(~(available & ~extra).any(axis=(1, 2))):
        candidates = np.where(available[row], scores[row], -np.inf)
        extra[row].flat[np.argmax(candidates)] = False
    return extra


def hide_future(available: Mask, rng: np.random.Generator | None = None) -> Mask:
    """Hide and score every observed future point: SSSD as a forecaster."""
    extra = np.zeros_like(available)
    extra[:, CONTEXT:] = available[:, CONTEXT:]
    return extra


def train_imputer(
    train: tuple[Array, Mask],
    validation: tuple[Array, Mask],
    *,
    config: TrainingConfig,
    steps: int,
    seed: int,
    targets: Callable[[Mask, np.random.Generator], Mask] = hide_some,
) -> nn.Module:
    """Fit an SSSD imputer to gappy windows with CSDI's self-supervised recipe.

    Each (values, missing) pair holds windows with True at missing points.
    targets(observed, rng) picks the observed points to hide on purpose (by
    default a random share; hide_future trains a forecaster instead). The loss
    is the noise-prediction error at those points; truly missing points are
    hidden as well (their values are unknown) but unscored. Returns the
    weights with the lowest validation loss.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(config.device)
    model = ImputationModel(SSSDDenoiser(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    alpha_bar = (1 - config.betas(dtype=torch.float32, device=device)).cumprod(0)

    def tensors(values, missing, extra, noise, levels):
        def put(array, dtype=torch.float32):
            return torch.as_tensor(array, dtype=dtype, device=device)

        return (
            put(np.where(missing, 0, values)),
            put(missing, torch.bool),
            put(extra, torch.bool),
            put(noise),
            put(levels, torch.long),
        )

    def loss(values, missing, extra, noise, levels):
        hidden = missing | extra
        weight = alpha_bar[levels][:, None, None]
        noisy = weight.sqrt() * values + (1 - weight).sqrt() * noise
        noisy = torch.where(hidden, noisy, values)
        observed = values.masked_fill(hidden, 0)
        prediction = model(noisy, levels, observed=observed, mask=hidden, dt=1.0)
        return (prediction - noise)[extra].square().mean()

    def draw(values, missing, rows):
        return tensors(
            values[rows],
            missing[rows],
            targets(~missing[rows], rng),
            rng.standard_normal(values[rows].shape),
            rng.integers(config.diffusion_steps, size=len(rows)),
        )

    held_out = draw(*validation, np.arange(len(validation[0])))
    best, best_state, stale = np.inf, None, 0
    for step in range(1, steps + 1):
        model.train()
        rows = rng.choice(len(train[0]), size=min(BATCH, len(train[0])), replace=False)
        optimizer.zero_grad(set_to_none=True)
        value = loss(*draw(*train, rows))
        value.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        if step % EVAL_EVERY and step != steps:
            continue
        model.eval()
        with torch.no_grad():
            score = loss(*held_out).item()
        logging.info("[imputer] step=%d validation_loss=%.4f", step, score)
        if score < best:
            best, best_state, stale = score, deepcopy(model.state_dict()), 0
        elif (stale := stale + 1) >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model.eval()


def impute_pasts(
    values: Array, missing: Mask, sampler: Densifier, *, n_samples: int, seed: int
) -> Array:
    """(batch, n_samples, CONTEXT, 1) imputer fills of the past, from the past alone."""
    past, gaps = values[:, :CONTEXT], missing[:, :CONTEXT]
    return impute(
        np.where(gaps, 0, past),
        sampler,
        factor=1,
        n_samples=n_samples,
        seed=seed,
        mask=gaps,
    )


class GRUForecaster(nn.Module):
    """A GRU reads (value, observed) pairs; a Gaussian for each future step."""

    def __init__(self, *, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(2, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 2 * HORIZON)

    def forward(self, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states, _ = self.gru(past)
        mean, scale = self.head(states[:, -1]).chunk(2, dim=-1)
        return mean, F.softplus(scale) + 1e-3


def observation_gaps(
    values: torch.Tensor, observed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(last, gap, seen) for (batch, time) values and 0/1 observed indicators.

    last is the value observed most recently before each step and seen says
    whether there was one; gap counts steps since that observation, or since
    the window start before the first one (GRU-D's delta, 0 at the start).
    """
    steps = torch.arange(values.shape[1], device=values.device).expand_as(values)
    marked = torch.where(observed > 0, steps, torch.full_like(steps, -1))
    latest = marked.cummax(dim=1).values
    previous = torch.cat((torch.full_like(latest[:, :1], -1), latest[:, :-1]), 1)
    seen = previous >= 0
    gap = torch.where(seen, steps - previous, steps).to(values.dtype)
    return values.gather(1, previous.clamp(min=0)), gap, seen


class GRUDForecaster(nn.Module):
    """GRU-D (Che et al., 2018) on the gappy past; a Gaussian per future step.

    A missing input decays from the last observed value toward the training
    mean, and the memory decays, as the gap since the last observation grows,
    at learned rates. A plain GRU layer and the head then match GRUForecaster.
    """

    def __init__(self, mean: float, *, hidden: int = 64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.input_decay = nn.Linear(1, 1)
        self.memory_decay = nn.Linear(1, hidden)
        self.cell = nn.GRUCell(2, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * HORIZON)

    def forward(self, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, observed = past[..., 0], past[..., 1]
        last, gap, seen = observation_gaps(values, observed)
        last = torch.where(seen, last, self.mean)
        keep = torch.exp(-F.relu(self.input_decay(gap[..., None])))[..., 0]
        values = observed * values + (1 - observed) * (
            keep * last + (1 - keep) * self.mean
        )
        memory = torch.exp(-F.relu(self.memory_decay(gap[..., None])))
        state = past.new_zeros(len(past), self.cell.hidden_size)
        states = []
        for step in range(past.shape[1]):
            reading = torch.stack((values[:, step], observed[:, step]), dim=-1)
            state = self.cell(reading, memory[:, step] * state)
            states.append(state)
        states, _ = self.gru(torch.stack(states, dim=1))
        mean, scale = self.head(states[:, -1]).chunk(2, dim=-1)
        return mean, F.softplus(scale) + 1e-3


def gaussian_crps(mean, std, target):
    """Closed-form CRPS of N(mean, std^2) at target, elementwise."""
    z = (target - mean) / std
    density = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    return std * (z * (2 * cdf - 1) + 2 * density - 1 / np.sqrt(np.pi))


def gru_inputs(values: Array, missing: Mask) -> Array:
    """(..., CONTEXT, 2) inputs: filled past values and observed indicators."""
    return np.concatenate((values, (~missing).astype(float)), axis=-1)


def train_gru(
    train: tuple[Array, Array, Mask],
    validation: tuple[Array, Array, Mask],
    *,
    steps: int,
    seed: int,
    device: str = "cpu",
    factory: Callable[[], nn.Module] = GRUForecaster,
) -> nn.Module:
    """Fit on (inputs, future, scored) triples; keep the best validation CRPS."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def put(arrays):
        inputs, future, scored = arrays
        return (
            torch.as_tensor(inputs, dtype=torch.float32, device=device),
            torch.as_tensor(future[..., 0], dtype=torch.float32, device=device),
            torch.as_tensor(scored, dtype=torch.bool, device=device),
        )

    train, validation = put(train), put(validation)

    def loss(inputs, future, scored):
        mean, std = model(inputs)
        return gaussian_crps(mean, std, future)[scored].mean()

    best, best_state, stale = np.inf, None, 0
    for step in range(1, steps + 1):
        model.train()
        rows = rng.choice(len(train[0]), size=min(BATCH, len(train[0])), replace=False)
        rows = torch.as_tensor(rows, device=device)
        optimizer.zero_grad(set_to_none=True)
        value = loss(*(part[rows] for part in train))
        value.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % EVAL_EVERY and step != steps:
            continue
        model.eval()
        with torch.no_grad():
            score = loss(*validation).item()
        if score < best:
            best, best_state, stale = score, deepcopy(model.state_dict()), 0
        elif (stale := stale + 1) >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model.eval()


def filled_pasts(
    values: Array,
    missing: Mask,
    arm: str,
    sampler: Densifier,
    *,
    copies: int,
    seed: int,
) -> Array:
    """(batch, copies, CONTEXT, 1) pasts as arm `arm` sees them."""
    observed = np.where(missing, 0, values)[:, :CONTEXT]
    if arm in ("masked", "grud"):
        return observed[:, None]
    if arm == "linear":
        return linear_fill(observed, missing[:, :CONTEXT])[:, None]
    if arm == "complete":
        return values[:, None, :CONTEXT]
    return impute_pasts(values, missing, sampler, n_samples=copies, seed=seed)


def forecaster_data(
    values: Array, missing: Mask, pasts: Array, scored: Mask, *, complete: bool
) -> tuple[Array, Array, Mask]:
    """GRU (inputs, future, trained-on) triples, one per filled past.

    Future points count only where training data observes them, unless the
    arm sees complete data.
    """
    copies = pasts.shape[1]
    gaps = np.zeros_like(missing) if complete else missing
    inputs = gru_inputs(pasts, np.repeat(gaps[:, None, :CONTEXT], copies, axis=1))
    seen = ~gaps[:, CONTEXT:, 0] & (True if complete else scored)
    return (
        inputs.reshape(-1, CONTEXT, 2),
        np.repeat(values[:, CONTEXT:], copies, axis=0),
        np.repeat(seen, copies, axis=0),
    )


def gru_forecast(model: nn.Module, inputs: Array, *, n_draws: int, seed: int) -> Array:
    """(batch, histories * n_draws, HORIZON, 1) draws for (batch, histories, ...)."""
    batch, histories = inputs.shape[:2]
    device = next(model.parameters()).device
    flat = inputs.reshape(batch * histories, CONTEXT, 2)
    with torch.no_grad():
        mean, std = model(torch.as_tensor(flat, dtype=torch.float32, device=device))
    mean, std = mean.cpu().double().numpy(), std.cpu().double().numpy()
    noise = np.random.default_rng(seed).standard_normal((len(mean), n_draws, HORIZON))
    draws = mean[:, None] + std[:, None] * noise
    return draws.reshape(batch, histories * n_draws, HORIZON, 1)


def score_future(samples: Array, future: Array, scored: Mask) -> float:
    mask = np.broadcast_to(scored[None, :, None], future.shape)
    return crps(samples, future, mask=mask)


def score_fills(samples: Array, past: Array, missing: Mask) -> float:
    return crps(samples, past, mask=missing[:, :CONTEXT])


def parse_arms(text: str) -> list[str]:
    arms = [arm.strip() for arm in text.split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown or not arms:
        raise ValueError(f"Unknown arms {unknown}; choose from {', '.join(ARMS)}")
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", choices=("periodic", "electricity"), required=True)
    parser.add_argument("--missing", choices=("random", "stride"), default="random")
    parser.add_argument("--rate", type=float, default=0.5, help="random only")
    parser.add_argument("--sizes", default="32,128,512,2048")
    parser.add_argument("--imputer-steps", type=int, default=1500)
    parser.add_argument("--gru-steps", type=int, default=3000)
    parser.add_argument("--test-windows", type=int, default=TEST_WINDOWS)
    parser.add_argument("--device", default="cpu", help="for the imputer")
    parser.add_argument("--gru-device", default="cpu")
    parser.add_argument("--seeds", default="0", help="comma-separated, run in turn")
    parser.add_argument("--electricity", type=Path, default=ELECTRICITY)
    parser.add_argument("--arms", type=parse_arms, default=",".join(ARMS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.sizes = sorted(int(size) for size in args.sizes.split(","))
    if args.sizes[-1] > POOL:
        raise SystemExit(f"--sizes must be at most {POOL}")
    args.output = args.output or Path(
        f"outputs/impute_then_forecast/{args.data}_{args.missing}"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in (int(seed) for seed in args.seeds.split(",")):
        run(args, seed)


def run(args: argparse.Namespace, seed: int) -> None:
    """One seed: every training size; results go to <output>/seed<seed>.json."""
    path = args.output / f"seed{seed}.json"
    tag = f"[{args.data} {args.missing} seed {seed}]"

    # The test set is the same for every seed; seeds change the training data.
    if args.data == "periodic":
        pool = periodic_windows(POOL, seed=1000 + seed)
        test = periodic_windows(TEST_WINDOWS, seed=999)
    else:
        pool, test = electricity_windows(args.electricity, seed=999)
        pool = pool[np.random.default_rng(seed).permutation(POOL)]
    test = test[: args.test_windows]
    pool_missing = missing_mask(
        pool.shape, args.missing, args.rate, np.random.default_rng(seed)
    )
    test_missing = missing_mask(
        test.shape, args.missing, args.rate, np.random.default_rng(999)
    )
    past, future = test[:, :CONTEXT], test[:, CONTEXT:]
    scored = scored_future(args.missing)
    observed_past = np.where(test_missing[:, :CONTEXT], 0, past)

    results = {
        "data": args.data,
        "missing": args.missing,
        "rate": args.rate if args.missing == "random" else 0.5,
        "seed": seed,
        "test_windows": len(test),
        "references": {},
        "sizes": {},
    }
    linear_past = linear_fill(observed_past, test_missing[:, :CONTEXT])
    results["references"]["linear_fill_crps"] = score_fills(
        linear_past[:, None], past, test_missing
    )
    if args.data == "periodic":
        exact = GPPosteriorSampler(partial(gp_posterior, kernel=locally_periodic))
        hidden = test_missing.copy()
        hidden[:, CONTEXT:] = True
        draws = impute(
            np.where(hidden, 0, test),
            exact,
            factor=1,
            n_samples=N_HISTORIES * N_FUTURES,
            seed=seed,
            mask=hidden,
        )
        results["references"]["exact_gp_fill_crps"] = score_fills(
            draws[:, :, :CONTEXT], past, test_missing
        )
        results["references"]["exact_gp_forecast"] = score_future(
            draws[:, :, CONTEXT:], future, scored
        )
    else:
        yesterday = linear_past[:, None, CONTEXT - HORIZON :]
        results["references"]["previous_day"] = score_future(yesterday, future, scored)
    logging.info("%s references %s", tag, results["references"])

    config = TrainingConfig(
        diffusion_steps=50, beta_start=1e-4, beta_end=0.2, device=args.device
    )
    for size in args.sizes:
        start = time.perf_counter()
        timing = {}

        def lap(name, timing=timing):
            nonlocal start
            timing[name] = round(time.perf_counter() - start, 1)
            start = time.perf_counter()

        cut = max(2, size // 5)
        windows, missing = pool[:size], pool_missing[:size]
        train = (windows[cut:], missing[cut:])
        validation = (windows[:cut], missing[:cut])
        entry = {"windows": {"train": size - cut, "validation": cut}}

        sampler = test_fills = None
        if any(arm.startswith("diffusion") for arm in args.arms):
            imputer = train_imputer(
                train, validation, config=config, steps=args.imputer_steps, seed=seed
            )
            sampler = DDPMSampler(imputer, replace(config, batch_size=1024))
            lap("imputer training")
            test_fills = impute_pasts(
                test, test_missing, sampler, n_samples=N_HISTORIES, seed=seed + 1
            )
            entry["sssd_fill_crps"] = score_fills(test_fills, past, test_missing)
            lap("test imputation")
        mean = float(train[0][~train[1]].mean())  # GRU-D's fallback value

        entry["forecast_crps"] = {}
        for arm in args.arms:
            if arm == "sssd_forecast":
                forecaster = train_imputer(
                    train,
                    validation,
                    config=config,
                    steps=args.imputer_steps,
                    seed=seed,
                    targets=hide_future,
                )
                hidden = test_missing.copy()
                hidden[:, CONTEXT:] = True
                draws = impute(
                    np.where(hidden, 0, test),
                    DDPMSampler(forecaster, replace(config, batch_size=1024)),
                    factor=1,
                    n_samples=N_HISTORIES * N_FUTURES,
                    seed=seed + 2,
                    mask=hidden,
                )
                entry["forecast_crps"][arm] = score_future(
                    draws[:, :, CONTEXT:], future, scored
                )
                lap(arm)
                continue
            copies = AUGMENT if arm == "diffusion_aug" else 1
            complete = arm == "complete"
            sets = [
                forecaster_data(
                    values,
                    gaps,
                    filled_pasts(
                        values, gaps, arm, sampler, copies=n, seed=seed + 10 + i
                    ),
                    scored,
                    complete=complete,
                )
                for i, (values, gaps, n) in enumerate(
                    (train + (copies,), validation + (1,))
                )
            ]
            forecaster = train_gru(
                *sets,
                steps=args.gru_steps,
                seed=seed,
                device=args.gru_device,
                factory=partial(GRUDForecaster, mean)
                if arm == "grud"
                else GRUForecaster,
            )
            if arm.startswith("diffusion"):
                inputs = gru_inputs(
                    test_fills,
                    np.repeat(test_missing[:, None, :CONTEXT], N_HISTORIES, axis=1),
                )
                draws = N_FUTURES
            else:
                gaps = np.zeros_like(test_missing) if complete else test_missing
                inputs = gru_inputs(
                    filled_pasts(test, test_missing, arm, sampler, copies=1, seed=0),
                    gaps[:, None, :CONTEXT],
                )
                draws = N_HISTORIES * N_FUTURES
            samples = gru_forecast(forecaster, inputs, n_draws=draws, seed=seed)
            entry["forecast_crps"][arm] = score_future(samples, future, scored)
            lap(f"{arm} gru")
        entry["timing_seconds"] = timing
        results["sizes"][str(size)] = entry
        logging.info(
            "%s size %d fills %s forecasts %s",
            tag,
            size,
            round(entry.get("sssd_fill_crps", float("nan")), 4),
            {k: round(v, 4) for k, v in entry["forecast_crps"].items()},
        )
        path.write_text(json.dumps(results, indent=2))
    logging.info("%s done: %s", tag, path)


if __name__ == "__main__":
    main()
