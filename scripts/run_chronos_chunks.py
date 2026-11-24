"""Per-customer CSDI + Chronos-2 on 8-day electricity chunks.

Each chunk is one customer's 7-day hourly history and the following day. Gaps
are hidden in the history, a univariate CSDI fills them, and Chronos-2
forecasts the next day from each version of the history: with the gaps left
missing, filled by straight lines, filled by the median of the CSDI draws,
filled by each CSDI draw separately (forecasts pooled), and complete. See
docs/chronos_experiment_plan.md for the design and the main comparisons.

    uv run python scripts/prepare_electricity.py --n-channels 321 \
        --output data/electricity_all.npz
    uv run --extra research --extra chronos python scripts/run_chronos_chunks.py \
        --data data/electricity_all.npz --output outputs/chunks/random_selfsup \
        --gaps random --protocol selfsup --device cuda --seed 0
    uv run --extra research python scripts/run_chronos_chunks.py compare \
        --output outputs/chunks

Design:

- Customers: a fixed 20% are held out from training entirely. Time: the
  first 70% of the record trains CSDI, the next 15% selects its checkpoint,
  and test chunks come from the last 15%, for both training and held-out
  customers. Chunks that are mostly zero or constant are skipped.
- Gaps (``--gaps random``): a random number of gaps (1 to ``--max-gaps``),
  each with a random start and a random length (1 to ``--max-gap-hours``),
  possibly overlapping, so the missing fraction varies from chunk to chunk.
  ``--gaps heavy`` hides a random 50-80% of the history in gaps of 6-48
  hours, so outages can last up to two days. ``--gaps daily`` hides
  12:00-23:00 every day instead, as in run_csdi_chronos2.py; use it with
  ``--start-hour 12`` so histories end at the last observed hour.
- The test chunks and their gaps depend only on ``--test-seed``, so every
  method, protocol and seed is scored on the same chunks.
- CSDI (``--protocol``): ``supervised`` learns from the true values in the
  gaps; ``selfsup`` never sees them, and learns by filling one extra run of
  1-2 observed hours per history. Both see the same number of training
  histories and updates.
- Each history is scaled by the mean and spread of its observed hours only,
  so neither the hidden hours nor the target day leak into CSDI's input.
- Scores: CRPS and the absolute error of the median forecast over the 24
  target hours, divided by the mean absolute observed value of the chunk's
  history, so large and small customers count equally.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from densiffusion.architectures.csdi import CSDIDenoiser
from densiffusion.fx import load_multivariate_dataset
from densiffusion.models import ImputationModel
from densiffusion.pipeline import impute
from densiffusion.samplers import DDPMSampler
from densiffusion.training import TrainingConfig, train_densifier

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from run_csdi_chronos2 import (  # noqa: E402
    FINE_QUANTILES,
    QUANTILE_SAMPLES,
    extra_run_targets,
    load_chronos,
    quantile_samples,
)

LOGGER = logging.getLogger("chronos_chunks")
HISTORY = 168
HORIZON = 24
DAY = 24
SPLITS = (0.70, 0.85)  # train / validation / test boundaries, as fractions of time
ARMS = ("gappy", "linear", "csdi_median", "csdi_draws", "complete")
POINT_ARMS = ("seasonal_naive",)
BOOTSTRAP = 4000


# ---------------------------------------------------------------- sampling


def split_clients(n_clients: int, holdout: float, seed: int):
    """(training customers, held-out customers), a fixed random split."""
    order = np.random.default_rng(seed).permutation(n_clients)
    n_held = max(1, round(holdout * n_clients))
    return np.sort(order[n_held:]), np.sort(order[:n_held])


def usable(chunk: np.ndarray) -> bool:
    """Skip chunks whose history or target is mostly zero, or flat history."""
    history, target = chunk[:HISTORY], chunk[HISTORY:]
    return bool(
        np.mean(history == 0) < 0.5
        and np.mean(target == 0) < 0.5
        and np.ptp(history) > 0
        and np.isfinite(chunk).all()
    )


def sample_chunks(values, clients, lo, hi, n, rng, *, start_hour=0):
    """``n`` distinct usable (customer, start) pairs with chunks in [lo, hi).

    Starts fall at ``start_hour`` (the series starts at midnight), so every
    forecast begins at the same hour of the day.
    """
    first = lo + (start_hour - lo) % DAY
    starts = np.arange(first, hi - HISTORY - HORIZON + 1, DAY)
    if not len(starts) or not len(clients):
        raise ValueError("no room for chunks in this period")
    chosen, seen = [], set()
    for _ in range(50 * n):
        pair = (int(rng.choice(clients)), int(rng.choice(starts)))
        if pair in seen:
            continue
        seen.add(pair)
        client, start = pair
        if usable(values[start : start + HISTORY + HORIZON, client]):
            chosen.append(pair)
            if len(chosen) == n:
                break
    if len(chosen) < n:
        raise ValueError(f"found only {len(chosen)} usable chunks of {n}")
    return np.array(chosen, dtype=np.int64)


def random_gaps(rng, *, max_gaps=10, max_hours=24, min_observed=24):
    """Hide 1-``max_gaps`` gaps of 1-``max_hours`` hours at random starts.

    Gaps may overlap or run past the end of the history. Redraws until at
    least ``min_observed`` hours stay observed. True = hidden.
    """
    while True:
        hidden = np.zeros(HISTORY, dtype=bool)
        for _ in range(int(rng.integers(1, max_gaps + 1))):
            start = int(rng.integers(0, HISTORY))
            hidden[start : start + int(rng.integers(1, max_hours + 1))] = True
        if (~hidden).sum() >= min_observed:
            return hidden


def heavy_gaps(rng, *, missing=(0.5, 0.8), gap_hours=(6, 48)):
    """Hide long gaps at random starts until a random 50-80% is hidden.

    The target fraction is drawn uniformly from ``missing``; gaps of
    ``gap_hours`` hours (up to two days) are added until it is reached. Masks
    that overshoot the upper bound are redrawn. True = hidden.
    """
    low, high = missing
    while True:
        target = rng.uniform(low, high)
        hidden = np.zeros(HISTORY, dtype=bool)
        while hidden.mean() < target:
            start = int(rng.integers(0, HISTORY))
            length = int(rng.integers(gap_hours[0], gap_hours[1] + 1))
            hidden[start : start + length] = True
        if hidden.mean() <= high:
            return hidden


def daily_gaps(start_hour: int, *, first=12, last=23):
    """Hide the same hours (``first``-``last``) every day. True = hidden."""
    hours = (start_hour + np.arange(HISTORY)) % DAY
    return (hours >= first) & (hours <= last)


def gap_masks(n, rng, args, start_hour):
    if args.gaps == "daily":
        return np.tile(daily_gaps(start_hour), (n, 1))
    if args.gaps == "heavy":
        spec = {"missing": args.heavy_missing, "gap_hours": args.heavy_gap_hours}
        return np.stack([heavy_gaps(rng, **spec) for _ in range(n)])
    return np.stack(
        [
            random_gaps(rng, max_gaps=args.max_gaps, max_hours=args.max_gap_hours)
            for _ in range(n)
        ]
    )


def extract(values, chunks):
    """(n, HISTORY + HORIZON) values for (customer, start) pairs."""
    offsets = np.arange(HISTORY + HORIZON)
    return values[chunks[:, 1, None] + offsets, chunks[:, 0, None]]


# ------------------------------------------------------------ filling in


def observed_stats(history, hidden):
    """Per-chunk mean and spread of the observed hours only."""
    observed = np.where(hidden, np.nan, history)
    location = np.nanmean(observed, axis=1)
    spread = np.nanstd(observed, axis=1)
    spread = np.where(spread > 1e-8, spread, 1.0)
    return location, spread


def linear_fill(history, hidden):
    """Straight lines between observed hours; flat beyond the first and last."""
    filled = history.copy()
    times = np.arange(HISTORY)
    for row, mask in zip(filled, hidden, strict=True):
        if mask.any():
            row[mask] = np.interp(times[mask], times[~mask], row[~mask])
    return filled


def build_csdi(args) -> ImputationModel:
    return ImputationModel(
        CSDIDenoiser(
            n_channels=1,
            hidden_channels=args.hidden_channels,
            residual_layers=args.residual_layers,
            attention_heads=args.attention_heads,
            diffusion_embedding_dim=args.embedding_dim,
            time_embedding_dim=args.embedding_dim,
            feature_embedding_dim=8,
            feedforward_dim=args.hidden_channels,
            dropout=0.1,
        )
    )


def training_arrays(history, hidden, protocol, rng_seed):
    """CSDI inputs, loss masks and (selfsup) availability, all (n, T, 1).

    Self-supervised histories have their hidden hours zeroed, so the true
    values there never reach the model.
    """
    location, spread = observed_stats(history, hidden)
    scaled = ((history - location[:, None]) / spread[:, None])[..., None]
    if protocol == "supervised":
        return scaled, hidden[..., None], None
    available = ~hidden[..., None]
    scaled = np.where(available, scaled, 0.0)
    targets = extra_run_targets(available, min_length=1, max_length=2, seed=rng_seed)
    keep = targets.any(axis=(1, 2))
    return scaled[keep], targets[keep], available[keep]


# ---------------------------------------------------------------- Chronos


def chronos_quantiles(pipeline, contexts, starts, *, allow_nan, batch=2048):
    """Quantile forecasts {level: (n, HORIZON)} for (n, HISTORY) contexts."""
    import pandas as pd

    from densiffusion.fx import ELECTRICITY_START

    out = {q: [] for q in FINE_QUANTILES}
    hours = np.arange(HISTORY)
    for first in range(0, len(contexts), batch):
        block = contexts[first : first + batch]
        ids = np.repeat(np.arange(len(block)), HISTORY)
        offsets = starts[first : first + batch, None] - HISTORY + hours
        stamps = ELECTRICITY_START + offsets.astype("timedelta64[h]")
        frame = pd.DataFrame(
            {
                "item_id": ids.astype(str),
                "timestamp": stamps.reshape(-1),
                "target": block.reshape(-1),
            }
        )
        pred = pipeline.predict_df(
            frame,
            prediction_length=HORIZON,
            quantile_levels=list(FINE_QUANTILES),
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            validate_inputs=not allow_nan,
        )
        pred = pred.assign(order=pred["item_id"].astype(int)).sort_values(
            ["order", "timestamp"]
        )
        for q in FINE_QUANTILES:
            out[q].append(pred[str(q)].to_numpy(float).reshape(len(block), HORIZON))
    return {q: np.concatenate(parts) for q, parts in out.items()}


# ---------------------------------------------------------------- scoring


def ensemble_crps(samples, truth):
    """Mean CRPS per chunk for (S, n, H) samples and (n, H) truth."""
    errors = np.sort(samples - truth[None], axis=0)
    count = len(samples)
    weights = (2 * np.arange(1, count + 1) - count - 1) / count**2
    scores = np.abs(errors).mean(axis=0) - np.tensordot(weights, errors, axes=1)
    return scores.mean(axis=1)


def seasonal_naive(history, hidden):
    """Same hour on the most recent day it was observed (else last observed)."""
    forecast = np.empty((len(history), HORIZON))
    for row in range(len(history)):
        observed = np.flatnonzero(~hidden[row])
        for hour in range(HORIZON):
            index = HISTORY - DAY + hour
            while index >= 0 and hidden[row, index]:
                index -= DAY
            forecast[row, hour] = history[row, index if index >= 0 else observed[-1]]
    return forecast


def score_arm(samples, truth, level):
    """Scaled CRPS and scaled absolute error of the median, per chunk."""
    crps = ensemble_crps(samples, truth) / level
    mae = np.abs(np.median(samples, axis=0) - truth).mean(axis=1) / level
    return crps, mae


# -------------------------------------------------------------------- run


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(args.output / "training.log"),
            logging.StreamHandler(),
        ],
    )
    started = time.time()
    torch.manual_seed(args.seed)
    values, _times, meta = load_multivariate_dataset(args.data)
    values = np.asarray(values[0], dtype=np.float64)  # (T, clients)
    if args.customers:
        # Keep the highest-variance customers, as prepare_electricity.py does.
        top = np.sort(np.argsort(-values.var(axis=0), kind="stable")[: args.customers])
        values = values[:, top]
    length, n_clients = values.shape
    train_end, val_end = (int(f * length) for f in SPLITS)
    train_clients, held_clients = split_clients(n_clients, args.holdout, seed=0)

    # Test set: depends only on --test-seed.
    test_rng = np.random.default_rng(args.test_seed)
    groups = {}
    for name, clients in (("seen", train_clients), ("held_out", held_clients)):
        chunks = sample_chunks(
            values,
            clients,
            val_end,
            length,
            args.test_chunks,
            test_rng,
            start_hour=args.start_hour,
        )
        groups[name] = (chunks, gap_masks(len(chunks), test_rng, args, args.start_hour))

    # Training and validation histories: depend on --seed.
    rng = np.random.default_rng(args.seed)
    model = build_csdi(args)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=0.5,
        schedule="quadratic",
        weight_decay=1e-6,
        lr_decay_fractions=(0.75, 0.9),
        validation_steps="sampled",
        validation_interval=args.validation_interval,
        device=args.device,
    )
    checkpoint = args.output / f"csdi_{args.protocol}.pt"
    if checkpoint.exists():
        LOGGER.info("Loading CSDI checkpoint %s", checkpoint)
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        model.to(args.device).load_state_dict(state["model_state_dict"])
        model.eval()
    else:
        sets = []
        for lo, hi, count, seed in (
            (0, train_end, args.train_chunks, args.seed),
            (train_end, val_end, args.val_chunks, args.seed + 1),
        ):
            chunks = sample_chunks(
                values, train_clients, lo, hi, count, rng, start_hour=args.start_hour
            )
            history = extract(values, chunks)[:, :HISTORY]
            hidden = gap_masks(len(chunks), rng, args, args.start_hour)
            sets.append(training_arrays(history, hidden, args.protocol, seed))
        (train, train_mask, train_avail), (val, val_mask, val_avail) = sets
        LOGGER.info(
            "Training %s CSDI on %d histories (%d validation)",
            args.protocol,
            len(train),
            len(val),
        )
        availability = {}
        if train_avail is not None:
            availability = dict(
                train_availability=train_avail, validation_availability=val_avail
            )
        train_densifier(
            model,
            train,
            val,
            config=config,
            dt=1.0,
            seed=args.seed,
            checkpoint=checkpoint,
            train_mask=train_mask,
            validation_mask=val_mask,
            **availability,
        )
    sampler = DDPMSampler(model, replace(config, batch_size=args.sample_batch_size))
    pipeline = load_chronos(args.chronos_model, args.device)

    arrays = {}
    summary = {}
    for name, (chunks, hidden) in groups.items():
        values_chunk = extract(values, chunks)
        history, truth = values_chunk[:, :HISTORY], values_chunk[:, HISTORY:]
        level = np.abs(np.where(hidden, np.nan, history))
        level = np.nanmean(level, axis=1)
        level = np.where(level > 1e-8, level, 1.0)
        location, spread = observed_stats(history, hidden)
        scaled = ((history - location[:, None]) / spread[:, None])[..., None]
        scaled = np.where(hidden[..., None], 0.0, scaled)
        LOGGER.info(
            "Filling %d %s histories with %d draws", len(chunks), name, args.draws
        )
        draws = impute(
            scaled,
            sampler,
            factor=1,
            mask=hidden[..., None],
            n_samples=args.draws,
            seed=args.seed + 17,
        )[..., 0]
        draws = draws * spread[:, None, None] + location[:, None, None]
        median = np.median(draws, axis=1)
        linear = linear_fill(history, hidden)
        gappy = np.where(hidden, np.nan, history)

        # Fill quality on the hidden hours, scaled like the forecasts.
        def fill_error(samples, hidden=hidden, history=history, level=level):
            errors = []
            for row in range(len(history)):
                mask = hidden[row]
                errors.append(
                    ensemble_crps(
                        samples[:, row, mask][:, None], history[row, mask][None]
                    )[0]
                    / level[row]
                )
            return np.array(errors)

        scores = {
            "fill_csdi": fill_error(np.moveaxis(draws, 1, 0)),
            "fill_linear": fill_error(linear[None]),
        }
        contexts = {
            "gappy": gappy,
            "linear": linear,
            "csdi_median": median,
            "complete": history,
        }
        origins = chunks[:, 1] + HISTORY
        for arm, context in contexts.items():
            quantiles = chronos_quantiles(
                pipeline, context, origins, allow_nan=arm == "gappy"
            )
            samples = quantile_samples(quantiles, args.draws * QUANTILE_SAMPLES)
            scores[f"crps_{arm}"], scores[f"mae_{arm}"] = score_arm(
                samples, truth, level
            )
        per_draw = draws.transpose(1, 0, 2).reshape(-1, HISTORY)
        quantiles = chronos_quantiles(
            pipeline, per_draw, np.tile(origins, args.draws), allow_nan=False
        )
        pooled = quantile_samples(quantiles, QUANTILE_SAMPLES)
        pooled = pooled.reshape(QUANTILE_SAMPLES, args.draws, len(chunks), HORIZON)
        pooled = pooled.reshape(-1, len(chunks), HORIZON)
        scores["crps_csdi_draws"], scores["mae_csdi_draws"] = score_arm(
            pooled, truth, level
        )
        naive = seasonal_naive(history, hidden)
        scores["crps_seasonal_naive"], scores["mae_seasonal_naive"] = score_arm(
            naive[None], truth, level
        )
        arrays[name] = {
            "client": chunks[:, 0],
            "start": chunks[:, 1],
            "missing": hidden.mean(axis=1),
            "ends_in_gap": hidden[:, -1],
            **scores,
        }
        summary[name] = {key: float(np.mean(v)) for key, v in scores.items()}
        summary[name]["missing_fraction"] = float(hidden.mean())
        summary[name]["history_ends_in_gap"] = float(hidden[:, -1].mean())
        LOGGER.info("%s: %s", name, json.dumps(summary[name], indent=1))

    np.savez_compressed(
        args.output / "scores.npz",
        **{f"{g}/{k}": v for g, block in arrays.items() for k, v in block.items()},
    )
    results = {
        "summary": summary,
        "n_test_chunks": {g: len(v["client"]) for g, v in arrays.items()},
        "n_clients": {"train": len(train_clients), "held_out": len(held_clients)},
        "elapsed_sec": time.time() - started,
        "meta": meta,
        "args": vars(args),
    }
    (args.output / "results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    return results


# ---------------------------------------------------------------- compare

COMPARISONS = (
    # (label, (config, arm), (config, arm)); positive = the first is better.
    ("C1 CSDI per-draw beats straight lines", "csdi_draws", "linear"),
    ("C2 per-draw forecasts beat the median of the draws", "csdi_draws", "csdi_median"),
    ("C4a straight lines beat leaving the gaps", "linear", "gappy"),
    ("C4b CSDI per-draw beats leaving the gaps", "csdi_draws", "gappy"),
)


def load_config(folder: Path):
    """Seed-averaged per-chunk scores {group: {key: array}} for one config."""
    runs = sorted(folder.glob("seed*/scores.npz"))
    if not runs:
        return None
    loaded = [dict(np.load(path)) for path in runs]
    groups = {key.split("/")[0] for key in loaded[0]}
    out = {}
    for group in groups:
        keys = [k.split("/", 1)[1] for k in loaded[0] if k.startswith(group + "/")]
        out[group] = {
            key: np.mean([run[f"{group}/{key}"] for run in loaded], axis=0)
            for key in keys
        }
        out[group]["seeds"] = len(loaded)
    return out


def cluster_bootstrap(diff, clients, seed=0):
    """Mean of diff with a 95% interval resampling customers."""
    unique, index = np.unique(clients, return_inverse=True)
    sums = np.bincount(index, diff)
    counts = np.bincount(index)
    draws = np.random.default_rng(seed).integers(
        len(unique), size=(BOOTSTRAP, len(unique))
    )
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(low), float(high)


def compare(args):
    configs = {
        folder.name: data
        for folder in sorted(p for p in args.output.iterdir() if p.is_dir())
        if (data := load_config(folder)) is not None
    }
    if not configs:
        raise SystemExit(f"No runs under {args.output}")
    lines = [
        "# Per-customer CSDI + Chronos-2: results",
        "",
        "Scaled CRPS of the day-ahead forecast (CRPS / mean absolute observed "
        "history value; lower is better), averaged over seeds, then over test "
        "chunks. Differences are positive when the first method is better, with "
        "95% intervals from resampling customers.",
    ]
    for group in ("held_out", "seen"):
        lines += ["", f"## {group.replace('_', '-')} customers", ""]
        lines += [
            "| Config | Seeds | Missing | " + " | ".join(ARMS) + " | seasonal naive |",
            "|---" * (len(ARMS) + 4) + "|",
        ]
        for name, data in configs.items():
            block = data[group]
            cells = [f"{block[f'crps_{arm}'].mean():.3f}" for arm in ARMS]
            lines.append(
                f"| {name} | {block['seeds']} | {block['missing'].mean():.0%} | "
                + " | ".join(cells)
                + f" | {block['crps_seasonal_naive'].mean():.3f} |"
            )
        lines += ["", "| Comparison | Config | Difference [95%] |", "|---|---|---|"]
        for name, data in configs.items():
            block = data[group]
            for label, first, second in COMPARISONS:
                diff = block[f"crps_{second}"] - block[f"crps_{first}"]
                mean, low, high = cluster_bootstrap(diff, block["client"])
                lines.append(
                    f"| {label} | {name} | {mean:+.4f} [{low:+.4f}, {high:+.4f}] |"
                )
        for gaps in ("random", "heavy", "daily"):
            selfsup, supervised = f"{gaps}_selfsup", f"{gaps}_supervised"
            if selfsup in configs and supervised in configs:
                a, b = configs[selfsup][group], configs[supervised][group]
                diff = b["crps_csdi_draws"] - a["crps_csdi_draws"]
                mean, low, high = cluster_bootstrap(diff, a["client"])
                lines.append(
                    f"| C3 self-supervised beats supervised CSDI (per-draw) | {gaps} | "
                    f"{mean:+.4f} [{low:+.4f}, {high:+.4f}] |"
                )
        lines += ["", "### By missing fraction (scaled CRPS)", ""]
        bins = (0, 0.15, 0.3, 0.5, 1.01)
        lines += [
            "| Config | Missing | Chunks | " + " | ".join(ARMS) + " |",
            "|---" * (len(ARMS) + 3) + "|",
        ]
        for name, data in configs.items():
            block = data[group]
            for lo, hi in zip(bins, bins[1:], strict=False):
                pick = (block["missing"] >= lo) & (block["missing"] < hi)
                if pick.sum() < 20:
                    continue
                cells = [f"{block[f'crps_{arm}'][pick].mean():.3f}" for arm in ARMS]
                lines.append(
                    f"| {name} | {lo:.0%}-{min(hi, 1):.0%} | {pick.sum()} | "
                    + " | ".join(cells)
                    + " |"
                )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Comparison written to {args.output / 'comparison.md'}")


# ------------------------------------------------------------------- CLI


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", choices=("run", "compare"), default="run")
    parser.add_argument("--data", type=Path, default=Path("data/electricity_all.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=12345)
    parser.add_argument(
        "--protocol", choices=("selfsup", "supervised"), default="selfsup"
    )
    parser.add_argument(
        "--gaps", choices=("random", "heavy", "daily"), default="random"
    )
    parser.add_argument("--max-gaps", type=int, default=10)
    parser.add_argument("--max-gap-hours", type=int, default=24)
    parser.add_argument(
        "--heavy-missing",
        type=float,
        nargs=2,
        default=(0.5, 0.8),
        metavar=("LOW", "HIGH"),
        help="--gaps heavy: range of the hidden fraction",
    )
    parser.add_argument(
        "--heavy-gap-hours",
        type=int,
        nargs=2,
        default=(6, 48),
        metavar=("MIN", "MAX"),
        help="--gaps heavy: range of each gap's length in hours",
    )
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument(
        "--customers",
        type=int,
        default=0,
        help="Use only the N highest-variance (largest) customers; 0 = all",
    )
    parser.add_argument("--train-chunks", type=int, default=40000)
    parser.add_argument("--val-chunks", type=int, default=2000)
    parser.add_argument("--test-chunks", type=int, default=1000, help="per group")
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-batch-size", type=int, default=1024)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--chronos-model", default="autogluon/chronos-2-small")
    args = parser.parse_args(argv)
    if not 0 <= args.start_hour < DAY:
        parser.error("--start-hour must be in [0, 24)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "compare":
        compare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
