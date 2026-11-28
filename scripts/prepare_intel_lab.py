"""Download Intel Berkeley lab sensors and save temperature-only NPZ.

Source: https://db.csail.mit.edu/labdata/labdata.html
54 motes, ~31s sampling, Feb 28 – Apr 5 2004. Natural packet-loss / mote
failures leave many empty epochs (often cited ~40–50% missing).

This script keeps **temperature only**, one channel per mote, snaps readings
onto a regular minute grid (default 1 min ≈ native rate → ~50% observed),
and writes zeros + an ``observed`` mask for the self-sup Chronos pipeline.

    uv run python scripts/prepare_intel_lab.py --output data/intel_lab.npz
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

INTEL_LAB_URL = "https://db.csail.mit.edu/labdata/data.txt.gz"
INTEL_LAB_START = np.datetime64("2004-02-28T00:00")


def download_intel_lab(url: str = INTEL_LAB_URL) -> np.ndarray:
    """Return structured rows: time, moteid, temperature (finite only)."""
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read()
    text = gzip.decompress(raw).decode("utf-8", errors="replace")
    times = []
    mote_ids = []
    temps = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # date time epoch moteid temperature humidity light voltage
        try:
            stamp = np.datetime64(f"{parts[0]}T{parts[1]}")
            mote = int(float(parts[3]))
            temp = float(parts[4])
        except (ValueError, IndexError):
            continue
        if not (1 <= mote <= 54) or not np.isfinite(temp):
            continue
        # Drop absurd outliers common in this dump (failed sensors).
        if not (-10.0 <= temp <= 50.0):
            continue
        times.append(stamp)
        mote_ids.append(mote)
        temps.append(temp)
    if not times:
        raise ValueError("no valid Intel Lab temperature rows parsed")
    return np.rec.fromarrays(
        [np.asarray(times), np.asarray(mote_ids, dtype=np.int32), np.asarray(temps)],
        names=("time", "mote", "temp"),
    )


def build_minute_grid(
    rows: np.ndarray,
    *,
    bin_minutes: int,
    n_channels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Aggregate temps onto a regular grid; return values, observed, channels, stats."""
    if bin_minutes < 1:
        raise ValueError("bin_minutes must be >= 1")
    start = rows.time.min().astype("datetime64[m]")
    stop = rows.time.max().astype("datetime64[m]")
    # Inclusive minute range, then subsample by bin_minutes.
    all_minutes = np.arange(start, stop + np.timedelta64(1, "m"), dtype="datetime64[m]")
    grid = all_minutes[::bin_minutes]
    n_steps = len(grid)
    if n_steps < 2:
        raise ValueError("grid too short")

    # Map each reading to a grid index.
    reading_minute = rows.time.astype("datetime64[m]")
    # Floor to bin start.
    offsets = ((reading_minute - start) / np.timedelta64(1, "m")).astype(np.int64)
    bin_index = offsets // bin_minutes
    valid = (bin_index >= 0) & (bin_index < n_steps)
    bin_index = bin_index[valid]
    mote = rows.mote[valid]
    temp = rows.temp[valid]

    unique_motes = np.unique(mote)
    # Count observations per mote; keep the densest ``n_channels``.
    counts = {int(m): int(np.sum(mote == m)) for m in unique_motes}
    ranked = sorted(counts, key=lambda m: (-counts[m], m))
    if n_channels > len(ranked):
        raise ValueError(f"only {len(ranked)} motes available; asked for {n_channels}")
    # Tiny seed jitter only for exact-count ties (already broken by mote id).
    _ = seed
    chosen = sorted(ranked[:n_channels])
    channels = [f"mote_{m:02d}" for m in chosen]
    mote_to_col = {m: i for i, m in enumerate(chosen)}

    sums = np.zeros((n_steps, n_channels), dtype=np.float64)
    counts_grid = np.zeros((n_steps, n_channels), dtype=np.int32)
    for index, m, value in zip(bin_index, mote, temp, strict=True):
        col = mote_to_col.get(int(m))
        if col is None:
            continue
        sums[index, col] += value
        counts_grid[index, col] += 1

    observed = counts_grid > 0
    values = np.zeros_like(sums)
    values[observed] = sums[observed] / counts_grid[observed]
    # Force endpoints observed (carry nearest reading) so linear fill / CSDI
    # windows always have boundary conditioning per channel.
    for col in range(n_channels):
        known = np.flatnonzero(observed[:, col])
        if known.size == 0:
            raise ValueError(f"mote {chosen[col]} has no observations on the grid")
        if not observed[0, col]:
            values[0, col] = values[known[0], col]
            observed[0, col] = True
        if not observed[-1, col]:
            values[-1, col] = values[known[-1], col]
            observed[-1, col] = True
    observed_fraction = float(observed.mean())
    stats = {
        "n_steps": n_steps,
        "n_channels": n_channels,
        "bin_minutes": bin_minutes,
        "observed_fraction": observed_fraction,
        "mote_ids": chosen,
        "mote_observation_counts": {str(m): counts[m] for m in chosen},
        "grid_start": str(grid[0]),
        "grid_stop": str(grid[-1]),
    }
    return values[None, :, :], observed[None, :, :], channels, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data/intel_lab.npz"))
    parser.add_argument("--url", default=INTEL_LAB_URL)
    parser.add_argument(
        "--bin-minutes",
        type=int,
        default=1,
        help="Regular grid step. 1 min ≈ native rate and targets ~50%% observed",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=20,
        help="Keep the densest N motes (temperature channels)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite")

    print(f"Downloading {args.url} ...")
    rows = download_intel_lab(args.url)
    print(f"Parsed {len(rows):,} temperature readings")
    values, observed, channels, stats = build_minute_grid(
        rows,
        bin_minutes=args.bin_minutes,
        n_channels=args.n_channels,
        seed=args.seed,
    )
    print(
        f"Grid {stats['n_steps']} x {stats['n_channels']} "
        f"@ {args.bin_minutes}min; observed_fraction={stats['observed_fraction']:.3f}"
    )
    times = np.arange(values.shape[1], dtype=np.float64)
    config = {
        "source": args.url,
        "dataset": "intel_lab_temperature",
        "start": stats["grid_start"],
        "freq": "min" if args.bin_minutes == 1 else f"{args.bin_minutes}min",
        "dt": float(args.bin_minutes),
        "n_series": 1,
        "n_steps": int(values.shape[1]),
        "n_channels": int(values.shape[2]),
        "channels": channels,
        "item_id": "intel_lab",
        "natural_missing": True,
        "bin_minutes": args.bin_minutes,
        "observed_fraction": stats["observed_fraction"],
        "mote_ids": stats["mote_ids"],
        "citation": (
            "Intel Berkeley Research Lab sensor data "
            "(https://db.csail.mit.edu/labdata/labdata.html)"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.parent / f".{args.output.stem}.tmp.npz"
    tmp.unlink(missing_ok=True)
    np.savez_compressed(
        tmp,
        values=values,
        observed=observed,
        times=times,
        channels=np.asarray(channels),
        config=json.dumps(config),
    )
    tmp.replace(args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
