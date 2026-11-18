"""Download LSTNet hourly electricity and save a densiffusion dataset.

    uv run python scripts/prepare_electricity.py --output data/electricity.npz

Source: https://github.com/laiguokun/multivariate-time-series-data
(Lai et al., LSTNet) — 321 clients, hourly, 2012–2014.

By default keeps the ``--n-channels`` highest-variance clients so multivariate
CSDI feature attention stays tractable (full 321 is usually too heavy).
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

from densiffusion.fx import ELECTRICITY_N_CLIENTS, ELECTRICITY_START, ELECTRICITY_URL


def download_electricity(url: str = ELECTRICITY_URL) -> np.ndarray:
    """Return float array shaped (n_steps, n_clients)."""
    with urllib.request.urlopen(url) as response:
        raw = response.read()
    text = gzip.decompress(raw).decode("utf-8")
    rows = [
        [float(part) for part in line.split(",")]
        for line in text.splitlines()
        if line.strip()
    ]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ELECTRICITY_N_CLIENTS:
        raise ValueError(
            f"expected (*, {ELECTRICITY_N_CLIENTS}) electricity matrix, "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("electricity file contains non-finite values")
    return values


def select_channels(
    values: np.ndarray, *, n_channels: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the highest-variance clients (stable tie-break by index)."""
    if not 1 <= n_channels <= values.shape[1]:
        raise ValueError(f"n_channels must be in [1, {values.shape[1]}]")
    variance = values.var(axis=0)
    # Tiny jitter from seed only breaks exact ties reproducibly.
    jitter = np.random.default_rng(seed).uniform(0, 1e-12, size=variance.shape)
    order = np.argsort(-(variance + jitter), kind="stable")
    chosen = np.sort(order[:n_channels])
    return values[:, chosen], chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data/electricity.npz"))
    parser.add_argument("--url", default=ELECTRICITY_URL)
    parser.add_argument(
        "--n-channels",
        type=int,
        default=20,
        help="Number of clients to keep (highest variance). Use 321 for all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional truncate to the last N hours (faster experiments)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite")

    values = download_electricity(args.url)
    values, client_ids = select_channels(
        values, n_channels=args.n_channels, seed=args.seed
    )
    if args.max_steps is not None:
        if args.max_steps < 2:
            raise SystemExit("--max-steps must be at least 2")
        values = values[-args.max_steps :]
    channels = [f"client_{int(index):03d}" for index in client_ids]
    series = values[None, :, :]
    times = np.arange(values.shape[0], dtype=np.float64)
    config = {
        "source": args.url,
        "dataset": "lstnet_electricity",
        "start": str(ELECTRICITY_START),
        "freq": "h",
        "dt": 1.0,
        "n_series": 1,
        "n_steps": int(values.shape[0]),
        "n_channels": int(values.shape[1]),
        "channels": channels,
        "client_ids": [int(index) for index in client_ids],
        "item_id": "electricity",
        "selection": "highest_variance",
        "seed": args.seed,
        "citation": (
            "Lai et al., Modeling Long- and Short-Term Temporal Patterns with "
            "Deep Neural Networks (LSTNet), 2018; electricity subset "
            "(UCI ElectricityLoadDiagrams20112014, hourly)."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.parent / f".{args.output.stem}.tmp.npz"
    tmp.unlink(missing_ok=True)
    np.savez_compressed(
        tmp,
        values=series,
        times=times,
        channels=np.asarray(channels),
        config=json.dumps(config, sort_keys=True),
    )
    written = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    written.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shape": list(series.shape),
                "channels": channels,
                "client_ids": config["client_ids"],
                "n_steps": values.shape[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
