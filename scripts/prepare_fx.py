"""Download the LSTNet daily FX rates and save a densiffusion dataset.

    uv run python scripts/prepare_fx.py --output data/fx.npz

Source: https://github.com/laiguokun/multivariate-time-series-data
(Lai et al., LSTNet) — 8 currencies, daily, 1990–2016.
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

from densiffusion.fx import FX_CURRENCIES, FX_START, FX_URL


def download_exchange_rates(url: str = FX_URL) -> np.ndarray:
    """Return float array shaped (n_steps, n_currencies)."""
    with urllib.request.urlopen(url) as response:
        raw = response.read()
    text = gzip.decompress(raw).decode("utf-8")
    rows = [
        [float(part) for part in line.split(",")]
        for line in text.splitlines()
        if line.strip()
    ]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FX_CURRENCIES):
        raise ValueError(
            f"expected (*, {len(FX_CURRENCIES)}) FX matrix, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("FX file contains non-finite values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data/fx.npz"))
    parser.add_argument("--url", default=FX_URL)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite")

    values = download_exchange_rates(args.url)
    # densiffusion layout: (series, time, channels) with one multivariate series.
    series = values[None, :, :]
    times = np.arange(values.shape[0], dtype=np.float64)
    config = {
        "source": args.url,
        "dataset": "lstnet_exchange_rate",
        "start": str(FX_START),
        "freq": "D",
        "dt": 1.0,
        "n_series": 1,
        "n_steps": int(values.shape[0]),
        "n_channels": int(values.shape[1]),
        "currencies": list(FX_CURRENCIES),
        "citation": (
            "Lai et al., Modeling Long- and Short-Term Temporal Patterns with "
            "Deep Neural Networks (LSTNet), 2018; exchange_rate subset."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # numpy always appends ".npz"; write beside the target then rename.
    tmp = args.output.parent / f".{args.output.stem}.tmp.npz"
    tmp.unlink(missing_ok=True)
    np.savez_compressed(
        tmp,
        values=series,
        times=times,
        channels=np.asarray(FX_CURRENCIES),
        currencies=np.asarray(FX_CURRENCIES),  # legacy key
        config=json.dumps(config, sort_keys=True),
    )
    written = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    written.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shape": list(series.shape),
                "currencies": list(FX_CURRENCIES),
                "n_steps": values.shape[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
