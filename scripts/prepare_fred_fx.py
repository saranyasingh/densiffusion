"""Download FRED daily exchange rates and save aligned log prices.

Run from the repository root:

    uv run python scripts/prepare_fred_fx.py

Downloads the H.10 series in densiffusion.fred.SERIES (public domain, citation
requested) into data/raw/fred/, keeps the trading days every series has from
--start on, and saves data/fred_fx.npz with:

- values: (series, time, 1) natural logs of the quoted rates
- times: trading-day index 0, 1, 2, ... (holidays removed)
- dates: ISO date of each time step
- series: FRED series id of each row
- config: JSON with the source, citation, units, and processing notes

Load with numpy.load(path, allow_pickle=False). Cut experiment windows with
densiffusion.fred.fx_windows, which scales by the training period only.
"""

import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

from densiffusion.fred import CITATION, CSV_URL, SERIES, align, parse_fred_csv


def download(series_id: str, folder: Path, *, refresh: bool) -> str:
    path = folder / f"{series_id}.csv"
    if refresh or not path.exists():
        with urllib.request.urlopen(CSV_URL.format(series_id), timeout=60) as reply:
            path.write_bytes(reply.read())
    return path.read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="1999-01-04")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/fred"))
    parser.add_argument("--output", type=Path, default=Path("data/fred_fx.npz"))
    parser.add_argument("--refresh", action="store_true", help="download again")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    args.raw.mkdir(parents=True, exist_ok=True)

    rates = {}
    for series_id in SERIES:
        rates[series_id] = parse_fred_csv(
            download(series_id, args.raw, refresh=args.refresh)
        )
        print(f"{series_id}: {len(rates[series_id])} trading days")
    dates, values = align(rates, start=args.start)
    config = {
        "source": CITATION,
        "license": "Public domain, citation requested",
        "retrieved": date.today().isoformat(),
        "units": SERIES,
        "start": dates[0],
        "end": dates[-1],
        "processing": (
            "Natural log of each quoted rate. FRED leaves holidays blank; only "
            "days with a rate for every series are kept, and consecutive kept "
            "days are one time step apart."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        values=np.log(values)[..., None],
        times=np.arange(len(dates), dtype=float),
        dates=np.array(dates),
        series=np.array(list(SERIES)),
        config=json.dumps(config, sort_keys=True),
    )
    print(
        f"Saved {values.shape[0]} series x {len(dates)} days "
        f"({dates[0]} to {dates[-1]}) to {args.output}"
    )


if __name__ == "__main__":
    main()
