"""Download EPA AQS daily PM2.5 (1999-2025) and save data/aqs_pm25.npz.

    uv run python scripts/prepare_aqs.py

Yearly zips are cached in data/raw/aqs/. The saved file holds a (days, sites)
array of site-day means with NaN on unsampled days, the dates, site ids,
coordinates and names, plus each site's year-by-year sampling regime (D daily,
3 one-in-three, 6 one-in-six; see densiffusion.aqs.year_regimes). It also lists
the sites whose daily era is followed by a sparse era, which the forecasting
experiment uses.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from densiffusion.aqs import (
    CITATION,
    DAILY_URL,
    LICENSE,
    assemble,
    choose_neighbours,
    dense_then_sparse,
    gap_coverage,
    year_regimes,
)


def download(year: int, folder: Path) -> Path:
    path = folder / f"daily_88101_{year}.zip"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            DAILY_URL.format(year=year), headers={"User-Agent": "densiffusion-research"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            partial = path.with_suffix(".part")
            partial.write_bytes(response.read())
            partial.rename(path)
        print(f"downloaded {path.name}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first-year", type=int, default=1999)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/aqs"))
    parser.add_argument("--output", type=Path, default=Path("data/aqs_pm25.npz"))
    args = parser.parse_args()

    paths = [download(y, args.raw) for y in range(args.first_year, args.last_year + 1)]
    data = assemble(paths)
    years, codes = year_regimes(data["values"], data["dates"])

    candidates = []
    for index, code in enumerate(codes):
        eras = dense_then_sparse(code, years)
        if eras is None:
            continue
        neighbours = choose_neighbours(data, index, eras)
        candidates.append(
            {
                "site": str(data["sites"][index]),
                "name": str(data["names"][index]),
                "dense": list(eras.dense),
                "sparse": list(eras.sparse),
                "neighbours": [str(data["sites"][n]) for n in neighbours],
                "gap_coverage": round(gap_coverage(data, index, neighbours, eras), 3),
                "regimes": code,
            }
        )
    candidates.sort(key=lambda c: (-c["gap_coverage"], c["sparse"][0] - c["sparse"][1]))

    config = {
        "source": DAILY_URL,
        "license": LICENSE,
        "citation": CITATION,
        "years": [int(years[0]), int(years[-1])],
        "regime_years": [int(y) for y in years],
        "dense_then_sparse": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **data,
        regimes=codes,
        config=json.dumps(config),
    )
    good = [
        c
        for c in candidates
        if c["gap_coverage"] >= 0.9 and c["sparse"][1] - c["sparse"][0] >= 2
    ]
    print(
        f"saved {args.output}: {data['values'].shape[1]} sites x "
        f"{len(data['dates'])} days; "
        f"{len(candidates)} dense-then-sparse sites, {len(good)} with >=3 sparse years "
        "and >=90% neighbour coverage"
    )


if __name__ == "__main__":
    main()
