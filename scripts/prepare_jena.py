"""Download the Jena rooftop weather station's solar radiation onto a 10-min grid.

The Max Planck Institute for Biogeochemistry publishes its rooftop station's
10-minute measurements under CC-BY-4.0
(https://www.bgc-jena.mpg.de/wetter/weather_data.html), one zip per half
year (whole years from 2024). This keeps shortwave downward radiation (SWDR,
W/m^2) and writes

    data/jena/swdr.npz: start (ISO timestamp of the first grid point),
                        values (float, NaN where missing), step_minutes = 10

on a strictly regular grid. Timestamps are the station's local standard time
(no daylight saving), marking the end of each 10-minute interval.

    uv run python scripts/prepare_jena.py
"""

import argparse
import csv
import io
import re
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PAGE = "https://www.bgc-jena.mpg.de/wetter/weather_data.html"
URL = "https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_{part}.zip"
STEP = timedelta(minutes=10)
MISSING = -9999.0


def parts(first: int, last: int):
    """File names for the given years, as listed on the download page.

    Files are half years ("2020a", "2020b") until 2023 and whole years after.
    """
    with urllib.request.urlopen(PAGE, timeout=60) as response:
        page = response.read().decode("latin-1")
    names = sorted(set(re.findall(r"data/mpi_roof_(\d{4}[ab]?)\.zip", page)))
    return [name for name in names if first <= int(name[:4]) <= last]


def download(part: str, folder: Path) -> Path:
    path = folder / f"mpi_roof_{part}.zip"
    if not path.exists():
        with urllib.request.urlopen(URL.format(part=part), timeout=120) as response:
            path.write_bytes(response.read())
    return path


def read_swdr(path: Path):
    """(timestamps, SWDR) from one zip; SWDR is the column starting 'SWDR'."""
    with zipfile.ZipFile(path) as archive:
        (name,) = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        text = archive.read(name).decode("latin-1")
    rows = csv.reader(io.StringIO(text))
    header = [field.strip() for field in next(rows)]
    column = next(i for i, field in enumerate(header) if field.startswith("SWDR"))
    times, values = [], []
    for row in rows:
        if len(row) <= column:
            continue
        times.append(datetime.strptime(row[0].strip(), "%d.%m.%Y %H:%M:%S"))
        values.append(float(row[column]))
    return times, values


def to_grid(times, values):
    """Place readings on a regular 10-min grid; gaps and -9999 become NaN."""
    start = min(times)
    index = np.array([(t - start) // STEP for t in times])
    grid = np.full(index.max() + 1, np.nan)
    readings = np.array(values)
    readings[readings <= MISSING + 1] = np.nan
    grid[index] = readings  # later duplicates overwrite earlier ones
    return start, grid


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-year", type=int, default=2003)
    parser.add_argument("--last-year", type=int, default=2024)
    parser.add_argument("--output", type=Path, default=Path("data/jena"))
    args = parser.parse_args()
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    times, values = [], []
    for part in parts(args.first_year, args.last_year):
        part_times, part_values = read_swdr(download(part, raw))
        times += part_times
        values += part_values
        print(f"{part}: {len(part_times)} readings", flush=True)
    start, grid = to_grid(times, values)
    np.savez_compressed(
        args.output / "swdr.npz",
        start=start.isoformat(),
        values=grid,
        step_minutes=10,
    )
    missing = np.isnan(grid).mean()
    print(f"{grid.size} grid points from {start}, {missing:.2%} missing")


if __name__ == "__main__":
    main()
