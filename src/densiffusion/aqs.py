"""EPA AQS daily PM2.5 with naturally dense and sparse sampling eras.

Many PM2.5 monitors ran daily for years and later moved to the EPA's national
1-in-3 or 1-in-6 day calendar (or the reverse). Nothing here removes data: the
sparse days are the agency's own sampling schedule. A site's daily era trains a
model; its later sparse era is where the model is used and scored on the real
readings that exist.

Values are site-day means over monitors (POCs), in micrograms per cubic metre.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from densiffusion.contracts import Array

DAILY_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"
LICENSE = "Public domain (U.S. EPA Air Quality System, pre-generated files)"
CITATION = "U.S. EPA Air Quality System, daily PM2.5 FRM/FEM (parameter 88101)"

DENSE, ONE_IN_THREE, ONE_IN_SIX, OTHER, EMPTY = "D", "3", "6", "?", " "
EARTH_RADIUS_KM = 6371.0


def read_daily_csv(text: Iterable[str]) -> tuple[dict, dict]:
    """Parse one AQS daily file into site-day means and site coordinates.

    Returns ({(site, date): mean}, {site: (lat, lon, name)}). A monitor's day
    appears once per pollutant standard in the file; it counts once here.
    """
    sums: dict[tuple[str, str], list[float]] = {}
    seen: set[tuple[str, str, str]] = set()
    sites: dict[str, tuple[float, float, str]] = {}
    for row in csv.DictReader(text):
        site = f"{row['State Code']}-{row['County Code']}-{row['Site Num']}"
        date = row["Date Local"]
        key = (site, row["POC"], date)
        if key in seen:
            continue
        seen.add(key)
        value = float(row["Arithmetic Mean"])
        total = sums.setdefault((site, date), [0.0, 0])
        total[0] += value
        total[1] += 1
        if site not in sites:
            name = f"{row['City Name']}, {row['State Name']}"
            sites[site] = (float(row["Latitude"]), float(row["Longitude"]), name)
    means = {key: total / count for key, (total, count) in sums.items()}
    return means, sites


def read_daily_zip(path: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as archive:
        (name,) = archive.namelist()
        with archive.open(name) as raw:
            return read_daily_csv(io.TextIOWrapper(raw, encoding="utf-8"))


def assemble(paths: Iterable[Path]) -> dict[str, Array]:
    """Stack yearly files into a (days, sites) array with NaN where unsampled."""
    values: dict[tuple[str, str], float] = {}
    coordinates: dict[str, tuple[float, float, str]] = {}
    for path in sorted(paths):
        means, sites = read_daily_zip(path)
        values.update(means)
        for site, info in sites.items():
            coordinates.setdefault(site, info)
    site_ids = sorted(coordinates)
    dates = sorted({date for _, date in values})
    start, stop = np.datetime64(dates[0]), np.datetime64(dates[-1])
    days = np.arange(start, stop + 1, dtype="datetime64[D]")
    grid = np.full((len(days), len(site_ids)), np.nan, dtype=np.float32)
    column = {site: i for i, site in enumerate(site_ids)}
    for (site, date), value in values.items():
        grid[(np.datetime64(date) - start).astype(int), column[site]] = value
    return {
        "values": grid,
        "dates": days,
        "sites": np.array(site_ids),
        "latitude": np.array([coordinates[s][0] for s in site_ids]),
        "longitude": np.array([coordinates[s][1] for s in site_ids]),
        "names": np.array([coordinates[s][2] for s in site_ids]),
    }


def year_regimes(values: Array, dates: Array) -> tuple[Array, Array]:
    """Classify each site-year by its count of sampled days.

    Returns (years, codes) with codes[site_index] a string over years using
    D daily (>= 300 days), 3 one-in-three (100-130), 6 one-in-six (50-66),
    ? anything else, and a space for no data.
    """
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    unique = np.unique(years)
    counts = np.stack([np.isfinite(values[years == y]).sum(axis=0) for y in unique])
    codes = np.full(counts.shape, OTHER)
    codes[counts >= 300] = DENSE
    codes[(counts >= 100) & (counts <= 130)] = ONE_IN_THREE
    codes[(counts >= 50) & (counts <= 66)] = ONE_IN_SIX
    codes[counts == 0] = EMPTY
    return unique, np.array(["".join(column) for column in codes.T])


@dataclass(frozen=True)
class Eras:
    dense: tuple[int, int]  # first and last daily year, inclusive
    sparse: tuple[int, int]  # first and last 1-in-3 / 1-in-6 year, inclusive

    @property
    def n_dense(self) -> int:
        return self.dense[1] - self.dense[0] + 1

    @property
    def n_sparse(self) -> int:
        return self.sparse[1] - self.sparse[0] + 1


def dense_then_sparse(code: str, years: Array, *, min_dense: int = 4) -> Eras | None:
    """The longest daily run that is followed by a sparse run, if any.

    Up to two unclassified years may separate the runs; they belong to neither.
    """
    best = None
    for match in re.finditer(rf"{DENSE}{{{min_dense},}}[^36]{{0,2}}[36][36?]*", code):
        text = match.group()
        dense_len = len(text) - len(text.lstrip(DENSE))
        tail = text[dense_len:]
        first_sparse = dense_len + next(i for i, c in enumerate(tail) if c in "36")
        last_sparse = dense_len + max(i for i, c in enumerate(tail) if c in "36")
        start = match.start()
        eras = Eras(
            dense=(int(years[start]), int(years[start + dense_len - 1])),
            sparse=(int(years[start + first_sparse]), int(years[start + last_sparse])),
        )
        if best is None or (eras.n_sparse, eras.n_dense) > (
            best.n_sparse,
            best.n_dense,
        ):
            best = eras
    return best


def distances_km(latitude: Array, longitude: Array, index: int) -> Array:
    lat, lon = np.radians(latitude), np.radians(longitude)
    cos = np.sin(lat[index]) * np.sin(lat) + np.cos(lat[index]) * np.cos(lat) * np.cos(
        lon - lon[index]
    )
    distance = EARTH_RADIUS_KM * np.arccos(np.clip(cos, -1, 1))
    distance[index] = np.inf
    return distance


def era_slice(dates: Array, years: tuple[int, int]) -> slice:
    start = np.datetime64(f"{years[0]}-01-01")
    stop = np.datetime64(f"{years[1] + 1}-01-01")
    return slice(int(np.searchsorted(dates, start)), int(np.searchsorted(dates, stop)))


def choose_neighbours(
    data: dict[str, Array],
    target: int,
    eras: Eras,
    *,
    max_km: float = 100.0,
    count: int = 8,
    min_coverage: float = 0.3,
) -> list[int]:
    """Nearest sites that report on enough days in both of the target's eras."""
    distance = distances_km(data["latitude"], data["longitude"], target)
    dense = era_slice(data["dates"], eras.dense)
    sparse = era_slice(data["dates"], eras.sparse)
    available = np.isfinite(data["values"])
    chosen = []
    for other in np.argsort(distance):
        if distance[other] > max_km or len(chosen) == count:
            break
        if (
            available[dense, other].mean() >= min_coverage
            and available[sparse, other].mean() >= min_coverage
        ):
            chosen.append(int(other))
    return chosen


def gap_coverage(
    data: dict[str, Array], target: int, neighbours: list[int], eras: Eras
) -> float:
    """Share of the target's missing sparse-era days on which a neighbour reported."""
    if not neighbours:
        return 0.0
    sparse = era_slice(data["dates"], eras.sparse)
    values = data["values"][sparse]
    missing = ~np.isfinite(values[:, target])
    if not missing.any():
        return 1.0
    return float(np.isfinite(values[missing][:, neighbours]).any(axis=1).mean())
