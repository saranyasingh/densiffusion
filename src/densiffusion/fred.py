"""Daily exchange rates from FRED (Federal Reserve H.10), ready for windowing.

Every series here is public domain, citation requested: Board of Governors of
the Federal Reserve System (US), H.10 Foreign Exchange Rates, retrieved from
FRED, Federal Reserve Bank of St. Louis. Rates are noon buying rates in New
York, quoted as in each series' units (some per U.S. dollar, some per foreign
unit); taking logs makes the quoting direction only a sign.

Holidays are blank in FRED, so they are dropped and consecutive trading days
are treated as one time step apart (trading-day time).
"""

import csv
import io

import numpy as np

from densiffusion.contracts import Array
from densiffusion.data import make_windows, split_data

SERIES = {
    "DEXUSEU": "U.S. dollars per euro",
    "DEXJPUS": "Japanese yen per U.S. dollar",
    "DEXUSUK": "U.S. dollars per U.K. pound",
    "DEXCAUS": "Canadian dollars per U.S. dollar",
    "DEXSZUS": "Swiss francs per U.S. dollar",
    "DEXUSAL": "U.S. dollars per Australian dollar",
    "DEXUSNZ": "U.S. dollars per New Zealand dollar",
    "DEXSDUS": "Swedish kronor per U.S. dollar",
    "DEXNOUS": "Norwegian kroner per U.S. dollar",
}
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CITATION = (
    "Board of Governors of the Federal Reserve System (US), H.10 Foreign "
    "Exchange Rates, retrieved from FRED, Federal Reserve Bank of St. Louis"
)


def parse_fred_csv(text: str) -> dict[str, float]:
    """Map ISO dates to rates from a fredgraph CSV, skipping blank holidays."""
    rows = csv.reader(io.StringIO(text))
    next(rows)  # header: observation_date,<series id>
    values = {}
    for date, value in rows:
        if value.strip() in ("", "."):
            continue
        rate = float(value)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"Exchange rates must be finite and positive: {date}")
        values[date] = rate
    return values


def align(series: dict[str, dict[str, float]], *, start: str) -> tuple[list, Array]:
    """Return sorted dates from `start` that every series has, and their values.

    Values are shaped (number of series, number of dates), rows in the order of
    `series`. ISO dates sort correctly as strings.
    """
    common = set.intersection(*(set(values) for values in series.values()))
    dates = sorted(date for date in common if date >= start)
    values = np.array([[rates[date] for date in dates] for rates in series.values()])
    return dates, values


def fx_windows(
    log_prices: Array,
    *,
    window_length: int,
    stride: int = 1,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict:
    """Split (series, time) log prices by time, scale, and cut windows.

    Each series is divided by the standard deviation of its daily moves in the
    training period only, so a daily move is about one unit; each window is
    then shifted to start at 0. Returns "scales" and, per split ("train",
    "validation", "test"), "values" (windows, window_length, 1), "times"
    (trading-day indices) and "series" (row of `log_prices` per window).
    """
    log_prices = np.asarray(log_prices, dtype=float)
    n_series, length = log_prices.shape
    times = np.broadcast_to(np.arange(length, dtype=float), log_prices.shape)
    parts = split_data(
        log_prices[..., None],
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        axis="time",
        times=times,
    )
    scales = np.diff(parts[0][0][..., 0], axis=1).std(axis=1)
    if not (scales > 0).all():
        raise ValueError("Every series must move during the training period")
    result = {"scales": scales}
    for name, (values, part_times) in zip(
        ("train", "validation", "test"), parts, strict=True
    ):
        windows, window_times = make_windows(
            values / scales[:, None, None],
            window_length=window_length,
            stride=stride,
            times=part_times,
        )
        per_series = len(windows) // n_series
        result[name] = {
            "values": windows - windows[:, :1],
            "times": window_times,
            "series": np.repeat(np.arange(n_series), per_series),
        }
    return result
