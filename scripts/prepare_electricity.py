"""Download UCI ElectricityLoadDiagrams20112014 and save hourly client loads.

Run from the repository root:

    uv run python scripts/prepare_electricity.py

Downloads the archive (CC BY 4.0) into data/raw/electricity/, averages each
client's 15-minute readings (kW) to hourly values, keeps --start up to --end,
drops clients that read zero in more than --max-zero-fraction of those hours
(most of these had not started yet), and saves data/electricity_hourly.npz with:

- values: (clients, hours) mean load in kW over each hour
- start: ISO time at which the first hour begins
- clients: column name of each row (MT_001, ...)
- config: JSON with the source, license, units, and processing notes

Hour h covers start + h hours. The source has no daylight-saving gaps: each
day has 96 readings, so a spring-forward hour reads zero and a fall-back hour
sums two hours.
"""

import argparse
import io
import json
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

URL = (
    "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
)
CITATION = (
    "Artur Trindade, ElectricityLoadDiagrams20112014, UCI Machine Learning "
    "Repository (2015)"
)
FIRST_READING = datetime(2011, 1, 1, 0, 15)  # each reading closes a quarter hour


def download(folder: Path, *, refresh: bool) -> Path:
    path = folder / "electricityloaddiagrams20112014.zip"
    if refresh or not path.exists():
        partial = path.with_suffix(".part")
        with (
            urllib.request.urlopen(URL, timeout=120) as reply,
            partial.open("wb") as out,
        ):
            while chunk := reply.read(1 << 20):
                out.write(chunk)
        partial.replace(path)
    return path


def read_readings(archive: Path) -> tuple[list[str], np.ndarray]:
    """Return client names and (readings, clients) 15-minute loads in kW."""
    with zipfile.ZipFile(archive) as files:
        (name,) = [
            n
            for n in files.namelist()
            if n.endswith("LD2011_2014.txt") and not n.startswith("__MACOSX/")
        ]
        text = files.read(name).decode()
    header, body = text.split("\n", 1)
    clients = [column.strip('"') for column in header.split(";")[1:]]
    first = body.split(";", 1)[0].strip('"')
    if datetime.fromisoformat(first) != FIRST_READING:
        raise ValueError(f"Unexpected first reading time {first}")
    readings = np.loadtxt(
        io.StringIO(body.replace(",", ".")),
        delimiter=";",
        usecols=range(1, len(clients) + 1),
    )
    return clients, readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2014-12-31", help="last day kept")
    parser.add_argument("--max-zero-fraction", type=float, default=0.01)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/electricity"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/electricity_hourly.npz")
    )
    parser.add_argument("--refresh", action="store_true", help="download again")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    args.raw.mkdir(parents=True, exist_ok=True)

    clients, readings = read_readings(download(args.raw, refresh=args.refresh))
    # Reading i covers the quarter hour starting at origin + 15 i minutes.
    origin = FIRST_READING - timedelta(minutes=15)
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end) + timedelta(days=1)
    first = int((start - origin) / timedelta(minutes=15))
    count = int((end - start) / timedelta(hours=1))
    if first < 0 or first + 4 * count > len(readings):
        raise SystemExit("--start/--end fall outside the recorded period")
    hourly = readings[first : first + 4 * count].reshape(count, 4, -1).mean(axis=1)
    zero_fraction = (hourly == 0).mean(axis=0)
    kept = np.flatnonzero(zero_fraction <= args.max_zero_fraction)
    values = hourly[:, kept].T

    config = {
        "source": CITATION,
        "url": URL,
        "license": "CC BY 4.0",
        "retrieved": date.today().isoformat(),
        "units": "kW, mean over each hour",
        "start": start.isoformat(),
        "end": args.end,
        "processing": (
            "Hourly means of the 15-minute readings; clients reading zero in "
            f"more than {args.max_zero_fraction:.0%} of hours dropped."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        values=values,
        start=np.array(start.isoformat()),
        clients=np.array([clients[i] for i in kept]),
        config=json.dumps(config, sort_keys=True),
    )
    print(
        f"Saved {len(kept)} of {len(clients)} clients x {count} hours "
        f"({args.start} to {args.end}) to {args.output}"
    )


if __name__ == "__main__":
    main()
