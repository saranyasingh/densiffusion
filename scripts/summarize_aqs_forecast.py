"""Summarise run_aqs_forecast.py results: CSDI arms against the best baseline.

    uv run python scripts/summarize_aqs_forecast.py outputs/aqs_forecast

Reads every results.json under the folder. Per site, the best baseline is the
lowest sparse-era CRPS among persistence, recent_mean, ridge_direct and
fill_then_ridge; CSDI arms are reported as % change from it (negative is
better). "oracle" is the daily-era validation forecaster given the true missing
days, against the best baseline there: the most that perfect filling could buy.
"""

import json
import sys
from pathlib import Path

import numpy as np

BASELINES = ["persistence", "recent_mean", "ridge_direct", "fill_then_ridge"]
OURS = ["csdi", "csdi_fill_then_ridge"]


def main() -> None:
    rows = []
    for path in sorted(Path(sys.argv[1]).rglob("results.json")):
        for site, r in json.loads(path.read_text())["sites"].items():
            test = r["sparse_test"]
            best = min(BASELINES, key=lambda k: test[k]["crps"])
            row = {
                "site": site,
                "name": r["setup"]["name"][:22],
                "n": r["counts"]["test"],
                "best": best,
                "best_crps": test[best]["crps"],
            }
            for k in OURS:
                row[k] = 100 * (test[k]["crps"] / test[best]["crps"] - 1)
            val = r["dense_val"]
            best_val = min(val[k]["crps"] for k in BASELINES)
            oracle = val["oracle_complete_history"]["crps"]
            row["oracle"] = 100 * (oracle / best_val - 1)
            rows.append(row)

    print(
        f"{'site':<12}{'name':<23}{'tests':>6}  {'best baseline':<16}{'crps':>6}"
        f"  {'csdi':>7}  {'csdi+ridge':>10}  {'oracle':>7}"
    )
    for r in sorted(rows, key=lambda r: r["csdi_fill_then_ridge"]):
        print(
            f"{r['site']:<12}{r['name']:<23}{r['n']:>6}  {r['best']:<16}"
            f"{r['best_crps']:>6.2f}  {r['csdi']:>+6.1f}%  "
            f"{r['csdi_fill_then_ridge']:>+9.1f}%  {r['oracle']:>+6.1f}%"
        )
    for k in OURS:
        v = np.array([r[k] for r in rows])
        print(
            f"{k}: median {np.median(v):+.1f}%; wins by >=2%: {(v <= -2).sum()}"
            f"/{len(v)}; any win: {(v < 0).sum()}/{len(v)}; "
            f"loses by >=2%: {(v >= 2).sum()}/{len(v)}"
        )
    counts = {b: sum(r["best"] == b for r in rows) for b in BASELINES}
    print("best baseline counts:", counts)
    oracle = np.median([r["oracle"] for r in rows])
    print(f"oracle vs best baseline: median {oracle:+.1f}%")


if __name__ == "__main__":
    main()
