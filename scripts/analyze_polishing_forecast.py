"""Analyze saved raw, learned-polished, and median-polished history forecasts."""

import argparse
from pathlib import Path

from densiffusion.polishing_analysis import analyze_polishing_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()
    result = analyze_polishing_run(args.output, args.bootstrap_replicates, args.seed)
    for method, metrics in result["across_seed_descriptive"]["scores"].items():
        print(f"{method}: CRPS={metrics['crps']['mean']:.6f}")
    print(f"Saved analysis: {args.output / 'analysis.json'}")


if __name__ == "__main__":
    main()
