"""Recompute waveform forecast metrics, paired trajectory intervals, and plots."""

import argparse
from pathlib import Path

from densiffusion.waveform_analysis import analyze_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()
    result = analyze_run(args.output, args.bootstrap_replicates, args.seed)
    for method, scores in result["across_seed_descriptive"]["scores"].items():
        print(f"{method}: CRPS={scores['crps']['mean']:.6f}")
    print(
        f"Saved recomputed scores and paired intervals: {args.output / 'analysis.json'}"
    )


if __name__ == "__main__":
    main()
