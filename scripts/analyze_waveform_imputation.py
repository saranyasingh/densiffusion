"""Recompute larger waveform imputation metrics and render SVG figures."""

import argparse
from pathlib import Path

from densiffusion.waveform_imputation_analysis import (
    analyze_imputation_run,
    analyze_transfer_run,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--transfer", action="store_true")
    args = parser.parse_args()
    function = analyze_transfer_run if args.transfer else analyze_imputation_run
    function(args.output)


if __name__ == "__main__":
    main()
