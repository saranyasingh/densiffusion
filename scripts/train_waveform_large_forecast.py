"""Train or evaluate the forecasting comparison after both larger-q gates pass."""

import argparse
import json
from pathlib import Path

from densiffusion.waveform_large_forecast import (
    LargeForecastConfig,
    evaluate_run,
    train_run,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--imputation-source", type=Path)
    parser.add_argument("--development-source", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.phase == "train":
        if args.imputation_source is None:
            parser.error("--imputation-source is required for training")
        settings = json.loads(args.config.read_text()) if args.config else {}
        train_run(
            args.output,
            args.imputation_source,
            LargeForecastConfig(**settings),
            development_sources=args.development_source,
        )
    else:
        evaluate_run(args.output)


if __name__ == "__main__":
    main()
