"""Train, analyze/freeze, or diagnose larger waveform densifiers."""

import argparse
import json
from pathlib import Path

from densiffusion.waveform_imputation import (
    ImputationConfig,
    freeze_run,
    train_run,
    transfer_run,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("train", "freeze", "transfer"), default="train"
    )
    parser.add_argument(
        "--data-source", type=Path, default=Path("outputs/waveform-primary-20261027")
    )
    parser.add_argument(
        "--recipe-source", type=Path, default=Path("outputs/csdi-gp-better")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "train":
        settings = json.loads(args.config.read_text()) if args.config else {}
        if args.smoke:
            settings.update(
                epochs=2,
                score_interval=1,
                samples=4,
                seeds=[27101],
                workers=1,
                transfer_examples=4,
                label="smoke",
            )
        train_run(
            args.output,
            args.data_source,
            args.recipe_source,
            ImputationConfig(**settings),
        )
    elif args.phase == "freeze":
        print(json.dumps(freeze_run(args.output), indent=2))
    else:
        print(json.dumps(transfer_run(args.output), indent=2))


if __name__ == "__main__":
    main()
