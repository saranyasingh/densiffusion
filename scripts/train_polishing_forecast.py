"""Run the independent noisy-history polishing experiment in a fresh directory."""

import argparse
import json
from pathlib import Path

from densiffusion.polishing_experiment import PolishingConfig, evaluate_run, train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--recipe-source", type=Path, default=Path("outputs/csdi-gp-better")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "evaluate":
        evaluate_run(args.output)
    else:
        settings = json.loads(args.config.read_text()) if args.config else {}
        if args.smoke:
            settings.update(
                seeds=[28101],
                namespace=2026102799,
                n_train=16,
                n_validation=8,
                n_test=8,
                epochs=2,
                validation_interval=1,
                validation_epochs=[1, 2],
                samples=4,
                diffusion_steps=2,
                label="smoke",
            )
        train_run(args.output, PolishingConfig(**settings), args.recipe_source)


if __name__ == "__main__":
    main()
