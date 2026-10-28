"""Run the prospective resource-matched history-polishing comparison."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from densiffusion.budgeted_polishing import (
    BudgetConfig,
    evaluate_run,
    smoke_config,
    train_run,
)
from densiffusion.polishing_analysis import analyze_polishing_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--phase", choices=("all", "train", "evaluate", "analyze"), default="all"
    )
    args = parser.parse_args()
    if args.config and args.smoke:
        parser.error("Choose configuration or smoke")
    if args.phase in ("all", "train"):
        config = (
            smoke_config()
            if args.smoke
            else BudgetConfig(**json.loads(args.config.read_text()))
            if args.config
            else BudgetConfig()
        )
        train_run(args.output, config)
    if args.phase in ("all", "evaluate"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_budgeted_polishing.py")),
                str(args.output),
                "--training-only",
            ],
            check=True,
        )
        evaluate_run(args.output)
    if args.phase in ("all", "analyze"):
        analyze_polishing_run(
            args.output, bootstrap_replicates=1000 if args.smoke else 10000
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_budgeted_polishing.py")),
                str(args.output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
