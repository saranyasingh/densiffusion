"""Run the paired waveform experiment; see docs/waveform_forecasting.md."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from densiffusion.waveform_experiment import WaveformConfig, evaluate_run, train_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.phase in ("train", "all"):
        values = (
            json.loads(args.config.read_text())
            if args.config
            else asdict(WaveformConfig())
        )
        if args.smoke:
            values.update(
                seeds=[27999],
                n_train=16,
                n_validation=8,
                n_test=8,
                epochs=2,
                densifier_epochs=2,
                validation_interval=1,
                densifier_validation_interval=1,
                diffusion_steps=4,
                batch_size=8,
                sample_batch_size=64,
                label="smoke",
            )
        values["seeds"] = tuple(values["seeds"])
        train_run(WaveformConfig(**values), args.output)
    if args.phase in ("evaluate", "all"):
        evaluate_run(args.output)


if __name__ == "__main__":
    main()
