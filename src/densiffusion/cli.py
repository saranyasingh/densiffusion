import argparse
from collections.abc import Sequence
from pathlib import Path

from densiffusion.data import GenerationConfig, generate, save_dataset


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="densiffusion")
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser(
        "generate", help="Generate a dataset from a plug-in"
    )
    generation.add_argument("config", type=Path, help="TOML generation config")
    generation.add_argument(
        "--output", type=Path, required=True, help="Output .npz path"
    )
    args = parser.parse_args(argv)

    try:
        config = GenerationConfig.from_toml(args.config)
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        values = generate(config)
        save_dataset(args.output, values, config)
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    print(f"Saved {values.shape} to {args.output}")
