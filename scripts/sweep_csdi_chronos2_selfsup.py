"""1D threshold sweeps for self-supervised CSDI → Chronos-2.

Sweep A — base sparsity at fixed SSL rate (default 0.25):
  random: keep_probability in {0.1, 0.2, 0.35, 0.5, 0.75}
  block:  observe_length in {4, 6, 8, 12, 18} with period 24

Sweep B — SSL mask rate at fixed base sparsity:
  rates in {0.1, 0.25, 0.4, 0.55, 0.7}
  block default 12/12; random default keep_probability=0.5

Prints or runs one cell (for Slurm array) or the full local sweep:

    # list cells
    uv run python scripts/sweep_csdi_chronos2_selfsup.py --list

    # run cell 3 of sweep A / block
    uv run --extra research --extra chronos python \
        scripts/sweep_csdi_chronos2_selfsup.py \
        --sweep A --missing-mode block --index 3 \
        --data data/electricity.npz --output-root outputs/elec_selfsup \
        --device cuda --supervised-baseline

    # aggregate finished runs into threshold curves
    uv run --with matplotlib python scripts/plot_selfsup_thresholds.py \
        --root outputs/elec_selfsup
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BLOCK_PERIOD = 24
SWEEP_A_RANDOM = (0.1, 0.2, 0.35, 0.5, 0.75)
SWEEP_A_BLOCK_OBSERVE = (4, 6, 8, 12, 18)
SWEEP_B_RATES = (0.1, 0.25, 0.4, 0.55, 0.7)
FIXED_SSL_RATE = 0.25
FIXED_KEEP = 0.5
FIXED_OBSERVE = 12
FIXED_BLOCK = 12


@dataclass(frozen=True)
class Cell:
    sweep: str
    missing_mode: str
    index: int
    label: str
    flags: tuple[str, ...]

    @property
    def dirname(self) -> str:
        return f"elec_selfsup_{self.missing_mode}_{self.sweep}_{self.label}"


def cells_for(sweep: str, missing_mode: str) -> list[Cell]:
    sweep = sweep.upper()
    if sweep == "A":
        if missing_mode == "random":
            return [
                Cell(
                    "A",
                    "random",
                    i,
                    f"keep{p:g}",
                    (
                        "--missing-mode",
                        "random",
                        "--keep-probability",
                        str(p),
                        "--ssl-mask-rate",
                        str(FIXED_SSL_RATE),
                    ),
                )
                for i, p in enumerate(SWEEP_A_RANDOM)
            ]
        return [
            Cell(
                "A",
                "block",
                i,
                f"obs{obs}",
                (
                    "--missing-mode",
                    "block",
                    "--observe-length",
                    str(obs),
                    "--block-length",
                    str(BLOCK_PERIOD - obs),
                    "--ssl-mask-rate",
                    str(FIXED_SSL_RATE),
                ),
            )
            for i, obs in enumerate(SWEEP_A_BLOCK_OBSERVE)
        ]
    if sweep == "B":
        base = (
            (
                "--missing-mode",
                "random",
                "--keep-probability",
                str(FIXED_KEEP),
            )
            if missing_mode == "random"
            else (
                "--missing-mode",
                "block",
                "--observe-length",
                str(FIXED_OBSERVE),
                "--block-length",
                str(FIXED_BLOCK),
            )
        )
        return [
            Cell(
                "B",
                missing_mode,
                i,
                f"ssl{rate:g}",
                (*base, "--ssl-mask-rate", str(rate)),
            )
            for i, rate in enumerate(SWEEP_B_RATES)
        ]
    raise ValueError("sweep must be A or B")


def all_cells() -> list[Cell]:
    out: list[Cell] = []
    for sweep in ("A", "B"):
        for mode in ("block", "random"):
            out.extend(cells_for(sweep, mode))
    return out


def run_cell(cell: Cell, args) -> Path:
    output = Path(args.output_root) / cell.dirname
    if args.job_id:
        output = Path(f"{output}_{args.job_id}")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_csdi_chronos2_selfsup.py"),
        "--preset",
        "electricity",
        "--data",
        str(args.data),
        "--output",
        str(output),
        "--device",
        args.device,
        *cell.flags,
    ]
    if args.supervised_baseline and cell.sweep == "A":
        cmd.append("--supervised-baseline")
    if args.quick:
        cmd.append("--quick")
    if args.extra:
        cmd.extend(args.extra)
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="Print all cells and exit")
    parser.add_argument("--sweep", choices=("A", "B", "a", "b"), default=None)
    parser.add_argument("--missing-mode", choices=("block", "random"), default=None)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="0-based index within the chosen sweep/mode",
    )
    parser.add_argument(
        "--array-task-id",
        type=int,
        default=None,
        help="Flat index into --list order (for Slurm arrays)",
    )
    parser.add_argument("--data", type=Path, default=Path("data/electricity.npz"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--supervised-baseline", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra flags after -- forwarded to the runner",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    catalog = all_cells()
    if args.list:
        for i, cell in enumerate(catalog):
            print(
                json.dumps(
                    {
                        "array_index": i,
                        "sweep": cell.sweep,
                        "missing_mode": cell.missing_mode,
                        "index": cell.index,
                        "label": cell.label,
                        "dirname": cell.dirname,
                        "flags": list(cell.flags),
                    }
                )
            )
        print(f"# total cells: {len(catalog)}", file=sys.stderr)
        return

    if args.array_task_id is not None:
        if not 0 <= args.array_task_id < len(catalog):
            raise SystemExit(
                f"--array-task-id must be in [0, {len(catalog)}); got {args.array_task_id}"
            )
        cell = catalog[args.array_task_id]
    else:
        if args.sweep is None or args.missing_mode is None or args.index is None:
            raise SystemExit(
                "Provide --sweep, --missing-mode, and --index (or --array-task-id / --list)"
            )
        group = cells_for(args.sweep.upper(), args.missing_mode)
        if not 0 <= args.index < len(group):
            raise SystemExit(f"--index must be in [0, {len(group)}); got {args.index}")
        cell = group[args.index]

    output = run_cell(cell, args)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
