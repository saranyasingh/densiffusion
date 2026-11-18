"""Back-compat wrapper: FX preset for the generic CSDI→Chronos-2 pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_csdi_chronos2 import main  # noqa: E402

if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--preset" not in argv:
        argv = ["--preset", "fx", *argv]
    if "--data" not in argv:
        argv = ["--data", "data/fx.npz", *argv]
    main(argv)
