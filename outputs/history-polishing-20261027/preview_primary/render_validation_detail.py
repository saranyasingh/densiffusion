"""Reproduce the labeled validation detail SVG from saved analysis only.

Run from the repository root with the existing environment, then render:
  uv run --locked python outputs/history-polishing-20261027/preview_primary/render_validation_detail.py
  qlmanage -t -s 2000 -o outputs/history-polishing-20261027/preview_primary outputs/history-polishing-20261027/preview_primary/validation_crps_epoch50_detail.svg

No original plot, model, sample, score, or configuration is changed.
"""

import copy
import json
from pathlib import Path

from densiffusion.polishing_analysis import _validation_plot


def main():
    preview = Path(__file__).resolve().parent
    root = preview.parent
    analysis = json.loads((root / "primary/analysis.json").read_text())
    detail = copy.deepcopy(analysis)
    for entry in detail["seeds"].values():
        for review in entry["validation"].values():
            review["rows"] = [
                row for row in review["rows"] if row["epoch"] >= 50
            ]
    destination = preview / "validation_crps_epoch50_detail.svg"
    _validation_plot(detail, destination)
    svg = destination.read_text().replace(
        "Noisy-future validation CRPS checkpoint candidates",
        "Validation CRPS detail | epochs 50 through 1200",
    ).replace(
        "Each method selects its minimum recorded validation CRPS; "
        "curves do not certify convergence.",
        "Epoch 10 omitted from this labeled detail view; full-range original "
        "retained. Curves do not certify convergence.",
    ).replace(
        'height="1040" viewBox="0 0 1200 1040"',
        'height="1200" viewBox="0 0 1200 1200"',
    )
    destination.write_text(svg)


if __name__ == "__main__":
    main()
