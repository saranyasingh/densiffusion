"""Read-only progress summary for the running fixed experiment."""

import json
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parent
primary = root / "primary"
print(datetime.now(UTC).isoformat())
completed = 0
for directory in sorted(primary.glob("seed_*")):
    audit = directory / "training_audit.json"
    finished = json.loads(audit.read_text())["methods"] if audit.exists() else {}
    completed += len(finished)
    for name, result in finished.items():
        print(directory.name, name, "selected", result["selected_epoch"],
              f"CRPS={result['selected_validation_crps']:.6f}")
    for path in sorted(directory.glob("*_training.jsonl")):
        method = path.name.removesuffix("_training.jsonl")
        if method in finished:
            continue
        row = json.loads(path.read_text().splitlines()[-1])
        print(directory.name, method, "active epoch", row["epoch"],
              f"elapsed={row['elapsed_seconds']:.1f}s")
        validation = directory / f"{method}_validation.json"
        if validation.exists():
            scores = json.loads(validation.read_text())
            last = scores[-1]
            best = min(scores, key=lambda x: x["crps"])
            print("last score", last["epoch"], f"{last['crps']:.6f}",
                  "best", best["epoch"], f"{best['crps']:.6f}")
    forecasts = sorted(x.name for x in directory.glob("*_samples.npz"))
    if forecasts:
        print(directory.name, "test ensembles", forecasts)
print("completed forecasters", completed, "/ 9")
print("frozen", (primary / "FROZEN.json").exists(),
      "evaluated", (primary / "EVALUATED.json").exists(),
      "pipeline complete", (root / "PIPELINE_COMPLETE.json").exists())
