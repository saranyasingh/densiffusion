"""Read progress without loading data, forecasts, or changing scientific artifacts."""

import json
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parent
primary = root / "primary"


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


logs = list(primary.glob("seed_*/*_training.jsonl"))
latest = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
last = None
if latest:
    try:
        last = json.loads(latest.read_text().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        pass
trained = sum(len(read(p, {}).get("methods", {})) for p in primary.glob("seed_*/training_audit.json"))
evaluated = sum(len(read(p, {})) for p in primary.glob("seed_*/scores.json"))
result = {
    "utc": datetime.now(UTC).isoformat(),
    "phase": "complete" if (primary / "verification.json").exists() else "evaluation" if (primary / "EVALUATION_STARTED.json").exists() else "training",
    "trained_models": trained,
    "evaluated_models": evaluated,
    "total_models": 9,
    "latest_training_log": str(latest.relative_to(primary)) if latest else None,
    "latest_epoch": last["epoch"] if last else None,
    "last_message": (root / "run.log").read_text().splitlines()[-1],
}
print(json.dumps(result))
