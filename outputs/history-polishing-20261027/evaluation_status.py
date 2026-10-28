"""Read-only sampling progress; does not inspect or score forecast values."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parent
primary = root / "primary"
now = datetime.now(UTC)
start = json.loads((primary / "EVALUATION_STARTED.json").read_text())["started_utc"]
print(now.isoformat(), "evaluation elapsed minutes", round((now - datetime.fromisoformat(start)).total_seconds() / 60, 1))
complete = []
active = None
seconds = []
for seed in (28101, 28102, 28103):
    directory = primary / f"seed_{seed}"
    timings = directory / "evaluation_timings.json"
    done = json.loads(timings.read_text())["test_inference_seconds"] if timings.exists() else {}
    for method in ("raw", "learned", "median"):
        if method in done:
            complete.append(f"{seed}/{method}")
            seconds.append(done[method])
        elif active is None:
            active = f"{seed}/{method}"
print("completed ensembles", len(complete), "/ 9", ", ".join(complete))
print("active ensemble", active)
try:
    os.kill(73407, 0)
    print("evaluator process alive", True)
except ProcessLookupError:
    print("evaluator process alive", False)
if seconds:
    print("mean completed ensemble minutes", round(sum(seconds) / len(seconds) / 60, 2))
print("evaluated", (primary / "EVALUATED.json").exists(), "pipeline complete", (root / "PIPELINE_COMPLETE.json").exists())
