"""Continue the fixed, authorized experiment after the running training job."""

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from densiffusion.polishing_experiment import METHODS, read, verify_frozen

ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "primary"
TRAINING_PID = 66434


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


while alive(TRAINING_PID):
    time.sleep(2)
if not (PRIMARY / "FROZEN.json").is_file():
    raise RuntimeError("Training exited without a complete freeze; evaluation forbidden")
verify_frozen(PRIMARY)
manifest = read(PRIMARY / "config.json")
assert manifest["experiment"]["n_test"] == 2048
assert manifest["experiment"]["samples"] == 64
assert len(manifest["experiment"]["seeds"]) == 3
for seed in manifest["experiment"]["seeds"]:
    audit = read(PRIMARY / f"seed_{seed}/training_audit.json")
    for method in METHODS:
        item = audit["methods"][method]
        assert item["measured_optimizer_updates"] == 4800
        assert item["epochs_completed"] == 1200
        assert item["parameter_count"] == 413521
        assert item["validation_candidate_epochs"] == [10, 50, 100, 200, 400, 800, 1200]
    assert len({v["initialization_sha256"] for v in audit["methods"].values()}) == 1

commands = [
    ("evaluation", ["uv", "run", "--locked", "python", "scripts/train_polishing_forecast.py",
                    "--phase", "evaluate", "--output", str(PRIMARY)]),
    ("analysis", ["uv", "run", "--locked", "python", "scripts/analyze_polishing_forecast.py",
                  str(PRIMARY)]),
    ("verification", ["uv", "run", "--locked", "python", "scripts/verify_polishing_forecast.py",
                      str(PRIMARY)]),
]
records = []
for phase, command in commands:
    started = datetime.now(UTC).isoformat()
    print(f"{started}: {phase}", flush=True)
    with (ROOT / f"primary_{phase}.log").open("x") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    records.append({"phase": phase, "command": command, "started_utc": started,
                    "finished_utc": datetime.now(UTC).isoformat()})
    (ROOT / "pipeline_progress.json").write_text(json.dumps(records, indent=2) + "\n")
(ROOT / "PIPELINE_COMPLETE.json").write_text(json.dumps(records, indent=2) + "\n")
print("Training, frozen fresh-test evaluation, saved-sample analysis, and verification completed.", flush=True)
