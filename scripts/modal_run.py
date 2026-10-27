"""Run a script from scripts/ on a Modal GPU.

One-time setup: install the Modal CLI (`uv tool install modal`), run
`modal setup`, and choose the course workspace. Then, from the repository root:

    modal run --env Densiffusion scripts/modal_run.py \
        --script run_sssd_forecast.py --args "--process fbm --seed 1"

Separate runs with ';' to start them in parallel, one container each. Give each
run its own --output so their files do not collide:

    modal run --env Densiffusion scripts/modal_run.py \
        --script run_sssd_forecast.py \
        --args "--seed 1 --output outputs/seed1; --seed 2 --output outputs/seed2"

Each container installs the locked dependencies (uv.lock with the research
extra), runs the script from the repository root with `--device cuda` unless
another device is given, and keeps everything the script writes under
outputs/ in the shared `densiffusion-outputs` volume. Download results with:

    modal volume get --env Densiffusion densiffusion-outputs <path> <local dir>

Course rules: every function sets max_containers and a timeout, and the team's
first run on anything bigger than an L4 needs a one-line Ed post.
"""

import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = "L4"
TIMEOUT_HOURS = 4
MAX_CONTAINERS = 4

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), extras=["research"])
    .env({"PYTHONPATH": "/repo/src"})
    .add_local_dir(ROOT / "src", "/repo/src")
    .add_local_dir(ROOT / "scripts", "/repo/scripts")
)
volume = modal.Volume.from_name("densiffusion-outputs", create_if_missing=True)
app = modal.App("densiffusion")


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_HOURS * 3600,
    max_containers=MAX_CONTAINERS,
    volumes={"/repo/outputs": volume},
)
def run(script: str, arguments: list[str]) -> int:
    """Run scripts/<script> with arguments; return its exit code."""
    command = [sys.executable, "-u", f"scripts/{script}", *arguments]
    print("running:", shlex.join(command), flush=True)
    start = time.time()
    try:
        return subprocess.run(command, cwd="/repo").returncode
    finally:
        volume.commit()
        print(f"finished after {(time.time() - start) / 60:.1f} min", flush=True)


@app.local_entrypoint()
def main(script: str, args: str = "") -> None:
    if not (ROOT / "scripts" / script).is_file():
        raise SystemExit(f"No such script: scripts/{script}")
    runs = [shlex.split(part) for part in args.split(";")] if args.strip() else [[]]
    if len(runs) > MAX_CONTAINERS:
        raise SystemExit(f"At most {MAX_CONTAINERS} parallel runs")
    for arguments in runs:
        if "--device" not in arguments:
            arguments += ["--device", "cuda"]
    codes = list(run.starmap([(script, arguments) for arguments in runs]))
    for arguments, code in zip(runs, codes, strict=True):
        print(f"exit {code}: {shlex.join(arguments)}")
    if any(codes):
        raise SystemExit(1)
