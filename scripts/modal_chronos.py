"""Run the CSDI + Chronos-2 scripts on Modal L4 GPUs.

Jobs are spawned on a deployed app, so they keep running if the laptop sleeps.
From the repository root, with the course profile active:

    modal deploy --env Densiffusion scripts/modal_chronos.py
    uv run --with modal==1.6.4 python scripts/modal_chronos.py launch \
        --script run_csdi_chronos2.py --name task2 --seeds 0 1 2 \
        --args "--missing-mode block --origin-hour 12 --per-draw-forecasts 16 \
--stitch block"
    uv run --with modal==1.6.4 python scripts/modal_chronos.py status

Each run gets ``--seed``, ``--device cuda``, ``--data`` (the electricity data
prepared once on the volume: 20 highest-variance clients for
run_csdi_chronos2.py, all 321 for run_chronos_chunks.py) and ``--output``
``nihar/chronos/<name>/seed<seed>`` in the team's `densiffusion-outputs` volume.
Both scripts reuse checkpoints in their output folder, so a retried or
relaunched run resumes. Download results with:

    modal volume get --env Densiffusion densiffusion-outputs \
        nihar/chronos/<name> outputs/modal/chronos/

Deploy again after changing src/ or scripts/. The deployed app keeps no warm
containers, so it costs nothing while idle. Course rules: every function sets
max_containers and a timeout; an L4 needs no Ed post.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP = "densiffusion-chronos"
ENVIRONMENT = "Densiffusion"
FOLDER = "nihar/chronos"
DATA = {
    "run_csdi_chronos2.py": ("nihar/data/electricity.npz", 20),
    "run_chronos_chunks.py": ("nihar/data/electricity_all.npz", 321),
}
CALLS = ROOT / "outputs" / "modal_chronos_calls.json"
TIMEOUT_HOURS = 10
MAX_CONTAINERS = 3

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), extras=["research", "chronos"])
    .env({"PYTHONPATH": "/repo/src"})
    .add_local_dir(ROOT / "src", "/repo/src", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "scripts", "/repo/scripts", ignore=["**/__pycache__"])
)
volume = modal.Volume.from_name("densiffusion-outputs")
app = modal.App(APP, tags={"project": ENVIRONMENT, "purpose": "main"})


@app.function(
    image=image,
    timeout=1800,
    max_containers=1,
    volumes={"/repo/outputs": volume},
)
def prepare(path: str, clients: int) -> str:
    """Download and grid the electricity data onto the volume, once."""
    target = Path("/repo/outputs", path)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "scripts/prepare_electricity.py",
            "--output", str(target), "--n-channels", str(clients),
        ]  # fmt: skip
        subprocess.run(command, cwd="/repo", check=True)
        volume.commit()
    return path


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=TIMEOUT_HOURS * 3600,
    max_containers=MAX_CONTAINERS,
    retries=1,
    volumes={"/repo/outputs": volume},
)
def run(script: str, arguments: list[str], output: str) -> int:
    """Run scripts/<script> with arguments; return its exit code."""
    start = time.time()
    command = [
        sys.executable, "-u", f"scripts/{script}", *arguments,
        "--output", f"outputs/{output}", "--device", "cuda",
    ]  # fmt: skip
    print("running:", shlex.join(command), flush=True)
    try:
        code = subprocess.run(command, cwd="/repo").returncode
    finally:
        volume.commit()
    print(f"{output}: {(time.time() - start) / 60:.1f} min, exit {code}", flush=True)
    return code


def launch(args):
    """Prepare the data, then spawn one run per seed and record the call ids."""
    if args.script not in DATA:
        raise SystemExit(f"--script must be one of {', '.join(DATA)}")
    data, clients = DATA[args.script]
    prepare_data = modal.Function.from_name(
        APP, "prepare", environment_name=ENVIRONMENT
    )
    prepare_data.remote(data, clients)
    function = modal.Function.from_name(APP, "run", environment_name=ENVIRONMENT)
    calls = json.loads(CALLS.read_text()) if CALLS.exists() else {}
    for seed in args.seeds:
        output = f"{FOLDER}/{args.name}/seed{seed}"
        arguments = [
            *shlex.split(args.args),
            "--seed",
            str(seed),
            "--data",
            f"outputs/{data}",
        ]
        call = function.spawn(args.script, arguments, output)
        calls[call.object_id] = output
        print(f"spawned {output}: {call.object_id}")
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    CALLS.write_text(json.dumps(calls, indent=2))


def status(_):
    """Report each recorded call: running, or its exit code."""
    for call_id, output in json.loads(CALLS.read_text()).items():
        try:
            state = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            state = "running"
        except Exception as error:  # noqa: BLE001 - report any failure
            state = f"failed: {type(error).__name__}: {error}"
        print(f"{output}: {state}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(required=True)
    start = commands.add_parser("launch", help="spawn runs on the deployed app")
    start.add_argument("--script", required=True)
    start.add_argument("--name", required=True)
    start.add_argument("--seeds", nargs="+", type=int, default=[0])
    start.add_argument("--args", default="")
    start.set_defaults(func=launch)
    commands.add_parser("status", help="check spawned runs").set_defaults(func=status)
    arguments = parser.parse_args()
    arguments.func(arguments)
