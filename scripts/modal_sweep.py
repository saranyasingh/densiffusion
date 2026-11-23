"""Run the quasi-periodic sweep (scripts/run_densiffusion.py) on Modal CPUs.

One container per (model, seed) runs every task in parallel processes, one
thread each, so a container needs about one physical core per task. Jobs are
spawned on a deployed app, so they keep running if the laptop sleeps or
disconnects. From the repository root, with the course profile active:

    modal deploy --env Densiffusion scripts/modal_sweep.py
    uv run --with modal==1.6.4 python scripts/modal_sweep.py launch \
        --models pristi csdi --seeds 1 2 3
    uv run --with modal==1.6.4 python scripts/modal_sweep.py status

Deploy again after changing src/ or scripts/: containers run the deployed
code. The deployed app keeps no warm containers, so it costs nothing while
idle; stop it with `modal app stop --env Densiffusion densiffusion-qp-sweep`.

Results go to <name>/seed<seed>/<model>/<task> in the team's
`densiffusion-outputs` volume, under the `nihar/` folder so they stay clear of
teammates' runs. A task whose results.json exists is skipped and a partial one
is restarted, so relaunching or a retry after preemption resumes the sweep.
Download results with:

    modal volume get --env Densiffusion densiffusion-outputs \
        nihar/<name> outputs/modal/

--sweep jena runs scripts/run_jena.py on the Jena solar-radiation tasks instead
(the data, data/jena/swdr.npz, is packed into the image).

Pass extra run options with --extra, e.g. --extra="--forecaster-epochs 450";
with --extra=--extend, finished tasks are extended instead of skipped. Paths
are generated from fixed data seeds, so --seed changes network
initialisation, training order and sampling, not the data.

Course rules: every function sets max_containers and a timeout. CPU only, so
no Ed post is needed.
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP = "densiffusion-qp-sweep"
ENVIRONMENT = "Densiffusion"
FOLDER = "nihar"
CALLS = ROOT / "outputs" / "modal_calls.json"
TASKS = ("qp_1.6", "qp_2.5", "qp_3", "qp_4", "qp_6", "qp_8", "qp_16")
JENA_TASKS = tuple(f"jena_{r}" for r in ("1.64", "2.57", "3", "3.6", "6", "9", "18"))
SWEEPS = {  # --sweep: (script, tasks)
    "qp": ("run_densiffusion.py", TASKS),
    "jena": ("run_jena.py", JENA_TASKS),
}
CORES = 8
MEMORY_MIB = 16384
TIMEOUT_HOURS = 24  # the course maximum; billing follows actual use
MAX_CONTAINERS = 4  # the team may run 8 at once; leave room for teammates

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), extras=["research"])
    .env({"PYTHONPATH": "/repo/src", "OMP_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", "/repo/src", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "scripts", "/repo/scripts", ignore=["**/__pycache__"])
)
if (ROOT / "data" / "jena" / "swdr.npz").exists():
    # Public CC-BY data (MPI-BGC Jena weather station), 1.7 MB.
    image = image.add_local_file(
        ROOT / "data" / "jena" / "swdr.npz", "/repo/data/jena/swdr.npz"
    )
volume = modal.Volume.from_name("densiffusion-outputs")
app = modal.App(APP, tags={"project": ENVIRONMENT, "purpose": "main"})


@app.function(
    image=image,
    cpu=CORES,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_HOURS * 3600,
    max_containers=MAX_CONTAINERS,
    retries=2,
    volumes={"/repo/outputs": volume},
)
def run_model(
    model: str,
    seed: int,
    output: str,
    tasks: list[str],
    extra: list[str],
    script: str = "run_densiffusion.py",
) -> dict[str, int]:
    """Run every task for one model and seed; return each task's exit code."""
    start = time.time()
    processes = {}
    for task in tasks:
        folder = Path("/repo/outputs", output, task)
        if "--extend" not in extra:
            if (folder / "results.json").exists():
                print(f"{output}/{task}: finished, skipping", flush=True)
                continue
            shutil.rmtree(folder, ignore_errors=True)  # partial run
        command = [
            sys.executable, "-u", f"scripts/{script}", "run",
            "--model", model, "--task", task, "--seed", str(seed),
            "--output", f"outputs/{output}", "--threads", "1", *extra,
        ]  # fmt: skip
        print("running:", shlex.join(command), flush=True)
        processes[task] = subprocess.Popen(command, cwd="/repo")
    try:
        codes = {task: process.wait() for task, process in processes.items()}
    finally:
        volume.commit()
    print(f"{output}: {(time.time() - start) / 60:.1f} min, {codes}", flush=True)
    return codes


def launch(args):
    """Spawn one call per (model, seed) and record the call ids."""
    script, tasks = SWEEPS[args.sweep]
    chosen = args.tasks or list(tasks)
    if not set(chosen) <= set(tasks) or len(chosen) > CORES:
        raise SystemExit(f"Choose at most {CORES} tasks from {', '.join(tasks)}")
    function = modal.Function.from_name(APP, "run_model", environment_name=ENVIRONMENT)
    calls = json.loads(CALLS.read_text()) if CALLS.exists() else {}
    for model in args.models:
        for seed in args.seeds:
            output = f"{FOLDER}/{args.name}/seed{seed}/{model}"
            call = function.spawn(
                model, seed, output, chosen, shlex.split(args.extra), script
            )
            calls[call.object_id] = output
            print(f"spawned {output}: {call.object_id}")
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    CALLS.write_text(json.dumps(calls, indent=2))


def status(_):
    """Report each recorded call: running, or its exit codes."""
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
    start.add_argument("--models", nargs="+", default=["pristi"])
    start.add_argument("--seeds", nargs="+", type=int, default=[42])
    start.add_argument("--sweep", choices=tuple(SWEEPS), default="qp")
    start.add_argument("--tasks", nargs="+", help="default: every task of the sweep")
    start.add_argument("--name", default="qp_seeds")
    start.add_argument("--extra", default="")
    start.set_defaults(func=launch)
    commands.add_parser("status", help="check spawned runs").set_defaults(func=status)
    arguments = parser.parse_args()
    arguments.func(arguments)
