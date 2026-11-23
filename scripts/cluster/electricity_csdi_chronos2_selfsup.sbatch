#!/bin/bash
# Self-supervised CSDI → Chronos-2 1D threshold sweeps on electricity.
#
# One-time setup on a login node (needs internet):
#   uv sync --extra research --extra chronos
#   mkdir -p logs
#   uv run python scripts/prepare_electricity.py --output data/electricity.npz
#
# List cells (20 total: sweeps A/B × block/random):
#   uv run python scripts/sweep_csdi_chronos2_selfsup.py --list
#
# Submit the full array from the repository root:
#   sbatch scripts/cluster/electricity_csdi_chronos2_selfsup.sbatch
#
# Single cell override:
#   sbatch --export=ALL,ARRAY_TASK_ID=3 scripts/cluster/electricity_csdi_chronos2_selfsup.sbatch
#
# Results: outputs/elec_selfsup_<mode>_<sweep>_<label>_<jobid>/

#SBATCH --job-name=elec-selfsup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DATA="${DATA:-data/electricity.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TASK_ID="${ARRAY_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
export C_INCLUDE_PATH="/n/sw/Miniforge3-25.3.1-0.el9/include/python3.12:${C_INCLUDE_PATH:-}"
nvidia-smi || true

if [[ ! -f "$DATA" ]]; then
    echo "Missing $DATA — run scripts/prepare_electricity.py on a login node first" >&2
    exit 1
fi

mkdir -p logs "$OUTPUT_ROOT"

uv run --extra research --extra chronos python scripts/sweep_csdi_chronos2_selfsup.py \
    --array-task-id "$TASK_ID" \
    --data "$DATA" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda \
    --job-id "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${TASK_ID}" \
    --supervised-baseline \
    "$@"
