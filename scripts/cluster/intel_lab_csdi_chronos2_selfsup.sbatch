#!/bin/bash
# Self-supervised CSDI → Chronos-2 on Intel Lab temperature (natural missingness).
#
# One-time setup on a login node (needs internet):
#   uv sync --extra research --extra chronos
#   mkdir -p logs
#   uv run python scripts/prepare_intel_lab.py --output data/intel_lab.npz
#
# Submit from the repository root:
#   sbatch scripts/cluster/intel_lab_csdi_chronos2_selfsup.sbatch
#
# Results: outputs/intel_lab_selfsup_<jobid>/

#SBATCH --job-name=intel-selfsup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DATA="${DATA:-data/intel_lab.npz}"
OUTPUT="${OUTPUT:-outputs/intel_lab_selfsup_${SLURM_JOB_ID:-local}}"

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
export C_INCLUDE_PATH="/n/sw/Miniforge3-25.3.1-0.el9/include/python3.12:${C_INCLUDE_PATH:-}"
nvidia-smi || true

if [[ ! -f "$DATA" ]]; then
    echo "Missing $DATA — run scripts/prepare_intel_lab.py on a login node first" >&2
    exit 1
fi

mkdir -p logs "$(dirname "$OUTPUT")"

# --frozen avoids multi-job .venv races if other arrays are running.
uv run --frozen --extra research --extra chronos python \
    scripts/run_csdi_chronos2_selfsup.py \
    --preset intel_lab \
    --data "$DATA" \
    --output "$OUTPUT" \
    --device cuda \
    --ssl-mask-rate 0.25 \
    "$@"
