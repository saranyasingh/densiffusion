#!/bin/bash
# CSDI imputation of sparse FX, then Chronos-2 sparse vs imputed forecast.
#
# One-time setup on a login node (needs internet):
#   uv sync --extra research --extra chronos
#   mkdir -p logs
#   uv run python scripts/prepare_fx.py --output data/fx.npz
#
# Submit from the repository root:
#   sbatch scripts/cluster/fx_csdi_chronos2.sbatch
#   sbatch scripts/cluster/fx_csdi_chronos2.sbatch --sparse-step 4 --horizon 64
#
# Results: outputs/fx_csdi_chronos2_<jobid>/{results.json,imputed.npz,csdi.pt,...}

#SBATCH --job-name=fx-csdi-chronos2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

OUTPUT="${OUTPUT:-outputs/fx_csdi_chronos2_${SLURM_JOB_ID:-local}}"
DATA="${DATA:-data/fx.npz}"

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
# Triton JIT-compiles a CUDA driver shim with gcc, which needs Python.h; the
# compute node's system Python has no dev headers, so point gcc at FASRC's.
export C_INCLUDE_PATH="/n/sw/Miniforge3-25.3.1-0.el9/include/python3.12:${C_INCLUDE_PATH:-}"
nvidia-smi || true

if [[ ! -f "$DATA" ]]; then
    echo "Missing $DATA — run scripts/prepare_fx.py on a login node first" >&2
    exit 1
fi

uv run --extra research --extra chronos python scripts/run_fx_csdi_chronos2.py \
    --data "$DATA" \
    --output "$OUTPUT" \
    --device cuda \
    "$@"
