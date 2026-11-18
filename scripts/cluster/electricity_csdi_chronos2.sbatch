#!/bin/bash
# CSDI imputation of sparse electricity, then Chronos-2 sparse vs imputed forecast.
#
# One-time setup on a login node (needs internet):
#   uv sync --extra research --extra chronos
#   mkdir -p logs
#   uv run python scripts/prepare_electricity.py --output data/electricity.npz
#   # prefetch Chronos weights once, e.g.:
#   # uv run --extra chronos python -c "from chronos import Chronos2Pipeline; Chronos2Pipeline.from_pretrained('amazon/chronos-2')"
#
# Submit from the repository root:
#   sbatch scripts/cluster/electricity_csdi_chronos2.sbatch
#   sbatch scripts/cluster/electricity_csdi_chronos2.sbatch --n-samples 8
#
# Results: outputs/electricity_csdi_chronos2_<jobid>/{results.json,imputed.npz,...}

#SBATCH --job-name=elec-csdi-chronos2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

OUTPUT="${OUTPUT:-outputs/electricity_csdi_chronos2_${SLURM_JOB_ID:-local}}"
DATA="${DATA:-data/electricity.npz}"

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
nvidia-smi || true

if [[ ! -f "$DATA" ]]; then
    echo "Missing $DATA — run scripts/prepare_electricity.py on a login node first" >&2
    exit 1
fi

uv run --extra research --extra chronos python scripts/run_csdi_chronos2.py \
    --preset electricity \
    --data "$DATA" \
    --output "$OUTPUT" \
    --device cuda \
    "$@"
