#!/bin/bash
# Train and evaluate CSDI super-resolution on Oxford-IIIT Pet cats on one GPU.
#
# One-time setup on a login node (needs internet; compute nodes often lack it):
#   uv sync --extra research
#   uv run --with pillow python scripts/prepare_cats.py --output data/cats65.npz
#   uv run --extra research --with matplotlib python -c "import matplotlib, torch"
#   mkdir -p logs
#
# Submit from the repository root; extra arguments go to impute_cats_csdi.py:
#   sbatch scripts/cluster/impute_cats_csdi.sbatch
#   sbatch scripts/cluster/impute_cats_csdi.sbatch --factor 4 --epochs 300
#
# Results land in outputs/cats_csdi_<job id>/ (table.png, examples.png,
# results.json). Adjust the partition, account, and time limit for your cluster.
# Attention along rows spans 65 pixels x 3 colours; if the GPU runs out of
# memory, add --batch-size 16 or --attention-heads 4.

#SBATCH --job-name=csdi-cats
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
##SBATCH --partition=gpu
##SBATCH --account=your-account

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
nvidia-smi || true

uv run --extra research --with matplotlib python scripts/impute_cats_csdi.py \
    --data data/cats65.npz \
    --output "outputs/cats_csdi_${SLURM_JOB_ID:-local}" \
    --device cuda \
    "$@"
