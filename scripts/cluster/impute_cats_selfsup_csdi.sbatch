#!/bin/bash
# Self-supervised CSDI super-resolution on AFHQ-v2 cat faces on one GPU:
# trains on 33x33 low-resolution images only, then upscales to 65x65.
#
# Uses the same one-time setup as impute_cats_csdi.sbatch (data/afhq_cats65.npz and
# a logs/ directory). Submit from the repository root; extra arguments go to
# impute_cats_selfsup_csdi.py, for example the supervised x2 results to compare:
#   sbatch scripts/cluster/impute_cats_selfsup_csdi.sbatch
#   sbatch scripts/cluster/impute_cats_selfsup_csdi.sbatch \
#       --supervised outputs/cats_csdi_<job id>/results.json
#
# Results land in outputs/cats_selfsup_<job id>/. Without --supervised, the
# supervised rows can be added later by rerunning with the same --output: the
# checkpoint is reused, so only sampling and scoring run again.

#SBATCH --job-name=csdi-cats-selfsup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
nvidia-smi || true

uv run --extra research --with matplotlib python scripts/impute_cats_selfsup_csdi.py \
    --data data/afhq_cats65.npz \
    --output "outputs/cats_selfsup_${SLURM_JOB_ID:-local}" \
    --device cuda \
    "$@"
