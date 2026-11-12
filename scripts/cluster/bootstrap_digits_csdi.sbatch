#!/bin/bash
# Self-trained CSDI super-resolution of MNIST digits on one GPU: learns from
# 15x15 digits only, imputes the training set to 29x29, retrains on it, and
# produces clear 29x29 test digits.
#
# One-time setup on a login node (needs internet; about 12 MB):
#   uv sync --extra research
#   uv run python scripts/prepare_digits.py --output data/mnist29.npz
#   mkdir -p logs
#
# Submit from the repository root; extra arguments go to
# bootstrap_digits_csdi.py, for example the oracle trained on clear digits:
#   sbatch scripts/cluster/bootstrap_digits_csdi.sbatch
#   sbatch scripts/cluster/bootstrap_digits_csdi.sbatch --oracle --train-images 59500
#
# Results land in outputs/digits_<job id>/ (table.png, examples.png,
# imputed_train.png, results.json). Rerunning with the same --output reuses
# finished rounds.

#SBATCH --job-name=csdi-digits
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

uv run --extra research --with matplotlib python scripts/bootstrap_digits_csdi.py \
    --data data/mnist29.npz \
    --output "outputs/digits_${SLURM_JOB_ID:-local}" \
    --device cuda \
    "$@"
