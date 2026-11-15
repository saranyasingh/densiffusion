#!/bin/bash
# Round 3 only: trains a CSDI that draws new 29x29 digits from scratch on
# round 2's super-resolved training digits, then scores the new digits.
# bootstrap_csdi.sbatch already runs this after rounds 1–2 for DATASET=mnist;
# use this job to regenerate from a finished run without retraining those
# rounds. RUN needs round2.pt (or imputed.npz with --source imputed):
#   sbatch --export=ALL,RUN=outputs/digits_60074338 \
#       scripts/cluster/generate_digits_csdi.sbatch --oracle
# Extra arguments go to generate_digits_csdi.py, for example
# --source imputed to train on the round 1 imputations instead.
#
# Results land in RUN (generation_table.png, generated.png,
# generated_grid.png, generation.json). Reruns reuse finished checkpoints.

#SBATCH --job-name=csdi-generate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

: "${RUN:?Set RUN to a bootstrap_csdi.py output directory}"

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
nvidia-smi || true

uv run --extra research --with matplotlib python scripts/generate_digits_csdi.py \
    --output "$RUN" \
    --device cuda \
    "$@"
