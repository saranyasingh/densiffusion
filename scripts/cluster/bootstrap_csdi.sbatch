#!/bin/bash
# Self-trained CSDI super-resolution on one GPU: learns from low-resolution
# images only, imputes the training set at double size, retrains on it, and
# produces clear test images. DATASET picks the data (default mnist):
#   mnist: 15x15 -> 29x29 grey digits, translated copies for round 1
#   div2k: 33x33 -> 65x65 RGB crops of DIV2K photos, 10,000 training crops
#
# One-time setup on a login node (needs internet):
#   uv sync --extra research
#   mkdir -p logs
#   uv run python scripts/prepare_digits.py --output data/mnist29.npz          # 12 MB
#   uv run --with pillow python scripts/prepare_div2k.py --output data/div2k65.npz  # 4 GB
#
# Submit from the repository root; extra arguments go to bootstrap_csdi.py and
# override the defaults below, for example the oracle trained on clear images:
#   sbatch scripts/cluster/bootstrap_csdi.sbatch --oracle
#   sbatch --export=ALL,DATASET=div2k scripts/cluster/bootstrap_csdi.sbatch --oracle
#
# Results land in outputs/<dataset>_<job id>/ (table.png, examples.png,
# imputed_train.png, results.json). Rerunning with the same --output reuses
# finished rounds. DIV2K round 2 and the oracle attend along 65 x 3 values per
# row; if the GPU runs out of memory, add --batch-size 16.

#SBATCH --job-name=csdi-bootstrap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DATASET="${DATASET:-mnist}"
case "$DATASET" in
    mnist) defaults=(--data data/mnist29.npz) ;;
    div2k) defaults=(--data data/div2k65.npz --train-images 10000 --shift-copies 0 --batch-size 32) ;;
    *) echo "Unknown DATASET=$DATASET (use mnist or div2k)" >&2; exit 1 ;;
esac

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
nvidia-smi || true

uv run --extra research --with matplotlib python scripts/bootstrap_csdi.py \
    "${defaults[@]}" \
    --output "outputs/${DATASET}_${SLURM_JOB_ID:-local}" \
    --device cuda \
    "$@"
