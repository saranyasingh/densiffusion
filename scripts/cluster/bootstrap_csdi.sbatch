#!/bin/bash
# Self-trained CSDI on one GPU. Round 1 and 2 learn from low-resolution images
# only, impute the training set at double size, retrain, and produce clear test
# images. Round 3 then trains a generator on the self-trained images and draws
# new ones from scratch (mnist: generate_digits_csdi.py; div2k:
# generate_div2k_csdi.py). DATASET picks the data (default mnist):
#   mnist: 15x15 -> 29x29 grey digits, then generate new digits
#   div2k: 33x33 -> 65x65 RGB crops, then generate new crops
#
# One-time setup on a login node (needs internet):
#   uv sync --extra research
#   mkdir -p logs
#   uv run python scripts/prepare_digits.py --output data/mnist29.npz          # 12 MB
#   uv run --with pillow python scripts/prepare_div2k.py --output data/div2k65.npz  # 4 GB
#
# Submit from the repository root; extra arguments go to bootstrap_csdi.py and
# the matching Round 3 script, for example the oracle upper bounds:
#   sbatch scripts/cluster/bootstrap_csdi.sbatch --oracle
#   sbatch --export=ALL,DATASET=div2k scripts/cluster/bootstrap_csdi.sbatch --oracle
# To regenerate from a finished run without retraining rounds 1–2:
#   sbatch --export=ALL,RUN=outputs/digits_60074338 \
#       scripts/cluster/generate_digits_csdi.sbatch --oracle
#   sbatch --export=ALL,RUN=outputs/div2k_60201471 \
#       scripts/cluster/generate_div2k_csdi.sbatch --oracle
#
# Results land in outputs/<dataset>_<job id>/ (table.png, examples.png,
# imputed_train.png, results.json, plus generation_table.png / generated.png /
# generation.json). Rerunning with the same --output reuses finished rounds.
# DIV2K round 2 and the oracle attend along 65 x 3 values per row; if the GPU
# runs out of memory, add --batch-size 16.

#SBATCH --job-name=csdi-bootstrap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DATASET="${DATASET:-mnist}"
case "$DATASET" in
    mnist)
        defaults=(--data data/mnist29.npz)
        generate_script=scripts/generate_digits_csdi.py
        generate_defaults=()
        ;;
    div2k)
        defaults=(--data data/div2k65.npz --train-images 10000 --shift-copies 0 --batch-size 32)
        generate_script=scripts/generate_div2k_csdi.py
        generate_defaults=(--data data/div2k65.npz --train-images 10000 --batch-size 32)
        ;;
    *) echo "Unknown DATASET=$DATASET (use mnist or div2k)" >&2; exit 1 ;;
esac

OUTPUT="outputs/${DATASET}_${SLURM_JOB_ID:-local}"

export UV_OFFLINE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"
nvidia-smi || true

uv run --extra research --with matplotlib python scripts/bootstrap_csdi.py \
    "${defaults[@]}" \
    --output "$OUTPUT" \
    --device cuda \
    "$@"

# Round 3: train a generator on the self-trained images and draw new ones.
# Drop bootstrap-only flags that the generate scripts do not accept.
generate_args=()
skip=
for arg in "$@"; do
    if [[ -n "$skip" ]]; then
        skip=
        continue
    fi
    case "$arg" in
        --shift-copies|--imputations|--n-samples|--test-images)
            skip=1
            ;;
        --shift-copies=*|--imputations=*|--n-samples=*|--test-images=*)
            ;;
        *)
            generate_args+=("$arg")
            ;;
    esac
done
uv run --extra research --with matplotlib python "$generate_script" \
    "${generate_defaults[@]}" \
    --output "$OUTPUT" \
    --device cuda \
    "${generate_args[@]}"
