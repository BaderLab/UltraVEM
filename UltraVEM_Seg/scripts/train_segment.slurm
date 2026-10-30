#!/bin/bash
# =============================================================================
# Segmentation: training
#
# Standard, fully-supervised nnU-Net training: no custom trainer is needed for
# this task, only a trainer class picked from nnU-Net's own variants (or the
# default `nnUNetTrainer`) and, optionally, an encoder warm-started from a
# UltraVEM pre-trained checkpoint.
#
# Submit from the repository root:
#     DATASET=<id> sbatch --account=<your-account> scripts/train_segment.slurm
#
# Settings come from scripts/env.sh and can be overridden per submission, e.g.
#     DATASET=<id> TRAINER=<trainer-class> PLANS=<plans-identifier> \
#       sbatch --account=<your-account> scripts/train_segment.slurm
#
# Preprocessing (once per dataset, before the first training run):
#     nnUNetv2_plan_and_preprocess -d <DATASET> -c <CONFIGURATION> \
#         -pl <PLANNER> --verify_dataset_integrity
#
# Size --time, --cpus-per-task and --gres to your allocation.
# =============================================================================
#SBATCH --job-name=ultravem_seg_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/seg_train_%j.out

set -euo pipefail

# shellcheck disable=SC1090
source "${ULTRAVEM_SEG_ENV:-$SLURM_SUBMIT_DIR/scripts/env.sh}"

if [[ -z "${DATASET:-}" ]]; then
    echo "[ERROR] DATASET is not set. Pass the nnU-Net dataset id or name, e.g." >&2
    echo "        DATASET=<id> sbatch scripts/train_segment.slurm" >&2
    exit 1
fi

mkdir -p logs "$nnUNet_results"

module --force purge
module load StdEnv/2023
module load python/3.12.4
module load cuda
module load opencv
module load rust

export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1

bash "$SLURM_SUBMIT_DIR/scripts/setup_env.sh"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"

# Optional warm start from pre-trained encoder weights.
EXTRA_ARGS=()
if [[ -n "${PRETRAINED_WEIGHTS:-}" ]]; then
    if [[ ! -f "$PRETRAINED_WEIGHTS" ]]; then
        echo "[ERROR] PRETRAINED_WEIGHTS is set but does not exist: $PRETRAINED_WEIGHTS" >&2
        exit 1
    fi
    EXTRA_ARGS+=(-pretrained_weights "$PRETRAINED_WEIGHTS")
fi
# Resume from the latest checkpoint when one exists.
[[ "${CONTINUE:-0}" == "1" ]] && EXTRA_ARGS+=(--c)

echo "=============================================="
echo "Segmentation training"
echo "  Dataset:       $DATASET"
echo "  Configuration: $CONFIGURATION"
echo "  Plans:         $PLANS"
echo "  Trainer:       $TRAINER"
echo "  Fold:          $FOLD"
echo "  Pretrained:    ${PRETRAINED_WEIGHTS:-<none, training from scratch>}"
echo "=============================================="

srun python -u "$NNUNET_REPO/nnunetv2/run/run_training.py" \
    "$DATASET" "$CONFIGURATION" "$FOLD" \
    -tr "$TRAINER" \
    -p "$PLANS" \
    "${EXTRA_ARGS[@]}"

echo "Training finished with exit code $?"
