#!/bin/bash
# =============================================================================
# Super-resolution: training
#
# Every hyperparameter is required (see scripts/env.sh) -- there is no
# built-in default, so fill them in there or pass them on the command line:
#     DATASET=<id> nnUNet_SR_SCALE=<sx,sy,sz> nnUNet_SR_EPOCHS=<n> \
#     nnUNet_SR_LR=<lr> nnUNet_SR_WEIGHT_DECAY=<wd> nnUNet_SR_GRAD_CLIP_NORM=<norm> \
#     nnUNet_SR_FREEZE_EPOCHS=<n> nnUNet_SR_SSIM_WINDOW_SIZE=<size> nnUNet_SR_WEIGHT_CHARBONNIER=<w> \
#     nnUNet_SR_WEIGHT_SSIM=<w> nnUNet_SR_WEIGHT_PERCEPTUAL=<w> \
#       sbatch --account=<your-account> scripts/train_superres.slurm
#
# Scale and epochs may instead be encoded in TRAINER, e.g.
# TRAINER=nnUNetTrainer_SR_x1y1z5_100ep (see the trainer's docstring).
#
# The training set holds HIGH-RESOLUTION volumes only; the low-resolution input
# is synthesised on the fly by the trainer, so no paired acquisition is needed.
#
# Size --time, --cpus-per-task and --gres to your allocation.
# =============================================================================
#SBATCH --job-name=ultravem_sr_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/sr_train_%j.out

set -euo pipefail

# shellcheck disable=SC1090
source "${ULTRAVEM_SR_ENV:-$SLURM_SUBMIT_DIR/scripts/env.sh}"

if [[ -z "${DATASET:-}" ]]; then
    echo "[ERROR] DATASET is not set. Pass the nnU-Net dataset id or name, e.g." >&2
    echo "        DATASET=<id> sbatch scripts/train_superres.slurm" >&2
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
echo "Super-resolution training"
echo "  Dataset:       $DATASET"
echo "  Configuration: $CONFIGURATION"
echo "  Plans:         $PLANS"
echo "  Trainer:       $TRAINER"
echo "  Fold:          $FOLD"
echo "  Scale:         ${nnUNet_SR_SCALE:-<from trainer name, if encoded there>}"
echo "  Epochs:        ${nnUNet_SR_EPOCHS:-<from trainer name, if encoded there>}"
echo "  Pretrained:    ${PRETRAINED_WEIGHTS:-<none, training from scratch>}"
echo "=============================================="

srun python -u "$NNUNET_REPO/nnunetv2/run/run_training.py" \
    "$DATASET" "$CONFIGURATION" "$FOLD" \
    -tr "$TRAINER" \
    -p "$PLANS" \
    "${EXTRA_ARGS[@]}"

echo "Training finished with exit code $?"
