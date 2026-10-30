#!/bin/bash
# =============================================================================
# Denoising: training
#
# Every hyperparameter is required (see scripts/env.sh) -- there is no
# built-in default, so fill them in there or pass them on the command line:
#     DATASET=<id> nnUNet_DN_GAUSSIAN_SIGMA=<sigma> nnUNet_DN_POISSON_SCALE=<scale> \
#     nnUNet_DN_BASELINE_SIGMA=<sigma> nnUNet_DN_EPOCHS=<n> nnUNet_DN_LR=<lr> \
#     nnUNet_DN_SSIM_WEIGHT=<weight> nnUNet_DN_GRAD_CLIP_NORM=<norm> \
#     nnUNet_DN_SSIM_WINDOW_SIZE=<size> \
#       sbatch --account=<your-account> scripts/train_denoise.slurm
#
# The training set holds CLEAN volumes only; the noise is synthesised on the fly
# by the trainer, so no paired noisy/clean acquisition is required.
#
# Size --time, --cpus-per-task and --gres to your allocation.
# =============================================================================
#SBATCH --job-name=ultravem_dn_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/dn_train_%j.out

set -euo pipefail

# shellcheck disable=SC1090
source "${ULTRAVEM_DN_ENV:-$SLURM_SUBMIT_DIR/scripts/env.sh}"

if [[ -z "${DATASET:-}" ]]; then
    echo "[ERROR] DATASET is not set. Pass the nnU-Net dataset id or name, e.g." >&2
    echo "        DATASET=<id> sbatch scripts/train_denoise.slurm" >&2
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
echo "Denoising training"
echo "  Dataset:       $DATASET"
echo "  Configuration: $CONFIGURATION"
echo "  Plans:         $PLANS"
echo "  Trainer:       $TRAINER"
echo "  Fold:          $FOLD"
echo "  Gaussian sigma: ${nnUNet_DN_GAUSSIAN_SIGMA:-<NOT SET>}"
echo "  Poisson scale:  ${nnUNet_DN_POISSON_SCALE:-<NOT SET>}"
echo "  Baseline sigma: ${nnUNet_DN_BASELINE_SIGMA:-<NOT SET>}"
echo "  Epochs:         ${nnUNet_DN_EPOCHS:-<NOT SET>}"
echo "  Pretrained:     ${PRETRAINED_WEIGHTS:-<none, training from scratch>}"
echo "=============================================="

srun python -u "$NNUNET_REPO/nnunetv2/run/run_training.py" \
    "$DATASET" "$CONFIGURATION" "$FOLD" \
    -tr "$TRAINER" \
    -p "$PLANS" \
    "${EXTRA_ARGS[@]}"

echo "Training finished with exit code $?"
