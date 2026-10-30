#!/bin/bash
# =============================================================================
# Denoising: inference
#
# Applies a trained denoising model to a folder of volumes.
#
# Submit from the repository root:
#     INPUT_FOLDER=<dir> OUTPUT_FOLDER=<dir> MODEL_FOLDER=<dir> \
#       sbatch --account=<your-account> scripts/predict_denoise.slurm
#
# The noise preset and residual scaling are read from the checkpoint, so
# inference reproduces the trained model without further flags.
#
# Size --time, --cpus-per-task and --gres to your allocation.
# =============================================================================
#SBATCH --job-name=ultravem_dn_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/dn_predict_%j.out

set -euo pipefail

# shellcheck disable=SC1090
source "${ULTRAVEM_DN_ENV:-$SLURM_SUBMIT_DIR/scripts/env.sh}"

: "${INPUT_FOLDER:?Set INPUT_FOLDER to the directory of volumes to denoise}"
: "${OUTPUT_FOLDER:?Set OUTPUT_FOLDER to the directory to write predictions into}"
: "${MODEL_FOLDER:?Set MODEL_FOLDER to the trained model directory under \$nnUNet_results}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"

mkdir -p logs "$OUTPUT_FOLDER"

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

echo "=============================================="
echo "Denoising inference"
echo "  Input:      $INPUT_FOLDER"
echo "  Output:     $OUTPUT_FOLDER"
echo "  Model:      $MODEL_FOLDER"
echo "  Fold:       $FOLD"
echo "  Checkpoint: $CHECKPOINT"
echo "=============================================="

srun python -u "$NNUNET_REPO/nnunetv2/inference/predict_dn.py" \
    --input_folder  "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --model_folder  "$MODEL_FOLDER" \
    --fold          "$FOLD" \
    --checkpoint    "$CHECKPOINT"

echo "Inference finished with exit code $?"
