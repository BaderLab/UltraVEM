#!/bin/bash
# =============================================================================
# Segmentation: inference
#
# Uses nnU-Net's own prediction entry point -- no custom inference script is
# needed for this task.
#
# Submit from the repository root:
#     INPUT_FOLDER=<dir> OUTPUT_FOLDER=<dir> MODEL_FOLDER=<dir> \
#       sbatch --account=<your-account> scripts/predict_segment.slurm
#
# MODEL_FOLDER is the trainer output directory, i.e.
#     $nnUNet_results/<Dataset>/<trainer>__<plans>__<configuration>
#
# Size --time, --cpus-per-task and --gres to your allocation.
# =============================================================================
#SBATCH --job-name=ultravem_seg_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/seg_predict_%j.out

set -euo pipefail

# shellcheck disable=SC1090
source "${ULTRAVEM_SEG_ENV:-$SLURM_SUBMIT_DIR/scripts/env.sh}"

: "${INPUT_FOLDER:?Set INPUT_FOLDER to the directory of volumes to segment}"
: "${OUTPUT_FOLDER:?Set OUTPUT_FOLDER to the directory to write predictions into}"
: "${MODEL_FOLDER:?Set MODEL_FOLDER to the trained model directory under \$nnUNet_results}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"

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
echo "Segmentation inference"
echo "  Input:      $INPUT_FOLDER"
echo "  Output:     $OUTPUT_FOLDER"
echo "  Model:      $MODEL_FOLDER"
echo "  Fold:       $FOLD"
echo "  Checkpoint: $CHECKPOINT"
echo "=============================================="

# nnUNetv2_predict_from_modelfolder is the console entry point that pip
# installed from the nnU-Net repo above; it takes the trainer output directory
# directly (-m), unlike running predict_from_raw_data.py as a script, which
# dispatches to the -d/-c/-tr/-p variant instead.
srun nnUNetv2_predict_from_modelfolder \
    -i "$INPUT_FOLDER" \
    -o "$OUTPUT_FOLDER" \
    -m "$MODEL_FOLDER" \
    -f "$FOLD" \
    -chk "$CHECKPOINT"

echo "Inference finished with exit code $?"
