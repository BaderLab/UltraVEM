#!/bin/bash
# Shared environment for the UltraVEM super-resolution task.
#
# Copy this file, edit it for your cluster and your experiment, and point the
# SLURM scripts at it:
#     export ULTRAVEM_SR_ENV=/path/to/your/env.sh
#
# Every value can also be overridden per submission without editing the file:
#     DATASET=<id> nnUNet_SR_SCALE=<sx,sy,sz> sbatch scripts/train_superres.slurm

# --- Data roots -------------------------------------------------------------
# Parent directory holding the nnUNet_{raw,preprocessed,results} trees.
NNUNET_ROOT="${NNUNET_ROOT:-$SCRATCH/ultravem}"

export nnUNet_raw="${nnUNet_raw:-$NNUNET_ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_ROOT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$NNUNET_ROOT/nnUNet_results}"

# --- Repositories -----------------------------------------------------------
# The nnU-Net fork holding the denoising trainer and inference script.
export NNUNET_REPO="${NNUNET_REPO:-$NNUNET_ROOT/nnUNet}"

# --- Local wheels -----------------------------------------------------------
# Compute nodes on many HPC systems have no outbound network access, so pip is
# run with --no-index against a local wheelhouse. Leave empty to use PyPI.
export WHEELHOUSE="${WHEELHOUSE:-}"

# --- Experiment settings ----------------------------------------------------
# Set these for your run. DATASET is the nnU-Net dataset id or name holding the
# HIGH-RESOLUTION volumes; the low-resolution inputs are synthesised during training.
export DATASET="${DATASET:-}"
export CONFIGURATION="${CONFIGURATION:-3d_fullres}"
export PLANS="${PLANS:-nnUNetPlans}"
export FOLD="${FOLD:-0}"
export TRAINER="${TRAINER:-nnUNetTrainer_SuperResolution}"

# --- Super-resolution hyperparameters ---------------------------------------
# Every one of these is REQUIRED by the trainer -- there is no built-in default,
# so training refuses to start until each is set to a value you chose. Fill them
# in below (or override per submission, see the top of this file).
#
# nnUNet_SR_SCALE is the per-axis scale "sx,sy,sz" in image space: a factor of N
# on an axis means it was acquired N times coarser than the target. Scale and
# epochs can instead be encoded in TRAINER, e.g.
# TRAINER=nnUNetTrainer_SR_x1y1z5_100ep -- leave the two variables below empty
# in that case.
export nnUNet_SR_SCALE="${nnUNet_SR_SCALE:-}"
export nnUNet_SR_EPOCHS="${nnUNet_SR_EPOCHS:-}"
export nnUNet_SR_LR="${nnUNet_SR_LR:-}"                                   # initial learning rate
export nnUNet_SR_WEIGHT_DECAY="${nnUNet_SR_WEIGHT_DECAY:-}"               # optimizer weight decay
export nnUNet_SR_GRAD_CLIP_NORM="${nnUNet_SR_GRAD_CLIP_NORM:-}"           # gradient-norm clipping threshold
export nnUNet_SR_FREEZE_EPOCHS="${nnUNet_SR_FREEZE_EPOCHS:-}"             # epochs to freeze a pretrained encoder (0 disables)
export nnUNet_SR_SSIM_WINDOW_SIZE="${nnUNet_SR_SSIM_WINDOW_SIZE:-}"       # SSIM pooling window, in voxels
export nnUNet_SR_WEIGHT_CHARBONNIER="${nnUNet_SR_WEIGHT_CHARBONNIER:-}"   # weight of the Charbonnier loss term
export nnUNet_SR_WEIGHT_SSIM="${nnUNet_SR_WEIGHT_SSIM:-}"                 # weight of the SSIM loss term (0 disables it)
export nnUNet_SR_WEIGHT_PERCEPTUAL="${nnUNet_SR_WEIGHT_PERCEPTUAL:-}"     # weight of the perceptual loss term (0 disables it)

# Optional: initialise from pre-trained encoder weights (e.g. the UltraVEM MAE
# checkpoint from the nnssl repository). Leave empty to train from scratch.
export PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
