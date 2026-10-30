#!/bin/bash
# Shared environment for the UltraVEM denoising task.
#
# Copy this file, edit it for your cluster and your experiment, and point the
# SLURM scripts at it:
#     export ULTRAVEM_DN_ENV=/path/to/your/env.sh
#
# Every value can also be overridden per submission without editing the file:
#     DATASET=<id> nnUNet_DN_EPOCHS=<n> sbatch scripts/train_denoise.slurm

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
# CLEAN volumes; the noisy inputs are synthesised during training.
export DATASET="${DATASET:-}"
export CONFIGURATION="${CONFIGURATION:-3d_fullres}"
export PLANS="${PLANS:-nnUNetPlans}"
export FOLD="${FOLD:-0}"
export TRAINER="${TRAINER:-nnUNetTrainer_Denoising}"

# --- Denoising hyperparameters ----------------------------------------------
# Every one of these is REQUIRED by the trainer -- there is no built-in default,
# so training refuses to start until each is set to a value you chose. Fill them
# in below (or override per submission, see the top of this file).
export nnUNet_DN_GAUSSIAN_SIGMA="${nnUNet_DN_GAUSSIAN_SIGMA:-}"     # read-noise std, [0,1] intensity units
export nnUNet_DN_POISSON_SCALE="${nnUNet_DN_POISSON_SCALE:-}"       # photon-count scale (lower = stronger shot noise)
export nnUNet_DN_BASELINE_SIGMA="${nnUNet_DN_BASELINE_SIGMA:-}"     # Gaussian pre-filter std, in voxels
export nnUNet_DN_EPOCHS="${nnUNet_DN_EPOCHS:-}"                     # number of training epochs
export nnUNet_DN_LR="${nnUNet_DN_LR:-}"                             # initial learning rate
export nnUNet_DN_SSIM_WEIGHT="${nnUNet_DN_SSIM_WEIGHT:-}"           # weight of the SSIM loss term (0 disables it)
export nnUNet_DN_GRAD_CLIP_NORM="${nnUNet_DN_GRAD_CLIP_NORM:-}"     # gradient-norm clipping threshold
export nnUNet_DN_SSIM_WINDOW_SIZE="${nnUNet_DN_SSIM_WINDOW_SIZE:-}" # SSIM pooling window, in voxels

# Optional: initialise from pre-trained encoder weights (e.g. the UltraVEM MAE
# checkpoint from the nnssl repository). Leave empty to train from scratch.
export PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
