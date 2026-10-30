#!/bin/bash
# Shared environment for the UltraVEM segmentation task.
#
# Copy this file, edit it for your cluster and your experiment, and point the
# SLURM scripts at it:
#     export ULTRAVEM_SEG_ENV=/path/to/your/env.sh
#
# Every value can also be overridden per submission without editing the file:
#     DATASET=<id> TRAINER=<trainer-class> sbatch scripts/train_segment.slurm

# --- Data roots -------------------------------------------------------------
# Parent directory holding the nnUNet_{raw,preprocessed,results} trees.
NNUNET_ROOT="${NNUNET_ROOT:-$SCRATCH/ultravem}"

export nnUNet_raw="${nnUNet_raw:-$NNUNET_ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_ROOT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$NNUNET_ROOT/nnUNet_results}"

# --- Repositories -----------------------------------------------------------
# The nnU-Net fork this task runs against.
export NNUNET_REPO="${NNUNET_REPO:-$NNUNET_ROOT/nnUNet}"

# --- Local wheels -----------------------------------------------------------
# Compute nodes on many HPC systems have no outbound network access, so pip is
# run with --no-index against a local wheelhouse. Leave empty to use PyPI.
export WHEELHOUSE="${WHEELHOUSE:-}"

# --- Experiment settings ----------------------------------------------------
# Set these for your run. DATASET is the nnU-Net dataset id or name holding your
# labelled (image, label) volumes -- see the README for the raw data layout.
export DATASET="${DATASET:-}"
export CONFIGURATION="${CONFIGURATION:-3d_fullres}"
export PLANS="${PLANS:-nnUNetResEncUNetLPlans}"
# nnU-Net fold to train/predict with: an integer 0-4 for cross-validation, or
# "all" to train once on the full training set (no held-out validation split).
export FOLD="${FOLD:-all}"
# Any nnU-Net trainer class name, e.g. one of the epoch-count variants in
# nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py,
# or plain "nnUNetTrainer" for nnU-Net's own default schedule.
export TRAINER="${TRAINER:-nnUNetTrainer}"

# Optional: initialise the encoder from pre-trained weights (e.g. the UltraVEM
# MAE checkpoint from the pre-training repository). Leave empty to train from
# scratch. Loaded through nnUNet/nnunetv2/run/load_pretrained_weights.py, which
# adapts SSL checkpoint keys/shapes to the segmentation network.
export PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
