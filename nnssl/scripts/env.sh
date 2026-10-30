#!/bin/bash
# Shared environment for the UltraVEM pre-training pipeline.
#
# Copy this file, edit the paths to match your cluster, and point the SLURM
# scripts at it with:  export ULTRAVEM_ENV=/path/to/your/env.sh
#
# Every value below can also be overridden from the environment, so a one-off
# run does not require editing the file:
#   NNSSL_ROOT=/other/scratch sbatch scripts/pretrain_multinode.slurm

# --- Data roots -------------------------------------------------------------
# Parent directory holding the nnssl_{raw,preprocessed,results} trees.
NNSSL_ROOT="${NNSSL_ROOT:-$SCRATCH/ultravem}"

export nnssl_raw="${nnssl_raw:-$NNSSL_ROOT/nnssl_raw}"
export nnssl_preprocessed="${nnssl_preprocessed:-$NNSSL_ROOT/nnssl_preprocessed}"
export nnssl_results="${nnssl_results:-$NNSSL_ROOT/nnssl_results}"

# --- Repository -------------------------------------------------------------
# Root of this repository (the directory containing pyproject.toml).
export NNSSL_REPO="${NNSSL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- Local wheels -----------------------------------------------------------
# Compute nodes on many HPC systems have no outbound network access, so pip is
# run with --no-index against a local wheelhouse. Leave WHEELHOUSE empty to
# install from PyPI instead.
export WHEELHOUSE="${WHEELHOUSE:-}"

# --- Experiment settings ----------------------------------------------------
# Set these for your run. The values below are the ones used for the released
# model (see the README); change them to point at your own dataset and trainer.
# The trainer class determines the architecture, patch size, batch size and
# epoch count -- see src/nnssl/training/nnsslTrainer/ for the available classes.
export DATASET_ID="${DATASET_ID:-006}"
export CONFIGURATION="${CONFIGURATION:-onemmiso}"
export PLANS="${PLANS:-nnsslPlans}"
export TRAINER="${TRAINER:-BaseMAETrainer_BS32_ep10000_Arch_Width_B_Depth_B}"
