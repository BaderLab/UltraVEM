#!/bin/bash
# =============================================================================
# MAE pre-training -- multi-node DDP
#
# Submit from the repository root:
#     sbatch --account=<your-account> scripts/pretrain_multinode.slurm
#
# The dataset, configuration, trainer and plans come from scripts/env.sh and can
# be overridden per submission, e.g.
#     DATASET_ID=<id> TRAINER=<trainer-class> \
#       sbatch --account=<your-account> scripts/pretrain_multinode.slurm
#
# Scale is set by the #SBATCH directives below. Request one task per GPU:
# --ntasks-per-node must equal --gpus-per-node, since run_training_multinode.py
# runs one process per GPU and derives rank, world size and local rank from
# SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID. Adjust all three directives (and
# --time) to your allocation; the effective world size is nodes x gpus-per-node.
#
# The architecture, patch size, batch size and epoch count are properties of the
# TRAINER class, not of this script -- see the trainer definitions in
# src/nnssl/training/nnsslTrainer/, and the README for the released model's
# settings.
# =============================================================================
#SBATCH --job-name=ultravem_pretrain
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/pretrain_%j_%N.out

set -euo pipefail

# --- Configuration ----------------------------------------------------------
# Submit from the repository root, or point ULTRAVEM_ENV at your edited env.sh.
# shellcheck disable=SC1090
source "${ULTRAVEM_ENV:-$SLURM_SUBMIT_DIR/scripts/env.sh}"

# Continue from the latest checkpoint if one exists. Long pre-trainings span
# many jobs on time-limited clusters, so this is on by default; set CONTINUE=0
# to force a fresh run.
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
if [[ "${CONTINUE:-1}" == "1" ]]; then
    TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --c"
fi

mkdir -p logs "$nnssl_results"

# --- Modules ----------------------------------------------------------------
module --force purge
module load StdEnv/2023
module load python/3.12.4
module load cuda/12.2
module load opencv
module load rust

export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1

# --- Per-node environment build ---------------------------------------------
echo "Building environment on all $SLURM_NNODES nodes..."
srun --ntasks-per-node=1 --nodes="$SLURM_NNODES" bash "$NNSSL_REPO/scripts/setup_node_env.sh"
echo "All nodes ready."

# --- Distributed rendezvous -------------------------------------------------
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=============================================="
echo "Multi-node training: $SLURM_NNODES node(s), $SLURM_NTASKS rank(s) / GPU(s) total"
echo "Master:        $MASTER_ADDR:$MASTER_PORT"
echo "Dataset:       $DATASET_ID / $CONFIGURATION"
echo "Trainer:       $TRAINER"
echo "Plans:         $PLANS"
echo "=============================================="

# --- NCCL -------------------------------------------------------------------
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE=0            # keep InfiniBand enabled for inter-node all-reduce
export TORCH_NCCL_BLOCKING_WAIT=1   # surface collective timeouts as errors instead of hanging

# --- Launch -----------------------------------------------------------------
srun --export=ALL bash -c '
    source "$SLURM_TMPDIR/env/bin/activate"
    export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
    export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
    export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torch/inductor"

    # shellcheck disable=SC2086
    python -u -m nnssl.run.run_training_multinode \
        "$DATASET_ID" "$CONFIGURATION" \
        -tr "$TRAINER" \
        -p "$PLANS" \
        --slurm_multinode \
        $TRAIN_EXTRA_ARGS
'

echo "Training finished with exit code $?"
