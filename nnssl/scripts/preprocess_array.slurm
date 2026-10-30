#!/bin/bash
# =============================================================================
# UltraVEM pre-training data: planning + preprocessing, fanned out over a job array
#
# Preprocessing a large corpus on a single node is prohibitively slow, so the
# work is split into independent shards; each array task preprocesses one shard
# into $nnssl_preprocessed. The shards are disjoint and order-independent, so
# they can run concurrently and be re-run individually on failure.
#
# Set the shard count with the --array directive below, and size --cpus-per-task
# and --time to your allocation.
#
# Run the fingerprint extraction ONCE first (it must see the whole dataset):
#     sbatch --account=<your-account> scripts/preprocess_array.slurm fingerprint
#
# Then fan out the preprocessing:
#     sbatch --account=<your-account> scripts/preprocess_array.slurm
# =============================================================================
#SBATCH --job-name=ultravem_preprocess
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-19          # shard count; adjust to your corpus and allocation
#SBATCH --output=logs/preprocess_%A_%a.out

set -euo pipefail

# shellcheck disable=SC1090
source "${ULTRAVEM_ENV:-$SLURM_SUBMIT_DIR/scripts/env.sh}"

# Number of shards. Derived from the actual array size so that changing the
# --array directive above cannot silently desynchronise the two: a mismatch
# would make the shards overlap or leave gaps, i.e. silently duplicate or drop
# part of the dataset. Override only when driving this script outside an array.
TOTAL_PARTS="${TOTAL_PARTS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
NUM_PROC="${NUM_PROC:-${SLURM_CPUS_PER_TASK:-8}}"
MODE="${1:-preprocess}"

mkdir -p logs "$nnssl_preprocessed"

module --force purge
module load StdEnv/2023
module load python/3.12.4
module load cuda/12.2
module load opencv
module load rust

export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1

bash "$NNSSL_REPO/scripts/setup_node_env.sh"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"

if [[ "$MODE" == "fingerprint" ]]; then
    # Single pass over the whole dataset; run this before the array job.
    echo "Extracting dataset fingerprint for $DATASET_ID..."
    nnssl_extract_fingerprint -d "$DATASET_ID" -np "$NUM_PROC"
    exit 0
fi

PART="${SLURM_ARRAY_TASK_ID:-0}"
echo "Preprocessing dataset $DATASET_ID -- shard $PART of $TOTAL_PARTS (np=$NUM_PROC)"

# --no_fp: the fingerprint was already extracted above; without it every array
# task would redundantly re-extract the same fingerprint over the full dataset.
nnssl_plan_and_preprocess \
    -d "$DATASET_ID" \
    -np "$NUM_PROC" \
    --no_fp \
    --part "$PART" \
    --total_parts "$TOTAL_PARTS"

echo "Shard $PART done."
