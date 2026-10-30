#!/bin/bash
# Build the Python environment in node-local storage ($SLURM_TMPDIR).
#
# On multi-node jobs this must run once per node, before the training tasks are
# launched:
#     srun --ntasks-per-node=1 --nodes=$SLURM_NNODES bash scripts/setup_node_env.sh
#
# Building in node-local storage rather than on a shared filesystem avoids
# thousands of small-file reads over the network at import time, which is a
# common source of multi-node startup stalls.
set -euo pipefail

echo "Setting up environment on $(hostname)..."

python -m venv "$SLURM_TMPDIR/env"
# shellcheck disable=SC1091
source "$SLURM_TMPDIR/env/bin/activate"

# Keep compiler/JIT caches node-local as well.
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torch/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --no-index installs from the cluster's local wheelhouse; drop it if the
# compute nodes can reach PyPI.
PIP_FLAGS=(--no-cache-dir)
if [[ -n "${WHEELHOUSE:-}" ]]; then
    PIP_FLAGS+=(--no-index --find-links "$WHEELHOUSE")
fi

pip install --upgrade pip "${PIP_FLAGS[@]}"
pip install "${PIP_FLAGS[@]}" "torch==${TORCH_VERSION:-2.9.1}"
pip install "${PIP_FLAGS[@]}" -r "$NNSSL_REPO/requirements.txt"
pip install "${PIP_FLAGS[@]}" "blosc2>=3.0.0b1"
pip install "${PIP_FLAGS[@]}" maturin batchgeneratorsv2

# Vendored fork of dynamic-network-architectures (see third_party/ notes in the
# README); the pinned release on PyPI does not carry the architectures used here.
pip install "${PIP_FLAGS[@]}" "$NNSSL_REPO/dynamic-network-architectures"

# The package itself, installed without touching its dependency resolution
# (everything above is already pinned).
pip install --no-deps --no-build-isolation -e "$NNSSL_REPO"

python -c "import torch; print(f'torch {torch.__version__} | NCCL {torch.cuda.nccl.version()}')"
echo "Setup complete on $(hostname)"
