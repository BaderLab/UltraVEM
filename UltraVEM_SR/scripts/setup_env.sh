#!/bin/bash
# Build the Python environment in node-local storage ($SLURM_TMPDIR).
#
# Building in node-local storage rather than on a shared filesystem avoids
# thousands of small-file reads over the network at import time, a common source
# of slow job startup.
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

PIP_FLAGS=(--no-cache-dir)
if [[ -n "${WHEELHOUSE:-}" ]]; then
    PIP_FLAGS+=(--no-index --find-links "$WHEELHOUSE")
fi

pip install --upgrade pip "${PIP_FLAGS[@]}"
pip install "${PIP_FLAGS[@]}" -e "$NNUNET_REPO"
pip install "${PIP_FLAGS[@]}" nibabel SimpleITK

python -c "import torch; print(f'torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
echo "Setup complete on $(hostname)"
