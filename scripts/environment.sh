#!/usr/bin/env bash
# Source from any project launcher; all writable runtime caches stay in the project.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MUSIC_COVER_BASE_PREFIX="${PROJECT_ROOT}/envs/base"
export PATH="${MUSIC_COVER_BASE_PREFIX}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export TORCH_EXTENSIONS_DIR="${PROJECT_ROOT}/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${PROJECT_ROOT}/cache/triton"
export CUDA_CACHE_PATH="${PROJECT_ROOT}/cache/cuda"
export NUMBA_CACHE_DIR="${PROJECT_ROOT}/cache/numba"
export MPLCONFIGDIR="${PROJECT_ROOT}/cache/matplotlib"
export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export MAMBA_ROOT_PREFIX="${PROJECT_ROOT}/cache/mamba"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/cache/conda/pkgs"
export TMPDIR="${PROJECT_ROOT}/cache/tmp"
export WANDB_DIR="${PROJECT_ROOT}/cache/wandb"
export WANDB_CACHE_DIR="${PROJECT_ROOT}/cache/wandb"
export WANDB_DATA_DIR="${PROJECT_ROOT}/cache/wandb/data"
export WANDB_CONFIG_DIR="${PROJECT_ROOT}/cache/wandb/config"
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$WANDB_CONFIG_DIR"
