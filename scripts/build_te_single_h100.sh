#!/usr/bin/env bash
# Build Transformer Engine's PyTorch extension into MOLT_VENV on a node whose
# system toolchain is too old for scripts/bootstrap.sh's TE build (gcc 9 lacks
# -std=c++20; NCCL headers only in the venv). Uses the venv's pip CUDA 13
# nvcc/cuDNN/NCCL and a conda gcc 12. Version defaults to env/pins.sh.
# Only adds packages (core wheel + torch extension + onnxscript deps); existing
# packages are not upgraded.
#
#   bash scripts/build_te_single_h100.sh [TE_VERSION]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/env.sh"
source "$ROOT/env/pins.sh"
TE_VERSION="${1:-${TE_TORCH_SPEC##*==}}"
SP="$MOLT_VENV/lib/python3.12/site-packages"
GCC="${GCC12_PREFIX:-$CACHE_ROOT/gcc12}"
[ -x "$GCC/bin/x86_64-conda-linux-gnu-g++" ] || conda create -y -p "$GCC" -c conda-forge --override-channels gxx_linux-64=12 gcc_linux-64=12
"$MOLT_VENV/bin/pip" install --no-deps "transformer_engine_cu13==$TE_VERSION" "transformer_engine==$TE_VERSION"
INC="$SP/nvidia/nccl/include:$SP/nvidia/cudnn/include:$SP/nvidia/cu13/include"
export CPATH="$INC" CPLUS_INCLUDE_PATH="$INC" LIBRARY_PATH="$SP/nvidia/nccl/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cu13/lib"
export CC="$GCC/bin/x86_64-conda-linux-gnu-gcc" CXX="$GCC/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX" NVCC_CCBIN="$CXX" CUDA_HOME="$SP/nvidia/cu13" CUDNN_PATH="$SP/nvidia/cudnn"
export PATH="$SP/nvidia/cu13/bin:$GCC/bin:$MOLT_VENV/bin:$PATH"
export NVTE_FRAMEWORK=pytorch MAX_JOBS="${MAX_JOBS:-16}" NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}"
"$MOLT_VENV/bin/pip" install --no-deps --no-build-isolation "transformer_engine_torch==$TE_VERSION"
"$MOLT_VENV/bin/pip" install onnxscript
CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13 "$MOLT_VENV/bin/python" -c \
  "import torch, transformer_engine.pytorch as te; print('TE ok', te.Linear(8, 8).cuda()(torch.randn(2, 8, device='cuda')).shape)"
