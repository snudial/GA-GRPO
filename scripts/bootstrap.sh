#!/usr/bin/env bash
# One-command setup of a fresh node after `git clone`:
# the three Python environments, the pinned BFCL checkout, the model,
# config/env.sh and data/bfcl. Every package version and hash comes from
# env/ (read off the 4x B200 node that ran the JUMP s42 baseline).
#
#   bash scripts/bootstrap.sh --prefix /data/ga_grpo [--run-root DIR] [--with-crest]
#
# Idempotent: each stage leaves a marker in $PREFIX/.bootstrap and is skipped
# on a rerun, so a failed run continues where it stopped. Stages, in order:
#   check      GPUs, NVIDIA driver >= 580 (CUDA 13), disk, tools        (always runs)
#   uv         uv 0.9.18 and a managed CPython 3.12.3 under $PREFIX
#   molt       bundled third_party/molt runtime verification
#   gorilla    BFCL checkout 6ea5797 + its max-output-tokens patch (ea13468 with --with-crest)
#   venv-bfcl  BFCL CLI venv(s), bfcl_eval editable from the checkout
#   venv-molt  trainer venv: torch cu130, env/molt.txt, TE torch build, Molt editable, qwen3_5 guards
#   venv-eval  vLLM 0.28 venv for the official evaluation servers
#   appworld   AppWorld venv (env/appworld.txt) and its task data
#   model      Qwen3-4B-Instruct-2507 at the pinned revision, shard sha256 checked
#   envsh      config/env.sh for this node (an env.sh not written by this script is kept)
#   setup      scripts/setup.sh: patch check, BFCL protocol check, data/bfcl, protobuf overlay
#   verify     GPU smoke of torch/TE/vLLM/Molt imports, protocol, train.sh dry run (always runs)
#
# Options:
#   --prefix DIR        environments, checkouts, model, caches  (default: <repo>/.deps)
#   --run-root DIR      training outputs, ~50 GB per kept optimizer state (default: PREFIX/runs)
#   --topology NAME     config/topology/NAME.env (default: detected; multi_h100 on 4x H100 80-96 GB,
#                       single_h100 on 1x H100, multi_gpu4 on 4x B200)
#   --with-crest        also build the crest_compat BFCL environment (checkout ea13468)
#   --only STAGE        run one stage;  --redo STAGE  rerun a finished stage
#   --force-env         rewrite config/env.sh even if it was written by hand (old one is backed up)
#   --allow-old-driver  continue with a driver < 580 (needs CUDA forward compatibility, see check)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/pins.sh"

PREFIX="$ROOT/.deps"; RUN_ROOT=""; TOPOLOGY_ARG=""; WITH_CREST=0
ONLY=""; REDO=""; FORCE_ENV=0; ALLOW_OLD_DRIVER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --topology) TOPOLOGY_ARG="$2"; shift 2 ;;
    --with-crest) WITH_CREST=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --redo) REDO="$2"; shift 2 ;;
    --force-env) FORCE_ENV=1; shift ;;
    --allow-old-driver) ALLOW_OLD_DRIVER=1; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"
RUN_ROOT="${RUN_ROOT:-$PREFIX/runs}"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
CACHE_ROOT="$PREFIX/cache"
MARK="$PREFIX/.bootstrap"
mkdir -p "$MARK" "$CACHE_ROOT"
exec > >(tee -a "$PREFIX/bootstrap.log") 2>&1

export UV_CACHE_DIR="$PREFIX/uv-cache"            # same filesystem as the venvs: hardlinked, stored once
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
export UV_PYTHON_PREFERENCE=only-managed
UV="$PREFIX/bin/uv"
MOLT_VENV="$PREFIX/venvs/molt"
BFCL_T3RL_VENV="$PREFIX/venvs/bfcl-t3rl"
BFCL_V3_VENV="$PREFIX/venvs/bfcl-v3"
EVAL_VENV="$PREFIX/venvs/vllm-eval"
APPWORLD_VENV="$PREFIX/venvs/appworld"
APPWORLD_HOME="$PREFIX/appworld"          # AppWorld task data lives in $APPWORLD_HOME/data
T3RL_CHECKOUT="$PREFIX/benchmarks/gorilla-bfcl-t3rl"
V3_CHECKOUT="$PREFIX/benchmarks/gorilla-bfcl-v3"
MODEL_DIR="$PREFIX/models/$MODEL_DIRNAME"

die() { echo "[bootstrap] ERROR: $*" >&2; exit 1; }
warn() { echo "[bootstrap] WARNING: $*" >&2; }
free_gb() { df -Pk "$1" | awk 'NR==2 {print int($4 / 1048576)}'; }

detect_topology() {   # config/topology/<name>.env for this node's GPUs, or "" if none fits
  # H100 memory: 80 GB parts, and 94 GB parts that report 95,830 MiB.
  local names count mem
  names=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
  count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
  [ "$(printf '%s\n' "$names" | wc -l)" = 1 ] || { echo ""; return; }
  case "$names:$count" in
    # 80 GB and 94 GB H100s share a profile; single_h100 reads the size off nvidia-smi.
    *H100*:4) if [ "$mem" -ge 79000 ] && [ "$mem" -lt 100000 ]; then echo multi_h100; return; fi ;;
    *H100*:1) if [ "$mem" -ge 79000 ] && [ "$mem" -lt 100000 ]; then echo single_h100; return; fi ;;
    *B200*:4) echo multi_gpu4; return ;;
  esac
  echo ""
}

# ----------------------------------------------------------------------------- stages
stage_check() {
  local t
  for t in git curl df awk sha256sum findmnt nvidia-smi python3 patch; do
    command -v "$t" >/dev/null || die "missing tool: $t"
  done
  echo "GPUs (name, memory MiB, driver):"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
  local driver major count
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  major=${driver%%.*}
  if [ "$major" -lt "$MIN_DRIVER_MAJOR" ]; then
    local msg="NVIDIA driver $driver < $MIN_DRIVER_MAJOR: torch cu130 and vLLM 0.27/0.28 need a CUDA 13 driver. Upgrade the driver, or (data-center GPUs) install the cuda-compat-13-0 package, prepend its compat/ directory to LD_LIBRARY_PATH and rerun with --allow-old-driver."
    if [ "$ALLOW_OLD_DRIVER" = 1 ]; then warn "$msg"; else die "$msg"; fi
  fi
  count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  DETECTED_TOPOLOGY="${TOPOLOGY_ARG:-$(detect_topology)}"
  if [ -n "$DETECTED_TOPOLOGY" ]; then
    [ -f "$ROOT/config/topology/$DETECTED_TOPOLOGY.env" ] || die "no config/topology/$DETECTED_TOPOLOGY.env"
    echo "topology: $DETECTED_TOPOLOGY (config/topology/$DETECTED_TOPOLOGY.env)"
  else
    warn "no config/topology profile matches these $count GPUs; TOPOLOGY stays empty, i.e. the scripts' built-in 4x B200 defaults (VLLM_UTIL 0.50 of 180 GB). Add config/topology/<name>.env and pass --topology."
  fi
  local p_free r_free
  p_free=$(free_gb "$PREFIX"); r_free=$(free_gb "$RUN_ROOT")
  echo "free disk: prefix $PREFIX ${p_free} GB, run root $RUN_ROOT ${r_free} GB"
  [ "$p_free" -ge 60 ] || [ -e "$MARK/venv-eval" ] || die "prefix needs ~60 GB (3 venvs ~25 GB, model 8 GB, caches)"
  [ "$r_free" -ge 150 ] || warn "run root has ${r_free} GB; one 80k-rollout run keeps ~2 optimizer states (~45 GB each) plus HF snapshots (8 GB each)"
  if [ "$(findmnt -T "$RUN_ROOT" -n -o FSTYPE)" != lustre ]; then ALLOW_NON_LUSTRE_LINE='export ALLOW_NON_LUSTRE=1'; else ALLOW_NON_LUSTRE_LINE=''; fi
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$busy" = 0 ] || warn "$busy GPU processes are running now (verify/smoke share the GPUs with them)"
}

stage_uv() {
  if ! "$UV" --version 2>/dev/null | grep -q " $UV_VERSION"; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
      | env UV_INSTALL_DIR="$PREFIX/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
  fi
  "$UV" --version
  "$UV" python install "$PYTHON_VERSION"
}

stage_molt() {
  python3 "$ROOT/scripts/verify_vendor.py"
}

checkout_commit() {   # DIR COMMIT: shallow fetch of one commit (GitHub serves reachable SHAs)
  local dir="$1" commit="$2"
  if [ -d "$dir/.git" ]; then
    [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ] || die "$dir exists but is not at $commit"
    return 0
  fi
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$GORILLA_URL"
  git -C "$dir" fetch -q --depth 1 origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
}

stage_gorilla() {
  checkout_commit "$T3RL_CHECKOUT" "$GORILLA_T3RL_COMMIT"
  # Adds the BFCL_MAX_OUTPUT_TOKENS override; its default (4096) is upstream's value.
  local patch="$ROOT/env/gorilla-6ea5797-bfcl-max-output-tokens.patch"
  git -C "$T3RL_CHECKOUT" apply --reverse --check "$patch" 2>/dev/null || git -C "$T3RL_CHECKOUT" apply "$patch"
  if [ "$WITH_CREST" = 1 ]; then checkout_commit "$V3_CHECKOUT" "$GORILLA_V3_COMMIT"; fi
}

new_venv() {   # DIR: fresh venv (a half-built one from a failed run is replaced)
  [ ! -f "$1/pyvenv.cfg" ] || rm -rf "$1"
  "$UV" venv -q --python "$PYTHON_VERSION" "$1"
}

stage_venv_bfcl() {
  new_venv "$BFCL_T3RL_VENV"
  "$UV" pip install --python "$BFCL_T3RL_VENV/bin/python" --no-deps -r "$ROOT/env/bfcl-t3rl.txt"
  "$UV" pip install --python "$BFCL_T3RL_VENV/bin/python" --no-deps -e "$T3RL_CHECKOUT/berkeley-function-call-leaderboard"
  [ -x "$BFCL_T3RL_VENV/bin/bfcl" ] || die "bfcl CLI missing in $BFCL_T3RL_VENV"
  if [ "$WITH_CREST" = 1 ]; then
    new_venv "$BFCL_V3_VENV"
    "$UV" pip install --python "$BFCL_V3_VENV/bin/python" --no-deps -r "$ROOT/env/bfcl-v3.txt"
    "$UV" pip install --python "$BFCL_V3_VENV/bin/python" --no-deps -e "$V3_CHECKOUT/berkeley-function-call-leaderboard"
  fi
}

stage_venv_molt() {
  local py="$MOLT_VENV/bin/python" site cuda cpath
  new_venv "$MOLT_VENV"
  "$UV" pip install --python "$py" --no-deps --index-url "$TORCH_INDEX" "$TORCH_SPEC"
  "$UV" pip install --python "$py" --no-deps -r "$ROOT/env/molt.txt"
  # Transformer Engine's torch extension is built here against this torch. A
  # system CUDA >= 13 toolkit is used if present, else the pip-shipped one.
  site=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  cuda="${TE_CUDA_HOME:-}"          # TE_CUDA_HOME=pip forces the pip toolkit
  [ "$cuda" != pip ] || cuda="$site/nvidia/cu13"
  if [ -z "$cuda" ] && [ -x /usr/local/cuda/bin/nvcc ] \
      && /usr/local/cuda/bin/nvcc --version | grep -q -E 'release 13\.'; then
    cuda=/usr/local/cuda
  fi
  cuda="${cuda:-$site/nvidia/cu13}"
  [ -x "$cuda/bin/nvcc" ] || die "no nvcc at $cuda/bin/nvcc (env/molt.txt installs nvidia-cuda-nvcc)"
  cpath=$(ls -d "$site"/nvidia/*/include 2>/dev/null | paste -sd: -)
  echo "building $TE_TORCH_SPEC with CUDA_HOME=$cuda"
  CUDA_HOME="$cuda" CUDA_PATH="$cuda" PATH="$cuda/bin:$PATH" CPATH="$cpath${CPATH:+:$CPATH}" \
    NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90;100}" MAX_JOBS="${MAX_JOBS:-16}" \
    "$UV" pip install --python "$py" --no-deps --no-build-isolation "$TE_TORCH_SPEC"
  # Ray workers import molt from the venv: point it at the patched submodule.
  "$UV" pip install --python "$py" --no-deps -e "$ROOT/third_party/molt"
}

stage_venv_eval() {
  new_venv "$EVAL_VENV"
  "$UV" pip install --python "$EVAL_VENV/bin/python" --no-deps -r "$ROOT/env/vllm-eval.txt"
  [ -x "$EVAL_VENV/bin/vllm" ] || die "vllm CLI missing in $EVAL_VENV"
}

stage_appworld() {   # AppWorld (pydantic 1: its own venv) and its task data
  new_venv "$APPWORLD_VENV"
  "$UV" pip install --python "$APPWORLD_VENV/bin/python" --no-deps -r "$ROOT/env/appworld.txt"
  "$APPWORLD_VENV/bin/appworld" install
  [ -d "$APPWORLD_HOME/data" ] || "$APPWORLD_VENV/bin/appworld" download data --root "$APPWORLD_HOME"
}

stage_model() {
  [ -x "$MOLT_VENV/bin/python" ] || die "run the venv-molt stage first (it provides huggingface_hub)"
  HF_HOME="$CACHE_ROOT/huggingface" "$MOLT_VENV/bin/python" - "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, local_dir = sys.argv[1:4]
snapshot_download(repo_id=repo, revision=revision, local_dir=local_dir)
PY
  (cd "$MODEL_DIR" && printf '%s\n' "$MODEL_SHA256" | sha256sum -c -)
}

stage_envsh() {
  local f="$ROOT/config/env.sh" topology="${DETECTED_TOPOLOGY:-}"
  if [ -f "$f" ] && ! grep -q '^# Written by scripts/bootstrap.sh' "$f"; then
    if [ "$FORCE_ENV" = 1 ]; then
      cp "$f" "$f.bak.$(date +%Y%m%d%H%M%S)"
    else
      f="$ROOT/config/env.sh.bootstrap"
      warn "config/env.sh was written by hand; kept it and wrote $f (use --force-env to replace)"
    fi
  fi
  cat > "$f" <<EOF
# Written by scripts/bootstrap.sh on $(date -Is) for $(hostname).
# Machine-specific and gitignored. Regenerate: bash scripts/bootstrap.sh --prefix $PREFIX --redo envsh
export OPSD_ROOT="\${OPSD_ROOT:-\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)}"
export GA_GRPO_ROOT="\$OPSD_ROOT"

# ---- model
export MODELS_DIR="$PREFIX/models"
export QWEN3_4B_INSTRUCT_2507_PATH="$MODEL_DIR"

# ---- interpreters
export MOLT_VENV="$MOLT_VENV"
export MOLT_DIR="\$GA_GRPO_ROOT/third_party/molt"
export BFCL_T3RL_VENV="$BFCL_T3RL_VENV"       # BFCL_PROFILE=author_lineage (default)
export BFCL_VENV="$BFCL_V3_VENV"              # BFCL_PROFILE=crest_compat (bootstrap --with-crest)
export BFCL_VLLM_VENV="$EVAL_VENV"            # official evaluation servers

# ---- benchmark checkouts
export BFCL_T3RL_REPO_PATH="$T3RL_CHECKOUT"
export BFCL_REPO_PATH="$V3_CHECKOUT"
export BFCL_PROJECT_ROOT="$PREFIX/bfcl_project"
export APPWORLD_VENV="$APPWORLD_VENV"
export APPWORLD_DATA="$APPWORLD_HOME/data"
export APPWORLD_GUIDES_DIR="\${APPWORLD_GUIDES_DIR:-$PREFIX/appworld_guides}"
export APPWORLD_ROOT="$CACHE_ROOT/appworld_root"       # writable; AppWorld servers keep worlds here
export APPWORLD_SETUPS="$RUN_ROOT/appworld_setups"     # sealed setups built by run.py

# ---- protocol profile
export BFCL_PROFILE="\${BFCL_PROFILE:-author_lineage}"

# ---- node: hardware topology (config/topology/) and storage
export TOPOLOGY="\${TOPOLOGY-$topology}"   # an empty TOPOLOGY= means the scripts' built-in defaults
export RUN_ROOT="$RUN_ROOT"
export CACHE_ROOT="$CACHE_ROOT"
export BFCL_PROTOBUF_OVERLAY="$CACHE_ROOT/bfcl_protobuf_overlay"
${ALLOW_NON_LUSTRE_LINE:-}
EOF
  echo "wrote $f (TOPOLOGY=${topology:-none})"
}

stage_setup() {
  # setup.sh runs `python3` for data prep; give it the trainer venv's interpreter.
  PATH="$MOLT_VENV/bin:$PATH" bash "$ROOT/scripts/setup.sh"
}

stage_verify() {
  # shellcheck disable=SC1091
  (
    source "$ROOT/config/env.sh"
    "$MOLT_VENV/bin/python" - "$ROOT" <<'PY'
import sys
from pathlib import Path
import torch
assert torch.cuda.is_available(), "CUDA is not available in the Molt venv"
n = torch.cuda.device_count()
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), {n} GPUs: "
      f"{sorted({torch.cuda.get_device_name(i) for i in range(n)})}, capability {torch.cuda.get_device_capability(0)}")
import transformer_engine.pytorch as te
layer = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
layer(torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)).sum().item()
print("transformer_engine.pytorch: te.Linear forward ok")
import vllm
print("vllm", vllm.__version__)
import molt
root = Path(sys.argv[1]).resolve()
assert Path(molt.__file__).resolve().is_relative_to(root / "third_party" / "molt"), molt.__file__
print("molt imports from", Path(molt.__file__).parent)
PY
    "$BFCL_VLLM_VENV/bin/python" -c 'import vllm; print("evaluation vllm", vllm.__version__)'
    "$APPWORLD_VENV/bin/python" -c 'import appworld; print("appworld", appworld.__version__)'
    source "$ROOT/config/bfcl_protocol.sh"
    echo "BFCL profile $BFCL_PROFILE: $BFCL_ACTUAL_VERSION @ ${BFCL_RUNTIME_COMMIT:0:8} ($BFCL_COMPARABILITY)"
    DRY_RUN=1 BFCL_CURRICULUM=1 bash "$ROOT/scripts/train.sh" bootstrap_dry_run | sed -n '1,2p'
  )
  cat <<EOF

[bootstrap] done. Next:
  python3 run.py --env bfcl --model Qwen3-4B-Instruct-2507 --method call-bandit --seed 42 --dry-run
EOF
}

# ------------------------------------------------------------------------------- main
DETECTED_TOPOLOGY=""; ALLOW_NON_LUSTRE_LINE=""
for stage in check uv molt gorilla venv-bfcl venv-molt venv-eval appworld model envsh setup verify; do
  [ -z "$ONLY" ] || [ "$ONLY" = "$stage" ] || [ "$stage" = check ] || continue
  if [ "$stage" != check ] && [ "$stage" != verify ] && [ -e "$MARK/$stage" ] && [ "$REDO" != "$stage" ]; then
    echo "== $stage: done $(cat "$MARK/$stage"), skipped"
    continue
  fi
  echo "== $stage ($(date -Is))"
  "stage_${stage//-/_}"
  date -Is > "$MARK/$stage"
done
