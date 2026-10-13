# Machine-specific paths for a hand-set-up node. scripts/bootstrap.sh writes
# config/env.sh itself (same variables); copy this file only to set paths by hand:
#
#     cp config/env.example.sh config/env.sh
#
# Defaults mirror bootstrap's layout under PREFIX=<checkout>/.deps. Every
# environment, checkout and model there is rebuilt from pins in env/ and
# config/models.json, so a fresh clone needs no other directory.

export OPSD_ROOT="${OPSD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GA_GRPO_ROOT="$OPSD_ROOT"
_PREFIX="$OPSD_ROOT/.deps"

# ---- models (bootstrap: Qwen3-4B-Instruct-2507; others: scripts/fetch_models.py NAME)
export MODELS_DIR="${MODELS_DIR:-$_PREFIX/models}"
export QWEN3_4B_INSTRUCT_2507_PATH="${QWEN3_4B_INSTRUCT_2507_PATH:-$MODELS_DIR/Qwen3-4B-Instruct-2507}"

# ---- interpreters (not co-installable: Molt/vLLM/TE, bfcl_eval, appworld)
export MOLT_VENV="${MOLT_VENV:-$_PREFIX/venvs/molt}"
export MOLT_DIR="${MOLT_DIR:-$GA_GRPO_ROOT/third_party/molt}"
export BFCL_T3RL_VENV="${BFCL_T3RL_VENV:-$_PREFIX/venvs/bfcl-t3rl}"   # BFCL_PROFILE=author_lineage (default)
export BFCL_VENV="${BFCL_VENV:-$_PREFIX/venvs/bfcl-v3}"          # BFCL_PROFILE=crest_compat
export BFCL_VLLM_VENV="${BFCL_VLLM_VENV:-$_PREFIX/venvs/vllm-eval}"   # official evaluation servers
export APPWORLD_VENV="${APPWORLD_VENV:-$_PREFIX/venvs/appworld}"

# ---- benchmark checkouts and data
export BFCL_T3RL_REPO_PATH="${BFCL_T3RL_REPO_PATH:-$_PREFIX/benchmarks/gorilla-bfcl-t3rl}"
export BFCL_REPO_PATH="${BFCL_REPO_PATH:-$_PREFIX/benchmarks/gorilla-bfcl-v3}"
export BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-$_PREFIX/bfcl_project}"
# Locally supplied verified training guides: <task_id>.json with task_id, verifier_pass, turns[].code.
export APPWORLD_GUIDES_DIR="${APPWORLD_GUIDES_DIR:-$_PREFIX/appworld_guides}"
export APPWORLD_DATA="${APPWORLD_DATA:-$_PREFIX/appworld/data}"

# ---- protocol profile (config/bfcl_protocol.sh derives the BFCL source/venv paths)
export BFCL_PROFILE="${BFCL_PROFILE:-author_lineage}"

# ---- node: hardware topology (config/topology/<name>.env; empty = built-in B200 defaults)
export TOPOLOGY="${TOPOLOGY-multi_gpu4}"

# ---- outputs and caches
export RUN_ROOT="${RUN_ROOT:-$_PREFIX/runs}"                     # ~45 GB optimizer state per kept checkpoint
export CACHE_ROOT="${CACHE_ROOT:-$_PREFIX/cache}"                  # compile caches; rebuildable
export APPWORLD_ROOT="${APPWORLD_ROOT:-$CACHE_ROOT/appworld_root}"    # writable; AppWorld servers keep worlds here
export APPWORLD_SETUPS="${APPWORLD_SETUPS:-$RUN_ROOT/appworld_setups}"  # sealed setups built by run.py
export BFCL_PROTOBUF_OVERLAY="${BFCL_PROTOBUF_OVERLAY:-$CACHE_ROOT/bfcl_protobuf_overlay}"
# Optional: a GPU idle keeper with keeper_client.py (run.py reserves and waits for free GPUs).
# export GPU_KEEPER_DIR=/path/to/gpu_idle_keeper
