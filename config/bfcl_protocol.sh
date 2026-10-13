#!/bin/bash
# BFCL protocol pin. Runtime, data, simulator, evaluator and prompt-history
# handling are pinned as ONE unit through a named profile, because mixing them
# produces plausible-looking but incomparable numbers.
#
#   BFCL_PROFILE=author_lineage   (default, GA-GRPO)
#       T3RL author-lineage BFCL checkout 6ea5797 (dataset prefix BFCL_v4), its
#       own venv, Qwen FC handler that RETAINS prior assistant tool calls in the
#       prompt, registry Qwen/Qwen3-4B-Instruct-2507-FC. Training rollouts and
#       the official evaluation use the same checkout, so the model is scored in
#       the format it was trained in. Comparability label: author-lineage-candidate.
#       Reference: untrained Qwen3-4B-Instruct-2507 = 23.00% (100 eval tasks, 3 decodes).
#
#   BFCL_PROFILE=crest_compat
#       Strict BFCL V3 checkout ea13468 (dataset prefix BFCL_v3) with the
#       compat/bfcl_v3_qwen2507 sitecustomize that serializes prior tool calls
#       the v1.3 handler dropped. This is the CrEST-paper reproduction contract
#       used for every result dated 2026-08-28 .. 2026-09-04. Comparability
#       label: crest-bfcl-v3-qwen2507-compat.
#
# Numbers from the two profiles must never appear in the same table.
# Any variable below can still be overridden explicitly for a labelled ablation.

BFCL_PROTOCOL_REPO_ROOT="${ROOT:-${OPSD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
BFCL_PROFILE="${BFCL_PROFILE:-author_lineage}"

case "$BFCL_PROFILE" in
  author_lineage)
    BFCL_HISTORY_MODE="${BFCL_HISTORY_MODE:-author_lineage_qwen_handler}"
    BFCL_RUNTIME_SOURCE_ROOT="${BFCL_RUNTIME_SOURCE_ROOT:-$BFCL_T3RL_REPO_PATH/berkeley-function-call-leaderboard}"
    BFCL_RUNTIME_COMMIT="${BFCL_RUNTIME_COMMIT:-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8}"
    BFCL_RUNTIME_VENV="${BFCL_RUNTIME_VENV:-$BFCL_T3RL_VENV}"
    BFCL_MODEL_REGISTRY_NAME="${BFCL_MODEL_REGISTRY_NAME:-Qwen/Qwen3-4B-Instruct-2507-FC}"
    BFCL_EXPECTED_VERSION="${BFCL_EXPECTED_VERSION:-BFCL_v4}"
    BFCL_COMPARABILITY="${BFCL_COMPARABILITY:-author-lineage-candidate}"
    ;;
  crest_compat)
    BFCL_HISTORY_MODE="${BFCL_HISTORY_MODE:-qwen2507_v3_compat}"
    BFCL_RUNTIME_SOURCE_ROOT="${BFCL_RUNTIME_SOURCE_ROOT:-$BFCL_REPO_PATH/berkeley-function-call-leaderboard}"
    BFCL_RUNTIME_COMMIT="${BFCL_RUNTIME_COMMIT:-ea13468e4423454d0c213704fb87cf7cb3990433}"
    BFCL_RUNTIME_VENV="${BFCL_RUNTIME_VENV:-$BFCL_VENV}"
    # The v1.3 registry predates the 2507 checkpoint name; this entry selects the Qwen FC handler.
    BFCL_MODEL_REGISTRY_NAME="${BFCL_MODEL_REGISTRY_NAME:-Qwen/Qwen3-4B-FC}"
    BFCL_EXPECTED_VERSION="${BFCL_EXPECTED_VERSION:-BFCL_v3}"
    BFCL_COMPARABILITY="${BFCL_COMPARABILITY:-crest-bfcl-v3-qwen2507-compat}"
    ;;
  *)
    echo "invalid BFCL_PROFILE: $BFCL_PROFILE (author_lineage | crest_compat)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

BFCL_TOOL_MODE="${BFCL_TOOL_MODE:-native_fc}"
# 0 means omit top_k, matching the official BFCL generation request.
BFCL_GENERATION_TOP_K="${BFCL_GENERATION_TOP_K:-0}"
BFCL_DATA_ROOT="${BFCL_DATA_ROOT:-$BFCL_RUNTIME_SOURCE_ROOT/bfcl_eval/data}"
BFCL_PYTHON_COMPAT_ROOT="${BFCL_PYTHON_COMPAT_ROOT:-$BFCL_PROTOCOL_REPO_ROOT/compat/bfcl_v3_qwen2507}"
# Where ga_grpo/bfcl_agent.py (Molt rollouts) imports bfcl_eval from.
BFCL_SOURCE_ROOT="${BFCL_SOURCE_ROOT:-$BFCL_RUNTIME_SOURCE_ROOT}"
BFCL_VENV_SITE_PACKAGES="${BFCL_VENV_SITE_PACKAGES:-$BFCL_RUNTIME_VENV/lib/python3.12/site-packages}"

[[ "$BFCL_TOOL_MODE" == "native_fc" || "$BFCL_TOOL_MODE" == "prompting" ]] || {
    echo "BFCL_TOOL_MODE must be native_fc or prompting" >&2; return 1 2>/dev/null || exit 1
}
[[ "$BFCL_HISTORY_MODE" == "qwen2507_v3_compat" || \
   "$BFCL_HISTORY_MODE" == "author_lineage_qwen_handler" || \
   "$BFCL_HISTORY_MODE" == "v1_3_qwen_handler" || \
   "$BFCL_HISTORY_MODE" == "official_qwen_handler" || \
   "$BFCL_HISTORY_MODE" == "complete" ]] || {
    echo "invalid BFCL_HISTORY_MODE: $BFCL_HISTORY_MODE" >&2; return 1 2>/dev/null || exit 1
}
[[ "$BFCL_GENERATION_TOP_K" =~ ^[0-9]+$ ]] || {
    echo "BFCL_GENERATION_TOP_K must be a non-negative integer (0 disables it)" >&2
    return 1 2>/dev/null || exit 1
}
[[ -f "$BFCL_RUNTIME_SOURCE_ROOT/bfcl_eval/model_handler/local_inference/qwen_fc.py" ]] || {
    echo "missing pinned BFCL source for profile $BFCL_PROFILE: $BFCL_RUNTIME_SOURCE_ROOT" >&2
    return 1 2>/dev/null || exit 1
}
[[ -d "$BFCL_DATA_ROOT/possible_answer" ]] || {
    echo "missing BFCL data root: $BFCL_DATA_ROOT" >&2
    return 1 2>/dev/null || exit 1
}
[[ -x "$BFCL_RUNTIME_VENV/bin/bfcl" ]] || {
    echo "missing pinned BFCL CLI environment for profile $BFCL_PROFILE: $BFCL_RUNTIME_VENV" >&2
    return 1 2>/dev/null || exit 1
}
if [[ "$BFCL_HISTORY_MODE" == "qwen2507_v3_compat" ]]; then
    [[ -f "$BFCL_PYTHON_COMPAT_ROOT/sitecustomize.py" ]] || {
        echo "missing BFCL V3 Qwen2507 compatibility module: $BFCL_PYTHON_COMPAT_ROOT" >&2
        return 1 2>/dev/null || exit 1
    }
    BFCL_COMPAT_PYTHONPATH="$BFCL_PYTHON_COMPAT_ROOT:"
else
    BFCL_COMPAT_PYTHONPATH=""
fi
[[ "$BFCL_MODEL_REGISTRY_NAME" == *-FC ]] || {
    echo "registry must select an FC handler: $BFCL_MODEL_REGISTRY_NAME" >&2
    return 1 2>/dev/null || exit 1
}

BFCL_ACTUAL_VERSION="$({ PYTHONPATH="$BFCL_RUNTIME_SOURCE_ROOT" \
    "$BFCL_RUNTIME_VENV/bin/python" -c \
    'from bfcl_eval.constants.category_mapping import VERSION_PREFIX; print(VERSION_PREFIX)' 2>/dev/null; } || true)"
[[ "$BFCL_ACTUAL_VERSION" == "$BFCL_EXPECTED_VERSION" ]] || {
    echo "profile $BFCL_PROFILE requires $BFCL_EXPECTED_VERSION, got '${BFCL_ACTUAL_VERSION:-unreadable}' from $BFCL_RUNTIME_SOURCE_ROOT" >&2
    return 1 2>/dev/null || exit 1
}
[[ -f "$BFCL_DATA_ROOT/${BFCL_ACTUAL_VERSION}_multi_turn_base.json" ]] || {
    echo "missing ${BFCL_ACTUAL_VERSION} multi_turn_base data under $BFCL_DATA_ROOT" >&2
    return 1 2>/dev/null || exit 1
}
if [[ -n "${BFCL_RUNTIME_COMMIT:-}" && -d "$BFCL_RUNTIME_SOURCE_ROOT/../.git" ]]; then
    _actual_commit="$(git -C "$BFCL_RUNTIME_SOURCE_ROOT/.." rev-parse HEAD 2>/dev/null || true)"
    [[ -z "$_actual_commit" || "$_actual_commit" == "$BFCL_RUNTIME_COMMIT" ]] || {
        echo "BFCL checkout at $BFCL_RUNTIME_SOURCE_ROOT is $_actual_commit, profile $BFCL_PROFILE pins $BFCL_RUNTIME_COMMIT" >&2
        return 1 2>/dev/null || exit 1
    }
    unset _actual_commit
fi

export BFCL_PROFILE BFCL_COMPARABILITY BFCL_EXPECTED_VERSION
export BFCL_TOOL_MODE BFCL_HISTORY_MODE BFCL_GENERATION_TOP_K
export BFCL_RUNTIME_SOURCE_ROOT BFCL_DATA_ROOT BFCL_RUNTIME_COMMIT
export BFCL_MODEL_REGISTRY_NAME BFCL_RUNTIME_VENV
export BFCL_SOURCE_ROOT BFCL_VENV_SITE_PACKAGES
export BFCL_PYTHON_COMPAT_ROOT BFCL_COMPAT_PYTHONPATH
export BFCL_PROTOCOL_REPO_ROOT
export BFCL_ACTUAL_VERSION
export PYTHONPATH="${BFCL_COMPAT_PYTHONPATH}$BFCL_PROTOCOL_REPO_ROOT:$BFCL_RUNTIME_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
