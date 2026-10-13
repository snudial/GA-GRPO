#!/usr/bin/env bash
# Pinned BFCL official evaluation of an arbitrary HF checkpoint directory.
#   source config/env.sh
#   bash scripts/eval_official.sh HF_DIR OUT_RUN_DIR [SHARDS_PER_GPU]
# Builds the Qwen3-2507 compat tokenizer_config next to HF_DIR (Molt exports
# `extra_special_tokens` as a list, the pinned transformers wants a dict), then
# runs the frozen three-decode protocol via scripts/official_pool.sh.
# Existing results are reused only when explicitly requested.
set -euo pipefail

HF="$1"
OUT="$2"
SHARDS="${3:-25}"
ROOT="${GA_GRPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$ROOT/config/env.sh" ] && source "$ROOT/config/env.sh"
source "$ROOT/config/topology.sh"   # hardware profile (TOPOLOGY); see config/topology/
source "$ROOT/config/bfcl_protocol.sh"
COMPAT="${HF%/}_eval_compat_${BFCL_PROFILE}"
TASK_MANIFEST="${TASK_MANIFEST:-$ROOT/config/splits/bfcl_aworld_lineage_seed42_v1/base_test75.json}"
TASK_COUNT=$(python3 -c "import json; print(len(json.load(open('$TASK_MANIFEST'))))")
GPU_COUNT="${EVAL_GPU_COUNT:-4}"
# Every GPU-local shard must receive at least one task. Empty shards make BFCL
# generation omit its result file and fail the whole decode.
MAX_SHARDS_PER_GPU=$((TASK_COUNT / GPU_COUNT))
if [ "$MAX_SHARDS_PER_GPU" -lt 1 ]; then MAX_SHARDS_PER_GPU=1; fi
if [ "$SHARDS" -gt "$MAX_SHARDS_PER_GPU" ]; then SHARDS="$MAX_SHARDS_PER_GPU"; fi

if [ -s "$OUT/summary.json" ]; then
  if [ "${REUSE_EVAL:-0}" = "1" ]; then
    echo "reuse $OUT/summary.json"
    exit 0
  fi
  echo "$OUT/summary.json already exists; use a new output directory or set REUSE_EVAL=1" >&2
  exit 2
fi

mkdir -p "$COMPAT"
for name in config.json generation_config.json chat_template.jinja tokenizer.json \
    model.safetensors.index.json; do
  ln -sfn "$HF/$name" "$COMPAT/$name"
done
for f in "$HF"/model-*.safetensors; do
  ln -sfn "$f" "$COMPAT/$(basename "$f")"
done
jq '.extra_special_tokens = {} |
    .additional_special_tokens = ["<|im_start|>","<|im_end|>","<|object_ref_start|>","<|object_ref_end|>","<|box_start|>","<|box_end|>","<|quad_start|>","<|quad_end|>","<|vision_start|>","<|vision_end|>","<|vision_pad|>","<|image_pad|>","<|video_pad|"]' \
   "$HF/tokenizer_config.json" > "$COMPAT/tokenizer_config.json"

# Protocol profile selects checkout, venv, handler history mode and the
# comparability label; official_pool.sh validates the same profile again.
OVERLAY="${BFCL_PROTOBUF_OVERLAY:-$ROOT/compat/bfcl_protobuf_overlay}"
if [ "${BFCL_QWEN3_NONTHINKING:-0}" = 1 ]; then
  export PYTHONPATH="$ROOT/compat/bfcl_qwen3_nonthinking${PYTHONPATH:+:$PYTHONPATH}"
fi
PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}" \
  OPSD_ROOT="$ROOT" BFCL_PROFILE="${BFCL_PROFILE:-author_lineage}" \
  MODEL_PATH="$COMPAT" RUN_DIR="$OUT" \
  TASK_MANIFEST="$TASK_MANIFEST" \
  CATEGORY="${CATEGORY:-multi_turn_base}" DECODES="${DECODES:-8}" SHARDS_PER_GPU="$SHARDS" \
  TEMPERATURE=0.000001 \
  SEED_BASE=20260826 GPU_MEMORY_UTILIZATION=0.90 BFCL_MAX_OUTPUT_TOKENS=4096 \
  bash "$ROOT/scripts/official_pool.sh"
test -s "$OUT/summary.json"
python3 -c "
import json
s=json.load(open('$OUT/summary.json'))
print('OFFICIAL', s.get('comparability'), '$OUT', 'mean', round(s['mean_strict_official_session_success'],4),
      [d['strict_official_session_success'] for d in s['decodes']])"
