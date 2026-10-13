#!/usr/bin/env bash
# Fixed generated-rollout contract with val25 checkpoint selection and held-out
# test75 reporting. All generated samples count, including filtered groups and
# DAPO refill samples. Episode 30/50 boundaries retain exact 48k/80k recovery
# points; validation cadence is optimizer-step based.
set -euo pipefail

MODE="$1"
ALGO="$2"
SEED="$3"
ROOT="${GA_GRPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$ROOT/config/env.sh" ] && source "$ROOT/config/env.sh"
source "$ROOT/config/topology.sh"   # hardware profile (TOPOLOGY); see config/topology/
: "${MOLT_VENV:?set MOLT_VENV in config/env.sh}" "${MODELS_DIR:?set MODELS_DIR in config/env.sh}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs}"
TAG="$(date +%Y%m%d)"
TARGET_EPISODES="${TARGET_EPISODES:-30 50}"

case "$ALGO" in grpo|dapo) ;; *) echo "algo must be grpo or dapo" >&2; exit 2 ;; esac
case "$MODE" in
  jump) RUN="${4:-bfcl_ga_grpo_base100_4b_seed${SEED}_r80k_val25_test75_${TAG}}"
        DATA="$ROOT/data/bfcl/bfcl_jumpstart_train.jsonl"; CURR=1; METHOD=turn-threshold-end ;;
  grpo) RUN="${4:-bfcl_v3_${ALGO}_base100_4b_seed${SEED}_r80k_val25_test75_${TAG}}"
        DATA="$ROOT/data/bfcl/bfcl_train.jsonl"; CURR=0; METHOD="$ALGO" ;;
  *) echo "mode must be jump or grpo" >&2; exit 2 ;;
esac

DATA="${TRAIN_DATA:-$DATA}"
METHOD="${COMPARISON_METHOD:-$METHOD}"
export ALGO
BFCL_PROFILE="${BFCL_PROFILE:-author_lineage}"
LOG="$RUN_ROOT/${RUN}_queue.log"
RUN_DIR="$RUN_ROOT/$RUN"
DATASET_GROUPS=$(wc -l < "$DATA")
N_SAMPLES="${N_SAMPLES:-16}"
VAL_DATA="${EVAL_DATA:-$ROOT/data/bfcl/bfcl_val.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-$ROOT/config/splits/bfcl_aworld_lineage_seed42_v1/base_test75.json}"

# The one train.sh invocation of this protocol: train TARGET_EPISODES RESUME.
train() {
  MODEL="${MODEL:-$MODELS_DIR/Qwen3-4B-Instruct-2507}" \
    TRAIN_DATA="$DATA" EVAL_DATA="$VAL_DATA" BFCL_CURRICULUM="$CURR" TRAIN_SEED="$SEED" \
    N_SAMPLES="$N_SAMPLES" NUM_EPISODES="$1" MAX_TRAIN_GROUPS=0 \
    EVAL_STEPS="${EVAL_STEPS:-10}" EVAL_AT_START=0 CKPT_SAVE_STEPS=10 CKPT_SAVE_HF=1 \
    CKPT_MAX_NUM=16 CKPT_DCP_MAX_NUM=2 RESUME="$2" \
    bash "$ROOT/scripts/train.sh" "$RUN"
}
# DRY_RUN=1 (run.py --dry-run): print the final boundary's Molt command; touch nothing.
if [ "${DRY_RUN:-0}" = 1 ]; then train "${TARGET_EPISODES##* }" 1; exit; fi

mkdir -p "$RUN_ROOT"
FSTYPE=$(findmnt -T "$RUN_ROOT" -n -o FSTYPE)
# ALLOW_NON_LUSTRE=1: runs on local/NFS storage (single-node topologies such as
# config/topology/single_h100.env, and nodes set up by scripts/bootstrap.sh, which
# writes it into config/env.sh); keep ~50 GB free per kept optimizer state.
[ "$FSTYPE" = lustre ] || [ "${ALLOW_NON_LUSTRE:-0}" = 1 ] || { echo "RUN_ROOT must be on Lustre (got $FSTYPE: $RUN_ROOT); set ALLOW_NON_LUSTRE=1 on other sites" >&2; exit 2; }
log() { echo "$(date -Is) $*" >> "$LOG"; }
# The planned generated-rollout budget must equal the env contract
# (contracts/bfcl.json); train.sh repeats the model/env checks on every launch.
mkdir -p "$RUN_DIR"
PYTHONPATH="$ROOT" "$MOLT_VENV/bin/python" -m ga_grpo.admission --env bfcl \
  --model "${MODEL:-$MODELS_DIR/Qwen3-4B-Instruct-2507}" --group-size "$N_SAMPLES" \
  --budget $(( ${TARGET_EPISODES##* } * DATASET_GROUPS * N_SAMPLES )) \
  ${ALLOW_PROTOCOL_CHANGE:+--allow-protocol-change "$ALLOW_PROTOCOL_CHANGE"} \
  --report "$RUN_DIR/admission_contract.json" >> "$LOG" 2>&1 \
  || { log "ADMISSION FAILED: see $RUN_DIR/admission_contract.json"; exit 78; }

has_checkpoint() {
  [ -d "$RUN_DIR/state/_actor" ]
}

last_sampled_groups() {
  grep -a -o -E "'rollout/total_prompt_groups_sampled': [0-9.]+" \
    "$RUN_DIR/logs/train.log" | tail -1 | grep -o -E '[0-9]+' | head -1 || true
}

snapshot_export() {
  local target="$1" dst="$RUN_DIR/hf_ep$1"
  local step src
  step=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["global_step"])' "$progress")
  src="$RUN_DIR/state/_hf/global_step$step"
  if [ ! -e "$dst" ]; then
    # Versioned weights match the checked DCP counters; mutable hf/ might be
    # stale if the process died after DCP save but before the final HF export.
    cp -al "$src" "$dst"
  fi
  python - "$dst" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
index = json.loads((p / 'model.safetensors.index.json').read_text())
assert all((p / f).is_file() and (p / f).stat().st_size > 0 for f in set(index['weight_map'].values()))
PY
}

snapshot_resume_state() {
  local target="$1" src="$RUN_DIR/state" dst="$RUN_DIR/state_ep$1"
  [ -d "$src/_actor" ] || { echo "missing resumable actor state: $src/_actor" >&2; return 1; }
  if [ ! -d "$dst/_actor" ]; then
    mkdir -p "$dst"
    cp -al "$src/_actor" "$dst/_actor"
    [ ! -d "$src/_critic" ] || cp -al "$src/_critic" "$dst/_critic"
  fi
  find "$dst/_actor" -maxdepth 1 -type d -name 'global_step*' -print -quit | grep -q .
}

source "$MOLT_VENV/bin/activate"
cd "$ROOT"
mkdir -p "$RUN_DIR"
exec 7>"$RUN_DIR/run.lock"
flock -n 7 || { log "NON_RETRYABLE_RESUME another driver owns this run"; exit 78; }

previous=0
for target in $TARGET_EPISODES; do
  if [ "$target" -le "$previous" ]; then
    echo "TARGET_EPISODES must be strictly increasing" >&2
    exit 2
  fi
  previous="$target"
  expected_groups=$((target * DATASET_GROUPS))
  expected_rollouts=$((expected_groups * N_SAMPLES))
  marker="$RUN_DIR/boundaries/ep${target}.complete"

  if [ -s "$marker" ] && [ -s "$RUN_DIR/hf_ep$target/model.safetensors.index.json" ]; then
    log "reuse_boundary episode=$target expected_rollouts=$expected_rollouts marker=$marker"
    continue
  fi

  resume=""
  action=train
  guard_flags=()
  [ "$CURR" = 0 ] || guard_flags+=(--curriculum)
  if has_checkpoint; then
    progress=$(python -m ga_grpo.resume_guard "$RUN_DIR" --dataset-groups "$DATASET_GROUPS" \
      --target-groups "$expected_groups" "${guard_flags[@]}") || exit 78
    action=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["action"])' "$progress")
    resume=1
  fi
  log "train_start episode=$target resume=${resume:-0} generated_rollout_budget=$expected_rollouts"
  if [ "$action" != complete ]; then
  ray stop --force >/dev/null 2>&1 || true
  train "$target" "$resume" >> "$LOG" 2>&1
  fi

  # Read committed checkpoint counters, not the last optimizer log line:
  # all-filtered tails consume budget without logging an optimizer update.
  progress=$(python -m ga_grpo.resume_guard "$RUN_DIR" --dataset-groups "$DATASET_GROUPS" \
    --target-groups "$expected_groups" "${guard_flags[@]}") || exit 78
  sampled=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["sampled_groups"])' "$progress")
  if [ "${sampled:-}" != "$expected_groups" ]; then
    log "ERROR generated budget mismatch episode=$target expected_groups=$expected_groups actual=${sampled:-missing}"
    exit 78
  fi
  snapshot_export "$target"
  snapshot_resume_state "$target"
  mkdir -p "$(dirname "$marker")"
  printf 'episode=%s\nprompt_groups=%s\ngenerated_rollouts=%s\n' \
    "$target" "$sampled" "$expected_rollouts" > "$marker"
  log "boundary_complete episode=$target sampled_groups=$sampled generated_rollouts=$expected_rollouts"
done

# "Final" is the last TARGET_EPISODES boundary. The contract's 50 keeps the
# file names every earlier run used; an extension (TARGET_EPISODES="30 50 80"
# with ALLOW_PROTOCOL_CHANGE) gets its own final log and test75 directory so
# the 50-episode result is not reused or overwritten.
FINAL_EP="${TARGET_EPISODES##* }"
FINAL_ROLLOUTS=$((FINAL_EP * DATASET_GROUPS * N_SAMPLES))
FINAL_HF="$RUN_DIR/hf_ep$FINAL_EP"
test -s "$FINAL_HF/model.safetensors.index.json"
if [ "$FINAL_EP" = 50 ]; then FINAL_VAL_LOG="$RUN_DIR/logs/final_val25.log"; else FINAL_VAL_LOG="$RUN_DIR/logs/final_val25_ep$FINAL_EP.log"; fi
if ! grep -a -q "Eval-only completed:" "$FINAL_VAL_LOG" 2>/dev/null; then
  log "final_val_start split=val25 checkpoint=$FINAL_HF"
  EVAL_DATA="$VAL_DATA" bash "$ROOT/scripts/eval_monitor.sh" "$FINAL_HF" "$FINAL_VAL_LOG" >> "$LOG" 2>&1
fi
python3 "$ROOT/scripts/summarize_validation.py" "$RUN_DIR" \
  --final-log "$FINAL_VAL_LOG" --final-checkpoint "$FINAL_HF" --n-samples "$N_SAMPLES" \
  --output "$RUN_DIR/validation_summary.json" >> "$LOG"

BEST_STEP=$(python3 -c "import json; print(json.load(open('$RUN_DIR/validation_summary.json'))['best']['optimizer_step'])")
BEST_HF=$(python3 -c "import json; print(json.load(open('$RUN_DIR/validation_summary.json'))['best']['checkpoint'])")
test -s "$BEST_HF/model.safetensors.index.json" || { log "ERROR selected checkpoint missing path=$BEST_HF"; exit 1; }

BEST_OUT="$RUN_ROOT/${RUN}_best_step${BEST_STEP}_${BFCL_PROFILE}_test75_3decode"
FINAL_OUT="$RUN_ROOT/${RUN}_final_r$((FINAL_ROLLOUTS / 1000))k_${BFCL_PROFILE}_test75_3decode"
if [ "$BEST_HF" = "$FINAL_HF" ]; then FINAL_OUT="$BEST_OUT"; fi

if [ ! -s "$BEST_OUT/summary.json" ]; then
  log "test75_start kind=best step=$BEST_STEP checkpoint=$BEST_HF"
  TASK_MANIFEST="$TEST_MANIFEST" bash "$ROOT/scripts/eval_official.sh" "$BEST_HF" "$BEST_OUT" >> "$LOG" 2>&1
fi
if [ "$FINAL_OUT" != "$BEST_OUT" ] && [ ! -s "$FINAL_OUT/summary.json" ]; then
  log "test75_start kind=final generated_rollouts=$FINAL_ROLLOUTS checkpoint=$FINAL_HF"
  TASK_MANIFEST="$TEST_MANIFEST" bash "$ROOT/scripts/eval_official.sh" "$FINAL_HF" "$FINAL_OUT" >> "$LOG" 2>&1
fi

# The contract's result is comparison_result.json; an extension past 50 episodes
# keeps its own file so the 80k comparison is never overwritten.
if [ "$FINAL_EP" = 50 ]; then RESULT="$RUN_DIR/comparison_result.json"; else RESULT="$RUN_DIR/comparison_result_ep$FINAL_EP.json"; fi
python3 "$ROOT/scripts/write_comparison_result.py" \
  --run "$RUN_DIR" --method "$METHOD" --seed "$SEED" --expected-rollouts "$FINAL_ROLLOUTS" \
  --best-official "$BEST_OUT/summary.json" --final-official "$FINAL_OUT/summary.json" \
  --output "$RESULT" >> "$LOG"
log "SUMMARY_COMPLETE method=$METHOD seed=$SEED result=$RESULT"
