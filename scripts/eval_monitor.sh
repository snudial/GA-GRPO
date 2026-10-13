#!/usr/bin/env bash
# Molt-path monitor evaluation of a fixed HF checkpoint, no training.
#   source config/env.sh
#   bash scripts/eval_monitor.sh HF_DIR LOG_FILE
# Reproduces the in-training eval contract of train.sh (bfcl_agent.py,
# data/bfcl/bfcl_val.jsonl, 3 samples at temperature 0.01, 4096 new tokens,
# 10000 total context). This is the MONITOR number, not the official score;
# use scripts/eval_official.sh for anything that is compared.
set -euo pipefail

HF="$1"
LOG="$2"
ROOT="${GA_GRPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$ROOT/config/env.sh" ] && source "$ROOT/config/env.sh"
source "$ROOT/config/topology.sh"   # hardware profile (TOPOLOGY); see config/topology/
: "${MOLT_VENV:?set MOLT_VENV in config/env.sh}"
MOLT_DIR="${MOLT_DIR:-$ROOT/third_party/molt}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache}"

# Same protocol profile as training (config/bfcl_protocol.sh).
source "$ROOT/config/bfcl_protocol.sh"
export BFCL_OPSD_ROOT="$ROOT"
export BFCL_CURRICULUM=0
AGENT_PATH="$ROOT/ga_grpo/bfcl_agent.py"
TRAIN_DATA="$ROOT/data/bfcl/bfcl_train.jsonl"
EVAL_DATA="${EVAL_DATA:-$ROOT/data/bfcl/bfcl_val.jsonl}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
source "$MOLT_VENV/bin/activate"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXP2_CURRICULUM_STATE="$(dirname "$LOG")/evalonly_curriculum_$$.json"
export EXP2_CURRICULUM_WINDOW=16
export BFCL_TRACE_ROLLOUTS=0

mkdir -p "$(dirname "$LOG")"
ray stop --force >/dev/null 2>&1 || true
ray start --head --num-gpus="${NUM_GPUS:-4}" --disable-usage-stats >/dev/null
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

cd "$MOLT_DIR"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$HF" \
  --data.prompt_dataset "$TRAIN_DATA" \
  --data.input_key input \
  --data.label_key label \
  --data.apply_chat_template \
  --data.max_len "${MAX_LEN:-10000}" \
  --rollout.batch_size 32 \
  --rollout.n_samples_per_prompt 16 \
  --rollout.max_new_tokens "${MAX_NEW:-4096}" \
  --rollout.micro_batch_size 1 \
  --rollout.temperature 1.0 \
  --train.batch_size 32 \
  --train.micro_batch_size 1 \
  --train.seed 42 \
  --train.agent_path "$AGENT_PATH" \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "${NUM_GPUS:-4}" \
  --vllm.num_engines "${NUM_GPUS:-4}" \
  --vllm.tensor_parallel_size 1 \
  --vllm.gpu_memory_utilization 0.85 \
  --vllm.distributed_executor_backend mp \
  --rollout.vllm_generate_batch_size 16 \
  --algo.advantage.estimator reinforce_baseline \
  --algo.advantage.is_correction_level geo \
  --eval.dataset "$EVAL_DATA" \
  --eval.n_samples_per_prompt "${EVAL_SAMPLES:-3}" \
  --eval.temperature "${EVAL_TEMP:-0.01}" \
  --eval.eval_only \
  2>&1 | tee -a "$LOG"
grep -a "Eval-only completed:" "$LOG" | tail -1
