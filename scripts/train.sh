#!/usr/bin/env bash
# GA-GRPO training launcher: BFCL-v3 multi-turn RL with Molt on one 4x GPU node.
#
#   source config/env.sh
#   bash scripts/train.sh RUN_NAME                       # GRPO      (no curriculum, partial batches)
#   ALGO=dapo bash scripts/train.sh RUN_NAME             # DAPO      (fill 32 groups per step)
#   BFCL_CURRICULUM=1 bash scripts/train.sh RUN_NAME     # GA-GRPO (Jump-start handoff curriculum)
#   BFCL_CURRICULUM=1 ALGO=dapo bash scripts/train.sh RUN_NAME   # JUMP-DAPO
#   DRY_RUN=1 bash scripts/train.sh x                    # print the Molt command and exit
#
# Two axes, everything else shared:
#   ALGO             grpo | dapo   batch rule (see "Batch rule" below)
#   BFCL_CURRICULUM  0 | 1         reference roll-in with a per-task handoff turn h
#                                  that the curriculum walks toward 0 (ga_grpo/curriculum.py)
#
# Defaults reproduce the author-lineage training contract used for the paper's
# BFCL results: 100 Base tasks, group 16, 32 groups per step, temperature 1,
# 4096 new tokens per turn, 10k context, AdamW 1e-6 / wd 0.01, dynamic
# filtering of all-fail / all-pass groups. Outputs go to $RUN_ROOT/RUN_NAME
# (hf/ export, state/ optimizer, logs/train.log, curriculum.json).
set -euo pipefail

ROOT="${GA_GRPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$ROOT/config/env.sh" ] && source "$ROOT/config/env.sh"
source "$ROOT/config/topology.sh"   # hardware profile (TOPOLOGY); see config/topology/
: "${MOLT_VENV:?set MOLT_VENV in config/env.sh}" "${MODELS_DIR:?set MODELS_DIR in config/env.sh}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache}"
MOLT_DIR="${MOLT_DIR:-$ROOT/third_party/molt}"
RUN_NAME="${1:-bfcl_v3_$(date +%Y%m%d_%H%M)}"
RUN="$RUN_ROOT/$RUN_NAME"

# --- protocol profile: one BFCL checkout / handler / venv for rollouts AND the
# official evaluation (config/bfcl_protocol.sh; README.md).
source "$ROOT/config/bfcl_protocol.sh"
export BFCL_OPSD_ROOT="$ROOT"

# --- the two axes
ALGO="${ALGO:-grpo}"
export BFCL_CURRICULUM="${BFCL_CURRICULUM:-0}"
if [ "$BFCL_CURRICULUM" = "1" ]; then
  TRAIN_DATA="${TRAIN_DATA:-$ROOT/data/bfcl/bfcl_jumpstart_train.jsonl}"   # rows carry guide_turns=-1
else
  TRAIN_DATA="${TRAIN_DATA:-$ROOT/data/bfcl/bfcl_train.jsonl}"
fi
EVAL_DATA="${EVAL_DATA:-$ROOT/data/bfcl/bfcl_val.jsonl}"                   # autonomous validation only
AGENT_PATH="$ROOT/ga_grpo/bfcl_agent.py"

# --- model and training contract
MODEL="${MODEL:-$MODELS_DIR/Qwen3-4B-Instruct-2507}"
N_SAMPLES="${N_SAMPLES:-16}"          # group size
ROLLOUT_BS="${ROLLOUT_BS:-32}"        # accepted prompt groups per optimizer step
TRAIN_BS="${TRAIN_BS:-32}"
NUM_EPISODES="${NUM_EPISODES:-50}"    # epochs over the 100 prompts; upper bound when a group budget is set
# Budget in ACCEPTED prompt groups (the units that carry gradient), cumulative
# across resumes; 0 = unlimited. 960 = 30 full DAPO steps of 32 groups, or about
# 60 partial GRPO steps, i.e. the same learning signal for both axes. The run stops when it is spent.
MAX_TRAIN_GROUPS="${MAX_TRAIN_GROUPS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"         # optimizer passes per rollout batch
MAX_NEW="${MAX_NEW:-4096}"            # per assistant turn
MAX_LEN="${MAX_LEN:-10000}"           # whole episode (prompt + all turns)
ROLLOUT_TEMP="${ROLLOUT_TEMP:-1.0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"
TRAIN_SEED="${TRAIN_SEED:-42}"

# --- in-training MONITOR eval (Molt path). Not an official score: the same weights
# scored 12-15 points lower on the official pool for GRPO models under the old
# mixed protocol. Use scripts/eval_official.sh for anything that is compared.
EVAL_SAMPLES="${EVAL_SAMPLES:-3}"
EVAL_TEMP="${EVAL_TEMP:-0.01}"
EVAL_STEPS="${EVAL_STEPS:-10}"
CKPT_SAVE_HF="${CKPT_SAVE_HF:-1}"       # weights-only snapshots at validation steps
EVAL_AT_START="${EVAL_AT_START:-0}"
SAVE_HF_FLAG=""
[ "$CKPT_SAVE_HF" = "1" ] && SAVE_HF_FLAG="--ckpt.save_hf"
RETENTION_FLAGS=()
[ "${BEST_HF_ONLY:-0}" != 1 ] || RETENTION_FLAGS+=(--ckpt.best_hf_only)
[ "${DISABLE_FINAL_HF_EXPORT:-0}" != 1 ] || RETENTION_FLAGS+=(--ckpt.disable_final_save)
EVAL_START_FLAG=""
[ "$EVAL_AT_START" = "1" ] && EVAL_START_FLAG="--eval.eval_at_start"

# --- batch rule. Both axes drop uniform-reward groups (all fail / all pass).
#   grpo  train each step on whatever the epoch yielded: with 100 prompts and a
#         ~15% informative rate that is 3-32 groups (mean 15) — upstream Molt.
#   dapo  keep sampling across epoch boundaries until ROLLOUT_BS accepted groups
#         exist (DAPO dynamic sampling, the CrEST-loop rule): every step is a
#         full batch, about half as many optimizer steps per episode budget.
FILTER_FLAGS="--algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0.01 0.99"
case "$ALGO" in
  grpo)
    export BFCL_DAPO=0
    ;;
  dapo)
    FILTER_FLAGS="$FILTER_FLAGS --algo.dynamic_filtering_fill_batch"
    export BFCL_DAPO=1
    export BFCL_DAPO_MAX_RESPONSE_TOKENS="${BFCL_DAPO_MAX_RESPONSE_TOKENS:-$MAX_LEN}"
    export BFCL_DAPO_OVERLONG_BUFFER_TOKENS="${BFCL_DAPO_OVERLONG_BUFFER_TOKENS:-2048}"
    export BFCL_DAPO_OVERLONG_PENALTY_FACTOR="${BFCL_DAPO_OVERLONG_PENALTY_FACTOR:-1.0}"
    (( BFCL_DAPO_MAX_RESPONSE_TOKENS > 0 )) || { echo "BFCL_DAPO_MAX_RESPONSE_TOKENS must be positive" >&2; exit 2; }
    (( BFCL_DAPO_OVERLONG_BUFFER_TOKENS > 0 && BFCL_DAPO_OVERLONG_BUFFER_TOKENS <= BFCL_DAPO_MAX_RESPONSE_TOKENS )) || {
      echo "BFCL_DAPO_OVERLONG_BUFFER_TOKENS must be in [1, max response tokens]" >&2; exit 2;
    }
    ;;
  *) echo "ALGO must be grpo or dapo (got $ALGO)" >&2; exit 2 ;;
esac

# --- GPU topology. HYBRID=1 (default, measured 82% mean utilization): all four
# GPUs serve both roles — 4 vLLM engines generate, then sleep while the 4-way
# FSDP actor trains, then wake for the weight broadcast (gloo: NCCL refuses two
# communicator members on one device). HYBRID=0: 2 actor GPUs + 2 engines, NCCL.
HYBRID="${HYBRID:-1}"
if [ "$HYBRID" = "1" ]; then
  export EXP2_HYBRID=1
  ACTOR_GPUS="${ACTOR_GPUS:-4}"; VLLM_ENGINES="${VLLM_ENGINES:-$ACTOR_GPUS}"; ASYNC_QUEUE=1
  # 0.50 (was 0.30) from seed 44 on, 2026-09-06: generation is 81% of step time
  # and the vLLM scheduler showed ~15 waiting requests while KV usage sat at
  # 33-52%, i.e. KV was not the limit; the prefill budget below was. The extra
  # KV headroom only serves the larger prefill chunks. Actor peak 54 GB + engine
  # 0.50 x 179 GB fits a B200 with the engines asleep during training.
  VLLM_UTIL="${VLLM_UTIL:-0.50}"
  SYNC_FLAGS="--train.force_sync_mode --vllm.sync_backend gloo"
else
  ACTOR_GPUS="${ACTOR_GPUS:-2}"; VLLM_ENGINES="${VLLM_ENGINES:-2}"; ASYNC_QUEUE="${ASYNC_QUEUE:-2}"
  VLLM_UTIL="${VLLM_UTIL:-0.80}"
  SYNC_FLAGS="--vllm.sync_backend nccl"
fi
# vLLM scheduler token budget per iteration. vLLM's chunked-prefill default is
# 2048, but BFCL prompts average 6.8k tokens (max 10k) and every episode makes
# ~10 LLM calls, so each call spent 3-4 scheduler iterations just prefilling
# (measured 2026-09-06 on GRPO seed44: Running ~20 / Waiting ~19 requests with
# KV usage 33-52%). Setting it to MAX_LEN lets every prefill fit in one chunk, as
# Molt's own flag help recommends. This changes scheduling only, not sampling.
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-$MAX_LEN}"
# Rollout runner processes (molt AgentRunnerActor, 1 Ray CPU each). Every tool
# call, ground-truth replay, grade and Experience build of an episode runs
# in-process on its runner, so the 16 in-flight prompt groups (256 episodes)
# share NUM_RUNNERS GILs; Molt's default of 2 was every run up to 2026-10-08.
# A topology profile sizes it from its Ray CPU budget (config/topology/).
NUM_RUNNERS="${NUM_RUNNERS:-2}"
[[ "$NUM_RUNNERS" =~ ^[0-9]+$ ]] && (( NUM_RUNNERS >= 1 )) || { echo "NUM_RUNNERS must be a positive integer (got $NUM_RUNNERS)" >&2; exit 2; }
# Sequence packing through Transformer Engine (2.12.0 is the newest that loads
# on this node); micro_batch stays 1 — without packing it pads to the longest
# sequence (4.3% vs 13% MFU measured).
# Attention backend, packing, prefix caching and the Linear backend follow from
# the model (ga_grpo/model_profile.py) — one rule for BFCL and AppWorld, so a
# variation that changes --model changes nothing else by hand.
eval "$(PYTHONPATH="$ROOT" python3 -m ga_grpo.model_profile "$MODEL" --shell)"
if [ "$PACKING" = "1" ]; then PACK_FLAGS="--fsdp.packing_samples"; else PACK_FLAGS=""; fi
# TRAIN_MAX_TOKENS=N: pack several samples into one training forward up to N
# tokens (Molt --train.dynamic_batch_enable, first-fit by length) instead of
# one sample per forward. The optimizer step stays TRAIN_BS samples with the
# same global token-mean, so only the kernel shapes change. Measured on
# Qwen3-0.6B, one H100 94GB, 128 replayed samples of ~6.9k tokens: 0 -> 49 s, 14.9k tok/s,
# peak 19.5 GB; 32768 -> 61 s, 12.0k tok/s, peak 51.7 GB; 65536 -> OOM. The
# 152k-vocab fp32 logits/log-softmax/entropy traffic, not the transformer
# body, bounds that model, so larger forwards only cost memory; leave it 0.
TRAIN_MAX_TOKENS="${TRAIN_MAX_TOKENS:-0}"
[[ "$TRAIN_MAX_TOKENS" =~ ^[0-9]+$ ]] || { echo "TRAIN_MAX_TOKENS must be a non-negative integer (got $TRAIN_MAX_TOKENS)" >&2; exit 2; }
DYNAMIC_BATCH_FLAGS=()
if (( TRAIN_MAX_TOKENS > 0 )); then
  [ "$PACKING" = 1 ] || { echo "TRAIN_MAX_TOKENS needs packed batches (PACKING=1)" >&2; exit 2; }
  (( TRAIN_MAX_TOKENS >= MAX_LEN )) || { echo "TRAIN_MAX_TOKENS must hold one episode (MAX_LEN=$MAX_LEN)" >&2; exit 2; }
  DYNAMIC_BATCH_FLAGS=(--train.dynamic_batch_enable --train.max_tokens_per_gpu "$TRAIN_MAX_TOKENS"
                       --rollout.max_tokens_per_gpu "$TRAIN_MAX_TOKENS")
fi

# Ray workers must import this worktree, not the shared venv's editable checkout.
export PYTHONPATH="$MOLT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Independent rollback switches: neither changes the batch or sampling recipe.
export MOLT_GLOO_REUSE_BUFFER="${MOLT_GLOO_REUSE_BUFFER:-1}"
export MOLT_REUSE_ACTOR_CACHE="${MOLT_REUSE_ACTOR_CACHE:-1}"
for gpu_opt_flag in "$MOLT_GLOO_REUSE_BUFFER" "$MOLT_REUSE_ACTOR_CACHE"; do
  [[ "$gpu_opt_flag" == 0 || "$gpu_opt_flag" == 1 ]] || {
    echo "GPU optimization flags must be 0 or 1" >&2; exit 2;
  }
done

# --- environment for the Molt venv and the rollout agent
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"   # /tmp is noexec here
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXP2_CURRICULUM_STATE="$RUN/curriculum.json"          # per-task handoff state (flock'd JSON)
export EXP2_CURRICULUM_WINDOW="${EXP2_CURRICULUM_WINDOW:-$N_SAMPLES}"
export BFCL_TRACE_DIR="$RUN/rollout_trace"
export EXP2_VERSION_FILE="$RUN/policy_version.txt"           # stamped by the trainer at each weight sync
# OPSD=1: verifier-gated privileged distillation on all-zero groups (EMA
# self-teacher; see molt/trainer/workers/exp2_opsd.py and ga_grpo/opsd_data.py).
if [ "${OPSD:-0}" = "1" ]; then
  export EXP2_EMA_BETA="${EXP2_EMA_BETA:-0.995}"
  export EXP2_HOOK_DIR="$ROOT/ga_grpo"
  export EXP2_BANK_DIR="$RUN/opsd_bank"
  export EXP2_OPSD_LAMBDA="${EXP2_OPSD_LAMBDA:-1.0}"
fi
CHECK_WEIGHT_FLAG=""
[ "${CHECK_WEIGHT_UPDATE:-0}" = "1" ] && CHECK_WEIGHT_FLAG="--train.check_weight_update_equal"

REPLAY_FLAGS=()
[ -z "${ROLLOUT_DUMP_DIR:-}" ] || REPLAY_FLAGS+=(--train.rollout_dump_dir "$ROLLOUT_DUMP_DIR")
[ -z "${ROLLOUT_REPLAY_DIR:-}" ] || REPLAY_FLAGS+=(--train.rollout_replay_dir "$ROLLOUT_REPLAY_DIR")
VLLM_FLAGS=()
[ "${VLLM_ENFORCE_EAGER:-0}" != 1 ] || VLLM_FLAGS+=(--vllm.enforce_eager)
# Prefix KV cache: each multi-turn call reuses the KV of the unchanged history
# instead of re-prefilling it (98.8% of the s42 run's vLLM tokens were such
# re-prefill). vLLM drops the cache when the engine sleeps and Molt resets it
# after every weight broadcast, so KV from an older policy is never reused; only
# bf16 numerics move. model_profile turns it on for dense-attention models, where
# vllm_kl was measured unchanged on the B200 runs; a topology or the caller can
# still set VLLM_PREFIX_CACHING=0 (a separate GPU timing comparison compares both).
[ "$VLLM_PREFIX_CACHING" != 1 ] || VLLM_FLAGS+=(--vllm.enable_prefix_caching)

CMD=(python3 -u -m molt.cli.train_rl_ray
  --actor.model_name_or_path "$MODEL"
  --data.prompt_dataset "$TRAIN_DATA"
  --data.input_key input --data.label_key label --data.apply_chat_template
  --data.max_len "$MAX_LEN"
  --rollout.batch_size "$ROLLOUT_BS"
  --rollout.n_samples_per_prompt "$N_SAMPLES"
  --rollout.max_new_tokens "$MAX_NEW"
  --rollout.micro_batch_size 1
  --rollout.temperature "$ROLLOUT_TEMP"
  --rollout.vllm_generate_batch_size 16
  --rollout.num_runners "$NUM_RUNNERS"
  --train.batch_size "$TRAIN_BS"
  --train.micro_batch_size 1
  --train.max_epochs "$MAX_EPOCHS"
  --train.num_episodes "$NUM_EPISODES"
  --train.max_train_groups "$MAX_TRAIN_GROUPS"
  --train.seed "$TRAIN_SEED"
  --train.async_queue_size "$ASYNC_QUEUE"
  $CHECK_WEIGHT_FLAG
  $SYNC_FLAGS
  --train.agent_path "$AGENT_PATH"
  --actor.num_nodes 1
  --actor.num_gpus_per_node "$ACTOR_GPUS"
  --vllm.num_engines "$VLLM_ENGINES"
  --vllm.tensor_parallel_size 1
  --vllm.gpu_memory_utilization "$VLLM_UTIL"
  --vllm.max_num_batched_tokens "$VLLM_MAX_BATCHED_TOKENS"
  --vllm.distributed_executor_backend mp
  "${VLLM_FLAGS[@]}"
  --fsdp.param_dtype bf16
  --fsdp.offload "${FSDP_OFFLOAD:-none}"
  --fsdp.attn_implementation "${ATTN_IMPLEMENTATION:-te}"
  $PACK_FLAGS
  "${DYNAMIC_BATCH_FLAGS[@]}"
  --actor.gradient_checkpoint full
  --actor.adam.lr "$ACTOR_LR"
  --actor.adam.weight_decay "$WEIGHT_DECAY"
  --actor.entropy_coef "$ENTROPY_COEF"
  --actor.eps_clip_low_high 0.2 0.27
  --algo.advantage.estimator reinforce_baseline
  --algo.advantage.is_correction_level geo
  --algo.advantage.is_correction_threshold 0.99 1.01
  --algo.kl.init_coef 0
  $FILTER_FLAGS
  --reward.clip_range -10 10
  --eval.dataset "$EVAL_DATA"
  --eval.steps "$EVAL_STEPS"
  --eval.n_samples_per_prompt "$EVAL_SAMPLES"
  --eval.temperature "$EVAL_TEMP"
  $EVAL_START_FLAG
  --ckpt.output_dir "$RUN/hf"
  --ckpt.path "$RUN/state"
  --ckpt.save_steps "${CKPT_SAVE_STEPS:-10}"
  --ckpt.max_num "${CKPT_MAX_NUM:-16}"
  --ckpt.dcp_max_num "${CKPT_DCP_MAX_NUM:-2}"
  $SAVE_HF_FLAG
  "${RETENTION_FLAGS[@]}"
  ${RESUME:+--ckpt.load_enable}
  "${REPLAY_FLAGS[@]}"
  --logger.logging_steps 1)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "RUN=$RUN  ALGO=$ALGO  BFCL_CURRICULUM=$BFCL_CURRICULUM  BFCL_PROFILE=$BFCL_PROFILE"
  echo "TOPOLOGY=${TOPOLOGY:-none}  HYBRID=$HYBRID  VLLM_UTIL=$VLLM_UTIL  VLLM_MAX_BATCHED_TOKENS=$VLLM_MAX_BATCHED_TOKENS  VLLM_PREFIX_CACHING=${VLLM_PREFIX_CACHING:-0}  FSDP_OFFLOAD=${FSDP_OFFLOAD:-none}  NUM_RUNNERS=$NUM_RUNNERS  TRAIN_MAX_TOKENS=$TRAIN_MAX_TOKENS"
  echo "MOLT_GLOO_REUSE_BUFFER=$MOLT_GLOO_REUSE_BUFFER MOLT_REUSE_ACTOR_CACHE=$MOLT_REUSE_ACTOR_CACHE"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
export HYBRID VLLM_UTIL VLLM_MAX_BATCHED_TOKENS NUM_RUNNERS TRAIN_MAX_TOKENS FSDP_OFFLOAD="${FSDP_OFFLOAD:-none}"   # recorded by write_run_manifest.py

if [ -z "${RESUME:-}" ] && { [ -e "$RUN/manifest.json" ] || [ -e "$RUN/state" ] || [ -e "$RUN/hf" ] || [ -e "$RUN/logs/train.log" ]; }; then
  echo "Refusing to overwrite existing run: $RUN; use RESUME only for intentional continuation" >&2
  exit 78
fi
mkdir -p "$RUN/logs" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# Admission gate (ga_grpo/admission.py): every launch path — smoke, full run,
# resume — must pass the shared model/env checks before any GPU work.
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$MOLT_VENV/bin/python" -m ga_grpo.admission \
  --env bfcl --model "$MODEL" --group-size "$N_SAMPLES" --max-len "$MAX_LEN" \
  ${ALLOW_PROTOCOL_CHANGE:+--allow-protocol-change "$ALLOW_PROTOCOL_CHANGE"} \
  --report "$RUN/admission.json" || { echo "ADMISSION FAILED: $RUN/admission.json" >&2; exit 78; }
if [ -n "${RESUME:-}" ]; then
  guard_flags=()
  [ "$BFCL_CURRICULUM" = 0 ] || guard_flags+=(--curriculum)
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$MOLT_VENV/bin/python" -m ga_grpo.resume_guard \
    "$RUN" --dataset-groups "$(wc -l < "$TRAIN_DATA")" \
    --target-groups "$((NUM_EPISODES * $(wc -l < "$TRAIN_DATA")))" \
    "${guard_flags[@]}" || exit 78
fi
"$MOLT_VENV/bin/python" "$ROOT/scripts/write_run_manifest.py" --root "$ROOT" --run "$RUN" --command "${CMD[@]}"
source "$MOLT_VENV/bin/activate"
if [ "${ISOLATED_RAY:-0}" = 1 ]; then
  export RAY_ADDRESS=local
  STARTED_RAY=0
elif ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus="${RAY_NUM_GPUS:-4}" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && ray stop --force >/dev/null 2>&1 || true' EXIT

cd "$MOLT_DIR"   # `python -m molt` must resolve to the patched submodule, not another install
echo "[train.sh] $(date -Is) run=$RUN_NAME algo=$ALGO curriculum=$BFCL_CURRICULUM profile=$BFCL_PROFILE model=$MODEL" | tee -a "$RUN/logs/train.log"
if [ "${ISOLATED_RAY:-0}" = 1 ]; then
  python3 -u -c '
import ctypes, faulthandler, os, runpy, sys
ctypes.CDLL(None).prctl(4, 0)
import ray
ray_init = ray.init
def init_local_ray(*args, **kwargs):
    kwargs.update(address="local", num_cpus=int(os.environ.get("RAY_NUM_CPUS", "8")),
                  num_gpus=int(os.environ.get("RAY_NUM_GPUS", "1")),
                  object_store_memory=4 * 1024**3,
                  object_spilling_directory=os.path.join(os.environ["CACHE_ROOT"], "ray_spill"))
    result = ray_init(*args, **kwargs)
    faulthandler.enable(all_threads=True)
    return result
ray.init = init_local_ray
sys.argv = sys.argv[1:]
try:
    runpy.run_module(sys.argv[0], run_name="__main__")
finally:
    ray.shutdown()
' "${CMD[@]:3}" 2>&1 | tee -a "$RUN/logs/train.log"
else
  "${CMD[@]}" 2>&1 | tee -a "$RUN/logs/train.log"
fi
