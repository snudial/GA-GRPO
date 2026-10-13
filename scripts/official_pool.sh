#!/bin/bash
# Four independent B200 vLLM servers feeding the unmodified pinned BFCL CLI.
# Scoring is CPU-only and runs in the background, overlapped with the next
# decode's server boot and generation, so the GPUs never wait on the official
# scorer; all scoring jobs are joined before the summary is written.
set -euo pipefail

ROOT="${OPSD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/config/env.sh"
source "$ROOT/config/topology.sh"   # hardware profile (TOPOLOGY); see config/topology/
source "$ROOT/config/bfcl_protocol.sh"
MODEL_PATH="${MODEL_PATH:-$QWEN3_4B_INSTRUCT_2507_PATH}"
[[ -d "$MODEL_PATH" ]] || { echo "missing model/checkpoint: $MODEL_PATH" >&2; exit 1; }
MODEL_PATH="$(realpath "$MODEL_PATH")"

CATEGORY="${CATEGORY:-multi_turn_base}"
TASK_MANIFEST="${TASK_MANIFEST:-$ROOT/config/splits/bfcl_aworld_lineage_seed42_v1/base_eval.json}"
BFCL_CLI_VENV="${BFCL_CLI_VENV:-$BFCL_RUNTIME_VENV}"
BFCL_SOURCE_ROOT="${BFCL_SOURCE_ROOT:-$BFCL_RUNTIME_SOURCE_ROOT}"
BFCL_COMMIT="${BFCL_COMMIT:-$BFCL_RUNTIME_COMMIT}"
MODEL_REGISTRY_NAME="${MODEL_REGISTRY_NAME:-$BFCL_MODEL_REGISTRY_NAME}"
ALLOW_NON_FC_HANDLER="${ALLOW_NON_FC_HANDLER:-0}"
# Comparability label follows the protocol profile (config/bfcl_protocol.sh).
COMPARABILITY="${COMPARABILITY:-${BFCL_COMPARABILITY:-crest-bfcl-v3-candidate}}"
BFCL_SOURCE_ARGS=()
if [[ -n "$BFCL_SOURCE_ROOT" ]]; then
    BFCL_SOURCE_ARGS=(--bfcl-source-root "$BFCL_SOURCE_ROOT")
fi
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/bfcl_official_${CATEGORY}_${RUN_ID}}"
DECODES="${DECODES:-8}"  # 8 since 2026-10-05; decode d keeps seed SEED_BASE + d
TEMPERATURE="${TEMPERATURE:-0.000001}"
SEED_BASE="${SEED_BASE:-20260826}"
PORT_BASE="${PORT_BASE:-18001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
GPU_SAMPLE_SECONDS="${GPU_SAMPLE_SECONDS:-10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
BFCL_MAX_OUTPUT_TOKENS="${BFCL_MAX_OUTPUT_TOKENS:-4096}"
# vLLM serving venv for the evaluation servers (config/env.sh: BFCL_VLLM_VENV).
BFCL_VLLM_VENV="${BFCL_VLLM_VENV:-$ROOT/.venv/vllm-b200}"
VLLM_BIN="${VLLM_BIN:-$BFCL_VLLM_VENV/bin/vllm}"
VLLM_PYTHON="${VLLM_PYTHON:-$BFCL_VLLM_VENV/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NGPU=${#GPU_ARRAY[@]}
SHARDS_PER_GPU="${SHARDS_PER_GPU:-1}"
TOTAL_SHARDS=$((NGPU * SHARDS_PER_GPU))

[[ -x "$VLLM_BIN" ]] || { echo "missing B200 vLLM: $VLLM_BIN" >&2; exit 1; }
[[ -f "$TASK_MANIFEST" ]] || { echo "missing task manifest: $TASK_MANIFEST" >&2; exit 1; }
if [[ "$MODEL_REGISTRY_NAME" != *-FC && "$ALLOW_NON_FC_HANDLER" != "1" ]]; then
    echo "refusing non-FC BFCL registry '$MODEL_REGISTRY_NAME': it does not select QwenFCHandler" >&2
    echo "use the selected BFCL checkout's registry ending in -FC; set ALLOW_NON_FC_HANDLER=1 only for a labeled diagnostic" >&2
    exit 1
fi
(( NGPU > 0 )) || { echo "GPU_IDS must name at least one GPU" >&2; exit 1; }
available_gpus=",$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ',' )"
for gpu in "${GPU_ARRAY[@]}"; do
    [[ "$available_gpus" == *",$gpu,"* ]] || {
        echo "GPU_IDS contains unavailable GPU $gpu" >&2; exit 1;
    }
done
[[ $(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l) -eq "$NGPU" ]] || {
    echo "GPU_IDS contains duplicates" >&2; exit 1;
}

mkdir -p "$RUN_DIR/logs"
RUN_DIR="$(realpath "$RUN_DIR")"
cp "$TASK_MANIFEST" "$RUN_DIR/task_ids.json"
sha256sum "$RUN_DIR/task_ids.json" | awk '{print $1}' > "$RUN_DIR/task_ids.sha256"
TOTAL_TASKS=$("$BFCL_CLI_VENV/bin/python" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$TASK_MANIFEST")

MONITOR_PID=""
SERVER_PIDS=()
CLI_PIDS=()
SCORE_PIDS=()
cleanup() {
    local status=$?
    [[ -z "$MONITOR_PID" ]] || kill "$MONITOR_PID" 2>/dev/null || true
    for pid in "${CLI_PIDS[@]}" "${SCORE_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${SERVER_PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

echo "timestamp,gpu,memory_used_mib,memory_total_mib,utilization_percent" > "$RUN_DIR/gpu_metrics.csv"
(
    while true; do
        stamp=$(date --iso-8601=seconds)
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits | sed "s/^/$stamp,/"
        sleep "$GPU_SAMPLE_SECONDS"
    done
) >> "$RUN_DIR/gpu_metrics.csv" &
MONITOR_PID=$!

split_decode_ids() {
    local decode=$1
    "$BFCL_CLI_VENV/bin/python" - "$TASK_MANIFEST" "$RUN_DIR/decode_$decode" "$TOTAL_SHARDS" <<'PY'
import json, pathlib, sys
ids=json.load(open(sys.argv[1])); out=pathlib.Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
total_shards=int(sys.argv[3])
for shard_index in range(total_shards):
    shard=ids[shard_index::total_shards]; path=out/f"shard_{shard_index}"; path.mkdir(parents=True,exist_ok=True)
    (path/"task_ids.input.json").write_text(json.dumps(shard,indent=2)+"\n")
PY
}

start_servers() {
    local decode=$1
    SERVER_PIDS=()
    for ((local_gpu=0; local_gpu<NGPU; local_gpu++)); do
        physical_gpu=${GPU_ARRAY[$local_gpu]}
        port=$((PORT_BASE + local_gpu))
        CUDA_VISIBLE_DEVICES="$physical_gpu" OMP_NUM_THREADS=1 VLLM_LOGGING_LEVEL=INFO \
            "$VLLM_BIN" serve "$MODEL_PATH" \
            --host 127.0.0.1 --port "$port" --dtype bfloat16 \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" --generation-config vllm \
            --seed "$((SEED_BASE + decode))" \
            > "$RUN_DIR/logs/decode_${decode}_server_gpu_${physical_gpu}.log" 2>&1 &
        SERVER_PIDS+=("$!")
    done
    for ((local_gpu=0; local_gpu<NGPU; local_gpu++)); do
        physical_gpu=${GPU_ARRAY[$local_gpu]}
        port=$((PORT_BASE + local_gpu))
        for attempt in $(seq 1 300); do
            if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then break; fi
            if (( attempt == 300 )); then
                echo "server gpu=$physical_gpu port=$port did not become ready" >&2; return 1
            fi
            sleep 1
        done
    done
}

stop_servers() {
    for pid in "${SERVER_PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for pid in "${SERVER_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    SERVER_PIDS=()
}

score_shard() {
    # CPU-only official scoring of one generated shard; safe to run while the
    # next decode's servers occupy the GPUs.
    local decode=$1 shard_index=$2 physical_gpu=$3 local_gpu=$4
    local shard="$RUN_DIR/decode_$decode/shard_$shard_index"
    PYTHONPATH="$ROOT${BFCL_SOURCE_ROOT:+:$BFCL_SOURCE_ROOT}" "$BFCL_CLI_VENV/bin/python" \
        "$ROOT/scripts/pool_import_subset.py" \
        --run-dir "$shard" --task-id-manifest "$shard/task_ids.input.json" \
        --category "$CATEGORY" --model-path "$MODEL_PATH" \
        --model "$MODEL_REGISTRY_NAME" --bfcl-commit "$BFCL_COMMIT" \
        "${BFCL_SOURCE_ARGS[@]}" \
        --temperature "$TEMPERATURE" \
        --per-request-max-tokens "$BFCL_MAX_OUTPUT_TOKENS" \
        --backend-url "http://127.0.0.1:$((PORT_BASE + local_gpu))" \
        --backend-python "$VLLM_PYTHON" \
        --comparability "$COMPARABILITY"
}

for ((decode=0; decode<DECODES; decode++)); do
    split_decode_ids "$decode"
    start_servers "$decode"
    echo "$(date --iso-8601=seconds),decode_$decode,generation_start" >> "$RUN_DIR/events.csv"
    CLI_PIDS=()
    for ((shard_index=0; shard_index<TOTAL_SHARDS; shard_index++)); do
        local_gpu=$((shard_index % NGPU))
        physical_gpu=${GPU_ARRAY[$local_gpu]}
        shard="$RUN_DIR/decode_$decode/shard_$shard_index"
        project="$RUN_DIR/runtime/decode_$decode/shard_$shard_index"
        mkdir -p "$project"
        "$BFCL_CLI_VENV/bin/python" "$ROOT/scripts/pool_prepare_ids.py" \
            --ids "$shard/task_ids.input.json" --category "$CATEGORY" \
            --output "$project/test_case_ids_to_generate.json"
        (
            export BFCL_PROJECT_ROOT="$project"
            # v1.3 reads VLLM_*; the newer T3RL-pinned BFCL reads LOCAL_SERVER_*.
            # Export both so selecting the BFCL commit is the only behavior switch.
            export VLLM_ENDPOINT=127.0.0.1
            export VLLM_PORT=$((PORT_BASE + local_gpu))
            export LOCAL_SERVER_ENDPOINT=127.0.0.1
            export LOCAL_SERVER_PORT=$((PORT_BASE + local_gpu))
            export BFCL_MAX_OUTPUT_TOKENS
            if [[ -n "$BFCL_SOURCE_ROOT" ]]; then
                export PYTHONPATH="$BFCL_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
            fi
            "$BFCL_CLI_VENV/bin/bfcl" generate \
                --model "$MODEL_REGISTRY_NAME" --test-category "$CATEGORY" \
                --temperature "$TEMPERATURE" --backend vllm --num-gpus 1 \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --local-model-path "$MODEL_PATH" \
                --include-input-log --skip-server-setup --run-ids --allow-overwrite \
                --result-dir "$shard/official_result"
        ) > "$RUN_DIR/logs/decode_${decode}_cli_gpu_${physical_gpu}_shard_${shard_index}.log" 2>&1 &
        CLI_PIDS+=("$!")
    done
    failed=0
    for pid in "${CLI_PIDS[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    CLI_PIDS=()
    echo "$(date --iso-8601=seconds),decode_$decode,generation_end" >> "$RUN_DIR/events.csv"
    (( failed == 0 )) || { echo "decode $decode failed; inspect logs" >&2; exit 1; }
    stop_servers
    echo "$(date --iso-8601=seconds),decode_$decode,scoring_start" >> "$RUN_DIR/events.csv"
    for ((shard_index=0; shard_index<TOTAL_SHARDS; shard_index++)); do
        local_gpu=$((shard_index % NGPU))
        physical_gpu=${GPU_ARRAY[$local_gpu]}
        score_shard "$decode" "$shard_index" "$physical_gpu" "$local_gpu" \
            > "$RUN_DIR/logs/decode_${decode}_score_gpu_${physical_gpu}_shard_${shard_index}.log" 2>&1 &
        SCORE_PIDS+=("$!")
    done
done

score_failed=0
for pid in "${SCORE_PIDS[@]}"; do if ! wait "$pid"; then score_failed=1; fi; done
SCORE_PIDS=()
echo "$(date --iso-8601=seconds),all,scoring_end" >> "$RUN_DIR/events.csv"
(( score_failed == 0 )) || { echo "scoring failed; inspect logs/*_score_*.log" >&2; exit 1; }

"$BFCL_CLI_VENV/bin/python" "$ROOT/scripts/pool_summarize.py" \
    --run-dir "$RUN_DIR" --decodes "$DECODES" --expected-tasks "$TOTAL_TASKS" \
    --category "$CATEGORY" --model-path "$MODEL_PATH" \
    --temperature "$TEMPERATURE" --backend-python "$VLLM_PYTHON" \
    --per-request-max-tokens "$BFCL_MAX_OUTPUT_TOKENS" \
    --comparability "$COMPARABILITY" --bfcl-commit "$BFCL_COMMIT" \
    --shards-per-gpu "$SHARDS_PER_GPU" \
    "${BFCL_SOURCE_ARGS[@]}" \
    > "$RUN_DIR/summary.stdout.json"
echo "$RUN_DIR"
