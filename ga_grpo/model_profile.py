"""Runtime defaults derived from the model, shared by every launcher.

A variation changes `--model`; attention backend, packing and prefix caching
then follow from the model's own config, identically in BFCL (scripts/train.sh)
and AppWorld (scripts/run_appworld.py). Measured on 4x B200, TE 2.12, cuDNN 9.20
(2026-10-03): TE has a fused attention backend only for head_dim <= 128; above
that it falls back to unfused attention plus THD<->BSHD conversions (59% of
training GPU time on Qwen3.5-9B), while torch SDPA's flash kernel covers 256.

    python -m ga_grpo.model_profile MODEL_DIR            # JSON profile + defaults
    eval "$(python -m ga_grpo.model_profile MODEL_DIR --shell)"   # export defaults
    eval "$(python -m ga_grpo.model_profile MODEL_DIR --single-rank-shell GPU_GIB)"
                                                   # one-GPU memory split (see below)
"""
import json
import math
import sys
from pathlib import Path

TE_FUSED_MAX_HEAD_DIM = 128

# One-rank hybrid topology (config/topology/single_h100.env): the actor and one
# vLLM engine time-share a GPU. The engine sleeps (level 2, weights and KV freed)
# while the actor trains, but the actor stays resident while the engine
# generates, so the engine's share is what the actor leaves. On one rank the
# model is fp32 (no FSDP2 mixed precision; Molt autocasts the compute), so
# params and grads cost 8 bytes each; Adam's fp32 moments add 8 more when they
# stay on the GPU. The measured 48 GB training peak (bf16 compute, micro
# batch 1, 10k tokens) is logits-bound, not resident-bound, so it is not part
# of this split: vLLM is asleep then.
FP32_PARAM_GRAD_BYTES = 8
ADAM_BYTES = 8
# Adam stays on the GPU while params+grads+moments fit here (0.6B: 9 GiB,
# 1.7B: 26 GiB); beyond it the moments go to CPU (fused CPU AdamW), as the
# 4B (60 GiB) and 8B runs need.
GPU_OPTIMIZER_MAX_GIB = 32
# CUDA contexts of both processes, vLLM activations and CUDA graphs, allocator slack.
HEADROOM_GIB = 8
VLLM_UTIL_MIN, VLLM_UTIL_MAX, VLLM_UTIL_STEP = 0.30, 0.85, 0.05


def chat_template(model_dir):
    path = Path(model_dir) / "chat_template.jinja"
    if path.exists():
        return path.read_text()
    return json.loads((Path(model_dir) / "tokenizer_config.json").read_text()).get("chat_template") or ""


def profile(model_dir):
    config = json.loads((Path(model_dir) / "config.json").read_text())
    text = config.get("text_config", config)
    return {
        "head_dim": text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"],
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text.get("num_key_value_heads", text["num_attention_heads"]),
        "vocab_size": text["vocab_size"],
        "linear_attention": "linear_attention" in (text.get("layer_types") or []),
        # A template that parses </think> renders reasoning, so the model can emit it.
        "thinking_template": "</think>" in chat_template(model_dir),
    }


def runtime_defaults(p):
    fused = p["head_dim"] <= TE_FUSED_MAX_HEAD_DIM
    return {
        "ATTN_IMPLEMENTATION": "te" if fused else "sdpa",
        "PACKING": "1" if fused else "0",
        # Validated on dense attention (hit ~99%, vllm_kl unchanged); hybrid
        # linear-attention models keep it off until their state caching is.
        "VLLM_PREFIX_CACHING": "0" if p["linear_attention"] else "1",
        # TE Linear is BackendConfig's own default under TE attention; pinning
        # it keeps one checkpoint parameter layout across attention backends.
        "MOLT_LINEAR_BACKEND": "te",
    }


def param_count(model_dir):
    """Parameters of the checkpoint from its safetensors headers (shapes only, nothing
    loaded; derived views hardlink the same files). A tied lm_head stored on disk
    (Qwen3-0.6B ships one) is not a second tensor in memory. Without weights, the
    dense-transformer formula."""
    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text())
    text = config.get("text_config", config)
    tied = bool(text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    total = 0
    for file in sorted(d.glob("*.safetensors")):
        with file.open("rb") as f:
            header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
        for name, tensor in header.items():
            if name == "__metadata__" or (tied and name == "lm_head.weight"):
                continue
            total += math.prod(tensor["shape"])
    if total:
        return total
    hidden, layers, vocab = text["hidden_size"], text["num_hidden_layers"], text["vocab_size"]
    heads = text["num_attention_heads"]
    kv = text.get("num_key_value_heads", heads)
    head_dim = text.get("head_dim") or hidden // heads
    attention = hidden * head_dim * (2 * heads + 2 * kv)
    mlp = 3 * hidden * text["intermediate_size"]
    embeddings = vocab * hidden * (1 if tied else 2)
    return embeddings + layers * (attention + mlp)


def single_rank_defaults(params, gpu_gib):
    """Memory split for one GPU shared by the actor and one vLLM engine (module docstring)."""
    adam_on_gpu = params * (FP32_PARAM_GRAD_BYTES + ADAM_BYTES) / 2**30 <= GPU_OPTIMIZER_MAX_GIB
    resident_gib = params * (FP32_PARAM_GRAD_BYTES + (ADAM_BYTES if adam_on_gpu else 0)) / 2**30
    util = (gpu_gib - HEADROOM_GIB - resident_gib) / gpu_gib
    util = math.floor(util / VLLM_UTIL_STEP + 1e-9) * VLLM_UTIL_STEP
    util = min(VLLM_UTIL_MAX, max(VLLM_UTIL_MIN, util))
    return {
        "FSDP_OFFLOAD": "none" if adam_on_gpu else "optimizer",
        "VLLM_UTIL": f"{util:.2f}",
        # Recorded for the launch log; not read by any launcher.
        "ACTOR_RESIDENT_GIB": f"{resident_gib:.1f}",
    }


def logits_gib(p, max_len):
    """One fp32 [max_len, vocab] logits copy — the training backward keeps several."""
    return p["vocab_size"] * max_len * 4 / 2**30


def shell(defaults):
    # Caller-set values win, so an explicit override stays visible in the launch command.
    return "\n".join(f'export {k}="${{{k}:-{v}}}"' for k, v in defaults.items())


if __name__ == "__main__":
    if "--single-rank-shell" in sys.argv:   # needs only config.json and the weights
        gpu_gib = float(sys.argv[sys.argv.index("--single-rank-shell") + 1])
        print(shell(single_rank_defaults(param_count(sys.argv[1]), gpu_gib)))
        sys.exit()
    prof = profile(sys.argv[1])
    defaults = runtime_defaults(prof)
    if "--shell" in sys.argv:
        print(shell(defaults))
    else:
        params = param_count(sys.argv[1])
        print(json.dumps({"profile": dict(prof, params=params), "defaults": defaults,
                          "single_rank_80gb": single_rank_defaults(params, 79.6)}, indent=2))
