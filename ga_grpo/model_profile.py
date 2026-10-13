"""Runtime defaults derived from the model, shared by every launcher.

A variation changes `--model`; attention backend, packing and prefix caching
then follow from the model's own config, identically in BFCL (scripts/train.sh)
and AppWorld (scripts/run_appworld.py). Measured on 4x B200, TE 2.12, cuDNN 9.20
(2026-10-03): TE has a fused attention backend only for head_dim <= 128; above
that it falls back to unfused attention plus THD<->BSHD conversions (59% of
training GPU time on Qwen3.5-9B), while torch SDPA's flash kernel covers 256.

    python -m ga_grpo.model_profile MODEL_DIR            # JSON profile + defaults
    eval "$(python -m ga_grpo.model_profile MODEL_DIR --shell)"   # export defaults
"""
import json
import sys
from pathlib import Path

TE_FUSED_MAX_HEAD_DIM = 128


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


def logits_gib(p, max_len):
    """One fp32 [max_len, vocab] logits copy — the training backward keeps several."""
    return p["vocab_size"] * max_len * 4 / 2**30


if __name__ == "__main__":
    prof = profile(sys.argv[1])
    defaults = runtime_defaults(prof)
    if "--shell" in sys.argv:
        # Caller-set values win, so an explicit override stays visible in the launch command.
        print("\n".join(f'export {k}="${{{k}:-{v}}}"' for k, v in defaults.items()))
    else:
        print(json.dumps({"profile": prof, "defaults": defaults}, indent=2))
