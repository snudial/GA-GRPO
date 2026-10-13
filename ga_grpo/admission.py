"""Admission check for an (env, model) pair, run before any long run.

Catches in seconds the failure modes that otherwise surface hours into a run
(all four hit on 2026-10-03 when one AppWorld run swapped its model):
  1. history not append-only under the env's message pattern: molt then records
     every LLM call as its own training sample (measured 5.8 samples/episode)
  2. full-vocab fp32 logits too large for MAX_LEN: trainer OOM
  3. planned budget / group / length differ from the env contract
  4. (--gpu) attention falls back to an unfused kernel (measured MFU 3%)
Exit status 1 if any check FAILs.

    python -m ga_grpo.admission --env appworld --model MODEL_DIR [--budget N] [--max-len N] [--gpu]
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

from ga_grpo.model_profile import logits_gib, profile, runtime_defaults

ROOT = Path(__file__).resolve().parents[1]
# Largest full-vocab logits copy measured to train without OOM (Qwen3.5-9B, SDPA, 16k).
LOGITS_MEASURED_OK_GIB = 15.5


def molt_segments(tokenizer, calls):
    """Training segments molt records for one episode, given the message list
    sent at each LLM call. Mirrors molt/agents/_chat_server.py::_run_turn: a call
    continues the running segment only if it extends the previous request AND
    the template re-renders the earlier turns unchanged."""
    segments, last = 0, None
    for msgs in calls:
        extends = last is not None and msgs[:len(last)] == last
        if extends:
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prefix = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=False)
            extends = full.startswith(prefix)
        segments += not extends
        last = msgs
    return segments


def episode_calls(pattern, thinking, steps_per_query=3):
    """Synthetic episode in the env's message pattern; a thinking model's
    actions carry a <think> block, as it emits one every call."""
    msgs = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "Task 0"}]
    calls = []
    for q in range(pattern["queries_per_episode"]):
        if q:
            msgs = msgs + [{"role": pattern["new_query_role"], "content": f"Task {q}"}]
        for s in range(steps_per_query):
            calls.append(list(msgs))
            action = f"call_{q}_{s}()"
            content = f"<think>\nplan {q}.{s}\n</think>\n\n{action}" if thinking else action
            msgs = msgs + [{"role": "assistant", "content": content},
                           {"role": pattern["observation_role"], "content": f"result {q}.{s}"}]
    return calls


def probe_attention(prof, impl):
    """Run one tiny causal attention fwd+bwd on the resolved backend at the
    model's head shape; returns (ok, detail)."""
    os.environ.setdefault("NVTE_DEBUG", "1")
    os.environ.setdefault("NVTE_DEBUG_LEVEL", "2")
    import torch
    import torch.nn.functional as F
    H, KV, D, S = prof["num_attention_heads"], prof["num_key_value_heads"], prof["head_dim"], 512
    dev, dt = "cuda", torch.bfloat16
    if impl == "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        q = torch.randn(1, H, S, D, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(1, KV, S, D, device=dev, dtype=dt, requires_grad=True)
        v = torch.randn(1, KV, S, D, device=dev, dtype=dt, requires_grad=True)
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).float().sum().backward()
            return True, "SDPA flash kernel runs"
        except RuntimeError as e:
            return False, f"SDPA flash unavailable: {str(e)[:80]}"
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    import transformer_engine.pytorch as te
    attn = te.DotProductAttention(num_attention_heads=H, kv_channels=D, num_gqa_groups=KV,
                                  attn_mask_type="causal", qkv_format="bshd")
    q = torch.randn(1, S, H, D, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(1, S, KV, D, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(1, S, KV, D, device=dev, dtype=dt, requires_grad=True)
    attn(q, k, v).float().sum().backward()
    selected = [m.split("=", 1)[1].strip() for m in records if "Selected backend" in m]
    backend = selected[-1] if selected else "unknown"
    return "Unfused" not in backend and backend != "unknown", f"TE selected {backend}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=["bfcl", "appworld"])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--budget", type=int, help="planned total generated rollouts (default: contract)")
    ap.add_argument("--group-size", type=int)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--allow-protocol-change", metavar="REASON",
                    help="downgrade contract mismatches to WARN; the reason is recorded in the report")
    ap.add_argument("--gpu", action="store_true", help="also probe the attention backend on cuda:0")
    ap.add_argument("--report", type=Path, help="write the JSON report here")
    args = ap.parse_args()

    contract = json.loads((ROOT / "contracts" / f"{args.env}.json").read_text())
    prof = profile(args.model)
    defaults = runtime_defaults(prof)
    checks = []  # (status, name, detail)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    pattern = contract["message_pattern"]
    # A template that parses </think> but closes an empty think block in its own
    # generation prompt runs the model in non-thinking mode: no <think> is emitted.
    gen_prompt = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    prof["emits_thinking"] = prof["thinking_template"] and not gen_prompt.rstrip().endswith("</think>")
    segs = molt_segments(tok, episode_calls(pattern, prof["emits_thinking"]))
    queries = pattern["queries_per_episode"]
    checks.append(("PASS" if segs <= queries else "FAIL", "append-only history",
                   f"{segs} training segment(s) for {queries} query(ies) "
                   f"(observation role '{pattern['observation_role']}', emits_thinking={prof['emits_thinking']})"
                   + ("" if segs <= queries else " — every LLM call becomes its own sample; "
                      "send observations as tool responses or use a non-thinking/append-only template")))

    max_len = args.max_len or contract["max_len"]
    gib = logits_gib(prof, max_len)
    checks.append(("PASS" if gib <= LOGITS_MEASURED_OK_GIB else "WARN", "logits memory",
                   f"one fp32 [{max_len} x {prof['vocab_size']}] copy = {gib:.1f} GiB "
                   f"(largest measured to fit: {LOGITS_MEASURED_OK_GIB} GiB)"))

    if prof["linear_attention"]:
        has_fla = importlib.util.find_spec("fla") is not None
        checks.append(("PASS" if has_fla else "FAIL", "linear-attention kernels",
                       "fla installed" if has_fla else "fla missing: packed training silently corrupts and "
                       "the unpacked recurrence falls back to a torch loop"))
        if importlib.util.find_spec("causal_conv1d") is None:
            checks.append(("WARN", "linear-attention conv", "causal_conv1d missing: torch conv fallback (~1% of step, measured)"))

    for key, planned in (("total_generated_rollouts", args.budget), ("group_size", args.group_size),
                         ("max_len", args.max_len)):
        if planned is not None and planned != contract[key]:
            status = "WARN" if args.allow_protocol_change else "FAIL"
            checks.append((status, f"contract {key}", f"planned {planned} != contract {contract[key]}"
                           + (f" (allowed: {args.allow_protocol_change})" if args.allow_protocol_change else "")))

    if args.gpu:
        ok, detail = probe_attention(prof, defaults["ATTN_IMPLEMENTATION"])
        checks.append(("PASS" if ok else "FAIL", "attention backend", detail))

    print(f"admission: env={args.env} contract={contract['name']} model={args.model.name}")
    print(f"  profile  {json.dumps(prof)}")
    print(f"  defaults {json.dumps(defaults)}")
    for status, name, detail in checks:
        print(f"  [{status}] {name}: {detail}")
    failed = any(s == "FAIL" for s, _, _ in checks)
    if args.report:
        args.report.write_text(json.dumps(dict(env=args.env, contract=contract["name"], model=str(args.model),
                                               profile=prof, defaults=defaults, passed=not failed,
                                               allow_protocol_change=args.allow_protocol_change,
                                               checks=[dict(status=s, name=n, detail=d) for s, n, d in checks]),
                                          indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
