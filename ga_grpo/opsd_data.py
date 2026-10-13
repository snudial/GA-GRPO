"""Build paired (student, teacher) token sequences for boundary OPSD on BFCL and
AppWorld (records with env="appworld"; the rest is identical).

The distilled quantity is  KL( pi_EMA(. | x, z_priv, a_<t)  ||  pi_theta(. | x, a_<t) )
over the tokens of the boundary action the student actually took — the first
action of the officially-failed turn. Teacher and student see different
CONTEXTS but the exact same ACTION TOKENS, so alignment is by construction:
target token i sits at logit position ctx_len - 1 + i in both sequences.

Privilege boundary: the teacher message carries the runtime checker's
CERTIFICATE — mismatched state FIELD NAMES and the student's own failed
action — never ground-truth values (see banks.py).

Component weights follow the three decision kinds of a BFCL action: the tool
name and its arguments inside <tool_call> JSON, and the end-turn decision (a
response with no tool call). Char spans map to tokens via the fast tokenizer's
offset mapping; when nothing matches, the whole span trains at the base weight.
"""
import os
import re

import torch

W_BASE = 1.0
W_TOOL = float(os.environ.get("EXP2_OPSD_W_TOOL", "2.0"))
W_ARG = float(os.environ.get("EXP2_OPSD_W_ARG", "1.5"))
W_TERM = float(os.environ.get("EXP2_OPSD_W_TERM", "2.0"))

NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
ARGS_RE = re.compile(r'"arguments"\s*:\s*\{')

PRIV_TMPL = """\

--- PRIVILEGED HINDSIGHT (training-time only; the acting agent never sees this) ---
A completed attempt continued from exactly this state and FAILED the turn check.
Mismatched state fields after the turn:
{fields}
The attempted next action was:
{attempted}
With this hindsight, respond with the corrected next action."""

# AppWorld: the verifier's failed requirement sentences (never the test traces),
# and the decisions of a code action: which `apis.<app>.<fn>` it calls, with what
# arguments, and whether it ends the task.
APPWORLD_PRIV_TMPL = """\
--- PRIVILEGED HINDSIGHT (training-time only; the acting agent never sees this) ---
A completed attempt continued from exactly this state and FAILED verification.
Failed requirements:
{failed}
The attempted next action was:
{attempted}
With this hindsight, respond with the corrected next action as a single python code block."""
APP_CALL_RE = re.compile(r"apis\.\w+\.\w+")


def span_weights(action_text, offsets):
    """Per-token weights over the tokenized action text."""
    spans = []  # (start, end, weight)
    for m in NAME_RE.finditer(action_text):
        spans.append((m.start(1), m.end(1), W_TOOL))
    for m in ARGS_RE.finditer(action_text):
        depth, i = 0, m.end() - 1
        while i < len(action_text):
            if action_text[i] == "{":
                depth += 1
            elif action_text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        spans.append((m.end(), i, W_ARG))
    if "<tool_call>" not in action_text:
        # A call-free response ends the turn — the termination decision.
        spans.append((0, len(action_text), W_TERM))
    return _token_weights(spans, offsets)


def appworld_span_weights(action_text, offsets):
    """Per-token weights over an AppWorld code action."""
    spans = []  # (start, end, weight)
    for m in APP_CALL_RE.finditer(action_text):
        spans.append((m.start(), m.end(), W_TERM if 'complete_task' in m.group(0) else W_TOOL))
        if m.end() < len(action_text) and action_text[m.end()] == '(':
            depth, i = 0, m.end()
            while i < len(action_text):
                depth += {'(': 1, ')': -1}.get(action_text[i], 0)
                if depth == 0:
                    break
                i += 1
            spans.append((m.end() + 1, i, W_ARG))
    return _token_weights(spans, offsets)


def _token_weights(spans, offsets):
    weights = []
    for (a, b) in offsets:
        w = W_BASE
        for (s, e, sw) in spans:
            if a < e and b > s:  # token overlaps span
                w = max(w, sw)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def _merge_user(msgs, text):
    """Append text as user content, merging into a trailing user turn so chat
    templates never see two consecutive user messages."""
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n" + text}
    else:
        msgs.append({"role": "user", "content": text})
    return msgs


def _ctx_ids(tokenizer, msgs):
    """apply_chat_template(tokenize=True) returns a plain id list on some
    transformers versions and a BatchEncoding on others — normalize."""
    out = tokenizer.apply_chat_template(msgs, tokenize=True,
                                        add_generation_prompt=True)
    if hasattr(out, "input_ids"):
        out = out.input_ids
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def build_pairs(records, tokenizer, max_len):
    # Multimodal processors treat a bare string as an image source. OPSD only
    # encodes text, so use the processor's underlying tokenizer when present.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pairs = []
    for r in records:
        action, context = r.get("boundary_action"), r.get("context_messages")
        if not action or not context:
            continue
        enc = tokenizer(action, add_special_tokens=False,
                        return_offsets_mapping=True)
        target_ids = enc["input_ids"]
        if not target_ids:
            continue
        appworld = r.get("env") == "appworld"
        weights = (appworld_span_weights if appworld else span_weights)(action, enc["offset_mapping"])

        s_ctx = _ctx_ids(tokenizer, list(context))

        cert = r.get("certificate") or {}
        if appworld:
            failed = "\n".join(f"- {q}" for q in cert.get("failed_requirements", [])) \
                or "- (verification failed without requirement details)"
            note = APPWORLD_PRIV_TMPL.format(failed=failed, attempted=action)
        else:
            fields = "\n".join(f"- {f}" for f in cert.get("error_fields", [])) \
                or "- (turn check failed without field details)"
            note = PRIV_TMPL.format(fields=fields, attempted=action)
        t_msgs = _merge_user([dict(m) for m in context], note)
        t_ctx = _ctx_ids(tokenizer, t_msgs)

        if max(len(s_ctx), len(t_ctx)) + len(target_ids) > max_len:
            continue
        pairs.append({
            "student_ids": torch.tensor(s_ctx + target_ids, dtype=torch.long),
            "student_ctx_len": len(s_ctx),
            "teacher_ids": torch.tensor(t_ctx + target_ids, dtype=torch.long),
            "teacher_ctx_len": len(t_ctx),
            "target_len": len(target_ids),
            "weights": weights,
            "task_id": r.get("task_id"),
        })
    return pairs
