"""Qwen3-2507 prompt compatibility for the pinned BFCL V3 runtime.

BFCL v1.3 predates Qwen3-4B-Instruct-2507.  Its Qwen FC parser stores prior
calls in ``message['tool_calls']``, but its prompt formatter never serializes
that field, so later turns see tool results without the actions that produced
them.  Patch only that formatting omission; data loading, simulation, and
evaluation remain the unmodified BFCL V3 implementation.

Python imports ``sitecustomize`` during interpreter startup when this directory
is on PYTHONPATH, including for the ``bfcl`` console script.
"""

from __future__ import annotations

import copy
import json


def _serialize_call(call: dict) -> str:
    call = call.get("function", call)
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        rendered_arguments = arguments
    else:
        rendered_arguments = json.dumps(arguments)
    return (
        '<tool_call>\n{"name": "'
        + str(call["name"])
        + '", "arguments": '
        + rendered_arguments
        + "}\n</tool_call>"
    )


def _install() -> None:
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    if VERSION_PREFIX != "BFCL_v3":
        raise RuntimeError(
            "bfcl_v3_qwen2507 compatibility loaded against " + VERSION_PREFIX
        )
    original = QwenFCHandler._format_prompt
    if getattr(original, "_qwen2507_v3_compat", False):
        return

    def format_prompt_with_tool_history(self, messages, function):
        normalized = copy.deepcopy(messages)
        for message in normalized:
            calls = message.pop("tool_calls", None)
            if message.get("role") != "assistant" or not calls:
                continue
            rendered = "\n".join(_serialize_call(call) for call in calls)
            content = message.get("content") or ""
            message["content"] = content + (("\n" if content else "") + rendered)
        return original(self, normalized, function)

    format_prompt_with_tool_history._qwen2507_v3_compat = True
    QwenFCHandler._format_prompt = format_prompt_with_tool_history


_install()
