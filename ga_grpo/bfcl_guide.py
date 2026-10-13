"""Reference calls serialized exactly as in student rollout context."""
import ast
import json

from bfcl_env.rollout_protocol import StructuredAction, ToolCall


def _call_payload(call: str, documents: dict) -> dict:
    """Convert one BFCL reference Python call to Qwen native-FC JSON."""
    node = ast.parse(call, mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"reference is not a call: {call}")
    parts = []
    target = node.func
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    name = ".".join(reversed(parts))
    document = documents.get(name, {})
    properties = list(document.get("parameters", {}).get("properties", {}))
    arguments = {}
    for index, value in enumerate(node.args):
        if index >= len(properties):
            raise ValueError(f"no parameter name for positional argument in {call}")
        arguments[properties[index]] = ast.literal_eval(value)
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError(f"**kwargs reference is unsupported: {call}")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return {"name": name, "arguments": arguments}


def _guide_action(calls: list[str], documents: dict) -> tuple[str, StructuredAction]:
    raw = "\n".join(
        "<tool_call>\n" + json.dumps(_call_payload(call, documents))
        + "\n</tool_call>"
        for call in calls
    )
    action = StructuredAction(
        raw_text=raw,
        tool_calls=[
            ToolCall(tool=call.split("(", 1)[0], raw_text=call) for call in calls
        ],
        parse_ok=True,
        parse_kind="TOOL_CALL",
    )
    return raw, action
