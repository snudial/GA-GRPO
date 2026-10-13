"""Blocking byte/token-equivalence gates for the pinned BFCL Qwen protocol."""

from __future__ import annotations

import json
import importlib.util
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from bfcl_env.adapter import (
    available_tool_documents,
    load_task,
    load_tool_documents,
)
from bfcl_env.runtime import (
    _native_fc_system_prompt,
    _parse_native_fc,
)
from generation_backend import TransformersGenerationBackend

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_imported_bfcl_uses_configured_checkout():
    configured = os.environ.get("BFCL_SOURCE_ROOT")
    if not configured:
        pytest.skip("BFCL_SOURCE_ROOT is not configured")
    import bfcl_eval

    expected = (Path(configured) / "bfcl_eval").resolve()
    actual = Path(bfcl_eval.__file__).resolve()
    assert actual.is_relative_to(expected), (actual, expected)


def _dependencies():
    checkpoint = os.environ.get("QWEN3_4B_INSTRUCT_2507_PATH")
    if not checkpoint or not os.path.isfile(os.path.join(checkpoint, "tokenizer_config.json")):
        pytest.skip("QWEN3_4B_INSTRUCT_2507_PATH tokenizer is unavailable")
    transformers = pytest.importorskip("transformers")
    try:
        from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"pinned BFCL QwenFCHandler is unavailable: {error}")
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    if VERSION_PREFIX == "BFCL_v3":
        compat = REPO_ROOT / "compat" / "bfcl_v3_qwen2507" / "sitecustomize.py"
        spec = importlib.util.spec_from_file_location("_bfcl_v3_qwen2507_compat", compat)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return transformers.AutoTokenizer.from_pretrained(checkpoint), QwenFCHandler


def _user(task, turn):
    return "\n".join(message.get("content", "") for message in task.user_messages(turn))


def _raw_call(name="search", arguments=None):
    return "<tool_call>\n" + json.dumps(
        {"name": name, "arguments": arguments or {}}
    ) + "\n</tool_call>"


def _assert_equivalent(tokenizer, handler_class, messages, functions):
    handler = object.__new__(handler_class)
    canonical = handler._format_prompt(messages, functions)

    # The author-lineage handler stores parsed calls in ``tool_calls`` while our
    # generic manager stores the equivalent raw assistant generation.
    mirror_messages = []
    for source in messages:
        message = dict(source)
        calls = message.pop("tool_calls", None)
        if message.get("role") == "assistant" and calls:
            message["content"] = "\n".join(
                _raw_call(call["name"], call.get("arguments") or {}) for call in calls
            )
        mirror_messages.append(message)

    backend = object.__new__(TransformersGenerationBackend)
    backend.tokenizer = tokenizer
    backend.empty_tool_call_assistant_history = False
    backend.allow_thinking = False
    backend._checked = False
    mirror = backend._render([
        {"role": "system", "content": _native_fc_system_prompt(functions)},
        *mirror_messages,
    ])

    assert mirror.encode("utf-8") == canonical.encode("utf-8")
    assert tokenizer.encode(mirror) == tokenizer.encode(canonical)
    return mirror


def test_function_documents_use_official_preprocessing_and_order():
    for category in (
        "multi_turn_base", "multi_turn_miss_func",
        "multi_turn_miss_param", "multi_turn_long_context",
    ):
        task = load_task(f"{category}_0", category)
        documents = load_tool_documents(task.involved_classes, category=category)
        assert documents
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX
        if VERSION_PREFIX == "BFCL_v3":
            assert all(
                document["description"].endswith(
                    " Note that the provided function is in Python 3 syntax."
                )
                for document in documents.values()
            )


def test_qwen_prompt_is_byte_and_token_equal_for_all_required_history_shapes():
    tokenizer, handler_class = _dependencies()
    task = load_task("multi_turn_base_0", "multi_turn_base")
    documents = load_tool_documents(task.involved_classes, category=task.category)
    functions = list(available_tool_documents(task, 0, documents).values())
    first = [{"role": "user", "content": _user(task, 0)}]

    # First turn.
    _assert_equivalent(tokenizer, handler_class, first, functions)

    # Sequential call/result. The author-lineage formatter retains the call.
    sequential = first + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "cd", "arguments": {"folder": "document"}}
        ]},
        {"role": "tool", "content": '{"current_directory": "document"}'},
    ]
    _assert_equivalent(tokenizer, handler_class, sequential, functions)

    # Parallel results remain two messages and therefore two response envelopes.
    parallel = first + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "cd", "arguments": {"folder": "document"}},
            {"name": "sort", "arguments": {"file_name": "final_report.pdf"}},
        ]},
        {"role": "tool", "content": "first-result"},
        {"role": "tool", "content": "second-result"},
    ]
    rendered = _assert_equivalent(tokenizer, handler_class, parallel, functions)
    assert rendered.count("<tool_response>") == 2

    # Empty assistant response closes a turn before the next user turn.
    later = parallel + [
        {"role": "assistant", "content": ""},
        {"role": "user", "content": _user(task, 1)},
    ]
    _assert_equivalent(tokenizer, handler_class, later, functions)


@pytest.mark.parametrize("category", ["multi_turn_miss_param", "multi_turn_long_context"])
def test_qwen_prompt_is_equal_for_special_categories(category):
    tokenizer, handler_class = _dependencies()
    task = load_task(f"{category}_0", category)
    documents = load_tool_documents(task.involved_classes, category=category)
    functions = list(available_tool_documents(task, 0, documents).values())
    _assert_equivalent(
        tokenizer, handler_class,
        [{"role": "user", "content": _user(task, 0)}], functions,
    )


def test_missing_function_reveal_updates_tools_with_official_message():
    tokenizer, handler_class = _dependencies()
    from bfcl_eval.constants.default_prompts import (
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    )

    task = load_task("multi_turn_miss_func_0", "multi_turn_miss_func")
    documents = load_tool_documents(task.involved_classes, category=task.category)
    reveal_turn = next(iter(task.missed_function))
    before = list(available_tool_documents(task, reveal_turn - 1, documents).values())
    after = list(available_tool_documents(task, reveal_turn, documents).values())
    assert len(after) > len(before)

    messages = [{"role": "user", "content": _user(task, 0)}]
    for turn in range(1, reveal_turn):
        messages.extend([
            {"role": "assistant", "content": ""},
            {"role": "user", "content": _user(task, turn)},
        ])
    messages.extend([
        {"role": "assistant", "content": ""},
        {"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC},
    ])
    _assert_equivalent(tokenizer, handler_class, messages, after)


def test_native_extraction_is_exactly_the_pinned_handler():
    _, handler_class = _dependencies()
    handler = object.__new__(handler_class)
    raw = _raw_call("x", {"a": True, "b": None, "c": "q"}) + "\n" + _raw_call(
        "y", {"values": [1, 2]}
    )
    import inspect
    if "has_tool_call_tag" in inspect.signature(type(handler).decode_execute).parameters:
        expected = handler.decode_execute(raw, False)
    else:
        expected = handler.decode_execute(raw)
    actual, error = _parse_native_fc(raw)
    assert error is None
    assert actual == expected

    # The official regex is deliberately newline-sensitive.
    malformed = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
    if "has_tool_call_tag" in inspect.signature(type(handler).decode_execute).parameters:
        malformed_expected = handler.decode_execute(malformed, False)
    else:
        malformed_expected = handler.decode_execute(malformed)
    assert _parse_native_fc(malformed) == (malformed_expected, None)


def test_official_pool_propagates_checkpoint_override_everywhere():
    source = (REPO_ROOT / "scripts" / "official_pool.sh").read_text(
        encoding="utf-8"
    )
    assert 'MODEL_PATH="${MODEL_PATH:-$QWEN3_4B_INSTRUCT_2507_PATH}"' in source
    assert source.count('"$QWEN3_4B_INSTRUCT_2507_PATH"') == 0
    assert 'serve "$MODEL_PATH"' in source
    assert '--local-model-path "$MODEL_PATH"' in source
    assert source.count('--model-path "$MODEL_PATH"') == 2


def test_official_pool_rejects_accidental_non_fc_registry():
    source = (REPO_ROOT / "scripts" / "official_pool.sh").read_text(
        encoding="utf-8"
    )
    assert 'MODEL_REGISTRY_NAME="${MODEL_REGISTRY_NAME:-$BFCL_MODEL_REGISTRY_NAME}"' in source
    assert 'ALLOW_NON_FC_HANDLER="${ALLOW_NON_FC_HANDLER:-0}"' in source
    assert 'MODEL_REGISTRY_NAME" != *-FC' in source
    assert 'ALLOW_NON_FC_HANDLER" != "1"' in source


def test_official_eval_requires_explicit_result_reuse():
    source = (REPO_ROOT / "scripts" / "eval_official.sh").read_text(
        encoding="utf-8"
    )
    assert '"${REUSE_EVAL:-0}" = "1"' in source
    assert "use a new output directory or set REUSE_EVAL=1" in source


@pytest.mark.parametrize("category", [
    "multi_turn_base", "multi_turn_miss_func",
    "multi_turn_miss_param", "multi_turn_long_context",
])
def test_fresh_process_environment_matches_ground_truth_on_all_categories(category):
    """Gate 3 smoke: calls, result order, state, boundaries, and verdict."""
    from bfcl_env.runtime import BFCLRuntime, _state_hash
    from bfcl_env.environment_service import LocalEnvironmentBackend
    from bfcl_env.rollout_protocol import StructuredAction
    from bfcl_env.schemas import ToolCall

    task = load_task(f"{category}_0", category)
    completed = subprocess.run(
        [sys.executable, "-m", "bfcl_env.bfcl_process_worker", "ground_truth"],
        input=pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    trace = pickle.loads(completed.stdout)

    runtime = BFCLRuntime(
        category=category, runner_mode="recovery_continue",
        tool_mode="native_fc", isolate_process_state=True,
    )
    backend = LocalEnvironmentBackend(runtime, max_sessions=1)
    info = backend.create_session(task.task_id, 0, f"gate3-{category}")
    sequence = 0
    final = None
    try:
        for turn in range(task.num_turns):
            calls = list(task.ground_truth_calls(turn))
            if calls:
                action = StructuredAction(
                    raw_text="\n".join(calls),
                    tool_calls=[
                        ToolCall(tool=call.split("(", 1)[0], raw_text=call)
                        for call in calls
                    ],
                    parse_ok=True, parse_kind="TOOL_CALL",
                )
                result = backend.step(info.session_id, sequence, action)
                assert [
                    message["content"] for message in result.observation_messages
                    if message["role"] == "tool"
                ] == trace.turns[turn].responses
                assert result.state_after_hash == _state_hash(trace.turns[turn].state)
                assert result.turn == turn
                sequence += 1
            final = backend.step(
                info.session_id, sequence,
                StructuredAction(
                    raw_text="", parse_ok=True, parse_kind="END_TURN"
                ),
            )
            assert final.turn == turn
            assert final.state_after_hash == _state_hash(trace.turns[turn].state)
            sequence += 1
        assert final is not None and final.terminated
    finally:
        backend.close_session(info.session_id)

    score_payload = {
        "model_result": [
            ([list(task.ground_truth_calls(turn))] if task.ground_truth_calls(turn) else [])
            for turn in range(task.num_turns)
        ],
        "reference": [list(task.ground_truth_calls(turn)) for turn in range(task.num_turns)],
        "entry": {
            "id": task.task_id, "initial_config": task.initial_config,
            "involved_classes": task.involved_classes,
        },
        "category": category,
    }
    scored = subprocess.run(
        [sys.executable, "-m", "bfcl_env.bfcl_process_worker", "official"],
        input=pickle.dumps(score_payload, protocol=pickle.HIGHEST_PROTOCOL),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert scored.returncode == 0, scored.stderr.decode(errors="replace")
    assert pickle.loads(scored.stdout)["valid"] is True


@pytest.mark.parametrize("variant", ["reference", "empty", "omit_first", "bad_extra"])
def test_online_reward_agrees_with_official_checker(variant):
    from bfcl_env.rollout_protocol import StructuredAction, ToolCall
    from bfcl_env.runtime import BFCLRuntime

    task = load_task("multi_turn_base_0", "multi_turn_base")
    turns = [list(task.ground_truth_calls(turn)) for turn in range(task.num_turns)]
    if variant == "empty":
        turns = [[] for _ in turns]
    elif variant == "omit_first":
        turns[0] = turns[0][1:]
    elif variant == "bad_extra":
        turns[0].append("mkdir(dir_name='AGREEMENT_BAD')")

    runtime = BFCLRuntime(
        category=task.category, runner_mode="official_stop", tool_mode="native_fc"
    )
    reset = runtime.reset(task.task_id, 0)
    try:
        for calls in turns:
            if calls:
                step = runtime.execute(
                    reset.handle,
                    StructuredAction(
                        raw_text="\n".join(calls),
                        tool_calls=[
                            ToolCall(tool=call.split("(", 1)[0], raw_text=call)
                            for call in calls
                        ],
                        parse_ok=True,
                        parse_kind="TOOL_CALL",
                    ),
                )
                if step.terminated:
                    break
            step = runtime.execute(
                reset.handle,
                StructuredAction(raw_text="", parse_ok=True, parse_kind="END_TURN"),
            )
            if step.terminated:
                break
        online_valid = not reset.handle.official_stop_triggered
    finally:
        runtime.close(reset.handle)

    payload = {
        "model_result": [([calls] if calls else []) for calls in turns],
        "reference": [list(task.ground_truth_calls(turn)) for turn in range(task.num_turns)],
        "entry": {
            "id": task.task_id,
            "initial_config": task.initial_config,
            "involved_classes": task.involved_classes,
        },
        "category": task.category,
    }
    scored = subprocess.run(
        [sys.executable, "-m", "bfcl_env.bfcl_process_worker", "official"],
        input=pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert scored.returncode == 0, scored.stderr.decode(errors="replace")
    assert online_valid is bool(pickle.loads(scored.stdout)["valid"])


@pytest.fixture(autouse=True)
def require_benchmark_for_environment_tests(request):
    names = {'test_function_documents_use_official_preprocessing_and_order',
             'test_fresh_process_environment_matches_ground_truth_on_all_categories',
             'test_online_reward_agrees_with_official_checker'}
    if request.function.__name__ in names:
        from conftest import _bfcl_available
        if not _bfcl_available():
            pytest.skip('Configure the pinned BFCL checkout for benchmark integration tests')
