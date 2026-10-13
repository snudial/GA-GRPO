"""Plan section 7 -- BFCL v3 loading, comparison and action-error classification.

The pure-data tests run from fixtures with no ``bfcl_eval`` installed. The
real-data tests skip unless the benchmark is importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bfcl_env.adapter import (
    ACTION_RUBRICS,
    BFCL_CATEGORIES,
    EXECUTION_ERROR_PREFIX,
    BFCLTask,
    GroundTruthTrace,
    TurnSnapshot,
    available_tool_documents,
    classify_action_errors,
    compare_states,
    compile_rubrics,
    load_task_report,
    load_tasks,
    no_tool_call_rubric_id,
    response_rubric_id,
    responses_contain,
    snapshot_instances,
    state_rubric_id,
)
from bfcl_env.schemas import RubricKind, RubricState

pytestmark = []


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def make_task(**overrides) -> BFCLTask:
    base = dict(
        task_id="multi_turn_base_0",
        category="multi_turn_base",
        question=[[{"role": "user", "content": "do a"}], [{"role": "user", "content": "do b"}]],
        initial_config={"GorillaFileSystem": {"root": {}}},
        involved_classes=["GorillaFileSystem"],
        ground_truth=[["cd(folder='document')"], ["mkdir(dir_name='temp')"]],
    )
    base.update(overrides)
    return BFCLTask(**base)


def make_trace() -> GroundTruthTrace:
    return GroundTruthTrace(
        task_id="multi_turn_base_0",
        initial_state={"GorillaFileSystem": {"root": "r0", "long_context": False}},
        turns=[
            TurnSnapshot(0, {"GorillaFileSystem": {"root": "r1", "long_context": False}},
                         ["resp-a"], ["cd(folder='document')"]),
            TurnSnapshot(1, {"GorillaFileSystem": {"root": "r2", "long_context": False}},
                         [], []),
        ],
    )


TOOLS = {
    "cd": {"name": "cd", "parameters": {"required": ["folder"], "properties": {}}},
    "mkdir": {"name": "mkdir", "parameters": {"required": ["dir_name"], "properties": {}}},
    "ls": {"name": "ls", "parameters": {"required": [], "properties": {}}},
}


# --------------------------------------------------------------------------- #
# task model
# --------------------------------------------------------------------------- #


def test_from_entries_joins_prompt_and_answer():
    task = BFCLTask.from_entries(
        {"id": "multi_turn_base_0", "question": [[{"role": "user", "content": "x"}]],
         "initial_config": {"A": {}}, "involved_classes": ["A"]},
        {"id": "multi_turn_base_0", "ground_truth": [["f()"]]},
        "multi_turn_base",
    )
    assert task.num_turns == 1
    assert task.ground_truth_calls(0) == ["f()"]
    assert task.missed_function == {}


def test_id_mismatch_is_rejected():
    with pytest.raises(ValueError, match="id mismatch"):
        BFCLTask.from_entries(
            {"id": "a", "question": [], "initial_config": {}, "involved_classes": []},
            {"id": "b", "ground_truth": []}, "multi_turn_base",
        )


def test_missed_function_keys_become_integers():
    task = BFCLTask.from_entries(
        {"id": "multi_turn_miss_func_0", "question": [[], []], "initial_config": {},
         "involved_classes": [], "missed_function": {"1": ["sort", "cat"]}},
        {"id": "multi_turn_miss_func_0", "ground_truth": [[], ["sort()"]]},
        "multi_turn_miss_func",
    )
    assert task.missed_function == {1: ["sort", "cat"]}
    assert task.is_holdout_turn(1) and not task.is_holdout_turn(0)
    assert task.revealed_functions(0) == []
    assert task.revealed_functions(1) == ["sort", "cat"]


def test_held_out_tools_are_hidden_until_their_turn():
    task = make_task(missed_function={1: ["mkdir"]})
    assert "mkdir" not in available_tool_documents(task, 0, TOOLS)
    assert "mkdir" in available_tool_documents(task, 1, TOOLS)
    assert "cd" in available_tool_documents(task, 0, TOOLS)


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError, match="not one of the four"):
        load_tasks("single_turn_simple")


# --------------------------------------------------------------------------- #
# state comparison
# --------------------------------------------------------------------------- #


def test_equal_state_passes_every_attribute():
    verdicts = compare_states({"A": {"x": 1, "y": [1, 2]}}, {"A": {"x": 1, "y": [1, 2]}})
    assert verdicts == {("A", "x"): True, ("A", "y"): True}


def test_differing_attribute_is_reported_alone():
    verdicts = compare_states({"A": {"x": 1, "y": 9}}, {"A": {"x": 1, "y": 2}})
    assert verdicts[("A", "x")] is True
    assert verdicts[("A", "y")] is False


def test_missing_attribute_counts_as_a_mismatch():
    assert compare_states({"A": {}}, {"A": {"x": 1}}) == {("A", "x"): False}
    assert compare_states({}, {"A": {"x": 1}}) == {("A", "x"): False}


def test_extra_model_attributes_are_ignored_like_the_official_checker():
    """_compare_instances iterates the ground truth's attributes only."""
    verdicts = compare_states({"A": {"x": 1, "extra": 5}}, {"A": {"x": 1}})
    assert verdicts == {("A", "x"): True}


def test_a_comparison_that_raises_is_a_mismatch_not_a_crash():
    class Explodes:
        def __ne__(self, other):
            raise RuntimeError("boom")

    verdicts = compare_states({"A": {"x": Explodes()}}, {"A": {"x": Explodes()}})
    assert verdicts == {("A", "x"): False}


def test_snapshot_drops_private_attributes_and_deep_copies():
    class Instance:
        def __init__(self):
            self.public = {"a": 1}
            self._private = "hidden"

    instance = Instance()
    snapshot = snapshot_instances({"A": instance})
    assert snapshot == {"A": {"public": {"a": 1}}}

    instance.public["a"] = 99
    assert snapshot["A"]["public"] == {"a": 1}, "snapshot must not alias the live world"


# --------------------------------------------------------------------------- #
# response comparison
# --------------------------------------------------------------------------- #


def test_responses_are_matched_unordered():
    ok, missing = responses_contain(["c", "a", "b"], ["a", "b"])
    assert ok and missing == []


def test_duplicate_requirements_need_duplicate_responses():
    ok, missing = responses_contain(["a"], ["a", "a"])
    assert not ok and missing == ["a"]
    ok, _ = responses_contain(["a", "a"], ["a", "a"])
    assert ok


def test_missing_responses_are_listed():
    ok, missing = responses_contain(["a"], ["a", "z"])
    assert not ok and missing == ["z"]


def test_accumulated_responses_from_earlier_turns_count():
    """The official checker matches against all turns so far, not just this one."""
    ok, _ = responses_contain(["from_turn0", "from_turn1"], ["from_turn0"])
    assert ok


# --------------------------------------------------------------------------- #
# action-level errors
# --------------------------------------------------------------------------- #


def test_clean_call_violates_nothing():
    verdicts = classify_action_errors(
        parse_ok=True, decoded_calls=["cd(folder='x')"],
        available_tools=TOOLS, execution_results=['{"ok": true}'],
    )
    assert all(v.passed is True for v in verdicts.values())
    assert set(verdicts) == set(ACTION_RUBRICS)


def test_parse_failure_violates_only_the_parse_rubric():
    verdicts = classify_action_errors(
        parse_ok=False, decoded_calls=[], available_tools=TOOLS, execution_results=[],
    )
    assert verdicts["action:no_parse_error"].passed is False
    for rubric_id in ("action:no_unknown_tool", "action:no_invalid_arguments",
                      "action:no_execution_error"):
        assert verdicts[rubric_id].passed is None, "nothing ran, so there is no evidence"


def test_unknown_tool_is_flagged():
    verdicts = classify_action_errors(
        parse_ok=True, decoded_calls=["teleport(x=1)"],
        available_tools=TOOLS, execution_results=["whatever"],
    )
    assert verdicts["action:no_unknown_tool"].passed is False
    assert "teleport" in verdicts["action:no_unknown_tool"].trace


def test_missing_required_argument_is_flagged():
    verdicts = classify_action_errors(
        parse_ok=True, decoded_calls=["cd()"],
        available_tools=TOOLS, execution_results=["x"],
    )
    assert verdicts["action:no_invalid_arguments"].passed is False
    assert "folder" in verdicts["action:no_invalid_arguments"].trace


def test_positional_arguments_surface_as_missing_required_arguments():
    """The official decoder silently drops positional args, leaving cd()."""
    verdicts = classify_action_errors(
        parse_ok=True, decoded_calls=["cd()"], available_tools=TOOLS,
        execution_results=[f"{EXECUTION_ERROR_PREFIX} missing argument"],
    )
    assert verdicts["action:no_invalid_arguments"].passed is False
    assert verdicts["action:no_execution_error"].passed is False


def test_simulator_error_payloads_are_execution_failures():
    """The simulators mostly return {"error": ...} instead of raising."""
    from bfcl_env.adapter import is_execution_failure

    assert is_execution_failure('{"error": "cd: nope: No such directory."}')
    assert is_execution_failure(f"{EXECUTION_ERROR_PREFIX} boom")
    assert not is_execution_failure('{"current_working_directory": "document"}')
    assert not is_execution_failure("None")
    assert not is_execution_failure("")
    assert not is_execution_failure("[]")


def test_returned_error_payload_violates_the_execution_rubric():
    verdicts = classify_action_errors(
        parse_ok=True, decoded_calls=["cd(folder='nope')"], available_tools=TOOLS,
        execution_results=['{"error": "cd: nope: No such directory."}'],
    )
    assert verdicts["action:no_execution_error"].passed is False


def test_execution_exception_is_flagged():
    verdicts = classify_action_errors(
        parse_ok=True, decoded_calls=["cd(folder='nope')"], available_tools=TOOLS,
        execution_results=[f"{EXECUTION_ERROR_PREFIX} no such directory"],
    )
    assert verdicts["action:no_execution_error"].passed is False
    assert verdicts["action:no_unknown_tool"].passed is True


def test_a_turn_ending_step_asserts_nothing():
    """Claiming 'satisfied' here would wrongly close an open action episode."""
    verdicts = classify_action_errors(
        parse_ok=True, decoded_calls=[], available_tools=TOOLS, execution_results=[],
    )
    assert all(v.passed is None for v in verdicts.values())


def test_tools_with_no_required_arguments_are_not_flagged():
    verdicts = classify_action_errors(
        parse_ok=True, decoded_calls=["ls()"], available_tools=TOOLS,
        execution_results=["[]"],
    )
    assert verdicts["action:no_invalid_arguments"].passed is True


# --------------------------------------------------------------------------- #
# rubric compilation
# --------------------------------------------------------------------------- #


def test_three_rubric_families_are_compiled():
    specs = compile_rubrics(make_task(), make_trace())
    by_id = {s.rubric_id: s for s in specs}

    for rubric_id in ACTION_RUBRICS:
        assert by_id[rubric_id].kind is RubricKind.SAFETY
        assert by_id[rubric_id].initial_state is RubricState.SATISFIED

    assert by_id[state_rubric_id("GorillaFileSystem", "root")].kind is RubricKind.SAFETY
    assert by_id[state_rubric_id("GorillaFileSystem", "long_context")]

    response = by_id[response_rubric_id(0)]
    assert response.kind is RubricKind.GOAL
    assert response.initial_state is RubricState.PENDING, (
        "an absent reference response is an omission, never a violation"
    )
    assert response.turn == 0

    assert no_tool_call_rubric_id(1) in by_id, "empty-reference turn needs a safety rubric"
    assert by_id[no_tool_call_rubric_id(1)].kind is RubricKind.SAFETY


def test_turns_without_responses_get_no_response_rubric():
    specs = compile_rubrics(make_task(), make_trace())
    assert response_rubric_id(1) not in {s.rubric_id for s in specs}


def test_non_empty_reference_turn_gets_no_no_tool_call_rubric():
    specs = compile_rubrics(make_task(), make_trace())
    assert no_tool_call_rubric_id(0) not in {s.rubric_id for s in specs}


def test_every_compiled_rubric_is_hard():
    assert all(s.hard for s in compile_rubrics(make_task(), make_trace()))


def test_state_attributes_come_from_every_snapshot_including_the_initial_one():
    trace = make_trace()
    trace.turns[0].state["TwitterAPI"] = {"tweets": {}}
    pairs = trace.state_attributes()
    assert ("GorillaFileSystem", "root") in pairs
    assert ("TwitterAPI", "tweets") in pairs


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #


def _bfcl_available() -> bool:
    try:
        import bfcl_eval  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


requires_bfcl = pytest.mark.skipif(not _bfcl_available(), reason="bfcl_eval not importable")


@requires_bfcl
def test_all_four_categories_load_with_ground_truth():
    for category in BFCL_CATEGORIES:
        tasks = load_tasks(category)
        assert 190 <= len(tasks) <= 200, category
        for task in tasks.values():
            assert task.num_turns == len(task.ground_truth), task.task_id
            assert task.involved_classes


@requires_bfcl
def test_unscoreable_upstream_entries_are_excluded_and_named():
    """multi_turn_base_154 declares 5 user turns but 6 ground-truth turns.

    The official runner reports force_terminated for it whatever the model does,
    so it is dropped -- but never silently, because it changes every denominator.
    """
    report = load_task_report("multi_turn_base")
    assert len(report.tasks) + len(report.skipped) == 200
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    if VERSION_PREFIX != "BFCL_v3":
        assert report.skipped == {}
        return

    assert "multi_turn_base_154" in report.skipped
    assert "ground-truth turns" in report.skipped["multi_turn_base_154"]

    with pytest.raises(ValueError, match="force_terminated"):
        load_task_report("multi_turn_base", on_malformed="raise")

    kept = load_tasks("multi_turn_base", on_malformed="keep")
    assert "multi_turn_base_154" in kept


@requires_bfcl
def test_only_one_entry_in_the_suite_is_unscoreable():
    skipped = {}
    for category in BFCL_CATEGORIES:
        skipped.update(load_task_report(category).skipped)
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    expected = {"multi_turn_base_154"} if VERSION_PREFIX == "BFCL_v3" else set()
    assert set(skipped) == expected


@requires_bfcl
def test_miss_func_holdout_turns_have_no_user_message():
    tasks = load_tasks("multi_turn_miss_func")
    with_holdout = [t for t in tasks.values() if t.missed_function]
    assert len(with_holdout) == 200
    for task in with_holdout:
        for turn in task.missed_function:
            assert task.user_messages(turn) == [], (
                f"{task.task_id}: holdout turn {turn} should carry no user message"
            )


@requires_bfcl
def test_miss_param_has_no_holdout_field_but_has_empty_reference_turns():
    tasks = load_tasks("multi_turn_miss_param")
    assert all(not t.missed_function for t in tasks.values())
    with_empty = [t for t in tasks.values()
                  if any(not t.ground_truth_calls(u) for u in range(t.num_turns))]
    assert len(with_empty) == 200, (
        "every miss_param task should have a turn whose reference calls nothing"
    )


@requires_bfcl
def test_long_context_flag_follows_the_category():
    assert list(load_tasks("multi_turn_long_context").values())[0].long_context
    assert not list(load_tasks("multi_turn_base").values())[0].long_context
