"""BFCL v3 runtime end to end: rollout -> rubric states -> episodes.

Requires ``bfcl_eval`` and its data; skipped otherwise. No model is involved --
trajectories are scripted, which is what makes the expected episode structure
checkable.
"""

from __future__ import annotations

import os
import json

import pytest

os.environ.setdefault("BFCL_PROJECT_ROOT", "/tmp/claude-bfcl-project")


def _bfcl_available() -> bool:
    try:
        import bfcl_eval  # noqa: F401,PLC0415

        from bfcl_env.adapter import bfcl_data_root  # noqa: PLC0415

        bfcl_data_root()
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _bfcl_available(), reason="bfcl_eval or its data not available"
)

from bfcl_env.adapter import (  # noqa: E402
    load_task,
    state_rubric_id,
)
from bfcl_env.runtime import BFCLRuntime  # noqa: E402
from bfcl_env.bfcl_isolated_environment import (  # noqa: E402
    BFCLIsolatedEnvironmentBackend,
)
from bfcl_env.cost_ledger import CostLedger  # noqa: E402
from bfcl_env.environment_service import LocalEnvironmentBackend  # noqa: E402
from bfcl_env.episode_classifier import classify_rollout  # noqa: E402
from bfcl_env.rollout_protocol import PolicyVersion, StructuredAction  # noqa: E402
from bfcl_env.schemas import (  # noqa: E402
    CostVerdict,
    FailureKind,
    RolloutRecord,
    RubricState,
    RubricStateVector,
    StepRecord,
)

TASK = "multi_turn_base_0"
TURN_END = "All done with this turn."

# The reference answer for turn 2 uses a positional argument, which the official
# decoder drops. A model must use keywords, so the replay does too.
POSITIONAL_FIX = {"sort('final_report.pdf')": "sort(file_name='final_report.pdf')"}


def render(calls: list[str]) -> str:
    return "[" + ", ".join(calls) + "]"


def drive(actions_per_turn, *, runner_mode="recovery_continue", task_id=TASK):
    """Run a scripted trajectory and return everything the pipeline produces.

    ``actions_per_turn`` maps ``turn -> list of raw assistant texts`` (excluding
    the turn-ending text, which is appended automatically).
    """
    runtime = BFCLRuntime(category="multi_turn_base", runner_mode=runner_mode)
    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(task_id, seed=0, rollout_id="r0")

    rollout = RolloutRecord(
        benchmark="bfcl_v3", task_id=task_id, rollout_id="r0",
        policy_step=0, split="test",
    )
    vectors = [RubricStateVector(rollout_id="r0", step=-1,
                                 rubrics=dict(info.initial_rubric_states))]
    mutations = []
    sequence_no = 0
    result = None

    for turn in range(20):
        texts = list(actions_per_turn.get(turn, [])) + [TURN_END]
        for text in texts:
            action = runtime.parse_action(text)
            result = backend.step(info.session_id, sequence_no, action)
            rollout.steps.append(
                StepRecord(
                    step=sequence_no, turn=result.turn,
                    assistant_text=text, raw_observation=result.observation,
                    tool_execution_ok=result.tool_execution_ok,
                    state_before_hash=result.state_before_hash,
                    state_after_hash=result.state_after_hash,
                    assistant_token_span=(sequence_no * 4, (sequence_no + 1) * 4),
                )
            )
            vectors.append(
                RubricStateVector.from_evaluations(
                    "r0", sequence_no, result.rubric_states,
                    result.rubric_evaluations, turn=result.turn,
                )
            )
            mutations.extend(result.mutations)
            sequence_no += 1
            if result.finished:
                break
        if result is not None and result.finished:
            break

    rollout.terminated = bool(result and result.terminated)
    rollout.truncated = bool(result and result.truncated)
    if result and result.environment_error:
        rollout.abort_reason = result.environment_error

    classified = classify_rollout(
        rollout, info.rubric_specs, vectors,
        mutations=mutations,
        cost_ledger=CostLedger(default_when_unspecified=CostVerdict.UNKNOWN),
    )
    backend.close_session(info.session_id)
    return runtime, info, rollout, classified


def reference_actions(task_id=TASK, mutate=None):
    task = load_task(task_id, "multi_turn_base")
    actions = {}
    for turn in range(task.num_turns):
        calls = [POSITIONAL_FIX.get(c, c) for c in task.ground_truth_calls(turn)]
        if mutate:
            calls = mutate(turn, calls)
        actions[turn] = [render(calls)] if calls else []
    return actions


# --------------------------------------------------------------------------- #
# session setup
# --------------------------------------------------------------------------- #


def test_session_exposes_prompts_rubrics_and_metadata():
    runtime = BFCLRuntime()
    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(TASK, 0, "r0")
    try:
        assert "composing functions" in info.system_prompt
        assert "final_report.pdf" in info.task_prompt
        assert info.metadata["num_turns"] == 4
        assert set(info.metadata["involved_classes"]) == {"TwitterAPI", "GorillaFileSystem"}
        assert info.metadata["official_protocol"] is False

        ids = {s.rubric_id for s in info.rubric_specs}
        assert "action:no_parse_error" in ids
        assert state_rubric_id("GorillaFileSystem", "root") in ids
        assert all(s.hard for s in info.rubric_specs)
        assert info.initial_rubric_states[state_rubric_id("GorillaFileSystem", "root")] \
            is RubricState.SATISFIED
    finally:
        backend.close_session(info.session_id)


def test_tool_results_and_next_turn_use_distinct_chat_roles():
    runtime = BFCLRuntime()
    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(TASK, 0, "r0")
    try:
        # reference_actions already returns one rendered action for this turn.
        first_call = reference_actions()[0][0]
        tool_result = backend.step(
            info.session_id, 0, runtime.parse_action(first_call)
        )
        assert tool_result.observation
        assert tool_result.observation_role == "tool"
        assert len(tool_result.observation_messages) == 3
        assert all(m["role"] == "tool" for m in tool_result.observation_messages)

        next_turn = backend.step(
            info.session_id, 1, runtime.parse_action(TURN_END)
        )
        assert next_turn.observation
        assert next_turn.observation_role == "user"
    finally:
        backend.close_session(info.session_id)


def test_native_fc_prompt_and_parser_follow_qwen_tool_protocol():
    runtime = BFCLRuntime(tool_mode="native_fc")
    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(TASK, 0, "native")
    try:
        assert info.metadata["tool_mode"] == "native_fc"
        assert info.system_prompt.startswith("# Tools\n\n")
        assert "<tools>" in info.system_prompt
        assert "[func_name1(" not in info.system_prompt

        first_python_call = load_task(TASK, "multi_turn_base").ground_truth_calls(0)[0]
        name, arguments_text = first_python_call.split("(", 1)
        # The first reference call for this fixture has simple string kwargs.
        arguments = {}
        for pair in arguments_text[:-1].split(", "):
            key, value = pair.split("=", 1)
            arguments[key] = value.strip("'\"")
        raw = "<tool_call>\n" + json.dumps(
            {"name": name, "arguments": arguments}
        ) + "\n</tool_call>"
        parsed = runtime.parse_action(raw)
        assert parsed.parse_ok and len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].raw_text == first_python_call
    finally:
        backend.close_session(info.session_id)


def test_task_process_backend_serves_and_reaps_one_native_session():
    backend = BFCLIsolatedEnvironmentBackend(tool_mode="native_fc")
    info = backend.create_session(TASK, 0, "isolated-native")
    try:
        assert info.metadata["tool_mode"] == "native_fc"
        assert backend.health().num_open_sessions == 1
        result = backend.step(
            info.session_id, 0, StructuredAction(raw_text="I am done."),
        )
        assert result.session_id == info.session_id
    finally:
        backend.close_session(info.session_id)
    assert backend.health().num_open_sessions == 0


def test_the_reference_trajectory_is_a_clean_success():
    _, _, rollout, classified = drive(reference_actions())
    outcome = classified.outcome
    assert outcome.rollout_class == "clean_success", classified.episodes
    assert outcome.terminal_success and not outcome.any_hard_violation
    assert outcome.first_violation_step is None
    assert outcome.failure_kind is FailureKind.NONE
    assert rollout.terminated and not rollout.truncated


# --------------------------------------------------------------------------- #
# violations and recovery
# --------------------------------------------------------------------------- #


def test_a_stray_directory_opens_an_unresolved_state_episode():
    def mutate(turn, calls):
        if turn == 0:
            return ["cd(folder='document')", "mkdir(dir_name='WRONG')"]
        if turn == 1:
            return ["mkdir(dir_name='temp')",
                    "mv(source='final_report.pdf', destination='temp')"] + calls
        return calls

    _, _, _, classified = drive(reference_actions(mutate=mutate))
    outcome = classified.outcome
    assert outcome.rollout_class == "terminal_failure"
    assert outcome.failure_kind is FailureKind.LOCALIZED_VIOLATION
    assert outcome.first_violation_rubric_id == state_rubric_id("GorillaFileSystem", "root")

    episodes = [e for e in classified.episodes if e.rubric_id.startswith("state:")]
    assert episodes and all(not e.resolved for e in episodes), (
        "the stray directory is never removed, so the episode cannot close"
    )


def test_cross_turn_recovery_closes_a_state_episode():
    """Fall behind at turn 0, undo it at turn 1 -- the episode must close.

    This is the case a per-turn rubric id could not express (see the adapter
    module docstring).
    """
    def mutate(turn, calls):
        if turn == 0:
            return ["cd(folder='document')", "mkdir(dir_name='WRONG')"]
        if turn == 1:
            return ["rmdir(dir_name='WRONG')", "mkdir(dir_name='temp')",
                    "mv(source='final_report.pdf', destination='temp')"] + calls
        return calls

    _, _, _, classified = drive(reference_actions(mutate=mutate))
    state_episodes = [e for e in classified.episodes if e.rubric_id.startswith("state:")]
    assert state_episodes, "turn 0 must open a state episode"
    assert all(e.resolved for e in state_episodes), "turn 1 must close it"
    assert classified.outcome.rollout_class == "recovered_success"
    assert classified.outcome.num_recovered_episodes >= 1


def test_same_turn_recovery_from_an_execution_error():
    def mutate(turn, calls):
        if turn == 0:
            return ["cd(folder='does_not_exist')"] + calls
        return calls

    _, _, _, classified = drive(reference_actions(mutate=mutate))
    execution_episodes = [
        e for e in classified.episodes if e.rubric_id == "action:no_execution_error"
    ]
    assert execution_episodes, "a failing call must open an action episode"
    assert execution_episodes[0].resolved, "the next clean call closes it"
    assert execution_episodes[0].latency_steps is not None


def test_an_unparseable_call_attempt_is_a_violation():
    actions = reference_actions()
    actions[0] = ["[cd(folder='document'"] + actions[0]
    _, _, _, classified = drive(actions)
    parse_episodes = [
        e for e in classified.episodes if e.rubric_id == "action:no_parse_error"
    ]
    assert parse_episodes, "a malformed call must open a parse episode"


def test_prose_ends_the_turn_without_a_parse_violation():
    """The official decoder raises on prose too; that is a turn end, not an error."""
    runtime = BFCLRuntime()
    for text in ("I am done.", "The task is complete, nothing left (really).", ""):
        action = runtime.parse_action(text)
        assert action.parse_ok is True, text
        assert action.tool_calls == []

    broken = runtime.parse_action("[cd(folder='x'")
    assert broken.parse_ok is False
    assert broken.parse_error


def test_calling_an_unavailable_tool_is_a_violation():
    actions = reference_actions()
    actions[0] = ["[teleport(destination='mars')]"] + actions[0]
    _, _, _, classified = drive(actions)
    assert any(e.rubric_id == "action:no_unknown_tool" for e in classified.episodes)


# --------------------------------------------------------------------------- #
# runner modes (plan 7.2)
# --------------------------------------------------------------------------- #


def test_official_stop_ends_at_the_first_turn_mismatch():
    def mutate(turn, calls):
        if turn == 0:
            return ["cd(folder='document')", "mkdir(dir_name='WRONG')"]
        return calls

    _, _, stopped, _ = drive(reference_actions(mutate=mutate), runner_mode="official_stop")
    _, _, continued, _ = drive(
        reference_actions(mutate=mutate), runner_mode="recovery_continue"
    )
    assert len(stopped.steps) < len(continued.steps)
    assert max(s.turn for s in stopped.steps) == 0
    assert max(s.turn for s in continued.steps) == 3


def test_recovery_continue_preserves_the_history_after_the_first_error():
    def mutate(turn, calls):
        if turn == 0:
            return ["cd(folder='does_not_exist')"] + calls
        return calls

    _, _, rollout, _ = drive(reference_actions(mutate=mutate))
    from bfcl_env.adapter import is_execution_failure

    failed_steps = [
        s for s in rollout.steps
        if any(is_execution_failure(line) for line in s.raw_observation.splitlines())
    ]
    assert failed_steps, "the failed call must stay in the log"
    assert "No such" in failed_steps[0].raw_observation
    assert "directory" in failed_steps[0].raw_observation
    assert max(s.turn for s in rollout.steps) == 3, "and the rollout must continue"


def test_official_stop_does_not_change_the_clean_trajectory():
    _, _, _, classified = drive(reference_actions(), runner_mode="official_stop")
    assert classified.outcome.rollout_class == "clean_success"


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def test_every_hard_rubric_has_a_verdict_at_terminal():
    """A single UNKNOWN would make the whole rollout unjudgeable (plan 3.1)."""
    _, info, _, classified = drive(reference_actions())
    assert not classified.outcome.aborted
    assert classified.outcome.unknown_rubrics == []


def test_sessions_do_not_share_simulator_state():
    runtime = BFCLRuntime()
    backend = LocalEnvironmentBackend(runtime)
    first = backend.create_session(TASK, 0, "r0")
    second = backend.create_session(TASK, 0, "r1")
    try:
        assert first.session_id != second.session_id
        backend.step(first.session_id, 0,
                     runtime.parse_action("[mkdir(dir_name='ONLY_IN_FIRST')]"))
        a = backend.step(first.session_id, 1, runtime.parse_action(TURN_END))
        b = backend.step(second.session_id, 0, runtime.parse_action(TURN_END))
        assert a.state_after_hash != b.state_after_hash
    finally:
        backend.close_session(first.session_id)
        backend.close_session(second.session_id)


def test_closing_a_session_releases_the_cached_simulator_instances():
    """execute_multi_turn_func_call never evicts them; the runtime must."""
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

    runtime = BFCLRuntime()
    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(TASK, 0, "r0")
    backend.step(info.session_id, 0, runtime.parse_action("[ls()]"))

    leaked = [k for k in vars(multi_turn_utils) if k.endswith("_instance") and TASK in k]
    assert leaked, "instances should exist while the session is open"

    backend.close_session(info.session_id)
    still_there = [k for k in vars(multi_turn_utils)
                   if k.endswith("_instance") and TASK in k]
    assert len(still_there) < len(leaked)


def test_policy_version_stamping_round_trips():
    version = PolicyVersion(policy_step=3, model_checksum="abc")
    assert version.matches(PolicyVersion(policy_step=3, model_checksum="abc"))
    assert not version.matches(PolicyVersion(policy_step=4, model_checksum="abc"))


# --------------------------------------------------------------------------- #
# stage 5.5: a moving requirement must not manufacture recovery
# --------------------------------------------------------------------------- #


def test_running_ahead_of_the_script_is_superseded_not_recovered():
    """Do a later turn's work early, then sit still.

    At turn 0 the world does not match turn 0's reference, so an episode opens.
    By turn 3 the reference has advanced onto the world the model already built,
    and the violation disappears -- without the model touching anything. That is
    the goalpost moving, not a recovery.
    """
    from bfcl_env.schemas import EpisodeCloseReason

    def mutate(turn, calls):
        if turn == 0:
            # turn 0's own work, plus turn 3's mv done far too early
            return calls + ["cd(folder='..')",
                            "mv(source='previous_report.pdf', destination='temp')",
                            "cd(folder='temp')"]
        if turn == 3:
            # the reference moves onto our world; do nothing that changes it
            return [c for c in calls if c.startswith(("cd(", "diff("))]
        return calls

    _, _, _, classified = drive(reference_actions(mutate=mutate))
    state_episodes = [e for e in classified.episodes if e.rubric_id.startswith("state:")]
    assert state_episodes, "running ahead must open a state episode"

    episode = state_episodes[0]
    assert episode.open_target_revision != episode.close_target_revision or not episode.resolved
    assert not episode.recovered_by_action, (
        "the agent never repaired the mismatch it created; the reference moved"
    )
    assert classified.outcome.num_recovered_episodes == 0


def test_state_evaluations_carry_target_revisions():
    runtime = BFCLRuntime()
    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(TASK, 0, "r0")
    try:
        result = backend.step(info.session_id, 0, runtime.parse_action(TURN_END))
        root = result.rubric_evaluations[state_rubric_id("GorillaFileSystem", "root")]
        assert root.target_revision is not None
        assert root.expected_value and root.actual_value
        action = result.rubric_evaluations["action:no_parse_error"]
        assert action.target_revision is None, "action rubrics have a fixed expectation"
    finally:
        backend.close_session(info.session_id)


def test_parser_and_execution_policies_are_reported():
    runtime = BFCLRuntime()
    described = runtime.describe()
    assert described["parser_policy"] == "bfcl_python_call_attempt_v1"
    assert described["execution_outcome_policy"] == "bfcl_dual_channel_failure_v1"

    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(TASK, 0, "r0")
    try:
        assert info.metadata["parser_policy"] == "bfcl_python_call_attempt_v1"
        assert backend.health().detail["parser_policy"]
    finally:
        backend.close_session(info.session_id)


def test_parse_kind_travels_with_the_action():
    runtime = BFCLRuntime()
    assert runtime.parse_action("I am done.").parse_kind == "END_TURN"
    assert runtime.parse_action("[cd(folder='x'").parse_kind == "PARSE_ERROR"
    assert runtime.parse_action("[ls()]").parse_kind == "TOOL_CALL"


def test_mutation_is_read_from_simulator_state_not_from_tool_names():
    """A name-prefix heuristic left every BFCL episode abstaining on cost."""
    runtime = BFCLRuntime()
    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(TASK, 0, "r0")
    try:
        read_only = backend.step(
            info.session_id, 0, runtime.parse_action("[ls()]")
        )
        assert read_only.mutations
        assert all(m.mutating is False for m in read_only.mutations), (
            "ls changes nothing, so it must be recorded as read-only"
        )

        mutating = backend.step(
            info.session_id, 1,
            runtime.parse_action("[cd(folder='document'), mkdir(dir_name='NEW')]"),
        )
        assert mutating.mutations
        assert any(m.mutating is True for m in mutating.mutations), (
            "mkdir changes simulator state, so it must be recorded as mutating"
        )
        assert all(m.mutating is not None for m in read_only.mutations + mutating.mutations)
    finally:
        backend.close_session(info.session_id)


def test_bfcl_cost_is_decidable_so_episodes_can_be_labelled():
    from bfcl_env.cost_evidence import evaluate_cost_vector

    runtime = BFCLRuntime()
    backend = LocalEnvironmentBackend(runtime)
    info = backend.create_session(TASK, 0, "r0")
    try:
        result = backend.step(
            info.session_id, 0,
            runtime.parse_action("[cd(folder='document'), mkdir(dir_name='NEW')]"),
        )
        vector = evaluate_cost_vector("bfcl_v3", result.mutations)
        assert not vector.is_undecided, vector.unknown_dimensions()
        assert vector.has_external_cost
    finally:
        backend.close_session(info.session_id)
