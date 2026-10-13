"""BFCL v3 multi-turn as an :class:`~bfcl_env.rollout_protocol.EnvironmentRuntime`.

In-process behind :class:`~bfcl_env.environment_service.LocalEnvironmentBackend`
(ADR 0001): ``bfcl_eval`` ships torch and vLLM in the same virtualenv, so unlike
AppWorld it needs no separate service.

Turn structure, matching ``BaseHandler.inference_multi_turn_prompting``: a turn is
a run of steps that decode into calls, ended by a step that decodes into none (or
by ``MAXIMUM_STEP_LIMIT``). Rubrics are therefore evaluated at two resolutions:

* every step -> the four deterministic ``action:*`` rubrics,
* every turn boundary -> ``state:*`` against that turn's reference snapshot, and
  ``response:turn{u}``.

Two runner modes (plan section 7.2):

* ``official_stop`` reproduces the official score, which returns at the first
  turn-level mismatch,
* ``recovery_continue`` keeps the *same* trajectory going past the mismatch. It is
  not a counterfactual rollout and not an official score; it exists because
  recovery cannot be observed at all if the run stops at the first error.

Terminal completeness
---------------------
Every hard rubric must carry a definite verdict at the final step. A rubric left
``UNKNOWN`` at terminal makes the whole rollout ``aborted_or_unknown``
(plan section 3.1), which would silently delete most of the dataset. So the
terminal step re-emits the last known ``action:*`` verdicts (latched, never
invented), evaluates every ``response:turn{u}`` goal, and compares ``state:*``
against the last turn the model actually completed -- not against the reference
end state, which no incomplete rollout could ever match.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

from bfcl_env.adapter import (
    ACTION_RUBRICS,
    EXECUTION_OUTCOME_POLICY_VERSION,
    PARSER_POLICY_VERSION,
    RUNNER_MODES,
    ActionParseKind,
    BFCLTask,
    GroundTruthTrace,
    TurnSnapshot,
    available_tool_documents,
    build_expectations,
    canonical_value,
    classify_action_errors,
    classify_raw_action,
    compare_states,
    compile_rubrics,
    is_execution_failure,
    load_task,
    load_tool_documents,
    no_tool_call_rubric_id,
    response_rubric_id,
    responses_contain,
    snapshot_instances,
    state_rubric_id,
)
from bfcl_env.cost_ledger import MutationEvent
from bfcl_env.rollout_protocol import RuntimeReset, RuntimeStep, StructuredAction
from bfcl_env.schemas import RubricEvaluation, ToolCall

__all__ = ["BFCLRuntime", "MAXIMUM_STEP_LIMIT", "READ_ONLY_PREFIXES"]

MAXIMUM_STEP_LIMIT = 20
"""Per-turn step budget, from ``bfcl_eval.constants.default_prompts``."""

_CALL_ATTEMPT_RE = re.compile(r"[A-Za-z_]\w*\(")
"""Evidence that the model tried to emit a call. No whitespace is allowed before
the paren, so ordinary prose containing a parenthesis is not misread as one."""

_CALL_NAME_RE = re.compile(r"^\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")

READ_ONLY_PREFIXES = ()
"""Retained for compatibility. Mutation is no longer guessed from a tool name: the
simulators' own public state answers the question exactly (see
:func:`_mutation_events`)."""


@dataclass
class _BFCLSession:
    """Per-rollout mutable state. Only :class:`BFCLRuntime` interprets it."""

    task: BFCLTask
    trace: GroundTruthTrace
    tool_documents: dict[str, dict[str, Any]]
    instance_key: str
    """Unique ``model_name`` passed to ``execute_multi_turn_func_call``. The helper
    caches instances in module globals keyed by it and never evicts them, so a
    shared name would silently reuse another rollout's dirty world."""

    runner_mode: str
    expectations: dict[tuple[str, str], list[tuple[int, str]]] = field(
        default_factory=dict
    )
    turn: int = 0
    step_in_turn: int = 0
    calls_executed_in_turn: int = 0
    accumulated_responses: list[str] = field(default_factory=list)
    last_action_verdicts: dict[str, bool] = field(default_factory=dict)
    empty_turn_violations: dict[int, bool] = field(default_factory=dict)
    completed_turns: int = 0
    official_stop_triggered: bool = False
    finished: bool = False
    instances: dict[str, Any] = field(default_factory=dict)


class BFCLRuntime:
    """BFCL v3 multi-turn runtime.

    Args:
        category: one of the four multi-turn categories.
        runner_mode: ``official_stop`` or ``recovery_continue``.
        data_root: overrides ``BFCL_DATA_ROOT`` / the installed package data.
    """

    benchmark = "bfcl_v3"

    def __init__(
        self,
        category: str = "multi_turn_base",
        runner_mode: str = "recovery_continue",
        tool_mode: str = "prompting",
        isolate_process_state: bool = False,
        data_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if runner_mode not in RUNNER_MODES:
            raise ValueError(f"runner_mode must be one of {RUNNER_MODES}")
        if tool_mode not in ("prompting", "native_fc"):
            raise ValueError("tool_mode must be 'prompting' or 'native_fc'")
        self.category = category
        self.runner_mode = runner_mode
        self.tool_mode = tool_mode
        self.isolate_process_state = isolate_process_state
        self.data_root = data_root

    # ---- helpers ----------------------------------------------------------- #

    @staticmethod
    def _execute(
        calls: list[str], task: BFCLTask, instance_key: str
    ) -> tuple[list[str], dict[str, Any]]:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: PLC0415
            execute_multi_turn_func_call,
        )

        return execute_multi_turn_func_call(
            calls,
            initial_config=task.initial_config,
            involved_classes=task.involved_classes,
            model_name=instance_key,
            test_entry_id=task.task_id,
            long_context=task.long_context,
            is_evaL_run=False,
        )

    def _run_ground_truth(self, task: BFCLTask) -> GroundTruthTrace:
        """Execute the reference trajectory once, snapshotting after every turn."""
        if self.isolate_process_state:
            completed = subprocess.run(
                [sys.executable, "-m", "bfcl_env.bfcl_process_worker", "ground_truth"],
                input=pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "isolated BFCL ground truth failed: "
                    + completed.stderr.decode("utf-8", errors="replace")[-2000:]
                )
            return pickle.loads(completed.stdout)

        key = f"opsd_gt_{uuid.uuid4().hex[:12]}"
        trace = GroundTruthTrace(task_id=task.task_id)

        _, instances = self._execute([], task, key)
        trace.initial_state = snapshot_instances(instances)

        for turn in range(task.num_turns):
            calls = task.ground_truth_calls(turn)
            responses, instances = self._execute(list(calls), task, key)
            trace.turns.append(
                TurnSnapshot(
                    turn=turn,
                    state=snapshot_instances(instances),
                    responses=list(responses),
                    calls=list(calls),
                )
            )
        self._drop_instances(key, task)
        return trace

    @staticmethod
    def _drop_instances(instance_key: str, task: BFCLTask) -> None:
        """Remove the cached simulator instances from the helper's module globals.

        ``execute_multi_turn_func_call`` never evicts them, so a long-running
        service would otherwise accumulate one live world per rollout forever.
        """
        try:
            from bfcl_eval.eval_checker.multi_turn_eval import (  # noqa: PLC0415
                multi_turn_utils,
            )
        except ImportError:
            return
        sanitized = re.sub(r"[-./:]", "_", instance_key)
        namespace = vars(multi_turn_utils)
        for class_name in task.involved_classes:
            # v1.3 lower-cased the class suffix; the author-lineage commit keeps
            # its original case. Remove either spelling so switching the pinned
            # runtime cannot leak simulator worlds between rollouts.
            for spelling in {class_name, class_name.lower()}:
                namespace.pop(
                    f"{sanitized}_{task.task_id}_{spelling}_instance", None
                )

    def _user_message(self, task: BFCLTask, turn: int) -> str:
        """The user text that opens ``turn``.

        A ``multi_turn_miss_func`` holdout turn carries no user message on disk;
        the benchmark substitutes a synthetic one that reveals the held-back
        functions, and so does this.
        """
        messages = task.user_messages(turn)
        if messages:
            return "\n".join(m.get("content", "") for m in messages)

        if task.is_holdout_turn(turn):
            from bfcl_eval.constants.default_prompts import (  # noqa: PLC0415
                DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
            )

            if self.tool_mode == "native_fc":
                return DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC

            documents = [
                self._all_documents_cache[name]
                for name in task.missed_function.get(turn, [])
                if name in self._all_documents_cache
            ]
            return DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                functions=documents
            )
        return ""

    # ---- EnvironmentRuntime ------------------------------------------------ #

    def reset(self, task_id: str, seed: int) -> RuntimeReset:
        task = load_task(task_id, self.category, self.data_root)
        documents = load_tool_documents(
            task.involved_classes, self.data_root, category=task.category
        )
        self._all_documents_cache = documents

        trace = self._run_ground_truth(task)
        specs = compile_rubrics(task, trace)

        visible = available_tool_documents(task, 0, documents)
        if self.tool_mode == "native_fc":
            system_prompt = _native_fc_system_prompt(list(visible.values()))
        else:
            try:
                from bfcl_eval.constants.default_prompts import (  # noqa: PLC0415
                    DEFAULT_SYSTEM_PROMPT,
                )
            except ImportError:
                # Renamed to a reference template in the author-lineage commit.
                from bfcl_eval.constants.default_prompts import (  # type: ignore[no-redef]  # noqa: PLC0415
                    _DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
                )

            system_prompt = DEFAULT_SYSTEM_PROMPT.format(
                functions=list(visible.values())
            )

        session = _BFCLSession(
            task=task,
            trace=trace,
            tool_documents=documents,
            instance_key=f"opsd_{uuid.uuid4().hex[:12]}",
            runner_mode=self.runner_mode,
            expectations=build_expectations(trace),
        )
        _, session.instances = self._execute([], task, session.instance_key)

        return RuntimeReset(
            handle=session,
            system_prompt=system_prompt,
            task_prompt=self._user_message(task, 0),
            rubric_specs=specs,
            state_hash=_state_hash(snapshot_instances(session.instances)),
            max_steps=task.num_turns * MAXIMUM_STEP_LIMIT,
            metadata={
                "category": task.category,
                "runner_mode": self.runner_mode,
                "num_turns": task.num_turns,
                "involved_classes": list(task.involved_classes),
                # CrEST's privileged self-teacher is conditioned on reference
                # tool-call results from *preceding* turns.  Keep the exact
                # ordered calls/results on SessionInfo so replay-buffer
                # construction does not have to re-run a stateful simulator or
                # reconstruct privileged data from rubric prose.
                "ground_truth_tool_calls_by_turn": [
                    list(turn.calls) for turn in trace.turns
                ],
                "ground_truth_tool_results_by_turn": [
                    list(turn.responses) for turn in trace.turns
                ],
                "official_protocol": self.runner_mode == "official_stop",
                "tool_mode": self.tool_mode,
                "isolated_process_state": self.isolate_process_state,
                "parser_policy": PARSER_POLICY_VERSION,
                "execution_outcome_policy": EXECUTION_OUTCOME_POLICY_VERSION,
            },
        )

    def parse_action(self, raw_text: str) -> StructuredAction:
        """Decode into one of TOOL_CALL / END_TURN / PARSE_ERROR.

        The tri-state rule lives in the adapter (``classify_raw_action``) so that
        exactly one place decides what counts as a parser failure, and the policy
        version travels with the result.
        """
        if self.tool_mode == "native_fc":
            calls, error = _parse_native_fc(raw_text)
            return StructuredAction(
                raw_text=raw_text,
                tool_calls=[
                    ToolCall(tool=_call_name(c) or "", raw_text=c) for c in calls
                ],
                parse_ok=error is None,
                parse_error=error,
                parse_kind=(
                    ActionParseKind.TOOL_CALL.value
                    if calls
                    else ActionParseKind.END_TURN.value
                ),
            )

        parsed = classify_raw_action(raw_text)
        return StructuredAction(
            raw_text=raw_text,
            tool_calls=[
                ToolCall(tool=_call_name(c) or "", raw_text=c) for c in parsed.calls
            ],
            parse_ok=parsed.parse_ok,
            parse_error=parsed.error,
            parse_kind=parsed.kind.value,
        )

    def execute(self, handle: Any, action: StructuredAction) -> RuntimeStep:
        session: _BFCLSession = handle
        task = session.task
        calls = [c.raw_text or "" for c in action.tool_calls if c.raw_text]

        state_before = _state_hash(snapshot_instances(session.instances))
        ends_turn = (not calls) or session.step_in_turn >= MAXIMUM_STEP_LIMIT
        responses: list[str] = []
        mutations: list[MutationEvent] = []

        if calls and not ends_turn:
            responses, session.instances = self._execute(
                calls, task, session.instance_key
            )
            session.accumulated_responses.extend(responses)
            session.calls_executed_in_turn += len(calls)
            mutations = _mutation_events(
                session.turn, calls, responses,
                state_changed=_state_hash(snapshot_instances(session.instances))
                != state_before,
            )

        visible = available_tool_documents(task, session.turn, session.tool_documents)
        evaluations = dict(
            classify_action_errors(
                parse_ok=action.parse_ok,
                decoded_calls=calls if not ends_turn else [],
                available_tools=visible,
                execution_results=responses,
            )
        )
        for rubric_id, evaluation in evaluations.items():
            if evaluation.passed is not None:
                session.last_action_verdicts[rubric_id] = evaluation.passed

        # An empty-ground-truth turn must produce no tool call at all.
        gt_calls = task.ground_truth_calls(session.turn)
        if not gt_calls and calls and not ends_turn:
            session.empty_turn_violations[session.turn] = True
            rubric_id = no_tool_call_rubric_id(session.turn)
            evaluations[rubric_id] = RubricEvaluation(
                rubric_id, passed=False,
                trace=f"{len(calls)} call(s) executed in a turn whose reference is empty",
            )

        state = snapshot_instances(session.instances)
        evidence: dict[str, Any] = {
            "turn": session.turn,
            "step_in_turn": session.step_in_turn,
            "ends_turn": ends_turn,
            "num_calls": len(calls),
        }

        if not ends_turn:
            session.step_in_turn += 1
            return RuntimeStep(
                observation=_render_observation(responses),
                observation_role="tool",
                observation_messages=[
                    {"role": "tool", "content": response}
                    for response in responses
                ],
                rubric_evaluations=evaluations,
                evaluator_evidence=evidence,
                state_hash=_state_hash(state),
                state_delta={},
                mutations=mutations,
                tool_execution_ok=action.parse_ok and not any(
                    is_execution_failure(r) for r in responses
                ),
                terminated=False,
                turn=session.turn,
            )

        # ---- turn boundary ------------------------------------------------- #
        finished_turn = session.turn
        turn_ok = self._evaluate_turn(session, finished_turn, state, evaluations, evidence)
        session.completed_turns = finished_turn + 1
        session.turn += 1
        session.step_in_turn = 0
        session.calls_executed_in_turn = 0

        stop_officially = session.runner_mode == "official_stop" and not turn_ok
        if stop_officially:
            session.official_stop_triggered = True
            evidence["official_stop"] = True

        out_of_turns = session.turn >= task.num_turns
        terminated = out_of_turns or stop_officially

        if terminated:
            session.finished = True
            evaluations.update(self._terminal_evaluations(session, state))
            observation = ""
        else:
            observation = self._user_message(task, session.turn)

        system_prompt_update = None
        if (
            not terminated
            and self.tool_mode == "native_fc"
            and task.is_holdout_turn(session.turn)
        ):
            revealed = available_tool_documents(
                task, session.turn, session.tool_documents
            )
            system_prompt_update = _native_fc_system_prompt(list(revealed.values()))

        return RuntimeStep(
            observation=observation,
            observation_role="user",
            observation_messages=(
                [{"role": "user", "content": observation}] if observation else []
            ),
            system_prompt_update=system_prompt_update,
            rubric_evaluations=evaluations,
            evaluator_evidence=evidence,
            state_hash=_state_hash(state),
            state_delta={},
            mutations=mutations,
            tool_execution_ok=action.parse_ok,
            terminated=terminated,
            turn=finished_turn,
        )

    def close(self, handle: Any) -> None:
        session: _BFCLSession = handle
        self._drop_instances(session.instance_key, session.task)
        session.instances = {}

    def describe(self) -> dict[str, Any]:
        from bfcl_env.manifest import git_sha  # noqa: PLC0415

        detail: dict[str, Any] = {
            "runtime": "BFCLRuntime",
            "category": self.category,
            "runner_mode": self.runner_mode,
            "tool_mode": self.tool_mode,
            "isolated_process_state": self.isolate_process_state,
            "rubric_compiler_version": "bfcl-1",
            "parser_policy": PARSER_POLICY_VERSION,
            "execution_outcome_policy": EXECUTION_OUTCOME_POLICY_VERSION,
        }
        try:
            import bfcl_eval  # noqa: PLC0415

            package_root = os.path.dirname(os.path.dirname(bfcl_eval.__file__))
            detail["commit"] = git_sha(package_root)
            detail["bfcl_path"] = os.path.dirname(bfcl_eval.__file__)
        except ImportError:
            detail["commit"] = None
        return detail

    # ---- rubric evaluation ------------------------------------------------- #

    def _evaluate_turn(
        self,
        session: _BFCLSession,
        turn: int,
        state: dict[str, dict[str, Any]],
        evaluations: dict[str, RubricEvaluation],
        evidence: dict[str, Any],
    ) -> bool:
        """State and response checks for a just-finished turn. Returns official validity."""
        snapshot = session.trace.snapshot(turn)
        if snapshot is None:
            return True

        state_verdicts = compare_states(state, snapshot.state)
        differing = []
        for (class_name, attribute), passed in state_verdicts.items():
            rubric_id = state_rubric_id(class_name, attribute)
            evaluations[rubric_id] = self._state_evaluation(
                session, rubric_id, class_name, attribute, passed, state, turn
            )
            if not passed:
                differing.append(f"{class_name}.{attribute}")

        response_ok = True
        if snapshot.responses:
            response_ok, missing = responses_contain(
                session.accumulated_responses, snapshot.responses
            )
            rubric_id = response_rubric_id(turn)
            evaluations[rubric_id] = RubricEvaluation(
                rubric_id,
                passed=response_ok,
                trace="" if response_ok else f"missing {len(missing)} reference response(s)",
            )
            evidence["missing_responses"] = [] if response_ok else missing[:5]

        if not snapshot.calls:
            rubric_id = no_tool_call_rubric_id(turn)
            violated = session.empty_turn_violations.get(turn, False)
            evaluations[rubric_id] = RubricEvaluation(
                rubric_id,
                passed=not violated,
                trace="a tool executed in an empty-reference turn" if violated else "",
            )

        evidence["state_differences"] = differing[:8]
        evidence["turn_valid"] = not differing and response_ok
        return not differing and response_ok

    @staticmethod
    def _state_evaluation(
        session: _BFCLSession,
        rubric_id: str,
        class_name: str,
        attribute: str,
        passed: bool,
        state: dict[str, dict[str, Any]],
        turn: int,
    ) -> RubricEvaluation:
        """A state verdict carrying what was wanted and what was there.

        Both values are needed downstream: an episode that closes because the
        observed value changed is a recovery, one that closes because the
        expectation moved onto an unchanged world is not.
        """
        per_turn = session.expectations.get((class_name, attribute)) or []
        revision, expected = per_turn[turn] if 0 <= turn < len(per_turn) else (None, None)
        attributes = state.get(class_name, {})
        actual = (
            canonical_value(attributes[attribute])
            if attribute in attributes
            else "<absent>"
        )
        return RubricEvaluation(
            rubric_id,
            passed=passed,
            trace="" if passed else f"{class_name}.{attribute} differs at turn {turn}",
            target_revision=revision,
            expected_value=expected,
            actual_value=actual,
        )

    def _terminal_evaluations(
        self, session: _BFCLSession, state: dict[str, dict[str, Any]]
    ) -> dict[str, RubricEvaluation]:
        """Give every hard rubric a definite verdict at the final step.

        ``state:*`` is compared against the last turn the model actually completed:
        comparing an early-stopped rollout against the reference *end* state would
        mark every attribute violated at the terminal step and bury the real first
        violation.
        """
        evaluations: dict[str, RubricEvaluation] = {}

        for rubric_id in ACTION_RUBRICS:
            evaluations[rubric_id] = RubricEvaluation(
                rubric_id,
                passed=session.last_action_verdicts.get(rubric_id, True),
                trace="latched terminal verdict",
            )

        reference_turn = max(0, session.completed_turns - 1)
        reference = session.trace.snapshot(reference_turn)
        reference_state = reference.state if reference else session.trace.initial_state
        for (class_name, attribute), passed in compare_states(state, reference_state).items():
            rubric_id = state_rubric_id(class_name, attribute)
            evaluations[rubric_id] = self._state_evaluation(
                session, rubric_id, class_name, attribute, passed, state, reference_turn
            )

        for snapshot in session.trace.turns:
            if snapshot.responses:
                ok, _ = responses_contain(
                    session.accumulated_responses, snapshot.responses
                )
                rubric_id = response_rubric_id(snapshot.turn)
                evaluations[rubric_id] = RubricEvaluation(rubric_id, passed=ok)
            if not snapshot.calls:
                rubric_id = no_tool_call_rubric_id(snapshot.turn)
                evaluations[rubric_id] = RubricEvaluation(
                    rubric_id,
                    passed=not session.empty_turn_violations.get(snapshot.turn, False),
                )

        return evaluations


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _call_name(call: str) -> str | None:
    match = _CALL_NAME_RE.match(call)
    return match.group(1) if match else None


def _native_fc_system_prompt(functions: list[dict[str, Any]]) -> str:
    """System content rendered by BFCL's official ``QwenFCHandler``.

    ``TransformersGenerationBackend`` applies the checkpoint chat template
    without a separate ``tools=`` argument. Supplying this content as the system
    message yields the same serialized tool block while keeping the generic
    generation interface benchmark-independent.
    """
    documents = "".join(f"\n{json.dumps(function)}" for function in functions)
    return (
        "# Tools\n\nYou may call one or more functions to assist with the user "
        "query.\n\nYou are provided with function signatures within <tools></tools> "
        f"XML tags:\n<tools>{documents}\n</tools>\n\nFor each function call, "
        "return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def _parse_native_fc(raw_text: str) -> tuple[list[str], str | None]:
    """Delegate native extraction to the pinned official Qwen handler.

    This intentionally inherits its newline sensitivity and its treatment of a
    malformed envelope as no call.  Reconstructing the conversion locally had
    already introduced observable whitespace differences in executable calls.
    """
    from bfcl_eval.model_handler.local_inference.qwen_fc import (  # noqa: PLC0415
        QwenFCHandler,
    )

    handler = object.__new__(QwenFCHandler)
    try:
        # v1.3 accepted only result; the author-lineage handler additionally
        # requires has_tool_call_tag. Its value does not affect Qwen extraction.
        import inspect

        parameters = inspect.signature(type(handler).decode_execute).parameters
        if "has_tool_call_tag" in parameters:
            decoded = handler.decode_execute(raw_text, False)
        else:
            decoded = handler.decode_execute(raw_text)
        return list(decoded), None
    except Exception as error:  # noqa: BLE001 - model output is untrusted data
        return [], f"invalid native tool call: {type(error).__name__}: {error}"


def _render_observation(responses: list[str]) -> str:
    if not responses:
        return ""
    return "\n".join(responses)


def _state_hash(state: dict[str, dict[str, Any]]) -> str:
    import hashlib

    payload = repr(sorted(
        (class_name, sorted((k, repr(v)) for k, v in attributes.items()))
        for class_name, attributes in state.items()
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _mutation_events(
    turn: int, calls: list[str], responses: list[str], *, state_changed: bool
) -> list[MutationEvent]:
    """Mutation events derived from the simulators' own state, not from tool names.

    A prefix heuristic left almost every BFCL call ``mutating=None``, which made
    every BFCL episode abstain on cost. The state hash answers the question
    exactly: if the public attributes of the involved simulators changed across the
    step, the step mutated the world; if they did not, every call in it was
    read-only.

    The granularity is the step, not the call, because one step's calls share a
    single state transition. So a step that changed something is recorded as one
    mutating event naming its calls, and a step that changed nothing is recorded as
    one read-only event per call -- which keeps ``process_overhead_calls``
    meaningful.
    """
    if state_changed:
        failed = [r for r in responses if is_execution_failure(r)]
        return [
            MutationEvent(
                step=turn,
                tool=",".join(sorted({_call_name(c) or "?" for c in calls})),
                mutating=True,
                target=f"simulator_state@turn{turn}",
                detail=(
                    f"{len(calls)} call(s) changed simulator state"
                    + (f"; {len(failed)} failed" if failed else "")
                ),
            )
        ]

    return [
        MutationEvent(
            step=turn,
            tool=_call_name(call) or "",
            mutating=False,
            target=_call_name(call) or "",
            detail="no simulator state change",
        )
        for call in calls
    ]
