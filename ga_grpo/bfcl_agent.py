"""Molt agent for autonomous and reference-prefix BFCL-v3 rollouts."""
import json
import os
import sys
import time
from datetime import datetime, timezone

# Ray loads this file with importlib from an absolute path while its working
# directory is third_party/molt. Make sibling modules importable explicitly.
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from openai import AsyncOpenAI

# The repository root holds the `bfcl_env` environment layer; BFCL_SOURCE_ROOT is the pinned BFCL checkout. Both are
# exported by scripts/train.sh from config/env.sh.
REPO_ROOT = os.environ.get("BFCL_OPSD_ROOT", os.path.dirname(AGENT_DIR))
BFCL_SOURCE = os.environ.get("BFCL_SOURCE_ROOT") or os.environ.get("BFCL_RUNTIME_SOURCE_ROOT")
if not BFCL_SOURCE:
    raise RuntimeError("BFCL_SOURCE_ROOT is unset: source config/env.sh (see config/env.example.sh)")
sys.path[:0] = [REPO_ROOT, BFCL_SOURCE]
# bfcl_eval and its dependencies live in the BFCL venv, not in the Molt venv.
if os.environ.get("BFCL_VENV_SITE_PACKAGES"):
    sys.path.append(os.environ["BFCL_VENV_SITE_PACKAGES"])

from bfcl_env.runtime import BFCLRuntime  # noqa: E402
from bfcl_env.rollout_protocol import StructuredAction  # noqa: E402
from molt.agents import ChatAgent, ChatAgentRunner, Result  # noqa: E402

from curriculum import assign, call_positions, current_h, initial_h, record  # noqa: E402
import banks  # noqa: E402
from dapo import overlong_reward  # noqa: E402
from bfcl_guide import _call_payload, _guide_action  # noqa: E402,F401


class BFCLAgent(ChatAgent):
    async def run(self, ctx) -> Result:
        label = json.loads(ctx.label) if isinstance(ctx.label, str) else ctx.label
        runtime = BFCLRuntime(
            category=label.get("category", "multi_turn_base"),
            runner_mode="official_stop",
            tool_mode=os.environ.get("BFCL_TOOL_MODE", "native_fc"),
        )
        reset = runtime.reset(label["task_id"], int(label.get("seed", 42)))
        messages = [
            {"role": "system", "content": reset.system_prompt},
            {"role": "user", "content": reset.task_prompt},
        ]
        requested_guide_turns = int(label.get("guide_turns", 0))
        num_turns = int(reset.metadata["num_turns"])
        curriculum_enabled = (
            requested_guide_turns < 0
            and os.environ.get("BFCL_CURRICULUM", "0") == "1"
        )
        h_init = initial_h(
            num_turns, label.get("curriculum_init", "suffix"),
            selected_h=label.get("curriculum_initial_h"),
        )
        reference_turns = reset.metadata["ground_truth_tool_calls_by_turn"]
        # "call": the handoff may fall inside a user turn (see curriculum.call_positions).
        call_unit = os.environ.get("BFCL_HANDOFF_UNIT", "turn") == "call"
        # The curriculum leases this rollout a slot in its task's current group: the group's
        # id and its fixed h (curriculum.assign), so the 16 rollouts of a group share one h and
        # the outcome is credited to that group, not to whatever h the state holds later.
        group_id = None
        if call_unit:
            positions, turn_starts = call_positions(reference_turns)
            if curriculum_enabled:
                group_id, position = assign(label["task_id"], turn_starts[h_init], len(positions))
            elif "guide_position" in label:
                position = int(label["guide_position"])
            else:
                position = turn_starts[h_init if requested_guide_turns < 0
                                      else min(requested_guide_turns, num_turns - 1)]
            guide_turns, guide_calls = positions[position]
        else:
            positions = turn_starts = None
            position = guide_calls = 0
            if curriculum_enabled:
                group_id, guide_turns = assign(label["task_id"], h_init, num_turns)
            else:
                guide_turns = h_init if requested_guide_turns < 0 else min(requested_guide_turns, num_turns)
        client = AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)
        trace_started = datetime.now(timezone.utc).isoformat()
        trace_id = f"{os.getpid()}:{time.monotonic_ns()}"
        steps = 0
        student_response_tokens = 0
        dapo_enabled = os.environ.get("BFCL_DAPO", "0") == "1"
        # OPSD capture: first student action of each turn plus the context it
        # was generated from, and the latest turn-check evidence. On an
        # official-stop failure these become the boundary-distillation record.
        turn_first = {}
        last_evidence = {}
        cur_turn = guide_turns

        handoff_prompt_chars = 0        # context handed to the student (system + guide roll-in), characters
        handoff_prompt_tokens = None    # the same context as the server tokenized it (first student call)
        last_finish_reason = None       # vLLM finish_reason of the student's last response

        def result(success, *, terminated, truncated=False, turn=-1, reason=None):
            if not curriculum_enabled:
                next_h = position if call_unit else guide_turns
            elif call_unit:
                next_h = record(label["task_id"], success, len(positions), turn_starts,
                                group_id=group_id, rollout_id=trace_id)
            else:
                next_h = record(label["task_id"], success, num_turns, group_id=group_id, rollout_id=trace_id)
            h_used = position if call_unit else guide_turns
            trace_dir = os.environ.get("BFCL_TRACE_DIR")
            if os.environ.get("BFCL_TRACE_ROLLOUTS") == "1" and trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                version_file = os.environ.get("EXP2_VERSION_FILE")
                try:
                    with open(version_file) as stream:
                        policy_version = stream.read().strip()
                except (OSError, TypeError):
                    policy_version = None
                with open(os.path.join(trace_dir, f"rollouts_{os.getpid()}.jsonl"), "a") as stream:
                    stream.write(json.dumps(dict(trace_id=trace_id, task_id=label["task_id"], group_id=group_id,
                        started_at=trace_started, completed_at=datetime.now(timezone.utc).isoformat(),
                        policy_version=policy_version, curriculum_enabled=curriculum_enabled,
                        h_used=h_used, h_next_at_completion=next_h, num_turns=num_turns,
                        handoff_unit="call" if call_unit else "turn", guide_turns=guide_turns,
                        guide_calls_in_turn=guide_calls,
                        success=bool(success), terminated=terminated, truncated=truncated,
                        # Why the episode ended: the environment finished (terminated), the step cap
                        # (max_steps), or a guide roll-in that ended it before the student acted (guide).
                        # A context overflow surfaces as a server error, drops the rollout and never
                        # reaches here (its group then closes as incomplete); last_finish_reason tells a
                        # generation cut by max_tokens ("length") from a natural stop.
                        termination=reason or ("terminated" if terminated else "max_steps"),
                        last_finish_reason=last_finish_reason,
                        handoff_prompt_chars=handoff_prompt_chars, handoff_prompt_tokens=handoff_prompt_tokens,
                        student_response_tokens=student_response_tokens, tool_steps=steps)) + "\n")
            penalty = 0.0
            if dapo_enabled:
                penalty = overlong_reward(
                    student_response_tokens,
                    max_response_tokens=int(os.environ["BFCL_DAPO_MAX_RESPONSE_TOKENS"]),
                    buffer_tokens=int(os.environ.get("BFCL_DAPO_OVERLONG_BUFFER_TOKENS", "2048")),
                    penalty_factor=float(os.environ.get("BFCL_DAPO_OVERLONG_PENALTY_FACTOR", "1.0")),
                )
            # OPSD banks: raw side keeps ground-truth-bearing evidence for
            # analysis; the teacher bank gets field NAMES only (see banks.py).
            # uid rides as two float32-safe halves and is nonzero only when a
            # boundary action was captured for this failure.
            opsd_hi = opsd_lo = 0.0
            if (os.environ.get("EXP2_BANK_DIR") and curriculum_enabled
                    and terminated and not success):
                fail_turn = int(last_evidence.get("turn", -1))
                first = turn_first.get(fail_turn)
                uid = banks.new_uid()
                meta = dict(uid=uid, task_id=label["task_id"],
                            h_used=position if call_unit else guide_turns,
                            turn=fail_turn,
                            policy_version=banks.policy_version())
                banks.append("failures_raw.jsonl", {**meta,
                    "state_differences": list(last_evidence.get("state_differences", [])),
                    "missing_responses": list(last_evidence.get("missing_responses") or [])})
                if first:
                    hi, lo = banks.uid_halves(uid)
                    opsd_hi, opsd_lo = float(hi), float(lo)
                    banks.append("opsd_bank.jsonl", {**meta,
                        "certificate": {
                            "error_fields": list(last_evidence.get("state_differences", [])),
                            "missing_response_count": len(last_evidence.get("missing_responses") or []),
                        },
                        "boundary_action": first[0],
                        "context_messages": first[1]})
            # score is the unshaped verifier result used by DAPO group filtering;
            # reward is the shaped scalar used by the policy-gradient estimator.
            return Result(
                reward=float(success) + penalty, score=float(success),
                terminated=terminated, truncated=truncated,
                info={"bfcl_strict_success": float(success),
                      "opsd_uid_hi": opsd_hi, "opsd_uid_lo": opsd_lo,
                      "bfcl_steps": float(steps),
                      "bfcl_guide_turns": float(guide_turns),
                      "bfcl_next_guide_turns": float(next_h),
                      "bfcl_guide_calls_in_turn": float(guide_calls),
                      "bfcl_guide_position": float(h_used),
                      "bfcl_handoff_changed": float(next_h != h_used),
                      "bfcl_autonomous_train": float(h_used == 0),
                      "bfcl_turn": float(turn),
                      "dapo_response_tokens": float(student_response_tokens),
                      "dapo_overlong_reward": float(penalty)},
            )
        try:
            # Reference roll-in is context only: none of these guide tokens are
            # sampled by, or trained as actions of, the student.
            for turn in range(guide_turns):
                calls = list(reference_turns[turn])
                if calls:
                    raw, action = _guide_action(calls, reset.handle.tool_documents)
                    messages.append({"role": "assistant", "content": raw})
                    step = runtime.execute(reset.handle, action)
                    messages.extend(step.observation_messages)
                    steps += 1
                messages.append({"role": "assistant", "content": ""})
                step = runtime.execute(
                    reset.handle,
                    StructuredAction(
                        raw_text="", tool_calls=[], parse_ok=True,
                        parse_kind="END_TURN",
                    ),
                )
                messages.extend(step.observation_messages)
                steps += 1
                if step.system_prompt_update:
                    messages[0]["content"] = step.system_prompt_update
                if step.terminated:
                    success = not reset.handle.official_stop_triggered
                    return result(success, terminated=True, turn=step.turn, reason="guide")
            if guide_calls:
                # Mid-turn handoff: the first calls of this turn are context; the
                # student continues the same turn (no END_TURN is sent).
                calls = list(reference_turns[guide_turns])[:guide_calls]
                raw, action = _guide_action(calls, reset.handle.tool_documents)
                messages.append({"role": "assistant", "content": raw})
                step = runtime.execute(reset.handle, action)
                messages.extend(step.observation_messages)
                steps += 1
                if step.terminated:
                    raise RuntimeError(f"Guide terminated inside turn {guide_turns}: {label['task_id']}")
            handoff_prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
            while steps < reset.max_steps:
                response = await client.chat.completions.create(
                    model=ctx.model_name,
                    messages=messages,
                    max_tokens=ctx.sampling_params.max_tokens,
                    temperature=ctx.sampling_params.temperature,
                    top_p=ctx.sampling_params.top_p,
                )
                if response.usage is None or response.usage.completion_tokens is None:
                    if dapo_enabled:
                        raise RuntimeError("DAPO requires completion_tokens from the rollout server")
                else:
                    student_response_tokens += int(response.usage.completion_tokens)
                    if handoff_prompt_tokens is None and response.usage.prompt_tokens is not None:
                        handoff_prompt_tokens = int(response.usage.prompt_tokens)
                last_finish_reason = getattr(response.choices[0], "finish_reason", None)
                raw = response.choices[0].message.content or ""
                if cur_turn not in turn_first:
                    turn_first[cur_turn] = (raw, [dict(m) for m in messages])
                messages.append({"role": "assistant", "content": raw})
                step = runtime.execute(reset.handle, runtime.parse_action(raw))
                steps += 1
                if step.evaluator_evidence:
                    last_evidence = step.evaluator_evidence
                if step.terminated:
                    success = not reset.handle.official_stop_triggered
                    return result(success, terminated=True, turn=step.turn)
                messages.extend(step.observation_messages)
                cur_turn = step.turn
            return result(False, terminated=False, truncated=True, reason="max_steps")
        finally:
            runtime.close(reset.handle)


class AgentRunner(ChatAgentRunner):
    def __init__(self):
        super().__init__(BFCLAgent)
