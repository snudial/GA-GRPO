"""Fresh-process BFCL operations for simulators with leaking module state."""

from __future__ import annotations

import pickle
import sys
import uuid


def _ground_truth(task):
    from bfcl_env.adapter import (
        GroundTruthTrace,
        TurnSnapshot,
        snapshot_instances,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    key = f"opsd_isolated_gt_{uuid.uuid4().hex}"

    def execute(calls):
        return execute_multi_turn_func_call(
            list(calls), task.initial_config, task.involved_classes, key,
            task.task_id, long_context=task.long_context, is_evaL_run=False,
        )

    _, instances = execute([])
    trace = GroundTruthTrace(
        task_id=task.task_id, initial_state=snapshot_instances(instances)
    )
    for turn in range(task.num_turns):
        calls = task.ground_truth_calls(turn)
        responses, instances = execute(calls)
        trace.turns.append(
            TurnSnapshot(
                turn=turn, state=snapshot_instances(instances),
                responses=list(responses), calls=list(calls),
            )
        )
    return trace


def _official(payload):
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )

    return multi_turn_checker(
        payload["model_result"], payload["reference"], payload["entry"],
        payload["category"], f"opsd_isolated_score_{uuid.uuid4().hex}",
    )


def main() -> int:
    payload = pickle.loads(sys.stdin.buffer.read())
    if sys.argv[1] == "ground_truth":
        result = _ground_truth(payload)
    elif sys.argv[1] == "official":
        result = _official(payload)
    else:
        raise SystemExit(f"unknown mode: {sys.argv[1]}")
    sys.stdout.buffer.write(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
