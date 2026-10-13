#!/usr/bin/env python3
"""Build the val25 checkpoint-selection record for one BFCL training run."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GLOBAL_STEP = re.compile(r"(?:Global step |Generation progress at step )(\d+):\s*(\{.*\})")
EVAL_STEP = re.compile(r"Eval at step (\d+):\s*(\{.*\})")
EVAL_ONLY = re.compile(r"Eval-only completed:\s*(\{.*\})")


def metric_value(metrics: dict[str, object]) -> tuple[str, float]:
    keys = sorted(key for key in metrics if key.startswith("eval_") and key.endswith("_pass1"))
    if not keys:
        raise ValueError(f"no eval_*_pass1 metric in {sorted(metrics)}")
    key = keys[0]
    return key, float(metrics[key])


def parse_dict(raw: str) -> dict[str, object]:
    # A step dict can carry non-literal float reprs (an OPSD step once logged
    # 'opsd/kl': nan), which ast.literal_eval rejects. Neutralize them; the
    # summary only reads eval_* keys, which are always finite.
    raw = re.sub(r"\b(nan|inf|-inf)\b", "None", raw)
    value = ast.literal_eval(raw)
    if not isinstance(value, dict):
        raise ValueError("log metric payload is not a dictionary")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--final-log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--final-checkpoint", type=Path,
                        help="Explicit final checkpoint for non-80k pilots")
    args = parser.parse_args()

    train_log = args.run / "logs" / "train.log"
    step_budget: dict[int, dict[str, int]] = {}
    validations: dict[int, dict[str, object]] = {}
    for raw_line in train_log.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        if match := GLOBAL_STEP.search(line):
            step = int(match.group(1))
            metrics = parse_dict(match.group(2))
            groups = int(float(metrics.get("rollout/total_prompt_groups_sampled", 0)))
            step_budget[step] = {
                "prompt_groups": groups,
                "generated_rollouts": groups * args.n_samples,
            }
        if match := EVAL_STEP.search(line):
            step = int(match.group(1))
            if step == 0:
                continue
            metrics = parse_dict(match.group(2))
            key, score = metric_value(metrics)
            validations[step] = {
                "optimizer_step": step,
                "metric_key": key,
                "val_avg_at_3": score,
                **step_budget.get(step, {"prompt_groups": None, "generated_rollouts": None}),
                "checkpoint": str(args.run / "state" / "_hf" / f"best_global_step{step}"),
                "source": "in_training",
            }

    final_metrics = None
    for raw_line in args.final_log.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        if match := EVAL_ONLY.search(line):
            final_metrics = parse_dict(match.group(1))
    if final_metrics is None:
        raise SystemExit(f"missing 'Eval-only completed' result in {args.final_log}")
    if not step_budget:
        raise SystemExit(f"no Global step metrics in {train_log}")

    final_step = max(step_budget)
    key, score = metric_value(final_metrics)
    final_checkpoint = args.final_checkpoint or args.run / "hf_ep50"
    final_budget = dict(step_budget[final_step])
    # The optimizer log stops at the last update; an all-filtered tail (groups that were
    # sampled but produced no mixed group) still consumes budget. The boundary marker is the
    # committed counter (run_seed_rollouts verifies sampled_groups == expected_groups before
    # writing it), so the final entry reports that when the final checkpoint is a boundary.
    if match := re.search(r"hf_ep(\d+)$", str(final_checkpoint)):
        marker = args.run / "boundaries" / f"ep{match.group(1)}.complete"
        if marker.exists():
            fields = dict(line.split("=", 1) for line in marker.read_text().split() if "=" in line)
            if "prompt_groups" in fields and "generated_rollouts" in fields:
                final_budget = {
                    "prompt_groups": int(fields["prompt_groups"]),
                    "generated_rollouts": int(fields["generated_rollouts"]),
                    "last_update_prompt_groups": step_budget[final_step]["prompt_groups"],
                    "budget_source": "boundary_marker",
                }
    validations[final_step] = {
        "optimizer_step": final_step,
        "metric_key": key,
        "val_avg_at_3": score,
        **final_budget,
        "checkpoint": str(final_checkpoint),
        "source": "final_eval_only",
    }
    ordered = sorted(validations.values(), key=lambda row: int(row["optimizer_step"]))
    if not ordered:
        raise SystemExit("no eligible validation checkpoints")
    best = min(ordered, key=lambda row: (-float(row["val_avg_at_3"]), int(row["optimizer_step"])))
    output = {
        "selection_split": "bfcl-derived author-lineage val25",
        "selection_rule": "maximum val Avg@3; ties use earliest optimizer step",
        "best": best,
        "final": validations[final_step],
        "validations": ordered,
    }
    target = args.output or args.run / "validation_summary.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
