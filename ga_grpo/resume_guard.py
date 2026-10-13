"""Fail-closed generated-budget recovery checks, usable before allocating GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


class UnsafeResume(ValueError):
    pass


def validate_state(state: dict, *, dataset_groups: int, target_groups: int,
                   curriculum: bool = False) -> str:
    if dataset_groups <= 0 or target_groups <= 0:
        raise UnsafeResume("budget and dataset size must be positive")
    for key in ("episode", "global_step", "total_consumed_prompts", "total_trained_groups"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise UnsafeResume(f"missing/invalid checkpoint counter: {key}")
    consumed = state["total_consumed_prompts"]
    if consumed > target_groups:
        raise UnsafeResume(f"generated budget mismatch: {consumed} > {target_groups}; do not retry training")
    if "data_loader_state_dict" not in state:
        raise UnsafeResume("missing dataloader state")
    if state["episode"] != consumed // dataset_groups:
        raise UnsafeResume("episode/cumulative-count mismatch; legacy cursor requires explicit recovery")
    generator = state.get("rollout_generator_state_dict", {})
    if generator.get("pending_rollouts", 0) or generator.get("pending_finished_samples", 0):
        raise UnsafeResume("unsaved rollout pool: exact mid-batch resume is not supported; use a drained boundary")
    if "round_exhausted" not in generator:
        raise UnsafeResume("legacy checkpoint has no proof of a drained rollout pool")
    if curriculum and "curriculum_state" not in generator:
        raise UnsafeResume("Jump checkpoint lacks aligned curriculum state")
    if consumed == target_groups:
        if not generator["round_exhausted"]:
            raise UnsafeResume("budget spent but rollout pool has not finished draining")
        return "complete"
    return "resume"


def load_latest(run: Path) -> tuple[Path, dict]:
    import torch

    actor = run / "state" / "_actor"
    latest = actor / "latest"
    if not latest.is_file():
        raise UnsafeResume("missing latest checkpoint marker; refuse silent fresh start")
    tag = latest.read_text().strip()
    if not tag.startswith("global_step") or not tag[len("global_step"):].isdigit():
        raise UnsafeResume(f"invalid latest tag: {tag!r}")
    checkpoint = actor / tag
    if not (checkpoint / "model").is_dir() or not (checkpoint / "optim").is_dir():
        raise UnsafeResume("incomplete model/optimizer checkpoint")
    extra = checkpoint / "extra_state.pt"
    if not extra.is_file():
        raise UnsafeResume("incomplete checkpoint: missing extra_state.pt")
    state = torch.load(extra, map_location="cpu", weights_only=False)["client_state"]
    if state.get("global_step") != int(tag[len("global_step"):]):
        raise UnsafeResume("checkpoint directory and optimizer step disagree")
    return checkpoint, state


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--dataset-groups", type=int, required=True)
    p.add_argument("--target-groups", type=int, required=True)
    p.add_argument("--curriculum", action="store_true")
    args = p.parse_args()
    try:
        checkpoint, state = load_latest(args.run)
        action = validate_state(state, dataset_groups=args.dataset_groups,
                                target_groups=args.target_groups, curriculum=args.curriculum)
        print(json.dumps({"action": action, "checkpoint": str(checkpoint),
                          "global_step": state["global_step"],
                          "sampled_groups": state["total_consumed_prompts"]}))
    except (UnsafeResume, OSError, KeyError, ValueError) as exc:
        p.exit(78, f"NON_RETRYABLE_RESUME: {exc}\n")


if __name__ == "__main__":
    main()
