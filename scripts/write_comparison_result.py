#!/usr/bin/env python3
"""Join validation selection and held-out test75 scores into one run record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--best-official", type=Path, required=True)
    parser.add_argument("--final-official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=80000,
                        help="the contract's generated-rollout budget; an extension passes its own and is marked")
    args = parser.parse_args()
    validation = json.loads((args.run / "validation_summary.json").read_text())
    best_test = json.loads(args.best_official.read_text())
    final_test = json.loads(args.final_official.read_text())
    final_rollouts = validation["final"]["generated_rollouts"]
    if final_rollouts != args.expected_rollouts:
        raise SystemExit(f"Refuse comparison: final generated rollouts={final_rollouts}, expected {args.expected_rollouts}")
    payload = {
        "method": args.method,
        # 80000 is the contract (contracts/bfcl.json); anything else is a budget extension,
        # comparable only with runs of the same budget.
        "protocol_extension": final_rollouts != 80000,
        "train_seed": args.seed,
        "evaluation_set": "bfcl-derived author-lineage held-out test75",
        "best_optimizer_step": validation["best"]["optimizer_step"],
        "best_step_generated_rollouts": validation["best"]["generated_rollouts"],
        "best_val_avg_at_3": validation["best"]["val_avg_at_3"],
        "best_test_avg_at_3": best_test["mean_strict_official_session_success"],
        "final_generated_rollouts": final_rollouts,
        "final_test_avg_at_3": final_test["mean_strict_official_session_success"],
        "best_test_summary": str(args.best_official.resolve()),
        "final_test_summary": str(args.final_official.resolve()),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
