#!/usr/bin/env python
"""Aggregate four official BFCL shards for each decode without changing scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfcl_env.manifest import RunManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--decodes", type=int, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--per-request-max-tokens", type=int, default=4096)
    parser.add_argument("--backend-python", required=True, type=Path)
    parser.add_argument("--comparability", default="local-deterministic-split")
    parser.add_argument("--bfcl-commit", default="")
    parser.add_argument("--bfcl-source-root", type=Path)
    parser.add_argument("--shards-per-gpu", type=int, default=1)
    args = parser.parse_args()
    decode_rows = []
    all_validation = []
    aggregate_files = {
        "rollouts.jsonl": [], "prompt_differential.jsonl": [],
        "task_verdict_differential.jsonl": [], "errors.jsonl": [],
    }
    official_result_root = args.run_dir / "official_result"
    official_score_root = args.run_dir / "official_score"
    official_result_root.mkdir(exist_ok=True)
    official_score_root.mkdir(exist_ok=True)
    for decode in range(args.decodes):
        summaries = []
        seen = set()
        correct = scored = generated = 0
        for shard in sorted((args.run_dir / f"decode_{decode}").glob("shard_*")):
            summary = json.loads((shard / "summary.json").read_text(encoding="utf-8"))
            summaries.append(summary)
            generated += summary["number_generated"]
            scored += summary["number_officially_scored"]
            correct += summary["correct_count"]
            for line in (shard / "validation_rows.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["task_id"] in seen:
                    raise SystemExit(f"decode {decode}: duplicate {row['task_id']}")
                seen.add(row["task_id"])
                row["decode"] = decode
                all_validation.append(row)
            for filename in aggregate_files:
                path = shard / filename
                if path.exists():
                    aggregate_files[filename].append(path.read_text(encoding="utf-8"))
            for source_name, root in (
                ("official_result", official_result_root),
                ("official_score", official_score_root),
            ):
                target_parent = root / f"decode_{decode}"
                target_parent.mkdir(exist_ok=True)
                link = target_parent / shard.name
                if not link.exists():
                    link.symlink_to((shard / source_name).resolve(), target_is_directory=True)
        if len(seen) != args.expected_tasks or generated != args.expected_tasks:
            raise SystemExit(
                f"decode {decode}: expected {args.expected_tasks}, generated={generated}, unique={len(seen)}"
            )
        decode_rows.append({
            "decode": decode, "number_requested": args.expected_tasks,
            "number_generated": generated, "number_officially_scored": scored,
            "correct_count": correct,
            "strict_official_session_success": correct / scored if scored else None,
        })
    rates = [row["strict_official_session_success"] for row in decode_rows]
    task_scores = {}
    for row in all_validation:
        task_scores.setdefault(row["task_id"], []).append(float(row["official_success"]))
    task_macro = [statistics.fmean(values) for values in task_scores.values()]
    rng = random.Random(20260904)
    bootstrap = sorted(
        statistics.fmean(rng.choices(task_macro, k=len(task_macro)))
        for _ in range(10000)
    )
    ci95 = [bootstrap[249], bootstrap[9749]]
    # Utilization compliance is evaluated only inside the explicit generation
    # intervals. Model load and drain samples are retained in the CSV but excluded.
    intervals = []
    for line in (args.run_dir / "events.csv").read_text().splitlines():
        stamp, decode_name, event = line.split(",")
        if event == "generation_start":
            intervals.append([datetime.fromisoformat(stamp), None, decode_name])
        elif event == "generation_end":
            next(item for item in reversed(intervals) if item[2] == decode_name and item[1] is None)[1] = datetime.fromisoformat(stamp)
    gpu_samples = {gpu: [] for gpu in range(4)}
    with (args.run_dir / "gpu_metrics.csv").open() as handle:
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row["timestamp"])
            if any(start <= stamp <= end for start, end, _ in intervals if end is not None):
                gpu = int(row["gpu"])
                gpu_samples[gpu].append({
                    "memory_percent": 100 * int(row["memory_used_mib"]) / int(row["memory_total_mib"]),
                    "utilization_percent": int(row["utilization_percent"]),
                })
    gpu_compliance = {}
    for gpu, samples in gpu_samples.items():
        gpu_compliance[str(gpu)] = {
            "steady_state_samples": len(samples),
            "mean_memory_percent": statistics.fmean(s["memory_percent"] for s in samples) if samples else None,
            "mean_utilization_percent": statistics.fmean(s["utilization_percent"] for s in samples) if samples else None,
            "fraction_both_at_least_80_percent": (
                sum(s["memory_percent"] >= 80 and s["utilization_percent"] >= 80 for s in samples) / len(samples)
                if samples else None
            ),
        }

    summary = {
        "comparability": args.comparability,
        "track": (
            "bfcl-official-as-is"
            if args.comparability != "generation-cap-diagnostic"
            else "bfcl-official-generation-cap-diagnostic"
        ),
        "aggregation": {
            "per_decode": "strict successes / explicitly requested tasks",
            "mean": "arithmetic mean of the three decode rates",
            "pooled": "sum strict successes / sum officially scored tasks",
        },
        "decodes": decode_rows,
        "mean_strict_official_session_success": statistics.fmean(rates),
        "task_macro_success": statistics.fmean(task_macro),
        "task_bootstrap_95ci": ci95,
        "task_bootstrap_replicates": 10000,
        "pooled_strict_official_session_success": (
            sum(row["correct_count"] for row in decode_rows)
            / sum(row["number_officially_scored"] for row in decode_rows)
        ),
        "gpu_steady_state_compliance": gpu_compliance,
    }
    for filename, chunks in aggregate_files.items():
        (args.run_dir / filename).write_text("".join(chunks), encoding="utf-8")
    (args.run_dir / "validation_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_validation),
        encoding="utf-8",
    )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    task_payload = (args.run_dir / "task_ids.json").read_bytes()
    def version(package: str) -> str | None:
        result = subprocess.run([
            str(args.backend_python), "-c",
            f"from importlib.metadata import version; print(version({package!r}))",
        ], capture_output=True, text=True, check=False)
        return result.stdout.strip() or None
    RunManifest(
        run=f"bfcl-official-pool:{args.category}", model_path=args.model_path,
        config={
            "category": args.category, "decodes": args.decodes,
            "number_tasks_per_decode": args.expected_tasks,
            "temperature_nominal": args.temperature,
            "temperature_effective": max(args.temperature, 0.01) if args.temperature > 0 else 0,
            "per_request_max_tokens": args.per_request_max_tokens,
            "top_k": None, "top_p": None,
            "task_ids_sha256": hashlib.sha256(task_payload).hexdigest(),
            "server_generation_config": "vllm",
            "independent_client_shards_per_gpu": args.shards_per_gpu,
        },
        extra={
            "comparability": args.comparability,
            "bfcl_commit": args.bfcl_commit,
            "bfcl_source_root": str(args.bfcl_source_root) if args.bfcl_source_root else None,
            "backend": {name: version(name) for name in ("vllm", "torch", "transformers")},
        },
    ).write(args.run_dir / "manifest.json", repo_root=ROOT)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
