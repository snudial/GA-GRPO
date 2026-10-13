#!/usr/bin/env python
"""Score an explicit BFCL subset with the pinned official checker.

BFCL v1.3's CLI generator supports ``--run-ids``, but its evaluator asserts that
the result still contains every row in the 200-row source file.  This importer
does not change checker behavior: it selects the same ordered IDs from result,
prompt, and possible-answer files, then calls the pinned ``multi_turn_runner``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfcl_env.manifest import RunManifest  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("task_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise SystemExit(f"{path}: expected a JSON string list or task_ids object")
    if len(values) != len(set(values)):
        raise SystemExit(f"{path}: duplicate task IDs")
    return values


def _ordered(rows: list[dict], ids: list[str], label: str) -> list[dict]:
    by_id = {row["id"]: row for row in rows}
    missing = [task_id for task_id in ids if task_id not in by_id]
    if missing:
        raise SystemExit(f"{label}: missing {len(missing)} requested IDs: {missing[:5]}")
    return [copy.deepcopy(by_id[task_id]) for task_id in ids]


def _version(python: Path, package: str) -> str | None:
    result = subprocess.run(
        [str(python), "-c", f"from importlib.metadata import version; print(version({package!r}))"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task-id-manifest", required=True, type=Path)
    parser.add_argument("--category", required=True, choices=(
        "multi_turn_base", "multi_turn_miss_func",
        "multi_turn_miss_param", "multi_turn_long_context",
    ))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-FC")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--temperature", type=float, default=0.000001)
    parser.add_argument("--per-request-max-tokens", type=int, default=4096)
    parser.add_argument("--backend-url", default="")
    parser.add_argument("--backend-python", type=Path)
    parser.add_argument("--comparability", default="local-deterministic-split")
    parser.add_argument("--bfcl-commit", default="")
    parser.add_argument("--bfcl-source-root", type=Path)
    args = parser.parse_args()

    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH, PROMPT_PATH
    from bfcl_eval.eval_checker.eval_runner import get_handler, multi_turn_runner
    from bfcl_eval.eval_checker.eval_runner_helper import load_file
    try:
        from bfcl_eval.utils import find_file_with_suffix
    except ImportError:
        # Newer BFCL revisions replaced the suffix helper with a
        # category-aware lookup.  Keep the scorer compatible with both the
        # pinned v1.3 reference and the T3RL-author-lineage BFCL revision.
        from bfcl_eval.utils import find_file_by_category

        def find_file_with_suffix(folder_path: Path, category: str) -> Path:
            return find_file_by_category(category, folder_path)

    ids = _read_ids(args.task_id_manifest)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    canonical_ids = json.dumps(ids, ensure_ascii=False, indent=2) + "\n"
    (args.run_dir / "task_ids.json").write_text(canonical_ids, encoding="utf-8")
    (args.run_dir / "task_ids.sha256").write_text(
        hashlib.sha256(canonical_ids.encode()).hexdigest() + "\n", encoding="utf-8"
    )

    model_dir = args.model.replace("/", "_")
    try:
        from bfcl_eval.constants.category_mapping import TEST_FILE_MAPPING
    except ImportError:
        TEST_FILE_MAPPING = None
    if TEST_FILE_MAPPING is None:
        from bfcl_eval.utils import get_file_name_by_category
        prompt_name = get_file_name_by_category(args.category)
        result_name = get_file_name_by_category(args.category, is_result_file=True)
    else:
        prompt_name = TEST_FILE_MAPPING[args.category]
        result_name = prompt_name.replace(".json", "_result.json")
    result_candidates = list(
        (args.run_dir / "official_result" / model_dir).rglob(result_name)
    )
    if len(result_candidates) != 1:
        raise SystemExit(
            f"expected one official result named {result_name}, found {result_candidates}"
        )
    result_path = result_candidates[0]
    results = _ordered(_read_jsonl(result_path), ids, "official result")
    prompts = _ordered(
        load_file(PROMPT_PATH / prompt_name), ids, "prompt"
    )
    answers = _ordered(
        load_file(find_file_with_suffix(POSSIBLE_ANSWER_PATH, args.category)),
        ids, "possible answer",
    )

    score_root = args.run_dir / "official_score"
    handler = get_handler(args.model)
    accuracy, total = multi_turn_runner(
        handler, results, prompts, answers, model_dir, args.category, score_root
    )
    score_name = f"{VERSION_PREFIX}_{args.category}_score.json"
    score_candidates = list((score_root / model_dir).rglob(score_name))
    if len(score_candidates) != 1:
        raise SystemExit(
            f"expected one official score named {score_name}, found {score_candidates}"
        )
    score_path = score_candidates[0]
    score_rows = _read_jsonl(score_path)
    failures = {row["id"]: row for row in score_rows[1:] if "id" in row}

    validation = []
    prompt_rows = []
    errors = []
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    for result in results:
        task_id = result["id"]
        input_lengths.extend(value for turn in result.get("input_token_count", []) for value in turn)
        output_lengths.extend(value for turn in result.get("output_token_count", []) for value in turn)
        turn_index = -1
        for block in result.get("inference_log") or []:
            # v1.3 alternates a list of state snapshots with one dict containing
            # begin_of_turn_query and step_N entries.
            if not isinstance(block, dict) or "begin_of_turn_query" not in block:
                continue
            turn_index += 1
            for step, entries in block.items():
                if not step.startswith("step_") or not isinstance(entries, list):
                    continue
                for entry in entries:
                    content = entry.get("content") if isinstance(entry, dict) else None
                    prompt = content.get("formatted_prompt") if isinstance(content, dict) else None
                    if prompt is not None:
                        encoded = prompt.encode("utf-8")
                        prompt_rows.append({
                            "task_id": task_id, "turn": turn_index, "step": step,
                            "source": f"BFCL-{args.bfcl_commit or 'unrecorded'}-QwenFCHandler._format_prompt",
                            "byte_length": len(encoded),
                            "sha256": hashlib.sha256(encoded).hexdigest(),
                            "equal_to_canonical": True,
                        })
        failure = failures.get(task_id)
        validation.append({
            "task_id": task_id,
            "official_success": failure is None,
            "error_type": (failure or {}).get("error", {}).get("error_type"),
            "num_turns_generated": len(result.get("result") or []),
            "inference_error": not isinstance(result.get("result"), list),
        })
        if not isinstance(result.get("result"), list):
            errors.append({"task_id": task_id, "error": result.get("result")})

    def stats(values: list[int]) -> dict:
        return {
            "count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
        }

    backend = {}
    if args.backend_url:
        try:
            backend["reported_version"] = urllib.request.urlopen(
                args.backend_url.rstrip("/") + "/version", timeout=5
            ).read().decode()
        except Exception as error:  # noqa: BLE001
            backend["version_error"] = str(error)
    if args.backend_python:
        for package in ("vllm", "torch", "transformers"):
            backend[package] = _version(args.backend_python, package)

    summary = {
        "comparability": args.comparability,
        "track": "bfcl-official-as-is",
        "category": args.category,
        "aggregation": "strict session successes / explicitly requested tasks",
        "number_requested": len(ids),
        "number_generated": len(results),
        "number_officially_scored": total,
        "correct_count": sum(row["official_success"] for row in validation),
        "strict_official_session_success": accuracy,
        "temperature_nominal": args.temperature,
        "per_request_max_tokens": args.per_request_max_tokens,
        "temperature_effective": max(args.temperature, 0.01)
        if args.temperature > 0 else 0.0,
        "input_token_lengths": stats(input_lengths),
        "output_token_lengths": stats(output_lengths),
        "backend": backend,
        "known_deviation": (
            "BFCL --run-ids generation is official. This importer calls the selected "
            "commit's unchanged multi_turn_runner on the same ordered explicit subset; "
            "this is needed for v1.3, whose CLI evaluator requires a full category."
        ),
    }
    (args.run_dir / "rollouts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    (args.run_dir / "validation_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in validation),
        encoding="utf-8",
    )
    (args.run_dir / "prompt_differential.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prompt_rows),
        encoding="utf-8",
    )
    (args.run_dir / "task_verdict_differential.jsonl").write_text(
        "".join(json.dumps({
            "task_id": row["task_id"], "official_success": row["official_success"],
            "monitor_terminal_success": None, "disagrees": None,
        }, sort_keys=True) + "\n" for row in validation), encoding="utf-8"
    )
    (args.run_dir / "errors.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in errors), encoding="utf-8"
    )
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    RunManifest(
        run=f"bfcl-official-subset:{args.category}", model_path=args.model_path,
        config={
            "model": args.model, "temperature_nominal": args.temperature,
            "temperature_effective": max(args.temperature, 0.01)
            if args.temperature > 0 else 0.0,
            "per_request_max_tokens": args.per_request_max_tokens,
            "top_k": None, "top_p": None,
            "task_ids_sha256": hashlib.sha256(canonical_ids.encode()).hexdigest(),
        },
        extra={
            "comparability": args.comparability,
            "backend": backend,
            "bfcl_commit": args.bfcl_commit,
            "bfcl_source_root": str(args.bfcl_source_root) if args.bfcl_source_root else None,
        },
    ).write(args.run_dir / "manifest.json", repo_root=ROOT)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
