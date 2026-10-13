"""Convert pinned BFCL task manifests to Molt JSONL rows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root
SPLIT = ROOT / "config/splits/bfcl_aworld_lineage_seed42_v1"
OUT = ROOT / "data" / "bfcl"
OUT.mkdir(parents=True, exist_ok=True)

for source, destination, limit, guide_turns in (
    ("base_train.json", "bfcl_train.jsonl", None, 0),
    ("base_val.json", "bfcl_val.jsonl", None, 0),
    ("base_test75.json", "bfcl_test75.jsonl", None, 0),
    ("base_eval.json", "bfcl_eval.jsonl", None, 0),
    ("base_train.json", "bfcl_smoke8.jsonl", 8, 0),
    ("base_train.json", "bfcl_jumpstart_train.jsonl", None, -1),
    ("base_train.json", "bfcl_jumpstart_smoke8.jsonl", 8, -1),
):
    task_ids = json.loads((SPLIT / source).read_text())[:limit]
    rows = [
        {"input": [{"role": "user", "content": task_id}],
         "label": json.dumps({"task_id": task_id,
                              "category": "multi_turn_base", "seed": 42,
                              "guide_turns": guide_turns})}
        for task_id in task_ids
    ]
    (OUT / destination).write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
