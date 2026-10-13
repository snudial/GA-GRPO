#!/usr/bin/env python
"""Write BFCL v1.3's category-keyed run-ID file from an ordered ID list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True, type=Path)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.ids.read_text(encoding="utf-8"))
    ids = payload.get("task_ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise SystemExit("expected a JSON string list")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({args.category: ids}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
