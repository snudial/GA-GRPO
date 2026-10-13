"""Verify the source snapshot used by the training engine."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(molt=None):
    lock_path = ROOT/'third_party/molt.lock.json'
    lock = json.loads(lock_path.read_text())
    molt = Path(molt or ROOT/'third_party/molt')
    for name, expected in lock['files'].items():
        source = molt/name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Bundled Molt source differs: {source}')
    return lock


if __name__ == '__main__':
    lock = verify()
    print(f"Bundled Molt verified: {lock['source_commit']}")
