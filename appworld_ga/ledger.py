"""Training ledger: append-only records written while an AppWorld run trains.

Enabled by APPWORLD_LEDGER_DIR (scripts/run_appworld.py sets it for training phases).
Every record carries the restart segment (APPWORLD_SEGMENT_ID) and a wall-clock time.
Each process appends to its own files, <kind>.<pid>.jsonl (turn traces:
turns.<pid>.jsonl.gz, one gzip member per rollout), so no writer ever locks.

Logging never touches the training RNGs (ids come from uuid4/os.urandom in the
trainer, never from random/numpy/torch) and never changes what is sampled, scored
or optimized. Schema and file map: README.md.

  segments.jsonl                  driver: segment start/end, the checkpoint it resumed from
  groups.<pid>.jsonl              rollout runner: one row per 16-rollout training group
  rollouts.<pid>.jsonl            rollout runner: one row per training rollout
  turns.<pid>.jsonl.gz            rollout runner: per student turn code/output/hashes
  filter.<pid>.jsonl              sample generator: the trainer's keep/drop decision per group
  batch.<pid>.jsonl               trainer: samples per collector cycle, and samples it dropped
  updates.rank<r>.<pid>.jsonl     policy actor rank r: rollouts in each AdamW update
"""
import gzip
import json
import os
import time


def enabled():
    return bool(os.environ.get('APPWORLD_LEDGER_DIR'))


def stamp(record):
    return {'segment_id': os.environ.get('APPWORLD_SEGMENT_ID'), 'time': time.time(), **record}


def write(kind, record):
    """Append one JSON row to <ledger>/<kind>.<pid>.jsonl (no-op when the ledger is off)."""
    root = os.environ.get('APPWORLD_LEDGER_DIR')
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f'{kind}.{os.getpid()}.jsonl'), 'a') as stream:
        stream.write(json.dumps(stamp(record), default=str) + '\n')


def write_gz(kind, record):
    """Append one gzip member holding one JSON row to <ledger>/<kind>.<pid>.jsonl.gz."""
    root = os.environ.get('APPWORLD_LEDGER_DIR')
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    data = (json.dumps(stamp(record), default=str) + '\n').encode()
    with open(os.path.join(root, f'{kind}.{os.getpid()}.jsonl.gz'), 'ab') as stream:
        stream.write(gzip.compress(data))


def policy_version():
    """The weight version the rollout engines carry (stamped by the trainer after each update)."""
    path = os.environ.get('EXP2_VERSION_FILE')
    try:
        with open(path) as stream:
            return int(stream.read().strip())
    except (TypeError, OSError, ValueError):
        return None
