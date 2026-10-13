"""Append-only stores shared between rollout agents and the trainer.

Two files with deliberately different sensitivity:

  failures_raw.jsonl  — everything the runtime checker said, including
                        missing reference responses that embed ground-truth
                        values. Analysis only; NEVER read by teacher code.
  opsd_bank.jsonl     — the teacher certificate: mismatched state FIELD NAMES
                        and counts only (no ground-truth values), plus the
                        student's own context and boundary action so the
                        trainer can rebuild both OPSD sequences. Keyed by a
                        40-bit uid that rides through molt's float32 info
                        scalars as two 20-bit halves.

Writers run in many Ray processes, so records go through flock-append.
"""
import fcntl
import json
import os
import random


def _path(name):
    root = os.environ.get("EXP2_BANK_DIR", "/tmp/exp2_bank")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, name)


def append(name, record):
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(_path(name), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


UID_HALF_BITS = 20  # each half must survive a float32 round-trip (24-bit mantissa)


def new_uid():
    """40-bit uid carried through molt's info scalars as TWO float32-safe
    halves — molt tensorizes info values as float32, which silently corrupts
    integers above 2^24 (measured on AppWorld: 48-bit uids came back off by ~4.5M)."""
    return random.getrandbits(2 * UID_HALF_BITS)


def uid_halves(uid):
    return uid >> UID_HALF_BITS, uid & ((1 << UID_HALF_BITS) - 1)


def uid_join(hi, lo):
    return (int(round(hi)) << UID_HALF_BITS) | int(round(lo))


def policy_version():
    """Weight version last broadcast to the rollout engines, stamped by the
    trainer (EXP2_VERSION_FILE hook in molt's rl_trainer). -1 = unknown."""
    path = os.environ.get("EXP2_VERSION_FILE", "")
    try:
        with open(path) as f:
            return int(f.read().strip() or -1)
    except Exception:  # noqa: BLE001
        return -1


def load_opsd_records(uids):
    """uid -> record for the requested uids (trainer side)."""
    want = {int(u) for u in uids}
    out = {}
    path = _path("opsd_bank.jsonl")
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("uid") in want:
                out[r["uid"]] = r
    return out
