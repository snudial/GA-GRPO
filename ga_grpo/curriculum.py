"""Per-task handoff curriculum, shared across Ray rollout processes.

exp1 measured q(h) — the learner's success rate when it takes over a guide
trajectory at turn h — and found Delta_reach > 0 on 91.5% of tasks that the
learner never solves on its own (q(0)=0 on 90/90 train tasks). Training
straight from h=0 therefore yields all-zero reward groups and no advantage,
so each task starts at the h where exp1 saw a coin flip and walks h down
toward 0 as the policy improves. That walk IS the learning signal.

State is one JSON file guarded by flock: rollouts run in many processes and
each finished episode is a read-modify-write.
"""
import fcntl
import json
import os
from datetime import datetime, timezone

STATE_PATH = os.environ.get(
    "EXP2_CURRICULUM_STATE", "/tmp/exp2_curriculum.json")
WINDOW = int(os.environ.get("EXP2_CURRICULUM_WINDOW", "8"))
PROMOTE_AT = float(os.environ.get("EXP2_CURRICULUM_PROMOTE", "0.7"))
DEMOTE_AT = float(os.environ.get("EXP2_CURRICULUM_DEMOTE", "0.1"))


def _load(f):
    f.seek(0)
    raw = f.read()
    return json.loads(raw) if raw.strip() else {}


def _save(f, state):
    f.seek(0)
    f.truncate()
    json.dump(state, f)
    f.flush()
    os.fsync(f.fileno())


def _open():
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    return open(STATE_PATH, "a+")


def initial_h(num_turns, mode="suffix", *, selected_h=None):
    """Initial assistance only; all modes use the same adaptive update rule.

    The mode is carried in the task label so every Ray worker receives it,
    including workers attached to an existing Ray cluster.
    """
    if mode == "autonomous":
        return 0
    if mode == "suffix":
        return max(0, num_turns - 1)
    if mode == "second_last_turn":
        return max(0, num_turns - 2)
    if mode in ("jev", "local_llm"):
        if type(selected_h) is not int or not 0 <= selected_h < num_turns:
            raise ValueError("Selected initial h must be an integer in [0, num_turns)")
        return selected_h
    raise ValueError(f"unknown curriculum_init: {mode!r}")


def call_positions(reference_turns):
    """Call-level handoff positions, ordered by increasing assistance.

    Position (t, j) replays turns 0..t-1 in full plus the first j reference
    calls of turn t, then the student finishes turn t. (t, 0) is the
    turn-level handoff h=t. Returns (positions, turn_starts).
    """
    positions = []
    for turn, calls in enumerate(reference_turns):
        positions += [(turn, j) for j in range(max(1, len(calls)))]
    return positions, [i for i, (_, j) in enumerate(positions) if j == 0]


def _adaptive_step(h, rate, H, turn_starts):
    """All-pass drops to the previous turn boundary; otherwise move one call.

    Backing off is always one call: a coarse step up after a coarse step down
    would bounce between turn boundaries and never try the mid-turn positions
    where success changes.
    """
    if rate == 1.0 and h > 0:
        return max(s for s in turn_starts if s < h)
    if rate >= PROMOTE_AT and h > 0:
        return h - 1
    if rate <= DEMOTE_AT and h < H - 1:
        return h + 1
    return h


def current_h(task_id, h_init, H):
    """Handoff turn to use for the next rollout of this task."""
    with _open() as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            state = _load(f)
            entry = state.get(task_id)
            if entry is None:
                entry = {"h": int(h_init), "n": 0, "succ": 0}
                state[task_id] = entry
                _save(f, state)
            return max(0, min(int(entry["h"]), max(0, H - 1)))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def record(task_id, success, H, turn_starts=None):
    """Log one episode outcome; step h when the window says so.

    With `turn_starts` (call-level positions), h is a position index and the
    step size adapts to the window (see `_adaptive_step`); otherwise h moves
    exactly one turn.

    Returns the (possibly unchanged) h for logging.
    """
    with _open() as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            state = _load(f)
            entry = state.setdefault(task_id, {"h": 0, "n": 0, "succ": 0})
            entry["n"] += 1
            entry["succ"] += int(bool(success))
            if entry["n"] >= WINDOW:
                rate = entry["succ"] / entry["n"]
                previous_h = entry["h"]
                if turn_starts is not None:
                    entry["h"] = _adaptive_step(entry["h"], rate, H, turn_starts)
                elif rate >= PROMOTE_AT and entry["h"] > 0:
                    # BFCL tasks are short. Move exactly one turn per window so
                    # every handoff depth is trained and its transfer measured.
                    entry["h"] -= 1
                elif rate <= DEMOTE_AT and entry["h"] < max(0, H - 1):
                    entry["h"] += 1          # task went out of reach; back off
                entry["n"] = 0
                entry["succ"] = 0
                _save(f, state)
                event_path = STATE_PATH + ".events.jsonl"
                with open(event_path, "a") as events:
                    events.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "task_id": task_id,
                        "h_before": previous_h,
                        "h_after": entry["h"],
                        "success_rate": rate,
                        "window": WINDOW,
                        "unit": "call" if turn_starts is not None else "turn",
                    }, sort_keys=True) + "\n")
                    events.flush()
                    os.fsync(events.fileno())
            else:
                _save(f, state)
            return entry["h"]
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def snapshot():
    """All task states — for logging and post-run curriculum curves."""
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return _load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
