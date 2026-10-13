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

try:  # ga_grpo/ is also on sys.path as a plain directory (bfcl_agent.py: `from curriculum import ...`)
    from . import h_selector
except ImportError:
    import h_selector

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


def policy_version():
    """Weight version stamped by the trainer (EXP2_VERSION_FILE) at each sync; 0 before the first."""
    path = os.environ.get("EXP2_VERSION_FILE")
    try:
        with open(path) as f:
            return int(float(f.read().strip() or 0))
    except (OSError, TypeError, ValueError):
        return 0


def _entry(state, task_id, h_init):
    entry = state.get(task_id)
    if entry is None:
        entry = {"h": int(h_init), "n": 0, "succ": 0}
        state[task_id] = entry
    return entry


def _clamp(h, H):
    return max(0, min(int(h), max(0, H - 1)))


def current_h(task_id, h_init, H):
    """Handoff turn to use for the next rollout of this task (legacy: no group identity)."""
    with _open() as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            state = _load(f)
            entry = state.get(task_id)
            if entry is None:
                entry = {"h": int(h_init), "n": 0, "succ": 0}
                state[task_id] = entry
                _save(f, state)
            return _clamp(entry["h"], H)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def assign(task_id, h_init, H):
    """Lease one slot of the task's current group: returns (group_id, h).

    A group is fixed when it opens (id, h, planned size WINDOW, policy version); its
    WINDOW rollouts all receive that h, whatever the state does meanwhile. Once the
    group has handed out WINDOW slots the next caller opens a new group at the
    task's current h; a full group whose rollouts are still running waits in
    `pending` until they report (see record)."""
    with _open() as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            state = _load(f)
            entry = _entry(state, task_id, h_init)
            group = entry.get("open")
            if group is None or group["started"] >= group["planned"]:
                if group is not None:
                    entry.setdefault("pending", []).append(group)
                entry["seq"] = entry.get("seq", 0) + 1
                group = {"id": f"{task_id}:{entry['seq']}", "h": _clamp(entry["h"], H), "planned": WINDOW,
                         "started": 0, "done": [], "succ": 0, "policy_version": policy_version()}
                entry["open"] = group
            group["started"] += 1
            _save(f, state)
            return group["id"], group["h"]
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _event(payload):
    with open(STATE_PATH + ".events.jsonl", "a") as events:
        events.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **payload},
                                sort_keys=True) + "\n")
        events.flush()
        os.fsync(events.fileno())


def _step_h(task_id, entry, previous_h, successes, count, H, turn_starts, group_version):
    """Move the task's h from one closed group's outcome; returns the event fields."""
    rate = successes / count
    selector = os.environ.get("BFCL_H_SELECTOR", "window")
    fields = {"h_before": previous_h, "success_rate": rate, "selector": selector, "candidates": None,
              "selection": None, "decay_factor": None}
    if selector != "window":
        stats = entry.setdefault("stats", {})
        cfg = h_selector.settings()
        now = policy_version()
        factor = h_selector.decay_factor(now - entry.get("stats_version", now), cfg["decay"], cfg["updates_per_epoch"])
        h_selector.observe(stats, previous_h, successes, count - successes, factor)
        entry["stats_version"] = now
        entry["windows"] = entry.get("windows", 0) + 1
        positions = list(range(max(1, H)))
        entry["h"], selection = h_selector.choose(selector, stats, positions,
                                                  h_selector.rng_for(task_id, entry["windows"]), group=count)
        fields.update(candidates={str(p): h_selector.posterior(stats, p) for p in positions},
                      selection=selection, decay_factor=factor, policy_version_now=now)
    elif turn_starts is not None:
        entry["h"] = _adaptive_step(entry["h"], rate, H, turn_starts)
    elif rate >= PROMOTE_AT and entry["h"] > 0:
        # BFCL tasks are short. Move exactly one turn per window so
        # every handoff depth is trained and its transfer measured.
        entry["h"] -= 1
    elif rate <= DEMOTE_AT and entry["h"] < max(0, H - 1):
        entry["h"] += 1          # task went out of reach; back off
    fields["h_after"] = entry["h"]
    return fields


def record(task_id, success, H, turn_starts=None, *, group_id=None, rollout_id=None):
    """Log one episode outcome; step h when its group is complete.

    With `group_id` (from assign) the outcome joins that group: a rollout reported twice
    counts once, the group closes exactly once when all of its planned rollouts reported,
    and a group that can never complete (a dropped rollout) is closed as `incomplete`,
    excluded from the h update and the selector's records, when a newer group of the
    task closes. Without a group id the legacy per-task window counter is used.

    Returns the (possibly unchanged) h for logging.
    """
    with _open() as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            state = _load(f)
            entry = state.setdefault(task_id, {"h": 0, "n": 0, "succ": 0})
            unit = "call" if turn_starts is not None else "turn"
            if group_id is None:
                entry["n"] += 1
                entry["succ"] += int(bool(success))
                if entry["n"] >= WINDOW:
                    fields = _step_h(task_id, entry, entry["h"], entry["succ"], entry["n"], H, turn_starts, None)
                    entry["n"] = 0
                    entry["succ"] = 0
                    _save(f, state)
                    _event({"task_id": task_id, "window": WINDOW, "unit": unit, "group_id": None,
                            "rollouts": WINDOW, "status": "complete", **fields})
                else:
                    _save(f, state)
                return entry["h"]

            groups = ([entry["open"]] if entry.get("open") else []) + entry.get("pending", [])
            group = next((g for g in groups if g["id"] == group_id), None)
            if group is None:
                _event({"task_id": task_id, "group_id": group_id, "rollout_id": rollout_id, "status": "orphan",
                        "unit": unit, "h_before": entry["h"], "h_after": entry["h"]})
                return entry["h"]
            if rollout_id in group["done"]:
                _event({"task_id": task_id, "group_id": group_id, "rollout_id": rollout_id, "status": "duplicate",
                        "unit": unit, "h_before": entry["h"], "h_after": entry["h"]})
                return entry["h"]
            group["done"].append(rollout_id)
            group["succ"] += int(bool(success))
            if len(group["done"]) < group["planned"]:
                _save(f, state)
                return entry["h"]
            # The group is complete: it alone moves h. A group whose rollouts were started two
            # groups ago and still has not reported in full has lost a rollout (the runner drops
            # a failed one): an infrastructure loss, not a task failure, so it is closed as
            # incomplete and excluded from the records. One newer completion is not enough:
            # two groups of a task can legitimately run and finish interleaved.
            fields = _step_h(task_id, entry, group["h"], group["succ"], group["planned"], H, turn_starts,
                             group["policy_version"])
            pending = entry.get("pending", [])
            seq = int(group["id"].rsplit(":", 1)[1])
            older = [g for g in pending if int(g["id"].rsplit(":", 1)[1]) <= seq - 2]
            for g in older:
                _event({"task_id": task_id, "group_id": g["id"], "unit": unit, "status": "incomplete",
                        "rollouts": len(g["done"]), "planned": g["planned"], "h_before": g["h"], "h_after": entry["h"],
                        "success_rate": (g["succ"] / len(g["done"])) if g["done"] else None,
                        "policy_version_at_assign": g["policy_version"]})
            entry["pending"] = [g for g in pending if g is not group and g not in older]
            if entry.get("open") is group:
                entry["open"] = None
            _save(f, state)
            _event({"task_id": task_id, "group_id": group["id"], "window": WINDOW, "unit": unit,
                    "rollouts": group["planned"], "planned": group["planned"], "status": "complete",
                    "policy_version_at_assign": group["policy_version"], **fields})
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
