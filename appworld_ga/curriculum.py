"""One handoff per prompt group; update after all student rollouts finish.

EXP2_H_SELECTOR (training only, config/methods.json): how the next h is chosen when a group
closes. "window" (default) is the fixed rule below (success rate >= .7: h-1, <= .1: h+1).
"bandit" / "uniform" choose among every position 0..h_max from the task's per-position records
(ga_grpo/h_selector.py, the same selector as BFCL's call-level handoff): the records fade with
the policy version the ledger stamps (EXP2_VERSION_FILE), the choice is seeded by
(APPWORLD_SEED, task, group index) so a resume replays it, and the selection is logged to
curriculum.json.groups.jsonl and the ledger's group row.

EXP2_H_CANDIDATES (bandit / uniform only) sets which positions are candidates (2026-10-12):
"full" (default, unchanged): every position 0..h_max.
"fixedK" (fixed3, fixed4, ...): 0, K, 2K, ... and h_max; jumps between any two candidates are allowed.
"adaptiveK" (adaptive3, adaptive4, ...): starts from the fixedK set. When the latest completed groups at two initial candidates a < b
adjacent in that set were all-fail at a (0/16) and all-pass at b (16/16), and their creation policy versions
differ by at most EXP2_ADAPTIVE_MAX_VERSION_GAP (default 4), the integers strictly between a and b become
candidates for good (no records: the selector's prior, like any unvisited position). Only the group that
closes at a or b can open that interval; stale groups are skipped as before, incomplete groups raise before
any state change, and a group id already counted for the task never opens an interval twice. Candidates,
initial candidates, the latest outcome per candidate and every expansion live in the task's curriculum entry
(saved with each checkpoint), so a resume keeps them and replays the same choices."""
import fcntl
import json
import os
import random
import re
import time
from pathlib import Path

from appworld_ga import ledger
from ga_grpo import h_selector

CANDIDATE_STEP = 3


def candidate_mode():
    """'full', or (kind, step) for 'fixedK' / 'adaptiveK' with K >= 2."""
    mode = os.environ.get('EXP2_H_CANDIDATES', 'full')
    if mode == 'full':
        return mode
    match = re.fullmatch(r'(fixed|adaptive)([2-9]|[1-9][0-9]+)', mode)
    if not match:
        raise ValueError(f'unknown EXP2_H_CANDIDATES {mode!r} (full, fixedK, adaptiveK with K >= 2)')
    return match.group(1), int(match.group(2))


def initial_candidates(h_max, step=CANDIDATE_STEP):
    """0, step, 2*step, ... and h_max (h_max need not be a multiple of step)."""
    return sorted(set(range(0, h_max + 1, step)) | {h_max})


def max_version_gap():
    """Adaptive-3: the two outcomes that open an interval must come from groups whose creation policy
    versions differ by at most this many optimizer updates. 4 = the span over which the selector's own
    records keep at least 0.8**4 = 41% of their weight (decay 0.8 per update); a task is revisited about
    once per epoch (1.7 updates on AppWorld 4B), so this admits outcomes from consecutive and next-but-one
    visits and rejects pairs whose policies differ by several epochs."""
    return int(os.environ.get('EXP2_ADAPTIVE_MAX_VERSION_GAP', '4'))


def _adaptive_update(item, label, scores, now):
    """Record this group's outcome at its h and open the interval to a neighbouring initial candidate if the
    all-fail-below / all-pass-above pattern holds. Returns the expansion record or None."""
    h = label['h']
    group_id = str(label['group_id'])
    seen = item.setdefault('counted_groups', [])
    if group_id in seen:
        return None                                  # duplicate: never opens an interval
    seen.append(group_id)
    del seen[:-16]                                   # bounded: duplicates arrive close together
    created = label.get('ledger_created', {}).get('policy_version')
    version = created if created is not None else now
    k, n = int(sum(scores)), len(scores)
    item.setdefault('last_outcome', {})[str(h)] = dict(successes=k, n=n, policy_version=version, group_id=group_id)
    initial = item['initial_candidates']
    if h not in initial:
        return None                                  # opened positions never open further intervals
    i = initial.index(h)
    gap = max_version_gap()
    pairs = ([(initial[i - 1], h)] if i > 0 else []) + ([(h, initial[i + 1])] if i + 1 < len(initial) else [])
    for a, b in pairs:
        if b - a <= 1 or any(e['a'] == a and e['b'] == b for e in item.get('expansions', [])):
            continue
        la, lb = item['last_outcome'].get(str(a)), item['last_outcome'].get(str(b))
        if not la or not lb or not (la['successes'] == 0 and la['n'] == n and lb['successes'] == lb['n'] == n):
            continue
        if la['policy_version'] is None or lb['policy_version'] is None or abs(la['policy_version'] - lb['policy_version']) > gap:
            continue
        opened = [p for p in range(a + 1, b) if p not in item['candidates']]
        item['candidates'] = sorted(set(item['candidates']) | set(opened))
        record = dict(a=a, b=b, opened=opened, version_a=la['policy_version'], version_b=lb['policy_version'],
                      group_a=la['group_id'], group_b=lb['group_id'], trigger_group=group_id,
                      policy_version=now, task_groups=item['groups'])
        item.setdefault('expansions', []).append(record)
        return record
    return None


def prepare_group(label, group_id, n_samples):
    label = json.loads(label) if isinstance(label, str) else dict(label)
    label.update(group_id=group_id, group_size=n_samples, h=0)
    if ledger.enabled() and not label.get('eval'):
        # Creation-time facts for the group row (finish_group); the dispatch counters
        # ride in the label from the trainer's sample generator (prepare_appworld.py).
        label['ledger_created'] = dict(time=time.time(), policy_version=ledger.policy_version())
    if label.get('eval') or os.environ.get('APPWORLD_CURRICULUM') != '1':
        return label
    if not label.get('guide_path'):
        return label
    path = Path(os.environ['EXP2_CURRICULUM_STATE'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        state = json.loads(stream.read() or '{}')
        initial = label.get('h_max', label['guide_H'] - 1)   # End initialization: the guide's last code block
        item = state.setdefault(label['task_id'], {'h': initial, 'groups': 0})
        label['h'] = item['h']
        stream.seek(0)
        stream.truncate()
        json.dump(state, stream)
    return label


def finish_group(label, results, rollout_ids=None):
    scores = []
    for result in results:
        if isinstance(result, BaseException):
            raise RuntimeError('AppWorld rollout failed; do not count infrastructure errors as reward 0') from result
        trajectories = result if isinstance(result, list) else [result]
        if len(trajectories) != 1:
            raise RuntimeError(
                f'Expected one append-only training sample per trajectory, got {len(trajectories)} '
                f'(task {label["task_id"]}, h {label.get("h")}; tokens per segment '
                f'{[len(t.observation_tokens) for t in trajectories]})')
        trajectory = trajectories[0]
        scores.append(float(trajectory.reward))
    if len(scores) != label['group_size'] or any(s not in (0., 1.) for s in scores):
        raise RuntimeError('Incomplete group or nonbinary reward')
    if label.get('eval') or not label.get('guide_path') or os.environ.get('APPWORLD_CURRICULUM') != '1':
        _write_ledger(label, results, rollout_ids, scores, None)
        return
    with open(os.environ['EXP2_CURRICULUM_STATE'], 'r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        state = json.load(stream)
        item = state[label['task_id']]
        # A later epoch can dispatch this task before the previous group returns.
        stale = item['h'] != label['h']
        h_before = item['h']   # the task's position when this group returns (differs from h_used if stale)
        rate = sum(scores) / len(scores)
        selector = os.environ.get('EXP2_H_SELECTOR', 'window')
        selection = None
        h_max = label.get('h_max', label['guide_H'] - 1)
        if not stale:
            if selector != 'window':
                # Per-position records fade with the policy version (no update, no fade), then this
                # group's outcome is added at the h it used; the next h is chosen among 0..h_max.
                stats = item.setdefault('stats', {})
                cfg = h_selector.settings()
                now = ledger.policy_version()
                if now is None:
                    now = item.get('stats_version', 0)
                factor = h_selector.decay_factor(now - item.get('stats_version', now), cfg['decay'], cfg['updates_per_epoch'])
                h_selector.observe(stats, label['h'], sum(scores), len(scores) - sum(scores), factor)
                item['stats_version'] = now
                mode = candidate_mode()
                expansion = None
                if mode == 'full':
                    positions = list(range(h_max + 1))
                else:
                    kind, step = mode
                    if 'candidates' not in item:
                        item['initial_candidates'] = initial_candidates(h_max, step)
                        item['candidates'] = list(item['initial_candidates'])
                    if kind == 'adaptive':
                        expansion = _adaptive_update(item, label, scores, now)
                    positions = list(item['candidates'])
                rng = random.Random(f"{os.environ.get('APPWORLD_SEED', '42')}:{label['task_id']}:{item['groups']}")
                item['h'], chosen = h_selector.choose(selector, stats, positions, rng, group=len(scores))
                selection = dict(selector=selector, decay_factor=factor, policy_version=now,
                                 candidates={str(p): h_selector.posterior(stats, p) for p in positions}, **chosen)
                if mode != 'full':
                    selection.update(candidate_mode=os.environ['EXP2_H_CANDIDATES'], candidate_set=positions,
                                     reason='explore' if chosen.get('explore') else 'thompson',
                                     expansion=expansion)
            elif rate >= .7:
                item['h'] = max(0, item['h'] - 1)
            elif rate <= .1:
                item['h'] = min(h_max, item['h'] + 1)
            item['groups'] += 1
        event = dict(task_id=label['task_id'], group_id=label['group_id'], h=label['h'],
                     next_h=item['h'], success_rate=rate, stale=stale, selection=selection)
        with open(os.environ['EXP2_CURRICULUM_STATE'] + '.groups.jsonl', 'a') as log:
            log.write(json.dumps(event) + '\n')
        stream.seek(0)
        stream.truncate()
        json.dump(state, stream)
    _write_ledger(label, results, rollout_ids, scores,
                  dict(stale=stale, h_before=h_before, h_after=event['next_h'], selection=selection))


def _write_ledger(label, results, rollout_ids, scores, move):
    """One group row, its 16 rollout rows and their turn traces, under the trainer's rollout ids.

    `move` is the JUMP curriculum step (None when the task has no guide or the method has no
    curriculum, where every visit starts at h=0)."""
    if label.get('eval') or not ledger.enabled():
        return
    if rollout_ids is None or len(rollout_ids) != len(results):
        raise RuntimeError('Ledger needs one trainer rollout id per rollout (patched router.run_group)')
    wins = int(sum(scores))
    created = label.get('ledger_created', {})
    dispatch = label.get('ledger_dispatch', {})
    rollouts = []
    for rollout_id, result, reward in zip(rollout_ids, results, scores):
        trajectory = (result if isinstance(result, list) else [result])[0]
        logs = getattr(trajectory, 'extra_logs', None) or {}
        # Exact token ids the trainer sees: each action range starts at that turn's context length.
        spans = getattr(trajectory, 'action_ranges', None) or []
        record = dict(logs.get('ledger') or {}, rollout_id=rollout_id, group_id=label['group_id'],
                      task_id=label['task_id'], reward=reward,
                      sequence_tokens=len(getattr(trajectory, 'observation_tokens', None) or []),
                      initial_context_tokens=spans[0][0] if spans else None,
                      generated_tokens=sum(end - start for start, end in spans),
                      input_tokens_over_turns=sum(start for start, _ in spans))
        ledger.write('rollouts', record)
        ledger.write_gz('turns', dict(rollout_id=rollout_id, group_id=label['group_id'], task_id=label['task_id'],
                                      h_used=label['h'], reward=reward, turns=logs.pop('ledger_turns', None)))
        rollouts.append(dict(rollout_id=rollout_id, reward=reward))
    curriculum = move is not None
    before = dispatch.get('groups_dispatched_before')
    ledger.write('groups', dict(
        run_id=Path(os.environ['APPWORLD_LEDGER_DIR']).parent.name,
        training_seed=os.environ.get('APPWORLD_SEED'), method=os.environ.get('APPWORLD_METHOD'),
        task_id=label['task_id'], group_id=label['group_id'], group_size=len(scores),
        created_time=created.get('time'), policy_version_at_creation=created.get('policy_version'),
        dispatch_index_in_segment=dispatch.get('index'), groups_dispatched_before=before,
        rollouts_generated_before=None if before is None else before * len(scores),
        curriculum=curriculum, guide_id=Path(label['guide_path']).stem if label.get('guide_path') else None,
        guide_H=label.get('guide_H', 0), h_max=label.get('h_max', max(0, label.get('guide_H', 0) - 1)),
        h_before=move['h_before'] if curriculum else label['h'], h_used=label['h'],
        h_after=move['h_after'] if curriculum else label['h'], stale=move['stale'] if curriculum else False,
        rollouts=rollouts, successes=wins,
        outcome='all-fail' if wins == 0 else 'all-pass' if wins == len(scores) else 'mixed',
        selection=move.get('selection') if curriculum else None))
