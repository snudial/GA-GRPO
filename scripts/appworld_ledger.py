"""Join and check an AppWorld training ledger (appworld_ga/ledger.py; docs/APPWORLD_TRAINING_LEDGER_KO.md).

    python scripts/appworld_ledger.py RUN [--write]

Prints the integrity checks and per-group final status. --write also saves RUN/ledger/groups_final.jsonl:
each group row plus the trainer-confirmed outcome (filter decision, the AdamW updates that trained each
rollout, or why it never reached one). Exit status 1 if a check fails.

An update counts only if its segment's checkpoint survived: a cycle trained after the checkpoint the
next segment resumed from was discarded by the restart, and its groups were regenerated (new ids).
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path


def rows(ledger, kind):
    out = []
    for path in sorted(ledger.glob(f'{kind}.*.jsonl')) + sorted(ledger.glob(f'{kind}.jsonl')):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def turn_rollouts(ledger):
    """Rollout ids that have a turn trace (one gzip member per rollout)."""
    ids = []
    for path in sorted(ledger.glob('turns.*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            ids += [json.loads(line)['rollout_id'] for line in stream if line.strip()]
    return ids


def join(ledger):
    segments = [r for r in rows(ledger, 'segments') if r['event'] == 'start']
    segments.sort(key=lambda r: r['time'])
    # A segment's work up to the checkpoint the next one resumed from survives; the rest was discarded.
    survives = {s['segment_id']: (nxt['global_step'] if nxt else float('inf'))
                for s, nxt in zip(segments, segments[1:] + [None])}
    groups = rows(ledger, 'groups')
    rollouts = rows(ledger, 'rollouts')
    filters = {r['group_id']: r for r in rows(ledger, 'filter') if r['group_id']}
    cycle_of, dropped = {}, {}
    for r in rows(ledger, 'batch'):
        if r['event'] == 'cycle':
            for rid in r['trained_rollouts']:
                cycle_of[(r['segment_id'], rid)] = r['global_step_before'] + 1
            dropped.update({rid: r['dropped_reason'] for rid in r['dropped_rollouts']})
        else:
            dropped.update({rid: r['reason'] for rid in r['rollout_ids']})
    updates = collections.defaultdict(set)   # rollout -> {(segment, adam_step)} that survived
    discarded = set()
    for r in rows(ledger, 'updates'):
        for rid in r['rollouts']:
            step = cycle_of.get((r['segment_id'], rid))
            if step is not None and step > survives.get(r['segment_id'], float('inf')):
                discarded.add(rid)
            else:
                updates[rid].add((r['segment_id'], r['adam_step']))
    final = []
    for g in groups:
        ids = [r['rollout_id'] for r in g['rollouts']]
        f = filters.get(g['group_id'])
        trained = {rid: sorted(updates[rid]) for rid in ids if updates.get(rid)}
        if f is None:
            status = 'no_trainer_decision'   # still in flight when the segment ended
        elif f['decision'] == 'dropped':
            status = f'filtered:{f["reason"]}'
        elif len(trained) == len(ids):
            status = 'trained'
        elif any(rid in discarded for rid in ids):
            status = 'discarded_by_restart'
        elif trained:
            status = 'partly_trained'
        else:
            status = 'accepted_not_trained'  # buffered, or dropped by batching (see dropped_rollouts)
        final.append(dict(g, trainer_decision=f and f['decision'], exclusion_reason=f and f['reason'],
                          status=status, adam_steps=sorted({s for v in trained.values() for _, s in v}),
                          update_ids=sorted({f'{seg}:{s}' for v in trained.values() for seg, s in v}),
                          rollouts_trained=len(trained),
                          dropped_rollouts={rid: dropped[rid] for rid in ids if rid in dropped}))
    return segments, groups, rollouts, final


def check(ledger):
    segments, groups, rollouts, final = join(ledger)
    group_ids = [g['group_id'] for g in groups]
    rollout_ids = [r['rollout_id'] for g in groups for r in g['rollouts']]
    row_ids = [r['rollout_id'] for r in rollouts]
    traced = turn_rollouts(ledger)
    mixed = [g for g in final if g['outcome'] == 'mixed' and g['trainer_decision'] == 'accepted']
    # A live segment (no end row yet) still holds accepted groups for its next cycle.
    ended = {r['segment_id'] for r in rows(ledger, 'segments') if r['event'] == 'end'}
    waiting = [g for g in mixed if g['segment_id'] not in ended and g['status'] in ('accepted_not_trained', 'partly_trained')]
    checks = {
        'segments': len(segments),
        'groups': len(groups),
        'every group links exactly 16 distinct rollout ids': all(
            len(g['rollouts']) == 16 == len({r['rollout_id'] for r in g['rollouts']}) for g in groups),
        'group ids unique (all segments)': len(group_ids) == len(set(group_ids)),
        'rollout ids unique (all segments)': len(rollout_ids) == len(set(rollout_ids)),
        'one rollout row per group rollout': sorted(row_ids) == sorted(rollout_ids),
        'one turn trace per group rollout': sorted(traced) == sorted(rollout_ids),
        'every group has a segment id': all(g.get('segment_id') for g in groups),
        'mixed accepted groups': len(mixed),
        'mixed accepted groups with all 16 rollouts in an update':
            sum(g['rollouts_trained'] == 16 and bool(g['update_ids']) for g in mixed),
        'mixed accepted groups waiting for the next cycle (live segment)': len(waiting),
        'status counts': dict(collections.Counter(g['status'] for g in final)),
        'outcome counts': dict(collections.Counter(g['outcome'] for g in final)),
    }
    ok = (groups and checks['every group links exactly 16 distinct rollout ids']
          and checks['group ids unique (all segments)'] and checks['rollout ids unique (all segments)']
          and checks['one rollout row per group rollout'] and checks['one turn trace per group rollout']
          and checks['every group has a segment id']
          and all(g['status'] in ('trained', 'discarded_by_restart') for g in mixed
                  if not g['dropped_rollouts'] and g not in waiting))
    return bool(ok), checks, final


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    ledger = args.run/'ledger'
    ok, checks, final = check(ledger)
    print(json.dumps(checks, indent=2))
    if args.write:
        with (ledger/'groups_final.jsonl').open('w') as stream:
            stream.writelines(json.dumps(g) + '\n' for g in final)
    print('LEDGER OK' if ok else 'LEDGER CHECK FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
