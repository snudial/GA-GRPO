"""EXP2_H_CANDIDATES on the AppWorld bandit (2026-10-12): fixed3 / adaptive3 candidate sets; full unchanged."""
import importlib
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest



def _label(task='t1', h_max=24):
    return json.dumps(dict(task_id=task, guide_path=f'/g/{task}.json', guide_H=h_max + 1, h_max=h_max))


def _result(reward):
    return SimpleNamespace(reward=reward, observation_tokens=list(range(50)), action_ranges=[(10, 15)], extra_logs={})


def _curriculum(monkeypatch, tmp_path, mode=None):
    monkeypatch.setenv('APPWORLD_CURRICULUM', '1')
    monkeypatch.setenv('APPWORLD_METHOD', 'ga-grpo-call-bandit')
    monkeypatch.setenv('APPWORLD_SEED', '42')
    monkeypatch.setenv('EXP2_H_SELECTOR', 'bandit')
    monkeypatch.setenv('EXP2_CURRICULUM_STATE', str(tmp_path / 'curriculum.json'))
    monkeypatch.setenv('EXP2_VERSION_FILE', str(tmp_path / 'policy_version.txt'))
    monkeypatch.delenv('APPWORLD_LEDGER_DIR', raising=False)
    if mode:
        monkeypatch.setenv('EXP2_H_CANDIDATES', mode)
    else:
        monkeypatch.delenv('EXP2_H_CANDIDATES', raising=False)
    (tmp_path / 'policy_version.txt').write_text('0')
    import appworld_ga.curriculum as curriculum
    return importlib.reload(curriculum)


def _state(tmp_path):
    return json.load(open(tmp_path / 'curriculum.json'))


def _events(tmp_path):
    return [json.loads(l) for l in open(str(tmp_path / 'curriculum.json') + '.groups.jsonl')]


def _force_h(tmp_path, task, h):
    state = _state(tmp_path)
    state[task]['h'] = h
    json.dump(state, open(tmp_path / 'curriculum.json', 'w'))


def _group(c, label, gid, successes, n=16):
    g = c.prepare_group(label, gid, n)
    c.finish_group(g, [_result(float(i < successes)) for i in range(n)], [f'{gid}-{i}' for i in range(n)])
    return g


@pytest.mark.parametrize('h_max,expected', [(17, [0, 3, 6, 9, 12, 15, 17]), (18, [0, 3, 6, 9, 12, 15, 18]),
                                            (2, [0, 2]), (3, [0, 3]), (0, [0])])
def test_initial_candidates_keep_zero_and_hmax(monkeypatch, tmp_path, h_max, expected):
    c = _curriculum(monkeypatch, tmp_path, 'fixed3')
    assert c.initial_candidates(h_max) == expected


@pytest.mark.parametrize('h_max,expected', [(18, [0, 4, 8, 12, 16, 18]), (17, [0, 4, 8, 12, 16, 17]),
                                            (16, [0, 4, 8, 12, 16]), (3, [0, 3])])
def test_step4_candidates_keep_zero_and_hmax(monkeypatch, tmp_path, h_max, expected):
    c = _curriculum(monkeypatch, tmp_path, 'adaptive4')
    assert c.candidate_mode() == ('adaptive', 4) and c.initial_candidates(h_max, 4) == expected


@pytest.mark.parametrize('mode', ['adaptive1', 'fixed', 'adaptive04', 'window3'])
def test_unknown_candidate_modes_raise(monkeypatch, tmp_path, mode):
    c = _curriculum(monkeypatch, tmp_path, mode)
    with pytest.raises(ValueError):
        c.candidate_mode()


def test_adaptive4_opens_three_positions_and_logs_its_mode(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'adaptive4')
    label = _label('t1', 24)
    _group(c, label, 'init', successes=8)
    _force_h(tmp_path, 't1', 20); _group(c, label, 'g20', successes=0)     # 20F
    _force_h(tmp_path, 't1', 24); _group(c, label, 'g24', successes=16)    # 24P
    item = _state(tmp_path)['t1']
    assert item['initial_candidates'] == [0, 4, 8, 12, 16, 20, 24]
    assert [(e['a'], e['b'], e['opened']) for e in item['expansions']] == [(20, 24, [21, 22, 23])]
    assert _events(tmp_path)[-1]['selection']['candidate_mode'] == 'adaptive4'


def test_fixed3_picks_only_candidates_jumps_globally_and_explores_ten_percent(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'fixed3')
    rng = random.Random(0)
    big_jumps = explores = choices = 0
    for t in range(60):
        task, label = f't{t}', _label(f't{t}', 17)
        for k in range(30):
            _group(c, label, f'{task}-{k}', successes=rng.randint(0, 16))
    for e in _events(tmp_path):
        sel = e['selection']
        assert sel['candidate_mode'] == 'fixed3' and sel['candidate_set'] == [0, 3, 6, 9, 12, 15, 17]
        assert e['next_h'] in sel['candidate_set'] and e['h'] in sel['candidate_set']
        assert set(sel['candidates']) == {str(p) for p in sel['candidate_set']}
        assert sel['reason'] in ('explore', 'thompson') and sel['expansion'] is None
        big_jumps += abs(e['next_h'] - e['h']) > 3
        explores += sel['reason'] == 'explore'
        choices += 1
    assert big_jumps > 0.1 * choices                     # jumps across several candidates happen
    assert 0.07 < explores / choices < 0.13              # the 10% least-observed pick is kept


def _setup_adaptive(c, tmp_path, task='t1', h_max=24):
    label = _label(task, h_max)
    _group(c, label, 'init', successes=8)                 # first group at h_max creates the entry
    return label


def test_adaptive_opens_the_interval_for_fail_below_pass_above(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'adaptive3')
    label = _setup_adaptive(c, tmp_path)
    _force_h(tmp_path, 't1', 21); _group(c, label, 'g21', successes=0)     # 21F
    _force_h(tmp_path, 't1', 24); _group(c, label, 'g24', successes=16)    # 24P
    item = _state(tmp_path)['t1']
    assert {22, 23} <= set(item['candidates']) and item['initial_candidates'] == list(range(0, 25, 3))
    assert [(e['a'], e['b'], e['opened']) for e in item['expansions']] == [(21, 24, [22, 23])]
    exp = _events(tmp_path)[-1]['selection']['expansion']
    assert exp['a'] == 21 and exp['b'] == 24 and exp['trigger_group'] == 'g24'
    assert '22' not in item['stats'] and '23' not in item['stats']            # opened at the prior
    assert {22, 23} <= set(_events(tmp_path)[-1]['selection']['candidate_set'])


@pytest.mark.parametrize('k21,k24', [(5, 16), (0, 9), (16, 0), (3, 7)])
def test_adaptive_ignores_mixed_and_reversed_outcomes(monkeypatch, tmp_path, k21, k24):
    c = _curriculum(monkeypatch, tmp_path, 'adaptive3')
    label = _setup_adaptive(c, tmp_path)
    _force_h(tmp_path, 't1', 21); _group(c, label, 'g21', successes=k21)
    _force_h(tmp_path, 't1', 24); _group(c, label, 'g24', successes=k24)
    item = _state(tmp_path)['t1']
    assert item.get('expansions', []) == [] and item['candidates'] == item['initial_candidates']


def test_adaptive_requires_close_policy_versions(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'adaptive3')
    label = _setup_adaptive(c, tmp_path)
    _force_h(tmp_path, 't1', 21); _group(c, label, 'g21', successes=0)
    (tmp_path / 'policy_version.txt').write_text(str(c.max_version_gap() + 1))
    _force_h(tmp_path, 't1', 24); _group(c, label, 'g24', successes=16)
    assert _state(tmp_path)['t1'].get('expansions', []) == []
    (tmp_path / 'policy_version.txt').write_text(str(c.max_version_gap() + 2))   # fresh 21F, now close to 24P
    _force_h(tmp_path, 't1', 21); _group(c, label, 'g21b', successes=0)
    assert [(e['a'], e['b']) for e in _state(tmp_path)['t1']['expansions']] == [(21, 24)]


def test_adaptive_stale_duplicate_and_incomplete_groups_never_expand(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'adaptive3')
    label = _setup_adaptive(c, tmp_path)
    _force_h(tmp_path, 't1', 21); _group(c, label, 'g21', successes=0)
    # stale: dispatched at 24, the task moved before it returned
    _force_h(tmp_path, 't1', 24)
    g = c.prepare_group(label, 'g24', 16)
    _force_h(tmp_path, 't1', 6)
    c.finish_group(g, [_result(1.0)] * 16, [f'g24-{i}' for i in range(16)])
    assert _state(tmp_path)['t1'].get('expansions', []) == [] and _events(tmp_path)[-1]['stale']
    # duplicate: a group id already counted (g21) returning again at 24P
    _force_h(tmp_path, 't1', 24)
    _group(c, label, 'g21', successes=16)
    assert _state(tmp_path)['t1'].get('expansions', []) == []
    # incomplete: 15 of 16 rollouts raises before any state change
    before = _state(tmp_path)
    g = c.prepare_group(label, 'g24c', 16)
    with pytest.raises(RuntimeError):
        c.finish_group(g, [_result(1.0)] * 15, [f'g24c-{i}' for i in range(15)])
    assert _state(tmp_path) == before


def _outcomes(tasks=6, groups=25):
    rng = random.Random(7)
    return {(t, k): rng.choice([0, 0, 16, 16, rng.randint(1, 15)]) for t in range(tasks) for k in range(groups)}


def _play(c, outcomes, ks, tasks=6):
    for t in range(tasks):
        for k in ks:
            _group(c, _label(f't{t}', 13 + t), f't{t}-{k}', successes=outcomes[(t, k)])


def test_adaptive_state_survives_a_resume_and_replays(monkeypatch, tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    out = _outcomes()
    c = _curriculum(monkeypatch, a, 'adaptive3'); _play(c, out, range(25))
    c = _curriculum(monkeypatch, b, 'adaptive3'); _play(c, out, range(12))
    c = _curriculum(monkeypatch, b, 'adaptive3')          # reload = a new process after a resume
    _play(c, out, range(12, 25))
    sa, sb = _state(a), _state(b)
    assert any(sa[f't{t}'].get('expansions') for t in range(6))      # the sequence does open intervals
    for t in range(6):
        for key in ('h', 'candidates', 'initial_candidates', 'expansions', 'last_outcome', 'stats'):
            assert sa[f't{t}'].get(key) == sb[f't{t}'].get(key), (t, key)


RUNNER = r'''
import json, random, sys
from types import SimpleNamespace
import appworld_ga.curriculum as c
def res(r): return SimpleNamespace(reward=r, observation_tokens=list(range(50)), action_ranges=[(10, 15)], extra_logs={})
rng = random.Random(11)
for t in range(5):
    label = json.dumps(dict(task_id=f't{t}', guide_path='/g/x.json', guide_H=15 + 2 * t, h_max=14 + 2 * t))
    for k in range(30):
        open(sys.argv[1], 'w').write(str(k // 3))          # policy version advances
        g = c.prepare_group(label, f't{t}-{k}', 16)
        s = rng.randint(0, 16)
        c.finish_group(g, [res(float(i < s)) for i in range(16)], [f'r{i}' for i in range(16)])
'''


def _run_full(repo, out):
    out.mkdir()
    env = {**os.environ, 'PYTHONPATH': repo, 'APPWORLD_CURRICULUM': '1', 'APPWORLD_METHOD': 'ga-grpo-call-bandit',
           'APPWORLD_SEED': '42', 'EXP2_H_SELECTOR': 'bandit', 'EXP2_CURRICULUM_STATE': str(out / 'curriculum.json'),
           'EXP2_VERSION_FILE': str(out / 'v.txt')}
    env.pop('APPWORLD_LEDGER_DIR', None); env.pop('EXP2_H_CANDIDATES', None)
    subprocess.run([sys.executable, '-c', RUNNER, str(out / 'v.txt')], env=env, check=True, cwd=repo)
    return (out / 'curriculum.json').read_bytes(), (out / 'curriculum.json.groups.jsonl').read_bytes()
