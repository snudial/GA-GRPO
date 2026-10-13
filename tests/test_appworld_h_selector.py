"""EXP2_H_SELECTOR on the AppWorld curriculum: bandit / uniform choose the next h among 0..h_max
from per-position records; the default window rule is untouched."""
import importlib
import json
from types import SimpleNamespace

import pytest


def _label(task='t1', h_max=5):
    return json.dumps(dict(task_id=task, guide_path='/g/t1.json', guide_H=h_max + 1, h_max=h_max))


def _result(reward):
    return SimpleNamespace(reward=reward, observation_tokens=list(range(50)), action_ranges=[(10, 15)], extra_logs={})


def _curriculum(monkeypatch, tmp_path, selector=None, seed='42'):
    monkeypatch.setenv('APPWORLD_CURRICULUM', '1')
    monkeypatch.setenv('APPWORLD_METHOD', 'ga-grpo-call-bandit' if selector == 'bandit' else 'ga-grpo')
    monkeypatch.setenv('APPWORLD_SEED', seed)
    monkeypatch.setenv('EXP2_CURRICULUM_STATE', str(tmp_path / 'curriculum.json'))
    monkeypatch.setenv('EXP2_VERSION_FILE', str(tmp_path / 'policy_version.txt'))
    monkeypatch.delenv('APPWORLD_LEDGER_DIR', raising=False)
    if selector:
        monkeypatch.setenv('EXP2_H_SELECTOR', selector)
    else:
        monkeypatch.delenv('EXP2_H_SELECTOR', raising=False)
    (tmp_path / 'policy_version.txt').write_text('0')
    import appworld_ga.curriculum as curriculum
    return importlib.reload(curriculum)


def _run_group(curriculum, label, gid, successes, n=16):
    group = curriculum.prepare_group(label, gid, n)
    curriculum.finish_group(group, [_result(float(i < successes)) for i in range(n)], [f'{gid}-{i}' for i in range(n)])
    return group['h']


def test_window_rule_unchanged_without_selector(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path)
    assert _run_group(c, _label(), 'g1', successes=16) == 5           # starts at h_max
    state = json.load(open(tmp_path / 'curriculum.json'))
    assert state['t1'] == {'h': 4, 'groups': 1}                        # promote by one, no stats


def test_bandit_records_the_position_and_picks_among_all_candidates(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'bandit')
    assert _run_group(c, _label(), 'g1', successes=16) == 5
    state = json.load(open(tmp_path / 'curriculum.json'))['t1']
    assert state['stats'] == {'5': [16.0, 0.0]} and state['groups'] == 1 and state['stats_version'] == 0
    assert 0 <= state['h'] <= 5
    event = json.loads(open(str(tmp_path / 'curriculum.json') + '.groups.jsonl').readlines()[-1])
    sel = event['selection']
    assert sel['selector'] == 'bandit' and set(sel['candidates']) == {str(p) for p in range(6)}
    assert sel['candidates']['5']['n'] == 16.0 and sel['decay_factor'] == 1.0


def test_bandit_choice_replays_from_seed_task_and_group_index(monkeypatch, tmp_path):
    hs = []
    for i in range(2):
        d = tmp_path / str(i); d.mkdir()
        c = _curriculum(monkeypatch, d, 'bandit')
        seq = []
        for g in range(6):
            seq.append(_run_group(c, _label(), f'g{g}', successes=8))   # mixed groups: Thompson draws vary by position
        hs.append(seq)
    assert hs[0] == hs[1]


def test_records_fade_with_policy_updates(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'bandit')
    _run_group(c, _label(), 'g1', successes=16)
    (tmp_path / 'policy_version.txt').write_text('2')                  # two updates later (decay 0.8/epoch)
    _run_group(c, _label(), 'g2', successes=0)
    state = json.load(open(tmp_path / 'curriculum.json'))['t1']
    a5, b5 = state['stats']['5']
    assert a5 == pytest.approx(16 * 0.8 ** 2) and state['stats_version'] == 2


def test_uniform_moves_without_the_window_rule(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'uniform')
    seen = set()
    for g in range(30):
        seen.add(_run_group(c, _label(), f'g{g}', successes=16))
    assert len(seen) >= 4                                               # not a +-1 walk


def test_stale_group_does_not_move_h_or_record(monkeypatch, tmp_path):
    c = _curriculum(monkeypatch, tmp_path, 'bandit')
    g1 = c.prepare_group(_label(), 'g1', 16); g2 = c.prepare_group(_label(), 'g2', 16)
    c.finish_group(g1, [_result(1.0)] * 16, [f'a{i}' for i in range(16)])
    state_after_first = json.load(open(tmp_path / 'curriculum.json'))['t1']
    c.finish_group(g2, [_result(0.0)] * 16, [f'b{i}' for i in range(16)])   # g2 used the old h: stale
    state = json.load(open(tmp_path / 'curriculum.json'))['t1']
    assert state['h'] == state_after_first['h'] and state['stats'] == state_after_first['stats']
