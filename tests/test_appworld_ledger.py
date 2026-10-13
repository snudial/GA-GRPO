import gzip
import json
import py_compile
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from appworld_ga import curriculum
from scripts import appworld_ledger
from scripts.patch_appworld_actor import patch_actor
from scripts.prepare_appworld import patch_runtime

ROOT = Path(__file__).resolve().parents[1]


def rows(ledger, kind):
    return [json.loads(line) for path in sorted(ledger.glob(f'{kind}.*.jsonl')) for line in path.read_text().splitlines()]


def result(reward):
    return SimpleNamespace(reward=reward, observation_tokens=list(range(50)), action_ranges=[(10, 15), (30, 40)],
                           extra_logs={'ledger': {'termination': 'completed', 'student_turns': 3},
                                                       'ledger_turns': [{'turn': 0, 'code': 'print(1)'}]})


@pytest.mark.parametrize('method', ['grpo', 'ga-grpo'])
def test_group_row_links_the_trainer_rollout_ids(monkeypatch, tmp_path, method):
    ledger = tmp_path/'run'/'ledger'
    monkeypatch.setenv('APPWORLD_LEDGER_DIR', str(ledger))
    monkeypatch.setenv('APPWORLD_SEGMENT_ID', 'seg-a')
    monkeypatch.setenv('APPWORLD_SEED', '43')
    monkeypatch.setenv('APPWORLD_METHOD', method)
    monkeypatch.setenv('APPWORLD_CURRICULUM', '1' if 'ga-grpo' in method else '0')
    monkeypatch.setenv('EXP2_CURRICULUM_STATE', str(tmp_path/'state.json'))
    monkeypatch.setenv('EXP2_VERSION_FILE', str(tmp_path/'version.txt'))
    (tmp_path/'version.txt').write_text('7')
    label = json.dumps(dict(task_id='t1', guide_path='/g/t1.json', guide_H=18, h_max=17, eval=False,
                            ledger_dispatch=dict(index=2, groups_dispatched_before=40)))
    group = curriculum.prepare_group(label, 'g1', 16)
    ids = [f'r{i}' for i in range(16)]
    curriculum.finish_group(group, [result(float(i < 5)) for i in range(16)], ids)
    [row] = rows(ledger, 'groups')
    assert [r['rollout_id'] for r in row['rollouts']] == ids and row['successes'] == 5 and row['outcome'] == 'mixed'
    assert row['segment_id'] == 'seg-a' and row['run_id'] == 'run' and row['training_seed'] == '43'
    assert row['policy_version_at_creation'] == 7 and row['rollouts_generated_before'] == 640
    assert row['guide_id'] == 't1' and row['h_max'] == 17
    if method == 'grpo':
        assert (row['h_before'], row['h_used'], row['h_after'], row['curriculum']) == (0, 0, 0, False)
    else:   # 5/16 is neither promotion (>= .7) nor demotion (<= .1): h stays at the cap
        assert (row['h_before'], row['h_used'], row['h_after'], row['curriculum']) == (17, 17, 17, True)
    rollout = rows(ledger, 'rollouts')
    assert sorted(r['rollout_id'] for r in rollout) == sorted(ids) and rollout[0]['termination'] == 'completed'
    assert (rollout[0]['generated_tokens'], rollout[0]['input_tokens_over_turns'], rollout[0]['sequence_tokens']) == (15, 40, 50)
    with gzip.open(next(ledger.glob('turns.*.jsonl.gz')), 'rt') as stream:
        traces = [json.loads(line) for line in stream]
    assert [t['rollout_id'] for t in traces] == ids and traces[0]['turns'][0]['code'] == 'print(1)'
    # Eval groups write nothing; a missing rollout id refuses rather than guessing.
    curriculum.finish_group(curriculum.prepare_group({**json.loads(label), 'eval': True}, 'e', 3),
                            [result(1.)] * 3, ['a', 'b', 'c'])
    assert len(rows(ledger, 'groups')) == 1
    with pytest.raises(RuntimeError, match='rollout id'):
        curriculum.finish_group(curriculum.prepare_group(label, 'g2', 16), [result(0.)] * 16)


def test_ledger_off_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv('APPWORLD_LEDGER_DIR', raising=False)
    monkeypatch.setenv('APPWORLD_CURRICULUM', '0')
    group = curriculum.prepare_group(dict(task_id='t1', eval=False), 'g1', 16)
    assert 'ledger_created' not in group
    curriculum.finish_group(group, [result(1.)] * 16, [str(i) for i in range(16)])
    assert not list(tmp_path.iterdir())


def test_join_drops_updates_lost_to_a_restart(tmp_path):
    ledger = tmp_path/'ledger'
    ledger.mkdir()

    def write(kind, *records):   # the driver's segments file has no pid, as in a run
        with (ledger/(f'{kind}.jsonl' if kind == 'segments' else f'{kind}.1.jsonl')).open('a') as stream:
            stream.writelines(json.dumps(r) + '\n' for r in records)

    def group(gid, seg, wins):
        return dict(group_id=gid, segment_id=seg, outcome='mixed' if 0 < wins < 16 else 'all-fail',
                    rollouts=[dict(rollout_id=f'{gid}-{i}', reward=float(i < wins)) for i in range(16)])

    write('segments', dict(event='start', segment_id='A', time=1, global_step=0),
          dict(event='start', segment_id='B', time=2, global_step=1))
    # A trained cycles 1 and 2 but B resumed from step 1: cycle 2 (group a2) was discarded and regenerated as b1.
    write('groups', group('a1', 'A', 4), group('a2', 'A', 6), group('a3', 'A', 0), group('b1', 'B', 6))
    write('rollouts', *[dict(rollout_id=r['rollout_id']) for g in ('a1', 'a2', 'a3', 'b1')
                        for r in group(g, '', 0)['rollouts']])
    with gzip.open(ledger/'turns.1.jsonl.gz', 'wt') as stream:
        stream.writelines(json.dumps(dict(rollout_id=f'{g}-{i}')) + '\n' for g in ('a1', 'a2', 'a3', 'b1') for i in range(16))
    write('filter', dict(group_id='a1', decision='accepted', reason=None),
          dict(group_id='a2', decision='accepted', reason=None),
          dict(group_id='a3', decision='dropped', reason='uniform_reward'),
          dict(group_id='b1', decision='accepted', reason=None))
    ids = lambda g: [f'{g}-{i}' for i in range(16)]
    write('batch', dict(event='cycle', segment_id='A', global_step_before=0, trained_rollouts=ids('a1'), dropped_rollouts=[], dropped_reason='x'),
          dict(event='cycle', segment_id='A', global_step_before=1, trained_rollouts=ids('a2'), dropped_rollouts=[], dropped_reason='x'),
          dict(event='cycle', segment_id='B', global_step_before=1, trained_rollouts=ids('b1'), dropped_rollouts=[], dropped_reason='x'))
    write('updates', dict(segment_id='A', adam_step=1, rollouts=ids('a1')), dict(segment_id='A', adam_step=2, rollouts=ids('a2')),
          dict(segment_id='B', adam_step=2, rollouts=ids('b1')))
    ok, checks, final = appworld_ledger.check(ledger)
    status = {g['group_id']: (g['status'], g['update_ids']) for g in final}
    assert ok and checks['segments'] == 2 and checks['mixed accepted groups with all 16 rollouts in an update'] == 2
    assert status == {'a1': ('trained', ['A:1']), 'a2': ('discarded_by_restart', []),
                      'a3': ('filtered:uniform_reward', []), 'b1': ('trained', ['B:2'])}


def test_runtime_ledger_patches_apply_to_the_pinned_molt(tmp_path):
    runtime = tmp_path/'runtime'
    for name in ['molt', 'tests', 'examples/python']:
        shutil.copytree(ROOT/'third_party/molt'/name, runtime/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patch_runtime(runtime)
    patch_actor(runtime)
    for name in ['trainer/rollout/router.py', 'trainer/rollout/samples_generator.py', 'trainer/rl_trainer.py',
                 'cli/train_rl_ray.py', 'trainer/workers/policy_actor.py']:
        py_compile.compile(str(runtime/'molt'/name), doraise=True)
    generator = (runtime/'molt/trainer/rollout/samples_generator.py').read_text()
    assert generator.count('self._ledger_filter(') == 3 and 'label = self._ledger_dispatch(label)' in generator
    assert 'generate_eval_refill(self' in generator and 'APPWORLD_EVAL_REFILL' in generator
    # Module-level helpers go above decorators: the Ray actor class must keep its @ray.remote.
    actor = (runtime/'molt/trainer/workers/policy_actor.py').read_text()
    assert '@ray.remote(num_gpus=1)\nclass PolicyModelActor(' in actor
    assert '\n\n\nclass SamplesGenerator:' in generator
    cli = (runtime/'molt/cli/train_rl_ray.py').read_text()   # engines sleep while a resume loads Adam on GPU
    assert cli.index('engine.sleep.remote(2)') < cli.index('async_init_model_from_pretrained(') \
        < cli.index('ray.get(refs)') < cli.index('engine.wake_up.remote()')
    router = (runtime/'molt/trainer/rollout/router.py').read_text()
    assert 'finish_group(label, group_results, rollout_ids)' in router and 'rollout_id = uuid4().hex' not in router


def test_runtime_ledger_helpers_run_in_the_patched_runtime(monkeypatch, tmp_path):
    """Import the patched runtime (needs the Molt venv) and run each injected helper on fake inputs:
    compiling does not catch a helper landing between a decorator and its class."""
    pytest.importorskip('vllm')
    import importlib
    import sys
    import torch
    runtime = tmp_path/'runtime'
    for name in ['molt', 'tests', 'examples/python']:
        shutil.copytree(ROOT/'third_party/molt'/name, runtime/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patch_runtime(runtime)
    patch_actor(runtime)
    monkeypatch.setattr(sys, 'path', [str(runtime), *sys.path])
    for name in [m for m in sys.modules if m == 'molt' or m.startswith('molt.')]:
        monkeypatch.delitem(sys.modules, name)
    ledger = tmp_path/'ledger'
    monkeypatch.setenv('APPWORLD_LEDGER_DIR', str(ledger))
    sg = importlib.import_module('molt.trainer.rollout.samples_generator')
    pa = importlib.import_module('molt.trainer.workers.policy_actor')
    assert hasattr(pa.PolicyModelActor, 'options')
    gen = object.__new__(sg.SamplesGenerator)
    gen.set_prompt_budget(40, 100)
    assert json.loads(gen._ledger_dispatch(json.dumps({'task_id': 't'})))['ledger_dispatch'] == \
        {'index': 0, 'groups_dispatched_before': 40}
    exp = lambda i: SimpleNamespace(group_ids=['g'], rollout_ids=[f'r{i}'], scores=torch.tensor([float(i < 3)]))
    gen._ledger_filter([exp(i) for i in range(16)], [], 'accepted', None, True)
    pa._ledger_update(SimpleNamespace(actor_scheduler=SimpleNamespace(last_epoch=7)),
                      [SimpleNamespace(rollout_ids=['r0'], group_ids=['g'], action_mask=torch.ones(1, 5))])
    assert rows(ledger, 'filter')[0]['group_id'] == 'g' and len(rows(ledger, 'filter')[0]['usable_rollouts']) == 16
    assert [r['adam_step'] for r in rows(ledger, 'updates.rank0')] == [7]


def test_eval_refill_keeps_capacity_in_flight_and_order():
    from contextlib import contextmanager
    from appworld_ga.eval_refill import generate_eval_refill
    in_flight, peak, done = set(), [0], []

    class Ray:
        def wait(self, refs, num_returns, timeout):
            ref = min(refs)                     # finish the oldest dispatched group
            in_flight.discard(ref)
            return [ref], [r for r in refs if r != ref]

    class Gen:
        args = SimpleNamespace(rollout=SimpleNamespace(n_samples_per_prompt=16))
        eval_dataloader = [(None, [f'p{i}' for i in range(j, j + 4)], [f'l{i}' for i in range(j, j + 4)], [None] * 4, [None] * 4)
                           for j in range(0, 12, 4)]
        n = 0

        def _dispatch_to_agent_runners(self, prompts, labels, **kw):
            Gen.n += 1
            in_flight.add(Gen.n)
            peak[0] = max(peak[0], len(in_flight))
            return [Gen.n]

        def _filter_group(self, ref, dynamic, drops, **kw):
            done.append(ref)
            return [f'sample{ref}']

    @contextmanager
    def progress(**kw):
        yield SimpleNamespace(update=lambda *a, **k: None, set_postfix=lambda *a, **k: None)

    out = generate_eval_refill(Gen(), ray=Ray(), logger=SimpleNamespace(info=lambda *a: None), progress=progress,
                               capacity=9, n_samples_per_prompt=3)
    assert out == [f'sample{i}' for i in range(1, 13)] and peak[0] == 3   # 9 episodes = 3 groups of 3
