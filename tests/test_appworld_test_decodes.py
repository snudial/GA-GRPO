"""Test splits are scored over TEST_DECODES decodes (3 by default: Avg@3 / Best@3)."""
import json
from types import SimpleNamespace

import scripts.run_appworld as R


def _setup(tmp_path):
    setup = tmp_path/'setup'
    (setup/'datasets').mkdir(parents=True)
    (setup/'setup.json').write_text(json.dumps({'environment_revision': 'rev'}))
    for split, n in [('test_normal', 2), ('test_challenge', 3)]:
        (setup/'datasets'/f'{split}.jsonl').write_text(''.join('{}\n' for _ in range(n)))
    return setup


def test_every_test_split_gets_three_decodes_and_reports_their_mean(tmp_path, monkeypatch):
    calls = []

    def fake_job(setup, run, args, weights, split, decode, env, tag, gpu=None):
        calls.append((split, decode))
        return decode / 10, f'{split}-{decode}.jsonl'
    monkeypatch.setattr(R, '_eval_job', fake_job)
    monkeypatch.setattr(R, '_eval_gpus', lambda: 2)
    out = R.evaluate_tests_many(_setup(tmp_path), SimpleNamespace(), {}, [(tmp_path, 'w', 'final', ('test_normal', 'test_challenge'))])
    assert R.TEST_DECODES == 3
    assert sorted(calls) == sorted((s, d) for s in ('test_normal', 'test_challenge') for d in range(1, 4))
    normal = out[0]['test_normal']
    assert normal['decodes'] == 3 and normal['decode_scores'] == [d / 10 for d in range(1, 4)]
    assert abs(normal['mean'] - 0.2) < 1e-9 and abs(normal['mean_first3'] - 0.2) < 1e-9
    assert 'avg_at_3' not in normal
