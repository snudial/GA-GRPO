"""Reproducibility identity and fail-closed comparison, without loading a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_appworld_repro_eval import compare_records, digest, lane_for, request_seed


class ReproducibleEvaluationTests(unittest.TestCase):
    def test_identity_survives_process_hash_randomization(self):
        code = ('from scripts.run_appworld_repro_eval import request_seed,lane_for;'
                "print(request_seed('dev','example',20260827,4),lane_for('dev','example',4))")
        results = [subprocess.check_output([sys.executable, '-c', code], cwd=ROOT,
                   env={**os.environ, 'PYTHONHASHSEED': h}, text=True) for h in ['0', '12345']]
        self.assertEqual(results[0], results[1])

    def test_request_seeds_distinguish_task_turn_repeat_seed_and_split(self):
        keys = [('dev', t, s, turn) for t in ['a', 'b'] for s in [42, 43] for turn in range(30)]
        keys.append(('test_normal', 'a', 42, 0))
        seeds = [request_seed(*k) for k in keys]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(0 <= s < 2**31 for s in seeds))

    def test_assignment_does_not_depend_on_order(self):
        tasks = [f'task{i}' for i in range(100)]
        a = {t: lane_for('dev', t, 4) for t in tasks}
        b = {t: lane_for('dev', t, 4) for t in reversed(tasks)}
        self.assertEqual(a, b)
        self.assertEqual(set(a.values()), {0, 1, 2, 3})

    def test_equal_score_is_not_equal_trajectory(self):
        a = dict(split='dev', task_id='a', decode_seed=42, success=0,
                 token_trace_sha256='first', messages_sha256='obs')
        b = {**a, 'token_trace_sha256': 'second'}
        diff = compare_records([a], [b])
        self.assertEqual(diff[0]['fields'], ['token_trace_sha256'])
        self.assertEqual(compare_records([a], [dict(a)]), [])

    def test_missing_or_duplicate_cannot_pass(self):
        a = dict(split='dev', task_id='a', decode_seed=42)
        with self.assertRaises(ValueError):
            compare_records([a], [])
        with self.assertRaises(ValueError):
            compare_records([a], [a, a])

    def test_environment_and_gpu_changes_detected(self):
        a = dict(split='dev', task_id='a', decode_seed=42, gpu_uuid='gpu1',
                 messages_sha256=digest(['observation1']))
        b = {**a, 'gpu_uuid': 'gpu2', 'messages_sha256': digest(['observation2'])}
        self.assertEqual(set(compare_records([a], [b])[0]['fields']),
                         {'gpu_uuid', 'messages_sha256'})


if __name__ == '__main__':
    unittest.main()
