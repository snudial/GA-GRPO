import unittest

from scripts.appworld_eval_metrics import completed_eval_metrics


class CompletedEvalMetricsTests(unittest.TestCase):
    def test_concatenated_process_logs(self):
        metric = "{'eval_default_pass1': 0.5, 'eval_num_samples': 168.0}"
        text = '\x1b[36m(actor)\x1b[0m Eval-only completed: ' + metric + '[eval-only] ' + metric
        self.assertEqual(completed_eval_metrics(text)['eval_default_pass1'], 0.5)

    def test_multiline_nested_and_quoted_braces(self):
        text = "Eval-only completed: {'note': '} {',\n 'nested': {'a': 1}, 'eval_default_pass1': 1, 'eval_num_samples': 3}junk {"
        self.assertEqual(completed_eval_metrics(text)['nested'], {'a': 1})

    def test_latest_completed_evaluation(self):
        text = "Eval-only completed: {'eval_default_pass1': 0, 'eval_num_samples': 3}\nEval-only completed: {'eval_default_pass1': 1, 'eval_num_samples': 3}"
        self.assertEqual(completed_eval_metrics(text)['eval_default_pass1'], 1)

    def test_invalid_latest_does_not_fall_back(self):
        prefix = "Eval-only completed: {'eval_default_pass1': 1, 'eval_num_samples': 3}\n"
        for suffix in ["{'eval_default_pass1': 2, 'eval_num_samples': 3}",
                       "{'eval_default_pass1': 0.5, 'eval_num_samples': 0}",
                       "{'eval_default_pass1': 0.5}", "missing dictionary", "{'unfinished':"]:
            with self.subTest(suffix=suffix), self.assertRaises(Exception):
                completed_eval_metrics(prefix + 'Eval-only completed: ' + suffix)

    def test_missing_marker(self):
        with self.assertRaises(ValueError):
            completed_eval_metrics('[eval-only] {"eval_default_pass1": 1}')


if __name__ == '__main__':
    unittest.main()
