# BFCL v3 multi-turn splits

Task IDs for each BFCL multi-turn category, produced by
`random.Random(42).shuffle(list(range(200)))`: the first 100 tasks are `train`,
the next 25 `val`, and the last 75 `test75`. `eval` is `val` ∪ `test75`
(100 tasks). The Base IDs match the train/validation/test files published with
AWorld-RL EnvTuning; their source hashes are recorded in `split_manifest.json`.

| File | Use in the paper |
| --- | --- |
| `base_train.json` | training (100 Multi-Turn Base tasks) |
| `base_val.json` | in-training validation and checkpoint selection |
| `base_test75.json` | held-out test, evaluated by the training driver |
| `base_eval.json` | Base evaluation in Tables 1 and 2 (val + test, 100 tasks) |
| `miss_func_eval.json`, `miss_param_eval.json`, `long_context_eval.json` | untrained categories in Table 3 (100 tasks each) |

Only Base tasks are used for training. The `*_train`, `*_val` and `*_test75`
files of the other categories are provided for completeness. These are
research splits, not BFCL leaderboard splits.
