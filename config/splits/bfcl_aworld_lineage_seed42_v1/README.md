# BFCL V3 split manifests

These manifests preserve the seed-42 split recovered from the public
AWorld/EnvTuning/T3RL lineage.  The exact ID list has not been confirmed by the
CrEST authors, so it remains a lineage candidate rather than a claimed official
CrEST release.

## CrEST scope

CrEST trains on **`base_train.json` only** (100 Multi-Turn Base sessions).
Its final BFCL V3 evaluation uses 100 held-out sessions from each category:

- `base_eval.json`
- `miss_func_eval.json`
- `miss_param_eval.json`
- `long_context_eval.json`

The primary method-comparison protocol uses `base_train.json` (100 tasks),
`base_val.json` (25 tasks), and `base_test75.json` (75 tasks). They exactly
match the Base IDs in AWorld-RL EnvTuning's published
`bfcl_train_base.parquet`, `bfcl_val.parquet`, and `bfcl_test.parquet` at the
pinned commit. The source parquet SHA256 values are recorded in
`split_manifest.json`; the ID equality and source hashes were verified in the
2026-09-05 protocol audit.

This is therefore the published AWorld-RL/T3RL training protocol split, not a
locally invented split. It is not, however, a BFCL leaderboard split: upstream
BFCL treats the original 200 tasks as an evaluation collection. Test75 results
must therefore be labelled "AWorld-RL EnvTuning held-out Base test75", not BFCL
leaderboard scores.

The corresponding train/val/test75 projections for the other categories are
generated and committed for reproducibility, but are not inputs to the primary
Base-only experiment.

HINT-SD is a separate protocol: its paper uses BFCL V3 Base and Long Context
with a 5:1:4 train/eval/test partition.  Do not substitute these CrEST manifests
for a HINT-SD reproduction without separately verifying that paper's exact IDs.
