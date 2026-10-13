# GA-GRPO: Guide-Assisted GRPO

Code for **"Where to Start, How to Move: Analyzing Curricula under GRPO for Long-Horizon LLM Agents"** (NAACL 2027 submission).

Under GRPO with a binary terminal reward, a group whose rollouts all fail (or all succeed) gives no reward-relative signal. Guide-assisted training replays a successful reference trajectory (the *guide*) up to a handoff position `h` and lets the learner complete the episode. The paper studies three choices:

- **Handoff unit**: `Turn` places handoffs at BFCL user-turn boundaries; `Call` also allows validated boundaries between tool calls.
- **Initialization**: `Start` begins at `h = 0`; `End` begins near the end of the guide (see the table below for the exact position).
- **Movement rule**: `Threshold` moves one step after each group; `Uniform` samples any eligible handoff; `Bandit` uses Thompson sampling to favor handoffs likely to produce mixed groups.

Only learner-generated tokens receive policy loss. The terminal reward scores the whole episode, including the replayed guide prefix. Evaluation always uses `h = 0`, so the learner must complete the full task without assistance.

`run.py` is the single entry point for every experiment.

## Methods

Each row of the paper's tables corresponds to one `--method`. The switches are listed in `config/methods.json`.

| Paper | `--method` | Unit | Initial handoff | Movement | Environments | Paper tables |
| --- | --- | --- | --- | --- | --- | --- |
| GRPO | `grpo` | none | none | none | BFCL, AppWorld | 1, 3, 4 |
| DAPO | `dapo` | none | none | none (dynamic sampling) | BFCL | 1, 3 |
| Turn-Threshold (Start) | `turn-threshold-start` | Turn | `h = 0` | Threshold | BFCL | 1 |
| Turn-Threshold (End) | `turn-threshold-end` | Turn | `H − 1`: before the final user turn | Threshold | BFCL | 1, 2, 3 |
| Turn-Bandit | `turn-bandit` | Turn | `H − 2`: before the second-to-last user turn | Bandit | BFCL | 2, 3 |
| Call-Threshold | `call-threshold` | Call | `H − 2`: before the second-to-last user turn | Threshold | BFCL, AppWorld | 2, 3, 4 |
| Call-Uniform | `call-uniform` | Call | `H − 2`: before the second-to-last user turn | Uniform | BFCL | 2, 3 |
| Call-Bandit | `call-bandit` | Call | `H − 2`: before the second-to-last user turn | Bandit | BFCL | 2, 3 |

`H` is the number of user turns in the guide; `h` counts replayed user turns (Turn) or replayed positions (Call, where each user-turn start is also a call position). The initial handoff is set by `BFCL_CURRICULUM_INIT` (`suffix`, `second_last_turn` or `autonomous`).

In AppWorld, handoffs fall between executed code blocks, which the paper treats as Call-level. `call-threshold` there starts before the guide's last code block and moves one block per group with the same thresholds.

### Movement rules

Each group contains 16 continuations of one task from one handoff. The rule runs once per completed group.

- **Threshold** (`ga_grpo/curriculum.py`): move one step earlier after 12–16 successes (rate ≥ 0.7), one step later after 0–1 successes (rate ≤ 0.1), otherwise stay. Turn-level steps are one user turn. Call-level steps are one call, except that an all-pass group moves directly to the preceding user-turn boundary.
- **Uniform** (`ga_grpo/h_selector.py`): sample uniformly from all eligible handoffs without using outcomes.
- **Bandit** (`ga_grpo/h_selector.py`): each task–position pair keeps success and failure counts. The selector draws `p ~ Beta(1 + s, 1 + f)` and scores `s(p) = 1 − p^16 − (1 − p)^16`, the probability that a 16-rollout group is mixed. With probability 0.1 it explores the least-observed position. Otherwise it picks the earliest position whose sampled score is within 0.1 of the best. The counts decay by a factor of 0.8 per epoch of policy updates, so positions that have not been visited for a while drift back toward the prior (`EXP2_BANDIT_DECAY`, default `0.8`).

Uniform and Bandit can stay or jump in either direction. All moves are restricted to replay-eligible positions. Every rollout of a group shares the handoff that was fixed when the group opened (`curriculum.assign`).

## Protocol

| Setting | Value |
| --- | --- |
| Models | Qwen3-1.7B, Qwen3-4B-Instruct-2507, Qwen3-8B (non-thinking mode) |
| BFCL training data | 100 Multi-Turn Base tasks |
| BFCL evaluation | Base: 25 validation + 75 held-out test tasks; Missing Function, Missing Parameter and Long Context: 100 tasks each (never trained on) |
| AppWorld | 90 train, 57 dev, 168 Test Normal, 417 Test Challenge tasks |
| Budget | 80,000 generated rollouts (BFCL), counting filtered all-fail and all-pass groups |
| Group size | 16 continuations from the same handoff |
| Optimizer | AdamW, learning rate 1e-6, weight decay 0.01, BF16 |
| Generation | temperature 1.0 during training; up to 4,096 generated tokens per response |
| Metrics | Avg@3 (mean of three evaluation decodes) and Best@3 (any of three succeeds); Macro Avg. averages the three untrained BFCL categories |

All-fail and all-pass groups are filtered from the GRPO update, but they count toward the rollout budget and inform outcome-based movement. Protocol constants are in `contracts/bfcl.json` and `contracts/appworld.json`. The BFCL splits are in `config/splits/bfcl_aworld_lineage_seed42_v1/`.

Run names use these model identifiers:

| Paper | `--model` |
| --- | --- |
| Qwen3-1.7B | `Qwen3-1.7B-NonThinking-AppendOnly` |
| Qwen3-4B-Instruct-2507 | `Qwen3-4B-Instruct-2507` |
| Qwen3-8B | `Qwen3-8B-NonThinking-AppendOnly` |

The `NonThinking-AppendOnly` variants are derived locally from the pinned Qwen3 checkpoints. They change only the chat template, which disables thinking and keeps the serialized history append-only across turns. The weights are not modified.

## Installation

Training requires Linux and NVIDIA GPUs with CUDA 13-compatible drivers (driver 580 or newer).

```bash
bash scripts/bootstrap.sh --prefix "$PWD/.deps" --topology multi_h100
```

The installer builds separate environments for training (Molt), BFCL simulation, official BFCL evaluation (vLLM) and AppWorld. It also downloads the pinned BFCL checkout and the initial 4B model, generates the BFCL training rows, and writes `config/env.sh`. Package versions are pinned in `env/`. Model revisions are pinned in `config/models.json`. Expect about 60 GB for dependencies and at least 150 GB for BFCL run outputs. AppWorld checks for 260 GiB of free space.

To use existing installations instead, copy `config/env.example.sh` to `config/env.sh`, set the paths, and run `bash scripts/setup.sh`.

Download the other models with:

```bash
python3 scripts/fetch_models.py Qwen3-1.7B-NonThinking-AppendOnly Qwen3-8B-NonThinking-AppendOnly
```

GPU profiles are in `config/topology/`:

| Profile | Hardware |
| --- | --- |
| `multi_gpu4` | 4× B200 |
| `multi_h100` | 4× H100 |
| `dual_h100` | 2× H100 |
| `single_h100` | 1× H100 (optimizer offload is chosen from the model size) |

An admission check runs before every launch and refuses model/topology combinations that do not fit. On a single-H100 node whose system compiler cannot build Transformer Engine, run `scripts/build_te_single_h100.sh` once after bootstrap.

## Training

Run from the repository root:

```bash
# BFCL, one arm of Table 2
python3 run.py --env bfcl --model Qwen3-8B-NonThinking-AppendOnly \
  --method call-bandit --seed 42 --topology multi_h100

# BFCL baseline
python3 run.py --env bfcl --model Qwen3-8B-NonThinking-AppendOnly \
  --method grpo --seed 42 --topology multi_h100

# AppWorld (Table 4)
python3 run.py --env appworld --model Qwen3-8B-NonThinking-AppendOnly \
  --method call-threshold --seed 42 --topology multi_h100
```

`--dry-run` prints the resolved plan without downloading anything or using GPUs. A short smoke run precedes every full run, and the BFCL smoke checks that the run manifest records the method's switches; `--no-smoke` skips it. Outputs go to `RUN_ROOT/<run>`. Relaunching with the same `--run` name resumes from the last checkpoint.

During BFCL training the driver evaluates the 25 validation tasks every 10 optimizer steps. At the end it evaluates the 75 test tasks for the final checkpoint and for the best validation checkpoint. The paper reports the final checkpoints.

## Evaluation

The paper's BFCL tables use the official BFCL harness on 100 tasks per category with three decodes at temperature 1e-6:

```bash
source config/env.sh
SPLITS=config/splits/bfcl_aworld_lineage_seed42_v1
HF=/abs/path/to/run/hf_ep50                     # the final checkpoint (80,000 rollouts)

# Base (Tables 1 and 2)
TASK_MANIFEST=$PWD/$SPLITS/base_eval.json CATEGORY=multi_turn_base \
  bash scripts/eval_official.sh "$HF" /abs/path/to/out/base
# Untrained categories (Table 3)
TASK_MANIFEST=$PWD/$SPLITS/miss_func_eval.json CATEGORY=multi_turn_miss_func \
  bash scripts/eval_official.sh "$HF" /abs/path/to/out/miss_func
TASK_MANIFEST=$PWD/$SPLITS/miss_param_eval.json CATEGORY=multi_turn_miss_param \
  bash scripts/eval_official.sh "$HF" /abs/path/to/out/miss_param
TASK_MANIFEST=$PWD/$SPLITS/long_context_eval.json CATEGORY=multi_turn_long_context \
  bash scripts/eval_official.sh "$HF" /abs/path/to/out/long_context
```

Each output directory has a `summary.json` with one strict session-success rate per decode. Avg@3 is their mean. Best@3 counts a task as solved if any decode solves it. The per-task results are in `decode_*/`.

For AppWorld, `run.py` evaluates Test Normal and Test Challenge (three decodes, `h = 0`) on the final checkpoint and on the dev-selected checkpoint after training. To evaluate an exported checkpoint separately:

```bash
.deps/venvs/molt/bin/python scripts/run_appworld.py \
  --model 8b --method call-threshold --seed 42 \
  --setup /abs/path/to/sealed_setup --run /abs/path/inside/RUN_ROOT/eval \
  --eval-only tests --checkpoint /abs/path/to/hf_checkpoint --execute
```

## AppWorld guides

AppWorld's protected material and its derivatives must not be redistributed unencrypted, so this repository contains no AppWorld data, guides or task rows. Put verified training guides in a local directory and set it in `config/env.sh`:

```bash
export APPWORLD_GUIDES_DIR=/abs/path/to/training_guides
```

Each guide is `<task_id>.json` with `task_id`, `verifier_pass: true`, and a `turns` list whose entries have `code`. Setup accepts guides only for official training-split tasks. On first use it replays every guide in AppWorld to confirm success, measures the handoff limits for the selected model, and seals the setup. Guides are never used at evaluation.

## Repository layout

```text
run.py              entry point
config/             methods, models, splits, GPU topologies, path template
contracts/          BFCL and AppWorld protocol constants
ga_grpo/            guide replay agent, curriculum (Threshold), handoff selectors (Uniform, Bandit), checks
bfcl_env/           BFCL multi-turn runtime and rollout protocol
appworld_ga/        AppWorld agent, curriculum and environment server pool
scripts/            installation, training drivers, official evaluation
third_party/molt/   modified Molt training engine (lock: third_party/molt.lock.json)
tests/              CPU unit tests
```

## Checks

```bash
python3 scripts/verify_vendor.py        # bundled Molt matches its lock
.deps/bin/uv pip install --python .deps/venvs/molt/bin/python -r requirements-dev.txt
.deps/venvs/molt/bin/python -m pytest -q -m "not slow"
```

Tests that need installed benchmarks, models or GPUs are skipped when those are unavailable.

## Licenses

The training engine is derived from [NVIDIA-NeMo Molt](https://github.com/NVIDIA-NeMo/labs-molt) (Apache-2.0); its notices are kept in `third_party/molt`. BFCL and the Qwen3 checkpoints are released under Apache-2.0. AppWorld's public portion uses the same license; its protected portion carries the additional redistribution condition described above. See `THIRD_PARTY_NOTICES.md`.
