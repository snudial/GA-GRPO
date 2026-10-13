# Where to Start, How to Move: Analyzing Curricula under GRPO for Long-Horizon LLM Agents

Anonymous code release for the submission. The implementation is called **GA-GRPO** (Guide-Assisted GRPO), and `run.py` is the single entry point for every experiment in the paper.

## Study design (Section 2)

**Guide-assisted starts.** A successful reference trajectory serves as a guide. Each rollout resets the environment and replays the guide prefix up to handoff `h` before the learner takes control. The replay changes the environment state and supplies the executed actions and observations as dialogue history. The learner completes the episode without access to the unexecuted guide suffix.

**Handoff unit.** `Turn` places handoffs at BFCL user-turn boundaries. `Call` additionally permits validated boundaries between tool calls inside a user turn.

**Initialization.** `Start` initializes at `h = 0`, with no replay. `End` initializes near the end of the guide. Both allow later movement in either direction.

**Movement.** Each group contains 16 continuations of the same task from the same handoff. After each group:

- `Threshold` moves earlier after 12–16 successes, later after 0–1, and otherwise stays.
- `Uniform` samples uniformly from eligible handoffs without using outcomes.
- `Bandit` tracks outcomes by task and position and uses Thompson sampling to favor handoffs likely to produce both successes and failures.

**Training and evaluation.** Policy loss applies only to learner-generated tokens. A binary terminal task reward evaluates the whole episode, including the guide prefix and the learner's continuation. All-fail and all-pass groups are filtered from the reward-relative update, but they count toward the rollout budget and inform outcome-based movement. Evaluation uses `h = 0`, so the learner must complete the full task without assistance.

## Methods

Each method in the paper's tables is one `--method`. Its switches are in `config/methods.json`.

| Paper | `--method` | Unit | Movement | Tables |
| --- | --- | --- | --- | --- |
| GRPO | `grpo` | none | none | 1, 3, 4 |
| DAPO | `dapo` | none | none (dynamic sampling) | 1, 3 |
| Turn-Threshold, Start | `turn-threshold-start` | Turn | Threshold | 1 |
| Turn-Threshold, End | `turn-threshold-end` | Turn | Threshold | 1, 2, 3 |
| Turn-Bandit | `turn-bandit` | Turn | Bandit | 2, 3 |
| Call-Threshold | `call-threshold` | Call | Threshold | 2, 3, 4 |
| Call-Uniform | `call-uniform` | Call | Uniform | 2, 3 |
| Call-Bandit | `call-bandit` | Call | Bandit | 2, 3 |

All guided methods except Turn-Threshold, Start use End initialization.

AppWorld (Table 4) uses `grpo` and `call-threshold`. AppWorld agents act by executing Python code blocks, and handoffs between code-execution boundaries are Call-level. `call-threshold` starts before the guide's last code block and moves one block per group with the same thresholds.

## Experimental settings (Section 2 and Appendix B)

### Models and benchmarks

| Paper | `--model` | Benchmarks |
| --- | --- | --- |
| Qwen3-1.7B | `Qwen3-1.7B-NonThinking-AppendOnly` | BFCL |
| Qwen3-4B-Instruct-2507 | `Qwen3-4B-Instruct-2507` | BFCL |
| Qwen3-8B | `Qwen3-8B-NonThinking-AppendOnly` | BFCL, AppWorld |

All models run in non-thinking mode. The two `NonThinking-AppendOnly` variants are derived locally from the pinned Qwen3 checkpoints by `scripts/fetch_models.py`. Only their chat template changes: thinking is disabled and the serialized history stays append-only across turns. The weights are hardlinked, not modified. Revisions are pinned in `config/models.json`. All runs in the paper use seed 42.

### Data splits and checkpoint selection (B.1)

BFCL training uses 100 Multi-Turn Base tasks. The Base evaluation consists of 25 validation tasks and 75 held-out test tasks. Missing Function, Missing Parameter and Long Context are excluded from training and evaluated on 100 tasks each. The task lists are in `config/splits/bfcl_aworld_lineage_seed42_v1/`:

| File | Use |
| --- | --- |
| `base_train.json` | training (100 tasks) |
| `base_val.json`, `base_test75.json` | validation (25) and held-out test (75) |
| `base_eval.json` | Base evaluation for Tables 1 and 2: validation ∪ test (100) |
| `miss_func_eval.json`, `miss_param_eval.json`, `long_context_eval.json` | Table 3 (100 each) |

AppWorld uses 90 training tasks and 57 development tasks. Test Normal and Test Challenge contain 168 and 417 tasks.

The main tables report the **final checkpoint** at the generated-rollout budget. For BFCL that is 80,000 generated rollouts: 50 passes over 100 tasks × 16 rollouts, saved as `hf_ep50`. The budget counts every generated rollout, including filtered groups. The AppWorld contract saves a checkpoint every 14,400 generated rollouts, up to 129,600; `--budget` selects one of these milestones.

### Optimization and rollout generation (B.2)

| Setting | Value | Where |
| --- | --- | --- |
| Optimizer | AdamW, learning rate 1e-6, weight decay 0.01 | `scripts/train.sh` (`ACTOR_LR`, `WEIGHT_DECAY`) |
| Group | 16 learner continuations from the same handoff | `contracts/*.json` (`group_size`) |
| Batch | 32 accepted groups per optimizer step | `scripts/train.sh` (`ROLLOUT_BS`, `TRAIN_BS`) |
| Sampling temperature | 1.0 | `scripts/train.sh` (`ROLLOUT_TEMP`) |
| Response length | up to 4,096 generated tokens per response; 10,000-token context (BFCL), 20,480 (AppWorld) | `contracts/*.json` |
| Precision | BF16 | `scripts/train.sh` |
| Framework | Molt with vLLM for rollout generation | `third_party/molt` |

The paper's setup uses four NVIDIA B200 or four H100 GPUs (`multi_gpu4`, `multi_h100`). Each GPU layout is a profile in `config/topology/`. Profiles change only device placement and memory settings, not the contract above.

| Profile | Hardware |
| --- | --- |
| `multi_gpu4` | 4× B200 |
| `multi_h100` | 4× H100 |
| `dual_h100` | 2× H100 |
| `single_h100` | 1× H100 (optimizer offload chosen from the model size) |

### Movement step sizes (B.3)

`ga_grpo/curriculum.py` implements the Threshold rule with a 16-rollout window: success rate ≥ 0.7 moves one step earlier, and ≤ 0.1 moves one step later. Turn-Threshold moves one user turn when its threshold triggers. Call-Threshold moves one call, except that an all-pass group moves directly to the preceding user-turn boundary. All moves are restricted to replay-eligible positions. Uniform and Bandit can select any eligible position, including the current one, without an adjacent-step constraint. All 16 rollouts of a group share the handoff fixed when the group opened (`curriculum.assign`).

### Bandit selection (B.4)

`ga_grpo/h_selector.py` keeps success and failure counts for each task–position pair. With a Beta(1, 1) prior, the selector samples a success probability `p` from the Beta posterior and computes `s(p) = 1 − p^16 − (1 − p)^16`, the probability that a 16-rollout group contains both successes and failures. With probability 0.1 it explores a least-observed position. Otherwise it chooses the earliest position whose sampled score is within 0.1 of the highest score. Counts fade by a factor of 0.8 per epoch of policy updates, so the selector re-checks positions whose success rate may have changed as the learner improves.

| Setting | Value | Environment variable |
| --- | --- | --- |
| Exploration probability | 0.1 | `EXP2_BANDIT_EXPLORE` |
| Score tolerance | 0.1 | `EXP2_BANDIT_TIE` |
| Count decay per epoch of updates | 0.8 | `EXP2_BANDIT_DECAY` |

### Metrics

Avg@3 averages three evaluation outcomes per task, and Best@3 records whether any of them succeeds. Evaluation decodes at temperature 1e-6 with seeds 20260826 + d. Macro Avg. averages Missing Function, Missing Parameter and Long Context. The three evaluation repeats do not measure training-seed variability.

## Installation

Requirements: Linux and NVIDIA GPUs with CUDA 13-compatible drivers (580 or newer).

```bash
cd GA-GRPO
bash scripts/bootstrap.sh --prefix "$PWD/.deps" --topology multi_h100
python3 scripts/fetch_models.py Qwen3-1.7B-NonThinking-AppendOnly Qwen3-8B-NonThinking-AppendOnly
```

The installer builds separate environments for training (Molt), BFCL simulation, official BFCL evaluation (vLLM) and AppWorld. It also fetches the pinned BFCL checkout and Qwen3-4B-Instruct-2507, generates the BFCL training rows, and writes `config/env.sh`. Package versions are pinned in `env/`. Allow about 60 GB for dependencies and 150 GB for BFCL outputs; AppWorld checks for 260 GiB free. To use existing installations instead, copy `config/env.example.sh` to `config/env.sh`, edit the paths, and run `bash scripts/setup.sh`. On a single-H100 node whose system compiler cannot build Transformer Engine, run `scripts/build_te_single_h100.sh` once.

## Reproducing the paper

Every command runs from the repository root. A smoke run precedes each training run and checks that the run manifest records the method's switches. `--dry-run` prints the resolved plan without using GPUs. Relaunching with the same `--run` name resumes from the last checkpoint. Outputs go to `RUN_ROOT/<run>`.

### Training: Tables 1–3

```bash
MODELS="Qwen3-1.7B-NonThinking-AppendOnly Qwen3-4B-Instruct-2507 Qwen3-8B-NonThinking-AppendOnly"
METHODS="grpo dapo turn-threshold-start turn-threshold-end turn-bandit call-threshold call-uniform call-bandit"
for model in $MODELS; do
  for method in $METHODS; do
    python3 run.py --env bfcl --model "$model" --method "$method" --seed 42 --topology multi_h100
  done
done
```

During training the driver evaluates the 25 validation tasks every 10 optimizer steps. At the end it evaluates the 75 test tasks for the final checkpoint and for the best validation checkpoint.

### Evaluation: Tables 1–3

Evaluate each final checkpoint (`RUN_ROOT/<run>/hf_ep50`) with the official BFCL harness, three decodes per task:

```bash
source config/env.sh
SPLITS=$PWD/config/splits/bfcl_aworld_lineage_seed42_v1
HF=$RUN_ROOT/<run>/hf_ep50
OUT=$RUN_ROOT/<run>_eval

TASK_MANIFEST=$SPLITS/base_eval.json         CATEGORY=multi_turn_base         bash scripts/eval_official.sh "$HF" "$OUT/base"
TASK_MANIFEST=$SPLITS/miss_func_eval.json    CATEGORY=multi_turn_miss_func    bash scripts/eval_official.sh "$HF" "$OUT/miss_func"
TASK_MANIFEST=$SPLITS/miss_param_eval.json   CATEGORY=multi_turn_miss_param   bash scripts/eval_official.sh "$HF" "$OUT/miss_param"
TASK_MANIFEST=$SPLITS/long_context_eval.json CATEGORY=multi_turn_long_context bash scripts/eval_official.sh "$HF" "$OUT/long_context"
```

`summary.json` reports the strict session-success rate of each decode; their mean is Avg@3. Per-task outcomes are under `decode_*/`, and a task counts toward Best@3 if any decode solves it. Tables 1 and 2 use `base`. Table 3 averages the other three categories. The Initial row evaluates the untrained model directory (`$MODELS_DIR/<model>`).

### AppWorld: Table 4

AppWorld's protected material and its derivatives must not be redistributed unencrypted, so this release contains no AppWorld data, guides or task rows. Put verified training guides in a local directory and set `APPWORLD_GUIDES_DIR` in `config/env.sh`. Each guide is `<task_id>.json` with `task_id`, `verifier_pass: true`, and a `turns` list of entries with `code`. Setup accepts guides only for official training-split tasks. It replays every guide in AppWorld to confirm success, measures handoff limits for the selected model, and seals the setup on first use. Guides are never used at evaluation.

```bash
for method in grpo call-threshold; do
  python3 run.py --env appworld --model Qwen3-8B-NonThinking-AppendOnly --method "$method" --seed 42 --topology multi_h100
done
```

After training, `run.py` evaluates Test Normal and Test Challenge with `h = 0` and three decodes, on the final and dev-selected checkpoints. To evaluate an exported checkpoint separately:

```bash
.deps/venvs/molt/bin/python scripts/run_appworld.py --model 8b --method call-threshold --seed 42 \
  --setup <sealed_setup> --run <RUN_ROOT>/<eval_run> --eval-only tests --checkpoint <hf_checkpoint> --execute
```

### Figures

Each figure is drawn from files every run writes:

| Figure | Source |
| --- | --- |
| Figure 2: validation accuracy over rollouts | `RUN_ROOT/<run>/validation_summary.json` (validation every 10 optimizer steps, with generated rollouts) |
| Figure 3 left: group outcome composition | `logs/train.log`: `rollout/group_all_fail_rate`, `rollout/group_all_pass_rate` per step |
| Figure 3 right: training GPU time | `logs/train.log`: `timing/generation`, `timing/step_total` per step, times the number of GPUs |
| Figure 4: rollout starting positions | `curriculum.json.events.jsonl`: one event per completed group, with `h_before`, `selector`, `status` and the candidate positions |

## Repository layout

```text
run.py              entry point
config/             methods, models, splits, GPU topologies, path template
contracts/          BFCL and AppWorld protocol constants
ga_grpo/            guide-replay agent, Threshold curriculum, Uniform/Bandit selectors, launch checks
bfcl_env/           BFCL multi-turn runtime and rollout protocol
appworld_ga/        AppWorld agent, curriculum and environment server pool
scripts/            installation, training drivers, official evaluation
third_party/molt/   Molt training engine with the modifications used here (lock: third_party/molt.lock.json)
tests/              CPU unit tests
```

## Checks

```bash
python3 scripts/verify_vendor.py                 # bundled Molt matches its lock
.deps/venvs/molt/bin/python -m pytest -q -m "not slow"
```

Tests that need installed benchmarks, models or GPUs skip when those are unavailable.

## Licenses

The training engine derives from [NVIDIA-NeMo Molt](https://github.com/NVIDIA-NeMo/labs-molt) (Apache-2.0), with its notices kept in `third_party/molt`. BFCL and the Qwen3 checkpoints are released under Apache-2.0. AppWorld's public portion uses the same license; its protected portion requires encryption when redistributed. See `THIRD_PARTY_NOTICES.md`.
