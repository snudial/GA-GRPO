# GA-GRPO: Guide Assisted GRPO

GA-GRPO trains tool-using agents by replaying a successful guide up to a handoff position and letting the learner complete the remaining interaction. Only the learner's continuation contributes to the policy update. Handoff positions can follow a threshold curriculum or use uniform or bandit selection.

`run.py` is the main entry point for BFCL and AppWorld. This repository contains the implementation, fixed protocol settings, split definitions, and installation tools. Training outputs, checkpoints, result tables, and internal experiment history are excluded.

## Quick start

A fresh clone can display the CLI and resolve an execution plan using Python alone:

```bash
git clone https://github.com/snudial/GA-GRPO.git
cd GA-GRPO
python3 run.py --help
python3 run.py --env bfcl --model Qwen3-4B-Instruct-2507 \
  --method ga-grpo --seed 42 --topology multi_h100 --dry-run
```

`--dry-run` prints a JSON launch plan. It does not prepare datasets, download models, run admission checks, or use GPUs.

## Installation

Training requires Linux, NVIDIA GPUs, and CUDA 13-compatible drivers (driver 580 or newer for the pinned stack). The provided topology profiles cover four B200 GPUs, four H100 GPUs, and one H100 GPU. Model and context capacity depend on the profile; admission checks run before training.

```bash
bash scripts/bootstrap.sh --prefix "$PWD/.deps" --topology multi_h100
```

The installer builds separate training, BFCL, evaluation, and AppWorld environments, downloads the pinned benchmark and initial 4B model, generates BFCL rows, and writes the local `config/env.sh`. Dependencies are recorded in `env/`; model revisions are in `config/models.json`. Installation requires substantial disk space: the bootstrap checks approximately 60 GB for dependencies and recommends at least 150 GB for BFCL outputs. AppWorld execution checks for 260 GiB free.

For an existing installation, copy `config/env.example.sh` to `config/env.sh`, set its paths, then run `bash scripts/setup.sh`. The configuration is ignored by Git. Paths default to this checkout's `.deps` directory; use absolute paths for custom storage. No original cluster checkout is required.

The modified Molt training engine is included in `third_party/molt` with its licenses and source lock. It does not require access to the original private repository. Verify it with `python3 scripts/verify_vendor.py`.

## Training

Run commands from the repository root after installation:

```bash
# BFCL: turn-level threshold curriculum, last eligible starting position
python3 run.py --env bfcl --model Qwen3-4B-Instruct-2507 \
  --method ga-grpo --seed 42 --topology multi_h100

# BFCL: call-level bandit selection
python3 run.py --env bfcl --model Qwen3-4B-Instruct-2507 \
  --method ga-grpo-call-bandit --seed 42 --topology multi_h100

# AppWorld: turn-level threshold curriculum
python3 run.py --env appworld --model Qwen3-4B-Instruct-2507 \
  --method ga-grpo --seed 42 --topology multi_h100

# Unguided baseline
python3 run.py --env bfcl --model Qwen3-4B-Instruct-2507 \
  --method grpo --seed 42 --topology multi_h100
```

| Method | Guide assistance | Handoff selection | Environments |
| --- | --- | --- | --- |
| `grpo` | Off | Autonomous | BFCL, AppWorld |
| `dapo` | Off | Autonomous; DAPO sampling | BFCL, AppWorld |
| `ga-grpo` | On | Turn threshold, end initialization | BFCL, AppWorld |
| `need-based-ga-grpo` | On | Turn threshold, autonomous initialization | BFCL, AppWorld |
| `ga-grpo-call-second-last` | On | Call threshold | BFCL |
| `ga-grpo-call-uniform` | On | Call uniform | BFCL, AppWorld |
| `ga-grpo-call-bandit` | On | Call bandit | BFCL, AppWorld |

Method switches are defined in `config/methods.json`. Training budgets, group sizes, lengths, and evaluation settings are defined in `contracts/`. The default budgets are 80,000 generated rollouts for BFCL and 129,600 for AppWorld. The inherited test protocol uses eight decodes; validation uses three samples. These are implementation settings, not result claims.

A smoke run precedes full training. `--no-smoke` skips it. Outputs and run manifests are written under `RUN_ROOT`. Reusing `--run NAME` resumes an existing run through the checkpoint guards. AppWorld prepares, validates, and seals a setup for the selected model on first use; `--setup NAME` selects a setup directory beneath `APPWORLD_SETUPS`.

Additional registered checkpoints can be downloaded with:

```bash
python3 scripts/fetch_models.py Qwen3-8B-NonThinking-AppendOnly
```

The non-thinking variants preserve the tokenizer serialization required by the runtime. AppWorld supports the prepared 4B, 8B, 1.7B, and 0.6B non-thinking aliases; other registry models are BFCL-only.

## Evaluation

The BFCL training driver evaluates validation and held-out tasks. For standalone official evaluation:

```bash
bash scripts/eval_official.sh /absolute/path/to/hf_checkpoint /absolute/path/to/eval_output
```

For AppWorld, use a validated setup and an exported checkpoint:

```bash
.deps/venvs/molt/bin/python scripts/run_appworld.py \
  --model 4b --method ga-grpo --seed 42 \
  --setup /absolute/path/to/sealed_setup \
  --run /absolute/path/inside/RUN_ROOT/evaluation \
  --eval-only tests --checkpoint /absolute/path/to/hf_checkpoint --execute
```

BFCL pins its simulator, dataset, and evaluator together through `config/bfcl_protocol.sh`. The default `author_lineage` profile uses the pinned checkout's BFCL_v4 data prefix. The optional `crest_compat` profile is a separate strict BFCL_v3 configuration; outputs from these profiles should be reported separately.

## Repository layout

```text
run.py                 Main CLI
config/                Methods, models, portable paths, splits, GPU topology
contracts/             Training and evaluation protocols
ga_grpo/               BFCL agent, guide replay, curriculum, bandit, resume guards
bfcl_env/              BFCL environment adapter and rollout protocol
appworld_ga/           AppWorld agent, curriculum, environment pool
scripts/               Installation, data preparation, training and evaluation
third_party/molt/      Modified training engine and upstream licenses
tests/                 CPU unit tests and optional benchmark integration tests
```

## Data and licenses

BFCL datasets are prepared from the pinned official benchmark checkout. AppWorld data is downloaded through the official package. Successful AppWorld training guides must be supplied locally through `APPWORLD_GUIDES_DIR` in `config/env.sh`. This public repository does not redistribute protected AppWorld data, guides, or derived task rows. Use guides from an authorized local source; see [AppWorld's distribution policy](https://github.com/StonyBrookNLP/appworld#release-disclaimer).

Each guide file must be named `<task_id>.json` and contain `task_id`, `verifier_pass: true`, and a `turns` list whose entries have `code`. Preparation accepts only official training-split tasks, replays the guides to confirm success, and derives model-specific handoff limits. Task rows are generated locally from the official splits; no guidance is used during evaluation.

```bash
# In config/env.sh, set the directory containing your local verified guides:
export APPWORLD_GUIDES_DIR="/absolute/path/to/local/training_guides"
```

The engine derives from [NVIDIA-NeMo Molt](https://github.com/NVIDIA-NeMo/labs-molt), with Apache-2.0 notices retained in `third_party/molt`. Benchmark data and pretrained model weights retain their respective upstream terms. See `THIRD_PARTY_NOTICES.md`.

## Checks

Install the test runner in the training environment with `.deps/bin/uv pip install --python .deps/venvs/molt/bin/python -r requirements-dev.txt`.

```bash
python3 scripts/verify_vendor.py
python3 -m compileall -q run.py ga_grpo bfcl_env appworld_ga scripts tests
.deps/venvs/molt/bin/python -m pytest -q
```

Some integration tests require the installed benchmarks, model files, or optional packages and skip when unavailable. CPU checks and dry runs do not establish successful end-to-end GPU training on a new machine.
