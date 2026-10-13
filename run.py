#!/usr/bin/env python3
"""One entry point for every experiment: an env, a model, a method and a seed.

    python3 run.py --env bfcl --model Qwen3-4B-Instruct-2507 --method ga-grpo --seed 42
    python3 run.py ... --dry-run      # print the training command; touch nothing

Nothing else varies by hand: protocol (splits, budget, group, lengths) is
contracts/<env>.json, model rules are config/models.json + ga_grpo/model_profile.py,
a method is only its switches in config/methods.json, GPU topology is
config/topology/<name>.env (this machine's default: TOPOLOGY in config/env.sh), and
every path comes from config/env.sh. BFCL runs go through a smoke run, then the
generated-rollout driver (scripts/run_seed_rollouts.sh); AppWorld runs go through
scripts/run_appworld.py. Both resume from the last checkpoint after a crash.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ga_grpo.config import env as host_env, path  # noqa: E402


def registry(name):
    return {k: v for k, v in json.loads((ROOT/'config'/name).read_text()).items() if not k.startswith('_')}


METHODS, MODELS = registry('methods.json'), registry('models.json')
TOPOLOGIES = sorted(p.stem for p in (ROOT/'config/topology').glob('*.env'))
MAX_RETRIES = 5
NON_RETRYABLE = 78  # resume guard / budget contract / admission refusal


def labelled_rows(source, init, limit=0):
    """Jump-start rows with the curriculum's initial handoff written into each label."""
    rows = source.read_text().splitlines()
    out = []
    for line in rows[:limit] if limit else rows:
        row = json.loads(line)
        label = json.loads(row['label'])
        assert label['guide_turns'] == -1
        label['curriculum_init'] = init
        row['label'] = json.dumps(label)
        out.append(json.dumps(row) + '\n')
    return ''.join(out)


def admission(args, env, contract):
    if subprocess.call([str(path('MOLT_VENV')/'bin/python'), '-m', 'ga_grpo.admission', '--env', args.env,
                        '--model', env['MODEL'], '--group-size', str(contract['group_size']),
                        '--max-len', str(contract['max_len'])], cwd=ROOT, env=env):
        print('ADMISSION FAILED (see [FAIL] above)', file=sys.stderr)
        sys.exit(NON_RETRYABLE)


def bfcl(args, env, contract, ops):
    spec = METHODS[args.method]['bfcl']
    run_dir = path('RUN_ROOT')/args.run
    env.update(spec)
    env.update(COMPARISON_METHOD=args.method, BFCL_PROFILE='author_lineage', BFCL_TRACE_ROLLOUTS='1',
               MAX_NEW=str(contract['max_new_tokens']), MAX_LEN=str(contract['max_len']),
               BFCL_PROJECT_ROOT=str(run_dir/'bfcl_project'))
    mode = 'jump' if spec['BFCL_CURRICULUM'] == '1' else 'grpo'
    source = ROOT/'data/bfcl'/('bfcl_jumpstart_train.jsonl' if mode == 'jump' else 'bfcl_train.jsonl')
    driver = ['bash', str(ROOT/'scripts/run_seed_rollouts.sh'), mode, spec['ALGO'], str(args.seed), args.run]
    init = spec.get('BFCL_CURRICULUM_INIT')
    if args.dry_run:
        with tempfile.TemporaryDirectory(dir=path('CACHE_ROOT')) as tmp:
            if init:
                env['TRAIN_DATA'] = f'{tmp}/train.jsonl'
                Path(env['TRAIN_DATA']).write_text(labelled_rows(source, init))
            return subprocess.call(driver, cwd=ROOT, env={**env, 'DRY_RUN': '1'})
    if init:
        run_dir.mkdir(parents=True, exist_ok=True)
        env['TRAIN_DATA'] = str(run_dir/'train.jsonl')
        if not Path(env['TRAIN_DATA']).exists():
            Path(env['TRAIN_DATA']).write_text(labelled_rows(source, init))

    # Smoke: 2 accepted groups on 8 tasks, no checkpoints, then the manifest must
    # record exactly the registry's switches. Its run dir is removed after the
    # checks (a passed smoke once left 45G of checkpoint state on a full disk).
    if not args.no_smoke and not (ops/'smoke.passed').exists():
        smoke = f'{args.run}_smoke_{time.strftime("%Y%m%dT%H%M%S")}'
        smoke_dir = path('RUN_ROOT')/smoke
        smoke_dir.mkdir(parents=True, exist_ok=False)
        rows = labelled_rows(source, init, 8) if init else ''.join(r + '\n' for r in source.read_text().splitlines()[:8])
        (smoke_dir/'train.jsonl').write_text(rows)
        smoke_env = {**env, 'TRAIN_DATA': str(smoke_dir/'train.jsonl'), 'EVAL_DATA': str(ROOT/'data/bfcl/bfcl_smoke8.jsonl'),
                     'TRAIN_SEED': str(args.seed), 'N_SAMPLES': str(contract['group_size']), 'ROLLOUT_BS': '2',
                     'TRAIN_BS': '4', 'NUM_EPISODES': '1', 'MAX_TRAIN_GROUPS': '2', 'EVAL_STEPS': '-1',
                     'EVAL_AT_START': '0', 'CKPT_SAVE_STEPS': '-1', 'CKPT_SAVE_HF': '0', 'DISABLE_FINAL_HF_EXPORT': '1',
                     'BFCL_PROJECT_ROOT': str(smoke_dir/'bfcl_project'), 'RAY_TMPDIR': f'/tmp/jg-smoke-{os.getpid()}'}
        with (ops/'smoke.log').open('a') as log:
            if subprocess.call(['bash', str(ROOT/'scripts/train.sh'), smoke], cwd=ROOT, env=smoke_env,
                               stdout=log, stderr=subprocess.STDOUT):
                sys.exit(f'SMOKE FAILED: see {ops/"smoke.log"}')
        manifest = json.loads((smoke_dir/'manifest.json').read_text())
        expected = dict(algo=spec['ALGO'], curriculum=spec['BFCL_CURRICULUM'], curriculum_init=init or 'suffix',
                        handoff_unit=spec.get('BFCL_HANDOFF_UNIT', 'turn'))
        recorded = {k: str(manifest.get(k)) for k in expected}
        traces = list((smoke_dir/'rollout_trace').glob('*.jsonl'))
        if recorded != expected or not traces:
            sys.exit(f'SMOKE CHECK FAILED: manifest {recorded} != {expected} or no rollouts ({len(traces)} trace files)')
        shutil.rmtree(smoke_dir)
        (ops/'smoke.passed').write_text(smoke + '\n')

    return run_resuming(driver, env)


def run_resuming(driver, env):
    """Run a training driver; after a crash (e.g. an environment server's transient error),
    wait for the dead attempt's GPU processes to exit and resume from the last checkpoint."""
    for attempt in range(MAX_RETRIES + 1):
        status = subprocess.call(driver, cwd=ROOT, env={**env, 'RAY_TMPDIR': f'/tmp/jg-full-{os.getpid()}-{attempt}'})
        if status in (0, NON_RETRYABLE):
            return status
        print(f'driver exited {status}; resuming from the last checkpoint (attempt {attempt + 1})', flush=True)
        deadline = time.time() + 900
        while time.time() < deadline and subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                capture_output=True, text=True).stdout.strip():
            time.sleep(15)
    return 1


def appworld_setup(setup, model):
    """Build the sealed AppWorld setup (data, guides, isolated runtime) from this checkout:
    prepare, replay every guide in AppWorld, measure its tokens, then seal the hashes."""
    py = lambda venv: str(path(venv)/'bin/python')
    for cmd in [[py('MOLT_VENV'), 'scripts/prepare_appworld.py', '--out', str(setup)],
                [py('APPWORLD_VENV'), 'scripts/validate_appworld_guides.py', '--setup', str(setup)],
                [py('MOLT_VENV'), 'scripts/validate_appworld_guides.py', '--setup', str(setup), '--token-lengths-only', '--model', model],
                [py('MOLT_VENV'), 'scripts/prepare_appworld.py', '--out', str(setup), '--seal']]:
        subprocess.run(cmd, cwd=ROOT, env={**host_env(), 'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': ''},
                       stdout=subprocess.DEVNULL, check=True)


def appworld(args, env, contract, ops):
    from scripts.run_appworld import MODELS as ALIASES  # the setup's per-model datasets are keyed by alias
    alias = {name: key for key, name in ALIASES.items()}.get(args.model)
    if alias is None:
        sys.exit(f'AppWorld setups prepare datasets for {sorted(ALIASES.values())} only, not {args.model}')
    env.update(METHODS[args.method]['appworld'])
    setup = path('APPWORLD_SETUPS')/args.setup
    if 'artifact_sha256' not in (json.loads((setup/'setup.json').read_text()) if (setup/'setup.json').exists() else {}):
        print(f'preparing AppWorld setup {setup.name} (guide replay + seal, a few minutes)', flush=True)
        appworld_setup(setup, args.model)
    validated = json.loads((setup/'guide_validation/token_lengths.json').read_text())
    if args.model not in validated:
        sys.exit(f'Setup {setup.name} is not validated for {args.model}; choose a new --setup name')
    cmd = [str(path('MOLT_VENV')/'bin/python'), str(ROOT/'scripts/run_appworld.py'), '--model', alias,
           '--setup', str(setup), '--method', args.method, '--seed', str(args.seed),
           '--budget', str(args.budget or contract['total_generated_rollouts'])]
    if args.init_weights:
        cmd += ['--init-weights', args.init_weights]
    if args.dry_run:
        return subprocess.call([*cmd, '--run', str(path('RUN_ROOT')/args.run)], cwd=ROOT, env=env)
    # Smoke: 64 prompt groups (2 optimizer steps), no checkpoint or evaluation; an
    # OPSD method must also have run its distillation stage.
    if not args.no_smoke and not (ops/'smoke.passed').exists():
        smoke = path('RUN_ROOT')/f'{args.run}_smoke_{time.strftime("%Y%m%dT%H%M%S")}'
        with (ops/'smoke.log').open('a') as log:
            if subprocess.call([*cmd, '--run', str(smoke), '--smoke-groups', '64', '--execute'],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT):
                sys.exit(f'SMOKE FAILED: see {ops/"smoke.log"}')
        train_log = (smoke/'logs/train.log').read_text(errors='replace')
        if 'policy_loss' not in train_log or (env.get('OPSD') == '1' and "'opsd/kl'" not in train_log):
            sys.exit(f'SMOKE CHECK FAILED: no optimizer step{" or OPSD stage" if env.get("OPSD") == "1" else ""} '
                     f'in {smoke}/logs/train.log')
        (ops/'smoke.passed').write_text(smoke.name + '\n')
    return run_resuming([*cmd, '--run', str(path('RUN_ROOT')/args.run), '--execute'], env)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', required=True, choices=['bfcl', 'appworld'])
    ap.add_argument('--model', required=True, choices=sorted(MODELS))
    ap.add_argument('--method', required=True, choices=sorted(METHODS))
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--topology', choices=TOPOLOGIES, default=None,
                    help='config/topology/<name>.env (default: TOPOLOGY in config/env.sh; none = built-in B200 defaults)')
    ap.add_argument('--run', help='run name (default: <env>_<method>_<model>_seed<seed>_<topology>)')
    ap.add_argument('--no-smoke', action='store_true', help='skip the smoke run (BFCL)')
    ap.add_argument('--setup', help='AppWorld: sealed setup under APPWORLD_SETUPS (default: the run\'s '
                    'earlier one, else <contract>_<commit>, built on first use)')
    ap.add_argument('--budget', type=int, help='AppWorld: generated-rollout budget (a milestone of the contract; '
                    'default: the contract total)')
    ap.add_argument('--init-weights', help='AppWorld: start training from these HF weights with a fresh optimizer '
                    '(scripts/run_appworld.py --init-weights)')
    ap.add_argument('--dry-run', action='store_true', help='print the resolved launch plan without downloading data or using GPUs')
    args = ap.parse_args()
    if args.env not in METHODS[args.method]:
        ap.error(f'{args.method} is not implemented for {args.env}')
    host = host_env()
    args.topology = args.topology or host.get('TOPOLOGY') or None
    if args.topology and args.topology not in TOPOLOGIES:
        ap.error(f'Unknown topology in config/env.sh: {args.topology}')
    args.run = args.run or f'{args.env}_{args.method}_{args.model}_seed{args.seed}_{args.topology or "default"}'
    for name, value in [('run', args.run), ('setup', args.setup)]:
        if value and (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or value in {'.', '..'}):
            ap.error(f'--{name} must be a single directory name')
    contract = json.loads((ROOT/'contracts'/f'{args.env}.json').read_text())
    model_dir = path('MODELS_DIR')/args.model
    if args.budget is not None and (args.env != 'appworld' or args.budget not in contract['setup_fields']['milestones']):
        ap.error('--budget must be an AppWorld checkpoint milestone')
    if args.dry_run:
        print(json.dumps(dict(environment=args.env, model=str(model_dir), method=args.method,
                             seed=args.seed, topology=args.topology, run=str(path('RUN_ROOT')/args.run),
                             setup=args.setup, budget=args.budget or contract['total_generated_rollouts'],
                             switches=METHODS[args.method][args.env], contract=contract,
                             driver='scripts/run_seed_rollouts.sh' if args.env == 'bfcl' else 'scripts/run_appworld.py'),
                         indent=2))
        return
    if not (ROOT/'config/env.sh').exists():
        ap.error('Run bash scripts/bootstrap.sh first, or configure config/env.sh using env.example.sh')
    if not (model_dir/'config.json').exists():
        ap.error(f'{model_dir} missing: python3 scripts/fetch_models.py {args.model}')
    path('CACHE_ROOT').mkdir(parents=True, exist_ok=True)
    env = {**host, 'MODEL': str(model_dir), 'TRAIN_SEED': str(args.seed), 'TOPOLOGY': args.topology or ''}
    admission(args, env, contract)
    ops = path('RUN_ROOT')/f'{args.run}_ops'
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if args.env == 'appworld':  # a resumed run keeps the setup it started with
        earlier = json.loads((ops/'request.json').read_text()) if (ops/'request.json').exists() else {}
        args.setup = args.setup or earlier.get('setup') or f"{contract['name']}_{args.model}_{commit[:12]}"
    if not args.dry_run:
        ops.mkdir(parents=True, exist_ok=True)
        (ops/'request.json').write_text(json.dumps(dict(
            env=args.env, model=args.model, method=args.method, seed=args.seed, topology=args.topology,
            run=args.run, switches=METHODS[args.method][args.env],
            setup=args.setup, commit=commit),
            indent=2) + '\n')
    # This node's idle-GPU keeper (optional GPU_KEEPER_DIR in env.sh): reserve, then
    # wait until no other job holds a GPU — later launches queue behind earlier ones.
    keeper = host.get('GPU_KEEPER_DIR')
    if keeper and not args.dry_run:
        sys.path.insert(0, keeper)
        import keeper_client
        print(f'waiting for free GPUs (keeper {keeper})', flush=True)
        keeper_client.wait_gpus_free(f'run_py_{args.run}')
    try:
        sys.exit((bfcl if args.env == 'bfcl' else appworld)(args, env, contract, ops))
    finally:
        if keeper and not args.dry_run:
            keeper_client.release(f'run_py_{args.run}')


if __name__ == '__main__':
    main()
