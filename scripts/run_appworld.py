"""Plan or execute one BFCL-matched AppWorld arm, with sparse milestones."""
import argparse
import ast
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.process_pool import stop_process_pool
from scripts.prepare_appworld import DEFAULT_SETUP, prepare
from scripts.appworld_eval_metrics import completed_eval_metrics
from ga_grpo.config import path, topology_defaults
from ga_grpo.model_profile import profile, runtime_defaults
from ga_grpo.checkpoint import copy_checkpoint_file

PYTHON = path('MOLT_VENV')/'bin/python'
APPWORLD_PYTHON = path('APPWORLD_VENV')/'bin/python'
MODELS = {'4b': 'Qwen3-4B-Instruct-2507', '8b': 'Qwen3-8B-NonThinking-AppendOnly',
          '1p7b': 'Qwen3-1.7B-NonThinking-AppendOnly', '0p6b': 'Qwen3-0.6B-NonThinking-AppendOnly'}  # knob suffixes 1P7B / 0P6B
# Test decodes per (checkpoint, split): 3, the paper's Avg@3 / Best@3 protocol. Decode d uses seed
# 20260826 + d; a saved decode result is reused by identity. APPWORLD_TEST_DECODES raises the count.
TEST_DECODES = int(os.environ.get('APPWORLD_TEST_DECODES', '3'))
METHODS = ['grpo', 'call-threshold']
# Guide-assisted AppWorld methods: handoffs between executed code blocks (Call-level in the paper),
# initialized at the guide's last block and moved by the threshold rule (appworld_ga/curriculum.py).
CURRICULUM_METHODS = {'call-threshold'}


def pool_size():
    """AppWorld servers = episodes in flight (one global world each). APPWORLD_POOL_SIZE overrides 256.
    8B (2026-10-06): 256 in flight left KV cache 26-46% used and GPUs at 250-300 W while generating
    (815-945 W training); measure with --smoke-groups."""
    return int(os.environ.get('APPWORLD_POOL_SIZE', '256'))


def knob(name, model):
    """An AppWorld knob for this model: NAME (explicit, any model) wins over NAME_<MODEL>
    (a topology profile's per-model default, e.g. APPWORLD_VLLM_UTIL_8B)."""
    return os.environ.get(name) or os.environ.get(f'{name}_{model.upper()}')


def command(setup, run, model, method, seed, budget, *, evaluation=None, checkpoint=None, decode=0,
            eval_batch=None, single_gpu=False, init_weights=None):
    config = json.loads((setup/'setup.json').read_text())
    args = json.loads((setup/'bfcl_reference.json').read_text())['command'][4:]
    replacements = {
        # init_weights: training starts from these HF weights instead of the base model (a finished run's
        # final checkpoint whose optimizer state is lost); Adam, the data cursor and the LR warmup start fresh.
        '--actor.model_name_or_path': checkpoint or init_weights or path('MODELS_DIR')/MODELS[model],
        '--data.prompt_dataset': setup/'datasets'/f'train_{model}.jsonl',
        '--train.agent_path': Path(config.get('agent_source', ROOT))/'appworld_ga/agent.py', '--train.seed': seed,
        '--train.num_episodes': math.ceil(budget / 16 / 90),
        '--eval.dataset': setup/'datasets'/f'{evaluation or "dev"}.jsonl',
        # APPWORLD_EVAL_STEPS: in-training dev cadence (2026-10-12; a large value turns it off, e.g. when every
        # 10th step's weights are tested by separate 1-GPU jobs instead, slurm/test_every10_watch.sbatch).
        '--eval.steps': int(os.environ.get('APPWORLD_EVAL_STEPS', 10)),
        '--ckpt.output_dir': run/'hf', '--ckpt.path': run/'state',
    }
    if 'context_max_length' in config:
        replacements['--data.max_len'] = config['context_max_length']
    if evaluation:
        replacements.update({'--vllm.gpu_memory_utilization': .85, '--eval.n_samples_per_prompt': 3,
                             '--train.seed': 42})
        if evaluation.startswith('test'):
            replacements.update({'--data.max_len': config.get('context_max_length', 40960), '--eval.temperature': 1e-6,
                                 '--eval.n_samples_per_prompt': 1, '--train.seed': 20260826 + decode,
                                 '--vllm.gpu_memory_utilization': .90})
    for key, value in replacements.items():
        args[args.index(key) + 1] = str(value)
    if not evaluation:  # in-flight prompt groups fill the pool: every server runs an episode
        args[args.index('--rollout.vllm_generate_batch_size') + 1] = str(pool_size() // 16)
        # vLLM's share of each GPU while generating (hybrid: the engines sleep during training, so the
        # actor's 97 GB training peak never meets it). APPWORLD_VLLM_UTIL[_<MODEL>] overrides the setup's 0.50.
        # 8B, pool 512 at 0.50 (2026-10-06): KV cache 66-82% full evicted cached contexts (prefix hit
        # 95% -> 72-82%, prompt recompute 10k -> 27-38k tok/s per engine), generation 256 s -> 392 s.
        if knob('APPWORLD_VLLM_UTIL', model):
            args[args.index('--vllm.gpu_memory_utilization') + 1] = knob('APPWORLD_VLLM_UTIL', model)
        # APPWORLD_FSDP_OFFLOAD[_<MODEL>]=optimizer runs the AdamW step on CPU: the fp32 master and Adam
        # moments (12 bytes/param, a quarter per rank) never sit on the GPU, which removes the optimizer-step
        # peak; numerically the same step (molt/trainer/fsdp/optimizer_offload.py).
        if knob('APPWORLD_FSDP_OFFLOAD', model):
            args += ['--fsdp.offload', knob('APPWORLD_FSDP_OFFLOAD', model)]
    # Every method saves on the setup's cadence (5 cycles). Non-GRPO methods used to save every 2: at 8B one
    # save is ~53 s (92 GB model + Adam, 16 GB HF export), so JUMP spent ~1 h per run on saves that GRPO
    # did not. The curriculum state rides in the checkpoint (samples_generator curriculum_state), so a
    # resume restores it at any cadence.
    args.remove('--ckpt.load_enable')
    if eval_batch:  # concurrent eval prompts; Molt waits for each batch's slowest episode (default 32)
        args += ['--eval.batch_size', str(eval_batch)]
    if single_gpu:  # one decode per GPU; eval-only builds no policy actor, only the engine
        args[args.index('--vllm.num_engines') + 1] = '1'
        args[args.index('--actor.num_gpus_per_node') + 1] = '1'
    if evaluation:
        args.append('--eval.eval_only')
    else:
        if (run/'state/_actor').exists():
            args.append('--ckpt.load_enable')
        if method == 'dapo':
            args.append('--algo.dynamic_filtering_fill_batch')
    # Attention, packing and prefix caching follow the model (ga_grpo/model_profile.py), as in BFCL.
    # Prefix caching on AppWorld 4B (2026-10-03, appworld_ab_20261003/r2_pc{0,1}, 96 groups each):
    # cache hit rate 92%, prompt tokens prefilled 42.6k/s -> 5.3k/s, running requests per engine
    # 7.3 -> 2.6 at equal decode throughput (~2.8x shorter LLM time per turn), vllm_kl unchanged.
    kernels = runtime_defaults(profile(path('MODELS_DIR')/MODELS[model]))  # checkpoints share the base architecture
    args[args.index('--fsdp.attn_implementation') + 1] = kernels['ATTN_IMPLEMENTATION']
    if '--fsdp.packing_samples' in args and kernels['PACKING'] != '1':
        args.remove('--fsdp.packing_samples')
    elif '--fsdp.packing_samples' not in args and kernels['PACKING'] == '1':
        args.append('--fsdp.packing_samples')
    # VLLM_PREFIX_CACHING=0|1 overrides the model rule (measure with --smoke-groups).
    prefix_caching = os.environ.get('VLLM_PREFIX_CACHING') or kernels['VLLM_PREFIX_CACHING']
    if prefix_caching == '1' and '--vllm.enable_prefix_caching' not in args:
        args.append('--vllm.enable_prefix_caching')
    elif prefix_caching == '0' and '--vllm.enable_prefix_caching' in args:
        args.remove('--vllm.enable_prefix_caching')
    # Molt runs every episode's agent code, tokenization and grading inside --rollout.num_runners
    # processes (Molt default 2), each GIL-bound. 8 runners with 256 /dev/shm servers gave 12.9
    # rollouts/s vs 7.9 for 2 runners, 128 Lustre servers (2026-10-04, appworld_ab_20261004/s{a..e});
    # 16 runners measured 11.1. APPWORLD_NUM_RUNNERS overrides it (measure with --smoke-groups).
    args += ['--rollout.num_runners', os.environ.get('APPWORLD_NUM_RUNNERS', '8')]
    # Opt-in training-side knobs under measurement (same loss, different recompute / micro-batching):
    # APPWORLD_GRAD_CKPT=full|selective|none, APPWORLD_DYNAMIC_BATCH_TOKENS=<tokens per GPU micro-batch>.
    if os.environ.get('APPWORLD_GRAD_CKPT'):
        args[args.index('--actor.gradient_checkpoint') + 1] = os.environ['APPWORLD_GRAD_CKPT']
    if os.environ.get('APPWORLD_DYNAMIC_BATCH_TOKENS'):
        if '--fsdp.packing_samples' not in args:
            raise RuntimeError('APPWORLD_DYNAMIC_BATCH_TOKENS needs packed training (model_profile PACKING=1)')
        args += ['--train.dynamic_batch_enable', '--train.max_tokens_per_gpu', os.environ['APPWORLD_DYNAMIC_BATCH_TOKENS']]
    return [str(PYTHON), '-u', '-m', 'molt.cli.train_rl_ray', *args]


@contextlib.contextmanager
def environment_pool(setup, first_port, count):
    # APPWORLD_SERVER_DIR (a local tmpfs such as /dev/shm) holds the servers' working dirs and a
    # copy of the task data for this pool's lifetime. On Lustre, env-only episodes ran 27/s at 64
    # servers and 19/s at 192 (initialize p95 23 s); on /dev/shm 48/s and 33/s (2026-10-04,
    # appworld_ab_20261004/envbench). server.log is copied back to setup/servers/<port> at exit.
    fast = os.environ.get('APPWORLD_SERVER_DIR', '/dev/shm')  # '' keeps them on the setup's filesystem
    base = Path(fast)/f'appworld_servers_{os.getpid()}' if fast else setup/'servers'
    if fast:  # a killed driver never reaches the cleanup below; drop dirs whose driver is gone
        for stale in Path(fast).glob('appworld_servers_*'):
            pid = stale.name.rsplit('_', 1)[-1]
            if pid.isdigit() and not Path(f'/proc/{pid}').exists():
                shutil.rmtree(stale, ignore_errors=True)
    data = Path(json.loads((setup/'setup.json').read_text())['data'])
    if fast:
        shutil.copytree(data, base/'data', symlinks=True)
        data = base/'data'
    processes = []
    streams = []
    agent_source = str(json.loads((setup/'setup.json').read_text())['agent_source'])

    def launch(port):
        root = base/str(port)
        env = {**os.environ, 'APPWORLD_ROOT': str(root), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': agent_source}
        stream = (root/'server.log').open('a')
        streams.append(stream)
        return subprocess.Popen([str(APPWORLD_PYTHON), '-m', 'uvicorn',
            config_server_entrypoint(setup), '--host', '127.0.0.1', '--port', str(port), '--timeout-keep-alive', '30'],
            cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    stop = threading.Event()

    def supervise():
        # A client kills a server whose cell hung (appworld_ga.environment.kill_server);
        # restart it on its port. The hung episode keeps the lease until it answers again.
        restarts, started = {}, {}
        while not stop.wait(1.0):
            for i, process in enumerate(processes):
                port = first_port + i
                if process.poll() is not None and not stop.is_set():
                    # A server the client retired for memory (World.close) or killed for a hung cell is
                    # relaunched; the cap of 5 only stops a crash loop (exits within 60 s of launch).
                    if time.monotonic() - started.get(port, 0) >= 60:
                        restarts[port] = 0
                    if restarts.get(port, 0) >= 5:
                        continue
                    restarts[port] = restarts.get(port, 0) + 1
                    print(f'AppWorld server {port} exited ({process.returncode}); restart {restarts[port]}/5', flush=True)
                    processes[i] = launch(port); started[port] = time.monotonic()
    supervisor = threading.Thread(target=supervise, name='appworld-pool-supervisor', daemon=True)

    def wait_ready():
        deadline = time.monotonic() + 180
        pending = set(range(first_port, first_port + count))
        while pending:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('An AppWorld server exited; inspect servers/*/server.log')
            for port in list(pending):
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=1) as response:
                        if response.status == 200:
                            pending.remove(port)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f'AppWorld servers did not start: {pending}')
            time.sleep(.2)

    def recycle():
        """Replace every server with a fresh process (same port and working dir). The official AppWorld
        serve app keeps host memory per episode it ran (5-12 MB; 256 servers held 576 GB after 72k
        rollouts and a 1.7B guide-assisted job was OOM-killed, 2026-10-10), so the pool is
        recycled between training segments, when no episode is in flight. APPWORLD_POOL_RECYCLE=0 keeps
        the servers for the whole run."""
        nonlocal supervisor
        if os.environ.get('APPWORLD_POOL_RECYCLE', '1') == '0':
            return
        stop.set()
        if supervisor.is_alive():
            supervisor.join()
        stop_process_pool(processes)
        for stream in streams:
            stream.close()
        streams.clear()
        stop.clear()
        processes[:] = [launch(port) for port in range(first_port, first_port + count)]
        wait_ready()
        supervisor = threading.Thread(target=supervise, name='appworld-pool-supervisor', daemon=True)
        supervisor.start()
        print(f'AppWorld pool recycled ({count} servers)', flush=True)
    try:
        for port in range(first_port, first_port + count):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(('127.0.0.1', port))
            root = base/str(port)
            root.mkdir(parents=True, exist_ok=True)
            if not (root/'data').exists():
                (root/'data').symlink_to(data, target_is_directory=True)
            processes.append(launch(port))
        wait_ready()
        supervisor.start()
        yield recycle
    finally:
        stop.set()
        if supervisor.is_alive():
            supervisor.join()
        try:
            stop_process_pool(processes)
        finally:
            for stream in streams:
                stream.close()
            if fast:
                for port in range(first_port, first_port + count):
                    if (base/str(port)/'server.log').exists():
                        (setup/'servers'/str(port)).mkdir(parents=True, exist_ok=True)
                        with (setup/'servers'/str(port)/'server.log').open('ab') as out:
                            out.write((base/str(port)/'server.log').read_bytes())
                shutil.rmtree(base, ignore_errors=True)


def execute(cmd, setup, run, env, log):
    print('EXEC', ' '.join(cmd), flush=True)
    with log.open('a') as stream:
        subprocess.run(cmd, cwd=setup/'runtime', env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def train(cmd, setup, run, env, log):
    """One training process = one restart segment of the training ledger (appworld_ga/ledger.py).

    segments.jsonl records the checkpoint each segment resumed from and where it ended, so rollouts
    regenerated after a restart count as generation cost but join only the update that trained them."""
    from ga_grpo.resume_guard import load_latest

    def position():
        if not (run/'state/_actor/latest').is_file():
            return dict(checkpoint=None, global_step=0, consumed_groups=0)
        tag, state = load_latest(run)
        return dict(checkpoint=tag.name, global_step=state['global_step'],
                    consumed_groups=state['total_consumed_prompts'])

    ledger, segment = run/'ledger', f"{time.strftime('%Y%m%dT%H%M%S')}-{os.urandom(4).hex()}"
    ledger.mkdir(exist_ok=True)

    def record(**row):
        with (ledger/'segments.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(segment_id=segment, time=time.time(), **row)) + '\n')

    start = position()
    # The trainer stamps the policy version only after its first update; until then it is the checkpoint's.
    (run/'policy_version.txt').write_text(str(start['global_step']))
    record(event='start', slurm_job=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname(),
           budget_groups=int(env['APPWORLD_MAX_PROMPT_GROUPS']), **start)
    status = 'failed'
    try:
        execute(cmd, setup, run, {**env, 'APPWORLD_LEDGER_DIR': str(ledger), 'APPWORLD_SEGMENT_ID': segment,
                                  'EXP2_VERSION_FILE': str(run/'policy_version.txt')}, log)
        status = 'completed'
    finally:
        record(event='end', status=status, **position())


def _split_size(setup, split):
    return len((setup/'datasets'/f'{split}.jsonl').read_text().splitlines())


_EVAL_START_LOCK = threading.Lock()


def _eval_job(setup, run, args, weights, split, decode, env, tag, gpu=None):
    """One decode of one split: reuse a matching result, else run it and verify the records."""
    revision = json.loads((setup/"setup.json").read_text())["environment_revision"]
    log = run/f'{tag}_{split}_decode{decode}.log'
    result_file = log.with_suffix('.result.json')
    identity = dict(environment_revision=revision, checkpoint=str(Path(weights).resolve()), split=split, decode=decode,
                    shards=[(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                            for p in sorted(Path(weights).glob('*.safetensors'))])
    identity = json.loads(json.dumps(identity))
    if result_file.exists():
        saved = json.loads(result_file.read_text())
        if saved['identity'] != identity:
            raise RuntimeError('Evaluation output belongs to another checkpoint; choose a new output run')
        return saved['score'], saved['records']
    samples = 1 if decode else 3
    # The whole split in flight at once: Molt's eval waits for each batch's slowest episode, so
    # one batch leaves a single tail. Episodes beyond the env pool queue on server leases, which
    # parallel jobs share.
    batch = _split_size(setup, split)
    cmd = command(setup, run, args.model, args.method, args.seed, args.budget,
                  evaluation=split, checkpoint=weights, decode=decode, eval_batch=batch, single_gpu=gpu is not None)
    record_path = run/f'{tag}_{split}_decode{decode}_attempt{time.time_ns()}.jsonl'
    eval_env = {**env, 'APPWORLD_CURRICULUM': '0',
                'APPWORLD_EVAL_RECORDS': str(record_path),
                'APPWORLD_SEED': str(20260826 + decode if decode else 42)}
    eval_env.pop('EXP2_HYBRID', None)
    if gpu is not None:
        # The gpu-th GPU this process may use: inside a SLURM job, one of the job's own devices.
        visible = [g for g in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if g.strip()]
        eval_env.update(CUDA_VISIBLE_DEVICES=visible[gpu] if visible else str(gpu),
                        RAY_TMPDIR=f'/tmp/aw-eval-{os.getpid()}-{gpu}')
    log_offset = log.stat().st_size if log.exists() else 0
    with log.open('a') as stream:
        with _EVAL_START_LOCK:
            # Evaluation processes that start within the same second hang right after "Started a local Ray
            # instance" (3 of 6 final-test decodes, 0.6B and 1.7B guide-assisted runs, 2026-10-10/11: each sat
            # idle for hours on its GPU). Each process starts its own Ray instance (RAY_ADDRESS=local) and
            # the simultaneous port probing is the only thing they share, so starts are spaced out.
            started = subprocess.Popen(cmd, cwd=setup/'runtime', env=eval_env, stdout=stream, stderr=subprocess.STDOUT)
            print('EXEC', ' '.join(cmd), flush=True)
            time.sleep(float(os.environ.get('APPWORLD_EVAL_START_SPACING', '15')))
        if started.wait() != 0:
            raise subprocess.CalledProcessError(started.returncode, cmd)
    with log.open('rb') as stream:
        stream.seek(log_offset)
        completed = completed_eval_metrics(stream.read().decode(errors='replace'))
    score = float(completed['eval_default_pass1'])
    from collections import Counter
    rows = [json.loads(line) for line in record_path.read_text().splitlines()]
    task_ids = [json.loads(json.loads(line)['label'])['task_id'] for line in
                (setup/'datasets'/f'{split}.jsonl').read_text().splitlines()]
    assert Counter(row['task_id'] for row in rows) == Counter({t: samples for t in task_ids})
    assert completed['eval_num_samples'] == len(rows)
    assert all(row['handoff_h'] == 0 for row in rows)
    assert abs(sum(row['success'] for row in rows)/len(rows) - score) < 1e-5
    result_file.write_text(json.dumps(dict(identity=identity, score=score, records=str(record_path)), indent=2))
    return score, str(record_path)


def _eval_gpus():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        return len([g for g in visible.split(',') if g.strip()])
    listing = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.splitlines()
    return sum(1 for line in listing if line.startswith('GPU '))


def _summary(setup, weights, split, scores, records):
    revision = json.loads((setup/"setup.json").read_text())["environment_revision"]
    summary = {'environment_revision': revision, 'decode_scores': scores,
               'checkpoint': str(weights), 'split': split, 'h': 0, 'task_records': records}
    if split.startswith('test'):  # mean over TEST_DECODES decodes; mean_first3 = the pre-2026-10-05 protocol
        summary.update(mean=sum(scores)/len(scores), decodes=len(scores), mean_first3=sum(scores[:3])/len(scores[:3]))
    else:  # one dev job with 3 samples per task
        summary['avg_at_3'] = sum(scores)/len(scores)
    return summary


def evaluate(setup, run, args, weights, split, env, tag):
    decodes = range(1, TEST_DECODES + 1) if split.startswith('test') else [0]
    gpu = 0 if _eval_gpus() == 1 else None  # a one-GPU job evaluates on one engine
    done = [_eval_job(setup, run, args, weights, split, d, env, tag, gpu=gpu) for d in decodes]
    return _summary(setup, weights, split, [s for s, _ in done], [r for _, r in done])


def evaluate_tests_many(setup, args, env, items):
    """Every (checkpoint, split, decode) of ``items`` = [(run, weights, tag, splits)] through one GPU
    pool, one engine per decode, longest split first, so a straggling decode overlaps other work
    instead of idling the GPUs that finished (2026-10-05: per-checkpoint pools left GPUs idle 7-9
    min). Each decode keeps its own seeds, so only the schedule changes."""
    import concurrent.futures
    import queue
    gpus = queue.Queue()
    for g in range(_eval_gpus()):
        gpus.put(g)

    def job(i, split, decode):
        run, weights, tag, _ = items[i]
        g = gpus.get()
        try:
            return _eval_job(setup, run, args, weights, split, decode, env, tag, gpu=g)
        finally:
            gpus.put(g)
    jobs = sorted(((i, s, d) for i, it in enumerate(items) for s in it[3] for d in range(1, TEST_DECODES + 1)),
                  key=lambda j: -_split_size(setup, j[1]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpus.qsize()) as pool:
        out = dict(zip(jobs, pool.map(lambda j: job(*j), jobs)))
    decodes = range(1, TEST_DECODES + 1)
    return [{s: _summary(setup, it[1], s, [out[(i, s, d)][0] for d in decodes],
                         [out[(i, s, d)][1] for d in decodes]) for s in it[3]} for i, it in enumerate(items)]


def evaluate_tests(setup, run, args, weights, env, tag, splits=('test_normal', 'test_challenge')):
    return evaluate_tests_many(setup, args, env, [(run, weights, tag, splits)])[0]


def sweep_dataset(setup, run, model, fractions, tasks=0):
    """Guided train tasks as eval rows at fixed handoffs h = floor(f * guide_H), capped at the
    task's token-measured h_max, so the curriculum and the OPSD bank never see them."""
    rows, used = [], 0
    for line in (setup/'datasets'/f'train_{model}.jsonl').read_text().splitlines():
        row = json.loads(line)
        label = json.loads(row['label'])
        if not label.get('guide_path') or (tasks and used == tasks):
            continue
        used += 1
        for h in sorted({min(label['h_max'], int(f * label['guide_H'])) for f in fractions}):
            rows.append(json.dumps({**row, 'label': json.dumps({**label, 'eval': True, 'eval_h': h})}))
    text = '\n'.join(rows) + '\n'
    import hashlib
    dataset = run/f'handoff_sweep_{hashlib.sha256(text.encode()).hexdigest()[:12]}.jsonl'
    if not dataset.exists():
        dataset.write_text(text)
    return dataset


def _sweep_job(setup, run, args, weights, dataset, env, tag, gpu):
    """Every sweep row of one checkpoint on one GPU, sampled like training rollouts."""
    import hashlib
    from collections import Counter, defaultdict
    revision = json.loads((setup/"setup.json").read_text())["environment_revision"]
    log = run/f'{tag}_handoff_sweep.log'
    result_file = run/f'{tag}_handoff_sweep.json'
    identity = dict(environment_revision=revision, checkpoint=str(Path(weights).resolve()),
                    dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    samples=args.sweep_samples, temperature=args.sweep_temperature,
                    shards=[(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                            for p in sorted(Path(weights).glob('*.safetensors'))])
    identity = json.loads(json.dumps(identity))
    if result_file.exists():
        saved = json.loads(result_file.read_text())
        if saved['identity'] != identity:
            raise RuntimeError('Sweep output belongs to another checkpoint or dataset; choose a new output run')
        return saved
    labels = [json.loads(json.loads(line)['label']) for line in dataset.read_text().splitlines()]
    cmd = command(setup, run, args.model, args.method, args.seed, args.budget, evaluation='dev',
                  checkpoint=weights, eval_batch=len(labels), single_gpu=True)
    for key, value in [('--eval.dataset', dataset), ('--eval.n_samples_per_prompt', args.sweep_samples),
                       ('--eval.temperature', args.sweep_temperature)]:
        cmd[cmd.index(key) + 1] = str(value)
    attempt = time.time_ns()
    record_path = run/f'{tag}_handoff_sweep_attempt{attempt}.jsonl'
    eval_env = {**env, 'APPWORLD_CURRICULUM': '0', 'APPWORLD_EVAL_RECORDS': str(record_path), 'APPWORLD_SEED': '42',
                'APPWORLD_TRAJECTORY_DIR': str(run/f'{tag}_handoff_sweep_trajectories_attempt{attempt}'),
                'CUDA_VISIBLE_DEVICES': str(gpu), 'RAY_TMPDIR': f'/tmp/aw-sweep-{os.getpid()}-{gpu}'}
    eval_env.pop('EXP2_HYBRID', None)
    log_offset = log.stat().st_size if log.exists() else 0
    execute(cmd, setup, run, eval_env, log)
    with log.open('rb') as stream:
        stream.seek(log_offset)
        completed = completed_eval_metrics(stream.read().decode(errors='replace'))
    rows = [json.loads(line) for line in record_path.read_text().splitlines()]
    assert Counter((r['task_id'], r['handoff_h']) for r in rows) == Counter(
        {(l['task_id'], l['eval_h']): args.sweep_samples for l in labels})
    assert completed['eval_num_samples'] == len(rows)
    wins = defaultdict(lambda: defaultdict(int))
    for r in rows:
        wins[r['task_id']][r['handoff_h']] += r['success']
    tasks = {l['task_id']: dict(guide_H=l['guide_H'], h_max=l['h_max']) for l in labels}
    for task, item in tasks.items():
        item['success_by_h'] = {h: n / args.sweep_samples for h, n in sorted(wins[task].items())}
    curve = [dict(fraction=f, success=sum(t['success_by_h'][min(t['h_max'], int(f * t['guide_H']))]
                                          for t in tasks.values()) / len(tasks)) for f in args.sweep_fractions]
    result = dict(identity=identity, checkpoint=str(weights), success_by_fraction=curve, tasks=tasks,
                  records=str(record_path))
    result_file.write_text(json.dumps(result, indent=2))
    return result


def handoff_sweep_many(setup, args, env, items, dataset):
    """Each checkpoint of ``items`` = [(run, weights, tag)] on its own GPU, as in evaluate_tests_many."""
    import concurrent.futures
    import queue
    gpus = queue.Queue()
    for g in range(_eval_gpus()):
        gpus.put(g)

    def job(item):
        g = gpus.get()
        try:
            return _sweep_job(setup, item[0], args, item[1], dataset, env, item[2], g)
        finally:
            gpus.put(g)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpus.qsize()) as pool:
        return list(pool.map(job, items))


def reuse_final_evaluation(best, final):
    """Reuse the completed test only when selection identifies the exact final artifact."""
    if (best['optimizer_step'] != final['optimizer_step']
            or Path(best['checkpoint']).resolve() != Path(final['checkpoint']).resolve()):
        return None
    return dict(best=best, final=final, test_normal=final['test_normal'],
                test_challenge=final['test_challenge'],
                evaluation_provenance={'mode': 'reuse_exact_selected_final_artifact'})


def main():
    # The machine's topology profile (TOPOLOGY in config/env.sh, or run.py --topology) sets the
    # AppWorld knobs below as defaults, e.g. multi_h100's APPWORLD_VLLM_UTIL_4B; an exported value wins.
    # Before the parser: --pool-size reads APPWORLD_POOL_SIZE for its default.
    for key, value in topology_defaults(os.environ.get('TOPOLOGY'), 'APPWORLD_').items():
        os.environ.setdefault(key, value)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, default=DEFAULT_SETUP)
    parser.add_argument('--model', choices=MODELS, default='8b')
    parser.add_argument('--method', choices=METHODS, default='grpo')
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    parser.add_argument('--budget', type=int, help='Cumulative generated trajectories, including filtered groups')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--pool-size', type=int, default=pool_size())  # = in-flight episodes (groups x 16)
    parser.add_argument('--first-port', type=int, default=19300)
    parser.add_argument('--execute', action='store_true', help='Default is a CPU-only plan')
    parser.add_argument('--eval-only', choices=['dev', 'test_normal', 'test_challenge', 'tests'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--init-weights', type=Path, help='training only: start from these HF weights (a finished '
                        'run\'s weights/r<N>) with a fresh optimizer, data cursor and warmup; recorded in the contract')
    parser.add_argument('--eval-spec', type=Path, help='JSON [{run, weights, tag?, splits?}]: test decodes of '
                        'several checkpoints through one GPU pool (implies --eval-only tests)')
    parser.add_argument('--handoff-sweep', type=Path, help='JSON [{run, weights, tag?}]: success of each checkpoint on '
                        'the guided train tasks at fixed handoffs, one GPU per checkpoint (eval only)')
    parser.add_argument('--sweep-fractions', default='0,0.2,0.4,0.6,0.8',
                        type=lambda s: tuple(float(x) for x in s.split(',')), help='handoff h = floor(f * guide_H)')
    parser.add_argument('--sweep-samples', type=int, default=4)
    parser.add_argument('--sweep-temperature', type=float, default=1.0, help='training rollout temperature')
    parser.add_argument('--sweep-tasks', type=int, default=0, help='first N guided train tasks only (smoke); 0 = all')
    parser.add_argument('--smoke-groups', type=int, default=0,
                        help='train only this many prompt groups, no checkpoint or evaluation (smoke / A-B timing)')
    parser.add_argument('--smoke-resume', action='store_true', help='smoke that checkpoints every cycle and keeps '
                        'its state: a second smoke into the same --run with more groups resumes (restart check)')
    args = parser.parse_args()
    setup = args.setup.resolve()
    config = prepare(setup)
    args.budget = args.budget or config.get('total_generated_rollouts', 80000)
    if args.budget not in config['milestones'] and not args.smoke_groups:
        parser.error('--budget must be a configured checkpoint milestone')
    run = args.run or path('RUN_ROOT')/f'appworld_{args.model}_{args.method}_seed{args.seed}_{setup.name}'
    run = run.resolve()
    if not run.is_relative_to(path('RUN_ROOT')):
        parser.error('AppWorld outputs must be inside owned GA-GRPO runs')
    if args.handoff_sweep:
        args.eval_only = 'dev'
        args.checkpoint = Path(json.loads(args.handoff_sweep.read_text())[0]['weights'])
    if args.eval_spec:
        args.eval_only = 'tests'
        args.checkpoint = Path(json.loads(args.eval_spec.read_text())[0]['weights'])
    if args.eval_only and not args.checkpoint:
        parser.error('--eval-only requires --checkpoint')
    if not 1 <= args.pool_size <= 512 or args.pool_size % 16:
        parser.error('--pool-size must be a multiple of 16 in [16,512]')
    os.environ['APPWORLD_POOL_SIZE'] = str(args.pool_size)  # command() sizes the in-flight groups from it
    phases = [b for b in config['milestones'] if b <= args.budget]
    plan = dict(model=args.model, method=args.method, seed=args.seed, run=str(run),
                milestones=phases, splits=config['splits'], environment_workers=args.pool_size,
                command=command(setup, run, args.model, args.method, args.seed, phases[0],
                                evaluation='test_normal' if args.eval_only == 'tests' else args.eval_only,
                                checkpoint=args.checkpoint, decode=1, init_weights=args.init_weights),
                retention=dict(weights=config['milestones'], val_best=True,
                               optimizer='latest complete model + Adam + loader + curriculum; retained at final'),
                selection='in-training dev plus final dev; extra milestone evaluations excluded, earliest step on ties (BFCL)',
                reward='strict terminal binary; no overlong shaping',
                context_policy=dict(max_length=config.get('context_max_length', 10000),
                                    test_max_length=config.get('context_max_length', 40960),
                                    generation_ceiling=4096, reserve_generation_space=False),
                total_generated_rollouts=args.budget, data_epochs=args.budget//1440,
                init_weights=str(args.init_weights.resolve()) if args.init_weights else None,
                max_turns=config.get('max_turns', 30), max_interactions=200, observation_max_chars=config['observation_max_chars'], prompt_profile=config['prompt_profile'], server_entrypoint=config['server_entrypoint'],
                gpu_optimizations=config.get('gpu_optimizations', {}).get(args.model, {}))
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if not config.get('artifact_sha256'):
        raise RuntimeError('Run CPU validation and prepare_appworld.py --seal before launching GPUs')
    query = ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']
    if os.environ.get('CUDA_VISIBLE_DEVICES'):  # only the GPUs this job owns
        query += ['-i', os.environ['CUDA_VISIBLE_DEVICES']]
    busy = subprocess.check_output(query, text=True).strip()
    if busy:
        raise RuntimeError('GPUs have active processes; leave the BFCL run running and retry when idle')
    if shutil.disk_usage(run.parent).free < 260 * 1024**3:
        raise RuntimeError('Need at least 260 GiB of checkpoint headroom')
    run.mkdir(parents=True, exist_ok=True)
    # Node-wide locks keep two drivers off one machine's GPUs. One-GPU SLURM jobs already own their GPU and
    # may share a node, so they pass a job-private APPWORLD_LOCK_DIR (and their own --first-port).
    locks = Path(os.environ['APPWORLD_LOCK_DIR']) if os.environ.get('APPWORLD_LOCK_DIR') else None
    if locks:
        locks.mkdir(parents=True, exist_ok=True)
    with (locks/'execution.lock' if locks else setup/'execution.lock').open('a') as lock, \
            (locks/'gpu.lock' if locks else path('APPWORLD_SETUPS')/'comparison_queue_watchdog.lock').open('a') as gpu_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = run/'experiment_contract.json'
        identity = {k: v for k, v in plan.items() if k not in ('command', 'milestones')}
        if contract.exists() and json.loads(contract.read_text()) != identity:
            raise RuntimeError('Existing run contract differs; use a new run directory')
        contract.write_text(json.dumps(identity, indent=2))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('BFCL_', 'APPWORLD_'))
               and k not in ('MOLT_GLOO_REUSE_BUFFER', 'MOLT_REUSE_ACTOR_CACHE')}
        env.update(PYTHONPATH=os.pathsep.join([str(config.get('agent_source', ROOT)), str(ROOT), str(setup/'runtime')]),
                   RAY_ADDRESS='local', RAY_USAGE_STATS_ENABLED='0', EXP2_HYBRID='1',
                   VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='true',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   APPWORLD_METHOD=args.method, APPWORLD_SEED=str(args.seed),
                   APPWORLD_MAX_TURNS=str(config.get('max_turns', 30)), APPWORLD_MAX_PROMPT_GROUPS=str(args.budget//16),
                   APPWORLD_POOL=f'{args.first_port}-{args.first_port+args.pool_size-1}',
                   APPWORLD_CURRICULUM='1' if args.method in CURRICULUM_METHODS else '0',
                   EXP2_CURRICULUM_STATE=str(run/'curriculum.json'))
        # OPSD and other EXP2_ switches (config/methods.json via run.py) apply to
        # training only: evaluation must not allocate the EMA teacher.
        training_only = {k: env.pop(k) for k in [k for k in env if k == 'OPSD' or
                         (k.startswith('EXP2_') and k not in ('EXP2_HYBRID', 'EXP2_CURRICULUM_STATE'))]}
        if training_only.pop('OPSD', '0') == '1':
            training_only = {'EXP2_EMA_BETA': '0.99', 'EXP2_OPSD_LAMBDA': '1.0', **training_only,
                             'EXP2_HOOK_DIR': str(ROOT/'ga_grpo'), 'EXP2_BANK_DIR': str(run/'opsd_bank'),
                             'EXP2_VERSION_FILE': str(run/'policy_version.txt')}
        env.update(config.get('gpu_optimizations', {}).get(args.model, {}))
        env.update(MOLT_LINEAR_BACKEND='te')  # one Linear layout across attention backends (model_profile)
        env.update(APPWORLD_RELIABLE_RPC='1', APPWORLD_PROMPT_PROFILE=config['prompt_profile'], APPWORLD_OBSERVATION_MAX_CHARS=str(config['observation_max_chars']))
        if os.environ.get('APPWORLD_EVAL_REFILL'):  # continuous-refill evaluation (eval_refill.py); env drops APPWORLD_*
            env['APPWORLD_EVAL_REFILL'] = os.environ['APPWORLD_EVAL_REFILL']
        with environment_pool(setup, args.first_port, args.pool_size) as recycle_pool:
            if args.smoke_groups:
                env['APPWORLD_MAX_PROMPT_GROUPS'] = str(args.smoke_groups)
                cmd = command(setup, run, args.model, args.method, args.seed, args.smoke_groups * 16,
                              init_weights=args.init_weights)
                cmd[cmd.index('--ckpt.save_steps') + 1] = '1' if args.smoke_resume else '1000000'
                (run/'logs').mkdir(exist_ok=True)
                try:
                    train(cmd, setup, run, {**env, **training_only}, run/'logs/train.log')
                finally:
                    if not args.smoke_resume:
                        shutil.rmtree(run/'state', ignore_errors=True)  # a smoke is never resumed; ~53 GB
                return
            if args.handoff_sweep:
                items = [(Path(i['run']).resolve(), Path(i['weights']), i.get('tag', 'sweep'))
                         for i in json.loads(args.handoff_sweep.read_text())]
                for item_run, *_ in items:
                    if not item_run.is_relative_to(path('RUN_ROOT')):
                        raise RuntimeError(f'{item_run}: AppWorld outputs must be inside owned GA-GRPO runs')
                    item_run.mkdir(parents=True, exist_ok=True)
                dataset = sweep_dataset(setup, run, args.model, args.sweep_fractions, args.sweep_tasks)
                results = handoff_sweep_many(setup, args, env, items, dataset)
                (run/'handoff_sweep_summary.json').write_text(json.dumps(
                    [dict(tag=tag, run=str(item_run), checkpoint=r['checkpoint'], success_by_fraction=r['success_by_fraction'])
                     for (item_run, _, tag), r in zip(items, results)], indent=2))
                return
            if args.eval_spec:  # several checkpoints' test decodes through one GPU pool
                items = [(Path(i['run']).resolve(), Path(i['weights']), i.get('tag', 'evaluation'),
                          tuple(i.get('splits', ('test_normal', 'test_challenge'))))
                         for i in json.loads(args.eval_spec.read_text())]
                for item_run, *_ in items:
                    if not item_run.is_relative_to(path('RUN_ROOT')):
                        raise RuntimeError(f'{item_run}: AppWorld outputs must be inside owned GA-GRPO runs')
                    item_run.mkdir(parents=True, exist_ok=True)
                for (item_run, _, tag, _), result in zip(items, evaluate_tests_many(setup, args, env, items)):
                    (item_run/f'{tag}_tests.json').write_text(json.dumps(result, indent=2))
                return
            if args.eval_only == 'tests':  # every test split and decode in parallel, one GPU each
                result = evaluate_tests(setup, run, args, args.checkpoint, env, 'evaluation')
                (run/'evaluation_tests.json').write_text(json.dumps(result, indent=2))
                return
            if args.eval_only:
                result = evaluate(setup, run, args, args.checkpoint, args.eval_only, env, 'evaluation')
                (run/f'evaluation_{args.eval_only}.json').write_text(json.dumps(result, indent=2))
                return
            from ga_grpo.resume_guard import load_latest, validate_state
            from ga_grpo.checkpoint import read_curve
            records = []
            if not (run/'baseline_dev.json').exists():
                baseline = evaluate(setup, run, args, path('MODELS_DIR')/MODELS[args.model], 'dev', env, 'baseline')
                (run/'baseline_dev.json').write_text(json.dumps(baseline, indent=2))
            for budget in phases:
                marker = run/f'phase_r{budget}.json'
                if marker.exists():
                    records.append(json.loads(marker.read_text()))
                    continue
                env['APPWORLD_MAX_PROMPT_GROUPS'] = str(budget//16)
                complete = False
                if (run/'state/_actor').exists():
                    _, state = load_latest(run)
                    if type(state.get('rollout_generator_state_dict', {}).get('round_exhausted')) is not bool:
                        raise RuntimeError('Checkpoint lacks a verified data cursor')
                    # Pending counters establish drainage; round_exhausted encodes the DATA cursor.
                    check = {**state, 'rollout_generator_state_dict':
                             {**state['rollout_generator_state_dict'], 'round_exhausted': True}}
                    complete = validate_state(check, dataset_groups=90, target_groups=budget//16,
                                              curriculum=args.method in CURRICULUM_METHODS) == 'complete'
                if not complete:
                    cmd = command(setup, run, args.model, args.method, args.seed, budget,
                                  eval_batch=min(_split_size(setup, 'dev'), args.pool_size // 3),
                                  init_weights=args.init_weights)
                    (run/'logs').mkdir(exist_ok=True)
                    recycle_pool()  # fresh env servers for each segment (host-memory leak, see environment_pool)
                    train(cmd, setup, run, {**env, **training_only}, run/'logs/train.log')
                _, state = load_latest(run)
                assert state['total_consumed_prompts'] * 16 == budget
                generator = state['rollout_generator_state_dict']
                assert not generator['pending_rollouts'] and not generator['pending_finished_samples']
                weights = run/'weights'/f'r{budget}'
                if not weights.exists():
                    shutil.copytree(run/'state/_hf'/f"global_step{state['global_step']}", weights,
                                    copy_function=copy_checkpoint_file)
                final = budget == config.get('total_generated_rollouts', 80000)
                # Intermediate milestones get no separate dev eval (user, 2026-10-06): the in-training dev
                # eval every 10 steps already tracks progress, milestone rows were never selection
                # candidates, and each one booted its own eval process (8B r14400: step 23 at 20:50:11, eval
                # engines up 20:54:34, GPUs idle meanwhile). The weights stay in run/weights for a later
                # evaluation; APPWORLD_MILESTONE_DEV=1 restores the eval.
                if final or os.environ.get('APPWORLD_MILESTONE_DEV') == '1':
                    dev = evaluate(setup, run, args, weights, 'dev', env, f'r{budget}')
                else:
                    dev = {'avg_at_3': None}
                row = dict(environment_revision=config["environment_revision"], generated_rollouts=budget, optimizer_step=state['global_step'],
                           val_avg_at_3=dev['avg_at_3'], checkpoint=str(weights),
                           source='final_eval_only' if final else 'milestone_eval_only')
                # Test only the artifacts that are reported: the final one here and the
                # dev-selected one below. Intermediate milestones are never selection
                # candidates; their weights stay in run/weights for later evaluation.
                if budget == config.get('total_generated_rollouts', 80000):
                    row.update(evaluate_tests(setup, run, args, weights, env, f'r{budget}'))
                marker.write_text(json.dumps(row, indent=2))
                records.append(row)
                curve = read_curve(run) + records
                (run/'learning_curve.json').write_text(json.dumps(curve, indent=2))
            if args.budget == config.get('total_generated_rollouts', 80000):
                candidates = [r for r in read_curve(run)+records if r['source'] in ('in_training', 'final_eval_only')]
                best = min(candidates, key=lambda r: (-r['val_avg_at_3'], r['optimizer_step']))
                reused = reuse_final_evaluation(best, records[-1])
                if reused is not None:
                    (run/'comparison_result.json').write_text(json.dumps(reused, indent=2))
                    return
                weights = run/'weights'/'val_best'
                if not weights.exists():
                    best_source = next((r['checkpoint'] for r in records if r['optimizer_step'] == best['optimizer_step']), best['checkpoint'])
                    shutil.copytree(best_source, weights, copy_function=copy_checkpoint_file)
                result = {'best': best, 'final': records[-1]}
                result.update(evaluate_tests(setup, run, args, weights, env, 'val_best'))
                (run/'comparison_result.json').write_text(json.dumps(result, indent=2))



def config_server_entrypoint(setup):
    return json.loads((setup/'setup.json').read_text())['server_entrypoint']

if __name__ == '__main__':
    main()
