"""Isolated, repeatable AppWorld evaluation using the sealed agent and Molt serializer.

No Ray router or live training engine is used. Each GPU UUID owns a stable set of
tasks and executes one episode at a time. Repeats restart both model and world
processes; odd repeats reverse task order to detect state leaking across episodes.
"""
import argparse
import asyncio
import contextlib
from copy import deepcopy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ga_grpo.config import path
OWNED = path('RUN_ROOT')
PROTOCOL = 'appworld_repro_eval_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def request_seed(split, task_id, decode_seed, turn):
    # Exclude checkpoint, method, process ID, repeat and GPU: paired arms share randomness.
    return int(digest([PROTOCOL, split, task_id, decode_seed, turn])[:16], 16) % (2**31)


def lane_for(split, task_id, gpu_count):
    return int(digest([PROTOCOL, split, task_id])[:16], 16) % gpu_count


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare_records(reference, other):
    def indexed(rows):
        result = {}
        for row in rows:
            key = (row['split'], row['task_id'], row['decode_seed'])
            if key in result:
                raise ValueError(f'Duplicate evaluation identity: {key}')
            result[key] = row
        return result
    a, b = indexed(reference), indexed(other)
    if a.keys() != b.keys():
        raise ValueError('Incomplete or different evaluation task sets')
    return [dict(split=k[0], task_id=k[1], decode_seed=k[2],
                 fields=[f for f in a[k].keys() | b[k].keys() if a[k].get(f) != b[k].get(f)])
            for k in sorted(a) if a[k] != b[k]]


@contextlib.contextmanager
def world_server(plan, lane):
    port = plan['first_port'] + lane
    root = Path(plan['output']) / 'servers' / str(port)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'data').exists():
        (root / 'data').symlink_to(plan['config']['data'], target_is_directory=True)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'APPWORLD_ROOT': str(root),
           'PYTHONPATH': str(Path(plan['setup']) / 'source')}
    with (root / 'server.log').open('a') as log:
        process = subprocess.Popen([str(path('APPWORLD_VENV')/'bin/python'),
            '-m', 'uvicorn', plan['config']['server_entrypoint'], '--host', '127.0.0.1',
            '--port', str(port), '--timeout-keep-alive', '30'], cwd=root, env=env,
            stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('AppWorld server exited; inspect server.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('AppWorld server startup timeout')
                    time.sleep(.2)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def worker(plan, lane, repeat):
    uuid = plan['gpu_uuids'][lane]
    port = plan['first_port'] + lane
    out = Path(plan['output']) / f'repeat{repeat}' / f'lane{lane}'
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, VLLM_BATCH_INVARIANT='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        TOKENIZERS_PARALLELISM='false', APPWORLD_POOL=f'{port}-{port}',
        APPWORLD_SEED=str(plan['environment_seed']), APPWORLD_RELIABLE_RPC='1',
        APPWORLD_PROMPT_PROFILE=plan['config']['prompt_profile'],
        APPWORLD_OBSERVATION_MAX_CHARS=str(plan['config']['observation_max_chars']),
        APPWORLD_MAX_TURNS=str(plan['config']['max_turns']), APPWORLD_CURRICULUM='0',
        APPWORLD_TRAJECTORY_DIR=str(out / 'trajectories'))
    os.environ.pop('APPWORLD_EVAL_RECORDS', None)
    sys.path[:0] = [str(Path(plan['setup']) / 'source'), str(Path(plan['setup']) / 'runtime')]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from appworld_ga.agent import AgentRunner

    tokenizer = AutoTokenizer.from_pretrained(plan['checkpoint'], local_files_only=True)
    llm = LLM(model=plan['checkpoint'], dtype='bfloat16', tensor_parallel_size=1,
        seed=0, max_model_len=plan['config']['context_max_length'],
        gpu_memory_utilization=.5, max_num_seqs=1, max_num_batched_tokens=10000,
        enable_prefix_caching=False, enforce_eager=True, generation_config='auto')
    runner = AgentRunner()
    tasks = [t for t in plan['tasks'] if t['lane'] == lane]
    if repeat % 2:
        tasks.reverse()

    async def evaluate():
        for item in tasks:
            for decode_seed in plan['decode_seeds']:
                trace = []

                async def generate(prompt_token_ids, sampling_params, **kwargs):
                    if kwargs.get('multi_modal_data'):
                        raise ValueError('This evaluator supports text-only AppWorld')
                    sp = deepcopy(sampling_params)
                    sp.seed = request_seed(item['split'], item['task_id'], decode_seed, len(trace))
                    sp.__dict__.pop('sampling_type', None)
                    output = llm.generate([{'prompt_token_ids': list(prompt_token_ids)}], sp,
                                          use_tqdm=False)[0]
                    generated = output.outputs[0]
                    trace.append(dict(turn=len(trace), seed=sp.seed,
                        prompt_tokens=list(prompt_token_ids), output_tokens=list(generated.token_ids),
                        finish_reason=generated.finish_reason, temperature=sp.temperature))
                    return output, 0

                label = {**item['label'], 'eval': True, 'h': 0}
                trajectories = await runner.execute(prompt=item['input'], label=label,
                    sampling_params=SamplingParams(temperature=plan['temperature'], top_p=1.,
                        top_k=-1, min_tokens=1, max_tokens=4096, logprobs=1,
                        skip_special_tokens=False),
                    max_length=plan['config']['context_max_length'], hf_tokenizer=tokenizer,
                    llm_engine=SimpleNamespace(generate=generate))
                if not trajectories:
                    raise RuntimeError('No trajectory returned; refusing a fabricated failure score')
                info = dict(trajectories[-1].extra_logs)
                full = json.loads(Path(info.pop('trajectory_path')).read_text())
                reward = trajectories[-1].reward
                if reward not in (0., 1.) or info['handoff_h'] != 0:
                    raise ValueError('Invalid autonomous terminal reward')
                identity = dict(split=item['split'], task_id=item['task_id'], decode_seed=decode_seed)
                name = f"{item['split']}_{item['task_id']}_{decode_seed}"
                (out / f'{name}.trace.json').write_text(json.dumps(trace))
                row = dict(**identity, gpu_uuid=uuid, environment_seed=plan['environment_seed'],
                    success=reward, info=info, token_trace_sha256=digest(trace),
                    messages_sha256=digest(full['messages']))
                with (out / 'records.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
    try:
        with world_server(plan, lane):
            asyncio.run(evaluate())
    finally:
        # EngineCore is a child process; interpreter exit otherwise waits for it
        # while the request callback still retains the LLM through its closure.
        llm.llm_engine.engine_core.shutdown(timeout=15)


def execute(plan):
    out = Path(plan['output'])
    lock_path = OWNED / 'operations/comparison_queue_watchdog.lock'
    with lock_path.open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader'], text=True).strip()
                if not busy:
                    break
                fcntl.flock(lock, fcntl.LOCK_UN)
            except BlockingIOError:
                pass
            if not plan['wait_for_idle']:
                raise RuntimeError('GPUs or campaign lock busy; use --wait-for-idle')
            (out / 'status.json').write_text(json.dumps({'status': 'waiting_for_idle', 'time': time.time()}))
            time.sleep(30)
        for package, version in plan['packages'].items():
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f'Package changed since planning: {package}')
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                              '--format=csv,noheader'], text=True)
        if inventory != plan['gpu_inventory']:
            raise RuntimeError('GPU UUID, order, model or driver changed since planning')
        appworld_version = subprocess.check_output([str(path('APPWORLD_VENV')/'bin/python'),
            '-c', 'import importlib.metadata; print(importlib.metadata.version("appworld"))'], text=True).strip()
        if appworld_version != plan['appworld_version']:
            raise RuntimeError('AppWorld package changed since planning')
        for filename, expected in plan['artifact_sha256'].items():
            if file_hash(filename) != expected:
                raise RuntimeError(f'Artifact changed since planning: {filename}')
        repeats = []
        for repeat in range(plan['repeats']):
            processes, logs = [], []
            try:
                for lane in range(len(plan['gpu_uuids'])):
                    log = (out / f'repeat{repeat}_lane{lane}.log').open('x')
                    logs.append(log)
                    env = {k: v for k, v in os.environ.items()
                           if not k.startswith(('APPWORLD_', 'VLLM_', 'RAY_', 'MOLT_', 'CUDA_'))}
                    env['PYTHONHASHSEED'] = '0'
                    processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                        '--worker', str(out / 'plan.json'), '--lane', str(lane), '--repeat', str(repeat)],
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                (out / 'status.json').write_text(json.dumps(dict(status='evaluating', repeat=repeat,
                                                               pids=[p.pid for p in processes])))
                while any(p.poll() is None for p in processes):
                    if any(p.poll() not in (None, 0) for p in processes):
                        raise RuntimeError('Evaluation worker failed; inspect worker logs')
                    time.sleep(1)
                if any(p.returncode for p in processes):
                    raise RuntimeError('Evaluation worker failed')
            finally:
                for p in processes:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                for p in processes:
                    try:
                        p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
                for log in logs:
                    log.close()
            rows = [json.loads(line) for p in (out / f'repeat{repeat}').glob('lane*/records.jsonl')
                    for line in p.read_text().splitlines()]
            expected = {(t['split'], t['task_id'], seed) for t in plan['tasks'] for seed in plan['decode_seeds']}
            actual = {(r['split'], r['task_id'], r['decode_seed']) for r in rows}
            if len(rows) != len(expected) or actual != expected:
                raise RuntimeError('Missing/duplicate evaluation records; refusing partial score')
            repeats.append(rows)
        comparisons = [dict(repeat=i, differences=compare_records(repeats[0], rows))
                       for i, rows in enumerate(repeats[1:], 1)]
        passed = all(not c['differences'] for c in comparisons)
        report = dict(status='passed' if passed else 'diverged', protocol=PROTOCOL,
            comparisons=comparisons, repetitions=len(repeats), episodes_per_repeat=len(repeats[0]),
            scores=[sum(r['success'] for r in rows) / len(rows) for rows in repeats],
            scope='Exact token/observation/outcome match on these tasks and this pinned hardware/software only')
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        (out / 'status.json').write_text(json.dumps(report, indent=2))
        if not passed:
            raise RuntimeError('Repeated evaluation diverged; inspect report and per-turn traces')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--lane', type=int)
    p.add_argument('--repeat', type=int)
    p.add_argument('--setup', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--splits', nargs='+', choices=['dev', 'test_normal', 'test_challenge'], default=['dev'])
    p.add_argument('--tasks', nargs='+', help='Explicit pilot task IDs; omit for entire splits')
    p.add_argument('--gpus', nargs='+', help='Ordered GPU UUIDs; defaults to all visible physical GPUs')
    p.add_argument('--seeds', nargs='+', type=int, default=[20260827, 20260828, 20260829])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--temperature', type=float, choices=[0., .01], default=.01)
    p.add_argument('--environment-seed', type=int, default=42)
    p.add_argument('--first-port', type=int, default=19620)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--wait-for-idle', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(json.loads(a.worker.read_text()), a.lane, a.repeat)
        return
    if not all([a.setup, a.checkpoint, a.output]) or a.repeats < 2:
        p.error('--setup, --checkpoint, --output and at least two repeats are required')
    setup, checkpoint, out = a.setup.resolve(), a.checkpoint.resolve(), a.output.resolve()
    if not out.is_relative_to(OWNED.resolve()):
        p.error('Output must be within the owned experiment root')
    available = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
    uuids = a.gpus or available
    if not uuids or len(set(uuids)) != len(uuids) or not set(uuids) <= set(available):
        p.error('GPU list must contain unique available physical GPU UUIDs')
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.splits)) != len(a.splits):
        p.error('Seeds and splits must be unique')
    config = json.loads((setup / 'setup.json').read_text())
    tasks = []
    for split in a.splits:
        for line in (setup / 'datasets' / f'{split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            label = json.loads(row['label']) if isinstance(row['label'], str) else row['label']
            if a.tasks and label['task_id'] not in a.tasks:
                continue
            tasks.append(dict(split=split, task_id=label['task_id'], input=row['input'], label=label,
                              lane=lane_for(split, label['task_id'], len(uuids))))
    if not tasks or (a.tasks and {t['task_id'] for t in tasks} != set(a.tasks)):
        p.error('Missing requested pilot tasks')
    artifacts = [Path(__file__).resolve(), setup / 'setup.json']
    artifacts += [setup / 'datasets' / f'{s}.jsonl' for s in a.splits]
    artifacts += [p for root in [setup / 'source', setup / 'runtime'] for p in root.rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.txt', '.jinja', '.json', '.yaml', '.md')]
    artifacts += [p for p in checkpoint.iterdir() if p.is_file() and p.suffix in ('.safetensors', '.json', '.jinja')]
    if not any(p.suffix == '.safetensors' for p in artifacts):
        p.error('Checkpoint contains no model weights')
    plan = dict(protocol=PROTOCOL, setup=str(setup), checkpoint=str(checkpoint), output=str(out),
        config=config, tasks=tasks, gpu_uuids=uuids, decode_seeds=a.seeds, repeats=a.repeats,
        temperature=a.temperature, environment_seed=a.environment_seed, first_port=a.first_port,
        wait_for_idle=a.wait_for_idle, execute_requested=a.execute,
        packages={n: importlib.metadata.version(n) for n in ['vllm', 'torch', 'transformers']},
        appworld_version=subprocess.check_output([str(path('APPWORLD_VENV')/'bin/python'),
            '-c', 'import importlib.metadata; print(importlib.metadata.version("appworld"))'], text=True).strip(),
        gpu_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                               '--format=csv,noheader'], text=True),
        artifact_sha256={str(p): file_hash(p) for p in sorted(set(artifacts))})
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    print(json.dumps(dict(output=str(out), tasks=len(tasks), repeats=a.repeats,
                         episodes=len(tasks)*len(a.seeds)*a.repeats, gpu_uuids=uuids)), flush=True)
    if a.execute:
        try:
            execute(plan)
        except BaseException as exc:
            (out / 'error.json').write_text(json.dumps(dict(error=repr(exc), time=time.time())))
            raise


if __name__ == '__main__':
    main()
