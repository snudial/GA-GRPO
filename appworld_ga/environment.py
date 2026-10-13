"""HTTP client for the AppWorld environment server pool.

AppWorld pins pydantic 1.x, which cannot coexist with the Molt/vLLM stack
(pydantic 2). So AppWorld runs as a pool of `appworld serve environment`
processes and we talk to it over plain HTTP — no `appworld` import here.

Each server holds ONE global world (see appworld/serve/environment.py), so a
port is an exclusive lease for the duration of an episode. Leases are taken
with a non-blocking flock on a per-port file, which works across the many
Ray rollout processes that each import this module.
"""
import asyncio
import fcntl
import logging
import os
import re
import signal
import sys

import httpx
from appworld_ga.pool_control import allowed_ports
from appworld_ga.reliable_client import CellHang

POOL_SPEC = os.environ.get("APPWORLD_POOL", "19300-19363")
LEASE_DIR = os.environ.get(
    "APPWORLD_LEASE_DIR", "/dev/shm/jump_appworld_leases")


def pool_ports(spec=POOL_SPEC):
    lo, hi = (int(x) for x in spec.split("-"))
    return list(range(lo, hi + 1))


class Lease:
    """Exclusive hold on one environment server port."""

    def __init__(self, port, fd):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self._fd = fd

    def release(self):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def _try_lease(port):
    os.makedirs(LEASE_DIR, exist_ok=True)
    fd = os.open(os.path.join(LEASE_DIR, f"{port}.lock"),
                 os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Lease(port, fd)
    except OSError:
        os.close(fd)
        return None


async def acquire(ports=None, poll=0.5, timeout=1800):
    """Wait for a free environment server. Ports are probed in a rotated
    order so concurrent rollouts do not all collide on the low ports."""
    ports = list(ports or pool_ports())
    start = asyncio.get_event_loop().time()
    offset = os.getpid() % max(1, len(ports))
    ports = ports[offset:] + ports[:offset]
    while True:
        eligible = set(allowed_ports(sorted(ports), os.environ.get("APPWORLD_POOL_CONTROL")))
        for port in ports:
            if port not in eligible:
                continue
            lease = await asyncio.to_thread(_try_lease, port)
            if lease is not None:
                return lease
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError(
                f"no free AppWorld server in {POOL_SPEC} after {timeout}s")
        await asyncio.sleep(poll)


class World:
    """One AppWorld episode on a leased server."""

    def __init__(self, lease, task_id, timeout=300.0):
        self.lease = lease
        self.task_id = task_id
        reliable = os.environ.get('APPWORLD_RELIABLE_RPC') == '1'
        options = dict(limits=httpx.Limits(max_keepalive_connections=0), trust_env=False) if reliable else {}
        self.http = httpx.AsyncClient(base_url=lease.url, timeout=timeout, **options)
        if reliable:
            from appworld_ga.reliable_client import ReliableClient
            self.rpc = ReliableClient(self.http)
        else:
            self.rpc = self.http
        self.hung = False
        self.observation_truncations = 0
        self.observation_chars_dropped = 0

    async def initialize(self, experiment_name, max_interactions=200):
        r = await self.rpc.post("/initialize", json={
            "task_id": self.task_id, "experiment_name": experiment_name,
            "max_interactions": max_interactions,
            "random_seed": int(os.environ.get("APPWORLD_SEED", "42"))})
        r.raise_for_status()
        task = r.json()["output"]
        if os.environ.get('APPWORLD_PROMPT_PROFILE', 'legacy') != 'legacy':
            metadata = await self.rpc.post('/public_task_metadata', json={'task_id': self.task_id})
            metadata.raise_for_status()
            task['app_descriptions'] = metadata.json()['output']['app_descriptions']
        return task

    async def execute(self, code, max_chars=None):
        # 0 explicitly disables character truncation; the model context cap is
        # enforced separately by Molt. Preserve the historical default.
        if max_chars is None:
            max_chars = int(os.environ.get('APPWORLD_OBSERVATION_MAX_CHARS', '3000'))
        if max_chars < 0:
            raise ValueError('Observation character limit must be nonnegative')
        try:
            r = await self.rpc.post(
                "/execute", json={"task_id": self.task_id, "code": code})
        except CellHang:
            # The world is lost with the server; the pool supervisor
            # (run_appworld.environment_pool) restarts it on the same port.
            self.hung = True
            logging.getLogger('appworld.reliable').warning(
                'Cell hang on port %s task %s; killing the server. Code:\n%s',
                self.lease.port, self.task_id, code[:4000])
            kill_server(self.lease.port)
            raise
        r.raise_for_status()
        out = r.json()["output"]
        out = out if isinstance(out, str) else str(out)
        if max_chars and len(out) > max_chars:
            self.observation_truncations += 1
            self.observation_chars_dropped += len(out)-max_chars
            out = out[:max_chars] + f"\n...[truncated {len(out)-max_chars} chars]"
        return out

    async def completed(self):
        r = await self.rpc.post('/task_completed', json={'task_id': self.task_id})
        r.raise_for_status()
        return bool(r.json()['output'])

    async def evaluate(self, with_failures=False):
        r = await self.rpc.post('/evaluate', json={
            'task_id': self.task_id, 'suppress_errors': True})
        r.raise_for_status()
        out = r.json()['output']
        total = len(out.get('passes', [])) + len(out.get('failures', []))
        result = bool(out['success']), len(out.get('passes', [])) / max(1, total)
        # Each failure has a `requirement` sentence (teacher-safe) and a `trace`
        # holding test source and ground-truth values; callers keep them apart.
        return (*result, out.get('failures', [])) if with_failures else result

    async def close(self, restart_wait=300.0):
        try:
            if self.hung:
                # Keep the lease until the restarted server answers, so the next
                # holder never meets a booting port.
                deadline = asyncio.get_event_loop().time() + restart_wait
                while not await health(self.lease.port):
                    if asyncio.get_event_loop().time() > deadline:
                        raise RuntimeError(f'AppWorld server {self.lease.port} did not restart after a cell hang')
                    await asyncio.sleep(1.0)
            else:
                r = await self.rpc.post("/close", json={"task_id": self.task_id})
                try:
                    rss = float(r.json().get('server_rss_gib', 0.0))
                except Exception:  # noqa: BLE001
                    rss = 0.0
                if rss > float(os.environ.get('APPWORLD_SERVER_MAX_RSS_GIB', '2')):
                    # The server keeps memory from past episodes (see appworld_ga.server.close); retire
                    # it now, between episodes, and keep the lease until its replacement answers.
                    logging.getLogger('appworld.reliable').info(
                        'Retiring AppWorld server %s at %.1f GiB resident', self.lease.port, rss)
                    kill_server(self.lease.port)
                    deadline = asyncio.get_event_loop().time() + restart_wait
                    await asyncio.sleep(1.0)
                    while not await health(self.lease.port):
                        if asyncio.get_event_loop().time() > deadline:
                            raise RuntimeError(f'AppWorld server {self.lease.port} did not restart after retirement')
                        await asyncio.sleep(1.0)
        except RuntimeError:
            if self.hung or 'did not restart' in str(sys.exc_info()[1]):
                raise
        except Exception:  # noqa: BLE001
            pass  # server is re-initialized by the next lease holder anyway
        finally:
            await self.http.aclose()
            self.lease.release()


def kill_server(port):
    """SIGKILL this user's AppWorld server on ``port``: a hung cell holds its event loop,
    so no request can stop it. Only a process whose command line names the server
    entrypoint and exactly this port is touched."""
    victims = []
    for entry in os.scandir('/proc'):
        if not entry.name.isdigit():
            continue
        try:
            if os.stat(entry.path).st_uid != os.getuid():
                continue
            with open(os.path.join(entry.path, 'cmdline'), 'rb') as stream:
                argv = stream.read().split(b'\0')
        except OSError:
            continue
        if (b'appworld_ga.server:app' in argv and b'--port' in argv
                and argv[argv.index(b'--port') + 1:argv.index(b'--port') + 2] == [str(port).encode()]):
            victims.append(int(entry.name))
    if len(victims) != 1:
        raise RuntimeError(f'expected one AppWorld server on port {port}, found {victims}')
    os.kill(victims[0], signal.SIGKILL)
    return victims[0]


async def health(port, timeout=3.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"http://127.0.0.1:{port}/")
            return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------- transcripts

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Execute the first complete block, as in SAGE's ignore_multiple_calls mode.

    Do not rewrite the recorded assistant message: its generated tokens remain
    the conditioning context for subsequent actions and policy log-probabilities.
    """
    visible = THINK_RE.sub("", text or "").strip()
    blocks = CODE_RE.findall(visible)
    return blocks[0].strip() if blocks else None
