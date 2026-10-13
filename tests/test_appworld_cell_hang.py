"""A cell that outlives AppWorld's deadline and holds its server is a failed episode
(CellHang), not an infrastructure fault; a slow but answering server keeps the
ordinary response recovery."""
import asyncio
import subprocess
import sys
import time

import httpx
import pytest

from appworld_ga.environment import kill_server
from appworld_ga.reliable_client import CellHang, ReliableClient


def _client(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url='http://test')
    client = ReliableClient(http, attempts=4, backoff=0, hang_seconds=1, probe_seconds=1)
    client.server_boot, client.generation = 'boot', 0
    return http, client


def test_execute_with_unresponsive_server_is_a_cell_hang():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        raise httpx.ReadTimeout('held', request=request)

    async def run():
        http, client = _client(handler)
        async with http:
            with pytest.raises(CellHang):
                await client.post('/execute', json={'task_id': 't', 'code': 'while True: pass'})
    asyncio.run(run())
    assert calls == ['/reliable/request', '/reliable/info']  # one probe, no resubmission


def test_lost_execute_response_with_answering_server_is_recovered():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == '/reliable/request':
            raise httpx.ReadTimeout('lost', request=request)
        if request.url.path == '/reliable/info':
            return httpx.Response(200, json={'protocol': 1, 'server_boot': 'boot', 'next_generation': 1})
        return httpx.Response(200, json={'state': 'done', 'response': {'output': 'ok'}})

    async def run():
        http, client = _client(handler)
        async with http:
            response = await client.post('/execute', json={'task_id': 't', 'code': 'print(1)'})
            assert response.json() == {'output': 'ok'}
    asyncio.run(run())
    assert calls == ['/reliable/request', '/reliable/info', '/reliable/result']


def test_only_execute_is_classified_as_a_hang():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        raise httpx.ReadTimeout('held', request=request)

    async def run():
        http, client = _client(handler)
        async with http:
            with pytest.raises(RuntimeError, match='recovery exhausted') as raised:
                await client.post('/evaluate', json={'task_id': 't'})
            assert not isinstance(raised.value, CellHang)
    asyncio.run(run())
    assert '/reliable/info' not in calls


def test_kill_server_matches_entrypoint_and_exact_port_only():
    sleeper = 'import time; time.sleep(60)'
    target = subprocess.Popen([sys.executable, '-c', sleeper, 'appworld_ga.server:app', '--port', '54321'])
    other = subprocess.Popen([sys.executable, '-c', sleeper, 'appworld_ga.server:app', '--port', '543210'])
    try:
        time.sleep(.5)
        assert kill_server(54321) == target.pid
        assert target.wait(timeout=5) == -9
        assert other.poll() is None
        with pytest.raises(RuntimeError, match='expected one AppWorld server'):
            kill_server(54322)
    finally:
        for process in (target, other):
            if process.poll() is None:
                process.kill()
