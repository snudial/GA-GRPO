import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
import pytest

from appworld_ga.operation_diagnostics import OperationDiagnostics
from appworld_ga.reliable_server import install_reliable_routes


def test_records_before_execution_preserves_output_and_deduplication(tmp_path):
    async def run():
        calls = []
        code = "print('value') # \ud800"
        async def initialize(args):
            return {'output': 'ready'}
        async def execute(**payload):
            current = json.loads((tmp_path/'current.json').read_text())
            assert current['state'] == 'started'
            assert current['code'] == code and current['request_seq'] == 1
            calls.append(payload)
            return {'output': 'value\n'}
        official = SimpleNamespace(initialize=initialize, execute=execute, AppWorldArgs=lambda **kw: kw)
        app = FastAPI()
        install_reliable_routes(app, official, JSONResponse, OperationDiagnostics(tmp_path))
        handler = signal.getsignal(signal.SIGALRM)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            info = (await client.get('/reliable/info')).json()
            envelope = dict(server_boot=info['server_boot'], world_generation=info['next_generation'],
                            request_seq=0, operation='initialize', payload={'task_id': 'task'})
            assert (await client.post('/reliable/request', json=envelope)).status_code == 200
            envelope.update(request_seq=1, operation='execute', payload={'task_id': 'task', 'code': code})
            # ASCII JSON also tests preservation of unpaired surrogates.
            for _ in range(2):
                response = await client.post('/reliable/request', content=json.dumps(envelope),
                                             headers={'content-type': 'application/json'})
                assert response.json() == {'output': 'value\n'}
        assert len(calls) == 1
        assert signal.getsignal(signal.SIGALRM) == handler
        records = [json.loads(line) for line in (tmp_path/'operations.jsonl').read_text().splitlines()]
        assert [r['state'] for r in records] == ['started', 'done', 'started', 'done']
    asyncio.run(run())


def test_original_error_is_preserved_even_when_diagnostics_cannot_write(tmp_path):
    bad_directory = tmp_path/'file'
    bad_directory.write_text('not a directory')
    diagnostic = OperationDiagnostics(bad_directory)
    with pytest.raises(ValueError, match='original failure'):
        with diagnostic.operation(server_boot='b', world_generation=1, request_seq=2,
                                  operation='execute', payload={'code': 'raise ValueError()'}):
            raise ValueError('original failure')


def test_busy_cell_journaling_never_arms_native_stack_dump(tmp_path):
    source = '''
import asyncio, sys, time, faulthandler
from appworld_ga.operation_diagnostics import OperationDiagnostics
def unsafe_timer(*args, **kwargs):
    raise AssertionError('Native stack inspection must not run inside AppWorld')
faulthandler.dump_traceback_later = unsafe_timer
faulthandler.cancel_dump_traceback_later = unsafe_timer
async def blocked():
    with OperationDiagnostics(sys.argv[1]).operation(server_boot='boot', world_generation=1,
            request_seq=7, operation='execute', payload={'code':'busy loop'}):
        deadline = time.monotonic() + .35
        while time.monotonic() < deadline:
            pass
asyncio.run(blocked())
'''
    child = subprocess.run([sys.executable, '-c', source, str(tmp_path)],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr
    assert not (tmp_path/'stacks.log').exists()
    records = [json.loads(line) for line in (tmp_path/'operations.jsonl').read_text().splitlines()]
    assert [r['state'] for r in records] == ['started', 'done']
    assert json.loads((tmp_path/'current.json').read_text())['state'] == 'done'
