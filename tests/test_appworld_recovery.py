import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
import pytest

from appworld_ga.reliable_client import ReliableClient
from appworld_ga.reliable_server import install_reliable_routes


def test_unknown_execution_is_not_retried_or_followed_by_another_action():
    async def run():
        counter = []
        async def initialize(args):
            return {'output': 'initialized'}
        async def execute(**kwargs):
            counter.append(1)  # mutation happened before an infrastructure exception
            raise RuntimeError('injected failure after mutation')
        official = SimpleNamespace(initialize=initialize, execute=execute,
                                   AppWorldArgs=lambda **kw:kw)
        app = FastAPI()
        install_reliable_routes(app, official, JSONResponse)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app, raise_app_exceptions=False),
                                    base_url='http://test') as client:
            info = (await client.get('/reliable/info')).json()
            envelope = dict(server_boot=info['server_boot'], world_generation=info['next_generation'],
                            request_seq=0, operation='initialize', payload={})
            assert (await client.post('/reliable/request', json=envelope)).status_code == 200
            envelope.update(request_seq=1, operation='execute', payload=dict(code='increment'))
            assert (await client.post('/reliable/request', json=envelope)).status_code == 500
            assert (await client.post('/reliable/request', json=envelope)).status_code == 409
            envelope['request_seq'] = 2
            assert (await client.post('/reliable/request', json=envelope)).status_code == 409
            assert len(counter) == 1
    asyncio.run(run())


def test_transport_retries_are_bounded_and_keep_identical_request_id():
    async def run():
        posts = []
        async def transport(request):
            if request.url.path == '/reliable/info':
                return httpx.Response(200, json=dict(protocol=1, server_boot='boot', next_generation=1))
            if request.url.path == '/reliable/result':
                return httpx.Response(200, json=dict(state='not_seen'))
            posts.append(request.content)
            raise httpx.ReadError('injected connection reset', request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport),base_url='http://test') as http:
            client = ReliableClient(http, attempts=4, backoff=0)
            with pytest.raises(RuntimeError, match='recovery exhausted'):
                await client.post('/initialize', json={'task_id':'test'})
            assert len(posts) == 4 and len(set(posts)) == 1 and client.seq == 0
    asyncio.run(run())
