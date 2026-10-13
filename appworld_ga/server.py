"""Isolated AppWorld server entrypoint; no modifications to site-packages.

Fix UTF-8 serialization of unpaired surrogates in logs and HTTP transport.
Execution, observations, world state and scoring are not rewritten or retried.
"""
import json
import os
from pathlib import Path

from fastapi import Body
from fastapi.responses import JSONResponse
from appworld_ga.reliable_server import install_reliable_routes
from appworld_ga.log_serialization import save_request_log
from appworld_ga.operation_diagnostics import OperationDiagnostics
from appworld.requester import RequestTracker
import appworld.environment as environment
import appworld.serve.environment as official


def save_environment_io_log(self):
    self._maybe_raise_remote_environment_error('_save_environment_io_log')
    path = Path(self.output_logs_directory)/'environment_io.md'
    with path.open('w', encoding='utf-8', errors='backslashreplace') as stream:
        for index, entry in enumerate(self.environment_io):
            stream.write('\n'.join([
                f'\n### Environment Interaction {index + 1}\n{environment.SINGLE_HORIZONTAL_RULE}',
                f"```python\n{entry['input']}\n```\n",
                f"```\n{entry['output']}\n```\n\n",
            ]))


class EscapedJSONResponse(JSONResponse):
    def render(self, content):
        return json.dumps(content, ensure_ascii=True, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')


environment.AppWorld._save_environment_io_log = save_environment_io_log
RequestTracker.save = save_request_log
from appworld_ga.execution_compat import install as install_execution_compat
install_execution_compat()

app = official.app
diagnostics = (OperationDiagnostics(Path(os.environ['APPWORLD_ROOT'])/'diagnostics'/str(os.getpid()))
               if os.environ.get('APPWORLD_ROOT') else None)
install_reliable_routes(app, official, EscapedJSONResponse, diagnostics=diagnostics)


def _resident_gib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1048576
    return 0.0


_official_close = official.close


async def close(task_id):
    """The official close, plus this process's resident memory so the client (appworld_ga.environment
    World.close) can retire a server that has grown too large between episodes: the official serve app
    keeps memory an episode used (2026-10-11: 0.8-11 GiB per server after hours of training)."""
    out = await _official_close(task_id=task_id)
    out = dict(out) if isinstance(out, dict) else {'output': out}
    out['server_rss_gib'] = round(_resident_gib(), 3)
    return out


official.close = close  # the reliable route calls official.close at request time
for route in app.routes:
    if getattr(route, 'path', None) == '/execute':
        # Rebuilding the route handler is needed: FastAPI captures response_class
        # at route construction, so assigning that attribute alone is insufficient.
        route.response_class = EscapedJSONResponse
        from starlette.routing import request_response
        route.app = request_response(route.get_route_handler())


@app.post('/public_task_metadata')
async def public_task_metadata(task_id: str = Body(..., embed=True)):
    official.maybe_raise_exception(task_id)
    return {'output': {'app_descriptions': [
        {'name': name, 'description': description}
        for name, description in official.world.task.app_descriptions.items()]}}
