"""Journal environment operations without inspecting live native thread stacks.

The former faulthandler timer segfaulted the AppWorld Python 3.12 server during
long cells. Keep the flushed before/after records, without that crash source.
Cell deadlines remain the responsibility of execution_compat, unchanged here.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time


class OperationDiagnostics:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.logger = logging.getLogger('appworld.diagnostics')

    def _record(self, record):
        self.directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record, ensure_ascii=True, allow_nan=False) + '\n'
        # Close/flush the record BEFORE invoking model-generated code.
        with (self.directory/'operations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(text)
        current = self.directory/'current.json'
        temporary = self.directory/'current.json.tmp'
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(current)

    @contextmanager
    def operation(self, *, server_boot, world_generation, request_seq, operation, payload):
        started = time.monotonic()
        record = dict(pid=os.getpid(), server_boot=server_boot,
                      world_generation=world_generation, request_seq=request_seq,
                      operation=operation, task_id=payload.get('task_id'),
                      started_at=datetime.now(timezone.utc).isoformat(), state='started')
        if operation == 'execute':
            record['code'] = payload.get('code')
        error_type = None
        try:
            self._record(record)
        except Exception:
            self.logger.exception('Could not start operation diagnostics')
        try:
            yield
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            record.update(state='failed' if error_type else 'done', error_type=error_type,
                          elapsed_seconds=time.monotonic()-started,
                          finished_at=datetime.now(timezone.utc).isoformat())
            try:
                self._record(record)
            except Exception:
                self.logger.exception('Could not finish operation diagnostics')
