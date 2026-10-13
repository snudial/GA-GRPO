"""Deduplicate HTTP requests within a live, exclusively leased AppWorld.

The official environment executes each operation unchanged. Cache identity is
(server boot, world generation, request sequence), not task_id (tasks repeat).
Unknown outcomes and server restarts fail closed; this is not crash recovery.
"""
import asyncio
from contextlib import nullcontext
import hashlib
import json
import logging
import uuid

from fastapi import Body, HTTPException
from fastapi.encoders import jsonable_encoder


def install_reliable_routes(app, official, response_class, diagnostics=None):
    boot = uuid.uuid4().hex
    generation = 0
    entries = {}
    closed = False
    lock = asyncio.Lock()
    logger = logging.getLogger("appworld.reliable")

    def validate_epoch(server_boot, world_generation):
        if server_boot != boot:
            raise HTTPException(409, "Server restarted; world state is unknown. Do not replay.")
        if world_generation != generation:
            raise HTTPException(409, "Stale world generation; refusing to touch another episode.")

    @app.get('/reliable/info')
    async def info():
        return dict(server_boot=boot, next_generation=generation + 1, protocol=1)

    @app.get('/reliable/result', response_class=response_class)
    async def result(server_boot: str, world_generation: int, request_seq: int, digest: str):
        async with lock:
            validate_epoch(server_boot, world_generation)
            entry = entries.get(request_seq)
            if entry is None:
                return dict(state='not_seen')
            if entry['digest'] != digest:
                raise HTTPException(409, "Request ID reused with different content.")
            if entry['state'] != 'done':
                raise HTTPException(409, "Execution outcome unknown; refusing to repeat operation.")
            logger.info("Recovered cached response generation=%s seq=%s", generation, request_seq)
            return dict(state='done', response=entry['response'])

    @app.post('/reliable/request', response_class=response_class)
    async def request(envelope: dict = Body(...)):
        nonlocal generation, entries, closed
        server_boot = envelope['server_boot']
        requested_generation = envelope['world_generation']
        seq = envelope['request_seq']
        operation = envelope['operation']
        payload = envelope['payload']
        digest = hashlib.sha256(json.dumps([operation, payload], sort_keys=True,
                                          ensure_ascii=True).encode()).hexdigest()
        if operation not in ('initialize', 'execute', 'public_task_metadata',
                             'task_completed', 'evaluate', 'close'):
            raise HTTPException(400, 'Unsupported operation')
        if type(seq) is not int or seq < 0 or type(requested_generation) is not int:
            raise HTTPException(400, 'Invalid request identity')
        async with lock:
            if server_boot != boot:
                raise HTTPException(409, "Server restarted; refusing to replay.")
            if operation == 'initialize' and requested_generation == generation + 1 and seq == 0:
                # A new exclusive lease may reset a failed previous episode, as
                # the original initialize endpoint does. Older packets are fenced.
                generation = requested_generation
                entries = {}
                closed = False
            validate_epoch(server_boot, requested_generation)
            if seq in entries:
                entry = entries[seq]
                if entry['digest'] != digest:
                    raise HTTPException(409, 'Request ID reused with different content')
                if entry['state'] != 'done':
                    raise HTTPException(409, 'Execution outcome unknown; refusing to repeat')
                return entry['response']
            if closed or seq != len(entries) or (operation == 'initialize' and seq != 0):
                raise HTTPException(409, 'Closed episode or out-of-order request')
            if any(entry['state'] != 'done' for entry in entries.values()):
                raise HTTPException(409, 'Previous operation has unknown outcome; episode is fenced')
            entries[seq] = dict(digest=digest, state='started')
            try:
                diagnostic = diagnostics.operation(server_boot=boot, world_generation=generation,
                    request_seq=seq, operation=operation, payload=payload) if diagnostics else nullcontext()
                with diagnostic:
                    if operation == 'initialize':
                        response = await official.initialize(official.AppWorldArgs(**payload))
                    elif operation == 'execute':
                        response = await official.execute(**payload)
                    elif operation == 'public_task_metadata':
                        official.maybe_raise_exception(payload['task_id'])
                        response = {'output': {'app_descriptions': [
                            {'name': k, 'description': v}
                            for k, v in official.world.task.app_descriptions.items()]}}
                    elif operation == 'evaluate':
                        response = await official.evaluate(report=False, **payload)
                    elif operation == 'task_completed':
                        response = await official.task_completed(**payload)
                    else:
                        response = await official.close(**payload)
                        closed = True
                    # Validate serialization before declaring a recoverable result.
                    response = json.loads(json.dumps(jsonable_encoder(response), ensure_ascii=True, allow_nan=False))
                    entries[seq].update(state='done', response=response)
                    return response
            except BaseException:
                entries[seq]['state'] = 'unknown'
                logger.exception('Unrecoverable operation generation=%s seq=%s op=%s',
                                 generation, seq, operation)
                raise
