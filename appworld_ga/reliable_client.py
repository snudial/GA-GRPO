"""Recover lost responses without regenerating model actions or rerunning code."""
import asyncio
import hashlib
import json as jsonlib
import logging
import os

import httpx


class CellHang(RuntimeError):
    """An execute outlived AppWorld's own 100 s cell deadline and the server stopped
    answering even a trivial request, so the model's code holds its event loop. (A plain
    loop, with or without a bare ``except:``, is stopped by the deadline; the 2026-10-05
    hang was not, and its code was lost with the server.) The episode is scored as a
    failure instead of aborting the run."""


class ReliableClient:
    def __init__(self, http, attempts=4, backoff=0.25, hang_seconds=None, probe_seconds=10.0):
        self.http = http
        self.attempts = attempts
        self.backoff = backoff
        # Cells end within AppWorld's 100 s deadline; an execute far past it with the
        # whole server unresponsive is a hang (2026-10-05: one cell held its server for
        # 25 min while every other episode of the batch waited on it).
        self.hang_seconds = (float(os.environ.get('APPWORLD_CELL_HANG_SECONDS', '240'))
                             if hang_seconds is None else hang_seconds)
        self.probe_seconds = probe_seconds
        self.server_boot = None
        self.generation = None
        self.seq = 0
        self.recoveries = 0

    async def _info(self):
        for attempt in range(self.attempts):
            try:
                response = await self.http.get('/reliable/info')
                response.raise_for_status()
                info = response.json()
                if info['protocol'] != 1:
                    raise RuntimeError('Unsupported AppWorld recovery protocol')
                self.server_boot = info['server_boot']
                self.generation = info['next_generation']
                return
            except httpx.TransportError:
                if attempt + 1 == self.attempts:
                    raise
                await asyncio.sleep(self.backoff * (2 ** attempt))

    async def _blocked(self):
        """True when the server cannot answer a trivial request: its event loop is held."""
        try:
            (await self.http.get('/reliable/info', timeout=self.probe_seconds)).raise_for_status()
            return False
        except httpx.TimeoutException:
            return True
        except httpx.HTTPError:
            return False  # refused or reset: a dead server, left to the transport recovery

    async def post(self, path, *, json):
        operation = path.lstrip('/')
        if self.server_boot is None:
            if operation != 'initialize':
                raise RuntimeError('Initialize the reliable AppWorld session first')
            await self._info()
        envelope = dict(server_boot=self.server_boot, world_generation=self.generation,
                        request_seq=self.seq, operation=operation, payload=json)
        digest = hashlib.sha256(jsonlib.dumps([operation, json], sort_keys=True,
                                         ensure_ascii=True).encode()).hexdigest()
        response = None
        timeout = self.hang_seconds if operation == 'execute' else httpx.USE_CLIENT_DEFAULT
        for attempt in range(self.attempts):
            try:
                if attempt:
                    # Query before resubmitting. If the original packet was never
                    # delivered, the identical request ID can safely be submitted.
                    status = await self.http.get('/reliable/result', params=dict(
                        server_boot=self.server_boot, world_generation=self.generation,
                        request_seq=self.seq, digest=digest), timeout=timeout)
                    # Initialize may have failed before reaching the server; the
                    # info endpoint identifies that exact, still-unused generation.
                    if status.status_code == 409 and operation == 'initialize':
                        info = await self.http.get('/reliable/info')
                        info.raise_for_status()
                        info = info.json()
                        if (info['server_boot'], info['next_generation']) != (self.server_boot, self.generation):
                            status.raise_for_status()
                    else:
                        status.raise_for_status()
                        state = status.json()
                        if state['state'] == 'done':
                            response = httpx.Response(200, content=jsonlib.dumps(
                                state['response'], ensure_ascii=True).encode(), request=status.request)
                            break
                        if state['state'] != 'not_seen':
                            raise RuntimeError('Unknown recovery state')
                response = await self.http.post('/reliable/request', json=envelope, timeout=timeout)
                response.raise_for_status()
                break
            except httpx.TransportError as exc:
                if operation == 'execute' and isinstance(exc, httpx.ReadTimeout) and await self._blocked():
                    raise CellHang(f'AppWorld execute seq={self.seq} still running after '
                                   f'{self.hang_seconds:.0f}s; server unresponsive') from exc
                logging.getLogger('appworld.reliable').warning(
                    'Transport interrupted op=%s seq=%s attempt=%s type=%s; recover same request',
                    operation, self.seq, attempt + 1, type(exc).__name__)
                if attempt + 1 == self.attempts:
                    raise RuntimeError(f'AppWorld transport recovery exhausted: op={operation} seq={self.seq}') from exc
                await asyncio.sleep(self.backoff * (2 ** attempt))
        if attempt:
            self.recoveries += 1
            logging.getLogger('appworld.reliable').warning(
                'Recovered AppWorld response op=%s seq=%s without repeating execution', operation, self.seq)
        self.seq += 1
        return response
