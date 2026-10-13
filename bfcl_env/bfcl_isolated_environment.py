"""A fresh BFCL simulator process per rollout while the model stays resident.

BFCL's simulator keeps task state in module globals.  Unique instance names are
not sufficient: executing one task can change the verdict of a later task in the
same interpreter.  Spawning the *environment* per session gives task isolation
without repeatedly loading the multi-gigabyte language model.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass
from typing import Any

from bfcl_env.rollout_protocol import (
    HealthInfo,
    InfrastructureError,
    SessionInfo,
    StepResult,
    StructuredAction,
)

__all__ = ["BFCLIsolatedEnvironmentBackend"]


def _child_main(connection, config: dict[str, Any]) -> None:
    """Serve exactly one session in a spawn-created interpreter."""
    from bfcl_env.runtime import BFCLRuntime
    from bfcl_env.environment_service import LocalEnvironmentBackend

    runtime = BFCLRuntime(
        category=config["category"],
        runner_mode=config["runner_mode"],
        tool_mode=config["tool_mode"],
        # Ground-truth execution gets its own process too; it must not dirty the
        # simulator process in which the policy's actions will execute.
        isolate_process_state=True,
        data_root=config.get("data_root"),
    )
    backend = LocalEnvironmentBackend(runtime, max_sessions=1)
    try:
        while True:
            try:
                command, args = connection.recv()
            except EOFError:
                # The model process may be terminated deliberately to rebalance
                # GPU memory.  Its private simulator should exit quietly too.
                break
            if command == "shutdown":
                break
            try:
                if command == "create":
                    result = backend.create_session(*args)
                elif command == "step":
                    result = backend.step(*args)
                elif command == "close":
                    result = backend.close_session(*args)
                else:
                    raise ValueError(f"unknown isolated-environment command: {command}")
                connection.send((True, result))
            except Exception as error:  # noqa: BLE001 - cross-process boundary
                connection.send((False, type(error).__name__, str(error)))
    finally:
        connection.close()


@dataclass
class _Child:
    process: Any
    connection: Any
    lock: threading.RLock


class BFCLIsolatedEnvironmentBackend:
    """Environment backend that owns one fresh spawned process per session."""

    benchmark = "bfcl_v3"

    def __init__(
        self,
        *,
        category: str = "multi_turn_base",
        runner_mode: str = "recovery_continue",
        tool_mode: str = "native_fc",
        data_root: str | None = None,
    ) -> None:
        self.config = {
            "category": category,
            "runner_mode": runner_mode,
            "tool_mode": tool_mode,
            "data_root": data_root,
        }
        self._context = mp.get_context("spawn")
        self._children: dict[str, _Child] = {}
        self._registry_lock = threading.RLock()

    @staticmethod
    def _rpc(child: _Child, command: str, *args):
        with child.lock:
            if not child.process.is_alive():
                raise InfrastructureError(
                    f"isolated BFCL process exited with code {child.process.exitcode}"
                )
            child.connection.send((command, args))
            try:
                response = child.connection.recv()
            except EOFError as error:
                raise InfrastructureError("isolated BFCL process closed its pipe") from error
        if response[0]:
            return response[1]
        raise InfrastructureError(
            f"isolated BFCL {command} failed: {response[1]}: {response[2]}"
        )

    def create_session(self, task_id: str, seed: int, rollout_id: str) -> SessionInfo:
        parent, child_pipe = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_child_main, args=(child_pipe, self.config), daemon=True
        )
        process.start()
        child_pipe.close()
        child = _Child(process=process, connection=parent, lock=threading.RLock())
        try:
            info = self._rpc(child, "create", task_id, seed, rollout_id)
        except Exception:
            parent.close()
            process.terminate()
            process.join(timeout=5)
            raise
        with self._registry_lock:
            self._children[info.session_id] = child
        return info

    def _resolve(self, session_id: str) -> _Child:
        with self._registry_lock:
            child = self._children.get(session_id)
        if child is None:
            raise InfrastructureError(f"unknown isolated BFCL session {session_id!r}")
        return child

    def step(
        self, session_id: str, sequence_no: int, action: StructuredAction
    ) -> StepResult:
        return self._rpc(self._resolve(session_id), "step", session_id, sequence_no, action)

    def close_session(self, session_id: str) -> None:
        with self._registry_lock:
            child = self._children.pop(session_id, None)
        if child is None:
            return
        try:
            self._rpc(child, "close", session_id)
            if child.process.is_alive():
                child.connection.send(("shutdown", ()))
        finally:
            child.connection.close()
            child.process.join(timeout=10)
            if child.process.is_alive():
                child.process.terminate()
                child.process.join(timeout=5)

    def health(self) -> HealthInfo:
        with self._registry_lock:
            count = len(self._children)
        return HealthInfo(
            ok=True,
            benchmark=self.benchmark,
            num_open_sessions=count,
            detail={"task_process_isolation": True, **self.config},
        )
