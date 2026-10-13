"""Run with the isolated appworld-venv Python (not the training environment)."""
import signal
import sqlite3
import time
from types import SimpleNamespace

import pytest

pytest.importorskip('IPython')
pytest.importorskip('appworld')
from IPython.core.interactiveshell import InteractiveShell
from appworld.environment import AppWorld
from appworld.common.safety_guard import SafetyGuard
from appworld_ga.execution_compat import install, CellDeadline


def test_fts_exact_parity_without_disabling_global_guard():
    import appworld.common.utils as utils
    queries=['fitness tracker','backpack','no-such-item','']
    items=['fitness tracker','fitness mat','backpack','sofa']
    reference=[utils.search_fts(q,items) for q in queries]
    install()
    guard=SafetyGuard()
    guard.enable()
    try:
        assert sqlite3.connect(':memory:') is None
        assert [utils.search_fts(q,items) for q in queries]==reference
        connector=utils.search_fts.__globals__['sqlite3'].connect
        try: connector('/tmp/forbidden.db',uri=True)
        except ValueError: pass
        else: raise AssertionError('FTS connector accepted a disk path')
    finally:guard.disable()


def test_deadline_survives_exception_handler_and_restores_signal():
    install()
    shell=InteractiveShell()
    world=SimpleNamespace(timeout_seconds=1,shell=shell,
                          _maybe_raise_remote_environment_error=lambda _:None)
    handler=signal.getsignal(signal.SIGALRM)
    start=time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    result=AppWorld._shell_run_cell(world,'while True:\n try:\n  raise ValueError()\n except Exception:\n  pass')
    elapsed=time.clock_gettime(time.CLOCK_MONOTONIC_RAW)-start
    assert result is None
    assert .8<elapsed<4
    assert signal.getsignal(signal.SIGALRM)==handler
    assert AppWorld._shell_run_cell(world,'answer=42').success
    assert shell.user_ns['answer']==42


def test_ordinary_exceptions_and_disabled_deadline():
    install()
    world=SimpleNamespace(timeout_seconds=None,shell=InteractiveShell(),
                          _maybe_raise_remote_environment_error=lambda _:None)
    result=AppWorld._shell_run_cell(world,'raise ValueError("normal")')
    assert isinstance(result.error_in_exec,ValueError)
    assert not issubclass(CellDeadline,Exception)
