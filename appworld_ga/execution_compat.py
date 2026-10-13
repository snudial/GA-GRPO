"""Compatibility fixes for trusted AppWorld helpers under its execution guard.

No global sqlite permission is relaxed. The upstream FTS implementation alone
receives a connector restricted to its fixed, in-memory database URI.
"""
import functools
import signal
import sys
import types


class CellDeadline(BaseException):
    """Execution control; ordinary generated ``except Exception`` cannot eat it."""


def install_fts_compat():
    import sqlite3
    import appworld.common.utils as utils
    original = utils.search_fts
    if getattr(original, '_jump_fts_compat', False):
        return
    # Install before AppWorld enables its process-local safety guard.
    connect = sqlite3.connect

    def memory_connect(database, *, uri):
        if database != 'file:dynamic_fts?mode=memory&cache=shared' or uri is not True:
            raise ValueError('Only the official in-memory FTS database is allowed')
        return connect(database, uri=uri)

    namespace = dict(original.__globals__)
    namespace['sqlite3'] = types.SimpleNamespace(connect=memory_connect)
    replacement = types.FunctionType(original.__code__, namespace, original.__name__,
                                     original.__defaults__, original.__closure__)
    functools.update_wrapper(replacement, original)
    replacement.__kwdefaults__ = original.__kwdefaults__
    replacement._jump_fts_compat = True
    # Repair already-imported aliases too; later imports use utils.search_fts.
    for name, module in list(sys.modules.items()):
        if name.startswith('appworld.') and module is not None:
            for key, value in list(vars(module).items()):
                if value is original:
                    setattr(module, key, replacement)


def install_cell_deadline():
    from appworld.environment import AppWorld
    if getattr(AppWorld._shell_run_cell, '_jump_cell_deadline', False):
        return

    def run_cell(self, code):
        self._maybe_raise_remote_environment_error('_shell_run_cell')
        if self.timeout_seconds is None:
            return self.shell.run_cell(code)
        # Match AppWorld's existing integer-second alarm and duration. Restore
        # the caller's signal handler after execution, including error paths.
        previous = signal.getsignal(signal.SIGALRM)
        def expired(signum, frame):
            raise CellDeadline()
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(int(self.timeout_seconds))
        try:
            try:
                result = self.shell.run_cell(code)
            except CellDeadline:
                return None
            if isinstance(getattr(result, 'error_in_exec', None), CellDeadline):
                return None
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    run_cell._jump_cell_deadline = True
    AppWorld._shell_run_cell = run_cell


def install():
    install_fts_compat()
    install_cell_deadline()
