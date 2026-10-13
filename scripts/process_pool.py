"""Stop only the supplied child processes, sharing one shutdown deadline."""
import subprocess
import time


def stop_process_pool(processes, grace_seconds=15, kill_seconds=5):
    processes = list(processes)
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_seconds
    pending = [process for process in processes if process.poll() is None]
    while pending and time.monotonic() < deadline:
        time.sleep(min(.05, max(0, deadline-time.monotonic())))
        pending = [process for process in pending if process.poll() is None]
    for process in pending:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + kill_seconds
    remaining = []
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            remaining.append(process.pid)
    if remaining:
        raise RuntimeError(f'Environment children did not exit after SIGKILL: {remaining}')
