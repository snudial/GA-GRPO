import subprocess
import sys
import time

from scripts.process_pool import stop_process_pool


def test_shutdown_deadline_is_shared_and_other_processes_are_untouched():
    children = []
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        for _ in range(4):
            child = subprocess.Popen([sys.executable, '-u', '-c',
                'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready"); time.sleep(60)'],
                stdout=subprocess.PIPE, text=True)
            children.append(child)
            assert child.stdout.readline().strip() == 'ready'
        started = time.monotonic()
        stop_process_pool(children, grace_seconds=.4)
        assert time.monotonic()-started < 1.2  # four sequential .4s waits would fail
        assert all(child.poll() is not None for child in children)
        assert unrelated.poll() is None
        stop_process_pool(children, grace_seconds=0)  # already reaped children are safe
    finally:
        for child in [*children, unrelated]:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            if child.stdout:
                child.stdout.close()
