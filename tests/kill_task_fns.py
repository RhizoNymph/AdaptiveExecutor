"""Module-level task functions for worker-kill isolation tests.

Must be importable by (module, qualname) so worker subprocesses can resolve
them. ``tests/`` is on ``sys.path`` via ``conftest.py``.
"""

import os
import signal
import time


def busy_sleep(seconds):
    """Sleep for ``seconds`` and return it. Used as a long task to be killed."""
    time.sleep(seconds)
    return seconds


def crash_now():
    """Abruptly kill this worker (uncatchable SIGKILL), like an OOM death."""
    os.kill(os.getpid(), signal.SIGKILL)
