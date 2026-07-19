"""Make the tests directory importable so worker subprocesses (fork) and the
test modules can import helper modules like ``task_fns`` and ``fakes`` by name.
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
