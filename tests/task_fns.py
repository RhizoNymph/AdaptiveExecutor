"""Module-level task functions used by worker subprocesses in tests.

These must be importable by (module, qualname) so they can be resolved inside
worker processes.
"""

import time


def add(a, b):
    return a + b


def echo(x):
    return x


def boom():
    raise ValueError("boom in worker")


def slow(seconds=0.02):
    time.sleep(seconds)
    return seconds


class Adder:
    """Used to test bound-method rejection."""

    def add(self, a, b):
        return a + b


def make_closure():
    """Returns a closure whose __qualname__ contains '<locals>'."""

    def inner():
        return 42

    return inner
