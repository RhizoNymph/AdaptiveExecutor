"""Importable no-op callables used as submit targets by the scheduler sim.

Simulated tasks never actually execute. These functions exist only so the
executor's real submit-time validation (``validate_submittable``, which
re-resolves a callable by ``(module, qualname)``) succeeds, and so different
task classes map to distinct learned-profile keys (``module:qualname``).

They must remain module-level (no ``<locals>`` in ``__qualname__``) and
importable as ``sim.task_stubs`` so resolution matches ``__module__``.
"""

from typing import Any


def work(*args: Any, **kwargs: Any) -> None:  # generic CPU task
    return None


def gpu_work(*args: Any, **kwargs: Any) -> None:  # generic GPU task
    return None


def flaky(*args: Any, **kwargs: Any) -> None:  # OOM-prone task
    return None


def cold(*args: Any, **kwargs: Any) -> None:  # cold-start task (no profile/hint)
    return None
