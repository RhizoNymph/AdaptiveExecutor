"""Shared function resolution used by both submit-time validation and workers.

A submitted function is transported to worker subprocesses as a
``(module, qualname)`` pair and re-resolved there via import. This module
centralizes that resolution so the executor can reject unresolvable callables
(lambdas, closures, bound methods) *at submit time* with a clear error instead
of failing cryptically inside a worker.

Note: GPU ids elsewhere in this package are NVML device indices; they are
unrelated to function resolution.
"""

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger("adaptive_executor.resolve")

_IMPORTABILITY_HINT = (
    "The target must be importable by its module and qualified name from a "
    "worker subprocess (defined at module or class level in an importable "
    "module)."
)


class FunctionResolutionError(ValueError):
    """Raised when a ``(module, qualname)`` pair cannot be resolved to a callable."""


def resolve_function(module: str, qualname: str) -> Callable[..., Any]:
    """Resolve a callable from its module name and qualified name.

    Walks nested attribute access for qualnames like ``Outer.Inner.method``.

    Raises:
        FunctionResolutionError: if the qualname denotes a local (closure or
            lambda), the module cannot be imported, an attribute in the path is
            missing, or the resolved object is not callable.
    """
    if "<locals>" in qualname:
        raise FunctionResolutionError(
            f"Cannot resolve '{module}:{qualname}': it is defined inside another "
            f"function (a closure or lambda) and is not importable. {_IMPORTABILITY_HINT}"
        )

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise FunctionResolutionError(
            f"Cannot import module '{module}' to resolve '{qualname}': {exc}. "
            f"{_IMPORTABILITY_HINT}"
        ) from exc

    obj: Any = mod
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise FunctionResolutionError(
                f"Cannot resolve '{module}:{qualname}': attribute '{part}' not "
                f"found. {_IMPORTABILITY_HINT}"
            ) from exc

    if not callable(obj):
        raise FunctionResolutionError(
            f"Resolved '{module}:{qualname}' but it is not callable "
            f"(got {type(obj).__name__}). {_IMPORTABILITY_HINT}"
        )

    return obj


def validate_submittable(fn: Callable[..., Any]) -> None:
    """Validate that ``fn`` can be re-resolved in a worker subprocess.

    Raises:
        ValueError: if ``fn`` is a lambda/closure, cannot be resolved, or
            resolves to a different object than ``fn`` (e.g. a bound method).
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)

    if not isinstance(module, str) or not isinstance(qualname, str):
        raise ValueError(
            f"Cannot submit {fn!r}: it lacks a usable __module__/__qualname__. "
            f"{_IMPORTABILITY_HINT}"
        )

    if "<locals>" in qualname:
        raise ValueError(
            f"Cannot submit {qualname}: lambdas and closures (functions defined "
            f"inside other functions) are not importable in a worker subprocess. "
            f"{_IMPORTABILITY_HINT}"
        )

    try:
        resolved = resolve_function(module, qualname)
    except FunctionResolutionError as exc:
        raise ValueError(str(exc)) from exc

    if resolved is not fn:
        raise ValueError(
            f"Cannot submit {module}:{qualname}: it resolves to a different "
            f"object than the one submitted (e.g. a bound method or a decorated "
            f"wrapper). If this is a bound method, submit the underlying function "
            f"and pass the instance as the first argument, e.g. "
            f"submit(Cls.method, instance, ...). {_IMPORTABILITY_HINT}"
        )
