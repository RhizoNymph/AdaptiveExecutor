"""Bug 6: unresolvable functions must be rejected at submit time with a clear
error, using the same resolver the worker uses."""

import pytest

import adaptive_executor.adaptive_executor as ae
import adaptive_executor.worker as worker_mod
import task_fns
from adaptive_executor.resolve import (
    FunctionResolutionError,
    resolve_function,
    validate_submittable,
)


def test_resolve_top_level_function():
    fn = resolve_function("task_fns", "add")
    assert fn is task_fns.add


def test_resolve_nested_attribute():
    fn = resolve_function("task_fns", "Adder.add")
    assert fn is task_fns.Adder.add


def test_resolve_missing_module_raises():
    with pytest.raises(FunctionResolutionError):
        resolve_function("no_such_module_xyz", "add")


def test_resolve_missing_attribute_raises():
    with pytest.raises(FunctionResolutionError):
        resolve_function("task_fns", "does_not_exist")


def test_resolve_locals_rejected():
    with pytest.raises(FunctionResolutionError):
        resolve_function("task_fns", "make_closure.<locals>.inner")


def test_validate_accepts_top_level():
    validate_submittable(task_fns.add)  # must not raise


def test_validate_rejects_lambda():
    with pytest.raises(ValueError, match="lambda"):
        validate_submittable(lambda x: x)


def test_validate_rejects_closure():
    closure = task_fns.make_closure()
    with pytest.raises(ValueError, match="closure|lambda|importable"):
        validate_submittable(closure)


def test_validate_rejects_bound_method():
    inst = task_fns.Adder()
    with pytest.raises(ValueError, match="bound method"):
        validate_submittable(inst.add)


def test_submit_rejects_lambda_without_spawning(tmp_path):
    ex = ae.AdaptiveExecutor(max_workers=2)
    try:
        with pytest.raises(ValueError):
            ex.submit(lambda x: x, 1)
        # No worker should have been spawned by a rejected submit.
        assert ex.workers == {}
    finally:
        ex.shutdown()


def test_worker_and_submit_share_resolver():
    # Both call sites must use the exact same resolution helpers.
    assert worker_mod.resolve_function is resolve_function
    assert ae.validate_submittable is validate_submittable
