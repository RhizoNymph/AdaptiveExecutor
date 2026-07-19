"""End-to-end smoke test: real worker subprocesses execute real tasks, futures
resolve, profiles record observations, and shutdown flushes them to disk."""

import json

import pytest

from adaptive_executor import AdaptiveExecutor
import task_fns


def test_end_to_end_results_and_profiles(tmp_path):
    profile_path = tmp_path / "profiles.json"
    with AdaptiveExecutor(max_workers=4, profile_path=str(profile_path)) as ex:
        futures = [ex.submit(task_fns.add, i, i) for i in range(8)]
        results = [f.result(timeout=30) for f in futures]

    assert results == [i + i for i in range(8)]

    # An observation was recorded for the task function.
    prof = ex.profiles.get("task_fns", "add")
    assert prof.sample_count == 8

    # shutdown() flushed the debounced store to disk.
    assert profile_path.exists()
    data = json.loads(profile_path.read_text())
    assert "task_fns:add" in data
    assert len(data["task_fns:add"]) == 8


def test_task_exception_propagates_to_future(tmp_path):
    with AdaptiveExecutor(max_workers=2) as ex:
        fut = ex.submit(task_fns.boom)
        with pytest.raises(ValueError, match="boom"):
            fut.result(timeout=30)


def test_mixed_tasks_resolve(tmp_path):
    with AdaptiveExecutor(max_workers=3) as ex:
        f_add = ex.submit(task_fns.add, 2, 3)
        f_echo = ex.submit(task_fns.echo, "hi")
        f_slow = ex.submit(task_fns.slow, 0.02)
        assert f_add.result(timeout=30) == 5
        assert f_echo.result(timeout=30) == "hi"
        assert f_slow.result(timeout=30) == 0.02
