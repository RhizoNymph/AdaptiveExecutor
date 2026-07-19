"""Typed, structured errors raised by the executor.

Keeping these in their own module avoids import cycles and lets callers catch
executor-specific failures without depending on the whole executor package.
"""

from typing import Literal

ResourceKind = Literal["memory", "vram"]


class InfeasibleTaskError(Exception):
    """A task can never fit on this machine, so it must fail rather than hang.

    Raised synchronously from :meth:`AdaptiveExecutor.submit` when a fresh
    estimate already exceeds known capacity, and set on a task's future by the
    dispatch thread when a (possibly crash-penalized) estimate can never fit.
    Carries structured fields so callers can react programmatically instead of
    parsing a message string.

    Attributes:
        kind: Which resource is exhausted (``"memory"`` or ``"vram"``).
        estimate_gb: The task's estimated requirement, in GB.
        capacity_gb: The capacity limit it exceeded (total minus headroom), in GB.
        retry_count: Number of resource-crash retries already applied. When > 0
            the estimate was doubled by crash penalization, so infeasibility may
            stem from that doubling rather than a bad user hint.
    """

    def __init__(
        self,
        kind: ResourceKind,
        estimate_gb: float,
        capacity_gb: float,
        retry_count: int = 0,
    ):
        self.kind: ResourceKind = kind
        self.estimate_gb = estimate_gb
        self.capacity_gb = capacity_gb
        self.retry_count = retry_count
        self.message = self._build_message()
        super().__init__(self.message)

    def _build_message(self) -> str:
        base = (
            f"Task needs {self.estimate_gb:.2f} GB {self.kind} but this machine's "
            f"usable capacity is {self.capacity_gb:.2f} GB "
            f"(total minus headroom); it can never fit."
        )
        if self.retry_count > 0:
            base += (
                f" Estimate was doubled after {self.retry_count} resource-crash "
                f"retry/retries (the task was killed under memory pressure), so "
                f"the penalized estimate no longer fits even though the original "
                f"may have been feasible."
            )
        return base

    def __repr__(self) -> str:
        return (
            f"InfeasibleTaskError(kind={self.kind!r}, "
            f"estimate_gb={self.estimate_gb!r}, capacity_gb={self.capacity_gb!r}, "
            f"retry_count={self.retry_count!r})"
        )
