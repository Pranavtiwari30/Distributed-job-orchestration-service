"""Pure scheduling primitives: no I/O, no clock, no database.

Everything in this package is deterministic given its inputs, which is what
makes the scheduler's logic testable without spinning up Postgres.
"""

from conductor.core.dag import CycleError, Dag, DagCursor, UnknownNodeError
from conductor.core.heap import DelayQueue, Entry
from conductor.core.lru import LruCache
from conductor.core.ratelimit import Decision, RateLimiter, TokenBucket
from conductor.core.retry import (
    ExponentialBackoff,
    FixedDelay,
    NoRetry,
    RetryPolicy,
    build_policy,
)
from conductor.core.state import (
    IllegalTransitionError,
    RunState,
    TaskState,
    assert_run_transition,
    assert_task_transition,
    derive_run_state,
)

__all__ = [
    "CycleError",
    "Dag",
    "DagCursor",
    "Decision",
    "DelayQueue",
    "Entry",
    "ExponentialBackoff",
    "FixedDelay",
    "IllegalTransitionError",
    "LruCache",
    "NoRetry",
    "RateLimiter",
    "RetryPolicy",
    "RunState",
    "TaskState",
    "TokenBucket",
    "UnknownNodeError",
    "assert_run_transition",
    "assert_task_transition",
    "build_policy",
    "derive_run_state",
]
