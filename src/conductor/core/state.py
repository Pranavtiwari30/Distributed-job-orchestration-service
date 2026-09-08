"""Lifecycle state machines for runs and task attempts.

Every distributed-queue bug worth the name is an illegal state transition: a job
claimed twice, a job that succeeds after being dead-lettered, a run reported
complete while a task is still executing. Rather than scatter `if status == ...`
across the scheduler, the worker, and the API, the legal transitions are
declared once here and enforced at every mutation.

The table is the specification. Adding a state without adding its edges makes it
unreachable and untransitionable, which fails loudly rather than silently.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final


class TaskState(StrEnum):
    """The lifecycle of one task within one run.

    `StrEnum` so these serialise to plain strings in JSON and compare equal to
    the values stored in Postgres, with no converter at either boundary.
    """

    PENDING = "PENDING"  # dependencies not yet satisfied
    READY = "READY"  # dependencies met, awaiting a worker
    RUNNING = "RUNNING"  # claimed by a worker, lease held
    SUCCEEDED = "SUCCEEDED"  # terminal, happy path
    RETRYING = "RETRYING"  # failed, scheduled for another attempt
    FAILED = "FAILED"  # terminal, retries exhausted
    DEAD_LETTER = "DEAD_LETTER"  # terminal, quarantined for human inspection
    BLOCKED = "BLOCKED"  # terminal, an ancestor failed so this cannot run
    CANCELLED = "CANCELLED"  # terminal, withdrawn by the caller

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_TASK_STATES

    @property
    def is_active(self) -> bool:
        """True while the task still occupies scheduler or worker attention."""
        return self in {TaskState.PENDING, TaskState.READY, TaskState.RUNNING, TaskState.RETRYING}


class RunState(StrEnum):
    """The lifecycle of a whole workflow run, derived from its tasks."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_RUN_STATES


_TERMINAL_TASK_STATES: Final = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.DEAD_LETTER,
        TaskState.BLOCKED,
        TaskState.CANCELLED,
    }
)

#: Task states that mean "this run has not begun executing yet".
_NOT_YET_STARTED: Final = frozenset({TaskState.PENDING, TaskState.READY})

_TERMINAL_RUN_STATES: Final = frozenset({RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED})


# Read-only so no import site can mutate the specification at runtime.
TASK_TRANSITIONS: Final[MappingProxyType[TaskState, frozenset[TaskState]]] = MappingProxyType(
    {
        # A dependency completed, an ancestor failed, or the caller gave up.
        TaskState.PENDING: frozenset({TaskState.READY, TaskState.BLOCKED, TaskState.CANCELLED}),
        # A worker claimed it. READY -> BLOCKED is impossible: by definition
        # every dependency already succeeded.
        TaskState.READY: frozenset({TaskState.RUNNING, TaskState.CANCELLED}),
        # The three ways execution ends, plus a lease that expired because the
        # worker died -- which returns the task to RETRYING, not to FAILED,
        # since we cannot distinguish a crash from a slow task.
        TaskState.RUNNING: frozenset(
            {
                TaskState.SUCCEEDED,
                TaskState.RETRYING,
                TaskState.FAILED,
                TaskState.DEAD_LETTER,
                TaskState.CANCELLED,
            }
        ),
        # The backoff elapsed and the task is queueable again.
        TaskState.RETRYING: frozenset(
            {TaskState.READY, TaskState.FAILED, TaskState.DEAD_LETTER, TaskState.CANCELLED}
        ),
        # Terminal states have no outgoing edges. Replaying a task means
        # creating a new attempt row, never resurrecting an old one -- which is
        # what keeps the audit trail honest.
        TaskState.SUCCEEDED: frozenset(),
        TaskState.FAILED: frozenset(),
        TaskState.DEAD_LETTER: frozenset(),
        TaskState.BLOCKED: frozenset(),
        TaskState.CANCELLED: frozenset(),
    }
)

RUN_TRANSITIONS: Final[MappingProxyType[RunState, frozenset[RunState]]] = MappingProxyType(
    {
        RunState.PENDING: frozenset({RunState.RUNNING, RunState.CANCELLED}),
        RunState.RUNNING: frozenset({RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}),
        RunState.SUCCEEDED: frozenset(),
        RunState.FAILED: frozenset(),
        RunState.CANCELLED: frozenset(),
    }
)


class IllegalTransitionError(ValueError):
    """Raised on an attempt to move between states with no edge between them."""

    def __init__(self, current: StrEnum, target: StrEnum, allowed: frozenset[StrEnum]) -> None:
        self.current = current
        self.target = target
        self.allowed = allowed
        options = ", ".join(sorted(str(state) for state in allowed)) or "none (terminal)"
        super().__init__(f"cannot move {current} -> {target}; allowed: {options}")


def can_transition(current: TaskState, target: TaskState) -> bool:
    return target in TASK_TRANSITIONS[current]


def assert_task_transition(current: TaskState, target: TaskState) -> TaskState:
    """Validate a task transition, returning the target so it reads as an assignment.

    Called at the point of every status write, including inside the transaction
    that claims a task -- which is how a double-claim becomes a loud error
    instead of a task that runs twice.
    """
    if target not in TASK_TRANSITIONS[current]:
        raise IllegalTransitionError(current, target, TASK_TRANSITIONS[current])
    return target


def assert_run_transition(current: RunState, target: RunState) -> RunState:
    if target not in RUN_TRANSITIONS[current]:
        raise IllegalTransitionError(current, target, RUN_TRANSITIONS[current])
    return target


def derive_run_state(task_states: list[TaskState]) -> RunState:
    """Roll individual task states up into the run's state.

    Deliberately *derived* rather than stored-and-updated: a denormalised run
    status maintained by hand is guaranteed to drift out of sync with its tasks
    under concurrency. The cached column in the database is a read optimisation
    recomputed from this function, never an independent source of truth.

    The order of these checks is the specification, and it is load-bearing.
    """
    if not task_states:
        return RunState.PENDING

    if all(state is TaskState.SUCCEEDED for state in task_states):
        return RunState.SUCCEEDED

    # A run is PENDING until something actually starts. Note that "nothing has
    # started" is not "everything is PENDING": a freshly created run always has
    # at least one READY root, so testing for all-PENDING would make this state
    # unreachable for every real run.
    if all(state in _NOT_YET_STARTED for state in task_states):
        return RunState.PENDING

    active = any(state.is_active for state in task_states)

    # Cancellation only wins once nothing is still executing -- a worker that
    # has not yet noticed the cancellation is still doing work, and reporting
    # the run as finished while that is true would be a lie.
    if not active and any(state is TaskState.CANCELLED for state in task_states):
        return RunState.CANCELLED

    # Any live task keeps the run running, even alongside a failure: the
    # siblings of a failed task are still doing useful work.
    if active:
        return RunState.RUNNING

    return RunState.FAILED
