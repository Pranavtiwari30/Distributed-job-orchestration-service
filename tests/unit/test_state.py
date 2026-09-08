"""Unit tests for the lifecycle state machines.

These read as a specification: the transition table is the contract every other
layer is written against, so the tests assert the contract rather than the
implementation.
"""

from __future__ import annotations

import pytest

from conductor.core.state import (
    RUN_TRANSITIONS,
    TASK_TRANSITIONS,
    IllegalTransitionError,
    RunState,
    TaskState,
    assert_run_transition,
    assert_task_transition,
    can_transition,
    derive_run_state,
)

# --------------------------------------------------------------------------
# the happy path and its neighbours
# --------------------------------------------------------------------------


def test_the_nominal_lifecycle_is_legal_end_to_end() -> None:
    state = TaskState.PENDING
    for target in (TaskState.READY, TaskState.RUNNING, TaskState.SUCCEEDED):
        state = assert_task_transition(state, target)
    assert state is TaskState.SUCCEEDED


def test_the_retry_loop_is_legal() -> None:
    state = assert_task_transition(TaskState.RUNNING, TaskState.RETRYING)
    state = assert_task_transition(state, TaskState.READY)
    state = assert_task_transition(state, TaskState.RUNNING)
    assert assert_task_transition(state, TaskState.DEAD_LETTER) is TaskState.DEAD_LETTER


def test_an_expired_lease_returns_the_task_for_retry_not_for_failure() -> None:
    # A crashed worker and a slow worker look identical from here, so a lost
    # lease must not be recorded as the task having failed.
    assert can_transition(TaskState.RUNNING, TaskState.RETRYING)


# --------------------------------------------------------------------------
# the transitions that must not exist
# --------------------------------------------------------------------------


def test_a_task_cannot_run_before_its_dependencies_are_met() -> None:
    with pytest.raises(IllegalTransitionError):
        assert_task_transition(TaskState.PENDING, TaskState.RUNNING)


def test_a_second_claim_of_a_running_task_is_rejected() -> None:
    # This is the double-execution bug, caught at the type level rather than
    # discovered in production.
    with pytest.raises(IllegalTransitionError):
        assert_task_transition(TaskState.RUNNING, TaskState.RUNNING)


def test_a_terminal_task_can_never_move_again() -> None:
    for state in TaskState:
        if not state.is_terminal:
            continue
        assert TASK_TRANSITIONS[state] == frozenset(), f"{state} must be a sink"
        with pytest.raises(IllegalTransitionError):
            assert_task_transition(state, TaskState.RUNNING)


def test_a_dead_lettered_task_cannot_later_report_success() -> None:
    with pytest.raises(IllegalTransitionError):
        assert_task_transition(TaskState.DEAD_LETTER, TaskState.SUCCEEDED)


def test_a_ready_task_can_never_become_blocked() -> None:
    # READY means every dependency already succeeded; there is nothing left to
    # block on. If this edge existed it would mask a scheduler bug.
    assert not can_transition(TaskState.READY, TaskState.BLOCKED)


def test_the_error_names_the_legal_alternatives() -> None:
    with pytest.raises(IllegalTransitionError) as excinfo:
        assert_task_transition(TaskState.PENDING, TaskState.SUCCEEDED)
    assert "READY" in str(excinfo.value)


# --------------------------------------------------------------------------
# table integrity
# --------------------------------------------------------------------------


def test_every_state_appears_in_the_table() -> None:
    # Adding a state without wiring its edges leaves it unreachable; this
    # fails at test time rather than at 3am.
    assert set(TASK_TRANSITIONS) == set(TaskState)
    assert set(RUN_TRANSITIONS) == set(RunState)


def test_every_non_terminal_state_is_escapable() -> None:
    for state, targets in TASK_TRANSITIONS.items():
        assert bool(targets) != state.is_terminal, f"{state} is stuck or wrongly terminal"


def test_the_table_is_immutable_at_runtime() -> None:
    with pytest.raises(TypeError):
        TASK_TRANSITIONS[TaskState.SUCCEEDED] = frozenset({TaskState.RUNNING})  # type: ignore[index]


def test_states_serialise_as_plain_strings() -> None:
    # StrEnum means no converter at the JSON or the Postgres boundary.
    assert TaskState.RUNNING == "RUNNING"
    assert f"{TaskState.RUNNING}" == "RUNNING"


# --------------------------------------------------------------------------
# run state, derived from tasks
# --------------------------------------------------------------------------


def test_an_empty_run_is_pending() -> None:
    assert derive_run_state([]) is RunState.PENDING


def test_a_run_succeeds_only_when_every_task_succeeded() -> None:
    assert derive_run_state([TaskState.SUCCEEDED, TaskState.SUCCEEDED]) is RunState.SUCCEEDED
    assert derive_run_state([TaskState.SUCCEEDED, TaskState.FAILED]) is RunState.FAILED


def test_a_failure_alongside_live_siblings_keeps_the_run_running() -> None:
    # The siblings of a failed task are still doing useful work; reporting the
    # run as failed while they execute would be a lie the API tells its caller.
    assert derive_run_state([TaskState.FAILED, TaskState.RUNNING]) is RunState.RUNNING


def test_a_run_is_pending_until_something_actually_starts() -> None:
    assert derive_run_state([TaskState.PENDING, TaskState.PENDING]) is RunState.PENDING
    # A freshly created run always has a READY root, so READY must still count
    # as "not started" -- otherwise PENDING is a state no real run ever reaches.
    assert derive_run_state([TaskState.PENDING, TaskState.READY]) is RunState.PENDING
    assert derive_run_state([TaskState.RUNNING, TaskState.PENDING]) is RunState.RUNNING
    assert derive_run_state([TaskState.SUCCEEDED, TaskState.READY]) is RunState.RUNNING


def test_a_blocked_descendant_fails_the_run() -> None:
    assert derive_run_state([TaskState.FAILED, TaskState.BLOCKED]) is RunState.FAILED


def test_cancellation_wins_once_nothing_is_still_running() -> None:
    assert derive_run_state([TaskState.CANCELLED, TaskState.RUNNING]) is RunState.RUNNING
    assert derive_run_state([TaskState.SUCCEEDED, TaskState.CANCELLED]) is RunState.CANCELLED


def test_run_transitions_are_enforced_too() -> None:
    assert assert_run_transition(RunState.PENDING, RunState.RUNNING) is RunState.RUNNING
    with pytest.raises(IllegalTransitionError):
        assert_run_transition(RunState.SUCCEEDED, RunState.RUNNING)
