"""Integration tests for dependency resolution, retries, and run rollup.

Where `test_queue.py` covers "can two workers collide", this covers "does a
graph actually finish" -- the sequencing the scheduler exists to produce.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from conductor.core.state import RunState, TaskState
from conductor.db.models import TaskAttempt, TaskRun
from conductor.db.repositories.runs import RunRepository, TaskQueueRepository
from conductor.db.repositories.workflows import TaskSpec
from tests.integration.factories import fan_specs, linear_specs, make_run, make_workflow

pytestmark = pytest.mark.integration


def drain(session: Session, worker: str = "w1", max_steps: int = 100) -> int:
    """Run every claimable task to success until the queue is empty."""
    queue = TaskQueueRepository(session)
    executed = 0
    for _ in range(max_steps):
        batch = queue.claim_batch(worker, limit=10)
        if not batch:
            return executed
        for task in batch:
            queue.record_success(task.task_run_id, worker, {"ok": True})
            executed += 1
        session.flush()
    raise AssertionError("drain did not terminate -- the graph is not making progress")


# --------------------------------------------------------------------------
# sequencing
# --------------------------------------------------------------------------


def test_a_linear_chain_executes_in_order(session: Session) -> None:
    workflow = make_workflow(session, specs=linear_specs(5))
    make_run(session, workflow)
    queue = TaskQueueRepository(session)

    for expected in range(5):
        batch = queue.claim_batch("w1", limit=10)
        assert len(batch) == 1, f"only step{expected} should be claimable"
        task_run = session.get(TaskRun, batch[0].task_run_id)
        assert task_run.task.key == f"step{expected}"  # type: ignore[union-attr]
        queue.record_success(batch[0].task_run_id, "w1")
        session.flush()

    assert queue.claim_batch("w1") == []


def test_a_fan_out_releases_every_branch_at_once(session: Session) -> None:
    workflow = make_workflow(session, specs=fan_specs(width=6))
    make_run(session, workflow)
    queue = TaskQueueRepository(session)

    root = queue.claim_batch("w1", limit=10)
    assert len(root) == 1
    queue.record_success(root[0].task_run_id, "w1")
    session.flush()

    branches = queue.claim_batch("w1", limit=10)
    assert len(branches) == 6, "all six branches unblock together"


def test_a_join_waits_for_every_branch(session: Session) -> None:
    workflow = make_workflow(session, specs=fan_specs(width=3))
    make_run(session, workflow)
    queue = TaskQueueRepository(session)

    root = queue.claim_batch("w1")[0]
    queue.record_success(root.task_run_id, "w1")
    session.flush()

    branches = queue.claim_batch("w1", limit=10)
    for index, branch in enumerate(branches):
        queue.record_success(branch.task_run_id, "w1")
        session.flush()
        remaining = len(branches) - index - 1
        available = queue.claim_batch("peek", limit=10)
        if remaining:
            assert available == [], f"join must wait; {remaining} branches outstanding"
        else:
            assert len(available) == 1, "the join unblocks only after the last branch"
            # Put it back so the assertion above stays honest for future reads.
            queue.record_success(available[0].task_run_id, "peek")


def test_a_diamond_runs_each_task_exactly_once(session: Session) -> None:
    workflow = make_workflow(
        session,
        specs=[
            TaskSpec(key="a", executor="noop"),
            TaskSpec(key="b", executor="noop", depends_on=("a",)),
            TaskSpec(key="c", executor="noop", depends_on=("a",)),
            TaskSpec(key="d", executor="noop", depends_on=("b", "c")),
        ],
    )
    run = make_run(session, workflow)

    assert drain(session) == 4, "the join must run once, not once per parent"
    assert RunRepository(session).refresh_run_state(run.id) is RunState.SUCCEEDED


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------


def test_a_failure_retries_with_backoff_then_dead_letters(session: Session) -> None:
    workflow = make_workflow(
        session,
        specs=[
            TaskSpec(
                key="flaky",
                executor="noop",
                retry_policy={"type": "fixed", "delay": 0.0, "max_attempts": 3},
            )
        ],
    )
    make_run(session, workflow)
    queue = TaskQueueRepository(session)

    for attempt in (1, 2):
        claimed = queue.claim_batch("w1")[0]
        assert claimed.attempt == attempt
        assert queue.record_failure(claimed.task_run_id, "w1", "boom") is TaskState.RETRYING
        session.flush()

    final = queue.claim_batch("w1")[0]
    assert final.attempt == 3
    assert queue.record_failure(final.task_run_id, "w1", "boom") is TaskState.DEAD_LETTER
    session.flush()

    assert queue.claim_batch("w1") == [], "a dead-lettered task is not retried again"
    attempts = session.scalars(
        select(TaskAttempt).where(TaskAttempt.task_run_id == final.task_run_id)
    ).all()
    assert len(attempts) == 3, "every attempt is recorded, not just the last"
    assert all(a.duration_ms is not None for a in attempts), "durations feed the analytics"


def test_backoff_delays_the_next_claim(session: Session) -> None:
    workflow = make_workflow(
        session,
        specs=[
            TaskSpec(
                key="slow-retry",
                executor="noop",
                retry_policy={"type": "fixed", "delay": 3600.0, "max_attempts": 3},
            )
        ],
    )
    make_run(session, workflow)
    queue = TaskQueueRepository(session)

    claimed = queue.claim_batch("w1")[0]
    queue.record_failure(claimed.task_run_id, "w1", "boom")
    session.flush()

    assert queue.claim_batch("w1") == [], "the retry is scheduled an hour out"


def test_a_terminal_failure_blocks_everything_downstream(session: Session) -> None:
    # `type: none` so the first failure is the last one; with the default
    # policy this task would simply be retried and nothing would block.
    workflow = make_workflow(session, specs=linear_specs(4, retry_policy={"type": "none"}))
    run = make_run(session, workflow)
    queue = TaskQueueRepository(session)

    first = queue.claim_batch("w1")[0]
    queue.record_failure(first.task_run_id, "w1", "unrecoverable")
    session.flush()

    blocked = session.scalars(
        select(TaskRun).where(TaskRun.run_id == run.id, TaskRun.state == TaskState.BLOCKED)
    ).all()
    assert len(blocked) == 3, "step1..3 can never run"
    assert RunRepository(session).refresh_run_state(run.id) is RunState.FAILED


def test_an_unrelated_branch_survives_a_sibling_s_failure(session: Session) -> None:
    workflow = make_workflow(
        session,
        specs=[
            TaskSpec(key="doomed", executor="noop", retry_policy={"type": "none"}),
            TaskSpec(key="downstream", executor="noop", depends_on=("doomed",)),
            TaskSpec(key="independent", executor="noop"),
        ],
    )
    run = make_run(session, workflow)
    queue = TaskQueueRepository(session)

    batch = {session.get(TaskRun, c.task_run_id).task.key: c for c in queue.claim_batch("w1", 10)}  # type: ignore[union-attr]
    queue.record_failure(batch["doomed"].task_run_id, "w1", "nope")
    queue.record_success(batch["independent"].task_run_id, "w1")
    session.flush()

    states = {
        task_run.task.key: task_run.state
        for task_run in session.scalars(select(TaskRun).where(TaskRun.run_id == run.id))
    }
    assert states["independent"] == TaskState.SUCCEEDED
    assert states["downstream"] == TaskState.BLOCKED


# --------------------------------------------------------------------------
# run-level state and cancellation
# --------------------------------------------------------------------------


def test_run_state_tracks_its_tasks(session: Session) -> None:
    workflow = make_workflow(session, specs=linear_specs(3))
    run = make_run(session, workflow)
    runs = RunRepository(session)
    queue = TaskQueueRepository(session)

    assert runs.refresh_run_state(run.id) is RunState.PENDING

    claimed = queue.claim_batch("w1")[0]
    session.flush()
    assert runs.refresh_run_state(run.id) is RunState.RUNNING

    queue.record_success(claimed.task_run_id, "w1")
    drain(session)
    assert runs.refresh_run_state(run.id) is RunState.SUCCEEDED

    session.refresh(run)
    assert run.started_at is not None and run.finished_at is not None


def test_cancelling_a_run_stops_further_claims(session: Session) -> None:
    workflow = make_workflow(session, specs=linear_specs(5))
    run = make_run(session, workflow)
    runs = RunRepository(session)

    assert runs.cancel(run.id) == 5
    session.flush()
    assert TaskQueueRepository(session).claim_batch("w1") == []
    assert runs.refresh_run_state(run.id) is RunState.CANCELLED


def test_a_worker_mid_execution_discovers_the_cancellation(session: Session) -> None:
    # The worker is never interrupted; it learns its result is unwanted when
    # the completion update matches zero rows.
    workflow = make_workflow(session, specs=linear_specs(2))
    run = make_run(session, workflow)
    queue = TaskQueueRepository(session)

    claimed = queue.claim_batch("w1")[0]
    RunRepository(session).cancel(run.id)
    session.flush()

    assert queue.record_success(claimed.task_run_id, "w1") is False
