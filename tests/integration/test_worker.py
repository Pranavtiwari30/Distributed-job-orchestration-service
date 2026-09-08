"""End-to-end tests for the worker process against a real database.

These drive `Worker.run_once` rather than `run_forever`, so each test is
deterministic: no sleeping, no background loop to race with, and a failure
points at a specific batch.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from conductor.config import Settings
from conductor.core.state import RunState, TaskState
from conductor.db.models import TaskAttempt, TaskRun
from conductor.db.repositories.runs import RunRepository, TaskQueueRepository
from conductor.db.repositories.workflows import TaskSpec
from conductor.scheduler.reaper import Reaper
from conductor.worker.worker import Worker
from tests.integration.factories import linear_specs, make_run, make_workflow

pytestmark = pytest.mark.integration


def run_to_completion(worker: Worker, max_batches: int = 50) -> int:
    """Drive a worker until the queue is empty. Returns tasks executed."""
    executed = 0
    for _ in range(max_batches):
        count = worker.run_once()
        if count == 0:
            return executed
        executed += count
    raise AssertionError("worker did not drain the queue -- it is not making progress")


def test_a_worker_executes_a_whole_dag_in_dependency_order(
    settings: Settings, session: Session
) -> None:
    workflow = make_workflow(session, specs=linear_specs(4))
    run = make_run(session, workflow)

    worker = Worker(settings, worker_id="w-1")
    assert run_to_completion(worker) == 4

    session.expire_all()
    assert RunRepository(session).refresh_run_state(run.id) is RunState.SUCCEEDED
    assert worker.stats.succeeded == 4
    assert worker.stats.failed == 0


def test_task_output_is_persisted_and_readable(settings: Settings, session: Session) -> None:
    workflow = make_workflow(
        session,
        specs=[TaskSpec(key="greet", executor="shell", params={"command": "echo hi"})],
    )
    make_run(session, workflow)

    run_to_completion(Worker(settings, worker_id="w-1"))

    session.expire_all()
    task_run = session.scalars(select(TaskRun)).one()
    assert task_run.state == TaskState.SUCCEEDED
    assert task_run.output["stdout"].strip() == "hi"  # type: ignore[index]


def test_a_failing_task_is_retried_then_dead_lettered(settings: Settings, session: Session) -> None:
    workflow = make_workflow(
        session,
        specs=[
            TaskSpec(
                key="always-fails",
                executor="fail",
                params={"message": "kaboom"},
                retry_policy={"type": "fixed", "delay": 0.0, "max_attempts": 3},
            )
        ],
    )
    run = make_run(session, workflow)

    worker = Worker(settings, worker_id="w-1")
    assert run_to_completion(worker) == 3, "three attempts, then it stops"

    session.expire_all()
    task_run = session.scalars(select(TaskRun)).one()
    assert task_run.state == TaskState.DEAD_LETTER
    assert "kaboom" in (task_run.last_error or "")
    assert RunRepository(session).refresh_run_state(run.id) is RunState.FAILED

    attempts = session.scalars(select(TaskAttempt)).all()
    assert len(attempts) == 3, "every attempt is auditable after the fact"


def test_an_unknown_executor_fails_the_task_rather_than_the_worker(
    settings: Settings, session: Session
) -> None:
    workflow = make_workflow(
        session,
        specs=[
            TaskSpec(key="bogus", executor="does-not-exist", retry_policy={"type": "none"}),
            TaskSpec(key="fine", executor="noop"),
        ],
    )
    make_run(session, workflow)

    worker = Worker(settings, worker_id="w-1")
    run_to_completion(worker)

    session.expire_all()
    states = {tr.task.key: tr.state for tr in session.scalars(select(TaskRun))}
    assert states["bogus"] in (TaskState.FAILED, TaskState.DEAD_LETTER)
    assert states["fine"] == TaskState.SUCCEEDED, "one bad task must not stop the worker"


def test_two_workers_split_the_work_without_overlap(settings: Settings, session: Session) -> None:
    workflow = make_workflow(
        session, specs=[TaskSpec(key=f"t{i}", executor="noop") for i in range(20)]
    )
    make_run(session, workflow)

    first = Worker(settings, worker_id="w-1")
    second = Worker(settings, worker_id="w-2")
    executed = 0
    for _ in range(20):
        batch = first.run_once(limit=3) + second.run_once(limit=3)
        if batch == 0:
            break
        executed += batch

    assert executed == 20, "every task ran exactly once across both workers"
    session.expire_all()
    assert all(tr.state == TaskState.SUCCEEDED for tr in session.scalars(select(TaskRun)))


def test_the_reaper_recovers_a_task_from_a_crashed_worker(
    settings: Settings, session: Session
) -> None:
    """The crash-recovery path, without actually killing a process.

    A worker claims a task and then vanishes -- modelled by claiming the task
    and never reporting a result. The reaper requeues it and a second worker
    finishes the run.
    """
    workflow = make_workflow(session, specs=linear_specs(2))
    run = make_run(session, workflow)

    # The doomed worker claims, then dies before reporting.
    queue = TaskQueueRepository(session, settings.lease_seconds)
    claimed = queue.claim_batch("doomed", limit=1)[0]
    session.commit()

    assert Reaper(settings).reap_once() == 0, "a live lease must not be reaped"

    session.execute(
        text("UPDATE task_runs SET lease_expires_at = now() - interval '1 second' WHERE id = :i"),
        {"i": claimed.task_run_id},
    )
    session.commit()

    assert Reaper(settings).reap_once() == 1

    survivor = Worker(settings, worker_id="survivor")
    assert run_to_completion(survivor) == 2

    session.expire_all()
    assert RunRepository(session).refresh_run_state(run.id) is RunState.SUCCEEDED

    attempts = session.scalars(
        select(TaskAttempt).where(TaskAttempt.task_run_id == claimed.task_run_id)
    ).all()
    assert len(attempts) == 2, "the crashed attempt and the successful retry are both recorded"


def test_a_worker_registers_itself(settings: Settings, session: Session) -> None:
    worker = Worker(settings, worker_id="registered-worker")
    worker.register()

    session.expire_all()
    row = session.execute(
        text("SELECT hostname, pid FROM workers WHERE id = 'registered-worker'")
    ).one_or_none()
    assert row is not None


def test_worker_stats_summarise_the_run(settings: Settings, session: Session) -> None:
    workflow = make_workflow(
        session,
        specs=[
            TaskSpec(key="good", executor="noop"),
            TaskSpec(key="bad", executor="fail", retry_policy={"type": "none"}),
        ],
    )
    make_run(session, workflow)

    worker = Worker(settings, worker_id="w-1")
    run_to_completion(worker)

    stats = worker.stats.as_dict()
    assert stats["claimed"] == 2
    assert stats["succeeded"] == 1
    assert stats["failed"] == 1
    assert stats["uptime_seconds"] >= 0
