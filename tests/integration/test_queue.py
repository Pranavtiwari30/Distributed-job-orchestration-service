"""Integration tests for the task queue against a real PostgreSQL instance.

The concurrency tests here are the reason this project exists. They use real
threads and real connections, because the property under test -- that two
transactions racing on the same row produce exactly one winner -- is a property
of Postgres, not of the Python code wrapped around it.
"""

from __future__ import annotations

import threading
import time
from collections import Counter

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from conductor.config import Settings
from conductor.core.state import TaskState
from conductor.db.models import TaskAttempt, TaskRun
from conductor.db.repositories.runs import TaskQueueRepository
from conductor.db.repositories.workflows import TaskSpec
from conductor.db.session import get_sessionmaker
from tests.integration.factories import linear_specs, make_run, make_workflow

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------
# claiming
# --------------------------------------------------------------------------


def test_claiming_moves_a_ready_task_to_running(session: Session) -> None:
    make_run(session, make_workflow(session))
    queue = TaskQueueRepository(session)

    claimed = queue.claim_batch("worker-1", limit=5)
    assert len(claimed) == 1
    assert claimed[0].attempt == 1, "the first claim is attempt 1"

    task_run = session.get(TaskRun, claimed[0].task_run_id)
    assert task_run is not None
    assert task_run.state == TaskState.RUNNING
    assert task_run.worker_id == "worker-1"
    assert task_run.lease_expires_at is not None


def test_only_root_tasks_are_claimable_initially(session: Session) -> None:
    workflow = make_workflow(session, specs=linear_specs(4))
    make_run(session, workflow)

    claimed = TaskQueueRepository(session).claim_batch("worker-1", limit=10)
    assert len(claimed) == 1, "step1..3 are blocked behind step0"


def test_claims_respect_priority_then_age(session: Session) -> None:
    workflow = make_workflow(
        session,
        specs=[
            TaskSpec(key="low", executor="noop", priority=0),
            TaskSpec(key="urgent", executor="noop", priority=10),
            TaskSpec(key="medium", executor="noop", priority=5),
        ],
    )
    make_run(session, workflow)

    queue = TaskQueueRepository(session)
    # Claim one at a time so the ordering is observable rather than batched.
    keys = []
    for _ in range(3):
        batch = queue.claim_batch(f"w{len(keys)}", limit=1)
        task_run = session.get(TaskRun, batch[0].task_run_id)
        keys.append(task_run.task.key)  # type: ignore[union-attr]

    assert keys == ["urgent", "medium", "low"], "higher priority must be served first"


def test_a_future_scheduled_at_hides_a_task_from_the_queue(session: Session) -> None:
    # This is how backoff is implemented: no timer, just a predicate the claim
    # query already evaluates.
    run = make_run(session, make_workflow(session))
    session.execute(
        text("UPDATE task_runs SET scheduled_at = now() + interval '1 hour' WHERE run_id = :r"),
        {"r": run.id},
    )
    session.flush()
    assert TaskQueueRepository(session).claim_batch("worker-1") == []


def test_an_empty_queue_returns_nothing_rather_than_blocking(session: Session) -> None:
    assert TaskQueueRepository(session).claim_batch("worker-1", limit=10) == []


# --------------------------------------------------------------------------
# the exactly-once property, under real concurrency
# --------------------------------------------------------------------------


def test_concurrent_workers_never_claim_the_same_task_twice(
    settings: Settings, session: Session
) -> None:
    """The headline guarantee: N threads, one queue, no double-claims.

    Twenty threads race to drain two hundred tasks. Every claim is recorded;
    at the end, no task id may appear twice, and every task must have been
    claimed exactly once. Without `SKIP LOCKED` this still passes -- but the
    threads serialise, which the throughput test below is there to catch.
    """
    task_count = 200
    worker_count = 20

    workflow = make_workflow(
        session,
        specs=[TaskSpec(key=f"t{i}", executor="noop") for i in range(task_count)],
    )
    make_run(session, workflow)
    session.commit()

    factory: sessionmaker[Session] = get_sessionmaker(settings)
    claims: list[str] = []
    lock = threading.Lock()
    start = threading.Barrier(worker_count)

    def drain(worker_index: int) -> None:
        start.wait()  # maximise contention: everyone starts together
        with factory() as worker_session:
            queue = TaskQueueRepository(worker_session, settings.lease_seconds)
            while True:
                batch = queue.claim_batch(f"worker-{worker_index}", limit=3)
                worker_session.commit()
                if not batch:
                    return
                with lock:
                    claims.extend(str(task.task_run_id) for task in batch)

    threads = [threading.Thread(target=drain, args=(i,)) for i in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    duplicates = [task_id for task_id, count in Counter(claims).items() if count > 1]
    assert duplicates == [], f"{len(duplicates)} tasks were claimed more than once"
    assert len(claims) == task_count, "every task must be claimed exactly once"


def test_every_attempt_row_is_unique_under_concurrency(
    settings: Settings, session: Session
) -> None:
    """The database-level backstop for exactly-once *effects*.

    Even granting a double-claim, `UNIQUE (task_run_id, attempt)` means only one
    of the two racers can commit its attempt row. This asserts the constraint is
    actually doing that work rather than merely being declared.
    """
    workflow = make_workflow(
        session, specs=[TaskSpec(key=f"t{i}", executor="noop") for i in range(50)]
    )
    make_run(session, workflow)
    session.commit()

    factory: sessionmaker[Session] = get_sessionmaker(settings)

    def drain(index: int) -> None:
        with factory() as worker_session:
            queue = TaskQueueRepository(worker_session, settings.lease_seconds)
            while queue.claim_batch(f"w{index}", limit=2):
                worker_session.commit()

    threads = [threading.Thread(target=drain, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    pairs = session.execute(
        text(
            "SELECT task_run_id, attempt, count(*) FROM task_attempts "
            "GROUP BY 1, 2 HAVING count(*) > 1"
        )
    ).all()
    assert pairs == [], "the unique constraint permitted a duplicate attempt"


def test_skip_locked_actually_lets_workers_overtake_each_other(
    settings: Settings, session: Session
) -> None:
    """Prove the claim is `SKIP LOCKED` and not merely `FOR UPDATE`.

    One transaction claims a task and holds its lock open. A second must be able
    to claim a *different* task immediately instead of blocking behind the
    first. Under plain `FOR UPDATE` this test times out -- which is exactly the
    throughput collapse the design note in `runs.py` describes.
    """
    workflow = make_workflow(
        session, specs=[TaskSpec(key="a", executor="noop"), TaskSpec(key="b", executor="noop")]
    )
    make_run(session, workflow)
    session.commit()

    factory: sessionmaker[Session] = get_sessionmaker(settings)
    holder_has_claimed = threading.Event()
    release_holder = threading.Event()

    def hold_a_lock() -> None:
        with factory() as holding_session:
            queue = TaskQueueRepository(holding_session, settings.lease_seconds)
            assert queue.claim_batch("holder", limit=1)
            holder_has_claimed.set()
            release_holder.wait(timeout=10)  # keep the transaction open
            holding_session.commit()

    holder = threading.Thread(target=hold_a_lock)
    holder.start()
    assert holder_has_claimed.wait(timeout=10)

    try:
        with factory() as overtaker_session:
            queue = TaskQueueRepository(overtaker_session, settings.lease_seconds)
            began = time.monotonic()
            claimed = queue.claim_batch("overtaker", limit=1)
            elapsed = time.monotonic() - began
            overtaker_session.commit()

        assert len(claimed) == 1, "the second worker should have taken the other task"
        assert elapsed < 2.0, f"claim blocked for {elapsed:.2f}s -- SKIP LOCKED is not in effect"
    finally:
        release_holder.set()
        holder.join(timeout=10)


# --------------------------------------------------------------------------
# leases and crash recovery
# --------------------------------------------------------------------------


def test_a_live_worker_can_renew_its_lease(session: Session) -> None:
    make_run(session, make_workflow(session))
    queue = TaskQueueRepository(session, lease_seconds=2.0)
    claimed = queue.claim_batch("worker-1")[0]

    assert queue.renew_lease(claimed.task_run_id, "worker-1") is True
    session.flush()
    task_run = session.get(TaskRun, claimed.task_run_id)
    session.refresh(task_run)  # type: ignore[arg-type]
    assert task_run.lease_expires_at > claimed.lease_expires_at  # type: ignore[union-attr,operator]


def test_a_worker_that_lost_its_lease_is_told_so(session: Session) -> None:
    # The single most important false return in the system: it is what stops a
    # revived zombie worker from overwriting the new owner's result.
    make_run(session, make_workflow(session))
    queue = TaskQueueRepository(session, lease_seconds=2.0)
    claimed = queue.claim_batch("worker-1")[0]

    session.execute(
        text("UPDATE task_runs SET lease_expires_at = now() - interval '1 second' WHERE id = :i"),
        {"i": claimed.task_run_id},
    )
    assert queue.reap_expired_leases() == 1
    assert queue.renew_lease(claimed.task_run_id, "worker-1") is False


def test_a_killed_worker_s_task_returns_to_the_queue(session: Session) -> None:
    """Simulated `SIGKILL`: the worker simply stops renewing."""
    make_run(session, make_workflow(session))
    queue = TaskQueueRepository(session, lease_seconds=2.0)
    claimed = queue.claim_batch("doomed-worker")[0]
    assert TaskQueueRepository(session).claim_batch("other") == [], "task is held"

    session.execute(
        text("UPDATE task_runs SET lease_expires_at = now() - interval '1 second' WHERE id = :i"),
        {"i": claimed.task_run_id},
    )
    assert queue.reap_expired_leases() == 1

    reclaimed = queue.claim_batch("rescuer")
    assert len(reclaimed) == 1
    assert reclaimed[0].task_run_id == claimed.task_run_id
    assert reclaimed[0].attempt == 2, "the retry is a new attempt, not a replay of the first"


def test_the_reaper_leaves_live_leases_alone(session: Session) -> None:
    make_run(session, make_workflow(session))
    queue = TaskQueueRepository(session, lease_seconds=300.0)
    queue.claim_batch("healthy-worker")
    assert queue.reap_expired_leases() == 0


def test_a_zombie_worker_cannot_overwrite_the_new_owner_s_result(session: Session) -> None:
    """The full split-brain scenario, end to end.

    Worker A is presumed dead and its task is reassigned to worker B. A then
    wakes up and reports success. A's write must be rejected, and B's must win.
    """
    make_run(session, make_workflow(session))
    queue = TaskQueueRepository(session, lease_seconds=2.0)
    zombie = queue.claim_batch("worker-a")[0]

    session.execute(
        text("UPDATE task_runs SET lease_expires_at = now() - interval '1 second' WHERE id = :i"),
        {"i": zombie.task_run_id},
    )
    queue.reap_expired_leases()
    new_owner = queue.claim_batch("worker-b")[0]
    assert new_owner.task_run_id == zombie.task_run_id

    assert queue.record_success(zombie.task_run_id, "worker-a", {"stale": True}) is False
    assert queue.record_success(new_owner.task_run_id, "worker-b", {"fresh": True}) is True

    task_run = session.get(TaskRun, zombie.task_run_id)
    session.refresh(task_run)  # type: ignore[arg-type]
    assert task_run.output == {"fresh": True}  # type: ignore[union-attr]


def test_a_crashed_attempt_is_not_recorded_as_a_success(session: Session) -> None:
    """Regression test: the audit trail must not launder a crash into a success.

    `_close_attempt` originally matched "the open attempt row for this task"
    rather than a specific attempt number. A worker killed mid-task leaves its
    attempt row open forever -- it never got to report anything -- so when a
    later attempt succeeded it closed *both* rows, stamping the crashed attempt
    SUCCEEDED with a duration covering the entire outage.

    Found by killing a worker in a live run, not by the suite: every existing
    test either had a single attempt, or failed explicitly rather than crashing.
    """
    make_run(session, make_workflow(session))
    queue = TaskQueueRepository(session, lease_seconds=2.0)

    crashed = queue.claim_batch("worker-that-dies")[0]
    session.execute(
        text("UPDATE task_runs SET lease_expires_at = now() - interval '1 second' WHERE id = :i"),
        {"i": crashed.task_run_id},
    )
    assert queue.reap_expired_leases() == 1

    rescued = queue.claim_batch("worker-that-survives")[0]
    assert rescued.attempt == 2
    assert queue.record_success(rescued.task_run_id, "worker-that-survives", {"ok": True}) is True
    session.flush()

    attempts = {
        attempt.attempt: attempt
        for attempt in session.scalars(
            select(TaskAttempt).where(TaskAttempt.task_run_id == crashed.task_run_id)
        )
    }
    assert set(attempts) == {1, 2}, "both attempts are on record"
    assert attempts[1].state == TaskState.FAILED, "the crashed attempt must not read as success"
    assert "lease expired" in (attempts[1].error or "")
    assert attempts[2].state == TaskState.SUCCEEDED
    assert attempts[2].worker_id == "worker-that-survives"
