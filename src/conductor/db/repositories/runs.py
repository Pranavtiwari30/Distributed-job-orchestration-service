"""Persistence for runs and the task queue.

This module contains the only concurrency-critical SQL in the service. Three
operations matter, and each one is a single statement so that it is atomic
without an explicit lock:

* **claim** -- `UPDATE ... WHERE id IN (SELECT ... FOR UPDATE SKIP LOCKED)`
* **complete** -- a conditional `UPDATE` guarded by the lease holder's identity
* **reap** -- `UPDATE` over rows whose lease expired

Everything else in Conductor is ordinary CRUD. If you read one file, read this
one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, Result, Select, and_, func, select, tuple_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from conductor.core.retry import RetryPolicy, build_policy
from conductor.core.state import RunState, TaskState, derive_run_state
from conductor.db.models import Task, TaskAttempt, TaskRun, Workflow, WorkflowRun
from conductor.db.repositories.base import ConflictError, Cursor, Page, Repository


def utcnow() -> datetime:
    return datetime.now(UTC)


def affected_rows(result: Result[Any]) -> int:
    """How many rows a conditional UPDATE actually matched.

    `rowcount` is declared on `CursorResult` rather than on the `Result` that
    `Session.execute` is typed to return, so the narrowing happens once here
    instead of at each of the six call sites that depend on it. That count is
    load-bearing: a conditional update matching zero rows is precisely how a
    worker learns it has lost its lease.
    """
    return cast("CursorResult[Any]", result).rowcount


@dataclass(frozen=True, slots=True)
class ClaimedTask:
    """A claimed task, flattened for the worker.

    A plain value object rather than an ORM instance: the worker holds this
    across the whole execution, and a detached ORM object that lazily loads an
    attribute mid-execution would open a second transaction from a thread that
    has no business owning one.
    """

    task_run_id: uuid.UUID
    run_id: uuid.UUID
    executor: str
    params: dict[str, object]
    attempt: int
    max_attempts: int
    timeout_seconds: float
    lease_expires_at: datetime
    retry_policy: dict[str, object]

    @property
    def policy(self) -> RetryPolicy:
        return build_policy(self.retry_policy or None)


class RunRepository(Repository[WorkflowRun]):
    """Creates runs and answers questions about them."""

    model = WorkflowRun

    def create_run(
        self,
        workflow: Workflow,
        dependencies: dict[str, list[str]],
        trigger_payload: dict[str, object] | None = None,
    ) -> WorkflowRun:
        """Materialise a workflow definition into an executable run.

        Every task row is inserted up front, in one flush, with its unmet
        dependency count precomputed. The alternative -- inserting tasks as they
        become ready -- means the run's shape is unknown until it finishes, so
        neither progress reporting nor cancellation can work.
        """
        run = WorkflowRun(
            workflow_id=workflow.id,
            state=RunState.PENDING,
            trigger_payload=dict(trigger_payload or {}),
        )
        self.session.add(run)
        self.session.flush()

        tasks = list(self.session.scalars(select(Task).where(Task.workflow_id == workflow.id)))
        by_key = {task.key: task for task in tasks}
        now = utcnow()

        for task in tasks:
            unmet = len(dependencies.get(task.key, []))
            self.session.add(
                TaskRun(
                    run_id=run.id,
                    task_id=task.id,
                    # Roots are immediately claimable; everything else waits.
                    state=TaskState.READY if unmet == 0 else TaskState.PENDING,
                    pending_deps=unmet,
                    attempt=0,
                    max_attempts=build_policy(task.retry_policy or None).max_attempts,
                    priority=task.priority,
                    scheduled_at=now,
                )
            )

        if not by_key:
            raise ValueError(f"workflow {workflow.name!r} has no tasks")

        self.session.flush()
        return run

    def get_with_tasks(self, run_id: uuid.UUID) -> WorkflowRun | None:
        """Load a run and its task rows in two queries, never N+1."""
        return self.session.scalar(
            select(WorkflowRun)
            .where(WorkflowRun.id == run_id)
            .options(selectinload(WorkflowRun.task_runs).joinedload(TaskRun.task))
        )

    def list_page(
        self,
        limit: int,
        cursor: str | None = None,
        state: RunState | None = None,
        workflow_id: uuid.UUID | None = None,
    ) -> Page[WorkflowRun]:
        query: Select[tuple[WorkflowRun]] = select(WorkflowRun).order_by(
            WorkflowRun.created_at.desc(), WorkflowRun.id.desc()
        )
        if state is not None:
            query = query.where(WorkflowRun.state == state)
        if workflow_id is not None:
            query = query.where(WorkflowRun.workflow_id == workflow_id)
        if cursor is not None:
            position = Cursor.decode(cursor)
            # A genuine SQL row-value comparison. Postgres satisfies this with
            # one index seek, and -- unlike comparing the columns separately --
            # it keeps `id` as a tiebreaker for rows sharing a timestamp, which
            # is what makes the page boundary stable under concurrent inserts.
            query = query.where(
                tuple_(WorkflowRun.created_at, WorkflowRun.id)
                < (position.created_at, uuid.UUID(position.id))
            )

        rows = list(self.session.scalars(query.limit(limit + 1)))
        if len(rows) > limit:
            last = rows[limit - 1]
            return Page(rows[:limit], Cursor(last.created_at, str(last.id)).encode())
        return Page(rows, None)

    def refresh_run_state(self, run_id: uuid.UUID) -> RunState:
        """Recompute the cached run state from its tasks.

        The stored column is a read optimisation, never a source of truth: it is
        always the output of `derive_run_state` over the current task rows.
        """
        states = [
            TaskState(state)
            for state in self.session.scalars(select(TaskRun.state).where(TaskRun.run_id == run_id))
        ]
        derived = derive_run_state(states)

        values: dict[str, object] = {"state": derived, "version": WorkflowRun.version + 1}
        if derived is not RunState.PENDING:
            values["started_at"] = func.coalesce(WorkflowRun.started_at, func.now())
        if derived.is_terminal:
            values["finished_at"] = func.coalesce(WorkflowRun.finished_at, func.now())

        self.session.execute(update(WorkflowRun).where(WorkflowRun.id == run_id).values(**values))
        return derived

    def cancel(self, run_id: uuid.UUID) -> int:
        """Cancel every task that has not reached a terminal state.

        Tasks already RUNNING are cancelled too, but a worker mid-execution is
        not interrupted: it discovers the cancellation when its completion
        update matches zero rows, and discards its result.
        """
        cancellable = (TaskState.PENDING, TaskState.READY, TaskState.RUNNING, TaskState.RETRYING)
        result = self.session.execute(
            update(TaskRun)
            .where(TaskRun.run_id == run_id, TaskRun.state.in_(cancellable))
            .values(
                state=TaskState.CANCELLED,
                finished_at=func.now(),
                lease_expires_at=None,
                worker_id=None,
                version=TaskRun.version + 1,
            )
        )
        self.refresh_run_state(run_id)
        return affected_rows(result)


class TaskQueueRepository(Repository[TaskRun]):
    """The queue itself: claiming, completing, and reaping task runs."""

    model = TaskRun

    def __init__(self, session: Session, lease_seconds: float = 60.0) -> None:
        super().__init__(session)
        self.lease_seconds = lease_seconds

    # ---- claiming --------------------------------------------------------

    def claim_batch(self, worker_id: str, limit: int = 1) -> list[ClaimedTask]:
        """Atomically claim up to `limit` ready tasks for `worker_id`.

        The whole design in one statement:

            UPDATE task_runs SET state='RUNNING', ... WHERE id IN (
                SELECT id FROM task_runs
                WHERE state='READY' AND scheduled_at <= now()
                ORDER BY priority DESC, scheduled_at
                LIMIT n
                FOR UPDATE SKIP LOCKED
            ) RETURNING ...

        * `FOR UPDATE` locks the candidate rows so no other transaction can
          claim them.
        * `SKIP LOCKED` is what makes it *scale*: a concurrent claimer steps
          over rows already locked by a peer instead of blocking behind them.
          Without it, N workers serialise on the head of the queue and total
          throughput equals that of one worker -- the exact benchmark in the
          README that shows flat scaling turn linear.
        * `RETURNING` hands back the claimed rows in the same round trip, so
          claiming costs one statement rather than a SELECT then an UPDATE
          (which would be a race in two parts).

        The `ORDER BY` matches `ix_task_runs_claimable` column for column, so
        Postgres walks the partial index and stops at `LIMIT` rather than
        sorting the ready set.
        """
        now = utcnow()
        lease_until = now + timedelta(seconds=self.lease_seconds)

        candidates = (
            select(TaskRun.id)
            .where(TaskRun.state == TaskState.READY, TaskRun.scheduled_at <= now)
            .order_by(TaskRun.priority.desc(), TaskRun.scheduled_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )

        claimed = self.session.execute(
            update(TaskRun)
            .where(TaskRun.id.in_(candidates))
            .values(
                state=TaskState.RUNNING,
                worker_id=worker_id,
                lease_expires_at=lease_until,
                attempt=TaskRun.attempt + 1,
                started_at=func.coalesce(TaskRun.started_at, func.now()),
                version=TaskRun.version + 1,
            )
            .returning(
                TaskRun.id,
                TaskRun.run_id,
                TaskRun.task_id,
                TaskRun.attempt,
                TaskRun.max_attempts,
                TaskRun.lease_expires_at,
            )
            .execution_options(synchronize_session=False)
        ).all()

        if not claimed:
            return []

        # Join the definitions in one query rather than one per claimed task.
        task_ids = {row.task_id for row in claimed}
        definitions = {
            task.id: task
            for task in self.session.scalars(select(Task).where(Task.id.in_(task_ids)))
        }

        results: list[ClaimedTask] = []
        for row in claimed:
            definition = definitions[row.task_id]
            # Append-only attempt record. The UNIQUE (task_run_id, attempt)
            # constraint on this insert is what makes a duplicate execution
            # impossible to *commit*, even if one somehow gets started.
            self.session.add(
                TaskAttempt(
                    task_run_id=row.id,
                    attempt=row.attempt,
                    worker_id=worker_id,
                    state=TaskState.RUNNING,
                )
            )
            results.append(
                ClaimedTask(
                    task_run_id=row.id,
                    run_id=row.run_id,
                    executor=definition.executor,
                    params=dict(definition.params),
                    attempt=row.attempt,
                    max_attempts=row.max_attempts,
                    timeout_seconds=definition.timeout_seconds,
                    lease_expires_at=row.lease_expires_at,
                    retry_policy=dict(definition.retry_policy),
                )
            )

        try:
            self.session.flush()
        except IntegrityError as exc:
            raise ConflictError("duplicate attempt detected while claiming") from exc
        return results

    def renew_lease(self, task_run_id: uuid.UUID, worker_id: str) -> bool:
        """Extend a lease. False means the lease was lost and work must stop.

        The `worker_id` predicate is the important half: if the reaper already
        requeued this task and another worker claimed it, this update matches
        nothing, the caller learns it no longer owns the task, and abandons it
        instead of writing a result over the new owner's.
        """
        result = self.session.execute(
            update(TaskRun)
            .where(
                TaskRun.id == task_run_id,
                TaskRun.worker_id == worker_id,
                TaskRun.state == TaskState.RUNNING,
            )
            .values(
                lease_expires_at=utcnow() + timedelta(seconds=self.lease_seconds),
                version=TaskRun.version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        return affected_rows(result) == 1

    # ---- completion ------------------------------------------------------

    def record_success(
        self, task_run_id: uuid.UUID, worker_id: str, output: dict[str, object] | None = None
    ) -> bool:
        """Mark a task succeeded and unblock its children.

        Returns False if the lease was lost, in which case the caller's work is
        discarded. That is the "loser discards" half of exactly-once effects.
        """
        # RETURNING the attempt number in the same statement: re-reading it
        # afterwards would race with a concurrent reap-and-reclaim, and would
        # then close the wrong attempt row.
        closed = self.session.execute(
            update(TaskRun)
            .where(
                TaskRun.id == task_run_id,
                TaskRun.worker_id == worker_id,
                TaskRun.state == TaskState.RUNNING,
            )
            .values(
                state=TaskState.SUCCEEDED,
                finished_at=func.now(),
                output=output or {},
                lease_expires_at=None,
                worker_id=None,
                version=TaskRun.version + 1,
            )
            .returning(TaskRun.attempt)
            .execution_options(synchronize_session=False)
        ).all()
        if len(closed) != 1:
            return False

        self._close_attempt(task_run_id, closed[0].attempt, TaskState.SUCCEEDED)
        self._unblock_children(task_run_id)
        return True

    def record_failure(
        self, task_run_id: uuid.UUID, worker_id: str, error: str
    ) -> TaskState | None:
        """Record a failure and apply the task's retry policy.

        Returns the resulting state, or None if the lease had already been lost.
        """
        task_run = self.session.get(TaskRun, task_run_id)
        if task_run is None or task_run.worker_id != worker_id:
            return None

        policy = build_policy(dict(task_run.task.retry_policy) or None)
        delay = policy.next_delay(task_run.attempt)

        if delay is None:
            # Retries exhausted. DEAD_LETTER rather than FAILED when the task
            # burned every attempt, so the two are distinguishable in triage:
            # one is a task that failed, the other is a task that kept failing.
            terminal = (
                TaskState.DEAD_LETTER
                if task_run.attempt >= task_run.max_attempts
                else TaskState.FAILED
            )
            self.session.execute(
                update(TaskRun)
                .where(TaskRun.id == task_run_id, TaskRun.worker_id == worker_id)
                .values(
                    state=terminal,
                    finished_at=func.now(),
                    last_error=error[:4000],
                    lease_expires_at=None,
                    worker_id=None,
                    version=TaskRun.version + 1,
                )
                .execution_options(synchronize_session=False)
            )
            self._close_attempt(task_run_id, task_run.attempt, terminal, error)
            self._block_descendants(task_run_id)
            return terminal

        self.session.execute(
            update(TaskRun)
            .where(TaskRun.id == task_run_id, TaskRun.worker_id == worker_id)
            .values(
                # Straight back to READY with a future `scheduled_at`: the claim
                # query already filters on that, so the backoff needs no
                # separate timer and survives a scheduler restart.
                state=TaskState.READY,
                scheduled_at=utcnow() + timedelta(seconds=delay),
                last_error=error[:4000],
                lease_expires_at=None,
                worker_id=None,
                version=TaskRun.version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        self._close_attempt(task_run_id, task_run.attempt, TaskState.RETRYING, error)
        return TaskState.RETRYING

    # ---- lease expiry ----------------------------------------------------

    def reap_expired_leases(self) -> int:
        """Return tasks whose worker stopped renewing to the ready queue.

        This is what makes a `SIGKILL`ed worker survivable, and it is also
        precisely why delivery is at-least-once: a task whose worker is merely
        slow gets requeued too, and may then run twice. The unique constraint on
        `task_attempts` is what stops that second run from committing anything.
        """
        reaped = self.session.execute(
            update(TaskRun)
            .where(
                TaskRun.state == TaskState.RUNNING,
                TaskRun.lease_expires_at < utcnow(),
            )
            .values(
                state=TaskState.READY,
                worker_id=None,
                lease_expires_at=None,
                last_error="lease expired; worker presumed dead",
                version=TaskRun.version + 1,
            )
            .returning(TaskRun.id, TaskRun.attempt)
            .execution_options(synchronize_session=False)
        ).all()

        # Close the attempt rows those workers left open. Nobody else can: the
        # worker that opened them is gone, and the next attempt is a different
        # row. Left open they would be closed by whichever attempt eventually
        # succeeds, silently recording a crash as a success.
        for row in reaped:
            self._close_attempt(
                row.id,
                row.attempt,
                TaskState.FAILED,
                "lease expired; worker presumed dead",
            )
        return len(reaped)

    # ---- internals -------------------------------------------------------

    def _close_attempt(
        self,
        task_run_id: uuid.UUID,
        attempt: int,
        state: TaskState,
        error: str | None = None,
    ) -> None:
        """Stamp one specific attempt row with its outcome and duration.

        Matching on `attempt` and not merely on "the open row for this task" is
        load-bearing. A worker killed mid-task leaves its attempt row open
        forever -- it never got to report anything -- so a later, successful
        attempt would otherwise close the crashed one too and record it as
        having succeeded, in a table whose whole purpose is to say truthfully
        what happened. The reaper closes orphaned rows instead.
        """
        self.session.execute(
            update(TaskAttempt)
            .where(
                TaskAttempt.task_run_id == task_run_id,
                TaskAttempt.attempt == attempt,
                TaskAttempt.finished_at.is_(None),
            )
            .values(
                state=state,
                finished_at=func.now(),
                error=error[:4000] if error else None,
                duration_ms=func.extract("epoch", func.now() - TaskAttempt.started_at) * 1000,
            )
            .execution_options(synchronize_session=False)
        )

    def _unblock_children(self, task_run_id: uuid.UUID) -> None:
        """Decrement dependants' counters and release those that hit zero.

        Two statements rather than a read-modify-write loop: `pending_deps - 1`
        is evaluated by the database, so two siblings finishing at the same
        instant cannot both read "2 remaining" and both write "1".
        """
        from conductor.db.models import TaskDependency  # local: avoids a cycle

        task_run = self.session.get(TaskRun, task_run_id)
        if task_run is None:
            return

        child_task_ids = select(TaskDependency.child_task_id).where(
            TaskDependency.parent_task_id == task_run.task_id
        )
        sibling_filter = and_(
            TaskRun.run_id == task_run.run_id,
            TaskRun.task_id.in_(child_task_ids),
            TaskRun.state == TaskState.PENDING,
        )

        self.session.execute(
            update(TaskRun)
            .where(sibling_filter, TaskRun.pending_deps > 0)
            .values(pending_deps=TaskRun.pending_deps - 1, version=TaskRun.version + 1)
            .execution_options(synchronize_session=False)
        )
        self.session.execute(
            update(TaskRun)
            .where(sibling_filter, TaskRun.pending_deps == 0)
            .values(state=TaskState.READY, version=TaskRun.version + 1)
            .execution_options(synchronize_session=False)
        )

    def _block_descendants(self, task_run_id: uuid.UUID) -> None:
        """Mark everything downstream of a terminal failure as BLOCKED.

        A recursive CTE would do this in one statement; the iterative version is
        used because it is bounded by graph depth (small, and validated at
        submit time) and is far easier to reason about when it appears in a
        query log during an incident.
        """
        from conductor.db.models import TaskDependency  # local: avoids a cycle

        task_run = self.session.get(TaskRun, task_run_id)
        if task_run is None:
            return

        frontier = {task_run.task_id}
        while frontier:
            children = set(
                self.session.scalars(
                    select(TaskDependency.child_task_id).where(
                        TaskDependency.parent_task_id.in_(frontier)
                    )
                )
            )
            if not children:
                return
            result = self.session.execute(
                update(TaskRun)
                .where(
                    TaskRun.run_id == task_run.run_id,
                    TaskRun.task_id.in_(children),
                    TaskRun.state.in_((TaskState.PENDING, TaskState.READY)),
                )
                .values(
                    state=TaskState.BLOCKED,
                    finished_at=func.now(),
                    version=TaskRun.version + 1,
                )
                .execution_options(synchronize_session=False)
            )
            if affected_rows(result) == 0:
                return  # already blocked by another failing branch
            frontier = children
