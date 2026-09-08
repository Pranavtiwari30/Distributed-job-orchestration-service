"""The relational schema.

Design notes worth defending in review:

* **The queue lives in Postgres.** No Redis, no RabbitMQ. One fewer moving part,
  and -- more importantly -- claiming a task and recording *why* it was claimed
  happen in the same transaction. With an external broker those are two systems
  that can disagree, and reconciling them is where the lost jobs come from.

* **Third normal form, with one deliberate exception.** `workflow_runs.state` is
  derivable from its task rows and is therefore redundant. It is stored anyway
  because the dashboard lists runs by state, and deriving it per row means an
  aggregate over `task_runs` for every entry on the page. It is written only by
  `derive_run_state`, never by hand.

* **The dependency edges are rows, not JSON.** A `depends_on` JSON array is
  easier to write and impossible to query: "what breaks if this task fails" is
  a recursive CTE over `task_dependencies` and nothing at all over a blob.

* **Attempts are append-only.** A retry inserts a new `task_attempts` row rather
  than mutating the previous one, so the history survives and
  `UNIQUE (task_run_id, attempt)` can enforce exactly-once effects.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from conductor.core.state import RunState, TaskState


class Base(DeclarativeBase):
    """Declarative base with the conventions every table follows."""

    type_annotation_map = {
        dict[str, object]: JSONB,
        datetime: DateTime(timezone=True),
    }


def _uuid_pk() -> Mapped[uuid.UUID]:
    """UUIDv4 primary key, generated client-side.

    Client-side generation lets the API build a whole run graph -- parents and
    children, cross-referenced -- in memory and insert it in one statement,
    with no round trip per row to learn a serial id.
    """
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    """A creation timestamp stamped by the database.

    `server_default` rather than a Python default: the database's clock is the
    only one every writer shares, and rows inserted by a worker whose clock has
    drifted must still sort correctly against everyone else's.

    Deliberately *not* `index=True`. Each table that needs to sort by this
    column already declares a composite index leading with it, and a redundant
    single-column index is pure write amplification on the hottest tables.
    """
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Workflow(Base):
    """An immutable, versioned workflow definition.

    Definitions are never edited in place. Editing one would retroactively
    change what a historical run *was*, which makes the audit trail worthless;
    a change publishes `version + 1` instead.
    """

    __tablename__ = "workflows"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()

    tasks: Mapped[list[Task]] = relationship(
        back_populates="workflow", cascade="all, delete-orphan", lazy="selectin"
    )
    runs: Mapped[list[WorkflowRun]] = relationship(back_populates="workflow")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_workflows_name_version"),
        CheckConstraint("version >= 1", name="ck_workflows_version_positive"),
        # Resolving "the current definition of X" is a hot lookup; DESC ordering
        # lets it be an index-only scan of the first row.
        Index("ix_workflows_name_version_desc", "name", text("version DESC")),
    )


class Task(Base):
    """One node of a workflow definition."""

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    executor: Mapped[str] = mapped_column(String(64), nullable=False)
    params: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    retry_policy: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    # Higher runs first. A signed integer so a task can be explicitly
    # deprioritised below the default without renumbering everything else.
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timeout_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=300.0)

    workflow: Mapped[Workflow] = relationship(back_populates="tasks")

    __table_args__ = (
        # Task keys are unique per workflow, not globally: "notify" is a
        # perfectly good name in two unrelated pipelines.
        UniqueConstraint("workflow_id", "key", name="uq_tasks_workflow_key"),
        CheckConstraint("timeout_seconds > 0", name="ck_tasks_timeout_positive"),
        CheckConstraint("length(key) > 0", name="ck_tasks_key_not_blank"),
    )


class TaskDependency(Base):
    """A single DAG edge: `child` waits for `parent`.

    Acyclicity is enforced in the application, at submit time, by attempting a
    topological sort. It could be enforced here with a recursive-CTE trigger,
    but that pays the cost on every insert to catch a condition the API already
    rejects with a far better error message.
    """

    __tablename__ = "task_dependencies"

    parent_task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True
    )
    child_task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True
    )

    __table_args__ = (
        CheckConstraint("parent_task_id <> child_task_id", name="ck_task_deps_no_self_loop"),
        # The PK covers parent->child lookups; this covers the reverse
        # ("what am I waiting on"), which the scheduler asks just as often.
        Index("ix_task_deps_child", "child_task_id"),
    )


class WorkflowRun(Base):
    """One execution of one workflow version."""

    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflows.id", ondelete="RESTRICT"), nullable=False
    )
    state: Mapped[str] = mapped_column(String(32), nullable=False, default=RunState.PENDING)
    trigger_payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = _created_at()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Optimistic lock. Every update carries `WHERE version = :seen`; a mismatch
    # means somebody else wrote first and this transaction must re-read.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    workflow: Mapped[Workflow] = relationship(back_populates="runs")
    task_runs: Mapped[list[TaskRun]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NOT NULL",
            name="ck_runs_finished_implies_started",
        ),
        # The dashboard's default view: newest runs in a given state.
        Index("ix_runs_state_created", "state", text("created_at DESC")),
        Index("ix_runs_workflow_created", "workflow_id", text("created_at DESC")),
    )


class TaskRun(Base):
    """One task within one run: the queue row workers compete for.

    This is the hottest table in the system. Every column on it is either read
    by the claim query or written by the worker holding the lease.
    """

    __tablename__ = "task_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="RESTRICT"), nullable=False
    )
    state: Mapped[str] = mapped_column(String(32), nullable=False, default=TaskState.PENDING)
    # Unmet dependency count, mirroring `DagCursor` in the database so a
    # restarted scheduler recovers its state instead of rebuilding the graph.
    pending_deps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # When this task becomes claimable: now for a ready task, now + backoff for
    # a retry. The claim query filters on it, so it is half of the hot index.
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_id: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    output: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    run: Mapped[WorkflowRun] = relationship(back_populates="task_runs")
    task: Mapped[Task] = relationship(lazy="joined")
    attempts: Mapped[list[TaskAttempt]] = relationship(
        back_populates="task_run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("run_id", "task_id", name="uq_task_runs_run_task"),
        CheckConstraint("attempt >= 0", name="ck_task_runs_attempt_non_negative"),
        CheckConstraint("attempt <= max_attempts", name="ck_task_runs_attempt_within_budget"),
        CheckConstraint("pending_deps >= 0", name="ck_task_runs_pending_deps_non_negative"),
        # A running task holds a lease and a worker; nothing else may. This is
        # the invariant the whole exactly-once story rests on, so it is a
        # constraint rather than a convention.
        CheckConstraint(
            "(state = 'RUNNING') = (lease_expires_at IS NOT NULL AND worker_id IS NOT NULL)",
            name="ck_task_runs_lease_iff_running",
        ),
        # THE claim index. Partial, because READY rows are a tiny fraction of
        # the table, and the queue must not slow down as history accumulates:
        # the index stays proportional to the backlog, not to all time.
        # Column order matches the claim query's ORDER BY exactly, so Postgres
        # walks it and stops at LIMIT 1 instead of sorting.
        Index(
            "ix_task_runs_claimable",
            text("priority DESC"),
            "scheduled_at",
            postgresql_where=text("state = 'READY'"),
        ),
        # The reaper's index: expired leases only. Also partial, and also tiny.
        Index(
            "ix_task_runs_expired_leases",
            "lease_expires_at",
            postgresql_where=text("state = 'RUNNING'"),
        ),
        Index("ix_task_runs_run_state", "run_id", "state"),
    )


class TaskAttempt(Base):
    """An append-only record of one execution attempt.

    `UNIQUE (task_run_id, attempt)` is what converts at-least-once *delivery*
    into exactly-once *effects*: two workers racing on the same attempt both try
    to insert this row, exactly one succeeds, and the loser discards its work
    instead of committing a duplicate side effect.
    """

    __tablename__ = "task_attempts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    task_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("task_runs.id", ondelete="CASCADE"), nullable=False
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    worker_id: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = _created_at()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)

    task_run: Mapped[TaskRun] = relationship(back_populates="attempts")

    __table_args__ = (
        UniqueConstraint("task_run_id", "attempt", name="uq_task_attempts_run_attempt"),
        CheckConstraint("attempt >= 1", name="ck_task_attempts_attempt_positive"),
        # Serves the p50/p95-by-executor analytics query.
        Index("ix_task_attempts_started", text("started_at DESC")),
    )


class Worker(Base):
    """A registered worker process and its last heartbeat."""

    __tablename__ = "workers"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(255), nullable=False)
    pid: Mapped[int] = mapped_column(Integer, nullable=False)
    registered_at: Mapped[datetime] = _created_at()
    last_heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    tasks_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (Index("ix_workers_heartbeat", text("last_heartbeat_at DESC")),)


class IdempotencyKey(Base):
    """Deduplicates run submissions.

    A client whose POST times out cannot know whether the run was created. It
    retries with the same key; the unique PK makes the second insert fail, and
    the API returns the original run instead of starting a second one.

    `request_hash` guards against the nastier case: the same key sent with
    *different* parameters, which is a client bug that must be reported (409),
    not silently resolved to whichever request happened to arrive first.
    """

    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = _created_at()
