"""Persistence for workflow definitions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select, tuple_
from sqlalchemy.exc import IntegrityError

from conductor.core.dag import Dag
from conductor.core.retry import build_policy
from conductor.db.models import Task, TaskDependency, Workflow
from conductor.db.repositories.base import ConflictError, Cursor, Page, Repository


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """A validated task definition, as accepted by the API."""

    key: str
    executor: str
    params: dict[str, object] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    retry_policy: dict[str, object] = field(default_factory=dict)
    priority: int = 0
    timeout_seconds: float = 300.0


class WorkflowRepository(Repository[Workflow]):
    model = Workflow

    def create(
        self,
        name: str,
        tasks: list[TaskSpec],
        description: str | None = None,
    ) -> Workflow:
        """Publish a new version of a workflow.

        Validation happens here rather than in the request handler because it is
        a property of the *data*, not of the transport: the scheduler creating a
        workflow internally must be held to the same rules as an HTTP client.
        """
        if not tasks:
            raise ValueError("a workflow must contain at least one task")

        keys = [task.key for task in tasks]
        duplicates = {key for key in keys if keys.count(key) > 1}
        if duplicates:
            raise ValueError(f"duplicate task keys: {sorted(duplicates)}")

        known = set(keys)
        for task in tasks:
            unknown = set(task.depends_on) - known
            if unknown:
                raise ValueError(f"task {task.key!r} depends on unknown tasks: {sorted(unknown)}")
            # Fail fast on a bad retry spec: discovering it at execution time
            # means the failure surfaces minutes later, in a worker log.
            build_policy(task.retry_policy or None)

        # Raises CycleError, which the API renders as a 422 naming the cycle.
        graph = Dag.from_dependencies({task.key: list(task.depends_on) for task in tasks})
        graph.topological_order()

        version = self._next_version(name)
        workflow = Workflow(name=name, version=version, description=description)
        self.session.add(workflow)
        self.session.flush()  # assign workflow.id without committing

        by_key: dict[str, Task] = {}
        for spec in tasks:
            row = Task(
                workflow_id=workflow.id,
                key=spec.key,
                executor=spec.executor,
                params=dict(spec.params),
                retry_policy=dict(spec.retry_policy),
                priority=spec.priority,
                timeout_seconds=spec.timeout_seconds,
            )
            self.session.add(row)
            by_key[spec.key] = row
        self.session.flush()

        for spec in tasks:
            for parent_key in spec.depends_on:
                self.session.add(
                    TaskDependency(
                        parent_task_id=by_key[parent_key].id,
                        child_task_id=by_key[spec.key].id,
                    )
                )

        try:
            self.session.flush()
        except IntegrityError as exc:
            # Two concurrent publishes of the same name raced on the version
            # number. The unique constraint caught it; report it as a conflict
            # so the client simply retries.
            raise ConflictError(f"workflow {name!r} version {version} already exists") from exc
        return workflow

    def _next_version(self, name: str) -> int:
        current = self.session.scalar(
            select(func.max(Workflow.version)).where(Workflow.name == name)
        )
        return (current or 0) + 1

    def get_latest(self, name: str) -> Workflow | None:
        """Resolve a name to its newest version -- what a run submission means."""
        return self.session.scalar(
            select(Workflow).where(Workflow.name == name).order_by(Workflow.version.desc()).limit(1)
        )

    def get_version(self, name: str, version: int) -> Workflow | None:
        return self.session.scalar(
            select(Workflow).where(Workflow.name == name, Workflow.version == version)
        )

    def dependencies(self, workflow_id: uuid.UUID) -> dict[str, list[str]]:
        """Reconstruct the `{task_key: [parent keys]}` map for a workflow.

        One join rather than a query per task: the N+1 version of this is the
        classic ORM performance bug, and it lands on the run-creation path.
        """
        parent = Task.__table__.alias("parent")
        child = Task.__table__.alias("child")
        rows = self.session.execute(
            select(child.c.key, parent.c.key)
            .select_from(TaskDependency)
            .join(parent, TaskDependency.parent_task_id == parent.c.id)
            .join(child, TaskDependency.child_task_id == child.c.id)
            .where(child.c.workflow_id == workflow_id)
        ).all()

        keys = self.session.scalars(select(Task.key).where(Task.workflow_id == workflow_id)).all()
        dependencies: dict[str, list[str]] = {key: [] for key in keys}
        for child_key, parent_key in rows:
            dependencies[child_key].append(parent_key)
        return dependencies

    def list_page(self, limit: int, cursor: str | None = None) -> Page[Workflow]:
        """Keyset-paginated listing, newest first."""
        query = select(Workflow).order_by(Workflow.created_at.desc(), Workflow.id.desc())
        if cursor is not None:
            position = Cursor.decode(cursor)
            # Row-value comparison: Postgres can satisfy this with a single
            # index seek, unlike the equivalent OR-expanded predicate.
            # A genuine SQL row-value comparison. Postgres can satisfy this
            # with one index seek, and -- unlike comparing the columns
            # separately -- it keeps `id` as a tiebreaker for rows that share a
            # timestamp, which is what makes the page boundary stable.
            query = query.where(
                tuple_(Workflow.created_at, Workflow.id)
                < (position.created_at, uuid.UUID(position.id))
            )

        # Fetch one extra row: its presence is what tells us another page
        # exists, without a second COUNT query over the whole table.
        rows = list(self.session.scalars(query.limit(limit + 1)))
        if len(rows) > limit:
            last = rows[limit - 1]
            return Page(rows[:limit], Cursor(last.created_at, str(last.id)).encode())
        return Page(rows, None)
