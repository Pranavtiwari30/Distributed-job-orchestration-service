"""Helpers for building test fixtures in the database.

Kept out of `conftest.py` so tests can compose them explicitly: a test that
builds its own graph reads better than one that receives an opaque fixture and
leaves the reader to go find out what shape it is.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from conductor.db.models import Workflow, WorkflowRun
from conductor.db.repositories.runs import RunRepository
from conductor.db.repositories.workflows import TaskSpec, WorkflowRepository


def make_workflow(
    session: Session,
    name: str = "etl",
    specs: list[TaskSpec] | None = None,
) -> Workflow:
    specs = specs or [TaskSpec(key="only", executor="noop")]
    workflow = WorkflowRepository(session).create(name=name, tasks=specs)
    session.flush()
    return workflow


def make_run(session: Session, workflow: Workflow) -> WorkflowRun:
    repo = WorkflowRepository(session)
    run = RunRepository(session).create_run(workflow, repo.dependencies(workflow.id))
    session.commit()
    return run


def linear_specs(
    count: int,
    prefix: str = "step",
    retry_policy: dict[str, object] | None = None,
) -> list[TaskSpec]:
    """`step0 -> step1 -> ... -> stepN`, the simplest non-trivial DAG.

    `retry_policy` is exposed because the default is five attempts of
    exponential backoff: a test that wants a failure to be *terminal* must say
    so, or it will merely observe a retry.
    """
    return [
        TaskSpec(
            key=f"{prefix}{index}",
            executor="noop",
            depends_on=() if index == 0 else (f"{prefix}{index - 1}",),
            retry_policy=dict(retry_policy or {}),
        )
        for index in range(count)
    ]


def fan_specs(width: int) -> list[TaskSpec]:
    """One root fanning out to `width` parallel tasks, joining at a sink."""
    specs = [TaskSpec(key="start", executor="noop")]
    specs += [
        TaskSpec(key=f"branch{i}", executor="noop", depends_on=("start",)) for i in range(width)
    ]
    specs.append(
        TaskSpec(
            key="join",
            executor="noop",
            depends_on=tuple(f"branch{i}" for i in range(width)),
        )
    )
    return specs
