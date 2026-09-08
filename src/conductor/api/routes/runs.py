"""Run submission and inspection endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response, status
from sqlalchemy import select

from conductor.api.deps import (
    IdempotencyKeyHeader,
    RateLimited,
    SessionDep,
    SettingsDep,
    request_fingerprint,
)
from conductor.api.errors import ProblemDetail
from conductor.api.schemas import PageMeta, RunCreate, RunPage, RunResponse, TaskRunResponse
from conductor.core.state import RunState
from conductor.db.models import IdempotencyKey, WorkflowRun
from conductor.db.repositories.base import NotFoundError
from conductor.db.repositories.runs import RunRepository
from conductor.db.repositories.workflows import WorkflowRepository

router = APIRouter(prefix="/runs", tags=["runs"], dependencies=[RateLimited])


def _to_response(run: WorkflowRun) -> RunResponse:
    return RunResponse(
        id=run.id,
        workflow_id=run.workflow_id,
        state=run.state,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        tasks=sorted(
            (
                TaskRunResponse(
                    id=task_run.id,
                    key=task_run.task.key,
                    state=task_run.state,
                    attempt=task_run.attempt,
                    max_attempts=task_run.max_attempts,
                    scheduled_at=task_run.scheduled_at,
                    started_at=task_run.started_at,
                    finished_at=task_run.finished_at,
                    last_error=task_run.last_error,
                    output=task_run.output,
                )
                for task_run in run.task_runs
            ),
            key=lambda task: task.key,
        ),
    )


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    body: RunCreate,
    session: SessionDep,
    response: Response,
    idempotency_key: IdempotencyKeyHeader = None,
) -> RunResponse:
    """Start a run of a workflow.

    Supports `Idempotency-Key`. A client whose POST times out cannot know
    whether the run started; it retries with the same key and gets the original
    run back (200 rather than 201) instead of starting a second one.

    Reusing a key with a *different* body is a client bug and returns 409 --
    silently returning the first run would hide it until someone noticed the
    second request never ran.
    """
    if idempotency_key is not None:
        fingerprint = request_fingerprint(body.model_dump())
        existing = session.get(IdempotencyKey, idempotency_key)
        if existing is not None:
            if existing.request_hash != fingerprint:
                raise ProblemDetail(
                    status.HTTP_409_CONFLICT,
                    "Idempotency key reused",
                    "this Idempotency-Key was already used with a different request body",
                    "idempotency-key-reused",
                )
            run = RunRepository(session).get_with_tasks(existing.run_id)
            if run is not None:
                response.status_code = status.HTTP_200_OK
                return _to_response(run)

    workflows = WorkflowRepository(session)
    workflow = (
        workflows.get_version(body.workflow_name, body.workflow_version)
        if body.workflow_version is not None
        else workflows.get_latest(body.workflow_name)
    )
    if workflow is None:
        raise NotFoundError(f"workflow {body.workflow_name!r} not found")

    runs = RunRepository(session)
    run = runs.create_run(workflow, workflows.dependencies(workflow.id), body.payload)

    if idempotency_key is not None:
        session.add(
            IdempotencyKey(
                key=idempotency_key,
                request_hash=request_fingerprint(body.model_dump()),
                run_id=run.id,
            )
        )
    session.flush()

    created = runs.get_with_tasks(run.id)
    assert created is not None
    return _to_response(created)


@router.get("", response_model=RunPage)
def list_runs(
    session: SessionDep,
    settings: SettingsDep,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    state: RunState | None = Query(default=None),
) -> RunPage:
    repo = RunRepository(session)
    page = repo.list_page(limit=min(limit, settings.max_page_size), cursor=cursor, state=state)
    # Re-load with tasks eagerly: rendering the page straight from `page.items`
    # would lazy-load `task_runs` per row, which is the N+1 this avoids.
    ids = [run.id for run in page.items]
    runs = (
        {run.id: run for run in session.scalars(select(WorkflowRun).where(WorkflowRun.id.in_(ids)))}
        if ids
        else {}
    )
    return RunPage(
        items=[_to_response(runs[run_id]) for run_id in ids],
        meta=PageMeta(next_cursor=page.next_cursor, has_more=page.has_more),
    )


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: uuid.UUID, session: SessionDep) -> RunResponse:
    run = RunRepository(session).get_with_tasks(run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    return _to_response(run)


@router.post("/{run_id}/cancel", response_model=RunResponse)
def cancel_run(run_id: uuid.UUID, session: SessionDep) -> RunResponse:
    """Cancel every task in a run that has not yet reached a terminal state.

    POST to a sub-resource rather than DELETE: cancelling is a state
    transition that leaves the run readable afterwards, not a deletion.
    """
    repo = RunRepository(session)
    run = repo.get_with_tasks(run_id)
    if run is None:
        raise NotFoundError(f"run {run_id} not found")
    if RunState(run.state).is_terminal:
        raise ProblemDetail(
            status.HTTP_409_CONFLICT,
            "Run already finished",
            f"run is {run.state} and cannot be cancelled",
            "run-not-cancellable",
        )

    repo.cancel(run_id)
    session.flush()
    session.expire_all()
    cancelled = repo.get_with_tasks(run_id)
    assert cancelled is not None
    return _to_response(cancelled)
