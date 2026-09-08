"""Workflow definition endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, status

from conductor.api.deps import RateLimited, SessionDep, SettingsDep
from conductor.api.schemas import (
    PageMeta,
    TaskSummary,
    WorkflowCreate,
    WorkflowPage,
    WorkflowResponse,
)
from conductor.db.repositories.base import NotFoundError
from conductor.db.repositories.workflows import TaskSpec, WorkflowRepository

router = APIRouter(prefix="/workflows", tags=["workflows"], dependencies=[RateLimited])


def _to_response(repo: WorkflowRepository, workflow: object) -> WorkflowResponse:
    dependencies = repo.dependencies(workflow.id)  # type: ignore[attr-defined]
    return WorkflowResponse(
        id=workflow.id,  # type: ignore[attr-defined]
        name=workflow.name,  # type: ignore[attr-defined]
        version=workflow.version,  # type: ignore[attr-defined]
        description=workflow.description,  # type: ignore[attr-defined]
        created_at=workflow.created_at,  # type: ignore[attr-defined]
        tasks=[
            TaskSummary(
                key=task.key,
                executor=task.executor,
                depends_on=sorted(dependencies.get(task.key, [])),
                priority=task.priority,
                timeout_seconds=task.timeout_seconds,
            )
            for task in sorted(workflow.tasks, key=lambda t: t.key)  # type: ignore[attr-defined]
        ],
    )


@router.post("", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
def create_workflow(body: WorkflowCreate, session: SessionDep) -> WorkflowResponse:
    """Publish a new version of a workflow.

    Always creates a *new version* rather than updating in place, so this is a
    POST to the collection and not a PUT to an item: submitting the same body
    twice legitimately yields two versions.
    """
    repo = WorkflowRepository(session)
    workflow = repo.create(
        name=body.name,
        description=body.description,
        tasks=[
            TaskSpec(
                key=task.key,
                executor=task.executor,
                params=task.params,
                depends_on=tuple(task.depends_on),
                retry_policy=task.retry_policy.to_spec(),
                priority=task.priority,
                timeout_seconds=task.timeout_seconds,
            )
            for task in body.tasks
        ],
    )
    session.flush()
    return _to_response(repo, workflow)


@router.get("", response_model=WorkflowPage)
def list_workflows(
    session: SessionDep,
    settings: SettingsDep,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> WorkflowPage:
    repo = WorkflowRepository(session)
    page = repo.list_page(limit=min(limit, settings.max_page_size), cursor=cursor)
    return WorkflowPage(
        items=[_to_response(repo, workflow) for workflow in page.items],
        meta=PageMeta(next_cursor=page.next_cursor, has_more=page.has_more),
    )


@router.get("/{workflow_id}", response_model=WorkflowResponse)
def get_workflow(workflow_id: uuid.UUID, session: SessionDep) -> WorkflowResponse:
    repo = WorkflowRepository(session)
    workflow = repo.get(workflow_id)
    if workflow is None:
        raise NotFoundError(f"workflow {workflow_id} not found")
    return _to_response(repo, workflow)
