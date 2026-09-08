"""Request and response models.

These are deliberately *not* the ORM models. Coupling the wire format to the
schema means every column rename is a breaking API change, and it makes it far
too easy to leak an internal field -- `version`, say, or `worker_id` -- to a
client that then starts depending on it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TaskKey = Annotated[str, Field(min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9_.-]+$")]


class RetryPolicySpec(BaseModel):
    """Retry configuration, validated before it can reach a worker."""

    type: Literal["none", "fixed", "exponential"] = "exponential"
    max_attempts: int = Field(default=5, ge=1, le=100)
    delay: float | None = Field(default=None, ge=0, le=86_400)
    base: float | None = Field(default=None, gt=0, le=3600)
    factor: float | None = Field(default=None, ge=1.0, le=10.0)
    max_delay: float | None = Field(default=None, gt=0, le=86_400)
    jitter: bool | None = None

    def to_spec(self) -> dict[str, object]:
        """Drop unset options so each policy receives only fields it accepts."""
        return {key: value for key, value in self.model_dump().items() if value is not None}


class TaskDefinition(BaseModel):
    key: TaskKey
    executor: str = Field(min_length=1, max_length=64)
    params: dict[str, object] = Field(default_factory=dict)
    depends_on: list[TaskKey] = Field(default_factory=list)
    retry_policy: RetryPolicySpec = Field(default_factory=RetryPolicySpec)
    priority: int = Field(default=0, ge=-100, le=100)
    timeout_seconds: float = Field(default=300.0, gt=0, le=86_400)

    @field_validator("depends_on")
    @classmethod
    def _no_duplicate_dependencies(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("depends_on contains duplicates")
        return value


class WorkflowCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9_.-]+$")
    description: str | None = Field(default=None, max_length=2000)
    # An upper bound on graph size, because this arrives from the network:
    # without it, a single request can allocate an unbounded number of rows.
    tasks: list[TaskDefinition] = Field(min_length=1, max_length=500)


class TaskSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    executor: str
    depends_on: list[str] = Field(default_factory=list)
    priority: int
    timeout_seconds: float


class WorkflowResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    version: int
    description: str | None
    created_at: datetime
    tasks: list[TaskSummary] = Field(default_factory=list)


class RunCreate(BaseModel):
    workflow_name: str = Field(min_length=1, max_length=255)
    # Omitted means "the latest version". Pinning is available for callers that
    # need a run to be reproducible against a definition they have tested.
    workflow_version: int | None = Field(default=None, ge=1)
    payload: dict[str, object] = Field(default_factory=dict)


class TaskRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    key: str
    state: str
    attempt: int
    max_attempts: int
    scheduled_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    last_error: str | None
    output: dict[str, object] | None


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workflow_id: uuid.UUID
    state: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    tasks: list[TaskRunResponse] = Field(default_factory=list)


class PageMeta(BaseModel):
    """Pagination envelope.

    `next_cursor` rather than a page number, and no total count: counting every
    matching row to render a page is a full scan the client almost never needs.
    """

    next_cursor: str | None = None
    has_more: bool = False


class WorkflowPage(BaseModel):
    items: list[WorkflowResponse]
    meta: PageMeta


class RunPage(BaseModel):
    items: list[RunResponse]
    meta: PageMeta


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: Literal["up", "down"]
