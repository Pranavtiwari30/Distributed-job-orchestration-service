"""Liveness and readiness endpoints.

Two endpoints, not one, because Kubernetes asks two different questions:

* `/healthz` -- "is this process alive?" It must not touch the database. A
  liveness probe that fails during a brief database blip restarts every healthy
  pod at once, turning a recoverable incident into an outage.
* `/readyz` -- "should this process receive traffic?" It checks the database,
  because a process that cannot reach Postgres can serve nothing useful.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from conductor import __version__
from conductor.api.deps import SessionDep
from conductor.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def liveness() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, database="up")


@router.get("/readyz", response_model=HealthResponse)
def readiness(session: SessionDep, response: Response) -> HealthResponse:
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", version=__version__, database="down")
    return HealthResponse(status="ok", version=__version__, database="up")
