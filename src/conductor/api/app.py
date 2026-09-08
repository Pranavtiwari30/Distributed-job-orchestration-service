"""Application factory.

A factory rather than a module-level `app = FastAPI()`: tests construct an app
with overridden settings, and a module-level instance would connect to the
developer's database at import time.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import APIRouter, FastAPI, Request, Response

from conductor import __version__
from conductor.api.errors import register_error_handlers
from conductor.api.routes import health, runs, workflows
from conductor.config import Settings, get_settings
from conductor.core.ratelimit import RateLimiter

logger = structlog.get_logger(__name__)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build process-wide state once, and tear it down on shutdown."""
    settings: Settings = app.state.settings
    app.state.rate_limiter = RateLimiter[str](
        capacity=settings.api_rate_limit,
        refill_rate=settings.api_rate_refill,
    )
    logger.info("api.startup", environment=settings.environment, version=__version__)
    yield
    logger.info("api.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Conductor",
        version=__version__,
        summary="Distributed job orchestration with exactly-once execution.",
        # Versioned in the path rather than in a header: it is visible in logs
        # and in a browser address bar, which is worth more day to day than
        # the purity of header-based negotiation.
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        lifespan=lifespan,
    )
    app.state.settings = settings

    register_error_handlers(app)

    versioned = APIRouter(prefix=API_PREFIX)
    versioned.include_router(workflows.router)
    versioned.include_router(runs.router)
    app.include_router(versioned)
    app.include_router(health.router)  # unversioned: probes are infrastructure

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a request id and log latency for every request.

        The id is echoed in `X-Request-ID` so that a user reporting a failure
        can quote one string that finds every log line for their request.
        """
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response

    return app


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "conductor.api.app:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104 - containerised; the gateway does the binding
        port=8000,
        log_level=settings.log_level.lower(),
    )
