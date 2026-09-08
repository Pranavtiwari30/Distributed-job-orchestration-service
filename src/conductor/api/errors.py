"""RFC 9457 `application/problem+json` error responses.

Every error the API returns has the same shape, so a client writes one error
handler rather than one per endpoint:

    {
      "type": "https://conductor.dev/problems/workflow-cycle",
      "title": "Workflow contains a cycle",
      "status": 422,
      "detail": "workflow contains a cycle: build -> test -> build",
      "instance": "/api/v1/workflows",
      "cycle": ["build", "test", "build"]
    }

`type` is a stable identifier clients may branch on; `detail` is prose for a
human and may change without notice. Extension members (`cycle` here) carry the
machine-readable specifics, which is what turns a 422 from "you did something
wrong" into something a caller can act on programmatically.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from conductor.core.dag import CycleError
from conductor.db.repositories.base import ConflictError, NotFoundError

PROBLEM_BASE = "https://conductor.dev/problems"
CONTENT_TYPE = "application/problem+json"


class ProblemDetail(Exception):  # noqa: N818 - named for RFC 9457, not for Python
    """An error carrying everything needed to render a problem document."""

    def __init__(
        self,
        status_code: int,
        title: str,
        detail: str,
        problem_type: str = "about:blank",
        **extensions: Any,
    ) -> None:
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.problem_type = (
            problem_type if problem_type.startswith("http") else f"{PROBLEM_BASE}/{problem_type}"
        )
        self.extensions = extensions
        super().__init__(detail)

    def to_response(self, instance: str) -> JSONResponse:
        body: dict[str, Any] = {
            "type": self.problem_type,
            "title": self.title,
            "status": self.status_code,
            "detail": self.detail,
            "instance": instance,
            **self.extensions,
        }
        return JSONResponse(status_code=self.status_code, content=body, media_type=CONTENT_TYPE)


def register_error_handlers(app: FastAPI) -> None:
    """Map every expected failure onto a problem document.

    Domain exceptions are translated here, at the boundary, rather than being
    caught in each route: a repository raising `NotFoundError` should not have
    to know that HTTP exists.
    """

    @app.exception_handler(ProblemDetail)
    async def _problem(request: Request, exc: ProblemDetail) -> JSONResponse:
        return exc.to_response(request.url.path)

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return ProblemDetail(
            status.HTTP_404_NOT_FOUND, "Resource not found", str(exc), "not-found"
        ).to_response(request.url.path)

    @app.exception_handler(ConflictError)
    async def _conflict(request: Request, exc: ConflictError) -> JSONResponse:
        return ProblemDetail(
            status.HTTP_409_CONFLICT, "Conflict", str(exc), "conflict"
        ).to_response(request.url.path)

    @app.exception_handler(CycleError)
    async def _cycle(request: Request, exc: CycleError) -> JSONResponse:
        # The cycle itself is the useful part: a client can highlight exactly
        # which edge to remove instead of re-deriving it from a message string.
        return ProblemDetail(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Workflow contains a cycle",
            str(exc),
            "workflow-cycle",
            cycle=[str(node) for node in exc.cycle],
        ).to_response(request.url.path)

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        return ProblemDetail(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Invalid request",
            str(exc),
            "invalid-request",
        ).to_response(request.url.path)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 body is a bare list; wrapping it keeps the
        # response shape identical to every other error the API emits.
        return ProblemDetail(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "Request failed validation",
            "one or more fields are invalid",
            "validation-failed",
            errors=[
                {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
                for error in exc.errors()
            ],
        ).to_response(request.url.path)
