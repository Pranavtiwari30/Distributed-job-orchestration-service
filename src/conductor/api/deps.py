"""FastAPI dependencies: sessions, settings, and rate limiting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, Request, status
from sqlalchemy.orm import Session

from conductor.api.errors import ProblemDetail
from conductor.config import Settings
from conductor.core.ratelimit import RateLimiter
from conductor.db.session import get_sessionmaker


def settings_dependency(request: Request) -> Settings:
    """The settings governing *this* application instance.

    Read from `app.state` rather than the process-global `get_settings()`. The
    factory already stores them there, and resolving them globally instead meant
    `create_app(settings)` governed only the parts that read `app.state` while
    every route handler quietly fell back to the environment -- so an app built
    with an explicit database URL would happily connect to a different one.
    """
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(settings_dependency)]


def db_session(settings: SettingsDep) -> Iterator[Session]:
    """One transaction per request, committed on success.

    A handler that raises rolls the whole request back, so a request that fails
    halfway cannot leave a run with some of its tasks inserted.
    """
    session = get_sessionmaker(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(db_session)]


def api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """Identify the caller for rate-limiting purposes.

    Authentication proper is out of scope for this service -- it belongs at the
    gateway. This exists so the limiter has a stable key; unauthenticated
    callers share the "anonymous" bucket, which is the conservative default.
    """
    return x_api_key or "anonymous"


ApiKeyDep = Annotated[str, Depends(api_key)]


def rate_limit(request: Request, key: ApiKeyDep) -> None:
    """Reject a caller that has exhausted its token bucket.

    The limiter lives on `app.state` so that all workers of a single process
    share one, and so tests can reach in and reset it.
    """
    limiter: RateLimiter[str] = request.app.state.rate_limiter
    decision = limiter.check(key)
    if not decision.allowed:
        raise ProblemDetail(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Rate limit exceeded",
            f"retry in {decision.retry_after:.1f}s",
            "rate-limited",
            retry_after=decision.retry_after,
        )


RateLimited = Depends(rate_limit)


def request_fingerprint(payload: object) -> str:
    """A stable hash of a request body, for idempotency-key comparison.

    `sort_keys` matters: two JSON objects with identical content but different
    key order are the same request, and must not be reported as a conflict.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


IdempotencyKeyHeader = Annotated[str | None, Header(alias="Idempotency-Key", max_length=255)]
