"""Engine and session management.

One engine per process, created lazily. Sessions are short-lived and always
scoped to a `with` block: a session held open across a request is a transaction
held open across a request, and that is how a connection pool gets exhausted by
a single slow client.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from conductor.config import Settings, get_settings

# Engines are cached by DSN rather than by `Settings` object: settings are an
# unhashable pydantic model, and the DSN is the only field that decides whether
# two callers may share a pool. Tests point at a second database and get a
# second engine, without the two evicting each other from a size-1 cache.
_ENGINES: dict[str, Engine] = {}
_SESSIONMAKERS: dict[str, sessionmaker[Session]] = {}


def get_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    cached = _ENGINES.get(settings.sync_database_url)
    if cached is not None:
        return cached

    engine = create_engine(
        settings.sync_database_url,
        pool_size=settings.pool_size,
        max_overflow=settings.pool_max_overflow,
        # Verify a pooled connection before handing it out. Without this, the
        # first request after a database restart or an idle-timeout kill fails
        # with a stale socket instead of transparently reconnecting.
        pool_pre_ping=True,
        # Recycle below any typical proxy/firewall idle timeout.
        pool_recycle=1800,
        future=True,
    )

    timeout_ms = settings.statement_timeout_ms

    @event.listens_for(engine, "connect")
    def _set_statement_timeout(dbapi_connection: object, _record: object) -> None:
        """Bound every query at the connection level.

        A runaway query on the claim path would hold locks and stall every
        worker; failing it after `statement_timeout_ms` turns an outage into a
        retried request.
        """
        with dbapi_connection.cursor() as cursor:  # type: ignore[attr-defined]
            cursor.execute(f"SET statement_timeout = {timeout_ms}")

    _ENGINES[settings.sync_database_url] = engine
    return engine


def get_sessionmaker(settings: Settings | None = None) -> sessionmaker[Session]:
    settings = settings or get_settings()
    cached = _SESSIONMAKERS.get(settings.sync_database_url)
    if cached is not None:
        return cached

    factory = sessionmaker(
        bind=get_engine(settings),
        # The worker reads a task's attributes *after* committing its claim;
        # expiring on commit would issue a fresh SELECT for each one.
        expire_on_commit=False,
        future=True,
    )
    _SESSIONMAKERS[settings.sync_database_url] = factory
    return factory


def reset_connections() -> None:
    """Dispose every pooled connection. Used by test fixtures between modules."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()
    _SESSIONMAKERS.clear()


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """A transactional scope: commit on success, roll back on any exception.

    Every write path in the service goes through this, so there is exactly one
    place where a transaction can be forgotten -- and it is this one.
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
