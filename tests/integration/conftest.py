"""Fixtures for tests that need a real PostgreSQL database.

These tests do not use SQLite. The behaviour under test -- `FOR UPDATE SKIP
LOCKED`, partial indexes, row-level locking between concurrent transactions --
does not exist in SQLite, so a test that passed there would be testing a
different program than the one that ships.

The database is created once per session and truncated between tests: dropping
and recreating the schema per test is correct but takes ~200ms a time, which is
the difference between a suite you run on every save and one you avoid.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from conductor.config import Settings
from conductor.db.models import Base
from conductor.db.session import get_engine, get_sessionmaker, reset_connections

TEST_DATABASE_URL = os.environ.get(
    "CONDUCTOR_TEST_DATABASE_URL",
    "postgresql+psycopg://localhost/conductor_test",
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    # A short lease so lease-expiry tests finish in milliseconds rather than
    # in the production default of a minute.
    return Settings(
        database_url=TEST_DATABASE_URL,  # type: ignore[arg-type]
        lease_seconds=2.0,
        heartbeat_seconds=0.5,
    )


@pytest.fixture(scope="session")
def engine(settings: Settings) -> Iterator[Engine]:
    engine = get_engine(settings)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment problem, not a bug
        pytest.skip(f"PostgreSQL unavailable at {TEST_DATABASE_URL}: {exc}")

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    reset_connections()


@pytest.fixture(autouse=True)
def clean_tables(engine: Engine) -> Iterator[None]:
    """Empty every table before each test.

    `TRUNCATE ... CASCADE` in one statement, so foreign keys never dictate a
    deletion order that has to be maintained by hand as tables are added.
    """
    tables = ", ".join(f'"{table.name}"' for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture
def session(settings: Settings, engine: Engine) -> Iterator[Session]:  # noqa: ARG001
    # `engine` is requested purely for ordering: it is what creates the schema.
    factory = get_sessionmaker(settings)
    with factory() as session:
        yield session
        session.rollback()
