"""Repository foundations.

Every database access in the service goes through a repository. The point is not
ceremony -- it is that the scheduler and the API depend on an interface they can
substitute in tests, and that raw SQL lives in exactly one layer instead of
leaking into request handlers.

Also here: keyset pagination. Offset pagination is the default everyone reaches
for and it is wrong for a feed that grows: `OFFSET 10000` makes Postgres walk
and discard ten thousand rows, and a row inserted between two page fetches
shifts every subsequent page, so the client silently skips an item. Keyset
pagination seeks straight to the cursor via the index and is stable under
concurrent inserts.
"""

from __future__ import annotations

import base64
import binascii
import json
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

from sqlalchemy.orm import Session

from conductor.db.models import Base

ModelT = TypeVar("ModelT", bound=Base)


class RepositoryError(Exception):
    """Base class for repository-level failures the API layer maps to responses."""


class ConflictError(RepositoryError):
    """A uniqueness or optimistic-locking conflict. Maps to HTTP 409."""


class NotFoundError(RepositoryError):
    """The requested entity does not exist. Maps to HTTP 404."""


@dataclass(frozen=True, slots=True)
class Cursor:
    """An opaque position in a `(created_at DESC, id DESC)` ordering.

    `id` breaks ties: two rows can share a timestamp, and without a tiebreaker
    the ordering is not total, so a page boundary landing between them either
    duplicates or drops a row.
    """

    created_at: datetime
    id: str

    def encode(self) -> str:
        """Base64 so clients treat it as opaque and do not build their own.

        A cursor whose structure clients depend on is a schema you can never
        change; making it look like a blob keeps it an implementation detail.
        """
        payload = json.dumps({"t": self.created_at.isoformat(), "i": self.id})
        return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> Cursor:
        try:
            padded = token + "=" * (-len(token) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            return cls(created_at=datetime.fromisoformat(payload["t"]), id=payload["i"])
        except (ValueError, KeyError, TypeError, binascii.Error) as exc:
            # A malformed cursor is a client error, not a 500. Clients do
            # truncate these in URLs.
            raise ValueError(f"malformed cursor: {token!r}") from exc


@dataclass(frozen=True, slots=True)
class Page(Generic[ModelT]):
    """One page of results plus the cursor for the next one."""

    items: list[ModelT]
    next_cursor: str | None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None


class Repository(ABC, Generic[ModelT]):
    """Holds a session; does not own it.

    Transaction boundaries belong to the caller -- typically one per request or
    one per scheduler tick. A repository that commits on its own makes it
    impossible to compose two of its methods atomically, which is exactly what
    creating a run and its tasks requires.
    """

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, entity_id: object) -> ModelT | None:
        return self.session.get(self.model, entity_id)

    def require(self, entity_id: object) -> ModelT:
        """`get`, but raises rather than returning None.

        Saves every call site an identical three-line None check, and keeps the
        404 message consistent across the API.
        """
        entity = self.get(entity_id)
        if entity is None:
            raise NotFoundError(f"{self.model.__tablename__} {entity_id} not found")
        return entity
