"""Runtime configuration, read from the environment.

Twelve-factor: every deployment difference is an environment variable, so the
same image runs in CI, locally, and in production with no code path selecting
between them. Defaults are the *local development* values -- never production
ones, so a missing variable in production fails loudly instead of silently
pointing at localhost.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONDUCTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- database ----
    database_url: PostgresDsn = Field(
        default="postgresql+psycopg://localhost/conductor",  # type: ignore[assignment]
        description="SQLAlchemy DSN. Must use the psycopg (v3) driver.",
    )
    pool_size: int = Field(default=5, ge=1, le=100)
    pool_max_overflow: int = Field(default=10, ge=0, le=100)
    statement_timeout_ms: int = Field(
        default=30_000,
        ge=100,
        description="Postgres statement_timeout. A query that outlives this is a bug.",
    )

    # ---- scheduling ----
    lease_seconds: float = Field(
        default=60.0,
        gt=0,
        description=(
            "How long a claim is valid before the reaper may requeue the task. "
            "Too short and healthy-but-slow workers get their work stolen; too "
            "long and a crash stalls that task for the whole interval."
        ),
    )
    heartbeat_seconds: float = Field(default=15.0, gt=0)
    claim_batch_size: int = Field(default=10, ge=1, le=1000)
    poll_interval_seconds: float = Field(default=1.0, gt=0)

    # ---- api ----
    api_rate_limit: int = Field(default=100, ge=1, description="Burst capacity per API key.")
    api_rate_refill: float = Field(default=10.0, gt=0, description="Tokens per second per key.")
    max_page_size: int = Field(default=100, ge=1, le=1000)
    workflow_cache_size: int = Field(default=256, ge=0)

    # ---- misc ----
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    @field_validator("heartbeat_seconds")
    @classmethod
    def _heartbeat_must_beat_the_lease(cls, value: float, info: object) -> float:
        """A worker must renew several times per lease.

        If the heartbeat interval approaches the lease duration, one slow
        network round trip is enough for a perfectly healthy worker to lose its
        task -- and then two workers run it at once.
        """
        data = getattr(info, "data", {})
        lease = data.get("lease_seconds")
        if lease is not None and value > lease / 3:
            raise ValueError(
                f"heartbeat_seconds ({value}) must be at most a third of "
                f"lease_seconds ({lease}) to tolerate a missed beat"
            )
        return value

    @property
    def sync_database_url(self) -> str:
        return str(self.database_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, cached so config is read once at startup.

    FastAPI depends on this rather than importing a module-level singleton, so
    tests can override the dependency without touching the environment.
    """
    return Settings()
