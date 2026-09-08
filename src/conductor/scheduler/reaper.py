"""The lease reaper.

A single, small responsibility: find tasks whose worker stopped renewing and
return them to the ready queue. It is what turns "a worker was killed" from data
loss into a delay bounded by the lease duration.

Run as its own process, or as a thread inside one. It is safe to run several:
the reaping statement is a conditional `UPDATE`, so two reapers racing on the
same expired task simply produce one update and one no-op.
"""

from __future__ import annotations

import threading
import time

import structlog

from conductor.config import Settings, get_settings
from conductor.db.repositories.runs import TaskQueueRepository
from conductor.db.session import get_sessionmaker

logger = structlog.get_logger(__name__)


class Reaper:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.session_factory = get_sessionmaker(self.settings)
        self._stop = threading.Event()

    def reap_once(self) -> int:
        """One sweep. Returns how many tasks were requeued."""
        with self.session_factory() as session:
            queue = TaskQueueRepository(session, self.settings.lease_seconds)
            requeued = queue.reap_expired_leases()
            session.commit()

        if requeued:
            logger.warning("reaper.requeued", count=requeued)
        return requeued

    def run_forever(self) -> None:  # pragma: no cover - process entry point
        # Sweep several times per lease period: waiting a whole lease between
        # sweeps would double the worst-case recovery time after a crash.
        interval = max(1.0, self.settings.lease_seconds / 3)
        logger.info("reaper.started", interval_seconds=interval)
        while not self._stop.wait(interval):
            try:
                self.reap_once()
            except Exception as exc:
                logger.error("reaper.sweep_failed", error=str(exc))
                time.sleep(interval)

    def stop(self) -> None:
        self._stop.set()
