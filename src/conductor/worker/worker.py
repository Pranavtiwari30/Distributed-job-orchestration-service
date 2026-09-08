"""The worker process: claim, execute, report, repeat.

The loop is deliberately boring. All of the difficulty lives in the SQL it
calls, which is where it belongs -- a worker is a stateless process that can be
killed at any instant without the system losing or duplicating work.

Two details are load-bearing:

* **The heartbeat runs on its own thread.** Renewing the lease from the main
  loop would mean a task that takes longer than one lease period gets reaped
  mid-execution, purely because the thread doing the work is the thread that was
  supposed to say it is alive.

* **Shutdown is graceful by default.** On `SIGTERM` the worker stops claiming
  but finishes what it holds. Dropping the task instead would be *correct* --
  the reaper recovers it -- but it wastes a lease period on every deploy, and a
  rolling restart of twenty workers is then twenty stalled tasks.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from types import FrameType

import structlog
from sqlalchemy.orm import Session, sessionmaker

from conductor.config import Settings, get_settings
from conductor.db.models import Worker as WorkerRow
from conductor.db.repositories.runs import ClaimedTask, RunRepository, TaskQueueRepository
from conductor.db.session import get_sessionmaker
from conductor.worker.executors import ExecutorError, build_executor

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class WorkerStats:
    """Counters for the run summary and for /metrics."""

    claimed: int = 0
    succeeded: int = 0
    failed: int = 0
    lost_leases: int = 0
    idle_polls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def as_dict(self) -> dict[str, float | int]:
        return {
            "claimed": self.claimed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "lost_leases": self.lost_leases,
            "idle_polls": self.idle_polls,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


class Heartbeat:
    """Renews a task's lease on a background thread for as long as it is held.

    Exposes `lost` so the executing thread can find out that the lease was
    stolen. It cannot interrupt the work -- Python offers no safe way to do that
    -- but the result is discarded, which is what actually matters.
    """

    def __init__(
        self,
        queue_factory: sessionmaker[Session],
        task_run_id: uuid.UUID,
        worker_id: str,
        interval: float,
        lease_seconds: float,
    ) -> None:
        self._queue_factory = queue_factory
        self._task_run_id = task_run_id
        self._worker_id = worker_id
        self._interval = interval
        self._lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lost = threading.Event()

    def __enter__(self) -> Heartbeat:
        self._thread = threading.Thread(target=self._run, daemon=True, name="heartbeat")
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)

    def _run(self) -> None:
        # A dedicated session: sharing the worker's would mean two threads
        # inside one SQLAlchemy session, which is not safe.
        while not self._stop.wait(self._interval):
            try:
                with self._queue_factory() as session:
                    queue = TaskQueueRepository(session, self._lease_seconds)
                    renewed = queue.renew_lease(self._task_run_id, self._worker_id)
                    session.commit()
                if not renewed:
                    logger.warning("worker.lease_lost", task_run_id=str(self._task_run_id))
                    self.lost.set()
                    return
            except Exception as exc:  # pragma: no cover - transient db failure
                # Do not give up on one failed renewal: a single dropped packet
                # would otherwise abandon a task that is running perfectly well.
                logger.warning("worker.heartbeat_failed", error=str(exc))


class Worker:
    """One worker process."""

    def __init__(self, settings: Settings | None = None, worker_id: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.session_factory = get_sessionmaker(self.settings)
        self.stats = WorkerStats()
        self._shutdown = threading.Event()

    # ---- lifecycle -------------------------------------------------------

    def install_signal_handlers(self) -> None:  # pragma: no cover - process-level
        def handle(signum: int, _frame: FrameType | None) -> None:
            logger.info("worker.shutdown_requested", signal=signal.Signals(signum).name)
            self._shutdown.set()

        signal.signal(signal.SIGTERM, handle)
        signal.signal(signal.SIGINT, handle)

    def register(self) -> None:
        with self.session_factory() as session:
            session.merge(
                WorkerRow(
                    id=self.worker_id,
                    hostname=socket.gethostname(),
                    pid=os.getpid(),
                )
            )
            session.commit()
        logger.info("worker.registered", worker_id=self.worker_id)

    def run_forever(self) -> WorkerStats:  # pragma: no cover - driven by run_once in tests
        self.register()
        while not self._shutdown.is_set():
            if self.run_once() == 0:
                self.stats.idle_polls += 1
                # Sleep on the event, not on the clock, so shutdown is
                # immediate rather than up to one poll interval late.
                self._shutdown.wait(self.settings.poll_interval_seconds)
        logger.info("worker.stopped", **self.stats.as_dict())
        return self.stats

    # ---- one iteration ---------------------------------------------------

    def run_once(self, limit: int | None = None) -> int:
        """Claim and execute one batch. Returns how many tasks were executed.

        Split out from `run_forever` so tests can drive the worker one batch at
        a time, deterministically, with no sleeping and no threads to join.
        """
        with self.session_factory() as session:
            queue = TaskQueueRepository(session, self.settings.lease_seconds)
            batch = queue.claim_batch(self.worker_id, limit=limit or self.settings.claim_batch_size)
            session.commit()

        self.stats.claimed += len(batch)
        for task in batch:
            self._execute(task)
        return len(batch)

    def _execute(self, task: ClaimedTask) -> None:
        log = logger.bind(task_run_id=str(task.task_run_id), attempt=task.attempt)
        heartbeat = Heartbeat(
            self.session_factory,
            task.task_run_id,
            self.worker_id,
            self.settings.heartbeat_seconds,
            self.settings.lease_seconds,
        )

        with heartbeat:
            started = time.perf_counter()
            try:
                output = build_executor(task.executor).execute(task.params, task.timeout_seconds)
                error: str | None = None
            except ExecutorError as exc:
                output, error = None, str(exc)
            except Exception as exc:  # an executor bug, not a task failure
                output, error = None, f"{type(exc).__name__}: {exc}"
                log.exception("worker.executor_crashed")
            duration_ms = (time.perf_counter() - started) * 1000

        if heartbeat.lost.is_set():
            # Someone else owns this task now. Writing the result would race
            # with them, so it is discarded -- this is the "loser discards"
            # half of exactly-once effects.
            self.stats.lost_leases += 1
            log.warning("worker.result_discarded", reason="lease lost during execution")
            return

        with self.session_factory() as session:
            queue = TaskQueueRepository(session, self.settings.lease_seconds)
            if error is None:
                accepted = queue.record_success(task.task_run_id, self.worker_id, output)
                self.stats.succeeded += int(accepted)
            else:
                accepted = queue.record_failure(task.task_run_id, self.worker_id, error) is not None
                self.stats.failed += int(accepted)

            # Roll the run's cached state forward in the same transaction, so a
            # run can never be observed complete while a task row says otherwise.
            RunRepository(session).refresh_run_state(task.run_id)
            session.commit()

        if not accepted:
            self.stats.lost_leases += 1
            log.warning("worker.result_rejected", reason="task was reassigned or cancelled")
        else:
            log.info(
                "worker.task_finished",
                outcome="success" if error is None else "failure",
                duration_ms=round(duration_ms, 2),
            )


def main() -> None:  # pragma: no cover - process entry point
    worker = Worker()
    worker.install_signal_handlers()
    worker.run_forever()
