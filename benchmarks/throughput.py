"""Measure queue throughput against worker count, and prove SKIP LOCKED matters.

Two questions this answers with numbers rather than assertion:

1. **Does the queue scale with workers?** If claiming were serialised, adding
   workers would leave throughput flat. The shape of the curve is the result.
2. **How much of that is `SKIP LOCKED`?** The same workload is run against a
   claim query using plain `FOR UPDATE`, where concurrent claimers block behind
   each other instead of stepping over locked rows.

**What the numbers actually showed.** With `--hold-ms 0` -- Conductor's real
design, where the claim transaction commits before any work begins -- the two
strategies perform the same. Locks are held for well under a millisecond, so
blocking on one costs nothing measurable. The assumption that `SKIP LOCKED` is
faster *per se* is simply wrong at this scale, and the benchmark is kept honest
rather than tuned until it agreed.

`SKIP LOCKED` earns its keep in the design it makes *possible to get wrong
safely*: run with `--hold-ms 20` to model a worker that executes the task inside
the claiming transaction, and plain `FOR UPDATE` collapses to single-worker
throughput while `SKIP LOCKED` keeps scaling. The lesson worth keeping is that
the short claim transaction is doing most of the work, and `SKIP LOCKED` is what
stops lock contention from mattering when a transaction is not short.

Run against a *local* database. Numbers from a shared or containerised Postgres
measure that Postgres, not this design.

    python benchmarks/throughput.py --tasks 2000 --workers 1,2,4,8,16
    python benchmarks/throughput.py --tasks 400 --workers 1,2,4,8 --hold-ms 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from conductor.config import Settings
from conductor.core.state import TaskState
from conductor.db.models import Base, Task, TaskRun, Workflow, WorkflowRun
from conductor.db.session import get_engine, get_sessionmaker

# Plain FOR UPDATE: identical to the production claim except for SKIP LOCKED.
# Concurrent claimers queue behind whoever holds the lock on the head row.
BLOCKING_CLAIM = text(
    """
    UPDATE task_runs SET state = 'RUNNING', worker_id = :worker,
        lease_expires_at = now() + interval '60 seconds',
        attempt = attempt + 1, version = version + 1
    WHERE id IN (
        SELECT id FROM task_runs
        WHERE state = 'READY' AND scheduled_at <= now()
        ORDER BY priority DESC, scheduled_at
        LIMIT :batch
        FOR UPDATE
    )
    RETURNING id
    """
)

SKIP_LOCKED_CLAIM = text(str(BLOCKING_CLAIM).replace("FOR UPDATE", "FOR UPDATE SKIP LOCKED"))


@dataclass
class Result:
    strategy: str
    hold_ms: float
    workers: int
    tasks: int
    seconds: float
    throughput_per_sec: float
    p50_claim_ms: float
    p95_claim_ms: float
    p99_claim_ms: float


def seed(session: Session, task_count: int) -> None:
    """Queue `task_count` independent, immediately-claimable tasks.

    Modelled as one run per task rather than one run of many tasks, because
    `uq_task_runs_run_task` (correctly) forbids the same task appearing twice in
    one run -- and because N independent single-task runs is what a queue under
    load actually looks like.

    Ids are generated client-side, which is what makes a single bulk insert
    possible: the run id each task row references is known before either is
    written, so there is no round trip per row.
    """
    session.execute(text("TRUNCATE workflows, workflow_runs, task_runs, tasks CASCADE"))

    workflow = Workflow(name=f"bench-{uuid.uuid4().hex[:8]}", version=1)
    session.add(workflow)
    session.flush()

    task = Task(workflow_id=workflow.id, key="bench", executor="noop")
    session.add(task)
    session.flush()

    now = datetime.now(UTC)
    run_ids = [uuid.uuid4() for _ in range(task_count)]
    session.bulk_save_objects(
        [WorkflowRun(id=run_id, workflow_id=workflow.id, state="RUNNING") for run_id in run_ids]
    )
    session.bulk_save_objects(
        [
            TaskRun(
                id=uuid.uuid4(),
                run_id=run_id,
                task_id=task.id,
                state=TaskState.READY,
                pending_deps=0,
                attempt=0,
                max_attempts=1,
                priority=0,
                scheduled_at=now,
            )
            for run_id in run_ids
        ]
    )
    session.commit()


def drain(
    factory: sessionmaker[Session],
    statement: text,  # type: ignore[valid-type]
    worker_count: int,
    batch_size: int,
    deadline: float,
    hold_ms: float = 0.0,
) -> tuple[int, list[float]]:
    """Run `worker_count` threads claiming until the queue empties or time runs out.

    `hold_ms` keeps the claim transaction open for that long before committing,
    modelling the naive design in which a worker executes the task *inside* the
    claiming transaction. That is the regime where `SKIP LOCKED` earns its
    keep -- see the note in the module docstring.
    """
    claimed = 0
    latencies: list[float] = []
    lock = threading.Lock()
    barrier = threading.Barrier(worker_count)

    def work(index: int) -> None:
        nonlocal claimed
        local: list[float] = []
        local_count = 0
        barrier.wait()  # start together, to maximise contention
        with factory() as session:
            while time.monotonic() < deadline:
                started = time.perf_counter()
                rows = session.execute(
                    statement, {"worker": f"bench-{index}", "batch": batch_size}
                ).all()
                if hold_ms:
                    time.sleep(hold_ms / 1000.0)  # lock held across "the work"
                session.commit()
                local.append((time.perf_counter() - started) * 1000)
                if not rows:
                    break
                local_count += len(rows)
        with lock:
            claimed += local_count
            latencies.extend(local)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return claimed, latencies


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def measure(
    factory: sessionmaker[Session],
    strategy: str,
    statement: text,  # type: ignore[valid-type]
    worker_count: int,
    task_count: int,
    batch_size: int,
    timeout: float,
    hold_ms: float = 0.0,
) -> Result:
    with factory() as session:
        seed(session, task_count)

    started = time.monotonic()
    claimed, latencies = drain(
        factory, statement, worker_count, batch_size, started + timeout, hold_ms
    )
    elapsed = time.monotonic() - started

    return Result(
        strategy=strategy,
        hold_ms=hold_ms,
        workers=worker_count,
        tasks=claimed,
        seconds=round(elapsed, 3),
        throughput_per_sec=round(claimed / elapsed, 1) if elapsed else 0.0,
        p50_claim_ms=round(statistics.median(latencies), 2) if latencies else 0.0,
        p95_claim_ms=round(percentile(latencies, 0.95), 2),
        p99_claim_ms=round(percentile(latencies, 0.99), 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--workers", default="1,2,4,8,16")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--hold-ms",
        type=float,
        default=0.0,
        help="Hold the claim transaction open this long, modelling a worker that "
        "executes inside the claiming transaction.",
    )
    parser.add_argument("--database-url", default="postgresql+psycopg://localhost/conductor_bench")
    args = parser.parse_args()

    settings = Settings(
        database_url=args.database_url,  # type: ignore[arg-type]
        pool_size=40,
        pool_max_overflow=40,
    )
    Base.metadata.create_all(get_engine(settings))
    factory = get_sessionmaker(settings)
    worker_counts = [int(value) for value in args.workers.split(",")]

    # Warm up and discard: the first pass pays for cold shared_buffers, an
    # empty plan cache, and connection establishment. Attributing that cost to
    # whichever strategy happens to run first is how benchmarks lie.
    measure(factory, "warmup", SKIP_LOCKED_CLAIM, 4, args.tasks, args.batch, args.timeout)

    results: list[Result] = []
    for worker_count in worker_counts:
        # Strategies are interleaved per worker count, not run in two blocks, so
        # that any drift over the run (thermal, background load) hits both
        # equally instead of biasing whichever ran second.
        for strategy, statement in (
            ("skip_locked", SKIP_LOCKED_CLAIM),
            ("for_update", BLOCKING_CLAIM),
        ):
            result = measure(
                factory,
                strategy,
                statement,
                worker_count,
                args.tasks,
                args.batch,
                args.timeout,
                args.hold_ms,
            )
            results.append(result)
            print(
                f"{strategy:<12} workers={result.workers:<3} "
                f"{result.throughput_per_sec:>9.1f} tasks/s  "
                f"p50={result.p50_claim_ms:>6.2f}ms  p95={result.p95_claim_ms:>7.2f}ms",
                flush=True,
            )

    output = Path("benchmarks/results") / f"{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(result) for result in results], indent=2))
    print(f"\nwrote {output}")

    baseline = {r.workers: r for r in results if r.strategy == "for_update"}
    print(f"\nspeedup from SKIP LOCKED (hold_ms={args.hold_ms}):")
    for result in results:
        if result.strategy != "skip_locked":
            continue
        reference = baseline.get(result.workers)
        if reference and reference.throughput_per_sec:
            ratio = result.throughput_per_sec / reference.throughput_per_sec
            print(f"  {result.workers:>3} workers: {ratio:.2f}x")


if __name__ == "__main__":
    main()
