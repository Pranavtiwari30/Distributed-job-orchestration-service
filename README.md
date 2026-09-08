# Conductor

A distributed job orchestration service. Submit a DAG of tasks over a REST API;
a scheduler resolves dependencies and a pool of stateless workers executes them
with leases, retries, and **exactly-once** semantics — including when a worker
is killed mid-task.

```
                 POST /api/v1/runs
                        │
                  ┌─────▼─────┐        ┌──────────────┐
                  │  FastAPI  │        │  Scheduler   │
                  │  gateway  │        │  (delay heap │
                  └─────┬─────┘        │   + DAG)     │
                        │              └──────┬───────┘
                        │                     │
                  ┌─────▼─────────────────────▼─────┐
                  │          PostgreSQL             │
                  │  the queue, the lease table,    │
                  │  and the single source of truth │
                  └─────┬─────────────────────┬─────┘
                        │  FOR UPDATE         │
                        │  SKIP LOCKED        │
                  ┌─────▼─────┐         ┌─────▼─────┐
                  │  Worker   │   ...   │  Worker   │
                  └───────────┘         └───────────┘
```

## Why the interesting part is interesting

The central problem is this: *N* workers poll one queue. Two of them must never
run the same task, and a task must not be lost when the worker holding it is
`SIGKILL`ed. Conductor solves it without a broker, using three mechanisms that
compose:

1. **Atomic claiming.** A worker takes a task with a single statement:

   ```sql
   UPDATE task_runs SET state = 'RUNNING', ...
   WHERE id = (
       SELECT id FROM task_runs
       WHERE state = 'READY' AND scheduled_at <= now()
       ORDER BY priority DESC, scheduled_at
       FOR UPDATE SKIP LOCKED          -- ← the whole trick
       LIMIT 1
   )
   RETURNING *;
   ```

   `SKIP LOCKED` makes concurrent claimers step over rows their peers have
   locked rather than queue behind them. Without it, twenty workers serialise on
   one row and throughput collapses to that of a single worker.

2. **Leases, not locks.** A claim writes a `lease_expires_at`. A worker that
   dies never releases anything — it simply stops renewing, and a reaper
   returns the expired task to the queue. This is what makes crashes survivable,
   and it is also why delivery is *at-least-once*, not exactly-once.

3. **Idempotency keys close the gap.** Because a slow worker and a dead worker
   are indistinguishable from the outside, a task can genuinely execute twice.
   Exactly-once *effects* come from a `UNIQUE` constraint on
   `(task_run_id, attempt)` in the effects table: the second writer loses the
   insert and discards its work.

Points 1–3 are the honest version. "Exactly-once delivery" is not a thing you
can buy; exactly-once *effect* is, and it costs you a unique index.

## Measurements

Apple M5 Pro (15 cores), PostgreSQL 16.15, local socket. Reproduce with
[`benchmarks/throughput.py`](benchmarks/throughput.py); raw JSON lands in
`benchmarks/results/`.

**Claim throughput, as Conductor actually runs** — the claim transaction commits
before any work begins:

| Workers | `SKIP LOCKED` | plain `FOR UPDATE` | p95 claim |
|--------:|--------------:|-------------------:|----------:|
| 1 | 31,723/s | 29,789/s | 0.36 ms |
| 2 | 52,352/s | 49,125/s | 0.44 ms |
| 4 | **70,955/s** | 49,893/s | 0.71 ms |
| 8 | 44,630/s | 46,962/s | 2.71 ms |

I expected `SKIP LOCKED` to dominate this table. **It does not, and the
benchmark was kept rather than tuned until it agreed.** With a sub-millisecond
claim transaction there is almost nothing to block *on*, so the two strategies
are within noise of each other. Throughput also peaks at four workers and
declines after: past that point the bottleneck is connection and scheduler
contention on one machine, not the queue.

**Where `SKIP LOCKED` actually earns its keep** — the same workload with 20 ms
of work held *inside* the claiming transaction (`--hold-ms 20`), modelling the
naive design where a worker executes before it commits:

| Workers | `SKIP LOCKED` | plain `FOR UPDATE` | Speedup |
|--------:|--------------:|-------------------:|--------:|
| 1 | 394/s | 387/s | 1.02x |
| 2 | 743/s | 394/s | 1.89x |
| 4 | 1,452/s | 371/s | 3.91x |
| 8 | **2,434/s** | 347/s | **7.01x** |

`SKIP LOCKED` scales 6.2x from one worker to eight. Plain `FOR UPDATE` is
*perfectly flat* — 387/s on one worker, 347/s on eight — which is the textbook
signature of full serialisation: every additional worker queues behind the first
and contributes nothing. Its p95 claim latency degrades from 27 ms to 1,102 ms.

So the honest conclusion is two-part, and the second half is the one that
survived contact with data: the short claim transaction is doing most of the
work, and `SKIP LOCKED` is what stops lock contention from mattering when a
transaction is *not* short — which, in a system where task duration is
user-supplied, is a guarantee worth having.

**Crash recovery.** A worker `SIGKILL`ed mid-task has its work requeued by the
reaper within one lease period and completed by another worker, with the failed
attempt and the successful retry both recorded. Covered by
[`test_worker.py`](tests/integration/test_worker.py) and
[`test_queue.py`](tests/integration/test_queue.py); 20 threads racing 200 tasks
produce zero double-claims.

## What is in here

| Area | Where | What it is |
|---|---|---|
| Data structures | [`src/conductor/core/`](src/conductor/core/) | Hand-written DAG + Kahn's topological sort, binary min-heap delay queue with O(log n) cancellation, LRU cache, token bucket |
| OOP design | [`core/retry.py`](src/conductor/core/retry.py), [`core/state.py`](src/conductor/core/state.py), [`db/repositories/`](src/conductor/db/repositories/) | Strategy (retry policies), State (lifecycle table), Repository (persistence), Factory + registry (executors) |
| SQL | [`migrations/`](migrations/), [`db/`](src/conductor/db/) | 3NF schema, `FOR UPDATE SKIP LOCKED`, partial + composite indexes, optimistic locking, window-function analytics |
| REST | [`api/`](src/conductor/api/) | Versioned routes, cursor pagination, idempotency keys, RFC 9457 errors, OpenAPI |
| Testing | [`tests/`](tests/) | Property-based unit tests (Hypothesis), integration tests against real Postgres, a concurrency test asserting exactly-once |
| CI/CD | [`.github/workflows/`](.github/workflows/) | Lint → typecheck → unit → integration → Docker build → deploy on tag |

## Quick start

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"

# unit tests need nothing but Python
pytest tests/unit -q

# integration tests need Postgres
brew services start postgresql@16      # or: docker compose up -d db
createdb conductor_test
alembic upgrade head
pytest -q
```

## Testing

164 tests, 93% line coverage, `mypy --strict` clean.

- **Unit** (`tests/unit/`) — no database, no clock, no network. Includes
  property-based tests: random DAGs must topologically sort with every parent
  before its child; the LRU cache is run against an `OrderedDict` reference
  model under arbitrary operation sequences; the heap invariant is re-checked
  after every interleaved push/pop/cancel.
- **Integration** (`tests/integration/`) — real PostgreSQL, never SQLite, because
  `SKIP LOCKED`, partial indexes and row-level locking do not exist there and a
  test that passed against SQLite would be testing a different program.

Tests that earned their place by catching real bugs:

- `test_pagination_is_stable_when_rows_share_a_timestamp` — the cursor predicate
  was written `(created_at, id) < (t, i)`, which Python evaluates as a *tuple*
  comparison and silently collapses to `created_at < t`, dropping the tiebreaker.
  Every existing test passed because their timestamps happened to differ.
- `test_a_zombie_worker_cannot_overwrite_the_new_owner_s_result` — full
  split-brain: worker A is reaped, B takes over, A wakes and reports success.
  A's write must lose.
- `test_skip_locked_actually_lets_workers_overtake_each_other` — holds a lock
  open in one transaction and asserts a second can still claim. Under plain
  `FOR UPDATE` this test hangs, which is the point.

## Status

Under active development. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for what is
built and what is next.
