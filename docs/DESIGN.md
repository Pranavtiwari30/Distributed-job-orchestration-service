# Design decisions

Why Conductor is built the way it is, including the parts that turned out to be
wrong. Each section states the decision, the alternatives, and the reason one
won — because "we used X" is not an engineering argument and "we used X instead
of Y because Z" is.

---

## 1. The problem

*N* worker processes poll one queue of tasks. Two requirements are in tension:

- **No task may run twice.** Two workers claiming the same task means duplicate
  side effects — a payment charged twice, an email sent twice.
- **No task may be lost.** A worker killed mid-execution must not take its task
  with it.

Naive solutions fail one or the other. "Mark it claimed, then run it" loses
tasks when the worker dies after marking. "Run it, then mark it done" runs tasks
twice when the worker dies before marking. The interesting work is in resolving
this, and everything else in the codebase is ordinary CRUD around it.

---

## 2. Why the queue lives in PostgreSQL

**Decision:** the queue is a table. No Redis, no RabbitMQ, no SQS.

**Alternative:** a dedicated broker, which is what most systems reach for.

**Reason:** claiming a task and recording *why* it was claimed must be atomic.
With a broker they are two systems that can disagree — the broker hands out a
message, the database write fails, and now the task is running with no record of
it. Reconciling that is where lost jobs come from. In one database, the claim and
its audit row are one transaction: both happen or neither does.

**What it costs:** Postgres will not match a purpose-built broker at extreme
throughput. Measured ceiling here is ~70k claims/s on one machine (§10), which is
far beyond what this system needs. Buying throughput you do not need with a
correctness problem you do not want is a bad trade.

**Interview framing:** this is a deliberate constraint, not an omission. Say what
it buys and what it costs.

---

## 3. The claim query

The whole design in one statement:

```sql
UPDATE task_runs SET state = 'RUNNING', worker_id = :worker,
    lease_expires_at = now() + interval '60 seconds',
    attempt = attempt + 1, version = version + 1
WHERE id IN (
    SELECT id FROM task_runs
    WHERE state = 'READY' AND scheduled_at <= now()
    ORDER BY priority DESC, scheduled_at
    LIMIT :batch
    FOR UPDATE SKIP LOCKED
)
RETURNING ...;
```

Four things are load-bearing.

**`FOR UPDATE`** locks the candidate rows so no other transaction can claim them.

**`SKIP LOCKED`** makes a concurrent claimer step *over* rows a peer has locked
rather than block behind them. Without it, *N* workers serialise on the head of
the queue and total throughput equals that of one worker. This is measured in
§10 — and the measurement is more interesting than the claim.

**`RETURNING`** hands back the claimed rows in the same round trip. The
alternative — `SELECT` the candidates, then `UPDATE` them — is a race in two
parts: another worker can claim between the two statements.

**`ORDER BY priority DESC, scheduled_at`** matches the partial index
`ix_task_runs_claimable` column for column, so Postgres walks the index and stops
at `LIMIT` instead of sorting the whole ready set.

> **Likely question:** *"What does `SKIP LOCKED` do if you remove it?"*
> The query still returns correct results — no double-claims — but concurrent
> claimers queue behind each other on the same row. Correctness is unaffected;
> throughput collapses. It is a performance mechanism, not a correctness one.

---

## 4. Leases, not locks

A claim writes `lease_expires_at`. The worker renews it on a background thread
while it works. A reaper requeues any task whose lease has expired.

**Why not hold a database lock for the duration?** Because the work happens
*outside* the transaction. Holding a row lock for the length of a task means one
long task blocks the connection, and a crashed worker's lock is released by the
database only when its connection dies — which can be minutes, and is invisible
to the application. A lease is state the application owns and can reason about.

**Why the heartbeat runs on its own thread:** renewing from the main loop would
mean any task outliving one lease period gets reaped mid-execution — the thread
doing the work is the thread that was supposed to say it is alive.

**Choosing the lease duration** is a real tradeoff:
- Too short: healthy-but-slow workers lose their tasks to the reaper.
- Too long: a crash stalls that task for the whole interval.

The config enforces `heartbeat_seconds <= lease_seconds / 3`, so a worker must
renew several times per lease and one missed beat is survivable.

**The failure mode leases do not cover:** a task that *hangs* rather than
crashing. Its worker faithfully keeps renewing, so it is never reaped. That is
what per-task `timeout_seconds` in the executor is for.

---

## 5. Exactly-once: delivery vs. effects

This distinction is the single most valuable thing in the project.

**Exactly-once *delivery* is not purchasable.** A slow worker and a dead worker
are indistinguishable from the outside. When a lease expires you must either
requeue the task (risking a second execution) or not (risking a lost task).
There is no third option. Conductor requeues, so delivery is **at-least-once**.

**Exactly-once *effects* are purchasable, and cost one unique index:**

```sql
UNIQUE (task_run_id, attempt)   -- on task_attempts
```

Two workers racing on the same attempt both try to insert this row. Exactly one
succeeds; the loser's transaction fails and it discards its work. The `worker_id`
predicate on every completion update does the same job at the other end:

```sql
UPDATE task_runs SET state = 'SUCCEEDED' ...
WHERE id = :id AND worker_id = :me AND state = 'RUNNING'
```

If the reaper already reassigned this task, that matches zero rows, the worker
learns it lost the lease, and it throws its result away rather than overwriting
the new owner's.

> **Likely question:** *"You claim exactly-once. Prove it."*
> Do not claim exactly-once delivery — say the above. Being precise here reads as
> significantly more senior than claiming a guarantee that does not exist.

---

## 6. Data structures, and why each

| Structure | Where | Why not the obvious alternative |
|---|---|---|
| Adjacency list + Kahn's topological sort | `core/dag.py` | Validates a submitted DAG in O(V+E). A graph is a DAG **iff** a topological order exists, so validation and ordering are the same operation. |
| In-degree counters (`DagCursor`) | `core/dag.py` | Completing a task decrements its children's counters — O(out-degree). Re-scanning the graph for ready tasks on every completion would be O(V+E) each time. |
| Iterative DFS for cycle reporting | `core/dag.py` | Kahn's tells you a cycle exists but not *where*. Iterative, not recursive, because graph depth arrives over the network — 10k nodes blows the stack. |
| Binary min-heap + key→slot index | `core/heap.py` | Sorted list: O(1) peek, O(n) insert. Unsorted: O(1) insert, O(n) peek. Heap: O(log n) both. The side index adds **O(log n) cancellation**, which `heapq` cannot do at all — needed to withdraw a timer when a task finishes early. |
| LRU cache (dict + doubly linked list) | `core/lru.py` | Workflow definitions are read on every dispatch, written almost never. O(1) get/put/evict. |
| Token bucket | `core/ratelimit.py` | Allows bursts (submitting 50 tasks at once is normal usage) while bounding sustained rate. A fixed window lets 2× the limit through either side of a boundary. Lazily refilled: no timer, O(1). |

> **Likely question:** *"Why write a heap instead of using `heapq`?"*
> The honest answer: cancellation. `heapq` has no way to remove an arbitrary
> element in less than O(n). If the answer were only "to show I can," say so —
> but here there is a real functional reason.

---

## 7. Schema decisions

**Dependency edges are rows, not JSON.** A `depends_on: ["a","b"]` array is
easier to write and impossible to query. "What breaks if this task fails" is a
recursive CTE over `task_dependencies` and nothing at all over a blob.

**Partial indexes on the hot paths:**
```sql
CREATE INDEX ix_task_runs_claimable ON task_runs (priority DESC, scheduled_at)
    WHERE state = 'READY';
```
READY rows are a tiny fraction of the table. A partial index stays proportional
to the *backlog*, not to all history — so the queue does not slow down as
completed tasks accumulate.

**A check constraint enforces the core invariant:**
```sql
CHECK ((state = 'RUNNING') = (lease_expires_at IS NOT NULL AND worker_id IS NOT NULL))
```
A running task holds a lease and a worker; nothing else may. The entire
exactly-once story rests on this, so it is a constraint the database enforces,
not a convention the code hopes to follow.

**Attempts are append-only.** A retry inserts a new row rather than mutating the
old one, so history survives and the unique constraint can do its job.

**One deliberate denormalisation.** `workflow_runs.state` is derivable from its
task rows. It is stored anyway because the dashboard lists runs by state, and
deriving it per row means an aggregate over `task_runs` for every entry on the
page. It is only ever written by `derive_run_state()`, never by hand — a
denormalised column with two writers is a column that drifts.

**Client-side UUID primary keys.** Lets the API build a whole run graph in memory
— parents and children cross-referenced — and insert it in one statement, with no
round trip per row to learn a serial id.

---

## 8. API decisions

**Keyset pagination, not offset.** `OFFSET 10000` makes Postgres walk and discard
10,000 rows. Worse, a row inserted between two page fetches shifts every
subsequent page, so the client silently *skips* an item. Keyset seeks straight to
the cursor via the index and is stable under concurrent inserts.

**Idempotency keys on `POST /runs`.** A client whose request times out cannot
know whether the run was created. It retries with the same key and gets the
original run back (200, not 201). Reusing a key with a *different* body returns
409 — silently returning the first run would hide a real client bug.

**RFC 9457 `problem+json` for every error.** One error shape means clients write
one error handler. A rejected cyclic workflow returns the cycle itself as an
extension member, so the caller can point at the edge to remove rather than
parsing prose.

**Liveness and readiness are separate endpoints.** `/healthz` must not touch the
database: a liveness probe that fails during a brief database blip restarts every
healthy pod at once, turning a recoverable incident into an outage. `/readyz`
does check, because a process that cannot reach Postgres can serve nothing.

---

## 9. What we got wrong

Four real bugs, each found by a different method. This section is worth more in
an interview than any feature — "how do you find bugs" is a question about
process, and process is what distinguishes engineers.

**Keyset pagination silently dropped its tiebreaker — found by `mypy --strict`.**
The cursor predicate was written `(created_at, id) < (t, i)`. That is a *Python*
tuple comparison, not SQL: Python evaluates `created_at == t` first, SQLAlchemy's
`__bool__` returns False for that expression, and the whole thing collapses to
`created_at < t`. The `id` tiebreaker vanished. Every test passed because their
timestamps happened to differ. Fixed with `tuple_()`, which emits a real SQL
row-value comparison. Pinned by a test that forces every row to share a timestamp.

**Crashed attempts were recorded as successes — found by `SIGKILL`ing a worker.**
Closing an attempt matched "the open attempt row for this task" rather than a
specific attempt number. A killed worker leaves its row open forever, so when a
later attempt succeeded it closed *both* rows and stamped the crashed one
`SUCCEEDED` with a duration spanning the whole outage. The audit trail was
laundering crashes into successes. Found only by killing a real worker — every
test either had one attempt or failed explicitly rather than crashing.

**The unit coverage gate was unsatisfiable — found by CI.** The unit job ran
`pytest tests/unit --cov`, measuring the whole package against an 85% gate, but
that job only exercises the pure modules. It reported 42.89% and failed by
construction. Invisible locally because the full suite was always run together.

**`docker compose --wait` failed on a healthy stack — found by CI.** `--wait`
waits for services to be *running or healthy*; the one-shot migration container
exits 0, which is neither.

The pattern worth naming: **three of the four were found by tools or environments
that behave differently from the developer's laptop.** That is the argument for
strict typing and for CI that runs jobs in isolation.

---

## 10. The benchmark, including the negative result

**The assumption:** `SKIP LOCKED` makes claiming dramatically faster.

**The measurement, with the real claim path** (transaction commits before any
work begins):

| Workers | `SKIP LOCKED` | plain `FOR UPDATE` |
|---:|---:|---:|
| 1 | 31,723/s | 29,789/s |
| 4 | **70,955/s** | 49,893/s |
| 8 | 44,630/s | 46,962/s |

**They are within noise of each other, and the assumption was wrong.** With a
sub-millisecond claim transaction there is almost nothing to block *on*. The
benchmark was kept rather than tuned until it agreed.

**Where it actually matters** — 20ms of work held *inside* the claiming
transaction, modelling the naive design:

| Workers | `SKIP LOCKED` | plain `FOR UPDATE` | Speedup |
|---:|---:|---:|---:|
| 1 | 394/s | 387/s | 1.02× |
| 4 | 1,452/s | 371/s | 3.91× |
| 8 | **2,434/s** | 347/s | **7.01×** |

`FOR UPDATE` is *perfectly flat* — 387/s on one worker, 347/s on eight. That flat
line is the signature of full serialisation: every additional worker queues
behind the first and contributes nothing. Its p95 claim latency degrades from
27ms to 1,102ms.

**The real conclusion is two-part.** The short claim transaction is doing most of
the work. `SKIP LOCKED` is what stops lock contention from mattering when a
transaction is *not* short — which, in a system where task duration is
user-supplied, is a guarantee worth having.

> **Likely question:** *"Did anything surprise you?"*
> This is the answer. Reporting a measurement that contradicted your own design
> assumption is a strong signal; claiming a number you never measured is the
> opposite, and is trivially exposed by one follow-up question.

---

## 11. Deliberately not done

- **A message broker** — §2.
- **Exactly-once delivery** — §5. Not achievable; the honest guarantee is stated
  instead.
- **Cross-language workers** — would require designing a wire protocol.
- **Priority ageing** — a low-priority task behind a steady stream of urgent ones
  currently starves. Known, and on the roadmap. Being able to name a limitation
  of your own system is worth more than pretending it has none.
