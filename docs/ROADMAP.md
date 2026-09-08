# Roadmap

## Shipped

- **Core scheduling primitives** — DAG with Kahn's topological sort and iterative
  cycle detection, binary min-heap delay queue with O(log n) cancellation, LRU
  cache, token-bucket rate limiter, retry policies, lifecycle state machines.
  All pure, all property-tested.
- **Schema and migrations** — 3NF schema, partial indexes on the claim and
  reaper paths, check constraints enforcing the lease invariant, Alembic
  migrations that round-trip.
- **The queue** — `FOR UPDATE SKIP LOCKED` claiming, leases with heartbeats,
  retry with backoff, dead-lettering, descendant blocking, lease reaping.
- **REST API** — versioned routes, RFC 9457 problem responses, cursor
  pagination, idempotency keys, rate limiting, OpenAPI.
- **Worker** — pluggable executor registry, background heartbeat, graceful
  shutdown, result discard on lease loss.
- **CI/CD** — lint, type-check, unit matrix, integration against real Postgres,
  migration drift check, image build, compose smoke test.

## Next

- [ ] **Prometheus `/metrics`** — queue depth, claim latency, per-executor
      success rate, worker liveness.
- [ ] **Analytics endpoint** — p50/p95 duration and success rate per executor,
      via window functions over `task_attempts`.
- [ ] **Scheduled workflows** — cron expressions, using the existing delay queue
      as the timer.
- [ ] **A small status UI** — one page listing runs and their task graphs.
- [ ] **Priority ageing** — a low-priority task behind a steady stream of urgent
      ones currently starves. Ageing its effective priority with time bounds the
      wait.

## Explicitly not doing

- **A message broker.** The queue lives in Postgres so that claiming a task and
  recording why are one transaction. Introducing Redis or RabbitMQ would add a
  second system that can disagree with the first.
- **Exactly-once *delivery*.** It is not purchasable. Conductor provides
  at-least-once delivery plus exactly-once effects via a unique constraint, and
  says so plainly rather than pretending otherwise.
- **Cross-language workers.** A worker is a Python process running a registered
  executor. Supporting other languages means designing a wire protocol, which is
  a different project.
