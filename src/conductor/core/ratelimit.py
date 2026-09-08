"""Token-bucket rate limiting, applied per API key.

A bucket holds up to `capacity` tokens and refills at `refill_rate` per second;
a request costs one token. Choosing this over a fixed window is what allows a
client to burst -- submitting fifty tasks at once is normal usage, not abuse --
while still bounding the sustained rate, and it avoids the fixed-window edge
where 2x the limit slips through either side of a boundary.

The bucket is *lazy*: no timer refills it. Each call computes how many tokens
would have accrued since the last one, which makes the whole thing O(1) with no
background work, and makes it exactly testable with an injected clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of one rate-limit check, shaped for RFC 6585 headers."""

    allowed: bool
    remaining: int
    retry_after: float
    limit: int

    def headers(self) -> dict[str, str]:
        """`X-RateLimit-*` headers, plus `Retry-After` when throttled.

        Telling a client exactly when to come back is the difference between a
        well-behaved retry and a hot loop against a 429.
        """
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if not self.allowed:
            # Retry-After is integer seconds; round up so we never invite a
            # client back a fraction of a second early.
            headers["Retry-After"] = str(max(1, int(self.retry_after + 0.999)))
        return headers


class TokenBucket:
    """A single lazily-refilled bucket."""

    __slots__ = ("_capacity", "_clock", "_refill_rate", "_tokens", "_updated_at")

    def __init__(self, capacity: int, refill_rate: float, clock: Clock = time.monotonic) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        self._capacity = capacity
        self._refill_rate = refill_rate
        # `time.monotonic` by default: a wall-clock adjustment (NTP, DST) must
        # not hand out free tokens or freeze the bucket.
        self._clock = clock
        self._tokens = float(capacity)
        self._updated_at = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._updated_at
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._updated_at = now

    def try_acquire(self, cost: int = 1) -> Decision:
        """Attempt to spend `cost` tokens; never blocks."""
        if cost < 1:
            raise ValueError("cost must be at least 1")
        if cost > self._capacity:
            raise ValueError(f"cost {cost} exceeds bucket capacity {self._capacity}")

        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return Decision(True, int(self._tokens), 0.0, self._capacity)

        deficit = cost - self._tokens
        return Decision(False, int(self._tokens), deficit / self._refill_rate, self._capacity)

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens


class RateLimiter(Generic[K]):
    """Per-key buckets with lazy creation and idle eviction.

    Without eviction this map is an unbounded memory leak keyed by anything a
    caller can put in a header -- so idle buckets are reaped, and reaping is
    safe precisely because a fully-refilled bucket is indistinguishable from a
    brand new one.
    """

    __slots__ = ("_buckets", "_capacity", "_clock", "_idle_ttl", "_last_seen", "_refill_rate")

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        clock: Clock = time.monotonic,
        idle_ttl: float = 3600.0,
    ) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._clock = clock
        self._idle_ttl = idle_ttl
        self._buckets: dict[K, TokenBucket] = {}
        self._last_seen: dict[K, float] = {}

    def check(self, key: K, cost: int = 1) -> Decision:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(self._capacity, self._refill_rate, self._clock)
            self._buckets[key] = bucket
        self._last_seen[key] = self._clock()
        return bucket.try_acquire(cost)

    def evict_idle(self) -> int:
        """Drop buckets untouched for `idle_ttl`. Returns how many were dropped."""
        cutoff = self._clock() - self._idle_ttl
        stale = [key for key, seen in self._last_seen.items() if seen < cutoff]
        for key in stale:
            del self._buckets[key]
            del self._last_seen[key]
        return len(stale)

    def __len__(self) -> int:
        return len(self._buckets)
