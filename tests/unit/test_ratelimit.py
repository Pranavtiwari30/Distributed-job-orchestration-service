"""Unit tests for the token bucket, driven by an injected clock."""

from __future__ import annotations

import pytest

from conductor.core.ratelimit import RateLimiter, TokenBucket


class FakeClock:
    """A clock the test advances by hand.

    Rate limiting is entirely about the passage of time, so testing it against
    the real clock means either sleeping (slow) or asserting ranges (weak).
    """

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_a_fresh_bucket_allows_a_full_burst() -> None:
    bucket = TokenBucket(capacity=5, refill_rate=1.0, clock=FakeClock())
    assert all(bucket.try_acquire().allowed for _ in range(5))
    assert not bucket.try_acquire().allowed


def test_tokens_accrue_with_elapsed_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=10, refill_rate=2.0, clock=clock)
    for _ in range(10):
        bucket.try_acquire()
    assert not bucket.try_acquire().allowed

    clock.advance(1.5)  # 3 tokens at 2/sec
    assert bucket.try_acquire().allowed
    assert bucket.try_acquire().allowed
    assert bucket.try_acquire().allowed
    assert not bucket.try_acquire().allowed


def test_refill_never_exceeds_capacity() -> None:
    # Otherwise an idle client accumulates an unbounded burst allowance and
    # the "limit" stops limiting anything.
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, refill_rate=1.0, clock=clock)
    clock.advance(10_000)
    assert bucket.tokens == 5.0


def test_retry_after_says_exactly_when_to_come_back() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, refill_rate=0.5, clock=clock)
    bucket.try_acquire()

    decision = bucket.try_acquire()
    assert not decision.allowed
    assert decision.retry_after == pytest.approx(2.0), "0.5 tokens/sec means 2s for one token"

    clock.advance(decision.retry_after)
    assert bucket.try_acquire().allowed, "the advice it gave must actually work"


def test_headers_round_up_retry_after_so_clients_never_return_early() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, refill_rate=1.0, clock=clock)
    bucket.try_acquire()

    clock.advance(0.5)
    headers = bucket.try_acquire().headers()
    assert headers["Retry-After"] == "1"
    assert headers["X-RateLimit-Limit"] == "1"


def test_allowed_requests_carry_no_retry_after_header() -> None:
    bucket = TokenBucket(capacity=3, refill_rate=1.0, clock=FakeClock())
    assert "Retry-After" not in bucket.try_acquire().headers()


def test_a_cost_larger_than_capacity_is_a_configuration_error() -> None:
    # Not a 429: no amount of waiting would ever let this request through, so
    # reporting it as "try later" would be a lie.
    bucket = TokenBucket(capacity=5, refill_rate=1.0, clock=FakeClock())
    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        bucket.try_acquire(cost=6)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(capacity=0, refill_rate=1.0)
    with pytest.raises(ValueError, match="refill_rate"):
        TokenBucket(capacity=1, refill_rate=0.0)


def test_a_clock_that_goes_backwards_does_not_hand_out_free_tokens() -> None:
    clock = FakeClock(now=100.0)
    bucket = TokenBucket(capacity=5, refill_rate=1.0, clock=clock)
    for _ in range(5):
        bucket.try_acquire()

    clock.now = 0.0  # e.g. an NTP step, if a wall clock were used
    assert not bucket.try_acquire().allowed


# --------------------------------------------------------------------------
# per-key limiter
# --------------------------------------------------------------------------


def test_each_api_key_gets_an_independent_budget() -> None:
    limiter: RateLimiter[str] = RateLimiter(capacity=2, refill_rate=1.0, clock=FakeClock())
    assert limiter.check("tenant-a").allowed
    assert limiter.check("tenant-a").allowed
    assert not limiter.check("tenant-a").allowed
    assert limiter.check("tenant-b").allowed, "one noisy tenant must not throttle another"


def test_idle_buckets_are_evicted_so_the_map_cannot_grow_without_bound() -> None:
    clock = FakeClock()
    limiter: RateLimiter[str] = RateLimiter(capacity=2, refill_rate=1.0, clock=clock, idle_ttl=60.0)
    limiter.check("old")
    clock.advance(30)
    limiter.check("recent")
    clock.advance(31)  # "old" is now 61s stale, "recent" is 31s stale

    assert limiter.evict_idle() == 1
    assert len(limiter) == 1


def test_eviction_is_safe_because_a_reaped_bucket_was_already_full() -> None:
    clock = FakeClock()
    limiter: RateLimiter[str] = RateLimiter(capacity=3, refill_rate=1.0, clock=clock, idle_ttl=10.0)
    limiter.check("tenant")
    clock.advance(100)
    limiter.evict_idle()

    # The recreated bucket is full -- but so was the evicted one, having
    # refilled over 100 idle seconds. No budget was handed out for free.
    assert limiter.check("tenant").remaining == 2
