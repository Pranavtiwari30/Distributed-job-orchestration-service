"""Unit tests for retry policies.

Delays are asserted exactly, never sampled: the RNG is injected, so a jittered
policy is as deterministic under test as a fixed one.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conductor.core.retry import (
    MAX_DELAY_SECONDS,
    ExponentialBackoff,
    FixedDelay,
    NoRetry,
    build_policy,
)


class SequenceRandom(random.Random):
    """A `Random` that returns a scripted fraction of each range.

    Injecting this makes "full jitter picked the top of the window" a case we
    can assert rather than a case we hope the fuzzer eventually hits.
    """

    def __init__(self, fractions: list[float]) -> None:
        super().__init__()
        self._fractions = fractions
        self._calls = 0

    def uniform(self, a: float, b: float) -> float:
        fraction = self._fractions[self._calls % len(self._fractions)]
        self._calls += 1
        return a + (b - a) * fraction


# --------------------------------------------------------------------------
# NoRetry
# --------------------------------------------------------------------------


def test_no_retry_gives_up_after_the_first_attempt() -> None:
    assert NoRetry().next_delay(attempt=1) is None


# --------------------------------------------------------------------------
# FixedDelay
# --------------------------------------------------------------------------


def test_fixed_delay_is_constant_until_attempts_run_out() -> None:
    policy = FixedDelay(delay=7.0, attempts=3)
    assert policy.next_delay(1) == 7.0
    assert policy.next_delay(2) == 7.0
    assert policy.next_delay(3) is None, "the third attempt was the last one"


def test_fixed_delay_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        FixedDelay(attempts=0)
    with pytest.raises(ValueError, match="non-negative"):
        FixedDelay(delay=-1.0)


# --------------------------------------------------------------------------
# ExponentialBackoff
# --------------------------------------------------------------------------


def test_exponential_backoff_doubles_without_jitter() -> None:
    policy = ExponentialBackoff(base=1.0, factor=2.0, attempts=6, jitter=False)
    assert [policy.next_delay(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_exponential_backoff_is_capped_by_max_delay() -> None:
    policy = ExponentialBackoff(base=1.0, factor=10.0, attempts=10, max_delay=30.0, jitter=False)
    assert policy.next_delay(5) == 30.0, "10^4 clamped to the ceiling"


def test_full_jitter_spans_zero_to_the_computed_window() -> None:
    policy = ExponentialBackoff(
        base=10.0, factor=2.0, attempts=5, jitter=True, rng=SequenceRandom([0.0, 0.5, 1.0])
    )
    assert policy.next_delay(1) == 0.0  # bottom of the window
    assert policy.next_delay(2) == 10.0  # middle of [0, 20]
    assert policy.next_delay(3) == 40.0  # top of [0, 40]


def test_jitter_prevents_a_synchronised_retry_storm() -> None:
    """The whole reason jitter exists: 500 tasks failing together must not
    all come back at the same instant."""
    policy = ExponentialBackoff(base=30.0, factor=2.0, attempts=5, jitter=True)
    delays = [policy.next_delay(2) for _ in range(500)]
    assert len(set(delays)) > 400, "delays are not meaningfully spread"
    assert all(0.0 <= delay <= 60.0 for delay in delays if delay is not None)


def test_a_huge_attempt_number_cannot_overflow_into_nonsense() -> None:
    # Without clamping the exponent, `2.0 ** 5000` raises OverflowError long
    # before `min()` ever sees the value.
    policy = ExponentialBackoff(base=1.0, factor=2.0, attempts=10_000, jitter=False)
    assert policy.next_delay(5000) == 300.0


def test_delay_is_globally_clamped_even_if_max_delay_is_absurd() -> None:
    policy = ExponentialBackoff(base=1.0, factor=2.0, attempts=99, max_delay=1e12, jitter=False)
    assert policy.next_delay(60) == MAX_DELAY_SECONDS


def test_shrinking_backoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="factor must be"):
        ExponentialBackoff(factor=0.5)


def test_attempt_numbering_is_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        ExponentialBackoff().next_delay(0)


# --------------------------------------------------------------------------
# the wire-format factory
# --------------------------------------------------------------------------


def test_build_policy_defaults_to_exponential_backoff() -> None:
    assert isinstance(build_policy(None), ExponentialBackoff)


def test_build_policy_accepts_the_api_spelling_of_max_attempts() -> None:
    policy = build_policy({"type": "fixed", "delay": 2.0, "max_attempts": 7})
    assert isinstance(policy, FixedDelay)
    assert policy.max_attempts == 7


def test_build_policy_rejects_an_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown retry policy"):
        build_policy({"type": "fibonacci"})


def test_build_policy_rejects_a_typo_rather_than_ignoring_it() -> None:
    # Silently defaulting on `max_attemps` is a bug that only ever surfaces
    # during an incident.
    with pytest.raises(ValueError, match="unknown options"):
        build_policy({"type": "fixed", "max_attemps": 3})


def test_describe_round_trips_through_build_policy() -> None:
    original = ExponentialBackoff(base=2.0, factor=3.0, attempts=4, max_delay=99.0, jitter=False)
    rebuilt = build_policy(original.describe())
    assert rebuilt == original


@given(
    st.floats(min_value=0.01, max_value=100),
    st.floats(min_value=1.0, max_value=5.0),
    st.integers(min_value=1, max_value=20),
)
@settings(max_examples=200)
def test_property_delays_are_bounded_and_terminate(
    base: float, factor: float, attempts: int
) -> None:
    """Two invariants every policy must satisfy, for any configuration:
    delays never exceed the cap, and retries always eventually stop."""
    policy = ExponentialBackoff(base=base, factor=factor, attempts=attempts, max_delay=600.0)

    for attempt in range(1, attempts):
        delay = policy.next_delay(attempt)
        assert delay is not None
        assert 0.0 <= delay <= 600.0

    assert policy.next_delay(attempts) is None, "retries must terminate"
