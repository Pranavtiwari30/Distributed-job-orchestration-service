"""Retry policies, expressed as a Strategy so the executor stays policy-free.

The worker's job is "run the task, then ask the policy what happens next". It
does not know what exponential backoff is, and adding a new policy requires no
change to it -- which is the entire argument for the pattern here rather than a
branch on a `retry_type` string.

Two properties every policy must have, learned the hard way by everyone who has
run a queue in production:

* **A cap.** Unbounded exponential backoff eventually schedules a retry after
  the heat death of the universe; it is indistinguishable from a lost job.
* **Jitter.** A thousand tasks that fail against the same downed dependency at
  the same instant will, without jitter, all retry at the same instant and
  knock it down again. The randomness is not a nicety; it is the point.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

# Clamp for any computed delay. A day is far past the point where a human should
# have been paged, and it keeps `2 ** attempt` from overflowing into nonsense.
MAX_DELAY_SECONDS = 86_400.0


class RetryPolicy(ABC):
    """Decides whether, and how long after, a failed attempt is retried."""

    #: The key this policy is addressed by in the API and in the database.
    wire_name: ClassVar[str] = "unknown"

    @property
    @abstractmethod
    def max_attempts(self) -> int:
        """Total attempts allowed, including the first. Always >= 1."""

    @abstractmethod
    def compute_delay(self, attempt: int) -> float:
        """Seconds to wait before attempt number `attempt + 1`."""

    def should_retry(self, attempt: int) -> bool:
        """True if a task that has made `attempt` attempts gets another.

        `attempt` is 1-based: after the first failure it is 1.
        """
        return attempt < self.max_attempts

    def next_delay(self, attempt: int) -> float | None:
        """Seconds until the next attempt, or None when retries are exhausted.

        The single call site the worker needs: `None` means "dead-letter it".
        """
        if attempt < 1:
            raise ValueError("attempt is 1-based; the first failure is attempt 1")
        if not self.should_retry(attempt):
            return None
        return min(max(0.0, self.compute_delay(attempt)), MAX_DELAY_SECONDS)

    def _options(self) -> dict[str, object]:
        """Policy-specific fields for `describe`. Overridden, never called directly.

        A template-method hook rather than a `super().describe()` chain: the
        zero-argument `super()` closure does not survive `@dataclass(slots=True)`,
        which rebuilds the class after the method body has already been compiled.
        """
        return {}

    def describe(self) -> dict[str, object]:
        """Serialisable form, persisted with the task and echoed by the API.

        Emits `wire_name`, not the class name, so that the output round-trips
        back through `build_policy` unchanged.
        """
        return {
            "type": self.wire_name,
            "max_attempts": self.max_attempts,
            **self._options(),
        }


@dataclass(frozen=True, slots=True)
class NoRetry(RetryPolicy):
    """Fail fast. Correct for non-idempotent side effects we cannot safely repeat."""

    wire_name: ClassVar[str] = "none"

    @property
    def max_attempts(self) -> int:
        return 1

    def compute_delay(self, attempt: int) -> float:  # noqa: ARG002 - abstract signature
        raise AssertionError("NoRetry never retries")  # pragma: no cover


@dataclass(frozen=True, slots=True)
class FixedDelay(RetryPolicy):
    """Constant wait between attempts. Fine for a flaky dependency with a known
    recovery time; poor for anything that fails because it is overloaded."""

    wire_name: ClassVar[str] = "fixed"

    delay: float = 5.0
    attempts: int = 3

    def __post_init__(self) -> None:
        _validate(self.attempts, self.delay)

    @property
    def max_attempts(self) -> int:
        return self.attempts

    def compute_delay(self, attempt: int) -> float:  # noqa: ARG002 - delay is constant
        return self.delay

    def _options(self) -> dict[str, object]:
        return {"delay": self.delay}


@dataclass(frozen=True, slots=True)
class ExponentialBackoff(RetryPolicy):
    """`base * factor ** (attempt - 1)`, capped, with optional full jitter.

    Full jitter (`uniform(0, computed)`) rather than the computed value itself:
    it spreads a thundering herd across the whole window instead of merely
    delaying it, at the cost of retrying sooner on average.

    `rng` is injectable so tests can assert exact delays instead of ranges --
    a policy whose behaviour can only be sampled is a policy nobody trusts.
    """

    wire_name: ClassVar[str] = "exponential"

    base: float = 1.0
    factor: float = 2.0
    attempts: int = 5
    max_delay: float = 300.0
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random, compare=False, repr=False)

    def __post_init__(self) -> None:
        _validate(self.attempts, self.base)
        if self.factor < 1.0:
            raise ValueError("factor must be >= 1.0, else backoff shrinks")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")

    @property
    def max_attempts(self) -> int:
        return self.attempts

    def compute_delay(self, attempt: int) -> float:
        # Cap the exponent before computing the power: at attempt ~1000 the
        # float itself overflows, and `min()` afterwards is too late.
        exponent = min(attempt - 1, 64)
        raw = min(self.base * (self.factor**exponent), self.max_delay)
        return self.rng.uniform(0.0, raw) if self.jitter else raw

    def _options(self) -> dict[str, object]:
        return {
            "base": self.base,
            "factor": self.factor,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
        }


def _validate(attempts: int, delay: float) -> None:
    if attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if delay < 0:
        raise ValueError("delay must be non-negative")


# Registry mapping the API's wire format onto policy classes. Keeping it here,
# beside the policies, means adding one is a single-file change.
_POLICIES: dict[str, type[RetryPolicy]] = {
    policy.wire_name: policy for policy in (NoRetry, FixedDelay, ExponentialBackoff)
}


def build_policy(spec: dict[str, object] | None) -> RetryPolicy:
    """Construct a policy from its persisted/wire form.

    Unknown keys are rejected rather than ignored: a typo'd `max_attemps` that
    silently leaves the default in place is a bug that surfaces only in an
    incident.
    """
    if not spec:
        return ExponentialBackoff()

    kind = str(spec.get("type", "exponential")).lower()
    policy_cls = _POLICIES.get(kind)
    if policy_cls is None:
        raise ValueError(f"unknown retry policy {kind!r}; expected one of {sorted(_POLICIES)}")

    kwargs = {key: value for key, value in spec.items() if key != "type"}
    if "max_attempts" in kwargs:
        kwargs["attempts"] = kwargs.pop("max_attempts")

    allowed = {f for f in policy_cls.__dataclass_fields__ if f != "rng"}  # type: ignore[attr-defined]
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(f"unknown options for {kind!r} policy: {sorted(unknown)}")
    return policy_cls(**kwargs)
