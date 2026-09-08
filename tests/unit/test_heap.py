"""Unit tests for the delay queue.

The heap invariant is checked explicitly after every mutating operation rather
than only inferred from pop order -- a heap can produce correct output for a
particular sequence while holding a corrupt array, and that bug surfaces later
under a different sequence.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conductor.core.heap import DelayQueue


def assert_invariants(queue: DelayQueue[str, str]) -> None:
    """Every parent is due no later than its children, and the index is exact."""
    heap = list(queue)
    for position, entry in enumerate(heap):
        parent = (position - 1) // 2
        if position > 0:
            assert heap[parent].due_at <= entry.due_at, f"heap property broken at {position}"
    assert len({e.key for e in heap}) == len(heap), "duplicate keys in the heap"
    assert len(heap) == len(queue)


def test_pop_returns_the_earliest_entry() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    queue.push("late", 300.0, "c")
    queue.push("early", 10.0, "a")
    queue.push("middle", 60.0, "b")
    assert_invariants(queue)

    assert [queue.pop().key for _ in range(3)] == ["early", "middle", "late"]


def test_peek_does_not_remove() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    queue.push("a", 1.0, "payload")
    assert queue.peek() is not None
    assert len(queue) == 1


def test_peek_on_empty_is_none_but_pop_raises() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    assert queue.peek() is None
    with pytest.raises(IndexError):
        queue.pop()


def test_pushing_a_known_key_reschedules_rather_than_duplicates() -> None:
    # Rescheduling a retry must move the timer, not add a second one that
    # fires spuriously after the task has already run.
    queue: DelayQueue[str, str] = DelayQueue()
    queue.push("job", 100.0, "first")
    queue.push("job", 5.0, "second")

    assert len(queue) == 1
    entry = queue.pop()
    assert (entry.due_at, entry.payload) == (5.0, "second")


def test_rescheduling_later_also_works() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    queue.push("a", 1.0, "a")
    queue.push("b", 2.0, "b")
    queue.push("a", 99.0, "a-delayed")  # must sift *down*
    assert_invariants(queue)
    assert queue.pop().key == "b"


def test_cancel_removes_a_scheduled_entry() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    for index in range(10):
        queue.push(f"k{index}", float(index), str(index))

    cancelled = queue.cancel("k4")
    assert cancelled is not None and cancelled.key == "k4"
    assert "k4" not in queue
    assert_invariants(queue)
    assert [entry.key for entry in queue.pop_due(now=100.0)] == [
        "k0",
        "k1",
        "k2",
        "k3",
        "k5",
        "k6",
        "k7",
        "k8",
        "k9",
    ]


def test_cancelling_an_unknown_key_is_a_no_op() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    assert queue.cancel("ghost") is None


def test_cancelling_the_last_element_does_not_corrupt_the_heap() -> None:
    # The `_remove_at` hole-filling path is skipped entirely when the removed
    # element *is* the last one; this pins that branch.
    queue: DelayQueue[str, str] = DelayQueue()
    queue.push("a", 1.0, "a")
    queue.push("b", 2.0, "b")
    assert queue.cancel("b") is not None
    assert_invariants(queue)
    assert queue.pop().key == "a"


def test_pop_due_drains_only_what_has_come_due() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    queue.push("now", 50.0, "a")
    queue.push("also-now", 100.0, "b")
    queue.push("later", 101.0, "c")

    assert [entry.key for entry in queue.pop_due(now=100.0)] == ["now", "also-now"]
    assert len(queue) == 1


def test_pop_due_respects_its_limit_so_a_backlog_cannot_starve_the_loop() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    for index in range(100):
        queue.push(f"k{index}", 1.0, "x")

    assert len(queue.pop_due(now=10.0, limit=25)) == 25
    assert len(queue) == 75


def test_seconds_until_next_drives_the_scheduler_sleep() -> None:
    queue: DelayQueue[str, str] = DelayQueue()
    assert queue.seconds_until_next(now=0.0) is None, "idle: sleep until woken"

    queue.push("a", 30.0, "a")
    assert queue.seconds_until_next(now=10.0) == 20.0
    assert queue.seconds_until_next(now=999.0) == 0.0, "overdue: do not sleep negative"


@given(
    st.lists(
        st.tuples(st.integers(min_value=0, max_value=50), st.floats(min_value=0, max_value=1e6)),
        max_size=60,
    )
)
@settings(max_examples=200)
def test_property_pops_emerge_in_nondecreasing_due_order(items: list[tuple[int, float]]) -> None:
    queue: DelayQueue[int, float] = DelayQueue()
    for key, due_at in items:
        queue.push(key, due_at, due_at)

    popped = [queue.pop().due_at for _ in range(len(queue))]
    assert popped == sorted(popped)


@given(
    st.lists(
        st.tuples(
            st.sampled_from(["push", "cancel", "pop"]),
            st.integers(min_value=0, max_value=15),
            st.floats(min_value=0, max_value=1000),
        ),
        max_size=80,
    )
)
@settings(max_examples=150)
def test_property_invariants_survive_arbitrary_operation_sequences(
    operations: list[tuple[str, int, float]],
) -> None:
    """Interleave pushes, cancels and pops at random and re-check the structure.

    This is the test that actually exercises `_remove_at`'s interaction with
    both sift directions -- the combination a hand-written case list misses.
    """
    queue: DelayQueue[int, float] = DelayQueue()
    for operation, key, due_at in operations:
        if operation == "push":
            queue.push(key, due_at, due_at)
        elif operation == "cancel":
            queue.cancel(key)
        elif queue:
            queue.pop()

        heap = list(queue)
        for position in range(1, len(heap)):
            assert heap[(position - 1) // 2].due_at <= heap[position].due_at
        assert all(key in queue for key in {entry.key for entry in heap})
