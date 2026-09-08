"""A binary min-heap keyed by wake-up time, backing the delay queue.

Tasks are frequently scheduled for the future: a retry with exponential backoff,
a `start_after` on submission, a visibility timeout that must fire if a worker
dies. The scheduler therefore repeatedly asks "what is due now?" and rarely asks
anything else, which is precisely a priority queue.

A sorted list gives O(1) peek but O(n) insert; an unsorted list gives O(1)
insert but O(n) peek. A binary heap gives O(log n) for both, and -- with a
side index from key to slot -- O(log n) cancellation too, which `heapq` cannot
offer at all. Cancellation matters: a job that completes before its visibility
timeout must be able to withdraw the timer rather than leave a tombstone behind.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True, slots=True)
class Entry(Generic[K, V]):
    """One scheduled item: fire `payload` at `due_at` (epoch seconds)."""

    key: K
    due_at: float
    payload: V


class DelayQueue(Generic[K, V]):
    """Min-heap over `due_at` with O(log n) push, pop, and cancel-by-key.

    Not thread-safe by design -- it is owned by a single scheduler loop. Sharing
    it across threads would mean a lock on the hottest path in the system, so
    instead each scheduler process owns one and they coordinate through Postgres.
    """

    __slots__ = ("_heap", "_index")

    def __init__(self) -> None:
        self._heap: list[Entry[K, V]] = []
        # key -> position in `_heap`, maintained by every sift. This is what
        # makes cancellation O(log n) instead of O(n).
        self._index: dict[K, int] = {}

    # ---- container protocol ---------------------------------------------

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def __iter__(self) -> Iterator[Entry[K, V]]:
        """Iterate in heap order, which is *not* due order. For tests/debug."""
        return iter(list(self._heap))

    # ---- public operations ----------------------------------------------

    def push(self, key: K, due_at: float, payload: V) -> None:
        """Schedule `payload` for `due_at`, replacing any entry with this key.

        Replace-on-duplicate is deliberate: rescheduling a retry should move the
        existing timer, not accumulate a second one that fires spuriously.
        """
        if key in self._index:
            self._replace(key, due_at, payload)
            return
        self._heap.append(Entry(key, due_at, payload))
        self._index[key] = len(self._heap) - 1
        self._sift_up(len(self._heap) - 1)

    def peek(self) -> Entry[K, V] | None:
        """The earliest entry, without removing it. O(1)."""
        return self._heap[0] if self._heap else None

    def pop(self) -> Entry[K, V]:
        """Remove and return the earliest entry. O(log n)."""
        if not self._heap:
            raise IndexError("pop from an empty DelayQueue")
        return self._remove_at(0)

    def cancel(self, key: K) -> Entry[K, V] | None:
        """Withdraw a scheduled entry. Returns it, or None if it was not queued."""
        position = self._index.get(key)
        if position is None:
            return None
        return self._remove_at(position)

    def pop_due(self, now: float, limit: int | None = None) -> list[Entry[K, V]]:
        """Drain every entry due at or before `now`, earliest first.

        `limit` caps a single drain so that a large backlog cannot starve the
        rest of the scheduler loop -- the remainder is simply picked up on the
        next tick.
        """
        due: list[Entry[K, V]] = []
        while self._heap and self._heap[0].due_at <= now:
            if limit is not None and len(due) >= limit:
                break
            due.append(self.pop())
        return due

    def seconds_until_next(self, now: float) -> float | None:
        """How long the loop may sleep. None means "nothing scheduled".

        Returning this instead of polling on a fixed interval is what keeps an
        idle scheduler at ~0% CPU while still firing timers promptly.
        """
        head = self.peek()
        if head is None:
            return None
        return max(0.0, head.due_at - now)

    # ---- heap internals --------------------------------------------------

    def _replace(self, key: K, due_at: float, payload: V) -> None:
        position = self._index[key]
        self._heap[position] = Entry(key, due_at, payload)
        # The new deadline may be earlier or later, so try both directions;
        # exactly one of them will move it (or neither, if it is already placed).
        self._sift_up(position)
        self._sift_down(position)

    def _remove_at(self, position: int) -> Entry[K, V]:
        entry = self._heap[position]
        last = self._heap.pop()
        del self._index[entry.key]
        if position < len(self._heap):
            # Move the former last element into the hole and re-heapify.
            self._heap[position] = last
            self._index[last.key] = position
            self._sift_up(position)
            self._sift_down(position)
        return entry

    def _swap(self, a: int, b: int) -> None:
        self._heap[a], self._heap[b] = self._heap[b], self._heap[a]
        self._index[self._heap[a].key] = a
        self._index[self._heap[b].key] = b

    def _sift_up(self, position: int) -> None:
        while position > 0:
            parent = (position - 1) // 2
            if self._heap[position].due_at >= self._heap[parent].due_at:
                break
            self._swap(position, parent)
            position = parent

    def _sift_down(self, position: int) -> None:
        size = len(self._heap)
        while True:
            left = 2 * position + 1
            right = left + 1
            smallest = position
            if left < size and self._heap[left].due_at < self._heap[smallest].due_at:
                smallest = left
            if right < size and self._heap[right].due_at < self._heap[smallest].due_at:
                smallest = right
            if smallest == position:
                return
            self._swap(position, smallest)
            position = smallest
