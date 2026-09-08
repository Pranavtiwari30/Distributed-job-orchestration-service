"""A fixed-capacity LRU cache: hash map over an intrusive doubly linked list.

Workflow definitions are read on every single task dispatch and rewritten almost
never, so they are the textbook case for caching. The requirement is O(1) get,
O(1) put, and O(1) eviction of the least recently used entry.

`OrderedDict` would do this in a line, and in production code it should. It is
written out here because the linked-list bookkeeping is the part worth being
able to explain -- and because the explicit structure makes the `capacity=0`
and self-referential-head edge cases visible rather than magic.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class _Node(Generic[K, V]):
    """A list node. `prev`/`next` are wired by the cache, never by callers."""

    __slots__ = ("key", "next", "prev", "value")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.prev: _Node[K, V] | None = None
        self.next: _Node[K, V] | None = None


class LruCache(Generic[K, V]):
    """Least-recently-used cache with O(1) get/put/evict.

    Ordering is maintained by a circular list between two sentinels: the node
    just after `_head` is the most recently used, the node just before `_tail`
    is the eviction candidate. Sentinels mean insertion and removal never need
    a null check, which is where hand-rolled linked lists usually go wrong.
    """

    __slots__ = ("_capacity", "_head", "_hits", "_misses", "_nodes", "_tail")

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self._capacity = capacity
        self._nodes: dict[K, _Node[K, V]] = {}
        # Sentinel nodes; their keys/values are never read.
        self._head: _Node[K, V] = _Node(None, None)  # type: ignore[arg-type]
        self._tail: _Node[K, V] = _Node(None, None)  # type: ignore[arg-type]
        self._head.next = self._tail
        self._tail.prev = self._head
        self._hits = 0
        self._misses = 0

    # ---- container protocol ---------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, key: object) -> bool:
        """Membership does *not* count as a use, so it cannot skew eviction."""
        return key in self._nodes

    def __iter__(self) -> Iterator[K]:
        """Keys from most to least recently used."""
        node = self._head.next
        while node is not None and node is not self._tail:
            yield node.key
            node = node.next

    # ---- operations ------------------------------------------------------

    def get(self, key: K, default: V | None = None) -> V | None:
        node = self._nodes.get(key)
        if node is None:
            self._misses += 1
            return default
        self._hits += 1
        self._move_to_front(node)
        return node.value

    def put(self, key: K, value: V) -> K | None:
        """Insert or update. Returns the evicted key, if one was evicted."""
        if self._capacity == 0:
            return key  # a zero-capacity cache evicts what it is handed

        existing = self._nodes.get(key)
        if existing is not None:
            existing.value = value
            self._move_to_front(existing)
            return None

        node = _Node(key, value)
        self._nodes[key] = node
        self._push_front(node)

        if len(self._nodes) > self._capacity:
            return self._evict()
        return None

    def get_or_load(self, key: K, loader: Callable[[K], V]) -> V:
        """Read through to `loader` on a miss, caching the result.

        The scheduler uses this so a cold cache degrades into a database read
        rather than into a special case at every call site.
        """
        node = self._nodes.get(key)
        if node is not None:
            self._hits += 1
            self._move_to_front(node)
            return node.value
        self._misses += 1
        value = loader(key)
        self.put(key, value)
        return value

    def invalidate(self, key: K) -> bool:
        """Drop an entry -- called when a workflow definition is updated."""
        node = self._nodes.pop(key, None)
        if node is None:
            return False
        self._unlink(node)
        return True

    def clear(self) -> None:
        self._nodes.clear()
        self._head.next = self._tail
        self._tail.prev = self._head

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def stats(self) -> dict[str, int | float]:
        """Hit rate, exported to /metrics so the capacity can be tuned on data."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._nodes),
            "capacity": self._capacity,
            "hit_rate": self._hits / total if total else 0.0,
        }

    # ---- linked-list internals -------------------------------------------

    def _push_front(self, node: _Node[K, V]) -> None:
        first = self._head.next
        assert first is not None
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node

    def _unlink(self, node: _Node[K, V]) -> None:
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

    def _move_to_front(self, node: _Node[K, V]) -> None:
        if self._head.next is node:
            return
        self._unlink(node)
        self._push_front(node)

    def _evict(self) -> K:
        victim = self._tail.prev
        assert victim is not None and victim is not self._head
        self._unlink(victim)
        del self._nodes[victim.key]
        return victim.key
