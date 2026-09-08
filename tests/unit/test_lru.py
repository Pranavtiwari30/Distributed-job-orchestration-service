"""Unit tests for the LRU cache, including a model-based property test."""

from __future__ import annotations

from collections import OrderedDict

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conductor.core.lru import LruCache


def test_evicts_the_least_recently_used_entry() -> None:
    cache: LruCache[str, int] = LruCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.put("c", 3) == "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_a_read_refreshes_recency() -> None:
    cache: LruCache[str, int] = LruCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # "b" is now the eviction candidate
    assert cache.put("c", 3) == "b"
    assert cache.get("a") == 1


def test_membership_does_not_count_as_a_use() -> None:
    # Otherwise a monitoring endpoint that probes the cache would silently
    # reorder production's eviction decisions.
    cache: LruCache[str, int] = LruCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert "a" in cache
    assert cache.put("c", 3) == "a"


def test_updating_an_existing_key_refreshes_without_growing() -> None:
    cache: LruCache[str, int] = LruCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.put("a", 100) is None
    assert len(cache) == 2
    assert cache.get("a") == 100
    assert cache.put("c", 3) == "b"


def test_zero_capacity_caches_nothing() -> None:
    cache: LruCache[str, int] = LruCache(capacity=0)
    assert cache.put("a", 1) == "a"
    assert cache.get("a") is None
    assert len(cache) == 0


def test_negative_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LruCache(capacity=-1)


def test_get_or_load_reads_through_exactly_once() -> None:
    cache: LruCache[str, str] = LruCache(capacity=4)
    calls: list[str] = []

    def loader(key: str) -> str:
        calls.append(key)
        return f"definition-of-{key}"

    assert cache.get_or_load("wf-1", loader) == "definition-of-wf-1"
    assert cache.get_or_load("wf-1", loader) == "definition-of-wf-1"
    assert calls == ["wf-1"], "the second call must be served from cache"


def test_invalidate_forces_the_next_read_through() -> None:
    cache: LruCache[str, int] = LruCache(capacity=4)
    cache.put("wf-1", 1)
    assert cache.invalidate("wf-1") is True
    assert cache.invalidate("wf-1") is False
    assert "wf-1" not in cache


def test_iteration_is_most_to_least_recently_used() -> None:
    cache: LruCache[str, int] = LruCache(capacity=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.get("a")
    assert list(cache) == ["a", "c", "b"]


def test_stats_report_a_usable_hit_rate() -> None:
    cache: LruCache[str, int] = LruCache(capacity=2)
    assert cache.stats["hit_rate"] == 0.0, "no divide-by-zero on a cold cache"
    cache.put("a", 1)
    cache.get("a")
    cache.get("missing")
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1
    assert cache.stats["hit_rate"] == 0.5


@given(
    st.integers(min_value=1, max_value=8),
    st.lists(
        st.tuples(st.booleans(), st.integers(min_value=0, max_value=12)),
        max_size=120,
    ),
)
@settings(max_examples=300)
def test_property_matches_an_ordered_dict_reference_model(
    capacity: int, operations: list[tuple[bool, int]]
) -> None:
    """Run the cache and an `OrderedDict` model side by side.

    The model is obviously correct and obviously slow; the cache is neither.
    Any divergence in contents or ordering after an arbitrary operation
    sequence is a bug in the linked-list bookkeeping.
    """
    cache: LruCache[int, int] = LruCache(capacity)
    model: OrderedDict[int, int] = OrderedDict()

    for is_read, key in operations:
        if is_read:
            cached = cache.get(key)
            expected = model.get(key)
            if key in model:
                model.move_to_end(key, last=False)
            assert cached == expected
        else:
            cache.put(key, key * 10)
            model[key] = key * 10
            model.move_to_end(key, last=False)
            if len(model) > capacity:
                model.popitem(last=True)

        assert len(cache) == len(model) <= capacity
        assert list(cache) == list(model), "recency ordering diverged"
