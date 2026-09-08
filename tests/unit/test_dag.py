"""Unit tests for the DAG primitives.

Split into example-based tests, which pin down the behaviour a reader needs to
understand, and property-based tests, which check invariants against thousands
of generated graphs -- including the malformed ones nobody thinks to write by
hand.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conductor.core.dag import CycleError, Dag, DagCursor, UnknownNodeError


def build(dependencies: dict[str, list[str]]) -> Dag[str]:
    return Dag.from_dependencies(dependencies)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_add_node_is_idempotent() -> None:
    dag: Dag[str] = Dag()
    dag.add_node("a")
    dag.add_node("a")
    assert len(dag) == 1


def test_duplicate_edges_are_collapsed() -> None:
    dag = build({"a": [], "b": ["a"]})
    dag.add_edge("a", "b")
    assert dag.edges == [("a", "b")]


def test_edge_to_unknown_node_is_rejected() -> None:
    dag = build({"a": []})
    with pytest.raises(UnknownNodeError):
        dag.add_edge("a", "typo")


def test_self_loop_is_rejected_immediately() -> None:
    dag = build({"a": []})
    with pytest.raises(CycleError) as excinfo:
        dag.add_edge("a", "a")
    assert excinfo.value.cycle == ["a", "a"]


# --------------------------------------------------------------------------
# topological ordering
# --------------------------------------------------------------------------


def test_topological_order_respects_dependencies() -> None:
    dag = build({"extract": [], "transform": ["extract"], "load": ["transform"]})
    assert dag.topological_order() == ["extract", "transform", "load"]


def test_diamond_orders_both_middles_before_the_join() -> None:
    dag = build({"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]})
    order = dag.topological_order()
    assert order.index("a") < order.index("b") < order.index("d")
    assert order.index("a") < order.index("c") < order.index("d")


def test_ordering_is_deterministic_across_calls() -> None:
    # A scheduler that shuffles identical submissions is untraceable in prod.
    dag = build({"a": [], "b": [], "c": ["a", "b"], "d": ["a"]})
    assert dag.topological_order() == dag.topological_order()


def test_cycle_is_detected_and_reported_concretely() -> None:
    dag = build({"a": ["c"], "b": ["a"], "c": ["b"]})
    with pytest.raises(CycleError) as excinfo:
        dag.topological_order()

    cycle = excinfo.value.cycle
    assert cycle[0] == cycle[-1], "a reported cycle should close on itself"
    assert set(cycle) == {"a", "b", "c"}


def test_cycle_detection_survives_a_graph_deeper_than_the_stack() -> None:
    # 10k nodes: a recursive DFS raises RecursionError here. The API accepts
    # this shape from the network, so the iterative implementation matters.
    depth = 10_000
    dependencies: dict[str, list[str]] = {"n0": []}
    for i in range(1, depth):
        dependencies[f"n{i}"] = [f"n{i - 1}"]
    dag = build(dependencies)
    assert len(dag.topological_order()) == depth

    dag.add_edge(f"n{depth - 1}", "n0")  # close the loop
    with pytest.raises(CycleError):
        dag.topological_order()


def test_empty_dag_is_vacuously_acyclic() -> None:
    dag: Dag[str] = Dag()
    assert dag.topological_order() == []
    assert dag.is_acyclic()


# --------------------------------------------------------------------------
# structural queries
# --------------------------------------------------------------------------


def test_roots_and_leaves() -> None:
    dag = build({"a": [], "b": [], "c": ["a", "b"], "d": ["c"]})
    assert set(dag.roots()) == {"a", "b"}
    assert dag.leaves() == ["d"]


def test_in_degrees_count_unmet_dependencies() -> None:
    dag = build({"a": [], "b": ["a"], "c": ["a", "b"]})
    assert dag.in_degrees() == {"a": 0, "b": 1, "c": 2}


# --------------------------------------------------------------------------
# cursor: run-time progress tracking
# --------------------------------------------------------------------------


def test_cursor_starts_ready_at_the_roots() -> None:
    cursor = DagCursor(build({"a": [], "b": [], "c": ["a", "b"]}))
    assert set(cursor.ready()) == {"a", "b"}


def test_completion_unblocks_only_fully_satisfied_children() -> None:
    cursor = DagCursor(build({"a": [], "b": [], "c": ["a", "b"]}))
    assert cursor.mark_succeeded("a") == [], "c still waits on b"
    assert cursor.mark_succeeded("b") == ["c"]


def test_marking_the_same_task_twice_is_a_no_op() -> None:
    # At-least-once delivery guarantees this happens; it must not double-
    # decrement an in-degree and release a task early.
    cursor = DagCursor(build({"a": [], "b": ["a"], "c": ["a", "b"]}))
    assert cursor.mark_succeeded("a") == ["b"]
    assert cursor.mark_succeeded("a") == []
    assert "c" not in cursor.ready()


def test_cursor_rejects_a_completion_that_was_still_blocked() -> None:
    cursor = DagCursor(build({"a": [], "b": ["a"]}))
    with pytest.raises(ValueError, match="scheduler bug"):
        cursor.mark_succeeded("b")


def test_failure_poisons_every_descendant() -> None:
    cursor = DagCursor(build({"a": [], "b": ["a"], "c": ["b"], "d": []}))
    cursor.mark_failed("a")
    assert set(cursor.blocked()) == {"b", "c"}
    assert "d" in cursor.ready(), "an unrelated branch keeps running"


def test_cursor_rejects_a_cyclic_definition() -> None:
    dag = build({"a": ["b"], "b": ["a"]})
    with pytest.raises(CycleError):
        DagCursor(dag)


def test_run_is_successful_only_when_every_task_succeeded() -> None:
    cursor = DagCursor(build({"a": [], "b": ["a"]}))
    cursor.mark_succeeded("a")
    assert not cursor.is_successful()
    cursor.mark_succeeded("b")
    assert cursor.is_successful()
    assert cursor.is_complete()


# --------------------------------------------------------------------------
# properties
# --------------------------------------------------------------------------


@st.composite
def acyclic_graphs(draw: st.DrawFn) -> Dag[int]:
    """Generate a random DAG.

    Acyclicity is structural rather than checked: edges only ever point from a
    lower index to a higher one, and any such graph is acyclic. That gives
    Hypothesis a generator with no rejection sampling, so it explores the space
    instead of fighting a filter.
    """
    size = draw(st.integers(min_value=0, max_value=25))
    dependencies: dict[int, list[int]] = {}
    for node in range(size):
        parents = draw(
            st.lists(
                st.integers(min_value=0, max_value=max(0, node - 1)),
                max_size=node,
                unique=True,
            )
        )
        dependencies[node] = [p for p in parents if p < node]
    return Dag.from_dependencies(dependencies)


@given(acyclic_graphs())
@settings(max_examples=200)
def test_property_topological_order_places_parents_first(dag: Dag[int]) -> None:
    order = dag.topological_order()
    position = {node: index for index, node in enumerate(order)}

    assert len(order) == len(dag), "every node appears exactly once"
    for parent, child in dag.edges:
        assert position[parent] < position[child]


@given(acyclic_graphs())
@settings(max_examples=200)
def test_property_cursor_drains_every_task_of_an_all_succeeding_run(dag: Dag[int]) -> None:
    """Simulate a run where nothing fails: it must reach every node.

    This is the invariant that catches an off-by-one in the in-degree
    bookkeeping -- a task left permanently blocked is a run that hangs forever.
    """
    cursor = DagCursor(dag)
    executed: list[int] = []

    while ready := cursor.ready():
        for node in ready:
            cursor.mark_succeeded(node)
            executed.append(node)

    assert sorted(executed) == sorted(dag.nodes)
    assert cursor.is_successful()
