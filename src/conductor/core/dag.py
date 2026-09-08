"""Directed acyclic graph primitives backing workflow scheduling.

A workflow is a DAG of tasks. Two questions dominate the scheduler's hot path:

1. *Is this submission valid?*  -- answered once, at submit time, by attempting a
   topological ordering (Kahn's algorithm). A graph is a DAG if and only if such
   an ordering exists.
2. *Which tasks may start now?* -- answered on every task completion. Rather than
   re-scanning the graph we keep in-degree counters and decrement them, which
   turns an O(V + E) sweep into O(out_degree(node)) per completion.

Nothing here touches the database or the clock, so the whole module is directly
unit-testable and property-testable.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Sequence
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)

# DFS colours for cycle detection: unvisited, on the current path, fully
# explored. A back edge to a GREY node -- one still on the path -- is a cycle.
_WHITE, _GREY, _BLACK = 0, 1, 2


class CycleError(ValueError):
    """Raised when a graph that must be acyclic contains a cycle.

    Carries the offending cycle so the API can hand the caller something
    actionable instead of a bare "invalid workflow".
    """

    def __init__(self, cycle: Sequence[object]) -> None:
        self.cycle = list(cycle)
        rendered = " -> ".join(str(node) for node in cycle)
        super().__init__(f"workflow contains a cycle: {rendered}")


class UnknownNodeError(KeyError):
    """Raised when an edge references a node that was never declared."""

    def __init__(self, node: object) -> None:
        self.node = node
        super().__init__(f"unknown node: {node!r}")


class Dag(Generic[T]):
    """An append-only directed graph stored as an adjacency list.

    Insertion order is preserved throughout so that topological orderings are
    deterministic; a scheduler that reorders tasks between identical submissions
    is miserable to debug.
    """

    __slots__ = ("_children", "_parents")

    def __init__(self) -> None:
        # dict preserves insertion order (guaranteed since 3.7), which is what
        # makes `topological_order` deterministic rather than merely correct.
        self._children: dict[T, list[T]] = {}
        self._parents: dict[T, list[T]] = {}

    # ---- construction ---------------------------------------------------

    def add_node(self, node: T) -> None:
        """Register a node. Idempotent, so callers need not de-duplicate."""
        self._children.setdefault(node, [])
        self._parents.setdefault(node, [])

    def add_edge(self, parent: T, child: T) -> None:
        """Declare that `child` may not start until `parent` succeeds.

        Both endpoints must already exist: silently conjuring nodes from a typo
        in a dependency list is exactly the bug this rejects.
        """
        if parent not in self._children:
            raise UnknownNodeError(parent)
        if child not in self._children:
            raise UnknownNodeError(child)
        if parent == child:
            raise CycleError([parent, parent])
        if child in self._children[parent]:
            return  # duplicate edge: harmless, but keep the structure minimal
        self._children[parent].append(child)
        self._parents[child].append(parent)

    @classmethod
    def from_dependencies(cls, dependencies: dict[T, Iterable[T]]) -> Dag[T]:
        """Build a DAG from a `{node: [nodes it depends on]}` mapping.

        This is the shape the REST API accepts, because "what do I wait for" is
        how people describe pipelines out loud.
        """
        dag: Dag[T] = cls()
        for node in dependencies:
            dag.add_node(node)
        for node, parents in dependencies.items():
            for parent in parents:
                dag.add_edge(parent, node)
        return dag

    # ---- inspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._children)

    def __contains__(self, node: object) -> bool:
        return node in self._children

    def __iter__(self) -> Iterator[T]:
        return iter(self._children)

    @property
    def nodes(self) -> list[T]:
        return list(self._children)

    @property
    def edges(self) -> list[tuple[T, T]]:
        return [
            (parent, child) for parent, children in self._children.items() for child in children
        ]

    def children(self, node: T) -> list[T]:
        """Tasks unblocked (in part) by `node` finishing."""
        if node not in self._children:
            raise UnknownNodeError(node)
        return list(self._children[node])

    def parents(self, node: T) -> list[T]:
        """Tasks `node` is waiting on."""
        if node not in self._parents:
            raise UnknownNodeError(node)
        return list(self._parents[node])

    def in_degrees(self) -> dict[T, int]:
        """Unmet-dependency counts, the state the scheduler decrements."""
        return {node: len(parents) for node, parents in self._parents.items()}

    def roots(self) -> list[T]:
        """Nodes with no dependencies -- the tasks a fresh run starts with."""
        return [node for node, parents in self._parents.items() if not parents]

    def leaves(self) -> list[T]:
        """Nodes nothing depends on. A run is complete when all of these are."""
        return [node for node, children in self._children.items() if not children]

    # ---- algorithms -----------------------------------------------------

    def topological_order(self) -> list[T]:
        """Return nodes in dependency order via Kahn's algorithm. O(V + E).

        Raises `CycleError` (naming a concrete cycle) if no such order exists.
        """
        in_degree = self.in_degrees()
        # Seed with every zero-in-degree node, in insertion order.
        queue = deque(node for node in self._children if in_degree[node] == 0)
        order: list[T] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self._children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._children):
            # Kahn's tells us a cycle exists but not where; DFS finds the proof.
            raise CycleError(self._find_cycle())
        return order

    def is_acyclic(self) -> bool:
        try:
            self.topological_order()
        except CycleError:
            return False
        return True

    def _find_cycle(self) -> list[T]:
        """Locate one concrete cycle with an iterative colouring DFS.

        Iterative rather than recursive so a pathological 10k-node submission
        cannot blow the interpreter's stack -- the API accepts this input from
        the network, so its depth is not ours to trust.
        """
        colour: dict[T, int] = dict.fromkeys(self._children, _WHITE)

        for start in self._children:
            if colour[start] != _WHITE:
                continue
            # Explicit stack of (node, index of next child to visit).
            stack: list[tuple[T, int]] = [(start, 0)]
            path: list[T] = [start]
            colour[start] = _GREY

            while stack:
                node, index = stack[-1]
                if index < len(self._children[node]):
                    stack[-1] = (node, index + 1)
                    child = self._children[node][index]
                    if colour[child] == _GREY:
                        # Back edge: the cycle is the path from `child` onward.
                        cut = path.index(child)
                        return [*path[cut:], child]
                    if colour[child] == _WHITE:
                        colour[child] = _GREY
                        stack.append((child, 0))
                        path.append(child)
                else:
                    colour[node] = _BLACK
                    stack.pop()
                    path.pop()

        raise AssertionError("_find_cycle called on an acyclic graph")  # pragma: no cover


class DagCursor(Generic[T]):
    """Tracks a single run's progress through a `Dag`.

    Holds the mutable half of scheduling -- which tasks are done, which are
    blocked -- so that `Dag` itself stays an immutable, shareable definition.
    One cursor exists per workflow run.
    """

    __slots__ = ("_dag", "_failed", "_pending", "_succeeded")

    def __init__(self, dag: Dag[T]) -> None:
        if not dag.is_acyclic():
            raise CycleError(dag._find_cycle())
        self._dag = dag
        self._pending = dag.in_degrees()
        self._succeeded: set[T] = set()
        self._failed: set[T] = set()

    def ready(self) -> list[T]:
        """Tasks whose dependencies are all satisfied and which have not run."""
        return [
            node
            for node, unmet in self._pending.items()
            if unmet == 0 and node not in self._succeeded and node not in self._failed
        ]

    def mark_succeeded(self, node: T) -> list[T]:
        """Record a success and return the tasks it newly unblocked.

        Returning the delta is the whole point: the scheduler enqueues exactly
        these instead of recomputing readiness across the graph.
        """
        if node not in self._dag:
            raise UnknownNodeError(node)
        if node in self._succeeded:
            return []  # idempotent: at-least-once delivery means this repeats
        if self._pending[node] != 0:
            raise ValueError(f"{node!r} succeeded while still blocked -- scheduler bug")

        self._succeeded.add(node)
        self._failed.discard(node)
        unblocked: list[T] = []
        for child in self._dag.children(node):
            self._pending[child] -= 1
            if self._pending[child] == 0:
                unblocked.append(child)
        return unblocked

    def mark_failed(self, node: T) -> None:
        """Record a terminal failure. Descendants stay blocked forever."""
        if node not in self._dag:
            raise UnknownNodeError(node)
        self._failed.add(node)

    @property
    def succeeded(self) -> set[T]:
        return set(self._succeeded)

    @property
    def failed(self) -> set[T]:
        return set(self._failed)

    def blocked(self) -> list[T]:
        """Tasks that can never run because an ancestor failed."""
        if not self._failed:
            return []
        poisoned: set[T] = set()
        queue = deque(self._failed)
        while queue:
            node = queue.popleft()
            for child in self._dag.children(node):
                if child not in poisoned:
                    poisoned.add(child)
                    queue.append(child)
        return [node for node in self._dag if node in poisoned]

    def is_complete(self) -> bool:
        """True once no task can make further progress."""
        return not self.ready()

    def is_successful(self) -> bool:
        return len(self._succeeded) == len(self._dag)
