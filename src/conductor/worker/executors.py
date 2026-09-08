"""Pluggable task executors, resolved through a registry.

The worker loop knows nothing about what a task *does*. It looks the executor up
by name, calls one method, and interprets the result. Adding a new task type is
therefore a new class and a decorator -- no change to the worker, the scheduler,
or the database.

Two rules every executor must follow:

* **Raise on failure.** Returning a result means success; the retry policy is
  applied by the caller, never here.
* **Respect the timeout.** A task that hangs forever holds a lease that its
  worker keeps renewing, so it is never reaped -- the one failure mode leases
  do not protect against.
"""

from __future__ import annotations

import shlex
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


class ExecutorError(RuntimeError):
    """A task failed. The message is persisted as `task_runs.last_error`."""


class Executor(ABC):
    """Runs one task and returns its output as a JSON-serialisable dict."""

    name: str

    @abstractmethod
    def execute(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Do the work. Raise `ExecutorError` on failure; return output on success."""


_REGISTRY: dict[str, type[Executor]] = {}


def register(cls: type[Executor]) -> type[Executor]:
    """Class decorator adding an executor to the registry.

    Registration lives beside the class rather than in a central list, so a new
    executor cannot be added and then forgotten at the wiring step.
    """
    if cls.name in _REGISTRY:
        raise ValueError(f"executor {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def build_executor(name: str) -> Executor:
    """Resolve an executor by name, or fail with the list of valid names."""
    executor_cls = _REGISTRY.get(name)
    if executor_cls is None:
        raise ExecutorError(f"unknown executor {name!r}; registered: {sorted(_REGISTRY)}")
    return executor_cls()


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


@register
class NoopExecutor(Executor):
    """Succeeds immediately. Used for tests and for structural DAG nodes."""

    name = "noop"

    def execute(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:  # noqa: ARG002
        return {"executed": True, "params": params}


@register
class FailingExecutor(Executor):
    """Always fails. Exists so retry and dead-letter paths are testable end to end."""

    name = "fail"

    def execute(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:  # noqa: ARG002
        raise ExecutorError(str(params.get("message", "deliberate failure")))


@register
class HttpExecutor(Executor):
    """Calls an HTTP endpoint.

    Treats 4xx and 5xx alike -- both raise, both retry. That is deliberately
    crude: distinguishing "retryable" from "permanent" per status code is a
    policy decision that belongs in the task's retry configuration, not
    hard-coded into the transport.
    """

    name = "http"

    def execute(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        url = params.get("url")
        if not url:
            raise ExecutorError("http executor requires a 'url' parameter")

        try:
            response = httpx.request(
                method=str(params.get("method", "GET")).upper(),
                url=str(url),
                json=params.get("body"),
                headers=params.get("headers") or {},
                # Bounded by the task's own timeout: the lease must outlive the
                # request, or the task gets reaped while it is still running.
                timeout=timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ExecutorError(
                f"{exc.response.status_code} from {url}: {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ExecutorError(f"request to {url} failed: {exc}") from exc

        return {
            "status_code": response.status_code,
            # Truncated: a task's output lands in a JSONB column that the API
            # returns in full, so an unbounded response body would be a memory
            # amplification bug reachable from a task definition.
            "body": response.text[:10_000],
        }


@register
class ShellExecutor(Executor):
    """Runs a local command.

    `shell=False` with `shlex.split`, never `shell=True`: task parameters come
    from the API, and a shell here would make `"; rm -rf /"` a valid workflow
    definition.
    """

    name = "shell"

    def execute(self, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        command = params.get("command")
        if not command:
            raise ExecutorError("shell executor requires a 'command' parameter")

        argv = shlex.split(command) if isinstance(command, str) else list(command)
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, never a shell string
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ExecutorError(f"command not found: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ExecutorError(f"command exceeded its {timeout}s timeout") from exc

        if completed.returncode != 0:
            raise ExecutorError(f"exit {completed.returncode}: {completed.stderr.strip()[:1000]}")
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[:10_000],
            "stderr": completed.stderr[:10_000],
        }


ExecutorFactory = Callable[[str], Executor]
