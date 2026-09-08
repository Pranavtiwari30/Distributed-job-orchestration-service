"""Unit tests for the executor registry and the built-in executors.

The HTTP executor is tested against a real local server rather than a mocked
`httpx`: mocking the client would test that the mock was configured correctly,
not that a non-2xx response actually raises.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator

import pytest

from conductor.worker.executors import (
    Executor,
    ExecutorError,
    build_executor,
    register,
    registered_names,
)

# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_the_builtin_executors_are_registered() -> None:
    assert {"noop", "fail", "http", "shell"} <= set(registered_names())


def test_an_unknown_executor_names_the_valid_options() -> None:
    with pytest.raises(ExecutorError) as excinfo:
        build_executor("does-not-exist")
    assert "registered:" in str(excinfo.value), "the error should be actionable"


def test_registering_a_duplicate_name_is_rejected() -> None:
    # Silently replacing an executor would mean a name collision changes what
    # every existing workflow using that name does.
    with pytest.raises(ValueError, match="already registered"):

        @register
        class Duplicate(Executor):
            name = "noop"

            def execute(self, _params: dict[str, object], _timeout: float) -> dict[str, object]:
                return {}


def test_a_new_executor_needs_no_change_to_the_worker() -> None:
    @register
    class Custom(Executor):
        name = "test-only-custom"

        def execute(self, _params: dict[str, object], _timeout: float) -> dict[str, object]:
            return {"custom": True}

    assert build_executor("test-only-custom").execute({}, 1.0) == {"custom": True}


# --------------------------------------------------------------------------
# shell
# --------------------------------------------------------------------------


def test_shell_returns_stdout_on_success() -> None:
    result = build_executor("shell").execute({"command": "echo hello"}, timeout=5.0)
    assert result["stdout"].strip() == "hello"  # type: ignore[union-attr]
    assert result["exit_code"] == 0


def test_a_nonzero_exit_is_a_failure_carrying_stderr() -> None:
    with pytest.raises(ExecutorError) as excinfo:
        build_executor("shell").execute(
            {"command": "python3 -c 'import sys; sys.stderr.write(\"nope\"); sys.exit(3)'"},
            timeout=10.0,
        )
    assert "exit 3" in str(excinfo.value)
    assert "nope" in str(excinfo.value)


def test_shell_does_not_invoke_a_shell() -> None:
    """The injection guard: task params come from the network.

    With `shell=True` this command would run `echo hi` and then `whoami`. With
    an argv list it is a single `echo` whose arguments happen to contain
    semicolons, so the payload is printed rather than executed.
    """
    result = build_executor("shell").execute({"command": "echo hi ; whoami"}, timeout=5.0)
    assert result["stdout"].strip() == "hi ; whoami"  # type: ignore[union-attr]


def test_a_missing_command_is_reported_clearly() -> None:
    with pytest.raises(ExecutorError, match="command not found"):
        build_executor("shell").execute({"command": "definitely-not-a-real-binary"}, timeout=5.0)


def test_shell_requires_a_command_parameter() -> None:
    with pytest.raises(ExecutorError, match="requires a 'command'"):
        build_executor("shell").execute({}, timeout=5.0)


def test_a_command_that_overruns_its_timeout_fails() -> None:
    # The timeout is what stops a hung task from holding a lease its worker
    # keeps faithfully renewing -- the one thing leases cannot recover from.
    with pytest.raises(ExecutorError, match="timeout"):
        build_executor("shell").execute({"command": "sleep 5"}, timeout=0.2)


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server's required spelling
        status = 500 if self.path == "/boom" else 200
        body = b"server error" if status == 500 else b"payload"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass  # keep the test output readable


@pytest.fixture
def http_server() -> Iterator[str]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_http_returns_the_body_on_success(http_server: str) -> None:
    result = build_executor("http").execute({"url": f"{http_server}/ok"}, timeout=5.0)
    assert result["status_code"] == 200
    assert result["body"] == "payload"


def test_an_error_status_raises_with_the_status_and_body(http_server: str) -> None:
    with pytest.raises(ExecutorError) as excinfo:
        build_executor("http").execute({"url": f"{http_server}/boom"}, timeout=5.0)
    assert "500" in str(excinfo.value)
    assert "server error" in str(excinfo.value)


def test_a_connection_failure_is_an_executor_error_not_a_crash() -> None:
    with pytest.raises(ExecutorError, match="failed"):
        # Port 1 is reserved and nothing listens on it.
        build_executor("http").execute({"url": "http://127.0.0.1:1/"}, timeout=2.0)


def test_http_requires_a_url_parameter() -> None:
    with pytest.raises(ExecutorError, match="requires a 'url'"):
        build_executor("http").execute({}, timeout=5.0)


# --------------------------------------------------------------------------
# noop / fail
# --------------------------------------------------------------------------


def test_noop_echoes_its_params() -> None:
    assert build_executor("noop").execute({"a": 1}, 1.0) == {"executed": True, "params": {"a": 1}}


def test_the_failing_executor_uses_the_supplied_message() -> None:
    with pytest.raises(ExecutorError, match="custom reason"):
        build_executor("fail").execute({"message": "custom reason"}, 1.0)
