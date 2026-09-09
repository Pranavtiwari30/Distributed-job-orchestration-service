"""End-to-end tests for the REST API against a real database.

These exercise the contract a client actually sees: status codes, error bodies,
pagination, and idempotency. They deliberately do not mock the repository layer
-- the interesting failures (a cycle rejected at 422, a duplicate idempotency
key at 409) only happen when real SQL runs.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conductor.api.app import API_PREFIX, create_app
from conductor.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def app(settings: Settings, engine: object) -> FastAPI:  # noqa: ARG001 - schema ordering
    # No dependency override needed: `create_app` stores these on app.state and
    # every handler resolves them from there, so the app genuinely runs against
    # the test database rather than whatever the environment happens to say.
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def etl_body(name: str = "etl") -> dict[str, object]:
    return {
        "name": name,
        "description": "extract, transform, load",
        "tasks": [
            {"key": "extract", "executor": "http"},
            {"key": "transform", "executor": "shell", "depends_on": ["extract"]},
            {"key": "load", "executor": "shell", "depends_on": ["transform"]},
        ],
    }


# --------------------------------------------------------------------------
# workflows
# --------------------------------------------------------------------------


def test_creating_a_workflow_returns_201_and_the_resolved_graph(client: TestClient) -> None:
    response = client.post(f"{API_PREFIX}/workflows", json=etl_body())
    assert response.status_code == 201

    body = response.json()
    assert body["version"] == 1
    assert [task["key"] for task in body["tasks"]] == ["extract", "load", "transform"]
    assert next(t for t in body["tasks"] if t["key"] == "load")["depends_on"] == ["transform"]


def test_republishing_a_name_creates_a_new_version(client: TestClient) -> None:
    assert client.post(f"{API_PREFIX}/workflows", json=etl_body()).json()["version"] == 1
    assert client.post(f"{API_PREFIX}/workflows", json=etl_body()).json()["version"] == 2


def test_a_cyclic_workflow_is_rejected_with_the_cycle_named(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/workflows",
        json={
            "name": "circular",
            "tasks": [
                {"key": "a", "executor": "noop", "depends_on": ["c"]},
                {"key": "b", "executor": "noop", "depends_on": ["a"]},
                {"key": "c", "executor": "noop", "depends_on": ["b"]},
            ],
        },
    )
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")

    body = response.json()
    assert body["type"].endswith("/workflow-cycle")
    assert set(body["cycle"]) == {"a", "b", "c"}, "the client is told which edge to remove"


def test_a_dependency_on_an_unknown_task_is_rejected(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/workflows",
        json={"name": "typo", "tasks": [{"key": "a", "executor": "noop", "depends_on": ["b"]}]},
    )
    assert response.status_code == 422
    assert "unknown tasks" in response.json()["detail"]


def test_field_validation_errors_use_the_same_problem_shape(client: TestClient) -> None:
    response = client.post(f"{API_PREFIX}/workflows", json={"name": "", "tasks": []})
    assert response.status_code == 422
    body = response.json()
    assert body["type"].endswith("/validation-failed")
    assert body["errors"], "the offending fields are enumerated"


def test_workflow_listing_paginates_by_cursor(client: TestClient) -> None:
    for index in range(5):
        client.post(f"{API_PREFIX}/workflows", json=etl_body(f"wf-{index}"))

    first = client.get(f"{API_PREFIX}/workflows", params={"limit": 2}).json()
    assert len(first["items"]) == 2
    assert first["meta"]["has_more"] is True

    second = client.get(
        f"{API_PREFIX}/workflows", params={"limit": 2, "cursor": first["meta"]["next_cursor"]}
    ).json()
    assert len(second["items"]) == 2

    first_ids = {item["id"] for item in first["items"]}
    second_ids = {item["id"] for item in second["items"]}
    assert first_ids.isdisjoint(second_ids), "pages must not overlap"


def test_a_malformed_cursor_is_a_422_not_a_500(client: TestClient) -> None:
    response = client.get(f"{API_PREFIX}/workflows", params={"cursor": "not-base64!!"})
    assert response.status_code == 422
    assert "malformed cursor" in response.json()["detail"]


def test_an_unknown_workflow_id_is_404(client: TestClient) -> None:
    response = client.get(f"{API_PREFIX}/workflows/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["type"].endswith("/not-found")


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------


def test_submitting_a_run_materialises_every_task(client: TestClient) -> None:
    client.post(f"{API_PREFIX}/workflows", json=etl_body())
    response = client.post(f"{API_PREFIX}/runs", json={"workflow_name": "etl"})
    assert response.status_code == 201

    body = response.json()
    assert body["state"] == "PENDING"
    states = {task["key"]: task["state"] for task in body["tasks"]}
    assert states == {"extract": "READY", "transform": "PENDING", "load": "PENDING"}


def test_a_run_can_pin_a_workflow_version(client: TestClient) -> None:
    client.post(f"{API_PREFIX}/workflows", json=etl_body())
    v1_id = client.get(f"{API_PREFIX}/workflows").json()["items"][0]["id"]
    client.post(f"{API_PREFIX}/workflows", json=etl_body())

    pinned = client.post(
        f"{API_PREFIX}/runs", json={"workflow_name": "etl", "workflow_version": 1}
    ).json()
    assert pinned["workflow_id"] == v1_id


def test_a_run_against_an_unknown_workflow_is_404(client: TestClient) -> None:
    response = client.post(f"{API_PREFIX}/runs", json={"workflow_name": "nope"})
    assert response.status_code == 404


def test_an_idempotency_key_makes_a_retried_post_safe(client: TestClient) -> None:
    client.post(f"{API_PREFIX}/workflows", json=etl_body())
    headers = {"Idempotency-Key": "client-generated-uuid"}

    first = client.post(f"{API_PREFIX}/runs", json={"workflow_name": "etl"}, headers=headers)
    second = client.post(f"{API_PREFIX}/runs", json={"workflow_name": "etl"}, headers=headers)

    assert first.status_code == 201
    assert second.status_code == 200, "the replay is not a creation"
    assert first.json()["id"] == second.json()["id"], "one run, not two"


def test_reusing_an_idempotency_key_with_a_different_body_is_409(client: TestClient) -> None:
    # Silently returning the first run would hide a real client bug.
    client.post(f"{API_PREFIX}/workflows", json=etl_body("one"))
    client.post(f"{API_PREFIX}/workflows", json=etl_body("two"))
    headers = {"Idempotency-Key": "reused"}

    client.post(f"{API_PREFIX}/runs", json={"workflow_name": "one"}, headers=headers)
    response = client.post(f"{API_PREFIX}/runs", json={"workflow_name": "two"}, headers=headers)

    assert response.status_code == 409
    assert response.json()["type"].endswith("/idempotency-key-reused")


def test_idempotency_ignores_key_ordering_in_the_body(client: TestClient) -> None:
    client.post(f"{API_PREFIX}/workflows", json=etl_body())
    headers = {"Idempotency-Key": "ordering"}

    first = client.post(
        f"{API_PREFIX}/runs",
        json={"workflow_name": "etl", "payload": {"a": 1, "b": 2}},
        headers=headers,
    )
    second = client.post(
        f"{API_PREFIX}/runs",
        json={"payload": {"b": 2, "a": 1}, "workflow_name": "etl"},
        headers=headers,
    )
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]


def test_runs_can_be_filtered_by_state(client: TestClient) -> None:
    client.post(f"{API_PREFIX}/workflows", json=etl_body())
    for _ in range(3):
        client.post(f"{API_PREFIX}/runs", json={"workflow_name": "etl"})

    pending = client.get(f"{API_PREFIX}/runs", params={"state": "PENDING"}).json()
    succeeded = client.get(f"{API_PREFIX}/runs", params={"state": "SUCCEEDED"}).json()
    assert len(pending["items"]) == 3
    assert succeeded["items"] == []


def test_cancelling_a_run_terminates_every_task(client: TestClient) -> None:
    client.post(f"{API_PREFIX}/workflows", json=etl_body())
    run_id = client.post(f"{API_PREFIX}/runs", json={"workflow_name": "etl"}).json()["id"]

    response = client.post(f"{API_PREFIX}/runs/{run_id}/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "CANCELLED"
    assert {task["state"] for task in body["tasks"]} == {"CANCELLED"}


def test_cancelling_an_already_finished_run_is_409(client: TestClient) -> None:
    client.post(f"{API_PREFIX}/workflows", json=etl_body())
    run_id = client.post(f"{API_PREFIX}/runs", json={"workflow_name": "etl"}).json()["id"]
    client.post(f"{API_PREFIX}/runs/{run_id}/cancel")

    response = client.post(f"{API_PREFIX}/runs/{run_id}/cancel")
    assert response.status_code == 409
    assert response.json()["type"].endswith("/run-not-cancellable")


# --------------------------------------------------------------------------
# cross-cutting
# --------------------------------------------------------------------------


def test_health_and_readiness_are_separate_endpoints(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    readiness = client.get("/readyz").json()
    assert readiness["status"] == "ok"
    assert readiness["database"] == "up"


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/healthz", headers={"X-Request-ID": "trace-me"})
    assert response.headers["X-Request-ID"] == "trace-me", "a caller-supplied id is preserved"
    assert client.get("/healthz").headers["X-Request-ID"], "one is generated otherwise"


def test_the_rate_limiter_returns_429_with_retry_after(app: FastAPI) -> None:
    limited = create_app(
        Settings(
            database_url=app.state.settings.database_url,
            api_rate_limit=3,
            api_rate_refill=0.001,
        )
    )
    with TestClient(limited) as client:
        for _ in range(3):
            assert client.get(f"{API_PREFIX}/workflows").status_code == 200

        response = client.get(f"{API_PREFIX}/workflows")
        assert response.status_code == 429
        assert response.json()["type"].endswith("/rate-limited")
        assert float(response.json()["retry_after"]) > 0


def test_the_openapi_document_is_generated(client: TestClient) -> None:
    schema = client.get(f"{API_PREFIX}/openapi.json").json()
    assert schema["info"]["title"] == "Conductor"
    assert f"{API_PREFIX}/runs" in schema["paths"]


def test_pagination_is_stable_when_rows_share_a_timestamp(
    client: TestClient, session: object
) -> None:
    """Regression test for a keyset-pagination bug.

    The cursor predicate was written as `(created_at, id) < (t, i)`, which is a
    *Python* tuple comparison, not SQL: Python evaluates `created_at == t`
    first, SQLAlchemy's `__bool__` returns False for that expression, and the
    whole thing collapses to `created_at < t` -- silently discarding the `id`
    tiebreaker. It passed every test that happened to produce distinct
    timestamps.

    Forcing every row to share one `created_at` is what exposes it: without a
    tiebreaker the second page either repeats or skips rows.
    """
    from sqlalchemy import text as sql_text

    for index in range(6):
        client.post(f"{API_PREFIX}/workflows", json=etl_body(f"same-time-{index}"))

    session.execute(sql_text("UPDATE workflows SET created_at = '2026-01-01T00:00:00Z'"))  # type: ignore[attr-defined]
    session.commit()  # type: ignore[attr-defined]

    seen: list[str] = []
    cursor: str | None = None
    for _ in range(10):
        params: dict[str, object] = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        page = client.get(f"{API_PREFIX}/workflows", params=params).json()
        seen.extend(item["id"] for item in page["items"])
        cursor = page["meta"]["next_cursor"]
        if not cursor:
            break

    assert len(seen) == 6, f"expected every row exactly once, walked {len(seen)}"
    assert len(set(seen)) == 6, "a row was returned on more than one page"
