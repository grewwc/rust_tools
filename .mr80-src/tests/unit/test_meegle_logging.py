import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.concurrency import run_in_threadpool
from structlog.contextvars import get_contextvars, merge_contextvars
from structlog.testing import capture_logs

from conciliator.api.meegle_logging import MeegleLoggingMiddleware
from conciliator.domain.aeolus_task import AeolusStartEvent, AeolusState
from conciliator.integrations.meegle.ai_node import AINodeRef, MeegleAINodeClient
from conciliator.integrations.trace import response_event, start_call
from conciliator.repositories.models import Base
from conciliator.services.aeolus_document_access import LarkAuthorizationRequired
from conciliator.services.aeolus_registry import AeolusTaskRegistry, InvalidTaskTransition
from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import logged_step, workflow_scope, workflow_step


@pytest.fixture(autouse=True)
def isolated_debug_loggers(monkeypatch):
    # Earlier app tests configure cached INFO loggers. Capture DEBUG deterministically
    # without changing the application's configured production level.
    from conciliator.api import meegle_logging as requests
    from conciliator.integrations import trace
    from conciliator.integrations.meegle import ai_node
    from conciliator.services import aeolus_registry
    from conciliator.utils import meegle_logging as steps

    original = structlog.get_config().copy()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG), cache_logger_on_first_use=False,
    )
    for module in (requests, trace, ai_node, aeolus_registry, steps):
        monkeypatch.setattr(module, "log", structlog.get_logger(module.__name__))
    yield
    structlog.configure(**original)


@pytest.mark.asyncio
async def test_concurrent_tasks_and_threadpool_have_isolated_scopes():
    arrived = 0
    ready = asyncio.Event()
    baseline = get_contextvars()

    @workflow_step("concurrent_test")
    async def execute(task_id, prompt):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            ready.set()
        await ready.wait()
        await run_in_threadpool(lambda: get_logger().info("threadpool_marker"))
        with logged_step("nested_step"):
            get_logger().info("nested_marker")

    with capture_logs(processors=[merge_contextvars]) as logs:
        await asyncio.gather(execute("task-a", "private prompt"), execute("task-b", "private prompt"))
        get_logger().info("outside_marker")
    assert get_contextvars() == baseline
    for task_id in ("task-a", "task-b"):
        events = [row for row in logs if row.get("task_id") == task_id]
        assert {"threadpool_marker", "nested_marker"} <= {row["event"] for row in events}
        assert len({row["execution_id"] for row in events}) == 1
        assert all(row["workflow"] == "meegle" for row in events)
    assert len({row["execution_id"] for row in logs if "execution_id" in row}) == 2
    assert "task_id" not in logs[-1]
    assert "private prompt" not in json.dumps(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [RuntimeError("private token=secret"), asyncio.CancelledError()])
async def test_failure_and_cancellation_are_safe_and_reset_scope(exception):
    @workflow_step("failure_test")
    async def execute(task_id):
        raise exception

    baseline = get_contextvars()
    with capture_logs(processors=[merge_contextvars]) as logs, pytest.raises(type(exception)):
        await execute("failed-task")
    assert get_contextvars() == baseline
    assert logs[-1]["event"] in {"meegle_step_failed", "meegle_step_cancelled"}
    assert logs[-1]["duration_ms"] >= 0
    assert not any(row["event"] == "meegle_step_finished" for row in logs)
    assert "secret" not in json.dumps(logs)


@pytest.mark.parametrize("method,path,expected_level,status", [
    ("GET", "/api/aeolus/tasks/task-a", "debug", 200),
    ("POST", "/api/deployment-workers/worker-a/claim", "debug", 200),
    ("POST", "/api/deployment-workers/worker-a/heartbeat", "debug", 200),
    ("POST", "/api/aeolus/tasks/task-a", "info", 200),
    ("GET", "/api/aeolus/tasks/task-a", "warning", 400),
])
def test_request_logs_use_route_template_and_omit_private_request(method, path, expected_level, status):
    app = FastAPI()
    app.add_middleware(MeegleLoggingMiddleware)

    async def handler():
        from fastapi.responses import JSONResponse
        return JSONResponse({"token": "private-response"}, status_code=status)

    route = "/api/aeolus/tasks/{task_id}" if path.startswith("/api/aeolus/") else (
        "/api/deployment-workers/{worker_id}/" + path.rsplit("/", 1)[-1]
    )
    app.add_api_route(route, handler, methods=[method])
    with capture_logs(processors=[merge_contextvars]) as logs, TestClient(app) as client:
        client.request(
            method, path + "?code=private-code&state=private-state", content="private-body",
            headers={"Authorization": "Bearer private-authorization"},
        )
    finished = [row for row in logs if row["event"] == "meegle_http_finished"]
    assert len(finished) == 1
    assert finished[0]["log_level"] == expected_level
    assert finished[0]["route"] == route
    assert finished[0]["http_status"] == status
    assert finished[0].get("task_id") == ("task-a" if path.startswith("/api/aeolus/") else None)
    assert "private-" not in json.dumps(logs)


def test_external_errors_keep_codes_and_request_id_without_payloads():
    with capture_logs(processors=[merge_contextvars]) as logs, workflow_scope(task_id="task-a"):
        fields, started = start_call("lark", "document_read", {"content": "private-prompt"})
        response_event(
            fields, started,
            httpx.Response(400, headers={"x-request-id": "platform-request"}, json={
                "code": 131006, "msg": "private-response", "access_token": "private-token",
            }),
        )
    result = logs[-1]
    assert result["http_status"] == 400 and result["code"] == 131006
    assert result["external_request_id"] == "platform-request"
    assert result["task_id"] == "task-a" and result["attempt"] == 1
    assert result["log_level"] == "warning"
    assert "private-" not in json.dumps(logs)


@pytest.mark.asyncio
async def test_meegle_calls_log_business_errors_and_never_tokens():
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.path.endswith("plugin_token"):
            return httpx.Response(200, json={"data": {"token": "private-token"}, "error": {"code": 0}})
        return httpx.Response(400, headers={"x-tt-logid": "meegle-request"}, json={
            "error": {"code": 10204, "msg": "private-secret"},
        })

    client = MeegleAINodeClient(site_url="https://example.com", plugin_id="plugin", plugin_secret="private-secret")
    await client._http.aclose()
    client._http = httpx.AsyncClient(base_url="https://example.com", transport=httpx.MockTransport(respond))
    try:
        with (
            capture_logs(processors=[merge_contextvars]) as logs,
            workflow_scope(task_id="task-a"), pytest.raises(RuntimeError),
        ):
            await client.confirm(AINodeRef("space", 1, "node", "user"), idem_key="private-idempotency")
    finally:
        await client.aclose()
    assert len(calls) == 2
    assert logs[-1]["code"] == 10204 and logs[-1]["external_request_id"] == "meegle-request"
    assert logs[-1]["task_id"] == "task-a"
    assert "private-" not in json.dumps(logs)


@pytest.mark.asyncio
async def test_frozen_authorization_exception_is_preserved():
    exception = LarkAuthorizationRequired("user-a")

    @workflow_step("authorization_test")
    async def execute(task_id):
        with logged_step("read_document"):
            raise exception

    baseline = get_contextvars()
    with capture_logs(processors=[merge_contextvars]) as logs, pytest.raises(LarkAuthorizationRequired) as caught:
        await execute("task-a")
    assert caught.value is exception
    assert get_contextvars() == baseline
    assert [row["error_type"] for row in logs if row["event"] == "meegle_step_failed"] == [
        "LarkAuthorizationRequired", "LarkAuthorizationRequired",
    ]


@pytest.mark.asyncio
async def test_only_committed_transitions_log_success(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    registry = AeolusTaskRegistry(sessionmaker(engine, expire_on_commit=False))
    event = AeolusStartEvent("task-a", "space", 1, "state", "node", "user", ())
    try:
        await registry.create_from_event(event)
        with capture_logs(processors=[merge_contextvars]) as logs:
            await registry.transition("task-a", {AeolusState.RECEIVED}, AeolusState.READING_DOCUMENT)
            with pytest.raises(InvalidTaskTransition):
                await registry.transition("task-a", {AeolusState.RECEIVED}, AeolusState.EVALUATING)
        assert len(logs) == 1 and logs[0]["previous_state"] == "received"
        assert logs[0]["state"] == "reading_document"

        def fail_commit(self):
            raise RuntimeError("private-db-error")

        with monkeypatch.context() as patch:
            patch.setattr(Session, "commit", fail_commit)
            with capture_logs(processors=[merge_contextvars]) as failed, pytest.raises(RuntimeError):
                await registry.transition("task-a", {AeolusState.READING_DOCUMENT}, AeolusState.EVALUATING)
        assert not failed
        assert (await registry.get("task-a")).state == AeolusState.READING_DOCUMENT
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_task_argument_does_not_log_document_or_feedback():
    @workflow_step("task_test")
    async def execute(task, feedback):
        return task

    task = SimpleNamespace(
        task_id="task-a", state=AeolusState.REVIEW, action_token="private-token", requirement_text="private-document",
    )
    with capture_logs(processors=[merge_contextvars]) as logs:
        assert await execute(task, "private-feedback") is task
    assert all(row["task_id"] == "task-a" for row in logs)
    assert "private-" not in json.dumps(logs)
