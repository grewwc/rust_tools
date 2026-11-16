import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.integrations.trace import traced_operation, traced_request
from conciliator.main import create_app
from conciliator.repositories._sync_repo import SyncTaskRepository
from conciliator.repositories.models import Base, ResolveTask, TaskEvent, TaskTraceBody, TaskTraceEvent
from conciliator.repositories.trace_repo import CHUNK_BYTES, append_event, link_task, now
from conciliator.services.trace_collector import TraceCollector
from conciliator.utils.redaction import redact, safe_text
from conciliator.utils.trace_context import context, observe_task, trace_scope


@pytest.fixture
def db(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "trace.db"), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sf = sessionmaker(engine, expire_on_commit=False)
    with sf() as session:
        for ident, version in [(1, 0), (2, 1), (3, 1)]:
            session.add(
                ResolveTask(
                    id=ident,
                    trace_schema_version=version,
                    repo_id="repo",
                    original_mr_id=ident,
                    original_mr_url="https://code.byted.org/team/repo/merge_requests/" + str(ident),
                    source_branch="feature",
                    target_branch="main",
                    resolve_branch="resolve/" + str(ident),
                    status="PENDING",
                    trigger_type="COMMENT",
                    trigger_payload={},
                    version=0,
                )
            )
        session.commit()
    yield sf
    engine.dispose()


@pytest.fixture
def client(db):
    app = create_app()
    app.state.settings = Settings()
    app.state.trace_session_factory = db
    # Do not start real clients or production worker lifespan.
    return TestClient(app)


def records(db, task_id=2):
    with db() as s:
        return list(
            s.scalars(select(TaskTraceEvent).where(TaskTraceEvent.task_id == task_id).order_by(TaskTraceEvent.id))
        )


def add(db, **overrides):
    with db.begin() as s:
        return append_event(
            s.connection(),
            dict(task_id=2, event_type="EXTERNAL_RESPONSE", **overrides),
            {"Authorization": "Bearer fake-secret", "value": "hello"},
            "response",
        )


def test_initial_status_and_old_tasks_hidden(db, client):
    assert len(records(db, 1)) == 0
    assert [(e.from_status, e.to_status) for e in records(db)] == [(None, "PENDING")]
    assert {t["id"] for t in client.get("/api/trace/tasks").json()["items"]} == {"2", "3"}
    assert client.get("/api/trace/tasks/1").status_code == 404
    assert client.get("/api/trace/events?task_id=1").status_code == 404
    with db.begin() as s:
        eid = append_event(s.connection(), dict(task_id=1, event_type="LOG"), {"old": "private"})
    assert client.get(f"/api/trace/events/{eid}").status_code == 404
    assert client.get(f"/api/trace/events/{eid}/body?kind=response").status_code == 404
    SyncTaskRepository(db).update_status(task_id=1, from_version=0, new_status=TaskStatus.FAILED)
    assert client.get("/api/trace/tasks/1").status_code == 404


def test_status_cas_and_noop_are_atomic(db):
    repo = SyncTaskRepository(db)
    assert repo.update_status(task_id=2, from_version=0, new_status=TaskStatus.PREPARING)
    assert not repo.update_status(task_id=2, from_version=0, new_status=TaskStatus.FAILED)
    assert repo.update_status(task_id=2, from_version=1, new_status=TaskStatus.PREPARING)
    assert [(e.from_status, e.to_status) for e in records(db)] == [(None, "PENDING"), ("PENDING", "PREPARING")]
    with pytest.raises(ValueError):
        repo.update_fields(task_id=2, from_version=2, status="FAILED")
    with pytest.raises(ValueError):
        repo.update_status(task_id=2, from_version=2, new_status=TaskStatus.FAILED, status="PENDING")


def test_failed_trace_rolls_back_status(db):
    engine = db.kw["bind"]

    def fail(conn, cursor, statement, parameters, ctx, many):
        if statement.startswith("INSERT INTO task_trace_events"):
            raise RuntimeError("injected trace failure")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError):
            SyncTaskRepository(db).update_status(task_id=2, from_version=0, new_status=TaskStatus.PREPARING)
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    with db() as s:
        assert s.get(ResolveTask, 2).status == "PENDING"


def test_orm_cancel_and_business_events_share_transaction(db):
    assert (
        SyncTaskRepository(db)
        .cancel_task(task_id=2, from_version=0, cancelled_by="test", webhook_event_id=None)
        .cancelled
    )
    events = records(db)
    assert len([e for e in events if e.to_status == "CANCELLED"]) == 1
    assert len([e for e in events if e.operation == "TASK_CANCELLED"]) == 1
    with db.begin() as s:
        s.add(
            TaskEvent(
                task_id=3,
                event_type="COCO_RESULT",
                payload={"session_id": "123", "prompt": "DO NOT STORE", "output": "SOURCE"},
            )
        )
    with db() as s:
        text = "".join(s.scalars(select(TaskTraceBody.content)))
    assert "DO NOT STORE" not in text and "SOURCE" not in text


def test_old_current_status_is_not_backfilled(db, client):
    eid = add(db, task_status="COCO_RUNNING")
    SyncTaskRepository(db).update_status(task_id=2, from_version=0, new_status=TaskStatus.ALL_APPROVED)
    data = client.get("/api/trace/events?current_status=ALL_APPROVED&task_status=COCO_RUNNING").json()["items"]
    assert len(data) == 1 and data[0]["id"] == str(eid)
    assert data[0]["task_status"] == "COCO_RUNNING" and data[0]["tasks"][0]["status"] == "ALL_APPROVED"


def test_pre_task_link_and_duplicate_visibility(db, client):
    with db.begin() as s:
        eid = append_event(s.connection(), dict(trace_id="pre-task", event_type="EXTERNAL_REQUEST"), {"query": "MR"})
        link_task(s.connection(), "pre-task", 2)
        link_task(s.connection(), "pre-task", 2)
        link_task(s.connection(), "pre-task", 3)
    rows = client.get("/api/trace/events?task_id=2").json()["items"]
    assert len([e for e in rows if e["id"] == str(eid)]) == 1
    detail = client.get(f"/api/trace/events/{eid}").json()
    assert detail["task_id"] is None


def test_cursor_pagination_filters_and_invalid_params(db, client):
    stamp = now()
    for _ in range(7):
        add(db, occurred_at=stamp)
    seen = []
    cursor = ""
    while True:
        result = client.get("/api/trace/events?limit=2&order=asc" + ("&cursor=" + cursor if cursor else ""))
        assert result.status_code == 200, result.text
        data = result.json()
        seen += [e["id"] for e in data["items"]]
        cursor = data["next_cursor"]
        if not cursor:
            break
        assert client.get("/api/trace/events?provider=coco&cursor=" + cursor).status_code == 422
    assert len(seen) == 9 and len(set(seen)) == 9
    for query in (
        "limit=201",
        "limit=0",
        "task_id=abc",
        "current_status=BAD",
        "cursor=notbase64",
        "start=2026-01-01T00:00:00",
    ):
        assert client.get("/api/trace/events?" + query).status_code == 422


def test_chunked_bodies_and_redaction(db, client):
    text = "中文" * 100000
    with db.begin() as s:
        eid = append_event(s.connection(), dict(task_id=2, event_type="LOG"), text, "response")
    result = ""
    chunk = 0
    while chunk is not None:
        data = client.get(f"/api/trace/events/{eid}/body?kind=response&chunk={chunk}").json()
        assert len(data["content"].encode()) <= CHUNK_BYTES
        result += data["content"]
        chunk = data["next_chunk"]
    assert result == text
    eid = add(db)
    body = client.get(f"/api/trace/events/{eid}/body?kind=response").json()["content"]
    assert "fake-secret" not in body


def test_no_body_or_task_payload_reads_in_lists(db, client):
    add(db)
    statements = []
    engine = db.kw["bind"]

    def record(conn, cursor, statement, parameters, ctx, many):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        assert client.get("/api/trace/events").status_code == 200
        assert client.get("/api/trace/tasks").status_code == 200
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert not any(
        "task_trace_event_bodies" in s or "trigger_payload" in s or "coco_result_json" in s for s in statements
    )
    assert len(statements) <= 4


def test_trace_headers_static_and_optional_access(client):
    response = client.get("/trace")
    assert response.status_code == 200 and "每一步" in response.text
    assert response.headers["cache-control"] == "private, no-store"
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert client.get("/trace/assets/trace.js").status_code == 200
    assert client.get("/trace/assets/no-file").status_code == 422
    client.app.state.settings = Settings(trace_viewers={"reader": "test-password"})
    for path in ["/trace", "/trace/assets/trace.js", "/api/trace/tasks", "/api/trace/events", "/api/trace/health"]:
        assert client.get(path).status_code == 401
        assert client.get(path, auth=("reader", "test-password")).status_code == 200
        assert client.get(path, auth=("reader", "wrong")).status_code == 401


@pytest.mark.asyncio
async def test_http_pairs_retry_errors_and_coco_no_body(db):
    collector = TraceCollector(db)
    await collector.start()
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if request.url.path == "/timeout":
            raise httpx.ReadTimeout("unknown", request=request)
        return httpx.Response(
            200, json={"Result": {"SessionId": "12345", "Prompt": "hidden input", "Output": "hidden output"}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

        @traced_operation
        async def twice():
            for _ in range(2):
                await traced_request(
                    client,
                    "coco",
                    "ListSessionEvents",
                    "POST",
                    "https://test/events",
                    json={"SessionId": "12345", "Prompt": "hidden input"},
                )

        with trace_scope(2):
            await twice()
            with pytest.raises(httpx.ReadTimeout):
                await traced_request(
                    client, "nextcode", "CreateComment", "POST", "https://test/timeout", json={"Content": "hello"}
                )
    await collector.stop()
    rows = [e for e in records(db) if e.provider]
    assert len(rows) == 6 and calls == 3
    coco = [e for e in rows if e.provider == "coco"]
    assert all(e.body_state == "EMPTY" for e in coco)
    assert len({e.operation_id for e in coco}) == 1
    assert {e.attempt for e in coco} == {1, 2}
    assert len({e.call_id for e in coco}) == 2
    assert all(e.details["SessionId"] == "12345" for e in coco)
    with db() as s:
        bodies = "".join(s.scalars(select(TaskTraceBody.content)))
    assert "hidden input" not in bodies and "hidden output" not in bodies


@pytest.mark.asyncio
async def test_interleaved_context_and_threadpool(db):
    from starlette.concurrency import run_in_threadpool

    collector = TraceCollector(db)
    await collector.start()

    async def worker(ident):
        with trace_scope():
            task = SimpleNamespace(id=ident, status="COCO_RUNNING", version=1, retry_count=0, trace_schema_version=1)
            observe_task(task)
            await asyncio.sleep(0)
            await run_in_threadpool(collector.emit, "LOG", body=f"task-{ident}", kind="raw_log")

    await asyncio.gather(worker(2), worker(3))
    assert not context.get()
    await collector.stop()
    with db() as s:
        rows = s.execute(
            select(TaskTraceEvent.task_id, TaskTraceBody.content)
            .join(TaskTraceBody, TaskTraceBody.event_id == TaskTraceEvent.id)
            .where(TaskTraceEvent.event_type == "LOG")
        ).all()
    assert set(rows) == {(2, "task-2"), (3, "task-3")}


def test_queue_capacity_and_idempotent_write(db):
    collector = TraceCollector(db, capacity=1)
    with trace_scope(2):
        collector.emit("LOG", body="first")
        collector.emit("LOG", body="second")
    assert collector.dropped == 1
    item = collector.queue.get_nowait()
    collector.write(item)
    collector.write(item)
    assert len([e for e in records(db) if e.event_type == "LOG"]) == 1


@pytest.mark.parametrize(
    "value",
    [
        {"headers": {"Authorization": "Bearer sentinel", "X-Code-App-Secret": "sentinel"}},
        "Authorization: Bearer sentinel",
        "https://test/path?access_token=sentinel&x=1",
        "mysql://name:sentinel@db/test",
        '{"nested":"{\\"password\\":\\"sentinel\\"}"}',
    ],
)
def test_redaction_nested_and_text(value):
    assert "sentinel" not in safe_text(redact(value))


@pytest.mark.asyncio
async def test_auth_and_source_payloads_are_not_retained(db):
    collector = TraceCollector(db)
    await collector.start()

    def handler(request):
        if "/auth/" in request.url.path:
            return httpx.Response(200, json={"code": 0, "tenant_access_token": "sentinel-auth-token"})
        return httpx.Response(
            200, json={"Result": {"Content": "sentinel-source-code"}, "ResponseMetadata": {"Error": {"Code": "Denied"}}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with trace_scope(2):
            await traced_request(
                client,
                "lark",
                "tenant_access_token",
                "POST",
                "https://test/auth/token",
                json={"app_id": "app", "app_secret": "sentinel-app-secret"},
            )
            await traced_request(client, "nextcode", "ListMergeRequestConflicts", "POST", "https://test/file")
    await collector.stop()
    with db() as s:
        bodies = "".join(s.scalars(select(TaskTraceBody.content)))
    assert "sentinel" not in bodies
    responses = [e for e in records(db) if e.event_type == "EXTERNAL_RESPONSE"]
    assert responses[-1].outcome == "BUSINESS_ERROR"


def test_coco_nested_session_identity():
    from conciliator.utils.redaction import coco_identity

    assert coco_identity({"Result": {"Session": {"Id": "123", "Prompt": "private"}}}) == {"SessionId": "123"}


@pytest.mark.asyncio
async def test_shared_entry_does_not_mix_child_task_calls(db, client):
    collector = TraceCollector(db)
    await collector.start()
    with trace_scope():
        collector.emit("EXTERNAL_REQUEST", body="shared list MR")
        for ident in (2, 3):
            with trace_scope(inherit_parent=True):
                collector.emit("EXTERNAL_REQUEST", body=f"private {ident}")
                observe_task(
                    SimpleNamespace(id=ident, status="PENDING", version=0, retry_count=0, trace_schema_version=1)
                )
    await collector.stop()
    with db.begin() as session:
        orphan = append_event(session.connection(), dict(trace_id="unlinked", event_type="LOG"), "private orphan")
    assert client.get(f"/api/trace/events/{orphan}").status_code == 404
    assert client.get(f"/api/trace/events/{orphan}/body?kind=response").status_code == 404
    for ident in (2, 3):
        data = client.get(f"/api/trace/events?task_id={ident}").json()["items"]
        request_ids = [e["id"] for e in data if e["event_type"] == "EXTERNAL_REQUEST"]
        texts = [client.get(f"/api/trace/events/{eid}/body?kind=response").json()["content"] for eid in request_ids]
        assert set(texts) == {"shared list MR", f"private {ident}"}


def test_create_records_marker_initial_state_and_business_event(tmp_path, monkeypatch):
    from sqlalchemy import Integer

    monkeypatch.setattr(ResolveTask.__table__.c.id, "type", Integer())
    engine = create_engine("sqlite:///" + str(tmp_path / "creation.db"))
    Base.metadata.create_all(engine)
    sf = sessionmaker(engine, expire_on_commit=False)
    task = SyncTaskRepository(sf).create(
        repo_id="r",
        original_mr_id=1,
        source_branch="a",
        target_branch="b",
        resolve_branch="r1",
        trigger_type="COMMENT",
        trigger_payload={},
        original_mr_url="https://code.byted.org/r/mr/1",
    )
    assert task.trace_schema_version == 1
    assert task.original_mr_url == "https://code.byted.org/r/mr/1"
    with sf() as s:
        assert s.scalar(select(TaskEvent.event_type)) == "TASK_CREATED"
        events = list(s.scalars(select(TaskTraceEvent)))
        assert len(events) == 2
        assert {e.event_type for e in events} == {"TASK_STATUS_CHANGED", "BUSINESS_EVENT"}
    engine.dispose()


def test_collector_keeps_nonnumeric_external_task_id_without_legacy_lookup(db):
    collector = TraceCollector(db)
    collector.write(
        (
            {
                "event_uid": "aeolus-event-1",
                "trace_id": "aeolus-trace-1",
                "task_id": "a713b57c5f4f",
                "event_type": "LOG",
                "occurred_at": now(),
            },
            None,
            "response",
            0,
        )
    )

    with db() as session:
        event = session.scalar(
            select(TaskTraceEvent).where(TaskTraceEvent.event_uid == "aeolus-event-1")
        )
        assert event is not None
        assert event.task_id is None
        assert event.details["external_task_id"] == "a713b57c5f4f"
