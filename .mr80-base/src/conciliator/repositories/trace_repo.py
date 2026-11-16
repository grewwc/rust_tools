"""Durable append-only trace storage. ORM hooks run on the business transaction."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import event, insert, inspect, select
from sqlalchemy.engine import Connection

from conciliator.repositories.models import ResolveTask, TaskEvent, TaskTraceBody, TaskTraceEvent, TaskTraceLink
from conciliator.utils.redaction import coco_identity, redact, safe_text, without_code
from conciliator.utils.trace_context import context

CHUNK_BYTES = 256 * 1024
BODY_LIMIT = 32 * 1024 * 1024


def now():
    return datetime.now(UTC).replace(tzinfo=None)


def append_event(connection: Connection, data: dict, body=None, kind="response") -> int:
    values = dict(data)
    values.setdefault("event_uid", uuid4().hex)
    values.setdefault("trace_id", context.get().get("trace_id") or uuid4().hex)
    values.setdefault("occurred_at", now())
    values.setdefault("recorded_at", now())
    values["summary"] = safe_text(values.get("summary", ""))[:1024]
    values["details"] = redact(values.get("details", {}))
    raw = safe_text(without_code(body)).encode("utf-8") if body is not None else b""
    limit = CHUNK_BYTES if kind == "raw_log" else BODY_LIMIT
    original_bytes = values.pop("original_body_bytes", len(raw))
    values.update(
        body_bytes=original_bytes, body_state="TRUNCATED" if original_bytes > limit else "COMPLETE" if raw else "EMPTY"
    )
    result = connection.execute(insert(TaskTraceEvent).values(**values))
    assert result.inserted_primary_key is not None
    event_id = result.inserted_primary_key[0]
    raw = raw[:limit]
    chunks: list[dict] = []
    while raw:
        text = raw[:CHUNK_BYTES].decode("utf-8", "ignore")
        size = len(text.encode("utf-8"))
        if not size:
            break
        chunks.append(
            {"event_id": event_id, "body_kind": kind, "chunk_index": len(chunks), "content": text, "byte_count": size}
        )
        raw = raw[size:]
    if chunks:
        connection.execute(insert(TaskTraceBody), chunks)
    return event_id


def link_task(connection, trace_id, task_id):
    if not trace_id:
        return
    # Same trace is handled sequentially; use a savepoint for duplicate links.
    from sqlalchemy.exc import IntegrityError

    try:
        with connection.begin_nested():
            connection.execute(insert(TaskTraceLink).values(trace_id=trace_id, task_id=task_id))
    except IntegrityError:
        if not connection.execute(
            select(TaskTraceLink.task_id).where(
                TaskTraceLink.trace_id == trace_id,
                TaskTraceLink.task_id == task_id,
            )
        ).first():
            raise


def status_event(connection, task_id, previous, current, version):
    enabled = connection.execute(select(ResolveTask.trace_schema_version).where(ResolveTask.id == task_id)).scalar()
    if not enabled or previous == current:
        return
    append_event(
        connection,
        dict(
            task_id=task_id,
            event_type="TASK_STATUS_CHANGED",
            task_status=current,
            from_status=previous,
            to_status=current,
            task_version=version,
            summary=f"{previous or '创建'} → {current}",
        ),
    )


@event.listens_for(ResolveTask, "after_insert")
def task_created(mapper, connection, task):
    if task.trace_schema_version >= 1:
        status_event(connection, task.id, None, task.status, task.version)
        link_task(connection, context.get().get("trace_id"), task.id)
        for parent in context.get().get("parent_trace_ids", ()):
            link_task(connection, parent, task.id)


@event.listens_for(ResolveTask, "after_update")
def task_updated(mapper, connection, task):
    history = inspect(task).attrs.status.history
    if history.has_changes():
        status_event(connection, task.id, history.deleted[0] if history.deleted else None, task.status, task.version)


@event.listens_for(TaskEvent, "after_insert")
def business_event(mapper, connection, item):
    task = connection.execute(
        select(ResolveTask.trace_schema_version, ResolveTask.status, ResolveTask.version).where(
            ResolveTask.id == item.task_id
        )
    ).first()
    if not task or task.trace_schema_version < 1:
        return
    payload = coco_identity(item.payload) if "COCO" in item.event_type else item.payload
    append_event(
        connection,
        dict(
            task_id=item.task_id,
            source_task_event_id=item.id,
            event_type="BUSINESS_EVENT",
            operation=item.event_type,
            task_status=task.status,
            task_version=task.version,
            occurred_at=item.occurred_at,
            summary=item.event_type,
        ),
        payload,
        "business",
    )
