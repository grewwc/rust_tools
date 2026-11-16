"""Read-only task timeline API. List queries never join the large body table."""

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.orm import load_only

from conciliator.domain.task import TaskStatus
from conciliator.repositories.models import ResolveTask as Task
from conciliator.repositories.models import TaskTraceBody as Body
from conciliator.repositories.models import TaskTraceEvent as Event
from conciliator.repositories.models import TaskTraceLink as Link
from conciliator.utils.redaction import redact

router = APIRouter()
WEB = Path(__file__).resolve().parents[1] / "web" / "trace"


def access(request: Request):
    # Authentication dependency is attached here so HTML, assets and JSON share one boundary.
    from conciliator.api.trace_auth import require_viewer

    return require_viewer(request)


class Filters(BaseModel):
    task_id: int | None = Field(None, ge=1)
    current_status: TaskStatus | None = None
    task_status: TaskStatus | None = None
    repo_id: str | None = Field(None, max_length=128)
    original_mr_id: int | None = Field(None, ge=1)
    new_mr_id: int | None = Field(None, ge=1)
    original_mr_url: str | None = Field(None, max_length=512)
    event_type: str | None = Field(None, max_length=64)
    provider: Literal["nextcode", "coco", "lark"] | None = None
    operation: str | None = Field(None, max_length=128)
    outcome: Literal["SUCCESS", "HTTP_ERROR", "BUSINESS_ERROR", "UNKNOWN"] | None = None
    call_id: str | None = Field(None, max_length=64)
    start: datetime | None = None
    end: datetime | None = None
    order: Literal["asc", "desc"] = "desc"
    limit: int = Field(50, ge=1, le=200)
    cursor: str | None = Field(None, max_length=2048)

    @model_validator(mode="after")
    def dates(self):
        for name in ("start", "end"):
            value = getattr(self, name)
            if value:
                if value.tzinfo is None:
                    raise ValueError("time must include a timezone")
                setattr(self, name, value.astimezone(UTC).replace(tzinfo=None))
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must precede end")
        return self


def task_conditions(f):
    terms = [Task.trace_schema_version >= 1]
    for name in ("repo_id", "original_mr_id", "new_mr_id", "original_mr_url"):
        if getattr(f, name) is not None:
            terms.append(getattr(Task, name) == getattr(f, name))
    if f.task_id:
        terms.append(Task.id == f.task_id)
    if f.current_status:
        terms.append(Task.status == f.current_status.value)
    return terms


def association():
    return or_(
        Event.task_id == Task.id,
        and_(
            Event.task_id.is_(None),
            exists()
            .where(
                Link.trace_id == Event.trace_id,
                Link.task_id == Task.id,
            )
            .correlate(Event, Task),
        ),
    )


def visible_event(event_id):
    return select(Event).where(
        Event.id == event_id,
        exists().where(
            Task.trace_schema_version >= 1,
            association(),
        ),
    )


def serialize(obj):
    result = {}
    for column in obj.__table__.columns:
        value = getattr(obj, column.key)
        if isinstance(value, datetime):
            value = value.replace(tzinfo=UTC).isoformat(timespec="milliseconds")
        elif value is not None and (column.key == "id" or column.key.endswith("_id")):
            value = str(value)
        result[column.key] = value
    return redact(result)


TASK_FIELDS = (
    "id",
    "repo_id",
    "original_mr_id",
    "original_mr_url",
    "new_mr_id",
    "new_mr_url",
    "status",
    "coco_session_id",
    "source_branch",
    "target_branch",
    "created_at",
    "updated_at",
    "retry_count",
    "version",
    "trace_schema_version",
)


def task_select():
    return select(Task).options(load_only(*(getattr(Task, key) for key in TASK_FIELDS), raiseload=True))


def task_snapshot(task):
    result = {}
    for key in TASK_FIELDS:
        value = getattr(task, key)
        if isinstance(value, datetime):
            value = value.replace(tzinfo=UTC).isoformat(timespec="milliseconds")
        elif value is not None and (key == "id" or key.endswith("_id")):
            value = str(value)
        result[key] = value
    return redact(result)


def cursor_hash(f, kind):
    raw = f.model_dump(mode="json", exclude={"cursor", "limit"})
    return hashlib.sha256(json.dumps([kind, raw], sort_keys=True).encode()).hexdigest()[:24]


def paginate(stmt, time_col, id_col, f, kind):
    if f.cursor:
        try:
            data = json.loads(base64.urlsafe_b64decode(f.cursor.encode()))
            if data["filter"] != cursor_hash(f, kind):
                raise ValueError()
            time = datetime.fromisoformat(data["time"])
            ident = int(data["id"])
            if f.order == "desc":
                stmt = stmt.where(or_(time_col < time, and_(time_col == time, id_col < ident)))
            else:
                stmt = stmt.where(or_(time_col > time, and_(time_col == time, id_col > ident)))
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            raise HTTPException(422, "游标无效或与筛选条件不匹配") from exc
    return stmt.order_by(
        time_col.desc() if f.order == "desc" else time_col.asc(), id_col.desc() if f.order == "desc" else id_col.asc()
    ).limit(f.limit + 1)


def next_cursor(rows, f, kind, time_key):
    if len(rows) <= f.limit:
        return None
    last = rows[f.limit - 1]
    return base64.urlsafe_b64encode(
        json.dumps(
            dict(filter=cursor_hash(f, kind), time=getattr(last, time_key).isoformat(), id=str(last.id))
        ).encode()
    ).decode()


@router.get("/trace", dependencies=[Depends(access)])
def page():
    return FileResponse(WEB / "index.html")


@router.get("/trace/assets/{filename}", dependencies=[Depends(access)])
def asset(filename: Literal["trace.js", "trace.css"]):
    return FileResponse(WEB / filename)


@router.get("/api/trace/options", dependencies=[Depends(access)])
def options():
    return {
        "statuses": [s.value for s in TaskStatus],
        "event_types": [
            "TASK_STATUS_CHANGED",
            "EXTERNAL_REQUEST",
            "EXTERNAL_RESPONSE",
            "EXTERNAL_ERROR",
            "EXTERNAL_STREAM_EVENT",
            "EXTERNAL_STREAM_END",
            "BUSINESS_EVENT",
            "LOG",
            "TASK_CONTEXT",
        ],
    }


@router.get("/api/trace/tasks", dependencies=[Depends(access)])
def tasks(request: Request, f: Annotated[Filters, Query()]):
    with request.app.state.trace_session_factory() as session:
        stmt = task_select().where(*task_conditions(f))
        if f.start:
            stmt = stmt.where(Task.created_at >= f.start)
        if f.end:
            stmt = stmt.where(Task.created_at <= f.end)
        rows = list(session.scalars(paginate(stmt, Task.created_at, Task.id, f, "tasks")))
        return {
            "items": [task_snapshot(t) for t in rows[: f.limit]],
            "next_cursor": next_cursor(rows, f, "tasks", "created_at"),
        }


@router.get("/api/trace/tasks/{task_id}", dependencies=[Depends(access)])
def task(request: Request, task_id: int):
    with request.app.state.trace_session_factory() as session:
        item = session.scalar(task_select().where(Task.id == task_id, Task.trace_schema_version >= 1))
        if item is None:
            raise HTTPException(404, "任务未收录")
        return task_snapshot(item)


@router.get("/api/trace/events", dependencies=[Depends(access)])
def events(request: Request, f: Annotated[Filters, Query()]):
    with request.app.state.trace_session_factory() as session:
        if f.task_id and not session.scalar(
            select(Task.id).where(Task.id == f.task_id, Task.trace_schema_version >= 1)
        ):
            raise HTTPException(404, "任务未收录")
        eligible = exists().where(*task_conditions(f), association())
        stmt = select(Event).where(eligible)
        for name in ("task_status", "provider", "event_type", "operation", "outcome", "call_id"):
            value = getattr(f, name)
            if value is not None:
                stmt = stmt.where(getattr(Event, name) == value)
        if f.start:
            stmt = stmt.where(Event.occurred_at >= f.start)
        if f.end:
            stmt = stmt.where(Event.occurred_at <= f.end)
        rows = list(session.scalars(paginate(stmt, Event.occurred_at, Event.id, f, "events")))
        items = rows[: f.limit]
        # Batch associated task snapshots; no body reads, per-event queries, or global counts.
        direct = {e.task_id for e in items if e.task_id is not None}
        traces = {e.trace_id for e in items if e.task_id is None}
        linked = (
            session.execute(select(Link.trace_id, Link.task_id).where(Link.trace_id.in_(traces))).all()
            if traces
            else []
        )
        ids = direct | {row.task_id for row in linked}
        snapshots = {
            t.id: task_snapshot(t)
            for t in session.scalars(task_select().where(Task.id.in_(ids), Task.trace_schema_version >= 1))
        }
        result = []
        for e in items:
            associated = [e.task_id] if e.task_id else [r.task_id for r in linked if r.trace_id == e.trace_id]
            result.append({**serialize(e), "tasks": [snapshots[i] for i in associated if i in snapshots]})
        return {"items": result, "next_cursor": next_cursor(rows, f, "events", "occurred_at")}


@router.get("/api/trace/events/{event_id}", dependencies=[Depends(access)])
def event_detail(request: Request, event_id: int):
    with request.app.state.trace_session_factory() as session:
        item = session.scalar(visible_event(event_id))
        if item is None:
            raise HTTPException(404, "明细未收录")
        directory = session.execute(
            select(Body.body_kind, func.count(), func.sum(Body.byte_count))
            .where(Body.event_id == event_id)
            .group_by(Body.body_kind)
        ).all()
        return {**serialize(item), "bodies": [dict(kind=k, chunks=n, bytes=b) for k, n, b in directory]}


@router.get("/api/trace/events/{event_id}/body", dependencies=[Depends(access)])
def body(request: Request, event_id: int, kind: str = Query(max_length=32), chunk: int = Query(0, ge=0)):
    with request.app.state.trace_session_factory() as session:
        if session.scalar(visible_event(event_id)) is None:
            raise HTTPException(404, "明细未收录")
        rows = list(
            session.scalars(
                select(Body)
                .where(Body.event_id == event_id, Body.body_kind == kind, Body.chunk_index >= chunk)
                .order_by(Body.chunk_index)
                .limit(2)
            )
        )
        if not rows:
            raise HTTPException(404, "正文不存在")
        return {"content": rows[0].content or "", "next_chunk": rows[1].chunk_index if len(rows) > 1 else None}


@router.get("/api/trace/health", dependencies=[Depends(access)])
def health(request: Request):
    with request.app.state.trace_session_factory() as session:
        latest = select(func.max(Event.id)).where(Event.event_type == "COLLECTOR_HEARTBEAT").group_by(Event.trace_id)
        rows = session.scalars(select(Event).where(Event.id.in_(latest)).order_by(Event.id.desc()).limit(100)).all()
        threshold = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=150)
        return {
            "instances": [
                {
                    "id": e.trace_id,
                    "last_seen": serialize(e)["occurred_at"],
                    **e.details,
                    "stale": e.occurred_at.replace(tzinfo=None) < threshold,
                }
                for e in rows
            ],
            "note": "请求与数据库不能原子提交；进程崩溃或采集故障可能造成缺失。刷新可查迟到记录。",
        }
