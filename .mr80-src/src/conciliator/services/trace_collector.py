"""Bounded, thread-safe trace queue. Failure never repeats a business HTTP call."""

import asyncio
import contextlib
import sys
from queue import Empty, Full, Queue
from threading import Lock
from uuid import uuid4

from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from conciliator.repositories.models import ResolveTask, TaskTraceEvent
from conciliator.repositories.trace_repo import append_event, link_task
from conciliator.utils.redaction import redact, safe_text, without_code
from conciliator.utils.trace_context import context

_active = None


def _legacy_task_id(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


class TraceCollector:
    def __init__(self, session_factory, capacity=1000):
        self.sf = session_factory
        self.queue: Queue = Queue(maxsize=capacity)
        self.dropped = 0
        self.errors = 0
        self.instance = uuid4().hex
        self.worker = None
        self.stopping = False
        self.queued_bytes = 0
        self.byte_limit = 64 * 1024 * 1024
        self.lock = Lock()

    async def start(self):
        global _active
        _active = self
        self.worker = asyncio.create_task(self.run())

    def emit(self, event_type, *, body=None, kind="response", **fields):
        ctx = context.get()
        if (not ctx and not fields.get("task_id")) or ctx.get("trace_enabled") is False:
            return
        data = {
            k: v
            for k, v in ctx.items()
            if k
            in {
                "trace_id",
                "task_id",
                "task_status",
                "task_version",
                "execution_id",
                "retry_round",
            }
        }
        data.update(fields, event_type=event_type, event_uid=uuid4().hex)
        data["_parent_trace_ids"] = list(ctx.get("parent_trace_ids", ()))
        from conciliator.repositories.trace_repo import now

        data["occurred_at"] = now()
        data.setdefault("trace_id", uuid4().hex)
        raw = safe_text(without_code(body)).encode("utf-8") if body is not None else b""
        data["original_body_bytes"] = len(raw)
        limit = 256 * 1024 if kind == "raw_log" else 32 * 1024 * 1024
        body_text = raw[:limit].decode("utf-8", "ignore")
        size = len(body_text.encode("utf-8"))
        try:
            # Redact before retaining anything in the queue, including exceptions.
            with self.lock:
                if self.queued_bytes + size > self.byte_limit:
                    raise Full()
                self.queue.put_nowait((data, body_text if body is not None else None, kind, size))
                self.queued_bytes += size
        except Full:
            self.dropped += 1
            print("trace_queue_full: task history may be incomplete", file=sys.stderr)

    def write(self, item):
        queued_data, body, kind, _size = item
        data = dict(queued_data)
        parents = data.pop("_parent_trace_ids", ())
        with self.sf() as session:
            if session.scalar(select(TaskTraceEvent.id).where(TaskTraceEvent.event_uid == data["event_uid"])):
                return
            task_ref = data.get("task_id")
            task_id = _legacy_task_id(task_ref)
            if task_ref is not None and task_id is None:
                data.pop("task_id", None)
                details = dict(data.get("details") or {})
                details["external_task_id"] = safe_text(task_ref)[:128]
                data["details"] = details
            if task_id is not None:
                data["task_id"] = task_id
                enabled = session.scalar(select(ResolveTask.trace_schema_version).where(ResolveTask.id == task_id))
                if not enabled:
                    return
                link_task(session.connection(), data.get("trace_id"), task_id)
                for parent in parents:
                    link_task(session.connection(), parent, task_id)
            append_event(session.connection(), data, body, kind)
            session.commit()

    def heartbeat(self):
        with self.sf() as session:
            append_event(
                session.connection(),
                dict(
                    trace_id=self.instance,
                    event_type="COLLECTOR_HEARTBEAT",
                    summary="采集进程心跳",
                    details={
                        "dropped": self.dropped,
                        "errors": self.errors,
                        "queued": self.queue.qsize(),
                        "stopping": self.stopping,
                    },
                ),
            )
            session.commit()

    async def run(self):
        next_heartbeat = 0.0
        while not self.stopping or not self.queue.empty():
            if asyncio.get_running_loop().time() >= next_heartbeat:
                try:
                    await run_in_threadpool(self.heartbeat)
                except Exception:
                    self.errors += 1
                next_heartbeat = asyncio.get_running_loop().time() + 60
            try:
                item = self.queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.1)
                continue
            for attempt in range(2):
                try:
                    await run_in_threadpool(self.write, item)
                    break
                except Exception:
                    self.errors += 1
                    if attempt:
                        self.dropped += 1
                        print("trace_write_failed: task history may be incomplete", file=sys.stderr)
                    else:
                        await asyncio.sleep(0.1)
            with self.lock:
                self.queued_bytes -= item[3]
            self.queue.task_done()

    async def stop(self):
        global _active
        _active = None
        self.stopping = True
        if self.worker:
            try:
                await asyncio.wait_for(self.worker, 10)
            except TimeoutError:
                self.dropped += self.queue.qsize()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(run_in_threadpool(self.heartbeat), 2)


def emit(event_type, **fields):
    if _active:
        try:
            _active.emit(event_type, **fields)
        except Exception:
            _active.errors += 1
            _active.dropped += 1


def trace_log(logger, method_name, event_dict):
    clean = redact(event_dict)
    ctx = context.get()
    if ctx or event_dict.get("task_id"):
        # Worker recovery logs often carry an explicit task_id before entering a scope.
        fields = {"task_id": event_dict["task_id"]} if event_dict.get("task_id") else {}
        emit("LOG", body=clean, kind="raw_log", summary=str(clean.get("event", "")), operation=method_name, **fields)
    # ``trace_id`` lives in this module's ContextVar rather than structlog's,
    # so it has to be copied into the event to be searchable in Argos. The
    # explicit values already on the event win; the stored trace body above
    # keeps its original shape.
    extra = {
        key: value
        for key, value in (
            ("trace_id", ctx.get("trace_id") if ctx else None),
            ("task_id", ctx.get("task_id") if ctx else None),
        )
        if value and not clean.get(key)
    }
    return {**clean, **extra} if extra else clean
