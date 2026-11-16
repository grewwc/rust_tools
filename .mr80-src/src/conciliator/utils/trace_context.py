"""Immutable request/worker context, propagated by AnyIO into repository threads."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from types import MappingProxyType
from uuid import uuid4

context: ContextVar[Mapping] = ContextVar("task_trace", default=MappingProxyType({}))


@contextmanager
def trace_scope(task_id=None, *, inherit_parent=False):
    parent = context.get()
    ancestors = tuple(parent.get("parent_trace_ids", ())) + ((parent["trace_id"],) if parent else ())
    token = context.set(
        {
            "trace_id": uuid4().hex,
            "execution_id": uuid4().hex,
            "task_id": task_id,
            "parent_trace_ids": ancestors if inherit_parent else (),
        }
    )
    try:
        yield
    finally:
        context.reset(token)


def traced_work(fn):
    sig = signature(fn)

    @wraps(fn)
    async def wrapped(*args, **kwargs):
        bound = sig.bind(*args, **kwargs).arguments
        task_id = bound.get("task_id")
        notification = bound.get("notification")
        if notification is not None:
            task_id = notification.task_id
        if context.get() and task_id in (None, context.get().get("task_id")):
            return await fn(*args, **kwargs)
        with trace_scope(task_id):
            return await fn(*args, **kwargs)

    return wrapped


def observe_task(task):
    if task is not None and context.get():
        previous_id = context.get().get("task_id")
        context.set(
            {
                **context.get(),
                "task_id": task.id,
                "task_status": task.status,
                "task_version": task.version,
                "retry_round": getattr(task, "retry_count", 0),
                "trace_enabled": getattr(task, "trace_schema_version", 0) >= 1,
            }
        )
        if previous_id != task.id:
            from conciliator.services.trace_collector import emit

            emit("TASK_CONTEXT", summary="关联任务")
    return task


def traced_child_work(fn):
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        with trace_scope(inherit_parent=True):
            return await fn(*args, **kwargs)

    return wrapped
