"""Safe, scoped diagnostics for the Meegle workflow, without retaining payloads."""

import asyncio
import time
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any
from uuid import uuid4

from structlog.contextvars import bind_contextvars, get_contextvars, reset_contextvars

from conciliator.utils.logging import get_logger

log = get_logger(__name__)


def error_fields(exc: BaseException) -> dict:
    """Keep error identity and its origin; exception messages can contain secrets."""
    fields: dict[str, Any] = {"error_type": type(exc).__name__}
    for key in ("code", "http_status", "retryable", "uncertain"):
        value = getattr(exc, key, None)
        if isinstance(value, (int, bool)):
            fields[key] = value
    tb = exc.__traceback__
    if tb is not None:
        while tb.tb_next:
            tb = tb.tb_next
        fields["error_origin"] = (
            f"{Path(tb.tb_frame.f_code.co_filename).name}:{tb.tb_lineno}:{tb.tb_frame.f_code.co_name}"
        )
    return fields


class workflow_scope:
    # Generator context managers assign __traceback__ on exit. Frozen domain
    # exceptions reject that assignment, so restore tokens without touching them.
    def __init__(self, **fields):
        self.fields = fields

    def __enter__(self):
        self.tokens = bind_contextvars(workflow="meegle", **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb):
        reset_contextvars(**self.tokens)
        return False


class logged_step(workflow_scope):
    """A successful step means the call returned, not that the task succeeded."""

    def __init__(self, step: str, *, level: str = "info", **fields):
        super().__init__(step=step, step_id=uuid4().hex, **fields)
        self.write = getattr(log, level)

    def __enter__(self):
        super().__enter__()
        self.started = time.monotonic()
        self.write("meegle_step_started")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            duration = int((time.monotonic() - self.started) * 1000)
            if isinstance(exc, asyncio.CancelledError):
                log.info("meegle_step_cancelled", duration_ms=duration)
            elif exc is not None:
                log.warning("meegle_step_failed", duration_ms=duration, **error_fields(exc))
            else:
                self.write("meegle_step_finished", duration_ms=duration)
        finally:
            super().__exit__(exc_type, exc, tb)
        return False


def workflow_step(step: str, *, level: str = "info"):
    """Log only explicitly selected identifiers, never arbitrary function arguments."""
    def decorate(fn):
        parameters = signature(fn)

        @wraps(fn)
        async def wrapped(*args, **kwargs):
            arguments = parameters.bind(*args, **kwargs).arguments
            fields: dict[str, Any] = {}
            for key in ("task_id", "deployment_id", "worker_id", "round_no"):
                if key in arguments:
                    fields[key] = arguments[key]
            task = arguments.get("task")
            if task is not None:
                fields.update(task_id=task.task_id, state=task.state.value)
            item = arguments.get("item") or arguments.get("record")
            if isinstance(item, dict):
                fields.update({
                    key: item[key] for key in ("task_id", "deployment_id", "round_no", "stage", "worker_id")
                    if key in item
                })
            if "execution_id" not in get_contextvars():
                fields["execution_id"] = uuid4().hex
            with logged_step(step, level=level, **fields):
                return await fn(*args, **kwargs)

        return wrapped

    return decorate
