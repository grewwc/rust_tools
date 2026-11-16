"""Capture each actual HTTP attempt before envelope parsing, without changing retry policy."""

import asyncio
import time
from contextvars import ContextVar
from functools import wraps
from uuid import uuid4

from structlog.contextvars import get_contextvars

from conciliator.services.trace_collector import emit
from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import error_fields
from conciliator.utils.redaction import coco_identity

_operation: ContextVar[dict | None] = ContextVar("trace_operation", default=None)
log = get_logger(__name__)


def _diagnostic_fields(fields):
    result = {key: fields[key] for key in ("provider", "operation", "call_id", "operation_id", "attempt")}
    result["operation"] = result["operation"].partition("?")[0]
    if fields["provider"] == "coco":
        details = fields.get("details") or {}
        result["session_id"] = details.get("SessionId") or details.get("session_id") or details.get("SessionID")
        result["run_id"] = details.get("RunId") or details.get("run_id") or details.get("RunID")
    return result


def traced_operation(fn):
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        token = _operation.set({"id": uuid4().hex, "attempt": 0})
        try:
            return await fn(*args, **kwargs)
        finally:
            _operation.reset(token)

    return wrapped


def start_call(provider, operation, body):
    logical = _operation.get()
    if logical is not None:
        logical["attempt"] += 1
    fields = dict(
        provider=provider,
        operation=operation[:128],
        call_id=uuid4().hex,
        operation_id=logical["id"] if logical else uuid4().hex,
        attempt=logical["attempt"] if logical else 1,
    )
    if provider == "coco":
        fields["details"] = coco_identity(body)
        body = None
    emit("EXTERNAL_REQUEST", body=body, kind="request", outcome="UNKNOWN", summary=operation, **fields)
    if get_contextvars().get("workflow") == "meegle":
        log.debug("meegle_external_started", **_diagnostic_fields(fields))
    return fields, time.monotonic()


def response_event(fields, started, response, *, body=None):
    if body is None:
        try:
            body = response.json()
        except ValueError:
            body = response.text
    error = None
    request_id = response.headers.get("x-request-id") or response.headers.get("x-tt-logid")
    if isinstance(body, dict):
        metadata = body.get("ResponseMetadata")
        if isinstance(metadata, dict):
            error = metadata.get("Error")
            request_id = metadata.get("RequestId") or request_id
        error = error or body.get("code")
    outcome = "HTTP_ERROR" if response.status_code >= 400 else "BUSINESS_ERROR" if error else "SUCCESS"
    if get_contextvars().get("workflow") == "meegle":
        write = log.debug if outcome == "SUCCESS" else log.warning
        code = error.get("Code") if isinstance(error, dict) else error
        write(
            "meegle_external_finished", **_diagnostic_fields(fields), http_status=response.status_code,
            duration_ms=int((time.monotonic() - started) * 1000), outcome=outcome,
            code=code if isinstance(code, (str, int)) else None,
            external_request_id=str(request_id)[:128] if request_id else None,
        )
    if fields["provider"] == "nextcode" and fields["operation"] in SOURCE_OPERATIONS:
        body = omit_source(body)
    if fields["provider"] == "coco":
        fields = {**fields, "details": {**fields.get("details", {}), **coco_identity(body)}}
        body = None
    emit(
        "EXTERNAL_RESPONSE",
        body=body,
        kind="response",
        http_status=response.status_code,
        duration_ms=int((time.monotonic() - started) * 1000),
        outcome=outcome,
        external_request_id=str(request_id)[:128] if request_id else None,
        summary=f"{fields['operation']} · HTTP {response.status_code} · {outcome}",
        **fields,
    )


def error_event(fields, started, exc):
    if get_contextvars().get("workflow") == "meegle":
        log.warning(
            "meegle_external_failed", **_diagnostic_fields(fields), **error_fields(exc),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    emit(
        "EXTERNAL_ERROR",
        body=None if fields["provider"] == "coco" else str(exc),
        kind="response",
        outcome="UNKNOWN",
        duration_ms=int((time.monotonic() - started) * 1000),
        summary=type(exc).__name__,
        **fields,
    )


async def traced_request(client, provider, operation, method, url, **kwargs):
    # Do not read file handles or consume iterators to collect multipart data.
    files = kwargs.get("files")
    upload = {}
    if isinstance(files, dict):
        for key, value in files.items():
            if isinstance(value, tuple):
                upload[key] = {
                    "filename": value[0],
                    "content": value[1] if isinstance(value[1], (str, bytes)) else "[stream not collected]",
                    "mime": value[2] if len(value) > 2 else None,
                }
    auth = "/auth/" in str(url)
    body = {
        "method": method,
        "url": str(url),
        "params": kwargs.get("params"),
        "json": {"app_id": (kwargs.get("json") or {}).get("app_id")} if auth else kwargs.get("json"),
        "data": kwargs.get("data"),
        "files": upload,
    }
    if provider == "nextcode" and operation == "CreateCommit":
        body = omit_source(body)
    fields, started = start_call(provider, operation, body)
    try:
        response = await client.request(method, url, **kwargs)
    except (Exception, asyncio.CancelledError) as exc:
        error_event(fields, started, exc)
        raise
    if auth:
        # Never retain an authentication response body, even non-JSON errors.
        try:
            auth_code = response.json().get("code")
        except (ValueError, AttributeError):
            auth_code = None
        response_event(
            fields, started, response, body={"code": auth_code, "authentication_response": "[not collected]"}
        )
    else:
        response_event(fields, started, response)
    return response


SOURCE_OPERATIONS = {"GetFile", "ListMergeRequestConflicts", "GetMergeRequestConflict", "CreateCommit"}


def omit_source(value):
    if isinstance(value, dict):
        return {k: "[源码未采集]" if k.lower() == "content" else omit_source(v) for k, v in value.items()}
    if isinstance(value, list):
        return [omit_source(v) for v in value]
    return value
