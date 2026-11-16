"""ASGI request diagnostics; no query strings, headers or bodies are collected."""

import time
from uuid import uuid4

from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import error_fields, workflow_scope

log = get_logger(__name__)
PREFIXES = (
    "/hooks/meegle/", "/hooks/lark/aeolus", "/api/aeolus/", "/api/deployment-workers/",
    "/oauth/lark/",
)


class MeegleLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(PREFIXES):
            return await self.app(scope, receive, send)
        started = time.monotonic()
        status = 500

        async def send_response(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        with workflow_scope(request_id=uuid4().hex):
            try:
                await self.app(scope, receive, send_response)
            except Exception as exc:
                status = 500
                log.warning("meegle_http_failed", method=scope["method"], **error_fields(exc))
                raise
            finally:
                route = scope.get("route")
                fields = {
                    key: value for key, value in scope.get("path_params", {}).items()
                    if key in {"task_id", "deployment_id", "worker_id"}
                }
                polling = scope["path"].startswith("/api/deployment-workers/") and scope["path"].endswith(
                    ("/heartbeat", "/claim")
                )
                write = log.warning if status >= 400 else (
                    log.info if not polling and (
                        scope["method"] != "GET" or scope["path"].startswith("/oauth/")
                    ) else log.debug
                )
                write(
                    "meegle_http_finished", method=scope["method"],
                    route=getattr(route, "path", "unmatched"), http_status=status,
                    duration_ms=int((time.monotonic() - started) * 1000), **fields,
                )
