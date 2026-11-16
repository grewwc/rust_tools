"""Metrics 上报薄包装。生产用 bytedance.metrics,测试/本地无 agent 时 no-op。"""
import contextlib
import os
from typing import Any

_TRY_IMPORT = True
_client: Any = None


def _get_client() -> Any | None:
    global _client, _TRY_IMPORT
    if not _TRY_IMPORT:
        return None
    if _client is None:
        try:
            from bytedance import metrics as _bm
            _client = _bm
        except Exception:
            _TRY_IMPORT = False
            return None
    return _client


def emit_counter(name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
    client = _get_client()
    if client is None:
        return
    with contextlib.suppress(Exception):
        client.emit_counter(name, value, tags=tags or {})


def emit_timer(name: str, seconds: float, tags: dict[str, str] | None = None) -> None:
    client = _get_client()
    if client is None:
        return
    with contextlib.suppress(Exception):
        client.emit_timer(name, seconds, tags=tags or {})


def emit_gauge(name: str, value: float, tags: dict[str, str] | None = None) -> None:
    client = _get_client()
    if client is None:
        return
    with contextlib.suppress(Exception):
        client.emit_gauge(name, value, tags=tags or {})


class MetricName:
    TASKS_TOTAL = "conciliator.tasks"
    TASK_DURATION = "conciliator.task_duration"
    COCO_TASK_DURATION = "conciliator.coco_task_duration"
    CONFLICT_BLOCKS_TOTAL = "conciliator.conflict_blocks"
    APPROVAL_TOTAL = "conciliator.approval"
    EXTERNAL_API_CALLS = "conciliator.external_api_calls"
    EXTERNAL_API_DURATION = "conciliator.external_api_duration"
    WEBHOOK_PROCESSED = "conciliator.webhook_processed"
    QUEUE_DEPTH = "conciliator.queue_depth"
    LARK_NOTIFICATIONS_TOTAL = "conciliator.lark_notifications"


def is_debug_enabled() -> bool:
    return bool(os.getenv("BYTEDMETRICS_DEBUG"))
