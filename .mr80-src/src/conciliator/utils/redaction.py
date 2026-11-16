"""Redact collected copies only; never mutate the business request."""

import json
import re
from typing import Any

_SECRET_KEY = re.compile(
    r"authorization|cookie|secret|token|password|passwd|credential|signature|jwt|api[-_]?key", re.I
)
_ASSIGNMENT = re.compile(
    r"""(?i)(?<![A-Za-z0-9_-])((?:[A-Za-z0-9_-]{0,64}(?:secret|token|password|passwd|authorization|cookie|signature|jwt|api[_-]?key)[A-Za-z0-9_-]{0,64})["']?\s*[:=]\s*)(?:"[^"\n]*"|'[^'\n]*'|[^\s,;&}\]]+)"""
)
_BEARER = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_DSN = re.compile(r"(\b[A-Za-z][A-Za-z0-9+.-]{0,31}://[^\s/:]+:)[^\s@]+(@)")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if _SECRET_KEY.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
                return json.dumps(redact(parsed), ensure_ascii=False)
            except (ValueError, RecursionError):
                pass
        value = _BEARER.sub(r"\1 [REDACTED]", value)
        value = _ASSIGNMENT.sub(r"\1[REDACTED]", value)
        value = _JWT.sub("[REDACTED]", value)
        return _DSN.sub(r"\1[REDACTED]\2", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


def safe_text(value: Any) -> str:
    value = redact(value)
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def without_code(value: Any) -> Any:
    """Omit explicit source/diff fields and code fences from display copies."""
    code_keys = {
        "diff",
        "patch",
        "ours",
        "theirs",
        "ours_text",
        "theirs_text",
        "source_content",
        "target_content",
        "ai_resolution",
        "resolved_text",
        "file_content",
        "normalfile",
        "conflictfiles",
    }
    if isinstance(value, dict):
        return {k: "[源码未采集]" if str(k).lower() in code_keys else without_code(v) for k, v in value.items()}
    if isinstance(value, list):
        return [without_code(v) for v in value]
    if isinstance(value, str):
        try:
            if value.lstrip().startswith(("{", "[")):
                return json.dumps(without_code(json.loads(value)), ensure_ascii=False)
        except ValueError:
            pass
        return re.sub(r"```[\s\S]*?(?:```|$)", "[代码块未采集]", value)
    return value


def coco_identity(value: Any) -> dict:
    """Only identifiers, never Coco prompt, output, message or validation contents."""
    result = {}
    allowed = {
        "sessionid",
        "runid",
        "workspaceid",
        "agentid",
        "eventid",
        "requestid",
        "status",
        "code",
        "session_id",
        "run_id",
        "workspace_id",
        "agent_id",
        "event_id",
        "coco_session_id",
        "coco_run_id",
    }

    def walk(item):
        if isinstance(item, dict):
            for key, val in item.items():
                if key in {"Session", "Run", "Workspace", "Agent"} and isinstance(val, dict) and val.get("Id"):
                    result[key + "Id"] = str(val["Id"])[:128]
                if str(key).lower() in allowed and isinstance(val, (str, int)):
                    result[str(key)] = str(val)[:128]
                elif isinstance(val, (dict, list)):
                    walk(val)
        elif isinstance(item, list):
            for entry in item:
                walk(entry)

    walk(value)
    return result
