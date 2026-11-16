"""Bundled redact/safe_text from Conciliator; keep sanitization behavior identical."""

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
