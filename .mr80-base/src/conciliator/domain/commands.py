import re
from dataclasses import dataclass

# 全局指令:/resolve approve all / /resolve retry / /resolve status / /resolve cancel
_GLOBAL_RE = re.compile(
    r"^\s*/resolve\s+(?P<action>approve\s+all|retry|status|cancel)\b\s*$",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(r"^\s*/resolve\s+cancel(?:\s+(?P<task_id>[1-9][0-9]*))?\s*$", re.IGNORECASE)

# 单块拒绝指令；指令与 reason 之间允许不留空格。
_REJECT_RE = re.compile(
    r"^\s*(?:/?reject|❌|-1|👎|\[(?:minusone|crossmark|thumbsdown)\])"
    r"\s*(?P<reason>.*?)\s*$",
    re.IGNORECASE,
)
_APPROVE_TOKENS = {
    "approve",
    "/approval",
    "lgtm",
    "+1",
    "[lgtm]",
    "[jiayi]",
    "[checkmark]",
    "[thumbsup]",
}
_ILLEGAL_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_REJECT_REASON_LENGTH = 1000


@dataclass(frozen=True)
class Ignored:
    body: str


@dataclass(frozen=True)
class ApproveBlock:
    pass


@dataclass(frozen=True)
class RejectBlock:
    reason: str


@dataclass(frozen=True)
class ApproveAll:
    pass


@dataclass(frozen=True)
class Retry:
    pass


@dataclass(frozen=True)
class Status:
    pass


@dataclass(frozen=True)
class Cancel:
    task_id: int | None = None


Command = ApproveBlock | RejectBlock | ApproveAll | Retry | Status | Cancel | Ignored


def parse_comment(body: str, in_discussion: bool) -> Command:
    """解析评论 body。

    匹配顺序:
      1. 全局指令(/resolve ...)  — 任何位置都识别
      2. 单块指令(approve/reject) — 仅 in_discussion=True 时识别
      3. 其余 → Ignored
    """
    body = body.strip()
    if not body:
        return Ignored(body=body)

    cancel = _CANCEL_RE.fullmatch(body)
    if cancel:
        task_id = cancel.group("task_id")
        return Cancel(task_id=int(task_id) if task_id else None)

    m = _GLOBAL_RE.match(body)
    if m:
        action = re.sub(r"\s+", " ", m.group("action").lower())
        mapping: dict[str, Command] = {
            "approve all": ApproveAll(),
            "retry": Retry(),
            "status": Status(),
            "cancel": Cancel(),
        }
        return mapping[action]

    if not in_discussion:
        return Ignored(body=body)

    first_token = body.split()[0] if body.split() else ""
    if first_token.lower() in _APPROVE_TOKENS or first_token in {"✅", "👍"}:
        return ApproveBlock()

    m_reject = _REJECT_RE.match(body)
    if m_reject:
        reason = _ILLEGAL_CONTROL_RE.sub("", m_reject.group("reason")).strip()
        if not reason or len(reason) > MAX_REJECT_REASON_LENGTH:
            return Ignored(body=body)
        return RejectBlock(reason=reason)

    return Ignored(body=body)
