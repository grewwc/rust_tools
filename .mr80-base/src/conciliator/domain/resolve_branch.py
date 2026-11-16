import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

MAX_RESOLVE_BRANCH_BYTES = 200
_SHANGHAI = ZoneInfo("Asia/Shanghai")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def build_resolve_branch(
    *,
    source_branch: str,
    target_branch: str,
    task_id: int,
    created_at: datetime,
    max_bytes: int = MAX_RESOLVE_BRANCH_BYTES,
) -> str:
    """Build the stable resolver branch name for one task.

    The date and task suffix are never truncated. The task persists the result,
    so retries reuse the same ref instead of deriving it from moving branches.
    """
    if task_id < 1:
        raise ValueError("task_id must be positive")
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    source = _slug(source_branch)
    target = _slug(target_branch)
    date = created_at.astimezone(_SHANGHAI).strftime("%Y%m%d")
    suffix = f"-conflict-{date}-t{task_id}"
    fixed = f"fix/{suffix}"
    if len(fixed.encode("utf-8")) + len("a-to-b") > max_bytes:
        raise ValueError("max_bytes is too small for the resolver branch suffix")

    stem_budget = max_bytes - len(b"fix/") - len(suffix.encode())
    source, target = _fit_sides(source, target, stem_budget)
    branch = f"fix/{source}-to-{target}{suffix}"
    if len(branch.encode("utf-8")) > max_bytes:
        raise AssertionError("resolver branch truncation exceeded max_bytes")
    return branch


def _slug(value: str) -> str:
    slug = _NON_ALNUM.sub("-", value.lower()).strip("-")
    return slug or "branch"


def _fit_sides(source: str, target: str, stem_budget: int) -> tuple[str, str]:
    separator_len = len("-to-")
    content_budget = stem_budget - separator_len
    if len(source) + len(target) <= content_budget:
        return source, target

    source_budget = min(len(source), content_budget // 2)
    target_budget = min(len(target), content_budget - source_budget)
    remaining = content_budget - source_budget - target_budget
    if remaining:
        source_extra = min(len(source) - source_budget, remaining)
        source_budget += source_extra
        remaining -= source_extra
        target_budget += min(len(target) - target_budget, remaining)
    return source[:source_budget].rstrip("-") or "b", target[:target_budget].rstrip("-") or "b"
