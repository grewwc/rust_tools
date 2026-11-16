from enum import StrEnum


class ApprovalStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


_ALLOWED = {
    ApprovalStatus.PENDING: {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED},
    ApprovalStatus.APPROVED: {ApprovalStatus.PENDING},  # retry 回退
    ApprovalStatus.REJECTED: {ApprovalStatus.PENDING},  # retry 回退
}


def can_transition(current: ApprovalStatus, target: ApprovalStatus) -> bool:
    return target in _ALLOWED.get(current, set())


def approval_counts(blocks) -> dict[str, int]:
    counts = {status.value: 0 for status in ApprovalStatus}
    counts["UNKNOWN"] = 0
    for block in blocks:
        approval = getattr(block, "approval", None)
        status = approval.status if approval is not None else "UNKNOWN"
        counts[status if status in counts else "UNKNOWN"] += 1
    return counts


def approval_not_required(block) -> bool:
    approval = getattr(block, "approval", None)
    return approval is not None and approval.status == ApprovalStatus.NOT_REQUIRED.value
