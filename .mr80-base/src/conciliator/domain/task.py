from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    BRANCH_CREATED = "BRANCH_CREATED"
    CONFLICTS_FETCHED = "CONFLICTS_FETCHED"
    COCO_SUBMITTED = "COCO_SUBMITTED"
    COCO_RUNNING = "COCO_RUNNING"
    COCO_PROCESSING = "COCO_PROCESSING"
    DOC_CREATED = "DOC_CREATED"
    MR_CREATED = "MR_CREATED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PARTIAL_REJECTED = "PARTIAL_REJECTED"
    RETRYING = "RETRYING"
    RETRY_PREPARING = "RETRY_PREPARING"
    RETRY_COCO_SUBMITTED = "RETRY_COCO_SUBMITTED"
    RETRY_COCO_RUNNING = "RETRY_COCO_RUNNING"
    RETRY_COCO_PROCESSING = "RETRY_COCO_PROCESSING"
    RETRY_RESULT_VERIFIED = "RETRY_RESULT_VERIFIED"
    RETRY_DOC_UPLOADED = "RETRY_DOC_UPLOADED"
    RETRY_PUBLISHING = "RETRY_PUBLISHING"
    ALL_APPROVED = "ALL_APPROVED"
    CANCELLED = "CANCELLED"
    COCO_TIMEOUT = "COCO_TIMEOUT"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.PREPARING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.PREPARING: {
        TaskStatus.BRANCH_CREATED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.BRANCH_CREATED: {
        TaskStatus.CONFLICTS_FETCHED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.CONFLICTS_FETCHED: {
        TaskStatus.COCO_SUBMITTED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.COCO_SUBMITTED: {
        TaskStatus.COCO_RUNNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.COCO_RUNNING: {
        TaskStatus.COCO_PROCESSING,
        TaskStatus.DOC_CREATED,
        TaskStatus.COCO_TIMEOUT,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.COCO_PROCESSING: {
        TaskStatus.COCO_RUNNING, TaskStatus.DOC_CREATED, TaskStatus.COCO_TIMEOUT,
        TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.COCO_TIMEOUT: {TaskStatus.FAILED},
    TaskStatus.DOC_CREATED: {
        TaskStatus.MR_CREATED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.MR_CREATED: {
        TaskStatus.ALL_APPROVED,
        TaskStatus.AWAITING_APPROVAL,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.AWAITING_APPROVAL: {
        TaskStatus.ALL_APPROVED,
        TaskStatus.PARTIAL_REJECTED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.PARTIAL_REJECTED: {
        TaskStatus.RETRYING,
        TaskStatus.AWAITING_APPROVAL,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RETRYING: {
        TaskStatus.RETRY_PREPARING,
        TaskStatus.PARTIAL_REJECTED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RETRY_PREPARING: {
        TaskStatus.RETRY_COCO_SUBMITTED,
        TaskStatus.PARTIAL_REJECTED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RETRY_COCO_SUBMITTED: {
        TaskStatus.RETRY_COCO_RUNNING,
        TaskStatus.PARTIAL_REJECTED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RETRY_COCO_RUNNING: {
        TaskStatus.RETRY_COCO_PROCESSING,
        TaskStatus.RETRY_RESULT_VERIFIED,
        TaskStatus.PARTIAL_REJECTED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RETRY_COCO_PROCESSING: {
        TaskStatus.RETRY_COCO_RUNNING, TaskStatus.RETRY_RESULT_VERIFIED,
        TaskStatus.PARTIAL_REJECTED, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.RETRY_RESULT_VERIFIED: {
        TaskStatus.RETRY_DOC_UPLOADED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RETRY_DOC_UPLOADED: {
        TaskStatus.RETRY_PUBLISHING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RETRY_PUBLISHING: {
        TaskStatus.AWAITING_APPROVAL,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.ALL_APPROVED: set(),
    TaskStatus.CANCELLED: set(),
    TaskStatus.FAILED: set(),
}


TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.ALL_APPROVED,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
        TaskStatus.COCO_TIMEOUT,
    }
)

SCHEDULABLE_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.RETRYING})
AWAITING_EXTERNAL_STATUSES = frozenset(
    {
        TaskStatus.COCO_RUNNING,
        TaskStatus.RETRY_COCO_RUNNING,
        TaskStatus.AWAITING_APPROVAL,
        TaskStatus.PARTIAL_REJECTED,
    }
)


def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, set())


class InvalidTransitionError(Exception):
    def __init__(self, current: TaskStatus, target: TaskStatus):
        super().__init__(f"Cannot transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


# A status claim excludes other Coco pollers; it is not a renewable execution lease.
COCO_POLL_CLAIMS = {
    TaskStatus.COCO_RUNNING: TaskStatus.COCO_PROCESSING,
    TaskStatus.RETRY_COCO_RUNNING: TaskStatus.RETRY_COCO_PROCESSING,
}
COCO_PROCESSING_TIMEOUT_SECONDS = 1800
