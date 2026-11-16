import pytest

from conciliator.domain.task import (
    SCHEDULABLE_STATUSES,
    TERMINAL_STATUSES,
    TaskStatus,
    can_transition,
)


@pytest.mark.parametrize(
    "frm,to,expected",
    [
        (TaskStatus.PENDING, TaskStatus.PREPARING, True),
        (TaskStatus.PREPARING, TaskStatus.BRANCH_CREATED, True),
        (TaskStatus.COCO_RUNNING, TaskStatus.DOC_CREATED, True),
        (TaskStatus.COCO_RUNNING, TaskStatus.COCO_TIMEOUT, True),
        (TaskStatus.AWAITING_APPROVAL, TaskStatus.ALL_APPROVED, True),
        (TaskStatus.AWAITING_APPROVAL, TaskStatus.PARTIAL_REJECTED, True),
        (TaskStatus.PARTIAL_REJECTED, TaskStatus.RETRYING, True),
        (TaskStatus.RETRYING, TaskStatus.RETRY_PREPARING, True),
        (TaskStatus.RETRY_PREPARING, TaskStatus.RETRY_COCO_SUBMITTED, True),
        (TaskStatus.RETRY_COCO_SUBMITTED, TaskStatus.RETRY_COCO_RUNNING, True),
        (TaskStatus.RETRY_COCO_RUNNING, TaskStatus.RETRY_RESULT_VERIFIED, True),
        (TaskStatus.RETRY_RESULT_VERIFIED, TaskStatus.RETRY_DOC_UPLOADED, True),
        (TaskStatus.RETRY_DOC_UPLOADED, TaskStatus.RETRY_PUBLISHING, True),
        (TaskStatus.RETRY_PUBLISHING, TaskStatus.AWAITING_APPROVAL, True),
        (TaskStatus.PENDING, TaskStatus.AWAITING_APPROVAL, False),
        (TaskStatus.AWAITING_APPROVAL, TaskStatus.COCO_RUNNING, False),
        (TaskStatus.ALL_APPROVED, TaskStatus.PENDING, False),
        (TaskStatus.FAILED, TaskStatus.PENDING, False),
    ],
)
def test_can_transition(frm, to, expected):
    assert can_transition(frm, to) is expected


def test_schedulable():
    assert TaskStatus.PENDING in SCHEDULABLE_STATUSES
    assert TaskStatus.RETRYING in SCHEDULABLE_STATUSES
    assert TaskStatus.AWAITING_APPROVAL not in SCHEDULABLE_STATUSES


def test_terminal():
    for s in (TaskStatus.ALL_APPROVED, TaskStatus.CANCELLED, TaskStatus.FAILED):
        assert s in TERMINAL_STATUSES
    assert TaskStatus.AWAITING_APPROVAL not in TERMINAL_STATUSES
