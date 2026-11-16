from conciliator.domain.approval import ApprovalStatus, can_transition


def test_pending_to_approved():
    assert can_transition(ApprovalStatus.PENDING, ApprovalStatus.APPROVED)


def test_pending_to_rejected():
    assert can_transition(ApprovalStatus.PENDING, ApprovalStatus.REJECTED)


def test_approved_to_rejected_blocked():
    assert not can_transition(ApprovalStatus.APPROVED, ApprovalStatus.REJECTED)


def test_retry_resets_to_pending():
    assert can_transition(ApprovalStatus.APPROVED, ApprovalStatus.PENDING)
    assert can_transition(ApprovalStatus.REJECTED, ApprovalStatus.PENDING)
