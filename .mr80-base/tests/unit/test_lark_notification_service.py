from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from conciliator.config import Settings
from conciliator.services import lark_notification_service as notification_module
from conciliator.services.lark_notification_service import (
    LarkNotificationService,
)
from conciliator.utils.retry import TransientError


def _settings() -> Settings:
    return Settings(
        env="dev",
        service_psm="x",
        lark_app_id="li",
        mysql_psm="x",
        lark_notification_retry_base_seconds=1,
        lark_notification_retry_max_seconds=10,
        lark_notification_max_attempts=3,
    )  # type: ignore[arg-type]


def _notification(*, attempts: int = 1):
    return SimpleNamespace(
        id=11,
        task_id=7,
        notification_type="ALL_APPROVED",
        recipient_email="trigger@example.com",
        recipient_open_id=None,
        mr_url="https://code.test/mr/99",
        request_uuid="conciliator-all-approved-7",
        status="SENDING",
        attempts=attempts,
        version=4,
        updated_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_type, sender, extra", [
    ("TASK_RECEIVED", "send_task_received_message", {"task_id": 7}),
    ("ALL_APPROVED", "send_all_approved_message", {}),
    ("NO_MANUAL_APPROVAL_REQUIRED", "send_no_manual_approval_message", {}),
    ("AWAITING_APPROVAL:0", "send_awaiting_approval_message", {"task_id": 7, "attempt": 1}),
    ("AWAITING_APPROVAL:2", "send_awaiting_approval_message", {"task_id": 7, "attempt": 3}),
])
async def test_process_once_resolves_recipient_and_marks_sent(monkeypatch, notification_type, sender, extra):
    item = _notification()
    item.notification_type = notification_type
    repo = SimpleNamespace(
        claim_lark_notification=AsyncMock(return_value=item),
        mark_lark_notification_sent=AsyncMock(return_value=True),
        add_event=AsyncMock(return_value=True),
    )
    resolve = AsyncMock(return_value="ou_trigger")
    send = AsyncMock(return_value="om_message")
    monkeypatch.setattr(
        notification_module,
        "get_user_open_id_by_email",
        resolve,
    )
    monkeypatch.setattr(
        notification_module,
        sender,
        send,
    )
    lark = object()
    service = LarkNotificationService(
        lark=lark,
        repo=repo,
        settings=_settings(),
    )

    assert await service.process_once() is True

    resolve.assert_awaited_once_with(lark, "trigger@example.com")
    send.assert_awaited_once()
    assert send.await_args.kwargs == {
        "open_id": "ou_trigger",
        "mr_url": "https://code.test/mr/99",
        "request_uuid": "conciliator-all-approved-7",
        **extra,
    }
    repo.mark_lark_notification_sent.assert_awaited_once_with(
        notification_id=11,
        from_version=4,
        recipient_open_id="ou_trigger",
        message_id="om_message",
    )
    assert repo.add_event.await_args.kwargs["event_type"] == (
        "LARK_NOTIFICATION_SENT"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_type, sender", [
    ("TASK_RECEIVED", "send_task_received_message"),
    ("ALL_APPROVED", "send_all_approved_message"),
    ("AWAITING_APPROVAL:0", "send_awaiting_approval_message"),
])
async def test_transient_failure_is_retried_without_changing_task(monkeypatch, notification_type, sender):
    item = _notification(attempts=1)
    item.notification_type = notification_type
    repo = SimpleNamespace(
        claim_lark_notification=AsyncMock(return_value=item),
        mark_lark_notification_failed=AsyncMock(return_value=True),
        add_event=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        notification_module,
        "get_user_open_id_by_email",
        AsyncMock(return_value="ou_trigger"),
    )
    monkeypatch.setattr(
        notification_module,
        sender,
        AsyncMock(side_effect=TransientError("rate limited")),
    )
    service = LarkNotificationService(
        lark=object(),
        repo=repo,
        settings=_settings(),
    )

    assert await service.process_once() is True

    failed = repo.mark_lark_notification_failed.await_args.kwargs
    assert failed["notification_id"] == 11
    assert failed["recipient_open_id"] == "ou_trigger"
    assert failed["retry_at"] is not None
    assert repo.add_event.await_args.kwargs["event_type"] == (
        "LARK_NOTIFICATION_RETRY"
    )


@pytest.mark.asyncio
async def test_retry_limit_marks_notification_failed(monkeypatch):
    item = _notification(attempts=3)
    repo = SimpleNamespace(
        claim_lark_notification=AsyncMock(return_value=item),
        mark_lark_notification_failed=AsyncMock(return_value=True),
        add_event=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        notification_module,
        "get_user_open_id_by_email",
        AsyncMock(side_effect=TransientError("network")),
    )
    service = LarkNotificationService(
        lark=object(),
        repo=repo,
        settings=_settings(),
    )

    assert await service.process_once() is True

    assert (
        repo.mark_lark_notification_failed.await_args.kwargs["retry_at"]
        is None
    )
    assert repo.add_event.await_args.kwargs["event_type"] == (
        "LARK_NOTIFICATION_FAILED"
    )
