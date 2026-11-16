from datetime import UTC, datetime, timedelta

from conciliator.config import Settings
from conciliator.integrations.lark.client import LarkClient, LarkError
from conciliator.integrations.lark.contacts import (
    LarkUserNotFoundError,
    get_user_open_id_by_email,
)
from conciliator.integrations.lark.messages import (
    send_all_approved_message,
    send_awaiting_approval_message,
    send_no_manual_approval_message,
    send_task_received_message,
)
from conciliator.repositories.models import LarkNotificationOutbox
from conciliator.repositories.task_repo import TaskRepository
from conciliator.utils.logging import get_logger
from conciliator.utils.metrics import MetricName, emit_counter
from conciliator.utils.retry import TransientError
from conciliator.utils.trace_context import traced_work

log = get_logger(__name__)


class LarkNotificationService:
    def __init__(
        self,
        *,
        lark: LarkClient,
        repo: TaskRepository,
        settings: Settings,
    ):
        self._lark = lark
        self._repo = repo
        self._settings = settings

    async def process_once(self) -> bool:
        stale_before = datetime.now(UTC) - timedelta(
            seconds=self._settings.lark_notification_sending_stale_seconds
        )
        notification = await self._repo.claim_lark_notification(
            stale_before=stale_before
        )
        if notification is None:
            return False
        await self._deliver(notification)
        return True

    @traced_work
    async def _deliver(self, notification: LarkNotificationOutbox) -> None:
        open_id = notification.recipient_open_id
        try:
            notification_type = notification.notification_type
            retry_round = notification_type.removeprefix("AWAITING_APPROVAL:")
            is_awaiting = notification_type.startswith("AWAITING_APPROVAL:") and retry_round.isdecimal()
            if notification_type not in {
                "TASK_RECEIVED", "ALL_APPROVED", "NO_MANUAL_APPROVAL_REQUIRED",
            } and not is_awaiting:
                raise ValueError(f"Unsupported Lark notification type: {notification_type}")
            if not open_id:
                open_id = await get_user_open_id_by_email(
                    self._lark,
                    notification.recipient_email,
                )
            if notification_type == "TASK_RECEIVED":
                message_id = await send_task_received_message(
                    self._lark, open_id=open_id, mr_url=notification.mr_url,
                    request_uuid=notification.request_uuid, task_id=notification.task_id,
                )
            elif is_awaiting:
                message_id = await send_awaiting_approval_message(
                    self._lark,
                    open_id=open_id,
                    mr_url=notification.mr_url,
                    request_uuid=notification.request_uuid,
                    task_id=notification.task_id,
                    attempt=int(retry_round) + 1,
                )
            elif notification_type == "NO_MANUAL_APPROVAL_REQUIRED":
                message_id = await send_no_manual_approval_message(
                    self._lark, open_id=open_id, mr_url=notification.mr_url,
                    request_uuid=notification.request_uuid,
                )
            else:
                message_id = await send_all_approved_message(
                    self._lark,
                    open_id=open_id,
                    mr_url=notification.mr_url,
                    request_uuid=notification.request_uuid,
                )
        except (LarkUserNotFoundError, LarkError, ValueError) as exc:
            await self._record_failure(
                notification,
                open_id=open_id,
                error=exc,
                retry=False,
            )
            return
        except TransientError as exc:
            await self._record_failure(
                notification,
                open_id=open_id,
                error=exc,
                retry=True,
            )
            return
        except Exception as exc:
            log.exception(
                "lark_notification_unexpected_error",
                notification_id=notification.id,
                task_id=notification.task_id,
            )
            await self._record_failure(
                notification,
                open_id=open_id,
                error=exc,
                retry=True,
            )
            return

        ok = await self._repo.mark_lark_notification_sent(
            notification_id=notification.id,
            from_version=notification.version,
            recipient_open_id=open_id,
            message_id=message_id,
        )
        if not ok:
            log.warning(
                "lark_notification_sent_state_race",
                notification_id=notification.id,
                task_id=notification.task_id,
            )
            return
        await self._repo.add_event(
            task_id=notification.task_id,
            event_type="LARK_NOTIFICATION_SENT",
            payload={
                "notification_id": notification.id,
                "notification_type": notification.notification_type,
                "message_id": message_id,
                "attempts": notification.attempts,
            },
        )
        emit_counter(
            MetricName.LARK_NOTIFICATIONS_TOTAL,
            1,
            tags={"status": "SENT", "type": notification.notification_type},
        )
        log.info(
            "lark_notification_sent",
            notification_id=notification.id,
            task_id=notification.task_id,
            message_id=message_id,
        )

    async def _record_failure(
        self,
        notification: LarkNotificationOutbox,
        *,
        open_id: str | None,
        error: Exception,
        retry: bool,
    ) -> None:
        should_retry = (
            retry
            and notification.attempts
            < self._settings.lark_notification_max_attempts
        )
        retry_at = None
        if should_retry:
            delay = min(
                self._settings.lark_notification_retry_base_seconds
                * (2 ** max(notification.attempts - 1, 0)),
                self._settings.lark_notification_retry_max_seconds,
            )
            retry_at = datetime.now(UTC) + timedelta(seconds=delay)
        ok = await self._repo.mark_lark_notification_failed(
            notification_id=notification.id,
            from_version=notification.version,
            recipient_open_id=open_id,
            error=str(error),
            retry_at=retry_at,
        )
        if not ok:
            log.warning(
                "lark_notification_failure_state_race",
                notification_id=notification.id,
                task_id=notification.task_id,
            )
            return
        status = "RETRY" if retry_at is not None else "FAILED"
        await self._repo.add_event(
            task_id=notification.task_id,
            event_type=f"LARK_NOTIFICATION_{status}",
            payload={
                "notification_id": notification.id,
                "notification_type": notification.notification_type,
                "attempts": notification.attempts,
                "next_retry_at": (
                    retry_at.isoformat() if retry_at is not None else None
                ),
                "error_type": type(error).__name__,
            },
        )
        emit_counter(
            MetricName.LARK_NOTIFICATIONS_TOTAL,
            1,
            tags={"status": status, "type": notification.notification_type},
        )
        log.warning(
            "lark_notification_delivery_failed",
            notification_id=notification.id,
            task_id=notification.task_id,
            attempts=notification.attempts,
            status=status,
            error_type=type(error).__name__,
        )
