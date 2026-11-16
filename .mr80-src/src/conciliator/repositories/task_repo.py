from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from conciliator.domain.task import TaskStatus
from conciliator.repositories._sync_repo import (
    ApproveAllResult,
    CancelTaskResult,
    RetryRequest,
    SyncTaskRepository,
    TaskClaim,
)
from conciliator.repositories.models import (
    ConflictBlock,
    LarkNotificationOutbox,
    RepoConfig,
    ResolveTask,
    TaskEvent,
)
from conciliator.utils.trace_context import observe_task


class TaskRepository:
    """async 包装层:DB 调用经 run_in_threadpool 调同步实现。"""

    def __init__(self, session_factory: sessionmaker[Session]):
        self._sync = SyncTaskRepository(session_factory)

    async def create(self, **kwargs: Any) -> ResolveTask:
        return observe_task(await run_in_threadpool(self._sync.create, **kwargs))

    async def get(self, task_id: int, *, include_bodies: bool = True) -> ResolveTask | None:
        return observe_task(await run_in_threadpool(self._sync.get, task_id, include_bodies=include_bodies))

    async def get_with_blocks(self, task_id: int) -> ResolveTask | None:
        return observe_task(await run_in_threadpool(self._sync.get_with_blocks, task_id))

    async def find_active_for_version(
        self, repo_id: str, mr_iid: int, source_sha: str, target_sha: str,
    ) -> ResolveTask | None:
        return observe_task(await run_in_threadpool(
            self._sync.find_active_for_version, repo_id, mr_iid, source_sha, target_sha,
        ))

    async def list_active_for_mr(self, repo_id: str, mr_iid: int) -> list[ResolveTask]:
        return await run_in_threadpool(self._sync.list_active_for_mr, repo_id, mr_iid)

    async def find_active_for_mr(
        self,
        repo_id: str,
        mr_iid: int,
    ) -> ResolveTask | None:
        return observe_task(await run_in_threadpool(
            self._sync.find_active_for_mr,
            repo_id,
            mr_iid,
        ))

    async def find_active_for_resolve_mr(
        self,
        repo_id: str,
        mr_iid: int,
    ) -> ResolveTask | None:
        return observe_task(await run_in_threadpool(
            self._sync.find_active_for_resolve_mr,
            repo_id,
            mr_iid,
        ))

    async def find_for_resolve_mr(self, repo_id: str, mr_iid: int) -> ResolveTask | None:
        return observe_task(await run_in_threadpool(
            self._sync.find_for_resolve_mr, repo_id, mr_iid,
        ))

    async def find_latest_for_original_mr(
        self,
        repo_id: str,
        mr_iid: int,
    ) -> ResolveTask | None:
        return observe_task(await run_in_threadpool(
            self._sync.find_latest_for_original_mr,
            repo_id,
            mr_iid,
        ))

    async def list_schedulable(self, limit: int = 10) -> list[ResolveTask]:
        return await run_in_threadpool(self._sync.list_schedulable, limit)

    async def claim_pending(self, limit: int = 10) -> int | None:
        return await run_in_threadpool(self._sync.claim_pending, limit)

    async def claim_schedulable(self, limit: int = 10) -> TaskClaim | None:
        return await run_in_threadpool(self._sync.claim_schedulable, limit)

    async def reject_block_and_mark_partial(
        self,
        **kwargs: Any,
    ) -> str:
        return await run_in_threadpool(
            self._sync.reject_block_and_mark_partial,
            **kwargs,
        )

    async def approve_block(self, **kwargs: Any) -> str:
        return await run_in_threadpool(self._sync.approve_block, **kwargs)

    async def request_retry(self, **kwargs: Any) -> RetryRequest | None:
        return await run_in_threadpool(self._sync.request_retry, **kwargs)

    async def reset_rejected_approvals(
        self,
        *,
        task_id: int,
        conflict_ids: tuple[str, ...],
    ) -> int:
        return await run_in_threadpool(
            self._sync.reset_rejected_approvals,
            task_id=task_id,
            conflict_ids=conflict_ids,
        )

    async def list_events(self, task_id: int) -> list[TaskEvent]:
        return await run_in_threadpool(self._sync.list_events, task_id)

    async def list_by_status_updated_before(
        self,
        *,
        status: TaskStatus,
        threshold: datetime,
        limit: int = 10,
    ) -> list[ResolveTask]:
        return await run_in_threadpool(
            self._sync.list_by_status_updated_before,
            status=status,
            threshold=threshold,
            limit=limit,
        )

    async def list_poll_candidates(
        self, *, status: TaskStatus, threshold: datetime,
        cursor: tuple[datetime, int] | None = None, limit: int = 10,
    ) -> list[tuple[int, datetime]]:
        return await run_in_threadpool(
            self._sync.list_poll_candidates, status=status, threshold=threshold, cursor=cursor, limit=limit,
        )

    async def list_pending_failure_notices(self, limit: int = 10) -> list[ResolveTask]:
        return await run_in_threadpool(self._sync.list_pending_failure_notices, limit)

    async def update_status(
        self,
        *,
        task_id: int,
        from_version: int,
        new_status: TaskStatus,
        from_status: TaskStatus | None = None,
        **fields: Any,
    ) -> bool:
        return await run_in_threadpool(
            self._sync.update_status,
            task_id=task_id,
            from_version=from_version,
            new_status=new_status,
            from_status=from_status,
            **fields,
        )

    async def update_fields(
        self,
        *,
        task_id: int,
        from_version: int,
        **fields: Any,
    ) -> bool:
        return await run_in_threadpool(
            self._sync.update_fields,
            task_id=task_id,
            from_version=from_version,
            **fields,
        )

    async def save_conflict_blocks(
        self,
        task_id: int,
        blocks: list[dict],
    ) -> list[int]:
        return await run_in_threadpool(
            self._sync.save_conflict_blocks,
            task_id,
            blocks,
        )

    async def update_block_discussion_and_resolution(
        self,
        *,
        block_id: int,
        discussion_id: str,
        ai_resolution: str,
    ) -> None:
        await run_in_threadpool(
            self._sync.update_block_discussion_and_resolution,
            block_id=block_id,
            discussion_id=discussion_id,
            ai_resolution=ai_resolution,
        )

    async def update_block_ai_resolution(
        self,
        *,
        block_id: int,
        ai_resolution: str,
    ) -> None:
        await run_in_threadpool(
            self._sync.update_block_ai_resolution,
            block_id=block_id,
            ai_resolution=ai_resolution,
        )

    async def update_block_resolution(
        self,
        *,
        block_id: int,
        fields: dict[str, Any],
    ) -> None:
        await run_in_threadpool(
            self._sync.update_block_resolution,
            block_id=block_id,
            fields=fields,
        )

    async def add_event(
        self,
        *,
        task_id: int,
        event_type: str,
        payload: dict,
        coco_event_id: str | None = None,
    ) -> bool:
        return await run_in_threadpool(
            self._sync.add_event,
            task_id=task_id,
            event_type=event_type,
            payload=payload,
            coco_event_id=coco_event_id,
        )

    async def complete_all_approved_and_enqueue(
        self,
        *,
        task_id: int,
        from_version: int,
        approved_count: int,
        no_manual_approval: bool = False,
    ) -> tuple[bool, bool]:
        return await run_in_threadpool(
            self._sync.complete_all_approved_and_enqueue,
            task_id=task_id,
            from_version=from_version,
            approved_count=approved_count,
            no_manual_approval=no_manual_approval,
        )

    async def approve_all_pending_and_enqueue(
        self,
        *,
        task_id: int,
        reviewer_email: str,
        webhook_event_id: str | None,
    ) -> ApproveAllResult:
        return await run_in_threadpool(
            self._sync.approve_all_pending_and_enqueue,
            task_id=task_id,
            reviewer_email=reviewer_email,
            webhook_event_id=webhook_event_id,
        )

    async def cancel_task(
        self,
        *,
        task_id: int,
        from_version: int,
        cancelled_by: str,
        webhook_event_id: str | None,
    ) -> CancelTaskResult:
        return await run_in_threadpool(
            self._sync.cancel_task,
            task_id=task_id,
            from_version=from_version,
            cancelled_by=cancelled_by,
            webhook_event_id=webhook_event_id,
        )

    async def claim_lark_notification(
        self,
        *,
        stale_before: datetime,
        limit: int = 10,
    ) -> LarkNotificationOutbox | None:
        return await run_in_threadpool(
            self._sync.claim_lark_notification,
            stale_before=stale_before,
            limit=limit,
        )

    async def mark_lark_notification_sent(
        self,
        *,
        notification_id: int,
        from_version: int,
        recipient_open_id: str,
        message_id: str,
    ) -> bool:
        return await run_in_threadpool(
            self._sync.mark_lark_notification_sent,
            notification_id=notification_id,
            from_version=from_version,
            recipient_open_id=recipient_open_id,
            message_id=message_id,
        )

    async def mark_lark_notification_failed(
        self,
        *,
        notification_id: int,
        from_version: int,
        recipient_open_id: str | None,
        error: str,
        retry_at: datetime | None,
    ) -> bool:
        return await run_in_threadpool(
            self._sync.mark_lark_notification_failed,
            notification_id=notification_id,
            from_version=from_version,
            recipient_open_id=recipient_open_id,
            error=error,
            retry_at=retry_at,
        )

    async def get_lark_notification(
        self,
        notification_id: int,
    ) -> LarkNotificationOutbox | None:
        return await run_in_threadpool(
            self._sync.get_lark_notification,
            notification_id,
        )

    async def get_lark_notification_for_task(
        self,
        *,
        task_id: int,
        notification_type: str,
    ) -> LarkNotificationOutbox | None:
        return await run_in_threadpool(
            self._sync.get_lark_notification_for_task,
            task_id=task_id,
            notification_type=notification_type,
        )

    async def claim_nextcode_webhook(
        self,
        *,
        delivery_key: str,
        event_id: str | None,
        delivery_id: str | None,
        event_type: str,
    ) -> bool:
        return await run_in_threadpool(
            self._sync.claim_nextcode_webhook,
            delivery_key=delivery_key,
            event_id=event_id,
            delivery_id=delivery_id,
            event_type=event_type,
        )

    async def complete_nextcode_webhook(self, *, delivery_key: str, processed_count: int) -> None:
        await run_in_threadpool(
            self._sync.complete_nextcode_webhook,
            delivery_key=delivery_key,
            processed_count=processed_count,
        )

    async def fail_nextcode_webhook(self, *, delivery_key: str, error: str) -> None:
        await run_in_threadpool(
            self._sync.fail_nextcode_webhook,
            delivery_key=delivery_key,
            error=error,
        )

    async def find_block_by_discussion(
        self,
        discussion_id: str,
    ) -> tuple[ConflictBlock, ResolveTask] | None:
        return await run_in_threadpool(
            self._sync.find_block_by_discussion,
            discussion_id,
        )

    async def get_repo_config(self, repo_id: str) -> RepoConfig | None:
        return await run_in_threadpool(self._sync.get_repo_config, repo_id)
