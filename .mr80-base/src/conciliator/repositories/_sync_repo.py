from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, defer, selectinload, sessionmaker

from conciliator.domain.approval import ApprovalStatus
from conciliator.domain.processing_version import build_active_key
from conciliator.domain.task import (
    SCHEDULABLE_STATUSES,
    TERMINAL_STATUSES,
    TaskStatus,
)
from conciliator.integrations.git.conflicts import is_blank_conflict
from conciliator.repositories.models import (
    Approval,
    ConflictBlock,
    LarkNotificationOutbox,
    NextCodeWebhookDelivery,
    RepoConfig,
    ResolveTask,
    TaskEvent,
)
from conciliator.repositories.trace_repo import status_event


class ActiveTaskExistsError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskClaim:
    task_id: int
    status: TaskStatus


@dataclass(frozen=True)
class RetryRequest:
    task_id: int
    retry_count: int
    rejected_conflict_ids: tuple[str, ...]
    locked_conflict_ids: tuple[str, ...]


@dataclass(frozen=True)
class ApproveAllResult:
    outcome: Literal["completed", "has_rejected", "invalid_task", "no_blocks"]
    total_count: int = 0
    approved_now_count: int = 0
    rejected_count: int = 0
    notification_queued: bool = False


@dataclass(frozen=True)
class CancelTaskResult:
    cancelled: bool
    previous_status: str | None = None


class SyncTaskRepository:
    """同步实现的 task 仓储。被 async TaskRepository 通过 run_in_threadpool 调用。"""

    def __init__(self, session_factory: sessionmaker[Session]):
        self._sf = session_factory

    def create(
        self,
        *,
        repo_id: str,
        source_branch: str,
        target_branch: str,
        resolve_branch: str,
        original_mr_id: int,
        trigger_type: str,
        trigger_payload: dict,
        mr_author_email: str | None = None,
        trigger_author_email: str | None = None,
        mr_version: int | None = None,
        source_commit_sha: str | None = None,
        target_commit_sha: str | None = None,
        conflict_files_snapshot: list[dict] | None = None,
        original_mr_url: str | None = None,
    ) -> ResolveTask:
        if bool(source_commit_sha) != bool(target_commit_sha):
            raise ValueError("both commit SHAs must be provided together")
        with self._sf() as session:
            task = ResolveTask(
                trace_schema_version=1,
                original_mr_url=original_mr_url,
                repo_id=repo_id,
                source_branch=source_branch,
                target_branch=target_branch,
                resolve_branch=resolve_branch,
                original_mr_id=original_mr_id,
                trigger_type=trigger_type,
                trigger_payload=trigger_payload,
                status=TaskStatus.PENDING.value,
                active_key=(
                    build_active_key(repo_id, original_mr_id, source_commit_sha, target_commit_sha)
                    if source_commit_sha and target_commit_sha else f"{repo_id}:{original_mr_id}"
                ),
                mr_version=mr_version,
                source_commit_sha=source_commit_sha,
                target_commit_sha=target_commit_sha,
                conflict_files_snapshot=conflict_files_snapshot,
                mr_author_email=mr_author_email,
                trigger_author_email=trigger_author_email,
            )
            session.add(task)
            try:
                session.flush()
                session.add(TaskEvent(
                    task_id=task.id, event_type="TASK_CREATED", occurred_at=datetime.now(UTC),
                    payload={"repo_id": repo_id, "original_mr_id": original_mr_id,
                             "trigger_type": trigger_type, "source_commit_sha": source_commit_sha,
                             "target_commit_sha": target_commit_sha,
                             "webhook_event_id": (trigger_payload.get("_webhook") or {}).get("event_id")},
                ))
                email = (trigger_author_email or "").strip()
                if trigger_type == "COMMENT" and email:
                    # Commit the receipt with the task; retries retain the same delivery identity.
                    notification_type = "TASK_RECEIVED"
                    session.add(LarkNotificationOutbox(
                        task_id=task.id,
                        notification_type=notification_type,
                        recipient_email=email,
                        mr_url=original_mr_url or "",
                        request_uuid=str(uuid5(NAMESPACE_URL, f"conciliator:{task.id}:{notification_type}")),
                        status="PENDING",
                        attempts=0,
                        version=0,
                    ))
                    session.add(TaskEvent(
                        task_id=task.id,
                        event_type="LARK_NOTIFICATION_QUEUED",
                        payload={"notification_type": notification_type},
                    ))
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                if "active_key" not in str(exc):
                    raise
                raise ActiveTaskExistsError(
                    f"active resolve task already exists for {repo_id}!{original_mr_id}"
                ) from exc
            session.refresh(task)
            return task

    def get(self, task_id: int, *, include_bodies: bool = True) -> ResolveTask | None:
        with self._sf() as session:
            options = [] if include_bodies else [
                defer(ResolveTask.conflict_files_snapshot), defer(ResolveTask.coco_result_json),
            ]
            return session.get(ResolveTask, task_id, options=options)

    def get_with_blocks(self, task_id: int) -> ResolveTask | None:
        with self._sf() as session:
            return session.execute(
                select(ResolveTask)
                .where(ResolveTask.id == task_id)
                .options(
                    selectinload(ResolveTask.blocks).selectinload(ConflictBlock.approval),
                )
            ).scalar_one_or_none()

    def find_active_for_version(
        self, repo_id: str, mr_iid: int, source_sha: str, target_sha: str,
    ) -> ResolveTask | None:
        with self._sf() as session:
            return session.execute(
                select(ResolveTask).where(
                    ResolveTask.repo_id == repo_id,
                    ResolveTask.original_mr_id == mr_iid,
                    ResolveTask.source_commit_sha == source_sha,
                    ResolveTask.target_commit_sha == target_sha,
                    ~ResolveTask.status.in_([s.value for s in TERMINAL_STATUSES]),
                ).order_by(ResolveTask.id.desc()).limit(1)
            ).scalar_one_or_none()

    def list_active_for_mr(self, repo_id: str, mr_iid: int) -> list[ResolveTask]:
        with self._sf() as session:
            return list(session.scalars(
                select(ResolveTask).where(
                    ResolveTask.repo_id == repo_id,
                    ResolveTask.original_mr_id == mr_iid,
                    ~ResolveTask.status.in_([s.value for s in TERMINAL_STATUSES]),
                ).order_by(ResolveTask.id.desc())
            ).all())

    def find_active_for_mr(self, repo_id: str, mr_iid: int) -> ResolveTask | None:
        """Find an active task by the original conflicted MR IID."""
        with self._sf() as session:
            return session.execute(
                select(ResolveTask)
                .where(
                    ResolveTask.repo_id == repo_id,
                    ResolveTask.original_mr_id == mr_iid,
                    ~ResolveTask.status.in_([s.value for s in TERMINAL_STATUSES]),
                )
                .order_by(ResolveTask.id.desc())
                .limit(1)
            ).scalar_one_or_none()

    def find_active_for_resolve_mr(
        self,
        repo_id: str,
        mr_iid: int,
    ) -> ResolveTask | None:
        """Find an active task by the resolver MR IID."""
        with self._sf() as session:
            return session.execute(
                select(ResolveTask)
                .where(
                    ResolveTask.repo_id == repo_id,
                    ResolveTask.new_mr_id == mr_iid,
                    ~ResolveTask.status.in_([s.value for s in TERMINAL_STATUSES]),
                )
                .order_by(ResolveTask.id.desc())
                .limit(1)
            ).scalar_one_or_none()

    def find_for_resolve_mr(self, repo_id: str, mr_iid: int) -> ResolveTask | None:
        """Find the resolver MR's task, including its final state."""
        with self._sf() as session:
            return session.scalars(
                select(ResolveTask).where(
                    ResolveTask.repo_id == repo_id,
                    ResolveTask.new_mr_id == mr_iid,
                ).order_by(ResolveTask.id.desc()).limit(1)
            ).first()

    def find_latest_for_original_mr(
        self,
        repo_id: str,
        mr_iid: int,
    ) -> ResolveTask | None:
        """Find the latest task for an original MR, including terminal tasks."""
        with self._sf() as session:
            return session.execute(
                select(ResolveTask)
                .where(
                    ResolveTask.repo_id == repo_id,
                    ResolveTask.original_mr_id == mr_iid,
                )
                .order_by(ResolveTask.id.desc())
                .limit(1)
            ).scalar_one_or_none()

    def list_schedulable(self, limit: int = 10) -> list[ResolveTask]:
        """List candidate tasks for observation only.

        Callers that intend to execute a PENDING task must use ``claim_pending``
        so multiple workers cannot start the same pipeline.
        """
        with self._sf() as session:
            return list(
                session.execute(
                    select(ResolveTask)
                    .where(ResolveTask.status.in_([s.value for s in SCHEDULABLE_STATUSES]))
                    .order_by(ResolveTask.updated_at)
                    .limit(limit)
                )
                .scalars()
                .all()
            )

    def claim_pending(self, limit: int = 10) -> int | None:
        """Atomically claim the oldest PENDING task for initial execution.

        The candidate SELECT is deliberately non-locking. Execution ownership is
        decided by the conditional UPDATE, which is safe across worker threads,
        processes, and Pods: exactly one caller can change a given task from
        PENDING at its observed version to PREPARING.
        """
        with self._sf() as session:
            candidates = list(
                session.execute(
                    select(ResolveTask.id, ResolveTask.version)
                    .where(ResolveTask.status == TaskStatus.PENDING.value)
                    .order_by(ResolveTask.updated_at, ResolveTask.id)
                    .limit(limit)
                ).all()
            )
            for task_id, version in candidates:
                result = session.execute(
                    update(ResolveTask)
                    .where(
                        ResolveTask.id == task_id,
                        ResolveTask.status == TaskStatus.PENDING.value,
                        ResolveTask.version == version,
                    )
                    .values(
                        status=TaskStatus.PREPARING.value,
                        version=version + 1,
                    )
                )
                if result.rowcount == 1:  # type: ignore[attr-defined]
                    status_event(session.connection(), task_id, "PENDING", "PREPARING", version + 1)
                    session.commit()
                    return task_id
                session.rollback()
        return None

    def claim_schedulable(self, limit: int = 10) -> TaskClaim | None:
        """Atomically claim one initial or retry task.

        The status change is the ownership token. A worker that loses the
        conditional UPDATE exits without modifying the task.
        """
        transitions = {
            TaskStatus.PENDING: TaskStatus.PREPARING,
            TaskStatus.RETRYING: TaskStatus.RETRY_PREPARING,
        }
        with self._sf() as session:
            candidates = list(
                session.execute(
                    select(
                        ResolveTask.id,
                        ResolveTask.version,
                        ResolveTask.status,
                    )
                    .where(ResolveTask.status.in_([status.value for status in transitions]))
                    .order_by(ResolveTask.updated_at, ResolveTask.id)
                    .limit(limit)
                ).all()
            )
            for task_id, version, raw_status in candidates:
                source = TaskStatus(raw_status)
                target = transitions[source]
                result = session.execute(
                    update(ResolveTask)
                    .where(
                        ResolveTask.id == task_id,
                        ResolveTask.status == source.value,
                        ResolveTask.version == version,
                    )
                    .values(
                        status=target.value,
                        version=version + 1,
                        updated_at=datetime.now(UTC),
                    )
                )
                if result.rowcount == 1:  # type: ignore[attr-defined]
                    status_event(session.connection(), task_id, source.value, target.value, version + 1)
                    session.add(
                        TaskEvent(
                            task_id=task_id,
                            event_type=("TASK_CLAIMED" if source is TaskStatus.PENDING else "RETRY_CLAIMED"),
                            payload={
                                "from_status": source.value,
                                "to_status": target.value,
                            },
                            occurred_at=datetime.now(UTC),
                        )
                    )
                    session.commit()
                    return TaskClaim(task_id=task_id, status=target)
                session.rollback()
        return None

    def reject_block_and_mark_partial(
        self,
        *,
        task_id: int,
        block_id: int,
        reviewer_email: str,
        reason: str,
        discussion_id: str,
        webhook_event_id: str | None,
    ) -> str:
        """Persist REJECTED, task state and audit event in one transaction."""
        now = datetime.now(UTC)
        with self._sf() as session:
            task = session.execute(
                select(ResolveTask).where(ResolveTask.id == task_id).with_for_update()
            ).scalar_one_or_none()
            if task is None or task.status not in {
                TaskStatus.AWAITING_APPROVAL.value,
                TaskStatus.PARTIAL_REJECTED.value,
            }:
                return "invalid_task"
            block = session.execute(
                select(ConflictBlock).where(
                    ConflictBlock.id == block_id,
                    ConflictBlock.task_id == task_id,
                )
            ).scalar_one_or_none()
            if block is None:
                return "missing_block"
            approval = session.execute(
                select(Approval).where(Approval.block_id == block_id).with_for_update()
            ).scalar_one_or_none()
            if approval is None:
                return "missing_approval"
            if approval.status == "REJECTED":
                return "already_rejected"
            if approval.status == "APPROVED":
                return "currently_approved"
            if approval.status != "PENDING":
                return "invalid_approval"

            approval.status = "REJECTED"
            approval.reviewer_email = reviewer_email
            approval.reviewed_at = now
            approval.comment = reason
            approval.version += 1
            if task.status == TaskStatus.AWAITING_APPROVAL.value:
                task.status = TaskStatus.PARTIAL_REJECTED.value
                task.version += 1
            session.add(
                TaskEvent(
                    task_id=task.id,
                    event_type="BLOCK_REJECTED",
                    payload={
                        "conflict_id": block.conflict_id,
                        "discussion_id": discussion_id,
                        "reviewer_email": reviewer_email,
                        "reason": reason,
                        "retry_count": task.retry_count,
                        "webhook_event_id": webhook_event_id,
                    },
                    occurred_at=now,
                )
            )
            session.commit()
            return "rejected"

    def approve_block(
        self,
        *,
        task_id: int,
        block_id: int,
        reviewer_email: str,
        discussion_id: str,
        webhook_event_id: str | None,
    ) -> str:
        """Approve a block only while the task remains reviewable."""
        now = datetime.now(UTC)
        with self._sf() as session:
            task = session.execute(
                select(ResolveTask).where(ResolveTask.id == task_id).with_for_update()
            ).scalar_one_or_none()
            if task is None or task.status not in {
                TaskStatus.AWAITING_APPROVAL.value,
                TaskStatus.PARTIAL_REJECTED.value,
            }:
                return "invalid_task"
            block = session.execute(
                select(ConflictBlock).where(
                    ConflictBlock.id == block_id,
                    ConflictBlock.task_id == task_id,
                )
            ).scalar_one_or_none()
            if block is None:
                return "missing_block"
            approval = session.execute(
                select(Approval).where(Approval.block_id == block_id).with_for_update()
            ).scalar_one_or_none()
            if approval is None:
                return "missing_approval"
            if approval.status == "APPROVED":
                return "already_approved"
            if approval.status == "REJECTED":
                return "currently_rejected"
            if approval.status != "PENDING":
                return "invalid_approval"
            approval.status = "APPROVED"
            approval.reviewer_email = reviewer_email
            approval.reviewed_at = now
            approval.comment = None
            approval.version += 1
            session.add(
                TaskEvent(
                    task_id=task.id,
                    event_type="BLOCK_APPROVED",
                    payload={
                        "conflict_id": block.conflict_id,
                        "discussion_id": discussion_id,
                        "reviewer_email": reviewer_email,
                        "webhook_event_id": webhook_event_id,
                    },
                    occurred_at=now,
                )
            )
            session.commit()
            return "approved"

    def request_retry(
        self,
        *,
        task_id: int,
        requested_by: str,
        webhook_event_id: str | None,
    ) -> RetryRequest | None:
        """Freeze this retry's rejected/locked sets and queue it atomically."""
        now = datetime.now(UTC)
        with self._sf() as session:
            task = session.execute(
                select(ResolveTask).where(ResolveTask.id == task_id).with_for_update()
            ).scalar_one_or_none()
            if task is None or task.status != TaskStatus.PARTIAL_REJECTED.value:
                return None
            rows = list(
                session.execute(
                    select(
                        ConflictBlock.conflict_id,
                        Approval.status,
                        Approval.comment,
                    )
                    .join(Approval, Approval.block_id == ConflictBlock.id)
                    .where(ConflictBlock.task_id == task_id)
                    .order_by(ConflictBlock.block_idx)
                    .with_for_update()
                ).all()
            )
            rejected = tuple(conflict_id for conflict_id, status, _comment in rows if status == "REJECTED")
            if not rejected:
                return None
            locked = tuple(conflict_id for conflict_id, status, _comment in rows if status != "REJECTED")
            rejected_with_reasons = [
                {"conflict_id": conflict_id, "reason": comment}
                for conflict_id, status, comment in rows
                if status == "REJECTED"
            ]
            previous_commit = task.resolve_commit_sha
            previous_lark_url = task.lark_doc_url
            task.retry_count += 1
            task.status = TaskStatus.RETRYING.value
            task.last_error = None
            task.version += 1
            session.add(
                TaskEvent(
                    task_id=task.id,
                    event_type="RETRY_REQUESTED",
                    payload={
                        "attempt": task.retry_count + 1,
                        "rejected_conflict_ids": list(rejected),
                        "rejected": rejected_with_reasons,
                        "locked_conflict_ids": list(locked),
                        "previous_resolve_commit": previous_commit,
                        "previous_lark_doc_url": previous_lark_url,
                        "requested_by": requested_by,
                        "webhook_event_id": webhook_event_id,
                    },
                    occurred_at=now,
                )
            )
            session.commit()
            return RetryRequest(
                task_id=task.id,
                retry_count=task.retry_count,
                rejected_conflict_ids=rejected,
                locked_conflict_ids=locked,
            )

    def reset_rejected_approvals(
        self,
        *,
        task_id: int,
        conflict_ids: tuple[str, ...],
    ) -> int:
        """Reset only this retry snapshot after all external publishing succeeds."""
        if not conflict_ids:
            return 0
        with self._sf() as session:
            block_ids = list(
                session.execute(
                    select(ConflictBlock.id).where(
                        ConflictBlock.task_id == task_id,
                        ConflictBlock.conflict_id.in_(conflict_ids),
                    )
                ).scalars()
            )
            result = session.execute(
                update(Approval)
                .where(
                    Approval.block_id.in_(block_ids),
                    Approval.status == "REJECTED",
                )
                .values(
                    status="PENDING",
                    reviewer_email=None,
                    reviewed_at=None,
                    comment=None,
                    version=Approval.version + 1,
                )
            )
            session.commit()
            return int(result.rowcount or 0)  # type: ignore[attr-defined]

    def list_events(self, task_id: int) -> list[TaskEvent]:
        with self._sf() as session:
            return list(
                session.execute(
                    select(TaskEvent).where(TaskEvent.task_id == task_id).order_by(TaskEvent.occurred_at, TaskEvent.id)
                )
                .scalars()
                .all()
            )

    def list_by_status_updated_before(
        self,
        *,
        status: TaskStatus,
        threshold: datetime,
        limit: int = 10,
    ) -> list[ResolveTask]:
        with self._sf() as session:
            return list(
                session.execute(
                    select(ResolveTask)
                    .where(
                        ResolveTask.status == status.value,
                        ResolveTask.updated_at < threshold,
                    )
                    .order_by(ResolveTask.updated_at)
                    .limit(limit)
                )
                .scalars()
                .all()
            )

    def list_poll_candidates(
        self, *, status: TaskStatus, threshold: datetime,
        cursor: tuple[datetime, int] | None = None, limit: int = 10,
    ) -> list[tuple[int, datetime]]:
        query = select(ResolveTask.id, ResolveTask.updated_at).where(
            ResolveTask.status == status.value, ResolveTask.updated_at < threshold,
        )
        if cursor is not None:
            updated_at, task_id = cursor
            query = query.where(or_(ResolveTask.updated_at > updated_at, and_(
                ResolveTask.updated_at == updated_at, ResolveTask.id > task_id,
            )))
        if status == TaskStatus.FAILED:
            query = query.where(
                ResolveTask.trigger_payload["_failure_notice"]["state"].as_string().in_(["pending", "sending"])
            )
        if status == TaskStatus.MR_CREATED:
            query = query.where(ResolveTask.trigger_payload["_blank_approval_publication_verified"].as_boolean())
        with self._sf() as session:
            return [(row.id, row.updated_at) for row in session.execute(
                query.order_by(ResolveTask.updated_at, ResolveTask.id).limit(limit)
            )]

    def list_pending_failure_notices(self, limit: int = 10) -> list[ResolveTask]:
        with self._sf() as session:
            return list(session.scalars(
                select(ResolveTask).where(
                    ResolveTask.status == TaskStatus.FAILED.value,
                    ResolveTask.trigger_payload["_failure_notice"]["state"].as_string().in_(["pending", "sending"]),
                ).order_by(ResolveTask.updated_at).limit(limit)
            ).all())

    def update_status(
        self,
        *,
        task_id: int,
        from_version: int,
        new_status: TaskStatus,
        from_status: TaskStatus | None = None,
        **fields: Any,
    ) -> bool:
        if {"status", "version", "trace_schema_version"} & fields.keys():
            raise ValueError("reserved task fields")
        values: dict[str, Any] = {"status": new_status.value, **fields}
        if new_status in TERMINAL_STATUSES:
            values["active_key"] = None
        values["version"] = from_version + 1
        conditions = [ResolveTask.id == task_id, ResolveTask.version == from_version]
        if from_status is not None:
            conditions.append(ResolveTask.status == from_status.value)
        with self._sf() as session:
            previous = session.execute(select(ResolveTask.status).where(
                *conditions,
            ).with_for_update()).scalar_one_or_none()
            result = session.execute(
                update(ResolveTask)
                .where(
                    *conditions,
                )
                .values(**values)
            )
            if result.rowcount == 1:  # type: ignore[attr-defined]
                status_event(session.connection(), task_id, previous, new_status.value, from_version + 1)
                if new_status == TaskStatus.AWAITING_APPROVAL and previous != new_status.value:
                    self._enqueue_awaiting_approval(session, task_id)
            session.commit()
            # SQLAlchemy CursorResult 的 rowcount;类型 stub 标注为无此属性
            return bool(result.rowcount) and result.rowcount == 1  # type: ignore[attr-defined]

    @staticmethod
    def _enqueue_awaiting_approval(session: Session, task_id: int) -> None:
        task = session.execute(select(
            ResolveTask.trigger_type, ResolveTask.trigger_author_email,
            ResolveTask.new_mr_url, ResolveTask.retry_count,
        ).where(ResolveTask.id == task_id)).one()
        email = (task.trigger_author_email or "").strip()
        if task.trigger_type != "COMMENT" or not email or not task.new_mr_url:
            return
        # The existing (task_id, notification_type) key isolates each published round.
        notification_type = f"AWAITING_APPROVAL:{task.retry_count}"
        if session.scalar(select(LarkNotificationOutbox.id).where(
            LarkNotificationOutbox.task_id == task_id,
            LarkNotificationOutbox.notification_type == notification_type,
        )) is not None:
            return
        session.add(LarkNotificationOutbox(
            task_id=task_id,
            notification_type=notification_type,
            recipient_email=email,
            mr_url=task.new_mr_url,
            request_uuid=str(uuid5(NAMESPACE_URL, f"conciliator:{task_id}:{notification_type}")),
            status="PENDING",
            attempts=0,
            version=0,
        ))
        session.add(TaskEvent(
            task_id=task_id,
            event_type="LARK_NOTIFICATION_QUEUED",
            payload={
                "notification_type": notification_type,
                "retry_count": task.retry_count,
                "new_mr_url": task.new_mr_url,
            },
        ))

    def update_fields(
        self,
        *,
        task_id: int,
        from_version: int,
        **fields: Any,
    ) -> bool:
        if {"status", "version", "trace_schema_version"} & fields.keys():
            raise ValueError("use update_status for status transitions")
        values = {**fields, "version": from_version + 1}
        with self._sf() as session:
            result = session.execute(
                update(ResolveTask)
                .where(
                    ResolveTask.id == task_id,
                    ResolveTask.version == from_version,
                )
                .values(**values)
            )
            session.commit()
            return bool(result.rowcount) and result.rowcount == 1  # type: ignore[attr-defined]

    def save_conflict_blocks(
        self,
        task_id: int,
        blocks: list[dict],
    ) -> list[int]:
        created: list[int] = []
        with self._sf() as session:
            task = session.scalar(select(ResolveTask).where(ResolveTask.id == task_id).with_for_update())
            if task is None:
                raise RuntimeError("conflict task does not exist")
            existing = list(
                session.execute(
                    select(ConflictBlock).where(ConflictBlock.task_id == task_id).order_by(ConflictBlock.block_idx)
                ).scalars()
            )
            if existing:
                actual = [
                    (
                        block.conflict_id,
                        block.block_idx,
                        block.file_path,
                        block.start_line,
                        block.end_line,
                        block.ours_text,
                        block.theirs_text,
                    )
                    for block in existing
                ]
                expected = [
                    (
                        block["conflict_id"],
                        block["block_idx"],
                        block["file_path"],
                        block["start_line"],
                        block["end_line"],
                        block["ours_text"],
                        block["theirs_text"],
                    )
                    for block in blocks
                ]
                if actual != expected:
                    raise RuntimeError("stored conflict manifest differs from the frozen manifest")
                return [block.id for block in existing]
            for b in blocks:
                block = ConflictBlock(
                    task_id=task_id,
                    conflict_id=b["conflict_id"],
                    block_idx=b["block_idx"],
                    file_path=b["file_path"],
                    start_line=b["start_line"],
                    end_line=b["end_line"],
                    ours_text=b["ours_text"],
                    theirs_text=b["theirs_text"],
                    author_email=b.get("author_email"),
                )
                session.add(block)
                session.flush()
                exempt = is_blank_conflict(b.get("ours_text"), b.get("theirs_text"))
                session.add(Approval(block_id=block.id, status="NOT_REQUIRED" if exempt else "PENDING"))
                if exempt:
                    session.add(TaskEvent(
                        task_id=task_id, event_type="BLOCK_APPROVAL_NOT_REQUIRED",
                        payload={"conflict_id": block.conflict_id, "reason": "BLANK_ONLY",
                                 "source_line_count": len(block.ours_text.splitlines()),
                                 "target_line_count": len(block.theirs_text.splitlines())},
                    ))
                created.append(block.id)
            session.commit()
            return created

    def update_block_resolution(
        self,
        *,
        block_id: int,
        fields: dict[str, Any],
    ) -> None:
        allowed = {
            "ai_resolution",
            "author_email",
            "source_author_email",
            "source_author_commit_sha",
            "source_path",
            "source_start_line",
            "source_end_line",
            "target_author_email",
            "target_author_commit_sha",
            "target_path",
            "target_start_line",
            "target_end_line",
            "attribution_status",
            "decision_summary",
            "introduced_fix",
            "resolved_path",
            "resolved_start_line",
            "resolved_end_line",
        }
        unexpected = set(fields) - allowed
        if unexpected:
            raise ValueError(f"unsupported conflict block fields: {sorted(unexpected)}")
        with self._sf() as session:
            session.execute(update(ConflictBlock).where(ConflictBlock.id == block_id).values(**fields))
            session.commit()

    def update_block_discussion_and_resolution(
        self,
        *,
        block_id: int,
        discussion_id: str,
        ai_resolution: str,
    ) -> None:
        with self._sf() as session:
            session.execute(
                update(ConflictBlock)
                .where(ConflictBlock.id == block_id)
                .values(discussion_id=discussion_id, ai_resolution=ai_resolution)
            )
            session.commit()

    def update_block_ai_resolution(
        self,
        *,
        block_id: int,
        ai_resolution: str,
    ) -> None:
        with self._sf() as session:
            session.execute(
                update(ConflictBlock).where(ConflictBlock.id == block_id).values(ai_resolution=ai_resolution)
            )
            session.commit()

    def add_event(
        self,
        *,
        task_id: int,
        event_type: str,
        payload: dict,
        coco_event_id: str | None = None,
    ) -> bool:
        with self._sf() as session:
            evt = TaskEvent(
                task_id=task_id,
                event_type=event_type,
                payload=payload,
                coco_event_id=coco_event_id,
                occurred_at=datetime.now(UTC),
            )
            session.add(evt)
            try:
                session.commit()
            except IntegrityError:
                return False
            return True

    def complete_all_approved_and_enqueue(
        self,
        *,
        task_id: int,
        from_version: int,
        approved_count: int,
        no_manual_approval: bool = False,
    ) -> tuple[bool, bool]:
        """Atomically finish a task and enqueue its one-shot Lark notification."""
        now = datetime.now(UTC)
        allowed_statuses = (
            (TaskStatus.MR_CREATED.value,)
            if no_manual_approval else
            (TaskStatus.AWAITING_APPROVAL.value, TaskStatus.PARTIAL_REJECTED.value)
        )
        with self._sf() as session:
            task = session.execute(
                select(ResolveTask).where(
                    ResolveTask.id == task_id,
                    ResolveTask.version == from_version,
                    ResolveTask.status.in_(allowed_statuses),
                )
            ).scalar_one_or_none()
            if task is None:
                return False, False
            total, approved, exempt = session.execute(
                select(
                    func.count(ConflictBlock.id),
                    func.sum(case((Approval.status == "APPROVED", 1), else_=0)),
                    func.sum(case((Approval.status == "NOT_REQUIRED", 1), else_=0)),
                )
                .select_from(ConflictBlock)
                .outerjoin(Approval, Approval.block_id == ConflictBlock.id)
                .where(ConflictBlock.task_id == task_id)
            ).one()
            total, approved, exempt = int(total or 0), int(approved or 0), int(exempt or 0)
            if (
                not total or total != approved + exempt or approved != approved_count
                or (no_manual_approval and (
                    approved or not task.new_mr_id or not task.new_mr_url
                    or not task.resolve_commit_sha
                    or not task.trigger_payload.get("_blank_approval_publication_verified")
                ))
            ):
                session.rollback()
                return False, False
            previous_status = task.status
            result = session.execute(
                update(ResolveTask)
                .where(
                    ResolveTask.id == task_id,
                    ResolveTask.version == from_version,
                    ResolveTask.status.in_(allowed_statuses),
                )
                .values(
                    status=TaskStatus.ALL_APPROVED.value,
                    active_key=None,
                    version=from_version + 1,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:  # type: ignore[attr-defined]
                session.rollback()
                return False, False

            status_event(session.connection(), task_id, previous_status, "ALL_APPROVED", from_version + 1)
            notification_type = "NO_MANUAL_APPROVAL_REQUIRED" if no_manual_approval else "ALL_APPROVED"
            should_enqueue = bool(task.trigger_type == "COMMENT" and task.trigger_author_email and task.new_mr_url)
            session.add(
                TaskEvent(
                    task_id=task.id,
                    event_type="ALL_APPROVED",
                    payload={
                        "new_mr_id": task.new_mr_id,
                        "new_mr_url": task.new_mr_url,
                        "approved_count": approved_count,
                        "not_required_count": exempt,
                        "completion_reason": ("NO_MANUAL_APPROVAL_REQUIRED" if no_manual_approval
                                              else "MANUAL_APPROVAL_COMPLETE"),
                        "lark_notification_queued": should_enqueue,
                    },
                    occurred_at=now,
                )
            )
            if should_enqueue:
                session.add(
                    TaskEvent(
                        task_id=task.id,
                        event_type="LARK_NOTIFICATION_QUEUED",
                        payload={
                            "notification_type": notification_type,
                            "new_mr_id": task.new_mr_id,
                            "new_mr_url": task.new_mr_url,
                        },
                        occurred_at=now,
                    )
                )
                session.add(
                    LarkNotificationOutbox(
                        task_id=task.id,
                        notification_type=notification_type,
                        recipient_email=task.trigger_author_email,
                        mr_url=task.new_mr_url,
                        request_uuid=f"conciliator-all-approved-{task.id}",
                        status="PENDING",
                        attempts=0,
                        version=0,
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.commit()
            return True, should_enqueue

    def approve_all_pending_and_enqueue(
        self,
        *,
        task_id: int,
        reviewer_email: str,
        webhook_event_id: str | None,
    ) -> ApproveAllResult:
        """Approve every PENDING block and finish the task in one transaction.

        A rejected block blocks the whole command. This prevents a misleading
        partial update and guarantees that ALL_APPROVED and its Lark outbox row
        are committed together with the approvals.
        """
        now = datetime.now(UTC)
        allowed_statuses = {
            TaskStatus.AWAITING_APPROVAL.value,
            TaskStatus.PARTIAL_REJECTED.value,
        }
        with self._sf() as session:
            task = session.execute(
                select(ResolveTask)
                .where(ResolveTask.id == task_id)
                .with_for_update()
            ).scalar_one_or_none()
            if task is None or task.status not in allowed_statuses:
                return ApproveAllResult(outcome="invalid_task")

            rows = list(
                session.execute(
                    select(Approval, ConflictBlock)
                    .select_from(ConflictBlock)
                    .outerjoin(Approval, Approval.block_id == ConflictBlock.id)
                    .where(ConflictBlock.task_id == task_id)
                    .order_by(ConflictBlock.block_idx)
                    .with_for_update()
                ).all()
            )
            if not rows:
                return ApproveAllResult(outcome="no_blocks")

            if any(approval is None or approval.status not in set(ApprovalStatus) for approval, _block in rows):
                return ApproveAllResult(outcome="invalid_task")
            exempt_count = sum(approval.status == "NOT_REQUIRED" for approval, _block in rows)
            manual_count = len(rows) - exempt_count
            rejected_count = sum(
                approval.status == ApprovalStatus.REJECTED.value
                for approval, _block in rows
            )
            if rejected_count:
                return ApproveAllResult(
                    outcome="has_rejected",
                    total_count=manual_count,
                    rejected_count=rejected_count,
                )

            pending = [
                (approval, block)
                for approval, block in rows
                if approval.status == ApprovalStatus.PENDING.value
            ]
            for approval, block in pending:
                approval.status = ApprovalStatus.APPROVED.value
                approval.reviewer_email = reviewer_email
                approval.reviewed_at = now
                approval.comment = None
                approval.version += 1
                session.add(
                    TaskEvent(
                        task_id=task.id,
                        event_type="BLOCK_APPROVED",
                        payload={
                            "conflict_id": block.conflict_id,
                            "discussion_id": block.discussion_id,
                            "reviewer_email": reviewer_email,
                            "approved_via": "approve_all",
                            "webhook_event_id": webhook_event_id,
                        },
                        occurred_at=now,
                    )
                )

            task.status = TaskStatus.ALL_APPROVED.value
            task.active_key = None
            task.version += 1
            should_enqueue = bool(
                task.trigger_type == "COMMENT"
                and task.trigger_author_email
                and task.new_mr_url
            )
            session.add(
                TaskEvent(
                    task_id=task.id,
                    event_type="ALL_APPROVED",
                    payload={
                        "new_mr_id": task.new_mr_id,
                        "new_mr_url": task.new_mr_url,
                        "approved_count": manual_count,
                        "not_required_count": exempt_count,
                        "approved_via": "approve_all",
                        "lark_notification_queued": should_enqueue,
                    },
                    occurred_at=now,
                )
            )
            if should_enqueue:
                session.add(
                    TaskEvent(
                        task_id=task.id,
                        event_type="LARK_NOTIFICATION_QUEUED",
                        payload={
                            "notification_type": "ALL_APPROVED",
                            "new_mr_id": task.new_mr_id,
                            "new_mr_url": task.new_mr_url,
                        },
                        occurred_at=now,
                    )
                )
                session.add(
                    LarkNotificationOutbox(
                        task_id=task.id,
                        notification_type="ALL_APPROVED",
                        recipient_email=task.trigger_author_email,
                        mr_url=task.new_mr_url,
                        request_uuid=f"conciliator-all-approved-{task.id}",
                        status="PENDING",
                        attempts=0,
                        version=0,
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.commit()
            return ApproveAllResult(
                outcome="completed",
                total_count=manual_count,
                approved_now_count=len(pending),
                notification_queued=should_enqueue,
            )

    def cancel_task(
        self,
        *,
        task_id: int,
        from_version: int,
        cancelled_by: str,
        webhook_event_id: str | None,
    ) -> CancelTaskResult:
        """Atomically move one non-terminal task to CANCELLED with an audit event."""
        now = datetime.now(UTC)
        with self._sf() as session:
            task = session.execute(
                select(ResolveTask)
                .where(ResolveTask.id == task_id)
                .with_for_update()
            ).scalar_one_or_none()
            if (
                task is None
                or task.version != from_version
                or TaskStatus(task.status) in TERMINAL_STATUSES
            ):
                return CancelTaskResult(cancelled=False)

            previous_status = task.status
            result = session.execute(
                update(ResolveTask).where(
                    ResolveTask.id == task_id, ResolveTask.version == from_version,
                    ResolveTask.status == previous_status,
                ).values(status=TaskStatus.CANCELLED.value, active_key=None, version=from_version + 1, updated_at=now)
            )
            if result.rowcount != 1:  # type: ignore[attr-defined]
                session.rollback()
                return CancelTaskResult(cancelled=False)
            status_event(session.connection(), task_id, previous_status, "CANCELLED", from_version + 1)
            session.add(
                TaskEvent(
                    task_id=task.id,
                    event_type="TASK_CANCELLED",
                    payload={
                        "from_status": previous_status,
                        "cancelled_by": cancelled_by,
                        "webhook_event_id": webhook_event_id,
                        "resolve_branch": task.resolve_branch,
                        "new_mr_id": task.new_mr_id,
                    },
                    occurred_at=now,
                )
            )
            session.commit()
            return CancelTaskResult(
                cancelled=True,
                previous_status=previous_status,
            )

    def claim_lark_notification(
        self,
        *,
        stale_before: datetime,
        limit: int = 10,
    ) -> LarkNotificationOutbox | None:
        """Claim one due notification across all worker processes and Pods."""
        now = datetime.now(UTC)
        with self._sf() as session:
            candidates = list(
                session.execute(
                    select(
                        LarkNotificationOutbox.id,
                        LarkNotificationOutbox.version,
                        LarkNotificationOutbox.status,
                    )
                    .where(
                        or_(
                            and_(
                                LarkNotificationOutbox.status.in_(("PENDING", "RETRY")),
                                or_(
                                    LarkNotificationOutbox.next_retry_at.is_(None),
                                    LarkNotificationOutbox.next_retry_at <= now,
                                ),
                            ),
                            and_(
                                LarkNotificationOutbox.status == "SENDING",
                                LarkNotificationOutbox.updated_at < stale_before,
                            ),
                        )
                    )
                    .order_by(
                        LarkNotificationOutbox.next_retry_at,
                        LarkNotificationOutbox.id,
                    )
                    .limit(limit)
                ).all()
            )
            for notification_id, version, status in candidates:
                result = session.execute(
                    update(LarkNotificationOutbox)
                    .where(
                        LarkNotificationOutbox.id == notification_id,
                        LarkNotificationOutbox.version == version,
                        LarkNotificationOutbox.status == status,
                    )
                    .values(
                        status="SENDING",
                        attempts=LarkNotificationOutbox.attempts + 1,
                        next_retry_at=None,
                        version=version + 1,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:  # type: ignore[attr-defined]
                    session.commit()
                    return session.get(LarkNotificationOutbox, notification_id)
                session.rollback()
        return None

    def mark_lark_notification_sent(
        self,
        *,
        notification_id: int,
        from_version: int,
        recipient_open_id: str,
        message_id: str,
    ) -> bool:
        with self._sf() as session:
            result = session.execute(
                update(LarkNotificationOutbox)
                .where(
                    LarkNotificationOutbox.id == notification_id,
                    LarkNotificationOutbox.status == "SENDING",
                    LarkNotificationOutbox.version == from_version,
                )
                .values(
                    status="SENT",
                    recipient_open_id=recipient_open_id,
                    message_id=message_id,
                    last_error=None,
                    version=from_version + 1,
                    updated_at=datetime.now(UTC),
                )
            )
            session.commit()
            return bool(result.rowcount) and result.rowcount == 1  # type: ignore[attr-defined]

    def mark_lark_notification_failed(
        self,
        *,
        notification_id: int,
        from_version: int,
        recipient_open_id: str | None,
        error: str,
        retry_at: datetime | None,
    ) -> bool:
        with self._sf() as session:
            result = session.execute(
                update(LarkNotificationOutbox)
                .where(
                    LarkNotificationOutbox.id == notification_id,
                    LarkNotificationOutbox.status == "SENDING",
                    LarkNotificationOutbox.version == from_version,
                )
                .values(
                    status="RETRY" if retry_at is not None else "FAILED",
                    recipient_open_id=recipient_open_id,
                    next_retry_at=retry_at,
                    last_error=error[:2000],
                    version=from_version + 1,
                    updated_at=datetime.now(UTC),
                )
            )
            session.commit()
            return bool(result.rowcount) and result.rowcount == 1  # type: ignore[attr-defined]

    def get_lark_notification(
        self,
        notification_id: int,
    ) -> LarkNotificationOutbox | None:
        with self._sf() as session:
            return session.get(LarkNotificationOutbox, notification_id)

    def get_lark_notification_for_task(
        self,
        *,
        task_id: int,
        notification_type: str,
    ) -> LarkNotificationOutbox | None:
        with self._sf() as session:
            return session.execute(
                select(LarkNotificationOutbox).where(
                    LarkNotificationOutbox.task_id == task_id,
                    LarkNotificationOutbox.notification_type == notification_type,
                )
            ).scalar_one_or_none()

    def claim_nextcode_webhook(
        self,
        *,
        delivery_key: str,
        event_id: str | None,
        delivery_id: str | None,
        event_type: str,
    ) -> bool:
        now = datetime.now(UTC)
        with self._sf() as session:
            delivery = NextCodeWebhookDelivery(
                delivery_key=delivery_key,
                event_id=event_id or None,
                delivery_id=delivery_id or None,
                event_type=event_type,
                status="PROCESSING",
                processed_count=0,
                created_at=now,
                updated_at=now,
            )
            session.add(delivery)
            try:
                session.commit()
                return True
            except IntegrityError:
                session.rollback()

            result = session.execute(
                update(NextCodeWebhookDelivery)
                .where(
                    NextCodeWebhookDelivery.delivery_key == delivery_key,
                    NextCodeWebhookDelivery.status == "FAILED",
                )
                .values(
                    status="PROCESSING",
                    processed_count=0,
                    last_error=None,
                    updated_at=now,
                )
            )
            session.commit()
            return bool(result.rowcount) and result.rowcount == 1  # type: ignore[attr-defined]

    def complete_nextcode_webhook(self, *, delivery_key: str, processed_count: int) -> None:
        with self._sf() as session:
            session.execute(
                update(NextCodeWebhookDelivery)
                .where(
                    NextCodeWebhookDelivery.delivery_key == delivery_key,
                    NextCodeWebhookDelivery.status == "PROCESSING",
                )
                .values(
                    status="COMPLETED",
                    processed_count=processed_count,
                    last_error=None,
                    updated_at=datetime.now(UTC),
                )
            )
            session.commit()

    def fail_nextcode_webhook(self, *, delivery_key: str, error: str) -> None:
        with self._sf() as session:
            session.execute(
                update(NextCodeWebhookDelivery)
                .where(
                    NextCodeWebhookDelivery.delivery_key == delivery_key,
                    NextCodeWebhookDelivery.status == "PROCESSING",
                )
                .values(
                    status="FAILED",
                    last_error=error[:1000],
                    updated_at=datetime.now(UTC),
                )
            )
            session.commit()

    def find_block_by_discussion(
        self,
        discussion_id: str,
    ) -> tuple[ConflictBlock, ResolveTask] | None:
        with self._sf() as session:
            row = session.execute(
                select(ConflictBlock, ResolveTask)
                .join(ResolveTask, ConflictBlock.task_id == ResolveTask.id)
                .where(ConflictBlock.discussion_id == discussion_id)
                .options(selectinload(ConflictBlock.approval))
            ).first()
            if row is None:
                return None
            return row[0], row[1]

    def get_repo_config(self, repo_id: str) -> RepoConfig | None:
        with self._sf() as session:
            return session.get(RepoConfig, repo_id)
