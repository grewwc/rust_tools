from collections.abc import Iterable

from sqlalchemy.orm import Session, sessionmaker

from conciliator.domain.approval import ApprovalStatus, approval_counts
from conciliator.domain.commands import (
    ApproveAll,
    ApproveBlock,
    Cancel,
    Ignored,
    RejectBlock,
    Retry,
    Status,
    parse_comment,
)
from conciliator.domain.task import TaskStatus
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.discussions import reply_discussion
from conciliator.integrations.nextcode.mrs import (
    add_mr_comment,
    get_mr,
    update_mr_description,
)
from conciliator.repositories.models import ConflictBlock, ResolveTask
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import templates
from conciliator.services.notification_service import NotificationService
from conciliator.services.task_status import original_mr_status_reply
from conciliator.utils.logging import get_logger
from conciliator.utils.metrics import MetricName, emit_counter
from conciliator.utils.trace_context import observe_task, traced_work

log = get_logger(__name__)


class ApprovalService:
    def __init__(
        self,
        *,
        nc: NextCodeClient,
        repo: TaskRepository,
        notification: NotificationService,
        session_factory: sessionmaker[Session],
        super_admin_emails: Iterable[str] = (),
    ):
        self._nc = nc
        self._repo = repo
        self._notification = notification
        self._sf = session_factory
        self._super_admin_emails = frozenset(
            email.strip().lower() for email in super_admin_emails if email.strip()
        )

    @traced_work
    async def handle_comment(
        self,
        *,
        repo_id: str,
        mr_iid: int,
        body: str,
        author_email: str,
        discussion_id: str | None,
        webhook_event_id: str | None = None,
    ) -> None:
        in_discussion = discussion_id is not None
        cmd = parse_comment(body, in_discussion=in_discussion)

        if isinstance(cmd, Ignored):
            return

        if isinstance(cmd, Cancel):
            await self._route_cancel(
                repo_id, mr_iid, cmd, author_email, discussion_id, webhook_event_id,
            )
            return

        if isinstance(cmd, Status):
            await self._route_status(repo_id, mr_iid, discussion_id)
            return

        task = await self._repo.find_active_for_resolve_mr(repo_id, mr_iid)
        if task is None:
            log.info("no_active_task_for_comment", repo_id=repo_id, mr_iid=mr_iid)
            return

        log.info("resolve_command_received", task_id=task.id, command=type(cmd).__name__, task_status=task.status)
        if isinstance(cmd, (ApproveAll, Retry)):
            if not await self._can_manage_task(task, author_email):
                await self._reply_no_permission(task, author_email)
                return
            if isinstance(cmd, ApproveAll):
                await self._handle_approve_all(
                    task,
                    author_email,
                    discussion_id=discussion_id,
                    webhook_event_id=webhook_event_id,
                )
            elif isinstance(cmd, Retry):
                await self._handle_retry(
                    task,
                    author_email,
                    webhook_event_id=webhook_event_id,
                )
            return

        # 单块指令需要 discussion_id
        if not discussion_id:
            return
        block_task = await self._repo.find_block_by_discussion(discussion_id)
        if block_task is None:
            return
        block, block_task_obj = block_task
        if block_task_obj.id != task.id or block_task_obj.new_mr_id != mr_iid:
            log.warning(
                "discussion_task_mismatch",
                task_id=task.id,
                discussion_id=discussion_id,
                mr_iid=mr_iid,
            )
            return

        # 此时 task 已过 MR_CREATED,new_mr_id 一定有值(类型上可能 None)
        assert task.new_mr_id is not None

        if task.status not in (
            TaskStatus.AWAITING_APPROVAL.value,
            TaskStatus.PARTIAL_REJECTED.value,
        ):
            await reply_discussion(
                self._nc,
                repo_id,
                task.new_mr_id,
                discussion_id,
                f"任务当前状态 {task.status},不可操作",
            )
            return

        if not await self._can_operate_block(block, task, author_email):
            await reply_discussion(
                self._nc,
                repo_id,
                task.new_mr_id,
                discussion_id,
                "无权限，仅 block author、`/resolve` 发起人或超级管理员可操作",
            )
            return

        if isinstance(cmd, ApproveBlock):
            await self._handle_approve_block(
                task,
                block,
                author_email,
                discussion_id,
                webhook_event_id=webhook_event_id,
            )
        elif isinstance(cmd, RejectBlock):
            await self._handle_reject_block(
                task,
                block,
                author_email,
                cmd.reason,
                discussion_id,
                webhook_event_id=webhook_event_id,
            )

    async def _route_status(self, repo_id: str, mr_iid: int, discussion_id: str | None) -> None:
        if not discussion_id:
            log.error("status_comment_missing_thread", repo_id=repo_id, mr_iid=mr_iid)
            raise ValueError("/resolve status comment requires ThreadId")
        task = await self._repo.find_for_resolve_mr(repo_id, mr_iid)
        if task is not None:
            await self._reply_status(task, discussion_id, reply_mr_iid=mr_iid)
            return

        tasks = await self._repo.list_active_for_mr(repo_id, mr_iid)
        if not tasks:
            latest = await self._repo.find_latest_for_original_mr(repo_id, mr_iid)
            tasks = [latest] if latest is not None else []
        await reply_discussion(
            self._nc, repo_id, mr_iid, discussion_id, original_mr_status_reply(tasks),
        )

    async def _route_cancel(
        self, repo_id: str, mr_iid: int, cmd: Cancel, author_email: str,
        discussion_id: str | None, webhook_event_id: str | None,
    ) -> None:
        task = await self._repo.find_active_for_resolve_mr(repo_id, mr_iid)
        if task is not None:
            candidates = [task] if cmd.task_id in (None, task.id) else []
        else:
            candidates = await self._repo.list_active_for_mr(repo_id, mr_iid)
            if cmd.task_id is not None:
                candidates = [item for item in candidates if item.id == cmd.task_id]
            elif discussion_id:
                in_thread = [item for item in candidates
                             if item.trigger_payload.get("_trigger_discussion_id") == discussion_id]
                if in_thread:
                    candidates = in_thread
        if len(candidates) != 1:
            if candidates:
                body = "当前 MR 有多个运行中的任务，请指定要取消的任务：\n" + "\n".join(
                    f"- #{item.id}（{(item.source_commit_sha or '')[:12]} → "
                    f"{(item.target_commit_sha or '')[:12]}）：`/resolve cancel {item.id}`"
                    for item in candidates
                )
            else:
                body = "当前 MR 下没有匹配的非终态任务。"
            if discussion_id:
                await reply_discussion(self._nc, repo_id, mr_iid, discussion_id, body)
            else:
                await add_mr_comment(self._nc, repo_id, mr_iid, body)
            return
        task = observe_task(candidates[0])
        log.info("resolve_command_received", task_id=task.id, command="Cancel", task_status=task.status)
        if not await self._can_manage_task(task, author_email):
            await self._reply_command(
                task, discussion_id, "无权限，仅 `/resolve` 发起人或超级管理员可取消该任务。", reply_mr_iid=mr_iid,
            )
            return
        await self._handle_cancel(
            task, author_email, discussion_id=discussion_id,
            webhook_event_id=webhook_event_id, reply_mr_iid=mr_iid,
        )

    async def _can_manage_task(
        self,
        task: ResolveTask,
        email: str,
    ) -> bool:
        normalized = email.strip().lower()
        return bool(normalized) and (
            normalized in self._super_admin_emails
            or normalized == (task.trigger_author_email or "").strip().lower()
        )

    async def _can_operate_block(
        self,
        block: ConflictBlock,
        task: ResolveTask,
        email: str,
    ) -> bool:
        allowed_authors = {
            author.strip().lower()
            for author in (
                block.source_author_email,
                block.target_author_email,
                block.author_email,
            )
            if author
        }
        if email.strip() and email.strip().lower() in allowed_authors:
            return True
        return await self._can_manage_task(task, email)

    async def _handle_approve_block(
        self,
        task,
        block,
        reviewer_email,
        discussion_id,
        *,
        webhook_event_id: str | None,
    ) -> None:
        result = await self._repo.approve_block(
            task_id=task.id,
            block_id=block.id,
            reviewer_email=reviewer_email,
            discussion_id=discussion_id,
            webhook_event_id=webhook_event_id,
        )
        if result == "currently_rejected":
            await reply_discussion(
                self._nc,
                task.repo_id,
                task.new_mr_id,
                discussion_id,
                "当前冲突块处于 REJECTED，请先在 MR 主线执行 `/resolve retry`，"
                "重解完成后再回复 `/approval`。",
            )
            return
        if result not in {"approved", "already_approved"}:
            return

        full = await self._sync_mr_description(task.id)
        approved = sum(
            bool(item.approval and item.approval.status == ApprovalStatus.APPROVED.value) for item in full.blocks
        )
        counts = approval_counts(full.blocks)
        total = counts["PENDING"] + counts["APPROVED"] + counts["REJECTED"]
        suffix = "（已通过）" if result == "already_approved" else ""
        await reply_discussion(
            self._nc,
            task.repo_id,
            task.new_mr_id,
            discussion_id,
            f"✅ Block #{block.block_idx + 1} 已 approve（{approved}/{total}）{suffix}",
        )
        if result == "approved":
            emit_counter(MetricName.APPROVAL_TOTAL, 1, tags={"action": "approve"})
        if approved == total and total and not counts["UNKNOWN"]:
            await self._complete_all_approved(full)

    async def _handle_reject_block(
        self,
        task,
        block,
        reviewer_email,
        reason,
        discussion_id,
        *,
        webhook_event_id: str | None,
    ) -> None:
        result = await self._repo.reject_block_and_mark_partial(
            task_id=task.id,
            block_id=block.id,
            reviewer_email=reviewer_email,
            reason=reason,
            discussion_id=discussion_id,
            webhook_event_id=webhook_event_id,
        )
        if result == "currently_approved":
            await reply_discussion(
                self._nc,
                task.repo_id,
                task.new_mr_id,
                discussion_id,
                "当前冲突块处于 APPROVED，不能再回复 `/reject <reason>`。",
            )
            return
        if result not in {"rejected", "already_rejected"}:
            return
        full = await self._sync_mr_description(task.id)
        rejected = sum(
            bool(item.approval and item.approval.status == ApprovalStatus.REJECTED.value) for item in full.blocks
        )
        await reply_discussion(
            self._nc,
            task.repo_id,
            task.new_mr_id,
            discussion_id,
            f"❌ Block #{block.block_idx + 1} 已 reject：{reason}"
            f"（当前 {rejected} 块待重解）\n"
            "可在 MR 主线回复 /resolve retry 重新解这些块",
        )
        if result == "rejected":
            emit_counter(MetricName.APPROVAL_TOTAL, 1, tags={"action": "reject"})

    async def _handle_approve_all(
        self,
        task: ResolveTask,
        reviewer_email: str,
        *,
        discussion_id: str | None,
        webhook_event_id: str | None,
    ) -> None:
        result = await self._repo.approve_all_pending_and_enqueue(
            task_id=task.id,
            reviewer_email=reviewer_email,
            webhook_event_id=webhook_event_id,
        )
        if result.outcome == "has_rejected":
            await self._reply_command(
                task,
                discussion_id,
                "无法执行 `/resolve approve all`："
                f"当前有 {result.rejected_count} 个 REJECTED 冲突块。"
                "请先在 MR 主线执行 `/resolve retry`，重解完成后再执行。",
            )
            return
        if result.outcome != "completed":
            await self._reply_command(
                task,
                discussion_id,
                "当前任务状态不能执行 `/resolve approve all`。",
            )
            return

        full = await self._repo.get_with_blocks(task.id)
        if full is None:
            raise RuntimeError("resolve task disappeared after approve all")
        await self._reply_command(
            full,
            discussion_id,
            f"✅ 已将 {result.approved_now_count} 个 PENDING 冲突块设为 APPROVED；"
            f"全部 {result.total_count} 个需人工审批的冲突块已通过。",
        )
        await self._publish_all_approved(full)

    async def _handle_retry(
        self,
        task,
        requested_by: str,
        *,
        webhook_event_id: str | None,
    ) -> None:
        request = await self._repo.request_retry(
            task_id=task.id,
            requested_by=requested_by,
            webhook_event_id=webhook_event_id,
        )
        if request is None:
            await add_mr_comment(
                self._nc,
                task.repo_id,
                task.new_mr_id,
                "当前状态不能重解，或没有 REJECTED 状态的冲突块。",
            )
            return
        log.info("resolve_retry_queued", task_id=task.id, retry_count=request.retry_count)
        await add_mr_comment(
            self._nc,
            task.repo_id,
            task.new_mr_id,
            f"🔄 已请求第 {request.retry_count} 次重解，仅处理 {', '.join(request.rejected_conflict_ids)}。",
        )

    async def _handle_cancel(
        self,
        task: ResolveTask,
        cancelled_by: str,
        *,
        discussion_id: str | None,
        webhook_event_id: str | None,
        reply_mr_iid: int | None = None,
    ) -> None:
        result = await self._repo.cancel_task(
            task_id=task.id,
            from_version=task.version,
            cancelled_by=cancelled_by,
            webhook_event_id=webhook_event_id,
        )
        if not result.cancelled:
            log.warning(
                "cancel_blocked_by_task_change",
                task_id=task.id,
                from_version=task.version,
            )
            await self._reply_command(
                task,
                discussion_id,
                "任务状态刚刚发生变化，本次取消未成功。"
                f"请执行 `/resolve status` 查看后，再重试 `/resolve cancel {task.id}`。",
                reply_mr_iid=reply_mr_iid,
            )
            return

        log.info("resolve_task_cancelled", task_id=task.id, from_status=task.status)
        task.version += 1
        task.status = TaskStatus.CANCELLED.value
        observe_task(task)
        task.active_key = None
        try:
            if task.new_mr_id is not None:
                await self._sync_mr_description(task.id)
        except Exception:
            log.exception(
                "cancelled_description_sync_failed",
                task_id=task.id,
                mr_iid=task.new_mr_id,
            )
        original_mr = (
            f"[原始 MR]({task.original_mr_url})" if task.original_mr_url
            else f"原始 MR !{task.original_mr_id}"
        )
        await self._reply_command(
            task,
            discussion_id,
            f"任务 #{task.id} 已取消，已解除当前代码版本的任务占用。\n"
            f"如需从头重试，请在{original_mr}再次评论 `/resolve`。新任务将使用当前分支版本。\n"
            "已创建的分支、MR 和报告会保留；已经发出的外部请求可能仍会完成。",
            reply_mr_iid=reply_mr_iid,
        )

    async def _sync_mr_description(self, task_id: int) -> ResolveTask:
        """Update the resolver MR until it matches one stable approval snapshot."""
        for _ in range(4):
            task = await self._repo.get_with_blocks(task_id)
            if task is None:
                raise RuntimeError("resolve task disappeared during description sync")
            if task.new_mr_id is None:
                raise RuntimeError("resolve task has no resolver MR")
            before = _approval_fingerprint(task)
            description = self._render_mr_description(task)
            await update_mr_description(
                self._nc,
                task.repo_id,
                task.new_mr_id,
                description,
            )
            readback = await get_mr(
                self._nc,
                task.repo_id,
                task.new_mr_id,
            )
            if readback.description != description:
                raise RuntimeError("resolver MR description readback differs after approval")
            latest = await self._repo.get_with_blocks(task_id)
            if latest is None:
                raise RuntimeError("resolve task disappeared after description readback")
            if _approval_fingerprint(latest) == before:
                return latest
        raise RuntimeError("approval state did not stabilize during description sync")

    @staticmethod
    def _render_mr_description(task: ResolveTask) -> str:
        block_dicts = [
            {
                "idx": block.block_idx + 1,
                "conflict_id": block.conflict_id,
                "file_path": block.file_path,
                "start_line": block.start_line,
                "end_line": block.end_line,
                "source_author_email": block.source_author_email,
                "target_author_email": block.target_author_email,
                "resolved_path": block.resolved_path,
                "resolved_start_line": block.resolved_start_line,
                "resolved_end_line": block.resolved_end_line,
                "approved": bool(block.approval and block.approval.status == ApprovalStatus.APPROVED.value),
                "approval_status": (block.approval.status if block.approval else "UNKNOWN"),
                "reject_reason": (
                    getattr(block.approval, "comment", None)
                    if block.approval and block.approval.status == ApprovalStatus.REJECTED.value
                    else None
                ),
            }
            for block in sorted(task.blocks, key=lambda item: item.block_idx)
        ]
        return templates.build_mr_description(
            original_mr_number=task.original_mr_id,
            source_branch=task.source_branch,
            source_commit=task.source_commit_sha or "",
            target_branch=task.target_branch,
            target_commit=task.target_commit_sha or "",
            resolve_branch=task.resolve_branch,
            resolve_commit=task.resolve_commit_sha or "",
            lark_doc_url=task.lark_doc_url or "",
            blocks=block_dicts,
            retry_count=getattr(task, "retry_count", 0),
            task_status=task.status,
        )

    async def _complete_all_approved(self, task: ResolveTask) -> None:
        if task.new_mr_id is None:
            raise RuntimeError("resolve task has no resolver MR")
        ok, _notification_queued = await self._repo.complete_all_approved_and_enqueue(
            task_id=task.id,
            from_version=task.version,
            approved_count=approval_counts(task.blocks)["APPROVED"],
        )
        if not ok:
            log.info(
                "all_approved_completion_skipped",
                task_id=task.id,
                from_version=task.version,
            )
            return
        task.status = TaskStatus.ALL_APPROVED.value
        observe_task(task)
        task.active_key = None
        task.version += 1
        await self._publish_all_approved(task)

    async def _publish_all_approved(self, task: ResolveTask) -> None:
        """Refresh user-visible state after the atomic ALL_APPROVED commit."""
        mr_iid = task.new_mr_id
        if mr_iid is None:
            raise RuntimeError("resolve task has no resolver MR")
        log.info("resolve_all_approved", task_id=task.id, new_mr_id=mr_iid)
        try:
            await self._sync_mr_description(task.id)
        except Exception:
            log.exception(
                "all_approved_description_sync_failed",
                task_id=task.id,
                mr_iid=mr_iid,
            )
        try:
            await self._notification.notify_mr_comment(
                project_id=task.repo_id,
                mr_iid=mr_iid,
                body="本任务冻结版本中需要人工审批的冲突块已全部 approve，请结合 MR 最新检查结果确认是否合入。",
                mention_emails=(
                    [task.trigger_author_email]
                    if task.trigger_author_email
                    else None
                ),
            )
        except Exception:
            log.exception(
                "all_approved_notification_failed",
                task_id=task.id,
                mr_iid=mr_iid,
            )

    async def _reply_command(
        self,
        task: ResolveTask,
        discussion_id: str | None,
        body: str,
        *,
        reply_mr_iid: int | None = None,
    ) -> None:
        mr_iid = reply_mr_iid if reply_mr_iid is not None else task.new_mr_id
        if mr_iid is None:
            raise RuntimeError("resolve task has no resolver MR")
        if discussion_id:
            await reply_discussion(
                self._nc,
                task.repo_id,
                mr_iid,
                discussion_id,
                body,
            )
            return
        await add_mr_comment(
            self._nc,
            task.repo_id,
            mr_iid,
            body,
        )

    async def _reply_status(
        self,
        task: ResolveTask,
        discussion_id: str,
        *,
        reply_mr_iid: int,
    ) -> None:
        full = await self._repo.get_with_blocks(task.id)
        if full is None:
            raise RuntimeError(f"Status task {task.id} disappeared during read")
        counts = approval_counts(full.blocks)
        lines = [
            f"任务状态：`{full.status}`",
            (
                f"进度：{counts[ApprovalStatus.APPROVED.value]}/"
                f"{counts['PENDING'] + counts['APPROVED'] + counts['REJECTED']} APPROVED，"
                f"{counts[ApprovalStatus.REJECTED.value]} REJECTED，"
                f"{counts[ApprovalStatus.PENDING.value]} PENDING"
            ),
            "",
            f"另有 {counts['NOT_REQUIRED']} 块纯空行冲突无需人工审批；{counts['UNKNOWN']} 块审批记录异常。",
            "",
            "| Block | File | Source author | Target author | Status |",
            "|---|---|---|---|---|",
        ]
        for b in sorted(full.blocks, key=lambda item: item.block_idx):
            status = b.approval.status if b.approval else "UNKNOWN"
            if status == "NOT_REQUIRED":
                status = "NOT_REQUIRED（纯空行，无需人工审批）"
            location = f"{b.file_path}:{b.start_line}-{b.end_line}"
            lines.append(
                f"| #{b.block_idx + 1} | {_markdown_table_cell(location)} | "
                f"{_markdown_table_cell(b.source_author_email or b.author_email or '-')} | "
                f"{_markdown_table_cell(b.target_author_email or '-')} | {status} |"
            )
        await self._reply_command(
            full,
            discussion_id,
            "\n".join(lines),
            reply_mr_iid=reply_mr_iid,
        )

    async def _reply_no_permission(self, task, email) -> None:
        await add_mr_comment(
            self._nc,
            task.repo_id,
            task.new_mr_id,
            f"@{email} 无权限执行此指令（限 `/resolve` 发起人或超级管理员）",
        )


def _approval_fingerprint(
    task: ResolveTask,
) -> tuple[tuple[int, str, str | None], ...]:
    return tuple(
        (
            block.id,
            block.approval.status if block.approval else "MISSING",
            getattr(block.approval, "comment", None) if block.approval else None,
        )
        for block in sorted(task.blocks, key=lambda item: item.block_idx)
    )


def _markdown_table_cell(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")
