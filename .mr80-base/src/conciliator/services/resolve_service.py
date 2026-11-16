import asyncio
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from time import monotonic

from conciliator.config import Settings
from conciliator.domain.approval import ApprovalStatus, approval_not_required
from conciliator.domain.processing_version import build_active_key
from conciliator.domain.resolve_branch import build_resolve_branch
from conciliator.domain.task import (
    COCO_POLL_CLAIMS,
    COCO_PROCESSING_TIMEOUT_SECONDS,
    InvalidTransitionError,
    TaskStatus,
    can_transition,
)
from conciliator.integrations.coco.client import (
    CocoClient,
    CocoError,
    CocoResolveResult,
    CocoRetryResult,
    CocoTaskResult,
    ConflictAttribution,
    parse_retry_result,
    result_to_dict,
)
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.lark.drive import upload_markdown
from conciliator.integrations.nextcode.branches import ensure_branch, get_branch
from conciliator.integrations.nextcode.client import NextCodeClient, NextCodeError
from conciliator.integrations.nextcode.commits import get_commit
from conciliator.integrations.nextcode.discussions import (
    create_discussion,
    find_position_for_version,
    get_thread,
    reply_discussion,
)
from conciliator.integrations.nextcode.files import get_file
from conciliator.integrations.nextcode.merges import (
    ConflictFile,
    ConflictSnapshotMismatchError,
    list_conflict_files,
)
from conciliator.integrations.nextcode.mrs import (
    MR,
    create_mr,
    get_mr,
    list_repo_mrs,
    update_mr_description,
)
from conciliator.integrations.nextcode.repos import get_blame
from conciliator.repositories._sync_repo import ActiveTaskExistsError
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import templates
from conciliator.services.blank_conflicts import verify_blank_files
from conciliator.services.local_artifacts import (
    LocalReportArtifacts,
    write_local_report_artifacts,
)
from conciliator.services.notification_service import NotificationService, send_resolve_feedback
from conciliator.utils.logging import get_logger
from conciliator.utils.metrics import MetricName, emit_counter
from conciliator.utils.redaction import safe_text
from conciliator.utils.retry import TransientError
from conciliator.utils.task_links import task_trace_url
from conciliator.utils.trace_context import observe_task, traced_work

log = get_logger(__name__)


def _logged_stage(fn):
    @wraps(fn)
    async def wrapped(self, task, *args, **kwargs):
        stage = fn.__name__.lstrip("_")
        fields = {"task_id": task.id, "stage": stage}
        started = monotonic()
        log.info("resolve_stage_started", **fields, task_status=getattr(task, "status", None))
        try:
            result = await fn(self, task, *args, **kwargs)
        except asyncio.CancelledError:
            log.warning("resolve_stage_interrupted", **fields, duration_ms=int((monotonic() - started) * 1000))
            raise
        except Exception as exc:
            log.error(
                "resolve_stage_failed", **fields, error_type=type(exc).__name__,
                duration_ms=int((monotonic() - started) * 1000),
            )
            raise
        log.info(
            "resolve_stage_finished", **fields, task_status=getattr(task, "status", None),
            duration_ms=int((monotonic() - started) * 1000),
        )
        return result

    return wrapped


class UnsupportedConflictError(RuntimeError):
    """Frozen inputs require a conflict mode that this resolver does not support."""


def _check_supported_conflicts(task) -> None:
    for item in getattr(task, "conflict_files_snapshot", None) or []:
        first, second = item.get("first_path"), item.get("second_path")
        if first and second and first != second:
            raise UnsupportedConflictError(
                f"暂不支持文件重命名冲突：两侧路径不同（{first} → {second}）。请人工处理该冲突。"
            )


class CocoValidationError(RuntimeError):
    """A model-fixable delivery/contract failure, not a transient dependency outage."""


@dataclass
class ResolveDeps:
    nc: NextCodeClient
    coco: CocoClient
    lark: LarkClient | None
    repo: TaskRepository
    notification: NotificationService | None
    settings: Settings


class ResolveService:
    def __init__(self, deps: ResolveDeps):
        self._deps = deps

    @traced_work
    async def handle_trigger(
        self,
        *,
        repo_id: str,
        mr_iid: int,
        author_email: str | None,
        payload: dict,
        trigger_type: str,
        discussion_id: str | None = None,
    ) -> int:
        log.info("resolve_trigger_received", repo_id=repo_id, mr_iid=mr_iid, trigger_type=trigger_type)
        # Capture the input before enqueueing; workers must not follow newer HEADs.
        for attempt in range(3):
            mr = await get_mr(self._deps.nc, repo_id, mr_iid)
            version = mr.latest_version
            if version is None or not mr.source_branch:
                raise RuntimeError("original MR has no branch or version to freeze")
            if mr.source_repo_id not in (None, repo_id):
                raise RuntimeError("cross-repository MR is not supported by the resolver")
            source_sha, target_sha = mr.source_head_sha, mr.target_head_sha
            if not source_sha or not target_sha:
                continue
            existing = await self._deps.repo.find_active_for_version(
                repo_id,
                mr_iid,
                source_sha,
                target_sha,
            )
            if existing:
                await self._reply_active_task_exists(
                    repo_id=repo_id,
                    mr_iid=mr_iid,
                    trigger_type=trigger_type,
                    discussion_id=discussion_id,
                    task=existing,
                )
                return existing.id

            log.info("resolve_snapshot_started", repo_id=repo_id, mr_iid=mr_iid)
            try:
                conflicts = await list_conflict_files(
                    self._deps.nc, repo_id, mr_iid,
                    expected_conflicts_id=f"{source_sha}..{target_sha}",
                )
            except ConflictSnapshotMismatchError:
                log.info(
                    "resolve_snapshot_retry", repo_id=repo_id, mr_iid=mr_iid,
                    attempt=attempt + 1, reason="conflict_snapshot_mismatch",
                    source_commit=source_sha, target_commit=target_sha,
                )
                continue
            log.info("resolve_snapshot_fetched", repo_id=repo_id, mr_iid=mr_iid, file_count=len(conflicts))
            # The snapshot binds this commit pair. Later branch movement does
            # not invalidate the captured input or change what workers process.
            frozen_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"_coco_round", "_coco_validation_repair", "_failure_notice"}
            }
            frozen_payload["_trigger_discussion_id"] = discussion_id
            frozen_payload["_trigger_mr_id"] = mr.id
            try:
                task = await self._deps.repo.create(
                    repo_id=repo_id,
                    source_branch=mr.source_branch,
                    target_branch=mr.target_branch,
                    resolve_branch="",
                    original_mr_id=mr_iid,
                    original_mr_url=mr.web_url,
                    trigger_type=trigger_type,
                    trigger_payload=frozen_payload,
                    trigger_author_email=author_email if trigger_type == "COMMENT" else None,
                    mr_author_email=mr.author_email,
                    mr_version=version.number,
                    source_commit_sha=source_sha,
                    target_commit_sha=target_sha,
                    conflict_files_snapshot=[asdict(item) for item in conflicts],
                )
            except ActiveTaskExistsError:
                # A concurrent trigger can win after our read. Re-read its version.
                existing = await self._deps.repo.find_active_for_version(
                    repo_id,
                    mr_iid,
                    source_sha,
                    target_sha,
                )
                if existing is None:
                    continue
                await self._reply_active_task_exists(
                    repo_id=repo_id,
                    mr_iid=mr_iid,
                    trigger_type=trigger_type,
                    discussion_id=discussion_id,
                    task=existing,
                )
                return existing.id
            break
        else:
            raise RuntimeError("could not capture a stable MR processing version; please retry /resolve")
        log.info("task_created", task_id=task.id, repo_id=repo_id, mr_iid=mr_iid)
        emit_counter(
            MetricName.TASKS_TOTAL,
            1,
            tags={"trigger": trigger_type, "status": "PENDING"},
        )
        if trigger_type == "COMMENT":
            await send_resolve_feedback(
                self._deps.nc, repo_id=repo_id, mr_id=mr.id, thread_id=discussion_id,
                body=(
                    f"解冲突任务 {task.id} 已受理，正在排队或处理中。\n"
                    f"[查看处理进度]({task_trace_url(task.id)})"
                ),
            )
        return task.id

    async def _reply_original(self, task, body: str) -> None:
        payload = getattr(task, "trigger_payload", None) or {}
        if not payload.get("_trigger_discussion_id"):
            return
        await send_resolve_feedback(
            self._deps.nc, repo_id=task.repo_id, mr_iid=task.original_mr_id,
            mr_id=payload.get("_trigger_mr_id"), thread_id=payload["_trigger_discussion_id"], body=body,
        )

    async def _reply_result(self, task) -> None:
        if not (getattr(task, "trigger_payload", None) or {}).get("_trigger_discussion_id"):
            return
        body = f"解冲突任务 #{task.id} 的结果已发布：[查看解冲突 MR]({task.new_mr_url})。"
        if task.lark_doc_url:
            body += f"\n[查看解冲突报告]({task.lark_doc_url})。"
        await self._reply_original(task, body)

    async def _fail_task(self, task, reason: str) -> None:
        stage = task.status
        ok = await self._deps.repo.update_status(
            task_id=task.id, from_version=task.version, new_status=TaskStatus.FAILED, last_error=reason,
        )
        if ok:
            task.version += 1
            task.status = TaskStatus.FAILED.value
            observe_task(task)
            log.warning("resolve_task_failed", task_id=task.id, from_status=stage)
            await self._reply_original(task, f"解冲突任务 #{task.id} 失败（阶段：{stage}）：{reason[:1000]}")

    async def _reply_active_task_exists(
        self,
        *,
        repo_id: str,
        mr_iid: int,
        trigger_type: str,
        discussion_id: str | None,
        task,
    ) -> None:
        log.info("resolve_task_reused", task_id=task.id, repo_id=repo_id, mr_iid=mr_iid, task_status=task.status)
        if trigger_type != "COMMENT":
            return
        if not discussion_id:
            log.warning(
                "active_task_reply_skipped_missing_discussion",
                task_id=task.id,
                repo_id=repo_id,
                mr_iid=mr_iid,
            )
            return
        message = f"当前代码版本已有解冲突任务 {task.id}（{task.status}），已复用该任务。"
        if task.new_mr_url:
            message += f"[查看解冲突 MR]({task.new_mr_url})。"
        try:
            async with asyncio.timeout(5):
                await reply_discussion(self._deps.nc, repo_id, mr_iid, discussion_id, message)
        except Exception:
            log.exception("active_task_reply_failed", task_id=task.id)
            return
        log.info(
            "active_task_reply_created",
            task_id=task.id,
            repo_id=repo_id,
            mr_iid=mr_iid,
            discussion_id=discussion_id,
        )

    async def _heads_match_version(self, repo_id: str, mr: MR) -> bool:
        version = mr.latest_version
        if version is None or not mr.source_branch:
            return False
        source, target = await asyncio.gather(
            get_branch(self._deps.nc, repo_id, mr.source_branch),
            get_branch(self._deps.nc, repo_id, mr.target_branch),
        )
        return bool(
            source
            and target
            and source.commit_sha == version.source_commit_id
            and target.commit_sha == version.target_commit_id
        )

    @traced_work
    async def run_pipeline(
        self,
        task_id: int,
        *,
        coco_instructions: str | None = None,
    ) -> None:
        task = await self._deps.repo.get(task_id)
        if task is None:
            log.error("task_not_found", task_id=task_id)
            return
        if TaskStatus(task.status) is not TaskStatus.PREPARING:
            log.info(
                "pipeline_skipped_without_claim",
                task_id=task_id,
                status=task.status,
            )
            return

        try:
            await self._step1_prepare(task)
            await self._step2_identify_conflicts(task)
            await self._step3_dispatch_coco(task, instructions=coco_instructions)
            # step4/5 等 Coco 完成事件触发,见 continue_after_coco
        except Exception as e:
            log.exception("pipeline_failed", task_id=task_id, error=str(e))
            await self._fail_task(task, str(e))
            emit_counter(
                MetricName.TASKS_TOTAL,
                1,
                tags={"trigger": task.trigger_type, "status": "FAILED"},
            )

    @traced_work
    async def run_retry_pipeline(self, task_id: int) -> None:
        task = await self._deps.repo.get_with_blocks(task_id)
        if task is None:
            log.error("retry_task_not_found", task_id=task_id)
            return
        if TaskStatus(task.status) is not TaskStatus.RETRY_PREPARING:
            log.info(
                "retry_pipeline_skipped_without_claim",
                task_id=task_id,
                status=task.status,
            )
            return
        try:
            rejected, locked = _partition_retry_blocks(task)
            if not rejected:
                raise RuntimeError("retry task has no rejected conflict blocks")
            await self._verify_retry_baseline(task)
            assert task.mr_version is not None
            assert task.source_commit_sha is not None
            assert task.target_commit_sha is not None
            assert task.resolve_commit_sha is not None
            agent_id = task.trigger_payload.get("agent_id") or self._deps.settings.coco_default_agent_id
            log.info("resolve_retry_submit_started", task_id=task.id, attempt=task.retry_count + 1)
            ref = await self._deps.coco.submit_retry_task(
                agent_id,
                repo_id=task.repo_id,
                task_id=task.id,
                original_mr_number=task.original_mr_id,
                mr_version=task.mr_version,
                resolve_branch=task.resolve_branch,
                source_branch=task.source_branch,
                source_commit=task.source_commit_sha,
                target_branch=task.target_branch,
                target_commit=task.target_commit_sha,
                base_commit_sha=task.resolve_commit_sha,
                attempt=task.retry_count + 1,
                rejected_conflicts=[
                    {
                        "conflict_id": block.conflict_id,
                        "path": block.file_path,
                        "source_text": block.ours_text or "",
                        "target_text": block.theirs_text or "",
                        "previous_resolution": block.ai_resolution or "",
                        "reject_reason": block.approval.comment,
                    }
                    for block in rejected
                ],
                locked_conflict_ids=[block.conflict_id for block in locked],
                instructions=("按 reject_reason 修订既有方案，同时保持其他冲突块和非冲突代码不变。"),
            )
            await self._transition_with_fields(
                task,
                TaskStatus.RETRY_COCO_SUBMITTED,
                coco_task_id=ref.session_id,
                coco_workspace_id=self._deps.settings.coco_workspace_id,
                coco_agent_id=agent_id,
                coco_session_id=ref.session_id,
                coco_run_id=ref.run_id,
                coco_last_event_id=ref.user_event_id,
                trigger_payload={
                    **task.trigger_payload,
                    "_coco_round": {
                        "session_id": ref.session_id,
                        "run_id": ref.run_id,
                        "event_id": ref.user_event_id,
                    },
                },
            )
            await self._transition(task, TaskStatus.RETRY_COCO_RUNNING)
            await self._record_event(
                task.id,
                "RETRY_COCO_SUBMITTED",
                {
                    "attempt": task.retry_count + 1,
                    "session_id": ref.session_id,
                    "run_id": ref.run_id,
                    "rejected_conflict_ids": [block.conflict_id for block in rejected],
                    "locked_conflict_ids": [block.conflict_id for block in locked],
                },
            )
        except UnsupportedConflictError as exc:
            log.warning("resolve_unsupported_conflict", task_id=task.id, error=str(exc))
            await self._fail_task(task, str(exc))
        except Exception as exc:
            log.exception("retry_pipeline_failed", task_id=task_id, error=str(exc))
            if TaskStatus(task.status) is TaskStatus.RETRY_COCO_SUBMITTED:
                await self._deps.repo.update_fields(
                    task_id=task.id,
                    from_version=task.version,
                    last_error=str(exc),
                )
            else:
                await self._deps.repo.update_status(
                    task_id=task.id,
                    from_version=task.version,
                    new_status=TaskStatus.PARTIAL_REJECTED,
                    last_error=str(exc),
                )
            await self._record_event(
                task.id,
                "RETRY_FAILED",
                {"stage": task.status, "error": str(exc)},
            )

    async def _claim_coco_poll(self, task) -> bool:
        source = TaskStatus(task.status)
        target = COCO_POLL_CLAIMS[source]
        claimed = await self._deps.repo.update_status(
            task_id=task.id, from_version=task.version, from_status=source, new_status=target,
        )
        if not claimed:
            log.info("resolve_coco_claim_lost", task_id=task.id, task_version=task.version)
            return False
        task.status = target.value
        task.version += 1
        observe_task(task)
        log.info("resolve_coco_claimed", task_id=task.id, task_version=task.version, task_status=task.status)
        return True

    async def _release_coco_poll(self, task, waiting: TaskStatus) -> None:
        ok = await self._deps.repo.update_status(
            task_id=task.id, from_version=task.version, from_status=TaskStatus(task.status), new_status=waiting,
        )
        if ok:
            task.status = waiting.value
            task.version += 1
            observe_task(task)
        log.info("resolve_coco_released" if ok else "resolve_coco_release_lost", task_id=task.id,
                 task_version=task.version)

    async def fail_stuck_coco_poll(self, task_id: int) -> None:
        task = await self._deps.repo.get(task_id, include_bodies=False)
        if task is None or TaskStatus(task.status) not in COCO_POLL_CLAIMS.values():
            return
        threshold = datetime.now(UTC) - timedelta(seconds=COCO_PROCESSING_TIMEOUT_SECONDS)
        if task.updated_at.replace(tzinfo=UTC) >= threshold:
            return
        # TODO: This only fences versioned task writes, not block writes or an
        # external request already in flight. Do not auto-requeue a timed-out claim.
        await self._fail_task(task, "Coco 结果处理超过 30 分钟未推进，已停止等待；请在原始 MR 重新 /resolve。")

    @traced_work
    async def continue_after_retry_coco(self, task_id: int) -> None:
        task = await self._deps.repo.get(task_id, include_bodies=False)
        if task is None or TaskStatus(task.status) is not TaskStatus.RETRY_COCO_RUNNING or not task.coco_task_id:
            return
        if not await self._claim_coco_poll(task):
            return
        interrupted = False
        try:
            if await self._recover_repair(task):
                return
            poll_started = monotonic()
            log.info("resolve_coco_poll_started", task_id=task.id, session_id=task.coco_task_id)
            result = await self._deps.coco.fetch_retry_task_result(task.coco_task_id, **self._round_context(task))
            log.info(
                "resolve_coco_polled", task_id=task.id, task_status=task.status,
                session_id=task.coco_task_id, coco_status=result.status,
                duration_ms=int((monotonic() - poll_started) * 1000),
            )
            if result.status == "invalid_result":
                raise CocoValidationError(result.error or "Coco retry result JSON is invalid")
            if result.status == "running":
                return
            if result.status != "succeeded" or result.retry_result is None:
                if self._repair_state(task):
                    await self._fail_validation(task, result.error or f"Coco repair ended with {result.status}")
                    return
                await self._handle_retry_coco_failure(
                    task,
                    result.error or f"Coco ended with {result.status}",
                )
                return
            full = await self._deps.repo.get_with_blocks(task_id)
            if full is None or full.status != task.status or full.version != task.version:
                return
            task = full
            rejected, locked = await self._retry_blocks_for_attempt(task)
            retry_result = self._bind_result(task, result.retry_result)
            resolved_files = await self._verify_retry_result(
                task,
                retry_result,
                rejected=rejected,
                locked=locked,
            )
            verified_commit = await self._get_result_commit(task, retry_result.commit_sha)
            retry_result = replace(retry_result, parent_commits=verified_commit.parents)
            result_by_id = {conflict.conflict_id: conflict for conflict in retry_result.conflicts}
            blank_positions = verify_blank_files(task, result_by_id, resolved_files)
            for block in locked:
                if block.conflict_id in blank_positions:
                    start, end = blank_positions[block.conflict_id]
                    await self._deps.repo.update_block_resolution(
                        block_id=block.id, fields={"resolved_start_line": start, "resolved_end_line": end},
                    )
            for block in rejected:
                conflict = result_by_id[block.conflict_id]
                await self._deps.repo.update_block_resolution(
                    block_id=block.id,
                    fields={
                        **self._author_fields(conflict),
                        "ai_resolution": _resolution_text(conflict.resolved.text),
                        "decision_summary": conflict.decision_summary,
                        "resolved_path": conflict.resolved.path,
                        "resolved_start_line": conflict.resolved.start_line,
                        "resolved_end_line": conflict.resolved.end_line,
                    },
                )
            await self._transition_with_fields(
                task,
                TaskStatus.RETRY_RESULT_VERIFIED,
                coco_last_event_id=(result.last_event_id or task.coco_last_event_id),
                coco_result_schema_version=retry_result.schema_version,
                coco_result_json=result_to_dict(retry_result),
                resolve_commit_sha=retry_result.commit_sha,
            )
            await self._record_event(
                task.id,
                "RETRY_COCO_VERIFIED",
                {
                    "attempt": retry_result.attempt,
                    "previous_resolve_commit": retry_result.base_commit_sha,
                    "resolve_commit": retry_result.commit_sha,
                    "conflict_ids": sorted(result_by_id),
                },
                coco_event_id=result.last_event_id or None,
            )
            await self._step_retry_document(task, retry_result, resolved_files)
            await self._publish_retry(task, rejected, retry_result)
        except asyncio.CancelledError:
            interrupted = True
            raise
        except UnsupportedConflictError as exc:
            log.warning("resolve_unsupported_conflict", task_id=task.id, error=str(exc))
            await self._fail_task(task, str(exc))
        except CocoValidationError as exc:
            await self._repair_or_fail(task, str(exc), retry=True)
        except (NextCodeError, CocoError, TransientError) as exc:
            log.warning("retry_validation_dependency_error", task_id=task.id, error=str(exc))
            if isinstance(exc, NextCodeError) and exc.action == "GetBlame" and exc.status == 422:
                await self._fail_task(task, str(exc))
            elif task.status != TaskStatus.RETRY_COCO_PROCESSING.value:
                await self._deps.repo.update_fields(task_id=task.id, from_version=task.version, last_error=str(exc))
        except Exception as exc:
            log.exception("retry_post_coco_failed", task_id=task_id, error=str(exc))
            if TaskStatus(task.status) is TaskStatus.RETRY_COCO_PROCESSING:
                await self._fail_task(task, str(exc))
            else:
                await self._deps.repo.update_fields(
                    task_id=task.id,
                    from_version=task.version,
                    last_error=str(exc),
                )
            await self._record_event(
                task.id,
                "RETRY_FAILED",
                {"stage": task.status, "error": str(exc)},
            )
        finally:
            current = asyncio.current_task()
            if not interrupted and current is not None and not current.cancelling() and (
                task.status == TaskStatus.RETRY_COCO_PROCESSING.value
            ):
                await self._release_coco_poll(task, TaskStatus.RETRY_COCO_RUNNING)

    @traced_work
    async def resume_retry_publication(self, task_id: int) -> None:
        """Resume verified retry side effects after a worker or Pod restart."""
        task = await self._deps.repo.get_with_blocks(task_id)
        if task is None:
            return
        status = TaskStatus(task.status)
        if status is TaskStatus.RETRY_COCO_RUNNING:
            await self.continue_after_retry_coco(task_id)
            return
        if status not in {
            TaskStatus.RETRY_COCO_SUBMITTED,
            TaskStatus.RETRY_RESULT_VERIFIED,
            TaskStatus.RETRY_DOC_UPLOADED,
            TaskStatus.RETRY_PUBLISHING,
        }:
            return
        claimed = await self._deps.repo.update_fields(
            task_id=task.id,
            from_version=task.version,
            last_error=task.last_error,
        )
        if not claimed:
            log.info(
                "retry_resume_claim_lost",
                task_id=task.id,
                status=task.status,
            )
            return
        task.version += 1
        if status is TaskStatus.RETRY_COCO_SUBMITTED:
            await self._transition(task, TaskStatus.RETRY_COCO_RUNNING)
            await self.continue_after_retry_coco(task.id)
            return
        rejected, _locked = await self._retry_blocks_for_attempt(task)
        retry_result = _retry_result_from_dict(task.coco_result_json)
        try:
            if status is TaskStatus.RETRY_RESULT_VERIFIED:
                resolved_files = await self._read_resolved_files(
                    task,
                    retry_result.commit_sha,
                )
                await self._step_retry_document(
                    task,
                    retry_result,
                    resolved_files,
                )
            await self._publish_retry(task, rejected, retry_result)
        except Exception as exc:
            log.exception(
                "retry_resume_failed",
                task_id=task.id,
                status=task.status,
                error=str(exc),
            )
            await self._deps.repo.update_fields(
                task_id=task.id,
                from_version=task.version,
                last_error=str(exc),
            )

    async def _retry_blocks_for_attempt(self, task) -> tuple[list, list]:
        events = await self._deps.repo.list_events(task.id)
        requested = [
            event
            for event in events
            if event.event_type == "RETRY_REQUESTED" and int(event.payload.get("attempt") or 0) == task.retry_count + 1
        ]
        if not requested:
            return _partition_retry_blocks(task)
        snapshot = requested[-1].payload
        rejected_ids = set(snapshot.get("rejected_conflict_ids") or [])
        locked_ids = set(snapshot.get("locked_conflict_ids") or [])
        by_id = {block.conflict_id: block for block in task.blocks}
        if rejected_ids & locked_ids or rejected_ids | locked_ids != set(by_id):
            raise RuntimeError("retry snapshot does not match the conflict manifest")
        return (
            [by_id[conflict_id] for conflict_id in sorted(rejected_ids)],
            [by_id[conflict_id] for conflict_id in sorted(locked_ids)],
        )

    async def _read_resolved_files(
        self,
        task,
        revision: str,
    ) -> dict[str, str]:
        paths = sorted({block.file_path for block in task.blocks})
        files = await asyncio.gather(
            *[
                get_file(
                    self._deps.nc,
                    task.repo_id,
                    revision=revision,
                    path=path,
                )
                for path in paths
            ]
        )
        if any(file is None for file in files):
            raise RuntimeError("retry commit is missing a conflict file")
        return {path: file.content for path, file in zip(paths, files, strict=True) if file is not None}

    @traced_work
    async def continue_after_coco(
        self,
        task_id: int,
        *,
        local_report_dir: Path | None = None,
    ) -> LocalReportArtifacts | None:
        """Poll a Managed Agent Session and continue only after a clean end turn."""
        task = await self._deps.repo.get(task_id, include_bodies=False)
        if task is None or task.status != TaskStatus.COCO_RUNNING.value or not task.coco_task_id:
            return None
        if not await self._claim_coco_poll(task):
            return None
        interrupted = False
        try:
            if await self._recover_repair(task):
                return None
            poll_started = monotonic()
            log.info("resolve_coco_poll_started", task_id=task.id, session_id=task.coco_task_id)
            result = await self._deps.coco.fetch_task_result(task.coco_task_id, **self._round_context(task))
            log.info(
                "resolve_coco_polled", task_id=task.id, task_status=task.status,
                session_id=task.coco_task_id, coco_status=result.status,
                duration_ms=int((monotonic() - poll_started) * 1000),
            )
            if result.status == "invalid_result":
                raise CocoValidationError(result.error or "Coco result JSON is invalid")
            if result.status == "running":
                return None
            if result.status == "requires_action":
                if self._repair_state(task):
                    await self._fail_validation(task, "自动修复等待工具权限确认，未完成修复")
                    return None
                log.warning(
                    "coco_requires_action",
                    task_id=task.id,
                    session_id=task.coco_task_id,
                )
                await self._transition_with_fields(
                    task,
                    TaskStatus.FAILED,
                    last_error="Coco Agent PermissionMode 配置错误",
                    coco_last_event_id=result.last_event_id or task.coco_last_event_id,
                )
                return None
            if result.status != "succeeded":
                if self._repair_state(task):
                    await self._fail_validation(task, result.error or f"Coco repair ended with {result.status}")
                    return None
                await self._transition_with_fields(
                    task,
                    TaskStatus.FAILED,
                    last_error=result.error or f"Coco ended with {result.status}",
                    coco_last_event_id=result.last_event_id or task.coco_last_event_id,
                )
                return None
            if result.resolve_result is None:
                raise CocoValidationError("Coco succeeded without a parsed resolve result")
            result = replace(result, resolve_result=self._bind_result(task, result.resolve_result))
            assert result.resolve_result is not None
            await self._update_fields(
                task,
                coco_last_event_id=result.last_event_id or task.coco_last_event_id,
                coco_result_schema_version=result.resolve_result.schema_version,
                coco_result_json=result_to_dict(result.resolve_result),
            )
            if local_report_dir is not None:
                return await self._step4_local_document(task, result, local_report_dir)
            await self._step4_document(task, result)
            await self._step5_create_mr_and_discussions(task)
        except asyncio.CancelledError:
            interrupted = True
            raise
        except UnsupportedConflictError as exc:
            log.warning("resolve_unsupported_conflict", task_id=task.id, error=str(exc))
            await self._fail_task(task, str(exc))
        except CocoValidationError as exc:
            await self._repair_or_fail(task, str(exc), retry=False)
        except (NextCodeError, CocoError, TransientError) as exc:
            log.warning("validation_dependency_error", task_id=task.id, error=str(exc))
            if (
                isinstance(exc, NextCodeError) and exc.action == "GetBlame" and exc.status == 422
            ) or task.status != TaskStatus.COCO_PROCESSING.value:
                await self._fail_task(task, str(exc))
        except Exception as e:
            log.exception("post_coco_failed", task_id=task_id, error=str(e))
            await self._fail_task(task, str(e))
        finally:
            current = asyncio.current_task()
            if not interrupted and current is not None and not current.cancelling() and (
                task.status == TaskStatus.COCO_PROCESSING.value
            ):
                await self._release_coco_poll(task, TaskStatus.COCO_RUNNING)
        return None

    @_logged_stage
    async def _step1_prepare(self, task) -> None:
        _check_supported_conflicts(task)
        source_sha = getattr(task, "source_commit_sha", None)
        target_sha = getattr(task, "target_commit_sha", None)
        mr_version = getattr(task, "mr_version", None)
        if not source_sha or not target_sha or mr_version is None:
            # Compatibility for tasks enqueued before this migration.
            mr = await get_mr(self._deps.nc, task.repo_id, task.original_mr_id)
            version = mr.latest_version
            if version is None or not mr.source_branch:
                raise RuntimeError("original MR has no branch or version to freeze")
            if mr.source_repo_id not in (None, task.repo_id):
                raise RuntimeError("cross-repository MR is not supported by the resolver")
            if task.source_branch and task.source_branch != mr.source_branch:
                raise RuntimeError("webhook source branch differs from GetMergeRequest")
            if task.target_branch and task.target_branch != mr.target_branch:
                raise RuntimeError("webhook target branch differs from GetMergeRequest")
            source_sha, target_sha, mr_version = (
                version.source_commit_id,
                version.target_commit_id,
                version.number,
            )
            task.source_branch, task.target_branch = mr.source_branch, mr.target_branch
            task.mr_author_email = mr.author_email or task.mr_author_email

        resolve_branch = task.resolve_branch or build_resolve_branch(
            source_branch=task.source_branch,
            target_branch=task.target_branch,
            task_id=task.id,
            created_at=task.created_at,
        )
        await self._update_fields(
            task,
            resolve_branch=resolve_branch,
            source_branch=task.source_branch,
            target_branch=task.target_branch,
            mr_version=mr_version,
            source_commit_sha=source_sha,
            target_commit_sha=target_sha,
            resolve_base_sha=source_sha,
            mr_author_email=task.mr_author_email,
            active_key=build_active_key(task.repo_id, task.original_mr_id, source_sha, target_sha),
        )
        await self._record_event(
            task.id,
            "MR_FROZEN",
            {"mr_version": mr_version, "source_commit": source_sha, "target_commit": target_sha},
        )
        await ensure_branch(self._deps.nc, task.repo_id, revision=source_sha, name=resolve_branch)
        await self._transition(task, TaskStatus.BRANCH_CREATED)
        await self._record_event(
            task.id,
            "RESOLVE_BRANCH_CREATED",
            {"resolve_branch": resolve_branch, "head": source_sha},
        )

    @_logged_stage
    async def _step2_identify_conflicts(self, task) -> None:
        snapshot = getattr(task, "conflict_files_snapshot", None)
        if snapshot is None:
            # Legacy tasks have no persisted manifest. Never associate live
            # conflicts with an older frozen version.
            mr = await get_mr(self._deps.nc, task.repo_id, task.original_mr_id)
            version = mr.latest_version
            if (
                version is None
                or version.source_commit_id != task.source_commit_sha
                or version.target_commit_id != task.target_commit_sha
            ):
                raise RuntimeError("legacy task has no conflict snapshot for its frozen commits")
            conflict_files = await list_conflict_files(self._deps.nc, task.repo_id, task.original_mr_id)
            readback = await get_mr(self._deps.nc, task.repo_id, task.original_mr_id)
            if readback.latest_version != version or not await self._heads_match_version(task.repo_id, readback):
                raise RuntimeError("MR changed while capturing legacy conflict snapshot")
            await self._update_fields(task, conflict_files_snapshot=[asdict(item) for item in conflict_files])
        else:
            conflict_files = [ConflictFile(**item) for item in snapshot]

        _check_supported_conflicts(task)
        from conciliator.integrations.git.conflicts import parse_conflict_file

        all_blocks: list[dict] = []
        for cf in sorted(
            conflict_files,
            key=lambda item: (item.file_path, item.first_path, item.second_path),
        ):
            parsed = parse_conflict_file(cf.file_path, cf.content)
            for p in parsed:
                block_idx = len(all_blocks)
                all_blocks.append(
                    {
                        "conflict_id": f"C{block_idx + 1:04d}",
                        "block_idx": block_idx,
                        "file_path": p.file_path,
                        "start_line": p.start_line,
                        "end_line": p.end_line,
                        "ours_text": p.ours_text,
                        "theirs_text": p.theirs_text,
                        # Coco 返回两侧原始行范围后，再用 NextCode GetBlame 复核。
                        "author_email": None,
                    }
                )

        if not all_blocks:
            raise RuntimeError("original MR no longer contains parseable conflict blocks")

        await self._deps.repo.save_conflict_blocks(task.id, all_blocks)
        await self._transition(task, TaskStatus.CONFLICTS_FETCHED)
        await self._record_event(
            task.id,
            "CONFLICTS_PERSISTED",
            {
                "block_count": len(all_blocks),
                "files": sorted({block["file_path"] for block in all_blocks}),
            },
        )

    @_logged_stage
    async def _step3_dispatch_coco(
        self,
        task,
        *,
        instructions: str | None = None,
    ) -> None:
        agent_id = task.trigger_payload.get("agent_id") or self._deps.settings.coco_default_agent_id
        full = await self._deps.repo.get_with_blocks(task.id)
        if full is None or task.mr_version is None or not task.source_commit_sha or not task.target_commit_sha:
            raise RuntimeError("task is missing its frozen conflict manifest")
        conflicts = [
            {
                "conflict_id": block.conflict_id,
                "path": block.file_path,
                "source_text": block.ours_text or "",
                "approval_not_required": approval_not_required(block),
                "target_text": block.theirs_text or "",
                "context_before": "",
                "context_after": "",
            }
            for block in sorted(full.blocks, key=lambda item: item.block_idx)
        ]
        ref = await self._deps.coco.submit_resolve_task(
            agent_id,
            repo_id=task.repo_id,
            task_id=task.id,
            original_mr_number=task.original_mr_id,
            mr_version=task.mr_version,
            resolve_branch=task.resolve_branch,
            source_branch=task.source_branch,
            source_commit=task.source_commit_sha,
            target_branch=task.target_branch,
            target_commit=task.target_commit_sha,
            attempt=task.retry_count + 1,
            conflicts=conflicts,
            instructions=(
                instructions or "根据代码上下文逐块选择保留 source、保留 target 或组合修改，并解释最终决策。"
            ),
        )
        await self._transition_with_fields(
            task,
            TaskStatus.COCO_SUBMITTED,
            coco_task_id=ref.session_id,
            coco_workspace_id=self._deps.settings.coco_workspace_id,
            coco_agent_id=agent_id,
            coco_session_id=ref.session_id,
            coco_run_id=ref.run_id,
            coco_last_event_id=ref.user_event_id,
            trigger_payload={
                **task.trigger_payload,
                "_coco_round": {
                    "session_id": ref.session_id,
                    "run_id": ref.run_id,
                    "event_id": ref.user_event_id,
                },
            },
        )
        await self._transition(task, TaskStatus.COCO_RUNNING)
        await self._record_event(
            task.id,
            "COCO_SUBMITTED",
            {
                "workspace_id": self._deps.settings.coco_workspace_id,
                "agent_id": agent_id,
                "session_id": ref.session_id,
                "run_id": ref.run_id,
            },
        )

    @_logged_stage
    async def _step4_document(self, task, result: CocoTaskResult) -> None:
        if self._deps.lark is None:
            raise RuntimeError("Lark client is unavailable")
        block_dicts, resolved_files = await self._prepare_report_data(task, result)
        resolve_result = result.resolve_result
        assert resolve_result is not None
        markdown = templates.build_lark_doc_markdown(
            original_mr_number=task.original_mr_id,
            source_branch=task.source_branch,
            source_commit=task.source_commit_sha or "",
            target_branch=task.target_branch,
            target_commit=task.target_commit_sha or "",
            resolve_branch=task.resolve_branch,
            resolve_commit=resolve_result.commit_sha,
            blocks=block_dicts,
            resolved_files=resolved_files,
        )
        uploaded = await upload_markdown(
            self._deps.lark,
            file_name=(f"conciliator-mr-{task.original_mr_id}-task-{task.id}.md"),
            content=markdown.encode("utf-8"),
            folder_token=self._deps.settings.lark_markdown_folder_token,
        )
        await self._transition_with_fields(
            task,
            TaskStatus.DOC_CREATED,
            lark_doc_url=uploaded.web_url,
        )
        await self._record_event(
            task.id,
            "LARK_MARKDOWN_UPLOADED",
            {
                "file_token": uploaded.file_token,
                "file_name": uploaded.file_name,
                "size_bytes": uploaded.size_bytes,
                "url": uploaded.web_url,
            },
        )

    @_logged_stage
    async def _step4_local_document(
        self,
        task,
        result: CocoTaskResult,
        output_root: Path,
    ) -> LocalReportArtifacts:
        block_dicts, resolved_files = await self._prepare_report_data(task, result)
        resolve_result = result.resolve_result
        assert resolve_result is not None
        resolve_result = replace(resolve_result, merge_parents=tuple(task.coco_result_json["merge_parents"]))
        artifacts = write_local_report_artifacts(
            output_root=output_root,
            task=task,
            blocks=block_dicts,
            resolve_result=resolve_result,
            resolved_files=resolved_files,
        )
        await self._transition(task, TaskStatus.DOC_CREATED)
        return artifacts

    @_logged_stage
    async def _prepare_report_data(
        self,
        task,
        result: CocoTaskResult,
    ) -> tuple[list[dict], dict[str, str]]:
        resolve_result = result.resolve_result
        if resolve_result is None:
            raise RuntimeError("Coco ended without a structured resolve result")
        full_task = await self._deps.repo.get_with_blocks(task.id)
        if full_task is None:
            raise RuntimeError("resolve task disappeared before result verification")
        _check_supported_conflicts(full_task)
        resolved_files = await self._verify_coco_result(full_task, resolve_result)
        verified_commit = await self._get_result_commit(full_task, resolve_result.commit_sha)
        await self._update_fields(
            task,
            resolve_commit_sha=resolve_result.commit_sha,
            coco_result_json=result_to_dict(replace(resolve_result, merge_parents=verified_commit.parents)),
        )

        result_by_id = {conflict.conflict_id: conflict for conflict in resolve_result.conflicts}
        block_dicts = []
        for block in sorted(full_task.blocks, key=lambda item: item.block_idx):
            conflict = result_by_id[block.conflict_id]
            ai_resolution = _resolution_text(
                conflict.resolved.text,
            )
            await self._deps.repo.update_block_resolution(
                block_id=block.id,
                fields={
                    "ai_resolution": ai_resolution,
                    "author_email": conflict.source_attribution.author_email,
                    "source_author_email": conflict.source_attribution.author_email,
                    "source_author_commit_sha": conflict.source_attribution.commit_sha,
                    "source_path": conflict.source_attribution.path,
                    "source_start_line": conflict.source_attribution.start_line,
                    "source_end_line": conflict.source_attribution.end_line,
                    "target_author_email": conflict.target_attribution.author_email,
                    "target_author_commit_sha": conflict.target_attribution.commit_sha,
                    "target_path": conflict.target_attribution.path,
                    "target_start_line": conflict.target_attribution.start_line,
                    "target_end_line": conflict.target_attribution.end_line,
                    "attribution_status": "VERIFIED",
                    "decision_summary": conflict.decision_summary,
                    "resolved_path": conflict.resolved.path,
                    "resolved_start_line": conflict.resolved.start_line,
                    "resolved_end_line": conflict.resolved.end_line,
                },
            )
            block_dicts.append(
                {
                    "idx": block.block_idx + 1,
                    "approval_status": ("NOT_REQUIRED" if approval_not_required(block) else "PENDING"),
                    "conflict_id": block.conflict_id,
                    "file_path": block.file_path,
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                    "source_text": block.ours_text or "",
                    "target_text": block.theirs_text or "",
                    "ai_resolution": ai_resolution,
                    "decision_summary": conflict.decision_summary,
                    "source_author_email": conflict.source_attribution.author_email,
                    "source_author_commit_sha": conflict.source_attribution.commit_sha,
                    "source_path": conflict.source_attribution.path,
                    "source_start_line": conflict.source_attribution.start_line,
                    "source_end_line": conflict.source_attribution.end_line,
                    "target_author_email": conflict.target_attribution.author_email,
                    "target_author_commit_sha": conflict.target_attribution.commit_sha,
                    "target_path": conflict.target_attribution.path,
                    "target_start_line": conflict.target_attribution.start_line,
                    "target_end_line": conflict.target_attribution.end_line,
                    "resolved_path": conflict.resolved.path,
                    "resolved_start_line": conflict.resolved.start_line,
                    "resolved_end_line": conflict.resolved.end_line,
                }
            )
        await self._record_event(
            task.id,
            "COCO_VERIFIED",
            {
                "session_id": task.coco_session_id,
                "run_id": task.coco_run_id,
                "resolve_commit": resolve_result.commit_sha,
                "schema_version": resolve_result.schema_version,
                "block_count": len(block_dicts),
            },
            coco_event_id=result.last_event_id or None,
        )
        return block_dicts, resolved_files

    @_logged_stage
    async def _step5_create_mr_and_discussions(self, task) -> None:
        if self._deps.notification is None:
            raise RuntimeError("notification service is unavailable")
        full_task = await self._deps.repo.get_with_blocks(task.id)
        if full_task is None:
            raise RuntimeError("resolve task disappeared before MR creation")
        manual_blocks = sorted(
            (b for b in full_task.blocks if not approval_not_required(b)),
            key=lambda b: b.block_idx,
        )
        all_automatic = bool(full_task.blocks) and not manual_blocks
        block_dicts = _mr_blocks(full_task)
        description = templates.build_mr_description(
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
            task_status=(TaskStatus.ALL_APPROVED.value if all_automatic else TaskStatus.AWAITING_APPROVAL.value),
        )
        title = (f"[resolve task-{task.id}] {task.trigger_payload.get('mr_title', '')}").strip()
        mr = await self._create_or_recover_resolve_mr(
            task,
            title=title,
            description=description,
        )
        readback = await get_mr(self._deps.nc, task.repo_id, mr.iid)
        if (
            readback.source_branch != task.resolve_branch
            or readback.target_branch != task.target_branch
            or readback.description != description
        ):
            raise RuntimeError("resolve MR readback differs from the create request")
        if readback.latest_version is None:
            raise RuntimeError("resolve MR readback has no version")
        if not readback.web_url:
            raise RuntimeError("resolve MR readback has no URL")
        if all_automatic and readback.latest_version.source_commit_id != task.resolve_commit_sha:
            raise RuntimeError("automatic resolver MR does not contain the verified commit")
        await self._transition_with_fields(
            task,
            TaskStatus.MR_CREATED,
            **({"trigger_payload": {**task.trigger_payload, "_blank_approval_publication_verified": True}}
               if all_automatic else {}),
            new_mr_id=mr.iid,
            new_mr_url=readback.web_url,
        )
        await self._record_event(
            task.id,
            "RESOLVE_MR_CREATED",
            {
                "new_mr_id": mr.iid,
                "new_mr_url": readback.web_url,
                "source_branch": task.resolve_branch,
                "target_branch": task.target_branch,
            },
        )

        if all_automatic:
            await self._complete_blank_approval(task)
            return

        for b in manual_blocks:
            if not b.resolved_path or b.resolved_start_line is None or b.resolved_end_line is None:
                raise RuntimeError(f"conflict {b.conflict_id} has no verified resolver position")
            body = templates.build_discussion_body(
                task_id=task.id,
                conflict_id=b.conflict_id,
                idx=b.block_idx + 1,
                file_path=b.resolved_path,
                start_line=b.resolved_start_line,
                end_line=b.resolved_end_line,
                author_email=b.author_email,
                source_author_email=b.source_author_email,
                target_author_email=b.target_author_email,
                source_path=b.source_path,
                source_start_line=b.source_start_line,
                source_end_line=b.source_end_line,
                target_path=b.target_path,
                target_start_line=b.target_start_line,
                target_end_line=b.target_end_line,
                ours=b.ours_text or "",
                theirs=b.theirs_text or "",
                ai_resolution=b.ai_resolution or "",
                decision_summary=b.decision_summary or "",
                lark_doc_url=task.lark_doc_url,
            )
            disc = await create_discussion(
                self._deps.nc,
                task.repo_id,
                task.new_mr_id,
                file_path=b.resolved_path,
                line=b.resolved_start_line,
                end_line=b.resolved_end_line,
                body=body,
            )
            thread = await get_thread(
                self._deps.nc,
                task.repo_id,
                disc.id,
            )
            version = readback.latest_version
            position = find_position_for_version(thread, version)
            if position is None or (
                position.get("Path") != b.resolved_path
                or int(position.get("StartLine") or 0) != b.resolved_start_line
                or int(position.get("EndLine") or 0) != b.resolved_end_line
            ):
                raise RuntimeError(f"Discussion position readback differs for {b.conflict_id}")
            await self._deps.repo.update_block_discussion_and_resolution(
                block_id=b.id,
                discussion_id=disc.id,
                ai_resolution=b.ai_resolution or "",
            )
            await self._record_event(
                task.id,
                "DISCUSSION_CREATED",
                {
                    "conflict_id": b.conflict_id,
                    "discussion_id": disc.id,
                    "comment_id": disc.comment_id,
                    "path": b.resolved_path,
                    "start_line": b.resolved_start_line,
                    "end_line": b.resolved_end_line,
                },
            )

        author_emails = sorted(
            {email for b in manual_blocks for email in (b.source_author_email, b.target_author_email) if email}
        )
        await self._deps.notification.notify_mr_comment(
            project_id=task.repo_id,
            mr_iid=task.new_mr_id,
            body=(f"已创建 {len(manual_blocks)} 个冲突块待确认；"
                  f"另有 {len(full_task.blocks) - len(manual_blocks)} 个纯空行块无需人工审批。"),
            mention_emails=author_emails,
        )
        await self._transition(task, TaskStatus.AWAITING_APPROVAL)
        await self._reply_result(task)
        await self._record_event(
            task.id,
            "AWAITING_APPROVAL",
            {
                "new_mr_id": task.new_mr_id,
                "block_count": len(manual_blocks),
                "not_required_count": len(full_task.blocks) - len(manual_blocks),
            },
        )
        emit_counter(
            MetricName.CONFLICT_BLOCKS_TOTAL,
            len(manual_blocks),
            tags={"repo_id": task.repo_id},
        )
        emit_counter(
            MetricName.TASKS_TOTAL,
            1,
            tags={"trigger": task.trigger_type, "status": "AWAITING_APPROVAL"},
        )

    async def _complete_blank_approval(self, task) -> None:
        ok, _queued = await self._deps.repo.complete_all_approved_and_enqueue(
            task_id=task.id, from_version=task.version, approved_count=0, no_manual_approval=True,
        )
        if ok:
            task.status = TaskStatus.ALL_APPROVED.value
            task.version += 1
            task.active_key = None
            observe_task(task)
            log.info("resolve_all_approved", task_id=task.id, new_mr_id=task.new_mr_id, automatic=True)
            await self._reply_result(task)
            emit_counter(MetricName.TASKS_TOTAL, 1, tags={"trigger": task.trigger_type, "status": "ALL_APPROVED"})

    async def resume_blank_approval(self, task_id: int) -> bool:
        task = await self._deps.repo.get_with_blocks(task_id)
        if (task is None or task.status != TaskStatus.MR_CREATED.value
                or not task.trigger_payload.get("_blank_approval_publication_verified")
                or not task.blocks or not all(approval_not_required(b) for b in task.blocks)):
            return False
        mr = await get_mr(self._deps.nc, task.repo_id, task.new_mr_id)
        if (mr.source_branch != task.resolve_branch or mr.target_branch != task.target_branch
                or mr.latest_version is None or mr.latest_version.source_commit_id != task.resolve_commit_sha):
            raise RuntimeError("automatic resolver MR changed before completion recovery")
        await self._complete_blank_approval(task)
        return True

    async def _get_result_commit(self, task, revision):
        try:
            return await get_commit(self._deps.nc, task.repo_id, revision)
        except NextCodeError as exc:
            if exc.status == 404 or exc.code.startswith("NotFound"):
                raise CocoValidationError(f"resolver commit {revision} does not exist") from exc
            raise

    @staticmethod
    def _check_conflict_markers(files: dict[str, str]) -> None:
        import re

        pattern = re.compile(r"^(?:<{7}(?:\s|$)|={7}$|>{7}(?:\s|$))")
        remaining = [
            f"{path}:{line_no}"
            for path, content in files.items()
            for line_no, line in enumerate(content.splitlines(), 1)
            if pattern.match(line)
        ]
        if remaining:
            raise CocoValidationError("resolver commit still contains conflict markers at " + ", ".join(remaining))

    @staticmethod
    def _bind_result(task, result):
        fields = dict(task_id=task.id, attempt=task.retry_count + 1, resolve_branch=task.resolve_branch)
        if isinstance(result, CocoRetryResult):
            return replace(
                result, **fields, base_commit_sha=task.resolve_commit_sha, parent_commits=(task.resolve_commit_sha,)
            )
        return replace(result, **fields, merge_parents=(task.source_commit_sha, task.target_commit_sha))

    @staticmethod
    def _round_context(task) -> dict:
        state = (getattr(task, "trigger_payload", None) or {}).get("_coco_round") or {}
        if state.get("session_id") != task.coco_task_id:
            return {}
        return {
            "workspace_id": task.coco_workspace_id,
            "run_id": state.get("run_id", ""),
            "after_event_id": state.get("event_id", ""),
        }

    @staticmethod
    def _repair_state(task) -> dict:
        state = (getattr(task, "trigger_payload", None) or {}).get("_coco_validation_repair") or {}
        return state if state.get("session_id") == getattr(task, "coco_task_id", None) else {}

    def _valid_parents(self, task, actual: tuple[str, ...], expected: tuple[str, ...]) -> bool:
        repair_base = self._repair_state(task).get("repair_base_sha")
        return actual == expected or bool(repair_base and actual == (repair_base,))

    @staticmethod
    def _author_fields(conflict) -> dict:
        fields = {"author_email": conflict.source_attribution.author_email, "attribution_status": "VERIFIED"}
        for side in ("source", "target"):
            attribution = getattr(conflict, f"{side}_attribution")
            fields.update(
                {
                    f"{side}_author_email": attribution.author_email,
                    f"{side}_author_commit_sha": attribution.commit_sha,
                    f"{side}_path": attribution.path,
                    f"{side}_start_line": attribution.start_line,
                    f"{side}_end_line": attribution.end_line,
                }
            )
        return fields

    @staticmethod
    def _check_author_shape(block, conflict) -> None:
        if not approval_not_required(block) and min(conflict.resolved.start_line, conflict.resolved.end_line) < 1:
            raise CocoValidationError(f"{conflict.conflict_id}: manual conflict requires a positive discussion anchor")
        for side, field in (("source", "ours_text"), ("target", "theirs_text")):
            attribution = getattr(conflict, f"{side}_attribution")
            if attribution.path != block.file_path:
                raise CocoValidationError(f"{conflict.conflict_id}: {side} attribution path differs from manifest")
            original = getattr(block, field, None)
            empty = original == "" or approval_not_required(block)
            if original is not None and empty != (attribution.author_email is None):
                raise CocoValidationError(f"{conflict.conflict_id}: {side} author must be null only for an empty side")

    async def _record_position_warning(self, task, conflict, content: str) -> None:
        lines = content.splitlines()
        start, end = conflict.resolved.start_line, conflict.resolved.end_line
        actual = "\n".join(lines[start - 1 : end])
        if end <= len(lines) and actual == _normalize_resolved_text(conflict.resolved.text):
            return
        details = {
            "conflict_id": conflict.conflict_id,
            "path": conflict.resolved.path,
            "start_line": start,
            "end_line": end,
            "file_line_count": len(lines),
            "reason": "resolved position/text differs from committed file; report uses Coco output",
        }
        log.warning("coco_report_position_warning", task_id=task.id, **details)
        try:
            if self._deps.repo is not None:
                await self._record_event(task.id, "COCO_REPORT_WARNING", details)
        except Exception:
            log.exception("coco_report_warning_record_failed", task_id=task.id)

    @_logged_stage
    async def _repair_or_fail(self, task, reason: str, *, retry: bool) -> None:
        """Claim a single send with task CAS before making an external write."""
        import time

        from conciliator.integrations.coco.prompts import repair_prompt

        if self._repair_state(task):
            await self._fail_validation(task, reason)
            return
        # Only an independently verified, already-pushed task commit can parent an additional repair.
        expected = (task.resolve_commit_sha,) if retry else (task.source_commit_sha, task.target_commit_sha)
        head = await get_branch(self._deps.nc, task.repo_id, task.resolve_branch)
        repair_base = ""
        if head is not None and head.commit_sha:
            try:
                commit = await self._get_result_commit(task, head.commit_sha)
            except CocoValidationError:
                commit = None
            if commit and commit.id == head.commit_sha and commit.parents == expected:
                repair_base = head.commit_sha
        full = await self._deps.repo.get_with_blocks(task.id)
        if full is None or full.version != task.version or full.status != task.status:
            return
        marker = f"conciliator:validation-repair:task={task.id}:session={task.coco_task_id}"
        state = {
            "session_id": task.coco_task_id,
            "phase": "sending",
            "marker": marker,
            "started_at": time.time(),
            "errors": [reason],
            "repair_base_sha": repair_base,
            "after_event_id": self._round_context(task).get("after_event_id", ""),
        }
        payload = {**task.trigger_payload, "_coco_validation_repair": state}
        claimed = await self._deps.repo.update_fields(
            task_id=task.id, from_version=task.version, trigger_payload=payload
        )
        if not claimed:
            return
        task.version += 1
        task.trigger_payload = payload
        rejected, locked = await self._retry_blocks_for_attempt(full) if retry else (full.blocks, [])
        context = {
            "repo_id": task.repo_id,
            "source_commit": task.source_commit_sha,
            "target_commit": task.target_commit_sha,
            "resolve_branch": task.resolve_branch,
            "expected_parents": list(expected),
            "repair_base_sha": repair_base,
            "observed_resolve_head": head.commit_sha if head else None,
            "conflicts": [
                {
                    "conflict_id": b.conflict_id,
                    "path": b.file_path,
                    "source_text": b.ours_text or "",
                    "approval_not_required": approval_not_required(b),
                    "target_text": b.theirs_text or "",
                }
                for b in rejected
            ],
            "locked_conflict_ids": [b.conflict_id for b in locked],
        }
        prompt = repair_prompt(marker=marker, errors=[reason], context=context, retry=retry)
        try:
            event = await self._deps.coco.send_user_message(
                workspace_id=task.coco_workspace_id,
                session_id=task.coco_task_id,
                text=prompt,
            )
        except Exception:
            # A timeout is not proof of non-delivery. Recover by the persisted message marker; never resend blindly.
            log.exception("coco_repair_send_uncertain", task_id=task.id)
            return
        await self._accept_repair_event(task, event)

    async def _accept_repair_event(self, task, event) -> None:
        if event.session_id != task.coco_task_id or not event.event_id or not event.run_id:
            raise CocoError(
                status=200,
                action="SendManagedAgentSessionEvents",
                message="repair message response has no matching Session/Event/Run identity",
            )
        state = {**self._repair_state(task), "phase": "running"}
        payload = {
            **task.trigger_payload,
            "_coco_validation_repair": state,
            "_coco_round": {"session_id": task.coco_task_id, "run_id": event.run_id, "event_id": event.event_id},
        }
        await self._update_fields(
            task, trigger_payload=payload, coco_run_id=event.run_id, coco_last_event_id=event.event_id
        )
        await self._record_event(
            task.id,
            "COCO_VALIDATION_REPAIR_SENT",
            {
                "session_id": task.coco_task_id,
                "run_id": event.run_id,
                "user_event_id": event.event_id,
                "errors": state["errors"],
            },
            coco_event_id=event.event_id,
        )

    async def _recover_repair(self, task) -> bool:
        import time

        from conciliator.integrations.coco.client import _event_text

        state = self._repair_state(task)
        if state.get("phase") != "sending":
            return False
        events = await self._deps.coco.read_round_events(
            workspace_id=task.coco_workspace_id,
            session_id=task.coco_task_id,
            after_event_id=state.get("after_event_id", ""),
        )
        matches = [e for e in events if e.kind == "UserMessage" and _event_text(e).startswith(state["marker"] + "\n")]
        if matches:
            await self._accept_repair_event(task, matches[0])
        elif time.time() - state["started_at"] > 120:
            await self._fail_validation(
                task, "自动修复消息发送结果无法确认，未重复发送。原校验问题：" + state["errors"][0]
            )
        return True

    async def _fail_validation(self, task, reason: str) -> None:
        payload = {**task.trigger_payload, "_failure_notice": {"state": "pending", "reason": reason[:4000]}}
        ok = await self._deps.repo.update_status(
            task_id=task.id,
            from_version=task.version,
            new_status=TaskStatus.FAILED,
            last_error=reason[:8000],
            trigger_payload=payload,
        )
        if not ok:  # cancellation or another worker won; never overwrite that decision
            return
        task.version += 1
        log.warning("resolve_validation_failed", task_id=task.id, from_status=task.status)
        task.status = TaskStatus.FAILED.value
        observe_task(task)
        task.trigger_payload = payload
        try:
            await self._record_event(task.id, "COCO_VALIDATION_FAILED", {"reason": reason[:4000], "repair_count": 1})
        finally:
            await self.notify_validation_failure(task.id)

    @traced_work
    async def notify_validation_failure(self, task_id: int) -> None:
        """Durable failed-task notification; thread readback avoids duplicates after uncertain writes."""
        import time

        task = await self._deps.repo.get(task_id, include_bodies=False)
        if task is None or task.status != TaskStatus.FAILED.value:
            return
        notice = (task.trigger_payload or {}).get("_failure_notice") or {}
        if notice.get("state") not in {"pending", "sending"}:
            return
        if notice.get("state") == "sending" and time.time() - notice.get("started_at", 0) < 300:
            return
        thread_id = task.trigger_payload.get("_trigger_discussion_id")
        if not thread_id:
            await self._update_fields(
                task,
                trigger_payload={**task.trigger_payload, "_failure_notice": {**notice, "state": "no_trigger_thread"}},
            )
            return
        sending = {**notice, "state": "sending", "started_at": time.time()}
        payload = {**task.trigger_payload, "_failure_notice": sending}
        claimed = await self._deps.repo.update_fields(
            task_id=task.id, from_version=task.version, trigger_payload=payload
        )
        if not claimed:
            return
        task.version += 1
        task.trigger_payload = payload
        marker = f"<!-- conciliator:validation-failed:task={task.id} -->"
        try:
            thread = await get_thread(self._deps.nc, task.repo_id, thread_id)
            existing = next((c for c in thread.comments if marker in str(c.get("Content") or "")), None)
            if existing is None:
                # JSON quoting keeps model/validation content visibly distinct from bot instructions.
                reason = json.dumps(notice.get("reason", "未知校验错误"), ensure_ascii=False)
                body = (
                    f"{marker}\n自动解冲突任务 #{task.id} 失败（FAILED）。\n\n"
                    "已尝试一次自动修复，仍未通过阻断校验，或无法确认修复请求已送达。\n\n"
                    f"失败原因：\n{templates._code_block(reason)}\n\n"
                    "本任务已停止后续处理。请检查原因，处理后可重新发送 `/resolve`。"
                )
                reply = await reply_discussion(self._deps.nc, task.repo_id, task.original_mr_id, thread_id, body)
                comment_id = reply.comment_id
            else:
                comment_id = str(existing.get("Id") or "")
            await self._update_fields(
                task,
                trigger_payload={
                    **task.trigger_payload,
                    "_failure_notice": {**sending, "state": "sent", "comment_id": comment_id},
                },
            )
            await self._record_event(
                task.id, "COCO_FAILURE_COMMENT_SENT", {"thread_id": thread_id, "comment_id": comment_id}
            )
        except Exception:
            # Leave the durable sending state. A later poll re-reads the thread before considering another send.
            log.exception("coco_failure_comment_pending", task_id=task.id)

    async def _verify_coco_result(
        self,
        task,
        result: CocoResolveResult,
    ) -> dict[str, str]:
        if not task.source_commit_sha or not task.target_commit_sha:
            raise CocoValidationError("task has no frozen source/target commits")
        if result.task_id != task.id:
            raise CocoValidationError("Coco result task_id does not match the local task")
        if result.attempt != task.retry_count + 1:
            raise CocoValidationError("Coco result attempt does not match the local task")
        if result.resolve_branch != task.resolve_branch:
            raise CocoValidationError("Coco result resolve_branch does not match the local task")
        expected_parents = (task.source_commit_sha, task.target_commit_sha)
        if result.merge_parents != expected_parents:
            raise CocoValidationError("Coco result merge parents do not match frozen commits")
        if result.unresolved_items:
            raise CocoValidationError("Coco unresolved items: " + "; ".join(result.unresolved_items))

        expected_blocks = {block.conflict_id: block for block in task.blocks}
        actual_blocks = {conflict.conflict_id: conflict for conflict in result.conflicts}
        if len(actual_blocks) != len(result.conflicts):
            raise CocoValidationError("Coco result contains duplicate conflict IDs")
        if set(actual_blocks) != set(expected_blocks):
            raise CocoValidationError(
                f"conflict IDs differ: missing={sorted(set(expected_blocks) - set(actual_blocks))}, "
                f"extra={sorted(set(actual_blocks) - set(expected_blocks))}"
            )
        for conflict_id, conflict in actual_blocks.items():
            if conflict.path != expected_blocks[conflict_id].file_path:
                raise CocoValidationError(f"Coco result path differs for {conflict_id}")

        resolve_branch = await get_branch(self._deps.nc, task.repo_id, task.resolve_branch)
        if resolve_branch is None or resolve_branch.commit_sha != result.commit_sha:
            raise CocoValidationError(
                f"resolver HEAD={resolve_branch.commit_sha if resolve_branch else None} "
                f"does not match returned commit_sha={result.commit_sha}"
            )

        commit = await self._get_result_commit(task, result.commit_sha)
        if commit.id != result.commit_sha or not self._valid_parents(task, commit.parents, expected_parents):
            raise CocoValidationError(
                f"resolver commit has unexpected parents={commit.parents}; expected={expected_parents}"
            )

        conflict_paths = sorted({conflict.path for conflict in result.conflicts})
        files = await asyncio.gather(
            *[
                get_file(
                    self._deps.nc,
                    task.repo_id,
                    revision=result.commit_sha,
                    path=path,
                )
                for path in conflict_paths
            ]
        )
        if any(file is None for file in files):
            raise CocoValidationError("resolver commit is missing a conflict file")
        resolved_files = {
            path: file.content for path, file in zip(conflict_paths, files, strict=True) if file is not None
        }
        self._check_conflict_markers(resolved_files)

        try:
            verify_blank_files(task, actual_blocks, resolved_files)
        except ValueError as exc:
            raise CocoValidationError(str(exc)) from exc
        for conflict in result.conflicts:
            if conflict.resolved.path != conflict.path:
                raise CocoValidationError(f"resolved path differs from conflict path for {conflict.conflict_id}")
            resolved_content = resolved_files.get(conflict.resolved.path)
            if resolved_content is None:
                raise CocoValidationError(f"resolved file is missing for {conflict.conflict_id}")
            if not approval_not_required(expected_blocks[conflict.conflict_id]):
                await self._record_position_warning(task, conflict, resolved_content)
            self._check_author_shape(expected_blocks[conflict.conflict_id], conflict)
            await self._verify_attribution(
                task,
                conflict.source_attribution,
                revision=task.source_commit_sha,
                side="source",
            )
            await self._verify_attribution(
                task,
                conflict.target_attribution,
                revision=task.target_commit_sha,
                side="target",
            )
        return resolved_files

    @_logged_stage
    async def _verify_retry_baseline(self, task) -> None:
        _check_supported_conflicts(task)
        required = (
            task.mr_version,
            task.source_commit_sha,
            task.target_commit_sha,
            task.resolve_commit_sha,
            task.new_mr_id,
        )
        if any(value in (None, "") for value in required):
            raise RuntimeError("retry task is missing its frozen baseline")
        resolver, resolver_mr = await asyncio.gather(
            get_branch(self._deps.nc, task.repo_id, task.resolve_branch),
            get_mr(self._deps.nc, task.repo_id, task.new_mr_id),
        )
        if resolver is None or resolver.commit_sha != task.resolve_commit_sha:
            raise RuntimeError("resolve branch moved before retry")
        if resolver_mr.source_branch != task.resolve_branch or (resolver_mr.target_branch != task.target_branch):
            raise RuntimeError("resolver MR branches changed before retry")

    async def _handle_retry_coco_failure(self, task, error: str) -> None:
        resolver = await get_branch(
            self._deps.nc,
            task.repo_id,
            task.resolve_branch,
        )
        safe_to_retry = bool(resolver is not None and resolver.commit_sha == task.resolve_commit_sha)
        target = TaskStatus.PARTIAL_REJECTED if safe_to_retry else TaskStatus.FAILED
        await self._transition_with_fields(
            task,
            target,
            last_error=error,
        )
        await self._record_event(
            task.id,
            "RETRY_FAILED",
            {
                "stage": TaskStatus.RETRY_COCO_RUNNING.value,
                "safe_to_retry": safe_to_retry,
                "error": error,
            },
        )

    @_logged_stage
    async def _verify_retry_result(
        self,
        task,
        result: CocoRetryResult,
        *,
        rejected: list,
        locked: list,
    ) -> dict[str, str]:
        _check_supported_conflicts(task)
        if result.task_id != task.id:
            raise CocoValidationError("Coco retry task_id does not match the local task")
        if result.attempt != task.retry_count + 1:
            raise CocoValidationError("Coco retry attempt does not match the local task")
        if result.resolve_branch != task.resolve_branch:
            raise CocoValidationError("Coco retry resolve_branch does not match the task")
        if result.base_commit_sha != task.resolve_commit_sha:
            raise CocoValidationError("Coco retry base commit does not match the task")
        if result.parent_commits != (task.resolve_commit_sha,):
            raise CocoValidationError("Coco retry commit must only parent the prior resolve commit")
        if result.unresolved_items:
            raise CocoValidationError("Coco retry unresolved items: " + "; ".join(result.unresolved_items))

        expected = {block.conflict_id: block for block in rejected}
        actual = {conflict.conflict_id: conflict for conflict in result.conflicts}
        if len(actual) != len(result.conflicts):
            raise CocoValidationError("Coco retry result contains duplicate conflict IDs")
        if set(actual) != set(expected):
            raise CocoValidationError(
                f"retry conflict IDs differ: missing={sorted(set(expected) - set(actual))}, "
                f"extra={sorted(set(actual) - set(expected))}"
            )
        for conflict_id, conflict in actual.items():
            if conflict.path != expected[conflict_id].file_path:
                raise CocoValidationError(f"Coco retry path differs for {conflict_id}")

        resolver = await get_branch(self._deps.nc, task.repo_id, task.resolve_branch)
        if resolver is None or resolver.commit_sha != result.commit_sha:
            raise CocoValidationError(
                f"retry resolver HEAD={resolver.commit_sha if resolver else None} "
                f"does not match returned commit_sha={result.commit_sha}"
            )
        commit = await self._get_result_commit(task, result.commit_sha)
        if commit.id != result.commit_sha or not self._valid_parents(task, commit.parents, (task.resolve_commit_sha,)):
            raise CocoValidationError(
                f"retry commit has unexpected parents={commit.parents}; expected={(task.resolve_commit_sha,)}"
            )

        paths = sorted({block.file_path for block in [*rejected, *locked]})
        files = await asyncio.gather(
            *[
                get_file(
                    self._deps.nc,
                    task.repo_id,
                    revision=result.commit_sha,
                    path=path,
                )
                for path in paths
            ]
        )
        if any(file is None for file in files):
            raise CocoValidationError("retry commit is missing a conflict file")
        resolved_files = {path: file.content for path, file in zip(paths, files, strict=True) if file is not None}
        self._check_conflict_markers(resolved_files)
        try:
            verify_blank_files(task, actual, resolved_files)
        except ValueError as exc:
            raise CocoValidationError(str(exc)) from exc
        for block in locked:
            if approval_not_required(block):
                continue
            content = resolved_files[block.file_path]
            if block.ai_resolution:
                if block.ai_resolution not in content:
                    raise CocoValidationError(f"locked conflict {block.conflict_id} changed during retry")
            else:
                before = await get_file(
                    self._deps.nc, task.repo_id, revision=task.resolve_commit_sha, path=block.file_path
                )
                if before is None or before.content != content:
                    raise CocoValidationError(f"cannot verify deleted locked conflict {block.conflict_id}")
        for conflict in result.conflicts:
            if conflict.resolved.path != conflict.path:
                raise CocoValidationError(f"retry resolved path differs for {conflict.conflict_id}")
            content = resolved_files[conflict.resolved.path]
            await self._record_position_warning(task, conflict, content)
            self._check_author_shape(expected[conflict.conflict_id], conflict)
            await self._verify_attribution(
                task,
                conflict.source_attribution,
                revision=task.source_commit_sha,
                side="source",
            )
            await self._verify_attribution(
                task,
                conflict.target_attribution,
                revision=task.target_commit_sha,
                side="target",
            )
        return resolved_files

    @_logged_stage
    async def _step_retry_document(
        self,
        task,
        result: CocoRetryResult,
        resolved_files: dict[str, str],
    ) -> None:
        if self._deps.lark is None:
            raise RuntimeError("Lark client is unavailable")
        full = await self._deps.repo.get_with_blocks(task.id)
        if full is None:
            raise RuntimeError("retry task disappeared before document upload")
        blocks = _report_blocks(full)
        markdown = templates.build_lark_doc_markdown(
            original_mr_number=task.original_mr_id,
            source_branch=task.source_branch,
            source_commit=task.source_commit_sha,
            target_branch=task.target_branch,
            target_commit=task.target_commit_sha,
            resolve_branch=task.resolve_branch,
            resolve_commit=result.commit_sha,
            blocks=blocks,
            resolved_files=resolved_files,
        )
        old_url = task.lark_doc_url
        uploaded = await upload_markdown(
            self._deps.lark,
            file_name=(f"conciliator-mr-{task.original_mr_id}-task-{task.id}-attempt-{result.attempt}.md"),
            content=markdown.encode("utf-8"),
            folder_token=self._deps.settings.lark_markdown_folder_token,
        )
        await self._transition_with_fields(
            task,
            TaskStatus.RETRY_DOC_UPLOADED,
            lark_doc_url=uploaded.web_url,
        )
        await self._record_event(
            task.id,
            "RETRY_LARK_MARKDOWN_UPLOADED",
            {
                "attempt": result.attempt,
                "file_token": uploaded.file_token,
                "file_name": uploaded.file_name,
                "old_url": old_url,
                "new_url": uploaded.web_url,
            },
        )

    @_logged_stage
    async def _publish_retry(
        self,
        task,
        rejected: list,
        result: CocoRetryResult,
    ) -> None:
        if task.new_mr_id is None:
            raise RuntimeError("retry task has no resolver MR")
        if TaskStatus(task.status) is TaskStatus.RETRY_DOC_UPLOADED:
            await self._transition(task, TaskStatus.RETRY_PUBLISHING)
        elif TaskStatus(task.status) is not TaskStatus.RETRY_PUBLISHING:
            raise RuntimeError(f"cannot publish retry from {task.status}")
        events = await self._deps.repo.list_events(task.id)
        already_replied = {
            str(event.payload.get("conflict_id"))
            for event in events
            if event.event_type == "RETRY_DISCUSSION_REPLIED"
            and int(event.payload.get("attempt") or 0) == result.attempt
        }
        replied_comment_ids: dict[str, str] = {}
        for block in rejected:
            if block.conflict_id in already_replied:
                continue
            if not block.discussion_id:
                raise RuntimeError(f"conflict {block.conflict_id} has no existing Discussion")
            marker = f"conciliator:task={task.id}:block={block.conflict_id}:attempt={result.attempt}"
            thread = await get_thread(
                self._deps.nc,
                task.repo_id,
                block.discussion_id,
            )
            existing = next(
                (comment for comment in thread.comments if marker in str(comment.get("Content") or "")),
                None,
            )
            if existing is not None:
                comment_id = str(existing.get("Id") or "")
                replied_comment_ids[block.conflict_id] = comment_id
                await self._record_event(
                    task.id,
                    "RETRY_DISCUSSION_REPLIED",
                    {
                        "attempt": result.attempt,
                        "conflict_id": block.conflict_id,
                        "discussion_id": block.discussion_id,
                        "comment_id": comment_id,
                        "recovered": True,
                    },
                )
                continue
            conflict = next(item for item in result.conflicts if item.conflict_id == block.conflict_id)
            reply = await reply_discussion(
                self._deps.nc,
                task.repo_id,
                task.new_mr_id,
                block.discussion_id,
                templates.build_retry_discussion_body(
                    task_id=task.id,
                    attempt=result.attempt,
                    conflict_id=block.conflict_id,
                    idx=block.block_idx + 1,
                    file_path=conflict.resolved.path,
                    start_line=conflict.resolved.start_line,
                    end_line=conflict.resolved.end_line,
                    ai_resolution=conflict.resolved.text,
                    decision_summary=conflict.decision_summary,
                    reject_reason=block.approval.comment,
                    lark_doc_url=task.lark_doc_url,
                    source_author_email=conflict.source_attribution.author_email,
                    target_author_email=conflict.target_attribution.author_email,
                    source_location=(
                        f"{conflict.source_attribution.path}:"
                        f"{conflict.source_attribution.start_line}-{conflict.source_attribution.end_line}"
                    ),
                    target_location=(
                        f"{conflict.target_attribution.path}:"
                        f"{conflict.target_attribution.start_line}-{conflict.target_attribution.end_line}"
                    ),
                ),
            )
            replied_comment_ids[block.conflict_id] = reply.comment_id
            await self._record_event(
                task.id,
                "RETRY_DISCUSSION_REPLIED",
                {
                    "attempt": result.attempt,
                    "conflict_id": block.conflict_id,
                    "discussion_id": block.discussion_id,
                    "comment_id": reply.comment_id,
                },
            )

        full = await self._deps.repo.get_with_blocks(task.id)
        if full is None:
            raise RuntimeError("retry task disappeared before MR update")
        retry_ids = {block.conflict_id for block in rejected}
        description = templates.build_mr_description(
            original_mr_number=task.original_mr_id,
            source_branch=task.source_branch,
            source_commit=task.source_commit_sha,
            target_branch=task.target_branch,
            target_commit=task.target_commit_sha,
            resolve_branch=task.resolve_branch,
            resolve_commit=task.resolve_commit_sha,
            lark_doc_url=task.lark_doc_url,
            blocks=_mr_blocks(full, pending_conflict_ids=retry_ids),
            retry_count=task.retry_count,
            task_status=TaskStatus.AWAITING_APPROVAL.value,
        )
        await update_mr_description(
            self._deps.nc,
            task.repo_id,
            task.new_mr_id,
            description,
        )
        mr = await get_mr(self._deps.nc, task.repo_id, task.new_mr_id)
        if (
            mr.description != description
            or mr.source_branch != task.resolve_branch
            or mr.target_branch != task.target_branch
            or mr.latest_version is None
            or mr.latest_version.source_commit_id != task.resolve_commit_sha
        ):
            raise RuntimeError("resolver MR description differs after retry update")
        reset = await self._deps.repo.reset_rejected_approvals(
            task_id=task.id,
            conflict_ids=tuple(sorted(retry_ids)),
        )
        if reset != len(retry_ids):
            latest = await self._deps.repo.get_with_blocks(task.id)
            if latest is None:
                raise RuntimeError("retry task disappeared after approval reset")
            latest_by_id = {block.conflict_id: block for block in latest.blocks}
            already_reset = True
            for conflict_id in retry_ids:
                approval = latest_by_id[conflict_id].approval
                if approval is None or approval.status != ApprovalStatus.PENDING.value:
                    already_reset = False
                    break
            if not already_reset:
                raise RuntimeError("retry approval reset count differs from snapshot")
        await self._transition(task, TaskStatus.AWAITING_APPROVAL)
        await self._reply_result(task)
        await self._record_event(
            task.id,
            "RETRY_PUBLISHED",
            {
                "attempt": result.attempt,
                "resolve_commit": task.resolve_commit_sha,
                "lark_doc_url": task.lark_doc_url,
                "conflict_ids": sorted(retry_ids),
                "reply_comment_ids": replied_comment_ids,
            },
        )

    async def _verify_attribution(
        self,
        task,
        attribution: ConflictAttribution,
        *,
        revision: str,
        side: str,
    ) -> None:
        if attribution.author_email is None:
            return
        try:
            segments = await get_blame(
                self._deps.nc,
                task.repo_id,
                revision=revision,
                path=attribution.path,
                start_line=attribution.start_line,
                end_line=attribution.end_line,
            )
        except NextCodeError as exc:
            log.warning(
                "resolve_blame_failed", task_id=task.id, side=side, action="GetBlame",
                request_body={
                    "RepoId": task.repo_id, "Revision": revision, "Path": attribution.path,
                    "StartLine": attribution.start_line, "EndLine": attribution.end_line,
                },
                http_status=exc.status, error_code=exc.code, request_id=exc.request_id,
                error_message=safe_text(exc.message)[:1000],
            )
            if exc.status in {400, 404} or exc.code.startswith(("NotFound", "InvalidParameter")):
                raise CocoValidationError(
                    f"{side} blame could not read {attribution.path}:{attribution.start_line}-{attribution.end_line}: "
                    f"{exc.code or exc.status}"
                ) from exc
            raise
        if not any(
            segment.commit_id == attribution.commit_sha
            and segment.author_email.lower() == attribution.author_email.lower()
            for segment in segments
        ):
            raise CocoValidationError(
                f"{side} attribution for {attribution.path}:"
                f"{attribution.start_line}-{attribution.end_line} does not match NextCode blame"
            )

    @_logged_stage
    async def _create_or_recover_resolve_mr(
        self,
        task,
        *,
        title: str,
        description: str,
    ):
        existing = await list_repo_mrs(
            self._deps.nc,
            task.repo_id,
            source_branch=task.resolve_branch,
            target_branch=task.target_branch,
            status="open",
        )
        owned = [mr for mr in existing if mr.title == title]
        if len(owned) > 1:
            raise RuntimeError("multiple resolver MRs exist for the same task")
        if owned:
            if owned[0].description != description:
                raise RuntimeError("existing resolver MR description differs from this task")
            return owned[0]
        try:
            return await create_mr(
                self._deps.nc,
                task.repo_id,
                source=task.resolve_branch,
                target=task.target_branch,
                title=title,
                description=description,
            )
        except (NextCodeError, TransientError):
            recovered = await list_repo_mrs(
                self._deps.nc,
                task.repo_id,
                source_branch=task.resolve_branch,
                target_branch=task.target_branch,
                status="open",
            )
            owned = [mr for mr in recovered if mr.title == title and mr.description == description]
            if len(owned) == 1:
                return owned[0]
            raise

    async def _transition(self, task, target: TaskStatus) -> None:
        current = TaskStatus(task.status)
        if not can_transition(current, target):
            raise InvalidTransitionError(current, target)
        ok = await self._deps.repo.update_status(
            task_id=task.id,
            from_version=task.version,
            new_status=target,
        )
        if not ok:
            log.warning(
                "resolve_transition_conflict", task_id=task.id, from_status=current.value,
                to_status=target.value, task_version=task.version,
            )
            raise InvalidTransitionError(current, target)
        log.info(
            "resolve_status_changed", task_id=task.id, from_status=current.value,
            to_status=target.value, task_version=task.version + 1,
        )
        task.version += 1
        task.status = target.value
        observe_task(task)

    async def _transition_with_fields(
        self,
        task,
        target: TaskStatus,
        **fields,
    ) -> None:
        current = TaskStatus(task.status)
        if not can_transition(current, target):
            raise InvalidTransitionError(current, target)
        ok = await self._deps.repo.update_status(
            task_id=task.id,
            from_version=task.version,
            new_status=target,
            **fields,
        )
        if not ok:
            log.warning(
                "resolve_transition_conflict", task_id=task.id, from_status=current.value,
                to_status=target.value, task_version=task.version,
            )
            raise InvalidTransitionError(current, target)
        log.info(
            "resolve_status_changed", task_id=task.id, from_status=current.value,
            to_status=target.value, task_version=task.version + 1,
        )
        task.version += 1
        task.status = target.value
        observe_task(task)
        # 同步内存字段
        for k, v in fields.items():
            setattr(task, k, v)
        observe_task(task)
        if target == TaskStatus.FAILED:
            reason = str(fields.get("last_error") or "")[:1000]
            await self._reply_original(task, f"解冲突任务 #{task.id} 失败（阶段：{current.value}）：{reason}")

    async def _update_fields(self, task, **fields) -> None:
        ok = await self._deps.repo.update_fields(
            task_id=task.id,
            from_version=task.version,
            **fields,
        )
        if not ok:
            raise RuntimeError("task changed while persisting external result fields")
        task.version += 1
        for key, value in fields.items():
            setattr(task, key, value)
        observe_task(task)

    async def _record_event(
        self,
        task_id: int,
        event_type: str,
        payload: dict,
        *,
        coco_event_id: str | None = None,
    ) -> None:
        await self._deps.repo.add_event(
            task_id=task_id,
            event_type=event_type,
            payload=payload,
            coco_event_id=coco_event_id,
        )
        log.info(
            "resolve_milestone", task_id=task_id, milestone=event_type,
            **{key: payload[key] for key in (
                "block_count", "attempt", "new_mr_id", "session_id", "run_id", "approved_count",
                "conflict_id", "discussion_id", "not_required_count", "recovered",
            ) if key in payload},
        )


def _resolution_text(resolved_text: str) -> str:
    return resolved_text


def _normalize_resolved_text(text: str) -> str:
    return "\n".join(text.splitlines())


def _webhook_event_id(payload: dict) -> str | None:
    metadata = payload.get("_webhook") or {}
    value = str(metadata.get("event_id") or "")
    return value or None


def _retry_result_from_dict(value: dict | None) -> CocoRetryResult:
    if not value:
        raise RuntimeError("retry task has no persisted Coco result")
    wire = {key: value[key] for key in ("schema_version", "status", "commit_sha", "conflicts", "unresolved_items")}
    result = parse_retry_result("CONCILIATOR_RESULT_BEGIN\n" + json.dumps(wire) + "\nCONCILIATOR_RESULT_END")
    return replace(
        result,
        task_id=value["task_id"],
        attempt=value["attempt"],
        resolve_branch=value["resolve_branch"],
        base_commit_sha=value["base_commit_sha"],
        parent_commits=tuple(value["parent_commits"]),
    )


def _partition_retry_blocks(task) -> tuple[list, list]:
    rejected = []
    locked = []
    for block in sorted(task.blocks, key=lambda item: item.block_idx):
        if block.approval is not None and block.approval.status == ApprovalStatus.REJECTED.value:
            if not block.approval.comment:
                raise RuntimeError(f"rejected conflict {block.conflict_id} has no reason")
            rejected.append(block)
        else:
            locked.append(block)
    return rejected, locked


def _mr_blocks(
    task,
    *,
    pending_conflict_ids: set[str] | None = None,
) -> list[dict]:
    pending_conflict_ids = pending_conflict_ids or set()
    rendered: list[dict] = []
    for block in sorted(task.blocks, key=lambda item: item.block_idx):
        approval = getattr(block, "approval", None)
        status = approval.status if approval is not None else "UNKNOWN"
        reason = getattr(approval, "comment", None)
        if block.conflict_id in pending_conflict_ids and not approval_not_required(block):
            status = ApprovalStatus.PENDING.value
            reason = None
        rendered.append(
            {
                "idx": block.block_idx + 1,
                "conflict_id": block.conflict_id,
                "file_path": block.file_path,
                "start_line": block.start_line,
                "end_line": block.end_line,
                "source_author_email": block.source_author_email,
                "target_author_email": block.target_author_email,
                "author_email": block.author_email,
                "resolved_path": block.resolved_path,
                "resolved_start_line": block.resolved_start_line,
                "resolved_end_line": block.resolved_end_line,
                "approved": status == ApprovalStatus.APPROVED.value,
                "approval_status": status,
                "reject_reason": reason,
            }
        )
    return rendered


def _report_blocks(task) -> list[dict]:
    return [
        {
            "idx": block.block_idx + 1,
            "approval_status": ("NOT_REQUIRED" if approval_not_required(block) else "PENDING"),
            "conflict_id": block.conflict_id,
            "file_path": block.file_path,
            "start_line": block.start_line,
            "end_line": block.end_line,
            "source_text": block.ours_text or "",
            "target_text": block.theirs_text or "",
            "ai_resolution": block.ai_resolution or "",
            "decision_summary": block.decision_summary or "",
            "source_author_email": block.source_author_email,
            "source_author_commit_sha": block.source_author_commit_sha,
            "source_path": block.source_path,
            "source_start_line": block.source_start_line,
            "source_end_line": block.source_end_line,
            "target_author_email": block.target_author_email,
            "target_author_commit_sha": block.target_author_commit_sha,
            "target_path": block.target_path,
            "target_start_line": block.target_start_line,
            "target_end_line": block.target_end_line,
            "resolved_path": block.resolved_path,
            "resolved_start_line": block.resolved_start_line,
            "resolved_end_line": block.resolved_end_line,
        }
        for block in sorted(task.blocks, key=lambda item: item.block_idx)
    ]
