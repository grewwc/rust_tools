from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from conciliator.domain.aeolus_task import (
    AeolusStartEvent,
    AeolusState,
    AeolusTask,
    CodeContext,
    DeploymentVerification,
    can_retry_coding,
    can_retry_completion,
    extract_requirement_url,
    parse_start_event,
)
from conciliator.integrations.coco.aeolus import CocoAeolusRunner, CocoRunFailed, CocoSessionContext, EvaluationResult
from conciliator.integrations.lark.client import LarkError
from conciliator.integrations.meegle.ai_node import AINodeRef, MeegleAINodeClient
from conciliator.integrations.nextcode.branches import create_branch, get_branch
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.commits import get_commit
from conciliator.integrations.nextcode.mrs import create_mr, list_repo_mrs
from conciliator.integrations.nextcode.repos import get_repository
from conciliator.services.aeolus_document_access import (
    LarkAuthorizationRequired,
    referenced_document_urls,
)
from conciliator.services.aeolus_registry import (
    AeolusTaskRegistry,
    InvalidTaskTransition,
    TaskLeaseLost,
    TaskNotFound,
    task_lease,
)
from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import error_fields, logged_step, workflow_step
from conciliator.utils.retry import TransientError

log = get_logger(__name__)

_LEASE_RENEW_SECONDS = 15

_question_answer_acceptor: ContextVar[Callable[[str], Awaitable[bool]] | None] = ContextVar(
    "aeolus_question_answer_acceptor", default=None
)


@contextmanager
def question_answer_acceptance(acceptor: Callable[[str], Awaitable[bool]]) -> Iterator[None]:
    """Persist a question result before its service releases the task lease."""
    token = _question_answer_acceptor.set(acceptor)
    try:
        yield
    finally:
        _question_answer_acceptor.reset(token)

# Appended by the terminal transition that drops acknowledged feedback it can
# no longer run, so the failure card states what happened to those comments.
_PENDING_FEEDBACK_NOTE = "\n另有 {count} 条已确认的评论反馈未完成处理，已随本次失败终止。"

# A missing operator token carries no Feishu code of its own. Keep reporting
# 131006 for older card versions, while document_authorization_required below
# distinguishes missing consent from a real document ACL denial.
_AUTHORIZATION_REQUIRED_CODE = 131006


def _document_read_error_code(exc: Exception) -> int | None:
    """Return the code the card uses to decide whether to show authorization."""
    if isinstance(exc, LarkError):
        return exc.code
    if isinstance(exc, LarkAuthorizationRequired):
        return _AUTHORIZATION_REQUIRED_CODE
    return None


def authorization_blocked(evaluation: dict[str, object] | None) -> bool:
    if not evaluation:
        return False
    if evaluation.get("document_authorization_required") is True:
        return True
    if evaluation.get("document_authorization_required") is False:
        return False
    missing_items = evaluation.get("missing_items")
    return (
        evaluation.get("document_read_error_code") == _AUTHORIZATION_REQUIRED_CODE
        and isinstance(missing_items, list)
        and any(
            isinstance(item, str) and "尚未完成飞书授权或授权已失效" in item
            for item in missing_items
        )
    )


def _reference_read_failure(url: str, exc: Exception) -> dict[str, object]:
    if isinstance(exc, LarkError):
        if exc.code == 131006:
            reason = "permission_denied"
            message = "应用和当前用户均无法读取该文档，请为应用添加阅读权限或为当前用户申请文档访问权限"
        elif exc.code == 3380004:
            reason = "permission_denied"
            message = "应用无该文档权限，请将应用添加为文档协作者或为当前用户申请文档访问权限"
        elif exc.code == 99991672:
            reason = "app_scope_missing"
            message = "飞书应用缺少读取该文档所需的权限，请联系应用管理员"
        else:
            reason = "read_failed"
            message = "飞书文档读取失败，请确认文档仍然存在且链接有效"
    elif isinstance(exc, TransientError):
        reason = "temporarily_unavailable"
        message = "飞书文档暂时无法读取，请稍后重试"
    elif isinstance(exc, ValueError):
        reason = "invalid_document"
        message = "引用文档为空、类型不受支持或链接无效，请检查引用文档"
    else:  # pragma: no cover - guarded by the caller
        raise TypeError("unexpected reference precheck failure") from exc
    failure: dict[str, object] = {"url": url, "reason": reason, "message": message}
    if isinstance(exc, LarkError):
        failure["code"] = exc.code
    return failure


def _reference_precheck_evaluation(
    total: int, failures: list[dict[str, object]]
) -> dict[str, object]:
    reasons = {item["reason"] for item in failures}
    permission_codes = [
        item["code"]
        for item in failures
        if item["reason"] == "permission_denied"
        and isinstance(item.get("code"), int)
    ]
    error_code = permission_codes[0] if permission_codes else (
        131006 if "permission_denied" in reasons else (
            99991672 if "app_scope_missing" in reasons else None
        )
    )
    return {
        "valid": False,
        "summary": "引用文档权限预检查未通过",
        "missing_items": [
            f'{item["message"]}：{item["url"]}' for item in failures
        ],
        "risks": [],
        "document_read_error_code": error_code,
        "document_authorization_required": False,
        "document_precheck": {
            "status": "failed",
            "total_documents": total,
            "failed_documents": failures,
        },
    }


@dataclass(frozen=True)
class PreparedBranch:
    repo_id: str
    branch: str


@dataclass(frozen=True)
class ReviewRef:
    id: str
    iid: int | None
    url: str


class CodeHost(Protocol):
    async def resolve_context(self, context: CodeContext) -> CodeContext: ...

    async def prepare_branch(self, task_id: str, context: CodeContext) -> PreparedBranch: ...

    async def verify_commit(
        self, repo_id: str, branch: str, commit_sha: str
    ) -> None: ...

    async def create_review(
        self, *, repo_id: str, branch: str, task_id: str, title: str, target_branch: str,
        requirement_contract: str = ""
    ) -> ReviewRef: ...


class DocumentAccess(Protocol):
    def instructions(
        self,
        task: AeolusTask,
        *,
        include_requirement_document: bool = False,
    ) -> str: ...


class NextCodeAeolusHost:
    def __init__(self, client: NextCodeClient) -> None:
        self._client = client

    async def resolve_context(self, context: CodeContext) -> CodeContext:
        repository = await get_repository(self._client, path=context.repository)
        base = await get_branch(self._client, repository.id, context.base_branch)
        if base is None or not base.commit_sha:
            raise ValueError(f"仓库 {context.repository} 的 base 分支 {context.base_branch} 不可读取")
        target = base if context.target_branch == context.base_branch else await get_branch(
            self._client, repository.id, context.target_branch,
        )
        if target is None or not target.commit_sha:
            raise ValueError(f"仓库 {context.repository} 的 MR 目标分支 {context.target_branch} 不可读取")
        commit = base.commit_sha
        if context.base_commit:
            details = await get_commit(self._client, repository.id, context.base_commit)
            if not details.id.lower().startswith(context.base_commit.lower()):
                raise ValueError("指定 base commit 与仓库返回的 commit 不一致")
            commit = details.id
        if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise ValueError("代码基线必须解析为完整 commit SHA")
        return replace(context, base_commit=commit)

    @workflow_step("branch_prepare")
    async def prepare_branch(self, task_id: str, context: CodeContext) -> PreparedBranch:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", context.base_commit):
            raise ValueError("创建编码分支前必须固定 base commit")
        repository = await get_repository(self._client, path=context.repository)
        branch_name = _branch_name(task_id)
        if branch_name in {context.base_branch, context.target_branch}:
            raise ValueError("编码分支不能与 base 或 MR 目标分支相同")
        branch = await get_branch(self._client, repository.id, branch_name)
        if branch is None:
            branch = await create_branch(
                self._client, repository.id, context.base_commit, branch_name
            )
        return PreparedBranch(repository.id, branch.name)

    async def verify_commit(
        self, repo_id: str, branch: str, commit_sha: str
    ) -> None:
        current = await get_branch(self._client, repo_id, branch)
        if current is None or current.commit_sha != commit_sha:
            raise RuntimeError(
                f"Coco commit was not pushed to {branch}: expected {commit_sha}"
            )

    @workflow_step("mr_create_or_reuse")
    async def create_review(
        self, *, repo_id: str, branch: str, task_id: str, title: str, target_branch: str,
        requirement_contract: str = ""
    ) -> ReviewRef:
        existing = await list_repo_mrs(
            self._client, repo_id,
            source_branch=branch, target_branch=target_branch,
        )
        if existing:
            mr = existing[0]
        else:
            description = f"由 aeolus 编码助手根据需求文档生成。\n\n任务 ID：{task_id}"
            excerpt = _review_contract_excerpt(requirement_contract)
            if excerpt:
                description += f"\n\n---\n\n## 已确认的需求要点\n\n{excerpt}"
            mr = await create_mr(
                self._client,
                repo_id,
                source=branch,
                target=target_branch,
                title=title,
                description=description,
            )
        log.info("meegle_mr_ready", mr_id=mr.id, mr_iid=mr.iid, reused=bool(existing))
        return ReviewRef(mr.id, mr.iid, mr.web_url)


class AeolusService:
    def __init__(
        self,
        *,
        registry: AeolusTaskRegistry,
        document_reader: Callable[[str, str], Awaitable[str]],
        coco: CocoAeolusRunner,
        code_host: CodeHost,
        meegle: MeegleAINodeClient,
        plugin_id: str = "",
        webhook_token: str = "",
        minimum_processing_seconds: float = 10.0,
        platform_retry_delay_seconds: float = 120.0,
        notify: Callable[[AeolusTask], Awaitable[None]] | None = None,
        deployments=None,
        ppe_deployments=None,
        document_access: DocumentAccess | None = None,
    ) -> None:
        self.registry = registry
        self.deployments = deployments
        self.ppe_deployments = ppe_deployments
        self._read_document = document_reader
        self._coco = coco
        self._code_host = code_host
        self._meegle = meegle
        self._plugin_id = plugin_id
        self._webhook_token = webhook_token
        self._minimum_processing_seconds = minimum_processing_seconds
        self._platform_retry_delay_seconds = platform_retry_delay_seconds
        self._notify = notify
        self._document_access = document_access
        self._background: set[asyncio.Task[None]] = set()
        self._recovery_worker: asyncio.Task[None] | None = None

    def _document_instructions(
        self,
        task: AeolusTask,
        *,
        include_requirement_document: bool = False,
    ) -> str:
        if self._document_access is None:
            return ""
        return self._document_access.instructions(
            task, include_requirement_document=include_requirement_document
        )

    async def _precheck_reference_documents(
        self, task: AeolusTask
    ) -> tuple[int, list[dict[str, object]]]:
        # App-only reading predates the operator-scoped document proxy. Do not
        # make its access to optional references a new evaluation prerequisite.
        if self._document_access is None:
            return 0, []
        urls = referenced_document_urls(task)
        if not urls:
            return 0, []
        semaphore = asyncio.Semaphore(5)

        async def read(url: str) -> str:
            async with semaphore:
                return await self._read_document(url, task.user_key)

        results = await asyncio.gather(
            *(read(url) for url in urls), return_exceptions=True
        )
        failures: list[dict[str, object]] = []
        for url, result in zip(urls, results, strict=True):
            if not isinstance(result, BaseException):
                continue
            if isinstance(result, LarkAuthorizationRequired):
                raise result
            if isinstance(result, (LarkError, TransientError, ValueError)):
                failures.append(_reference_read_failure(url, result))
                continue
            raise result
        return len(urls), failures

    async def _record_validation_failure(
        self,
        task: AeolusTask,
        started: float,
        evaluation: dict[str, object],
        *,
        requirement_text: str | None = None,
    ) -> None:
        await self._minimum_delay(started)
        task.evaluation_rounds[-1].update(
            status="error",
            result=evaluation,
            finished_at=datetime.now(UTC).isoformat(),
        )
        updates: dict[str, Any] = {
            "evaluation": evaluation,
            "evaluation_rounds": task.evaluation_rounds,
        }
        if requirement_text is not None:
            updates["requirement_text"] = requirement_text
        await self.registry.transition(
            task.task_id,
            {AeolusState.READING_DOCUMENT},
            AeolusState.VALIDATION_FAILED,
            pending_note=_PENDING_FEEDBACK_NOTE,
            **updates,
        )

    async def start(self) -> None:
        """Verify the pending-feedback schema and reclaim expired leases."""
        if self._recovery_worker is not None:
            return
        await self.registry.ensure_pending_feedback_schema()
        await self.recover_pending()

        async def scan() -> None:
            while True:
                await asyncio.sleep(5)
                try:
                    await self.recover_pending()
                except Exception as exc:
                    log.warning("aeolus_recovery_scan_failed", workflow="meegle", **error_fields(exc))

        self._recovery_worker = asyncio.create_task(scan(), name="aeolus:recovery")

    async def recover_pending(self) -> None:
        active = {task.get_name() for task in self._background}
        for task_id in await self.registry.recoverable_task_ids():
            name = f"aeolus:{task_id}"
            if name not in active:
                self._spawn(task_id, self._resume_task(task_id), recovery=True)
                active.add(name)
        # Feedback that was acknowledged before the aggregation window ended is
        # recovered independently of the round scan: its task may still be in
        # REVIEW and therefore absent from recoverable_task_ids().
        for task_id in await self.registry.pending_feedback_task_ids(datetime.now(UTC)):
            name = f"aeolus:{task_id}"
            if name not in active:
                # Pending feedback is legitimate in REVIEW, so it must bypass
                # the round-recovery guard; the lease still serializes it with
                # any coding run on the session.
                self._spawn(task_id, self._flush_pending_feedback(task_id))
                active.add(name)

    @workflow_step("task_recovery")
    async def _resume_task(self, task_id: str) -> None:
        task = await self.registry.get(task_id)
        if task.state == AeolusState.COMPLETING:
            await self._complete(task_id)
        elif task.state == AeolusState.READING_DOCUMENT:
            await self._run_validation(task_id, already_started=True)
        elif task.state == AeolusState.EVALUATING:
            cursor = task.evaluation_rounds[-1] if task.evaluation_rounds else {}
            if not all(cursor.get(key) for key in ("session_id", "run_id", "event_id")):
                raise RuntimeError("评估会话缺少恢复游标，无法安全恢复原 Run，请重试评估")
            log.info("aeolus_evaluation_resumed", task_id=task_id, run_id=cursor["run_id"])
            try:
                evaluation = await self._coco.resume_evaluation(
                    session_id=cursor["session_id"], run_id=cursor["run_id"], event_id=cursor["event_id"],
                    repair_attempted=bool(cursor.get("repair_attempted")),
                    on_started=lambda session, run, event: self._record_evaluation_run(task_id, session, run, event),
                )
            except CocoRunFailed as exc:
                # The cursor can outlive its run: a restart inside the platform
                # backoff leaves a session whose run is already terminal.
                if not (exc.retryable or exc.replacement_safe):
                    raise
                log.warning(
                    "aeolus_evaluation_resume_failed",
                    task_id=task_id,
                    run_id=cursor["run_id"],
                    **error_fields(exc),
                )
                await self._restart_evaluation(task_id, exc)
                return
            await self._accept_evaluation_result(task_id, evaluation, time.monotonic())
        else:
            await self._resume_coding(task_id)

    async def _restart_evaluation(self, task_id: str, exc: Exception) -> None:
        """Re-evaluate from a fresh session when the resumed run is terminal."""
        task = await self.registry.get(task_id)
        if task.state != AeolusState.EVALUATING:
            return
        if task.evaluation_rounds:
            # Keep the dead session out of the reuse selection below.
            task.evaluation_rounds[-1].update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                error_kind=_error_kind(exc),
                finished_at=datetime.now(UTC).isoformat(),
            )
            await self.registry.update(
                task_id, evaluation_rounds=task.evaluation_rounds
            )
        await self.registry.transition(
            task_id, {AeolusState.EVALUATING}, AeolusState.READING_DOCUMENT
        )
        await self._run_validation(task_id, already_started=True)

    async def _resume_coding(self, task_id: str) -> None:
        task = await self.registry.get(task_id)
        if task.state not in {AeolusState.CODING, AeolusState.RECODING}:
            return
        if task.rounds and task.rounds[-1].status == "retrying":
            await self._run_retry_coding(task_id)
            return
        if task.rounds and task.rounds[-1].status == "queued":
            await self._run_feedback(task_id)
            return
        if not task.rounds or not task.rounds[-1].attempts:
            # The initial coding round is committed before its first run exists
            # and is safe to start. A feedback round must carry the queued state;
            # otherwise the request may already have reached Coco before a crash.
            if task.state == AeolusState.CODING:
                await self._start_pending_round(task_id, task)
                return
            raise RuntimeError(
                "编码请求可能已发送但缺少恢复游标，请重新发起本轮编码"
            )
        attempt = task.rounds[-1].attempts[-1]
        if not attempt.session_id or not attempt.run_id or not attempt.event_id:
            raise RuntimeError("编码会话缺少恢复游标，请重新发起本轮编码")
        if attempt.status == "failed":
            # A run that failed on the platform is replaced after a backoff, and
            # the attempt is persisted as failed before that wait starts.  A
            # restart inside the wait lands here with a dead cursor: resuming it
            # only replays the terminal failure, which the worker turns into a
            # failed task, so start the replacement the interrupted flow meant
            # to start.
            log.info(
                "aeolus_coding_replaces_failed_attempt",
                task_id=task_id,
                attempt_no=attempt.attempt_no,
            )
            await self._run_retry_coding(task_id, resume_cursor=False)
            return
        started = time.monotonic()
        log.info("aeolus_coding_resumed", task_id=task_id, run_id=attempt.run_id)
        try:
            run = await self._coco.resume_coding(
                session_id=attempt.session_id, run_id=attempt.run_id, event_id=attempt.event_id,
                on_started=lambda session, run, event: self._record_coco_attempt(
                    task_id, session, run, event
                ),
            )
        except CocoRunFailed as exc:
            # A cursor can outlive its run when the process stopped before the
            # attempt was marked failed.  Replacing the run is safe once the
            # platform reports it terminal, so recover instead of failing the
            # whole task.
            if not (exc.retryable or exc.replacement_safe):
                raise
            log.warning(
                "aeolus_coding_resume_failed",
                task_id=task_id,
                run_id=attempt.run_id,
                **error_fields(exc),
            )
            await self._run_retry_coding(task_id, resume_cursor=False)
            return
        if await self._accept_coding_result(task_id, run):
            return
        if task.state == AeolusState.CODING or not task.mr_url:
            review = await self._code_host.create_review(
                repo_id=task.repo_id, target_branch=_task_code_context(task).target_branch,
                branch=task.work_branch, task_id=task_id, title=_review_title(task),
                requirement_contract=(task.evaluation or {}).get("requirement_contract") or "",
            )
        else:
            review = ReviewRef(task.mr_id, task.mr_iid, task.mr_url)
        await self._minimum_delay(started)
        await self.registry.update_last_round(task_id, code_url=review.url)
        await self.registry.transition(
            task_id, {task.state}, AeolusState.REVIEW,
            mr_id=review.id, mr_iid=review.iid, mr_url=review.url,
        )

    async def _start_pending_round(self, task_id: str, task: AeolusTask) -> None:
        """Start a committed round whose first run was never persisted."""
        log.info(
            "aeolus_coding_round_started", task_id=task_id, state=task.state.value
        )
        result = task.evaluation or {}
        evaluation = EvaluationResult(
            True,
            str(result.get("summary", "")),
            tuple(result.get("missing_items") or ()),
            tuple(result.get("risks") or ()),
        )
        await self._run_initial_coding(task_id, evaluation)

    async def accept_webhook(
        self, payload: dict
    ) -> tuple[AeolusTask | None, bool]:
        event = parse_start_event(
            payload,
            expected_plugin_id=self._plugin_id,
            webhook_token=self._webhook_token,
        )
        if event is None:
            log.debug("meegle_webhook_ignored", workflow="meegle")
            return None, False
        return await self.accept_event(event)

    async def accept_event(
        self, event: AeolusStartEvent
    ) -> tuple[AeolusTask, bool]:
        task, created = await self.registry.create_from_event(event)
        log.info(
            "meegle_webhook_accepted", workflow="meegle", task_id=task.task_id,
            event_id=event.task_id, created=created, state=task.state.value,
            project_key=task.project_key, work_item_id=task.work_item_id, node_uuid=task.node_uuid,
        )
        if created:
            self._spawn(task.task_id, self._run_validation(task.task_id))
        elif task.state == AeolusState.RECEIVED:
            # A previous process died between creating the task and starting
            # its first evaluation.
            self._spawn(task.task_id, self._run_validation_if_received(task.task_id))
        elif task.state in {AeolusState.CODING, AeolusState.RECODING} and (
            not task.rounds or not task.rounds[-1].attempts
        ):
            # The node was restarted after a crash that left the round without
            # its first run; reuse would otherwise park it forever.
            self._spawn(task.task_id, self._resume_coding(task.task_id))
        elif task.state in {
            AeolusState.COMPLETED,
            AeolusState.FAILED,
            AeolusState.VALIDATION_FAILED,
        }:
            # The platform flips a finished node back to running before it
            # sends the start event.  Reusing the task row is right, but
            # without a reply the node stays running with nothing left to do.
            self._spawn(
                task.task_id, self._restart_finished_task(task.task_id, event.task_id)
            )
        return task, created

    async def _run_validation_if_received(self, task_id: str) -> None:
        task = await self.registry.get(task_id)
        if task.state != AeolusState.RECEIVED:
            return
        await self._run_validation(task_id)

    async def _restart_finished_task(self, task_id: str, request_id: str) -> None:
        """Answer a start event for a task that already reached a terminal state.

        Reusing the row keeps the work already done, so the restart only has to
        put the node back into a state the platform understands: done for a
        completed task, and a retry for one that failed.
        """
        task = await self.registry.get(task_id)
        try:
            if task.state == AeolusState.COMPLETED:
                await self._reconfirm_completed(task_id, request_id)
            elif task.state == AeolusState.VALIDATION_FAILED:
                _, started = await self.registry.review_evaluation(
                    task_id, request_id, feedback=""
                )
                if started:
                    await self._run_validation(task_id, already_started=True)
            elif can_retry_completion(task):
                _, started = await self.registry.begin_completion(
                    task_id, request_id, True
                )
                if started:
                    await self._complete(task_id)
            elif can_retry_coding(task):
                _, started = await self.registry.start_coding_retry(
                    task_id, request_id
                )
                if started:
                    await self._run_retry_coding(task_id)
        except InvalidTaskTransition:
            # Another callback advanced the task after the state was read; this
            # restart has nothing left to answer, and failing here would turn a
            # duplicate delivery into a failed task.
            log.info(
                "aeolus_restart_skipped", task_id=task_id, state=task.state.value
            )

    async def _reconfirm_completed(self, task_id: str, request_id: str) -> None:
        task = await self.registry.get(task_id)
        if task.state != AeolusState.COMPLETED:
            return
        ref = AINodeRef(
            task.project_key, task.work_item_id, task.node_uuid, task.user_key
        )
        # The completion that ran first is deduplicated by its idempotency key,
        # so the restart needs keys of its own to put the node back into done.
        await self._meegle.edit_status(ref, "done", idem_key=f"{request_id}:done")
        await self._meegle.confirm(ref, idem_key=f"{request_id}:confirm")

    async def get(self, task_id: str) -> AeolusTask:
        task = await self.registry.get(task_id)
        if self.deployments is not None:
            snapshot = await self.deployments.run("snapshot", task_id)
            task.deployment = snapshot["deployment"]
            task.deployment_workers = snapshot["deployment_workers"]
        if self.ppe_deployments is not None:
            snapshot = await self.ppe_deployments.snapshot(task_id)
            task.ppe_deployment = snapshot["ppe_deployment"]
            task.ppe_capability = snapshot["ppe_capability"]
        return task

    async def find_by_node(
        self, project_key: str, work_item_id: int, node_id: str
    ) -> AeolusTask:
        task = await self.registry.find_by_node(project_key, work_item_id, node_id)
        return await self.get(task.task_id)

    @workflow_step("coding_retry_request")
    async def retry_coding(self, task_id: str, request_id: str) -> AeolusTask:
        task, started = await self.registry.start_coding_retry(task_id, request_id)
        if started:
            self._spawn(task_id, self._run_retry_coding(task_id))
        return task

    @workflow_step("evaluation_retry_request")
    async def retry_validation(self, task_id: str) -> AeolusTask:
        task, started = await self.registry.review_evaluation(
            task_id, str(uuid.uuid4()), feedback=""
        )
        if started:
            self._spawn(task_id, self._run_validation(task_id, already_started=True))
        return task

    async def resume_authorization_blocked(self, user_key: str) -> int:
        """Resume this user's evaluations that stopped only for missing consent."""
        tasks = await self.registry.validation_failed_for_user(user_key)
        resumed = 0
        for task in tasks:
            if not authorization_blocked(task.evaluation):
                continue
            try:
                await self.retry_validation(task.task_id)
            except InvalidTaskTransition:
                # Another callback or operator action already advanced the task.
                continue
            resumed += 1
        if resumed:
            log.info("aeolus_authorization_blocked_resumed", workflow="meegle", user_key=user_key, count=resumed)
        return resumed

    @workflow_step("evaluation_feedback_request")
    async def evaluation_feedback(
        self, task_id: str, feedback: str, request_id: str
    ) -> AeolusTask:
        feedback = feedback.strip()
        if not feedback:
            raise ValueError("feedback must not be empty")
        task, started = await self.registry.review_evaluation(
            task_id, request_id, feedback=feedback
        )
        if started:
            self._spawn(task_id, self._run_validation(task_id, already_started=True))
        return task

    @workflow_step("evaluation_confirm_request")
    async def confirm_evaluation(self, task_id: str, request_id: str) -> AeolusTask:
        task, started = await self.registry.review_evaluation(task_id, request_id)
        if started:
            result = task.evaluation or {}
            evaluation = EvaluationResult(
                True, result["summary"], tuple(result["missing_items"]), tuple(result["risks"])
            )
            self._spawn(task_id, self._run_initial_coding(task_id, evaluation))
        return task

    @workflow_step("coding_feedback_request")
    async def submit_feedback(
        self,
        task_id: str,
        feedback: str,
        request_ids: str | Sequence[str],
        *,
        entries: Sequence[tuple[str, str]] | None = None,
    ) -> AeolusTask:
        feedback = feedback.strip()
        if not feedback and entries is None:
            raise ValueError("feedback must not be empty")
        task, started = await self.registry.start_feedback(
            task_id, feedback, request_ids, entries=entries
        )
        if started:
            self._spawn(task_id, self._run_feedback(task_id))
        return task

    async def save_pending_feedback(
        self,
        task_id: str,
        entries: Sequence[tuple[str, str]],
        *,
        available_at: datetime,
    ) -> bool:
        """Persist accepted comments before the webhook acknowledges them."""
        return await self.registry.save_pending_feedback(
            task_id, entries, available_at=available_at
        )

    async def discard_pending_feedback(
        self, task_id: str, comment_ids: Sequence[str]
    ) -> None:
        await self.registry.discard_pending_feedback(task_id, comment_ids)

    async def due_pending_feedback(self, task_id: str) -> list[tuple[str, str]]:
        """Read the whole batch from the shared table, not from worker memory."""
        return await self.registry.due_pending_feedback(task_id)

    async def claim_comment(
        self, task_id: str, request_id: str, ttl: timedelta
    ) -> datetime | None:
        """Claim a comment id across workers before spending an agent run."""
        return await self.registry.claim_comment(task_id, request_id, ttl)

    async def release_comment(
        self, task_id: str, request_id: str, token: datetime | None
    ) -> None:
        await self.registry.release_comment(task_id, request_id, token)

    async def complete_comment(
        self, task_id: str, request_id: str, token: datetime | None
    ) -> bool:
        return await self.registry.complete_comment(task_id, request_id, token)

    async def renew_comment(
        self, task_id: str, request_id: str, ttl: timedelta, token: datetime | None
    ) -> bool:
        return await self.registry.renew_comment(task_id, request_id, ttl, token)

    async def _flush_pending_feedback(self, task_id: str) -> None:
        # Only rows whose window already closed: a stale due-task snapshot must
        # not flush a comment that a later window is still aggregating.
        entries = await self.registry.due_pending_feedback(task_id)
        if not entries:
            return
        try:
            task = await self.registry.get(task_id)
        except TaskNotFound:
            log.warning("aeolus_pending_feedback_task_missing", task_id=task_id)
            await self._discard_pending_feedback(task_id, entries, reason="task_missing")
            return
        if task.state in {
            AeolusState.FAILED,
            AeolusState.VALIDATION_FAILED,
            AeolusState.COMPLETED,
        }:
            # A terminal task can never consume this feedback. Resolve it now
            # instead of rediscovering the same rows on every recovery scan.
            await self._discard_pending_feedback(
                task_id, entries, reason=task.state.value
            )
            return
        feedback = "\n\n".join(text for _, text in entries)
        request_ids = [f"mr-comment:{comment_id}" for comment_id, _ in entries]
        try:
            await self.submit_feedback(
                task_id, feedback, request_ids, entries=entries
            )
        except InvalidTaskTransition:
            # The previous round still owns the session; the next recovery scan
            # retries once the task is reviewable again.
            return
        except TaskNotFound:
            log.warning("aeolus_pending_feedback_task_missing", task_id=task_id)
        await self._discard_pending_feedback(task_id, entries, reason="submitted")

    async def _discard_pending_feedback(
        self, task_id: str, entries: Sequence[tuple[str, str]], *, reason: str
    ) -> None:
        comment_ids = [comment_id for comment_id, _ in entries]
        try:
            await self.registry.discard_pending_feedback(task_id, comment_ids)
        except Exception as exc:
            # The rows stay durable and the recovery scan retries them, so a
            # failed cleanup must not fail the operation that already ran.
            log.warning(
                "aeolus_pending_feedback_discard_failed",
                task_id=task_id,
                reason=reason,
                error_type=type(exc).__name__,
            )
            return
        if reason == "submitted":
            log.info(
                "aeolus_pending_feedback_submitted",
                task_id=task_id,
                comments=len(comment_ids),
            )
        else:
            log.warning(
                "aeolus_pending_feedback_discarded",
                task_id=task_id,
                reason=reason,
                comments=len(comment_ids),
            )

    async def answer_question(
        self, task_id: str, question: str, position: str = "",
        on_started: Callable[[str, str, str], Awaitable[None]] | None = None,
    ) -> str:
        """Answer a review question in the existing coding session.

        The task lease serializes every run on the session. State is checked
        after acquiring it because feedback can start while a question waits.
        """
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        owner = uuid.uuid4().hex
        acquired = False
        try:
            while not acquired:
                acquired = await self.registry.acquire_lease(task_id, owner)
                if not acquired:
                    self._question_session(await self.registry.get(task_id))
                    await asyncio.sleep(0.25)
            with task_lease(owner):
                session_id = self._question_session(await self.registry.get(task_id))
                return await self._question_with_lease(
                    task_id, owner, self._coco.answer(
                        session_id=session_id,
                        question=question,
                        position=position,
                        on_started=on_started,
                    ),
                )
        finally:
            if acquired:
                try:
                    await self.registry.release_lease(task_id, owner)
                except Exception as exc:
                    log.warning(
                        "aeolus_lease_release_failed",
                        task_id=task_id,
                        error_type=type(exc).__name__,
                    )

    async def resume_answer(
        self, task_id: str, session_id: str, run_id: str, event_id: str
    ) -> str:
        """Collect an answer run started before a worker restart.

        Reading a run is not observation-only: the runner interrupts a stalled
        session and answers tool confirmations, so the task lease that
        serializes normal answers guards this path too.  Without it a concurrent
        question or coding round could prompt the session while this worker is
        still driving the old run.
        """
        owner = uuid.uuid4().hex
        acquired = False
        try:
            while not acquired:
                acquired = await self.registry.acquire_lease(task_id, owner)
                if not acquired:
                    self._question_session(await self.registry.get(task_id))
                    await asyncio.sleep(0.25)
            with task_lease(owner):
                task = await self.registry.get(task_id)
                expected = self._question_session(task)
                if expected != session_id:
                    # The task moved to another session; the recorded run can
                    # no longer belong to it.
                    raise InvalidTaskTransition(
                        f"task {task_id} now answers in session {expected}"
                    )
                return await self._question_with_lease(
                    task_id, owner, self._coco.resume_answer(session_id, run_id, event_id)
                )
        finally:
            if acquired:
                try:
                    await self.registry.release_lease(task_id, owner)
                except Exception as exc:
                    log.warning(
                        "aeolus_lease_release_failed",
                        task_id=task_id,
                        error_type=type(exc).__name__,
                    )

    @staticmethod
    def _question_session(task: AeolusTask) -> str:
        if task.state not in {
            AeolusState.REVIEW,
            AeolusState.CODING_BLOCKED,
        }:
            raise InvalidTaskTransition(
                f"task {task.task_id} is {task.state.value}; questions need a free session"
            )
        if not task.coco_session_id:
            raise InvalidTaskTransition(f"task {task.task_id} has no coding session")
        return task.coco_session_id

    @workflow_step("completion_request")
    async def confirm(
        self, task_id: str, request_id: str, verified_id: str = "", release_authorized: bool = False,
        deployment_target: str = "devbox",
        *, verification_targets: tuple[DeploymentVerification, ...] = (),
    ) -> AeolusTask:
        if verified_id and (self.ppe_deployments if deployment_target == "ppe" else self.deployments) is None:
            raise InvalidTaskTransition("deployment POC is disabled")
        for target in verification_targets:
            if (self.ppe_deployments if target.target == "ppe" else self.deployments) is None:
                raise InvalidTaskTransition("verification deployment integration is disabled")
        task = await self.registry.get(task_id)
        if task.state == AeolusState.COMPLETED:
            return task
        retrying_platform_confirm = (
            task.state == AeolusState.FAILED and (task.platform_status_done or task.completion_requested)
        )
        args = (task_id, request_id, retrying_platform_confirm, verified_id,
                release_authorized or bool(verified_id) or bool(verification_targets))
        if verification_targets:
            task, started = await self.registry.begin_completion(
                *args, verification_targets=verification_targets,
            )
        elif deployment_target == "devbox":
            task, started = await self.registry.begin_completion(*args)
        else:
            task, started = await self.registry.begin_completion(*args, deployment_target=deployment_target)
        if not started:
            return await self.get(task_id)
        if self.deployments is not None or self.ppe_deployments is not None:
            task = await self.get(task_id)
        self._spawn(task_id, self._complete(task_id))
        return task

    async def wait_for_idle(self) -> None:
        while self._background:
            await asyncio.gather(*tuple(self._background))

    async def aclose(self) -> None:
        if self._recovery_worker is not None:
            self._recovery_worker.cancel()
            await asyncio.gather(self._recovery_worker, return_exceptions=True)
            self._recovery_worker = None
        for task in tuple(self._background):
            task.cancel()
        if self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)

    def _spawn(self, task_id: str, operation: Awaitable[None], *, recovery: bool = False) -> None:
        async def guarded() -> None:
            owner = uuid.uuid4().hex
            acquired = False
            try:
                while not acquired:
                    acquired = await self.registry.acquire_lease(task_id, owner)
                    if not acquired:
                        if recovery:
                            return
                        await asyncio.sleep(0.25)
                with task_lease(owner):
                    if recovery:
                        current = await self.registry.get(task_id)
                        evaluating = current.state in {AeolusState.READING_DOCUMENT, AeolusState.EVALUATING}
                        coding = current.state in {AeolusState.CODING, AeolusState.RECODING}
                        completing = current.state == AeolusState.COMPLETING
                        # A round can be committed before its first run exists;
                        # _resume_coding starts that case instead of returning.
                        if not evaluating and not completing and not (coding and current.rounds):
                            return
                    await self._run_leased(task_id, owner, operation)
            finally:
                # Also close unscheduled coroutines when another instance owns recovery.
                if hasattr(operation, "close"):
                    operation.close()
                if acquired:
                    try:
                        await self.registry.release_lease(task_id, owner)
                    except Exception as exc:
                        log.warning(
                            "aeolus_lease_release_failed", workflow="meegle", task_id=task_id, **error_fields(exc),
                        )

        task = asyncio.create_task(guarded(), name=f"aeolus:{task_id}")
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    @workflow_step("task_processing")
    async def _run_leased(self, task_id: str, owner: str, operation: Awaitable[None]) -> None:
        async def process() -> None:
            try:
                await operation
            except TaskLeaseLost:
                raise
            except Exception as exc:
                log.error("aeolus_task_failed", task_id=task_id, **error_fields(exc))
                await self._mark_failed(task_id, exc)
            if self._notify is not None:
                try:
                    await asyncio.wait_for(self._notify(await self.get(task_id)), timeout=5)
                except Exception as exc:
                    log.warning("aeolus_notification_failed", task_id=task_id,
                                error_type=type(exc).__name__)

        worker = asyncio.create_task(process())
        renewal = asyncio.create_task(self._lease_heartbeat(task_id, owner))
        try:
            done, _ = await asyncio.wait({worker, renewal}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        except TaskLeaseLost:
            log.warning("aeolus_task_lease_lost", task_id=task_id)
        except Exception as exc:
            # A failed renewal must stop processing, not overwrite task state.
            log.warning("aeolus_leased_worker_stopped", task_id=task_id,
                        error_type=type(exc).__name__)
        finally:
            worker.cancel()
            renewal.cancel()
            await asyncio.gather(worker, renewal, return_exceptions=True)

    async def _question_with_lease(
        self, task_id: str, owner: str, operation: Awaitable[str],
    ) -> str:
        """Keep both new and resumed answers inside a renewable task lease."""
        worker = asyncio.ensure_future(operation)
        renewal = asyncio.create_task(self._lease_heartbeat(task_id, owner))
        try:
            done, _ = await asyncio.wait(
                {worker, renewal}, return_when=asyncio.FIRST_COMPLETED
            )
            if renewal in done:
                await renewal
                raise TaskLeaseLost("AI task processing lease renewal stopped")
            answer = await worker
            try:
                # Read failures still defer, but a successful owner snapshot
                # is advisory: acceptance below fences the write atomically.
                await self.registry.get(task_id)
            except TaskLeaseLost:
                raise
            except Exception as exc:
                raise TaskLeaseLost("AI task processing lease ownership could not be confirmed") from exc
            # A renewal can fail while the final registry read is awaiting I/O.
            if renewal.done():
                await renewal
                raise TaskLeaseLost("AI task processing lease renewal stopped")
            try:
                acceptor = _question_answer_acceptor.get()
                accepted = acceptor is not None and await acceptor(answer)
                if not accepted and not await self.registry.renew_lease(task_id, owner):
                    # Raw callers have no question row to persist; a conditional
                    # renewal fences acceptance of their return value instead.
                    raise TaskLeaseLost("AI task processing lease was lost before answer acceptance")
            except TaskLeaseLost:
                raise
            except Exception as exc:
                raise TaskLeaseLost("AI task answer acceptance could not be confirmed") from exc
            if renewal.done():
                await renewal
                raise TaskLeaseLost("AI task processing lease renewal stopped")
            return answer
        finally:
            worker.cancel()
            renewal.cancel()
            await asyncio.gather(worker, renewal, return_exceptions=True)

    async def _lease_heartbeat(self, task_id: str, owner: str) -> None:
        while True:
            await asyncio.sleep(_LEASE_RENEW_SECONDS)
            try:
                renewed = await self.registry.renew_lease(task_id, owner)
            except TaskLeaseLost:
                raise
            except Exception as exc:
                raise TaskLeaseLost("AI task processing lease renewal could not be confirmed") from exc
            if not renewed:
                raise TaskLeaseLost("AI task processing lease expired")

    @workflow_step("input_evaluation")
    async def _run_validation(
        self, task_id: str, *, already_started: bool = False
    ) -> None:
        started = time.monotonic()
        if not already_started:
            await self.registry.transition(
                task_id, {AeolusState.RECEIVED}, AeolusState.READING_DOCUMENT
            )
        task = await self.registry.get(task_id)
        if not task.evaluation_rounds:
            await self.registry.update(task_id, evaluation_rounds=[{
                "round_no": 1, "status": "evaluating", "feedback": "",
                "reviewed_round_no": None, "result": None,
                "started_at": datetime.now(UTC).isoformat(),
            }])
            task = await self.registry.get(task_id)
        # The props stored at webhook time freeze the requirement document
        # link. Re-read the *current* node props from Meegle on every
        # evaluation so "已更新文档，重新评估" picks up a replaced document
        # instead of re-reading the stale link; fall back to stored props.
        props = task.props
        try:
            data = await asyncio.wait_for(
                self._meegle.query(
                    AINodeRef(
                        task.project_key,
                        task.work_item_id,
                        task.node_uuid,
                        task.user_key,
                    )
                ),
                timeout=10.0,
            )
            fresh = (data.get("ai_info") or {}).get("props")
            if isinstance(fresh, list) and fresh:
                props = tuple(fresh)
        except Exception as exc:
            log.warning(
                "aeolus_meegle_props_refresh_failed",
                task_id=task_id,
                error_type=type(exc).__name__,
            )
        event = AeolusStartEvent(
            task_id=task.task_id,
            project_key=task.project_key,
            work_item_id=task.work_item_id,
            state_key=task.state_key,
            node_uuid=task.node_uuid,
            user_key=task.user_key,
            props=props,
        )
        requirement_url = extract_requirement_url(event)
        if props is not task.props:
            # Keep the stored snapshot consistent so later rounds and the card
            # see the document link that was actually evaluated.
            await self.registry.update(task_id, props=list(props))
        await self.registry.update(task_id, requirement_url=requirement_url)
        try:
            with logged_step("requirement_read", evaluation_round_no=len(task.evaluation_rounds)):
                requirement_text = await self._read_document(requirement_url, task.user_key)
                log.info("meegle_document_read", character_count=len(requirement_text))
        except Exception as exc:
            missing_item = "请确认文档链接有效、文档非空且类型受支持后重试"
            if isinstance(exc, LarkAuthorizationRequired):
                missing_item = (
                    "应用无法读取该文档，当前用户尚未完成飞书授权或授权已失效；"
                    "请为应用添加文档阅读权限后重试，或完成飞书用户授权"
                )
            elif isinstance(exc, TransientError):
                missing_item = "飞书文档服务暂时不可用，请稍后重新评估"
            elif isinstance(exc, LarkError) and exc.code == 99991672:
                missing_item = (
                    "飞书应用缺少读取文档所需的 API 权限（包括知识库 wiki:node:read 等），"
                    "请联系应用管理员开通后重试"
                )
            elif isinstance(exc, LarkError) and exc.code == 131006:
                missing_item = (
                    "应用和当前用户均无法读取该文档，"
                    "请为应用添加阅读权限或为当前用户申请文档访问权限后重试"
                )
            elif isinstance(exc, LarkError) and exc.code == 3380004:
                missing_item = (
                    "应用无该文档权限，"
                    "请将应用添加为文档协作者或为当前用户申请文档访问权限后重试"
                )
            log.warning(
                "aeolus_document_read_failed",
                task_id=task_id,
                error_type=type(exc).__name__,
                lark_code=exc.code if isinstance(exc, LarkError) else None,
                http_status=(
                    exc.http_status if isinstance(exc, LarkError) else None
                ),
            )
            await self._minimum_delay(started)
            evaluation_data = {
                "valid": False,
                "summary": "无法读取需求文档",
                "missing_items": [missing_item],
                "risks": [],
                # Structured code so the card can tell an authorization problem
                # (131006: this document is not shared with the app or with the
                # reading user, or the reading user has not authorized the app)
                # apart from a missing app scope (99991672: only an app
                # administrator can fix it by granting and publishing).
                "document_read_error_code": _document_read_error_code(exc),
                # Keep missing user consent distinct from a real document ACL
                # denial even though Feishu represents both with code 131006.
                "document_authorization_required": isinstance(
                    exc, LarkAuthorizationRequired
                ),
                "document_read_error_reason": (
                    "temporarily_unavailable"
                    if isinstance(exc, TransientError)
                    else ""
                ),
            }
            task.evaluation_rounds[-1].update(
                status="error", result=evaluation_data,
                finished_at=datetime.now(UTC).isoformat(),
            )
            await self.registry.transition(
                task_id,
                {AeolusState.READING_DOCUMENT},
                AeolusState.VALIDATION_FAILED,
                evaluation=evaluation_data,
                evaluation_rounds=task.evaluation_rounds,
            )
            return

        precheck_task = replace(
            task,
            requirement_url=requirement_url,
            requirement_text=requirement_text,
        )
        reference_count = len(referenced_document_urls(precheck_task))
        try:
            with logged_step("reference_precheck", reference_count=reference_count):
                _, reference_failures = await self._precheck_reference_documents(precheck_task)
        except LarkAuthorizationRequired:
            evaluation_data = {
                "valid": False,
                "summary": "飞书授权已失效，无法预检查引用文档",
                "missing_items": [
                    "当前用户飞书授权已失效，请重新授权；授权成功后系统会自动继续评估"
                ],
                "risks": [],
                "document_read_error_code": _AUTHORIZATION_REQUIRED_CODE,
                "document_authorization_required": True,
                "document_precheck": {
                    "status": "authorization_required",
                    "total_documents": reference_count,
                    "failed_documents": [],
                },
            }
            log.warning(
                "aeolus_reference_precheck_authorization_required",
                task_id=task_id,
                reference_count=reference_count,
            )
            await self._record_validation_failure(
                task, started, evaluation_data, requirement_text=requirement_text
            )
            return
        if reference_failures:
            evaluation_data = _reference_precheck_evaluation(
                reference_count, reference_failures
            )
            log.warning(
                "aeolus_reference_precheck_failed",
                task_id=task_id,
                reference_count=reference_count,
                failure_count=len(reference_failures),
                failure_reasons=sorted(
                    {item["reason"] for item in reference_failures}
                ),
            )
            await self._record_validation_failure(
                task, started, evaluation_data, requirement_text=requirement_text
            )
            return
        log.info("meegle_reference_precheck_passed", reference_count=reference_count)
        task = await self.registry.transition(
            task_id,
            {AeolusState.READING_DOCUMENT},
            AeolusState.EVALUATING,
            requirement_text=requirement_text,
        )
        history = [
            {key: item.get(key) for key in ("round_no", "reviewed_round_no", "feedback", "result")}
            for item in task.evaluation_rounds
        ]
        # A round that errored may have left its session broken on the platform
        # (e.g. cloud environment preparation failed permanently). Never reuse
        # such a session: create a fresh one so retries can recover.
        session_id = next((
            item["session_id"] for item in reversed(task.evaluation_rounds)
            if item.get("session_id") and item.get("status") != "error"
        ), "")
        document_instructions = (
            self._document_instructions(task)
            if self._document_access is not None
            else None
        )

        async def run_evaluation(
            session_id: str,
            on_started: Callable[[str, str, str], Awaitable[None]],
        ) -> EvaluationResult:
            session_kwargs: dict[str, Any] = {}
            if not session_id:
                session_kwargs["session_context"] = await self._coco_session_context(task)
            if document_instructions is None:
                return await self._coco.evaluate(
                    requirement_text,
                    session_id=session_id,
                    history=history,
                    on_started=on_started,
                    **session_kwargs,
                )
            return await self._coco.evaluate(
                requirement_text,
                session_id=session_id,
                history=history,
                on_started=on_started,
                document_access_instructions=document_instructions,
                **session_kwargs,
            )

        try:
            with logged_step("coco_evaluation", evaluation_round_no=len(task.evaluation_rounds)):
                evaluation = await run_evaluation(
                    session_id,
                    lambda session, run, event: self._record_evaluation_run(task_id, session, run, event),
                )
        except CocoRunFailed as exc:
            if not exc.retryable:
                raise
            # The old run is terminal and produced no Agent result, so a fresh
            # session is the only safe recovery.  Bound this to one replacement
            # to avoid amplifying platform capacity failures.
            if exc.platform:
                await self._platform_retry_delay()
                current = await self.registry.get(task_id)
                current.evaluation_rounds[-1]["platform_retry_attempted"] = True
                await self.registry.update(
                    task_id, evaluation_rounds=current.evaluation_rounds
                )
            log.warning(
                "aeolus_evaluation_run_replaced",
                task_id=task_id,
                had_session=bool(session_id),
                error_type=type(exc).__name__,
            )
            # A replacement is platform recovery, not a format repair.
            evaluation = await run_evaluation(
                "",
                lambda session, run, event: self._record_evaluation_run(
                    task_id, session, run, event, replacement=True
                ),
            )
        await self._accept_evaluation_result(task_id, evaluation, started)

    async def _record_evaluation_run(
        self, task_id: str, session_id: str, run_id: str, event_id: str,
        *, replacement: bool = False,
    ) -> None:
        task = await self.registry.get(task_id)
        current = task.evaluation_rounds[-1]
        if current.get("run_id") and not replacement:
            current["repair_attempted"] = True
        current.update(session_id=session_id, run_id=run_id, event_id=event_id)
        await self.registry.update(task_id, evaluation_rounds=task.evaluation_rounds)
        log.info(
            "meegle_evaluation_run_recorded", task_id=task_id, evaluation_round_no=len(task.evaluation_rounds),
            session_id=session_id, run_id=run_id, event_id=event_id,
        )

    async def _accept_evaluation_result(
        self, task_id: str, evaluation: EvaluationResult, started: float,
    ) -> None:
        task = await self.registry.get(task_id)
        log.info(
            "meegle_evaluation_result", task_id=task_id, valid=evaluation.valid,
            missing_count=len(evaluation.missing_items), risk_count=len(evaluation.risks),
        )
        if evaluation.valid:
            context = await self._code_host.resolve_context(evaluation.code_context)
            evaluation = replace(
                evaluation, code_context=context,
                requirement_contract=_contract_with_context(evaluation.requirement_contract, context),
            )
        await self._minimum_delay(started)
        evaluation_data = _evaluation_dict(evaluation)
        task.evaluation_rounds[-1].update(
            status="review", result=evaluation_data,
            finished_at=datetime.now(UTC).isoformat(),
        )
        await self.registry.transition(
            task_id,
            {AeolusState.EVALUATING},
            AeolusState.EVALUATION_REVIEW,
            evaluation=evaluation_data,
            evaluation_rounds=task.evaluation_rounds,
        )

    @workflow_step("coding_branch_prepare")
    async def _prepare_coding_branch(self, task: AeolusTask) -> AeolusTask:
        context = _task_code_context(task)
        if not context.base_commit:
            # Old tasks have no frozen context. Resolve once before their first branch creation.
            context = await self._code_host.resolve_context(context)
            evaluation = dict(task.evaluation or {})
            evaluation["code_context"] = asdict(context)
            evaluation["requirement_contract"] = _contract_with_context(
                evaluation.get("requirement_contract") or "", context,
            )
            task = await self.registry.update(task.task_id, evaluation=evaluation)
        prepared = await self._code_host.prepare_branch(task.task_id, context)
        task = await self.registry.update(
            task.task_id, repo_id=prepared.repo_id, work_branch=prepared.branch,
        )
        if task.rounds and task.rounds[-1].trigger_type == "initial":
            task = await self.registry.update_last_round(task.task_id, base_commit_sha=context.base_commit)
        return task

    @workflow_step("coding_retry")
    async def _run_retry_coding(
        self, task_id: str, *, resume_cursor: bool = True
    ) -> None:
        task = await self.registry.get(task_id)
        if task.state not in {AeolusState.CODING, AeolusState.RECODING}:
            return
        started = time.monotonic()
        if not task.repo_id or not task.work_branch:
            task = await self._prepare_coding_branch(task)
        round_ = task.rounds[-1]
        attempt = round_.attempts[-1] if round_.attempts else None
        has_cursor = attempt and attempt.session_id and attempt.run_id and attempt.event_id
        if has_cursor and attempt is not None and attempt.status != "running":
            # Preserve the failed attempt; result collection is a new execution record.
            await self._record_coco_attempt(task_id, attempt.session_id, attempt.run_id, attempt.event_id)
        if round_.commit_sha:
            # Coding succeeded previously; only MR/platform post-processing failed.
            await self._code_host.verify_commit(task.repo_id, task.work_branch, round_.commit_sha)
            if has_cursor:
                await self.registry.update_last_attempt(
                    task_id, status="succeeded", error="", finished_at=datetime.now(UTC),
                )
        else:
            run = None
            # A caller that already knows the recorded run is terminal skips the
            # resume: replaying a dead cursor can only fail again.
            if has_cursor and attempt is not None and resume_cursor:
                try:
                    run = await self._coco.resume_coding(
                        session_id=attempt.session_id, run_id=attempt.run_id, event_id=attempt.event_id,
                        on_started=lambda session, run, event: self._record_coco_attempt(
                            task_id, session, run, event
                        ),
                    )
                except CocoRunFailed as exc:
                    await self._fail_running_attempt(task_id, exc)
                    if not (exc.retryable or exc.replacement_safe):
                        raise
                    if exc.platform:
                        await self._platform_retry_delay()
                # Timeout/network/auth errors do not prove the remote run failed.
                if run is not None and run.result.status == "failed":
                    await self._fail_running_attempt(task_id, CocoRunFailed(run.result.summary))
                    run = None
            if run is None:
                run = await self._start_coco_session(
                    task_id=task_id, repo_id=task.repo_id, branch=task.work_branch,
                    requirement_url=_recovery_prompt(task),
                    evaluation_summary="恢复编码会话并继续当前修改轮次",
                )
            if await self._accept_coding_result(task_id, run):
                return
        review = ReviewRef(task.mr_id, task.mr_iid, task.mr_url)
        if not review.url:
            review = await self._code_host.create_review(
                repo_id=task.repo_id, target_branch=_task_code_context(task).target_branch,
                branch=task.work_branch, task_id=task_id, title=_review_title(task),
                requirement_contract=(task.evaluation or {}).get("requirement_contract") or "",
            )
        await self._minimum_delay(started)
        await self.registry.update_last_round(
            task_id, status="review", error="", code_url=review.url, finished_at=datetime.now(UTC),
        )
        await self.registry.transition(
            task_id, {task.state}, AeolusState.REVIEW,
            mr_id=review.id, mr_iid=review.iid, mr_url=review.url,
        )

    @workflow_step("coding_initial")
    async def _run_initial_coding(
        self, task_id: str, evaluation: EvaluationResult
    ) -> None:
        started = time.monotonic()
        task = await self.registry.get(task_id)
        if task.state != AeolusState.CODING:
            return
        task = await self._prepare_coding_branch(task)
        try:
            run = await self._start_coco_session(
                task_id=task_id,
                repo_id=task.repo_id,
                branch=task.work_branch,
                requirement_url=_coding_requirement(task),
                evaluation_summary=evaluation.summary,
            )
        except (CocoRunFailed, TransientError) as exc:
            # A write call has no transport retry of its own, so one transient
            # network error must not fail the whole task.
            await self._fail_running_attempt(task_id, exc)
            if isinstance(exc, CocoRunFailed) and not exc.retryable:
                raise
            if isinstance(exc, CocoRunFailed) and exc.platform:
                await self._platform_retry_delay()
            run = await self._start_coco_session(
                task_id=task_id,
                repo_id=task.repo_id,
                branch=task.work_branch,
                requirement_url=_coding_requirement(task),
                evaluation_summary=evaluation.summary,
            )
        if await self._accept_coding_result(task_id, run):
            return
        review = await self._code_host.create_review(
            repo_id=task.repo_id, target_branch=_task_code_context(task).target_branch,
            branch=task.work_branch, task_id=task_id, title=_review_title(task),
            requirement_contract=(task.evaluation or {}).get("requirement_contract") or "",
        )
        await self._minimum_delay(started)
        await self.registry.update_last_round(task_id, code_url=review.url)
        await self.registry.transition(
            task_id,
            {AeolusState.CODING},
            AeolusState.REVIEW,
            mr_id=review.id,
            mr_iid=review.iid,
            mr_url=review.url,
        )

    @workflow_step("coding_feedback")
    async def _run_feedback(self, task_id: str) -> None:
        started = time.monotonic()
        task = await self.registry.claim_queued_feedback(task_id)
        if task is None:
            return
        feedback = task.rounds[-1].feedback
        if not feedback:
            raise RuntimeError("queued feedback is empty")
        try:
            document_kwargs = {}
            if self._document_access is not None:
                document_kwargs["document_access_instructions"] = self._document_instructions(
                    task, include_requirement_document=True
                )
            run = await self._coco.recode(
                session_id=task.coco_session_id,
                feedback=feedback,
                on_started=lambda session_id, run_id, event_id: self._record_coco_attempt(
                    task_id, session_id, run_id, event_id
                ),
                **document_kwargs,
            )
        except (CocoRunFailed, TransientError) as exc:
            # A write call has no transport retry of its own, so one transient
            # network error must not fail the whole task.
            await self._fail_running_attempt(task_id, exc)
            if isinstance(exc, CocoRunFailed) and not exc.retryable:
                raise
            if isinstance(exc, CocoRunFailed) and exc.platform:
                await self._platform_retry_delay()
            task = await self.registry.get(task_id)
            run = await self._start_coco_session(
                task_id=task_id,
                repo_id=task.repo_id,
                branch=task.work_branch,
                requirement_url=_recovery_prompt(task),
                evaluation_summary="恢复编码会话并继续当前修改轮次",
            )
        if await self._accept_coding_result(task_id, run):
            return
        review = ReviewRef(task.mr_id, task.mr_iid, task.mr_url)
        if not review.url:
            review = await self._code_host.create_review(
                repo_id=task.repo_id, target_branch=_task_code_context(task).target_branch,
                branch=task.work_branch, task_id=task_id, title=_review_title(task),
                requirement_contract=(task.evaluation or {}).get("requirement_contract") or "",
            )
        await self._minimum_delay(started)
        await self.registry.update_last_round(task_id, code_url=review.url)
        await self.registry.transition(
            task_id, {AeolusState.RECODING}, AeolusState.REVIEW,
            mr_id=review.id, mr_iid=review.iid, mr_url=review.url,
        )

    async def _accept_coding_result(self, task_id: str, run) -> bool:
        """Return True when blocked has been saved and the caller must stop before normal Review."""
        task = await self.registry.get(task_id)
        result = run.result
        log.info(
            "meegle_coding_result", task_id=task_id, round_no=len(task.rounds),
            session_id=run.session_id, run_id=run.run_id, result_status=result.status,
            commit_sha=result.commit_sha, test_count=len(result.tests),
            unresolved_count=len(result.unresolved_items), failure_code=result.failure_code,
        )
        if result.status == "blocked":
            if result.commit_sha:
                await self._code_host.verify_commit(task.repo_id, task.work_branch, result.commit_sha)
            review = ReviewRef(task.mr_id, task.mr_iid, task.mr_url)
            code_url = review.url
            if result.commit_sha and result.commit_sha != task.rounds[-1].base_commit_sha:
                repository = _task_code_context(task).repository
                code_url = f"https://code.byted.org/{repository}/commit/{result.commit_sha}"
                try:
                    review = await self._code_host.create_review(
                        repo_id=task.repo_id, branch=task.work_branch, task_id=task_id,
                        target_branch=_task_code_context(task).target_branch, title=_review_title(task),
                        requirement_contract=(task.evaluation or {}).get("requirement_contract") or "",
                    )
                    code_url = review.url or code_url
                except Exception as exc:
                    # Partial delivery remains reviewable via its verified commit if MR creation is unavailable.
                    log.warning("aeolus_blocked_review_unavailable", task_id=task_id, error_type=type(exc).__name__)
            evaluation = dict(task.evaluation or {})
            evaluation.pop("coding_authorization_required", None)
            await self.registry.block_coding(
                task_id,
                round_changes={
                    "summary": result.summary, "commit_sha": result.commit_sha, "code_url": code_url,
                    "tests": list(result.tests), "unresolved_items": list(result.unresolved_items),
                    "failure_code": result.failure_code, "error": "",
                },
                task_changes={
                    "coco_session_id": run.session_id, "coco_run_id": run.run_id,
                    "coco_last_event_id": run.last_event_id, "evaluation": evaluation,
                    "mr_id": review.id, "mr_iid": review.iid, "mr_url": review.url,
                },
            )
            return True
        if result.status != "success":
            await self.registry.update_last_round(
                task_id, summary=result.summary, tests=list(result.tests),
                unresolved_items=list(result.unresolved_items), failure_code=result.failure_code,
            )
            if result.failure_code == "lark_authorization_required":
                evaluation = dict(task.evaluation or {})
                evaluation["coding_authorization_required"] = True
                await self.registry.update(task_id, evaluation=evaluation)
            await self.registry.update_last_attempt(
                task_id,
                status="failed",
                error=result.summary or "Coco coding failed",
                finished_at=datetime.now(UTC),
            )
            raise RuntimeError(result.summary or "Coco coding failed")
        await self._code_host.verify_commit(
            task.repo_id, task.work_branch, result.commit_sha
        )
        evaluation = dict(task.evaluation or {})
        evaluation.pop("coding_authorization_required", None)
        await self.registry.update(
            task_id,
            coco_session_id=run.session_id,
            coco_run_id=run.run_id,
            coco_last_event_id=run.last_event_id,
            evaluation=evaluation,
        )
        await self.registry.update_last_attempt(
            task_id,
            status="succeeded",
            error="",
            finished_at=datetime.now(UTC),
        )
        await self.registry.update_last_round(
            task_id,
            status="review",
            summary=result.summary,
            commit_sha=result.commit_sha,
            tests=list(result.tests),
            unresolved_items=list(result.unresolved_items),
            failure_code=result.failure_code,
            finished_at=datetime.now(UTC),
        )
        return False

    @workflow_step("coco_coding")
    async def _start_coco_session(
        self,
        *,
        task_id: str,
        repo_id: str,
        branch: str,
        requirement_url: str,
        evaluation_summary: str,
    ):
        task = await self.registry.get(task_id)
        document_kwargs = {}
        if self._document_access is not None:
            document_kwargs["document_access_instructions"] = self._document_instructions(
                task, include_requirement_document=True
            )
        return await self._coco.code(
            repo_id=repo_id,
            repository=_task_code_context(task).repository,
            branch=branch,
            requirement_url=requirement_url,
            evaluation_summary=evaluation_summary,
            session_context=await self._coco_session_context(task),
            on_started=lambda session_id, run_id, event_id: self._record_coco_attempt(
                task_id, session_id, run_id, event_id
            ),
            **document_kwargs,
        )

    async def _coco_session_context(self, task: AeolusTask) -> CocoSessionContext:
        ref = AINodeRef(task.project_key, task.work_item_id, task.node_uuid, task.user_key)

        async def item_name() -> str:
            if not task.work_item_type_key:
                return ""
            return await self._meegle.get_work_item_name(ref, task.work_item_type_key)

        async def user_name() -> str:
            users = await self._meegle.query_users(ref, [task.user_key])
            matched = [user for user in users if user.get("user_key") == task.user_key]
            if len(matched) != 1:
                return ""
            user = matched[0]
            email = user.get("email")
            if isinstance(email, str):
                prefix = email.split("@", 1)[0].strip()
                if prefix:
                    return prefix
            for field in ("name_en", "name_cn"):
                name = user.get(field)
                if isinstance(name, str) and name.strip():
                    return name.strip()
            return ""

        async def lookup(field: str, query: Callable[[], Awaitable[str]]) -> str:
            try:
                async with asyncio.timeout(10):
                    return await query()
            except Exception as exc:
                # A display-name lookup must not block evaluation or coding.
                log.warning("coco_session_name_lookup_failed", task_id=task.task_id, field=field,
                            error_type=type(exc).__name__)
                return ""

        work_item_name, initiator_name = await asyncio.gather(
            lookup("work_item", item_name), lookup("user", user_name),
        )
        return CocoSessionContext(
            task_id=task.task_id,
            project_key=task.project_key,
            work_item_id=task.work_item_id,
            work_item_name=work_item_name,
            initiator_user_key=task.user_key,
            initiator_name=initiator_name,
        )

    async def _record_coco_attempt(
        self, task_id: str, session_id: str, run_id: str, event_id: str
    ) -> None:
        await self.registry.append_attempt(
            task_id,
            session_id=session_id,
            run_id=run_id,
            event_id=event_id,
        )

    async def _fail_running_attempt(
        self, task_id: str, exc: Exception
    ) -> None:
        task = await self.registry.get(task_id)
        if not task.rounds or not task.rounds[-1].attempts:
            return
        if task.rounds[-1].attempts[-1].status != "running":
            return
        await self.registry.update_last_attempt(
            task_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(UTC),
        )
        log.warning(
            "meegle_coding_attempt_failed", task_id=task_id, round_no=len(task.rounds),
            session_id=task.coco_session_id, run_id=task.coco_run_id, **error_fields(exc),
        )

    @workflow_step("node_completion")
    async def _complete(self, task_id: str) -> None:
        task = await self.registry.get(task_id)
        ref = AINodeRef(
            task.project_key, task.work_item_id, task.node_uuid, task.user_key
        )
        if not task.platform_status_done:
            await self._meegle.edit_status(
                ref, "done", idem_key=f"{task_id}:done"
            )
            await self.registry.update(task_id, platform_status_done=True)
            log.info("meegle_node_status_saved", task_id=task_id, platform_status="done")
        else:
            log.debug("meegle_node_status_reused", task_id=task_id, platform_status="done")
        await self._meegle.confirm(ref, idem_key=f"{task_id}:confirm")
        log.info("meegle_node_confirmed", task_id=task_id)
        await self.registry.update_last_round(
            task_id, status="completed", finished_at=datetime.now(UTC)
        )
        await self.registry.transition(
            task_id, {AeolusState.COMPLETING}, AeolusState.COMPLETED
        )

    async def _mark_failed(self, task_id: str, exc: Exception) -> None:
        task = await self.registry.get(task_id)
        error = f"{type(exc).__name__}: {exc}"
        if task.state in {AeolusState.READING_DOCUMENT, AeolusState.EVALUATING}:
            if task.evaluation_rounds:
                task.evaluation_rounds[-1].update(
                    status="error", error=error,
                    error_kind=_error_kind(exc),
                    finished_at=datetime.now(UTC).isoformat(),
                )
            await self.registry.transition(
                task_id, {task.state}, AeolusState.VALIDATION_FAILED,
                pending_note=_PENDING_FEEDBACK_NOTE,
                evaluation=None, evaluation_rounds=task.evaluation_rounds,
                last_error=error,
            )
            return
        if task.state in {AeolusState.CODING, AeolusState.RECODING} and task.rounds:
            attempts = task.rounds[-1].attempts
            if not attempts or attempts[-1].status == "succeeded":
                # Also retain failures before starting Coco or after accepting its output.
                attempt = attempts[-1] if attempts else None
                await self._record_coco_attempt(
                    task_id, attempt.session_id if attempt else "",
                    attempt.run_id if attempt else "", attempt.event_id if attempt else "",
                )
        await self._fail_running_attempt(task_id, exc)
        if task.rounds and task.state in {AeolusState.CODING, AeolusState.RECODING}:
            await self.registry.update_last_round(
                task_id,
                status="error",
                error=error,
                finished_at=datetime.now(UTC),
            )
        try:
            await self.registry.transition(
                task_id,
                {
                    AeolusState.RECEIVED,
                    AeolusState.READING_DOCUMENT,
                    AeolusState.EVALUATING,
                    AeolusState.CODING,
                    AeolusState.RECODING,
                    AeolusState.COMPLETING,
                },
                AeolusState.FAILED,
                pending_note=_PENDING_FEEDBACK_NOTE,
                last_error=error,
            )
        except InvalidTaskTransition:
            log.warning("aeolus_failure_state_ignored", task_id=task_id)
            current = await self.registry.get(task_id)
            if current.state not in {
                AeolusState.FAILED,
                AeolusState.VALIDATION_FAILED,
                AeolusState.COMPLETED,
            }:
                return

    async def _minimum_delay(self, started: float) -> None:
        remaining = self._minimum_processing_seconds - (time.monotonic() - started)
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def _platform_retry_delay(self) -> None:
        """Wait before replacing a run that failed on the Coco platform itself.

        The runtime reports ``RetryStatus=exhausted`` after its own retries, so
        an immediate replacement would hit the same saturated queue.
        """
        if self._platform_retry_delay_seconds > 0:
            await asyncio.sleep(self._platform_retry_delay_seconds)


def _error_kind(exc: BaseException) -> str:
    """Classify a failure so cards can explain who has to act."""
    if isinstance(exc, CocoRunFailed) and exc.platform:
        return "platform"
    return ""


def _review_contract_excerpt(contract: str) -> str:
    """Keep scope and acceptance, not the full evaluation evidence/discussion."""
    lines = []
    keep_section = False
    acceptance = False
    requirement_heading = ""
    for line in contract.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level, name = len(heading.group(1)), heading.group(2).strip()
            acceptance = False
            if level <= 2:
                requirement_heading = ""
                keep_section = name in {"目标", "本次范围", "范围", "验收标准"} or name.startswith("非目标")
            elif level == 3:
                requirement_heading = line
                acceptance = name == "验收标准"
            if keep_section or acceptance:
                lines.append(line)
            continue
        if re.match(r"^\s*\*\*验收标准\*\*\s*[:：]?", line):
            acceptance = True
            if requirement_heading:
                lines.append(requirement_heading)
            lines.append(line)
            continue
        if line.lstrip().startswith("**"):
            acceptance = False
        if keep_section or acceptance:
            lines.append(line)
    excerpt = "\n".join(lines).strip()
    if len(excerpt) > 3000:
        excerpt = excerpt[:3000].rstrip() + "\n\n（摘录已截断，完整共识请在 Meego 查看。）"
    return excerpt


def _review_title(task: AeolusTask) -> str:
    contract = (task.evaluation or {}).get("requirement_contract") or ""
    heading = re.search(r"^#\s+(.+)$", contract, re.MULTILINE)
    title = re.sub(r"^需求共识\s*[:：]?\s*", "", heading.group(1)).strip() if heading else ""
    if not title:
        first_line = next((line.strip() for line in task.requirement_text.splitlines() if line.strip()), "")
        title = re.sub(r"^#{1,6}\s+", "", first_line)
    title = " ".join(title.split()).strip("# ") or "实现需求变更"
    if len(title) > 80:
        title = title[:79] + "…"
    return f"[Aeolus AI] {title}"


def _branch_name(task_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", task_id).strip("-") or "task"
    if len(safe) > 48:
        digest = hashlib.sha256(task_id.encode()).hexdigest()[:10]
        safe = f"{safe[:37]}-{digest}"
    return f"aeolus/ai-node/{safe}"


def _evaluation_dict(result: EvaluationResult) -> dict:
    return {
        "valid": result.valid,
        "summary": result.summary,
        "missing_items": list(result.missing_items),
        "risks": list(result.risks),
        "requirement_contract": result.requirement_contract,
        "code_context": asdict(result.code_context),
    }


def _task_code_context(task: AeolusTask) -> CodeContext:
    return CodeContext.from_dict((task.evaluation or {}).get("code_context"))


def _contract_with_context(contract: str, context: CodeContext) -> str:
    return contract.rstrip() + (
        "\n\n## 编码上下文（后端核验）\n"
        f"- 仓库：{context.repository}\n"
        f"- Base 分支：{context.base_branch}\n"
        f"- Base commit：{context.base_commit}\n"
        f"- MR 目标分支：{context.target_branch}"
    )


def _coding_requirement(task: AeolusTask) -> str:
    feedbacks = [
        f"- 针对第 {item.get('reviewed_round_no')} 轮评估：{item['feedback']}"
        for item in task.evaluation_rounds if item.get("feedback")
    ]
    text = (
        f"需求文档链接：{task.requirement_url}"
        if task.requirement_url
        else "需求文档链接：未提供"
    )
    contract = (task.evaluation or {}).get("requirement_contract", "")
    if contract:
        text += "\n\n用户已确认的需求共识（按此共识编码）：\n" + contract
    if feedbacks:
        text += "\n\n用户在输入评估阶段补充的信息：\n" + "\n".join(feedbacks)
    return text


def _recovery_prompt(task: AeolusTask) -> str:
    lines = [
        "请恢复并继续当前编码任务。",
        "",
        _coding_requirement(task),
        "",
        f"代码分支：{task.work_branch}",
    ]
    blocked_rounds = [round_ for round_ in task.rounds if round_.status == "blocked"]
    for round_ in blocked_rounds:
        lines.extend(["", f"第 {round_.round_no} 轮阻塞结果：{round_.summary}"])
        lines.extend(f"- {item}" for item in round_.unresolved_items)
    completed_rounds = [round_ for round_ in task.rounds if round_.commit_sha]
    if completed_rounds:
        lines.extend(
            ["", f"当前分支最新提交：{completed_rounds[-1].commit_sha}"]
        )
    feedbacks = [round_ for round_ in task.rounds if round_.feedback]
    if feedbacks:
        lines.extend(["", "历次修改要求："])
        lines.extend(
            f"- 第 {round_.round_no} 轮：{round_.feedback}"
            for round_ in feedbacks
        )
    lines.extend(
        [
            "",
            "请基于当前分支已有代码和以上完整上下文继续处理最后一轮修改要求，",
            "完成后提交并推送代码，不要重复已完成的改动。",
        ]
    )
    return "\n".join(lines)
