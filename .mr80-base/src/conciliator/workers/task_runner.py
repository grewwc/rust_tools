import asyncio
import contextlib
from datetime import UTC
from time import monotonic

from conciliator.config import Settings
from conciliator.domain.task import COCO_POLL_CLAIMS, COCO_PROCESSING_TIMEOUT_SECONDS, TaskStatus
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.resolve_service import ResolveService
from conciliator.utils.logging import get_logger
from conciliator.utils.trace_context import trace_scope

log = get_logger(__name__)

# poller 兜底:COCO_RUNNING 超过这个时间没收到 webhook,主动 fetch
_COCO_POLL_INTERVAL_SECONDS = 30
_COCO_STUCK_THRESHOLD_SECONDS = 60
_POLL_QUERY_TIMEOUT_SECONDS = 15
_POLL_TASK_TIMEOUT_SECONDS = 300

# 启动恢复:中间态任务超过这个时长视为孤儿(进程崩溃残留),标记 FAILED
_STUCK_INTERMEDIATE_TIMEOUT_SECONDS = 600
_STUCK_INTERMEDIATE_STATUSES = [
    TaskStatus.PREPARING,
    TaskStatus.BRANCH_CREATED,
    TaskStatus.CONFLICTS_FETCHED,
    TaskStatus.COCO_SUBMITTED,
    TaskStatus.DOC_CREATED,
    TaskStatus.MR_CREATED,
]
_RETRY_REQUEUE_STATUSES = [
    TaskStatus.RETRY_PREPARING,
]
_RETRY_RESUMABLE_STATUSES = [
    TaskStatus.RETRY_COCO_SUBMITTED,
    TaskStatus.RETRY_RESULT_VERIFIED,
    TaskStatus.RETRY_DOC_UPLOADED,
    TaskStatus.RETRY_PUBLISHING,
]


class TaskRunner:
    def __init__(
        self,
        *,
        repo: TaskRepository,
        resolve_service: ResolveService,
        settings: Settings,
    ):
        self._repo = repo
        self._svc = resolve_service
        self._settings = settings
        self._stop = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._poll_cursors: dict[TaskStatus, tuple] = {}
        self._poll_slots = asyncio.Semaphore(max(1, settings.worker_concurrency))

    async def start(self) -> None:
        log.info(
            "worker_started",
            concurrency=self._settings.worker_concurrency,
            interval=self._settings.worker_poll_interval_seconds,
        )
        self._stop.clear()
        # I5 fix:启动时先恢复孤儿中间态任务(进程崩溃残留)
        try:
            await self._recover_stuck_tasks()
        except Exception:
            log.exception("stuck_recovery_error")
        for i in range(self._settings.worker_concurrency):
            t = asyncio.create_task(self._loop(i))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)
        # SSE consumer/事件游标尚未持久化时的兜底轮询。
        self._tasks.add(asyncio.create_task(self._coco_poller()))

    async def _recover_stuck_tasks(self) -> None:
        """启动时把超过阈值的中间态任务标记 FAILED。"""
        from datetime import datetime, timedelta

        threshold = datetime.now(UTC) - timedelta(
            seconds=_STUCK_INTERMEDIATE_TIMEOUT_SECONDS,
        )
        for status in _STUCK_INTERMEDIATE_STATUSES:
            stuck = await self._repo.list_by_status_updated_before(
                status=status,
                threshold=threshold,
                limit=50,
            )
            for task in stuck:
                log.warning(
                    "recovering_stuck_task",
                    task_id=task.id,
                    status=task.status,
                )
                if status == TaskStatus.MR_CREATED and task.trigger_payload.get("_blank_approval_publication_verified"):
                    try:
                        await self._svc.resume_blank_approval(task.id)
                    except Exception:
                        log.exception("blank_approval_recovery_failed", task_id=task.id)
                    continue
                await self._repo.update_status(
                    task_id=task.id,
                    from_version=task.version,
                    new_status=TaskStatus.FAILED,
                    last_error=f"orphaned in {task.status} (process restart)",
                )
        for status in _RETRY_REQUEUE_STATUSES:
            stuck = await self._repo.list_by_status_updated_before(
                status=status,
                threshold=threshold,
                limit=50,
            )
            for task in stuck:
                log.warning(
                    "recovering_stuck_retry",
                    task_id=task.id,
                    status=task.status,
                )
                await self._repo.update_status(
                    task_id=task.id,
                    from_version=task.version,
                    new_status=TaskStatus.PARTIAL_REJECTED,
                    last_error=f"orphaned before retry run in {task.status}",
                )

    async def stop(self) -> None:
        self._stop.set()
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("worker_stopped")

    async def _loop(self, worker_id: int) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception:
                log.exception("worker_loop_error", worker_id=worker_id)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._settings.worker_poll_interval_seconds,
                )

    async def run_once(self) -> None:
        claim = await self._repo.claim_schedulable()
        if claim is None:
            return
        with trace_scope(claim.task_id):
            started = monotonic()
            log.info("worker_claimed_task", task_id=claim.task_id, status=claim.status.value)
            try:
                if claim.status is TaskStatus.PREPARING:
                    await self._svc.run_pipeline(claim.task_id)
                else:
                    await self._svc.run_retry_pipeline(claim.task_id)
            except Exception:
                log.exception("worker_task_failed", task_id=claim.task_id)
                raise
            else:
                log.info(
                    "worker_task_finished", task_id=claim.task_id,
                    duration_ms=int((monotonic() - started) * 1000),
                )

    async def _coco_poller(self) -> None:
        """兜底：定期回读 COCO_RUNNING Session 的事件和 Run 状态。"""
        while not self._stop.is_set():
            try:
                await self._poll_coco_once()
            except Exception:
                log.exception("coco_poller_error")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=_COCO_POLL_INTERVAL_SECONDS,
                )

    async def _poll_coco_once(self) -> None:
        # TODO: retry publication recovery and blank-approval recovery still have
        # no exclusive status claim. External publication idempotency is separate.
        handlers = [
            *((status, self._svc.fail_stuck_coco_poll) for status in COCO_POLL_CLAIMS.values()),
            (TaskStatus.FAILED, self._svc.notify_validation_failure),
            (TaskStatus.COCO_RUNNING, self._svc.continue_after_coco),
            (TaskStatus.MR_CREATED, self._svc.resume_blank_approval),
            (TaskStatus.RETRY_COCO_RUNNING, self._svc.continue_after_retry_coco),
            *((status, self._svc.resume_retry_publication) for status in _RETRY_RESUMABLE_STATUSES),
        ]
        await asyncio.gather(*(self._poll_status(status, handler) for status, handler in handlers))

    async def _poll_status(self, status, handler) -> None:
        from datetime import datetime, timedelta

        try:
            async with asyncio.timeout(_POLL_QUERY_TIMEOUT_SECONDS):
                batch = await self._repo.list_poll_candidates(
                    status=status,
                    threshold=datetime.now(UTC) - timedelta(seconds=(
                        COCO_PROCESSING_TIMEOUT_SECONDS if status in COCO_POLL_CLAIMS.values()
                        else _COCO_STUCK_THRESHOLD_SECONDS
                    )),
                    cursor=self._poll_cursors.get(status), limit=10,
                )
        except Exception:
            log.exception("task_poll_query_failed", status=status.value)
            return
        # Advance even when a task fails; the next sweep will revisit it.
        if len(batch) == 10:
            task_id, updated_at = batch[-1]
            self._poll_cursors[status] = (updated_at, task_id)
        else:
            self._poll_cursors.pop(status, None)
        await asyncio.gather(*(self._poll_task(task_id, handler) for task_id, _updated_at in batch))

    async def _poll_task(self, task_id, handler) -> None:
        async with self._poll_slots:
            with trace_scope(task_id):
                started = monotonic()
                log.info("task_poll_started", task_id=task_id)
                try:
                    async with asyncio.timeout(_POLL_TASK_TIMEOUT_SECONDS):
                        await handler(task_id)
                except TimeoutError:
                    log.warning("task_poll_timed_out", task_id=task_id, timeout_seconds=_POLL_TASK_TIMEOUT_SECONDS)
                except Exception:
                    log.exception("task_poll_advance_failed", task_id=task_id)
                else:
                    log.info("task_poll_finished", task_id=task_id, duration_ms=int((monotonic() - started) * 1000))
