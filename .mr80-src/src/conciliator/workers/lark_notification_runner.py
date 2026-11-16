import asyncio
import contextlib

from conciliator.config import Settings
from conciliator.services.lark_notification_service import (
    LarkNotificationService,
)
from conciliator.utils.logging import get_logger

log = get_logger(__name__)


class LarkNotificationRunner:
    def __init__(
        self,
        *,
        service: LarkNotificationService,
        settings: Settings,
    ):
        self._service = service
        self._settings = settings
        self._stop = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._stop.clear()
        for worker_id in range(
            self._settings.lark_notification_worker_concurrency
        ):
            task = asyncio.create_task(self._loop(worker_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        log.info(
            "lark_notification_worker_started",
            concurrency=self._settings.lark_notification_worker_concurrency,
        )

    async def stop(self) -> None:
        self._stop.set()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("lark_notification_worker_stopped")

    async def _loop(self, worker_id: int) -> None:
        while not self._stop.is_set():
            try:
                processed = await self._service.process_once()
            except Exception:
                log.exception(
                    "lark_notification_worker_error",
                    worker_id=worker_id,
                )
                processed = False
            if processed:
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=(
                        self._settings.lark_notification_poll_interval_seconds
                    ),
                )
