import asyncio
from unittest.mock import AsyncMock

import pytest

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.repositories._sync_repo import TaskClaim
from conciliator.workers.task_runner import TaskRunner


class _FakeRepo:
    def __init__(self, task_ids):
        self._task_ids = list(task_ids)
        self.calls = 0

    async def claim_schedulable(self, limit=10):
        self.calls += 1
        if self._task_ids:
            return TaskClaim(self._task_ids.pop(0), TaskStatus.PREPARING)
        return None


class _FakeSvc:
    def __init__(self):
        self.pipelines_run = []

    async def run_pipeline(self, task_id):
        self.pipelines_run.append(task_id)

    async def run_retry_pipeline(self, task_id):
        self.pipelines_run.append(("retry", task_id))


def _settings() -> Settings:
    return Settings(
        env="dev",
        service_psm="x",
        nextcode_app_id="x",
        coco_api_base="x",
        coco_default_agent_id="x",
        lark_app_id="x",
        mysql_psm="x",
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_run_once_picks_and_runs():
    repo = _FakeRepo([7])
    svc = _FakeSvc()
    runner = TaskRunner(repo=repo, resolve_service=svc, settings=_settings())

    await runner.run_once()
    assert svc.pipelines_run == [7]

    # 第二次没有任务
    await runner.run_once()
    assert svc.pipelines_run == [7]


@pytest.mark.asyncio
async def test_concurrent_workers_only_run_the_claimed_task_once():
    repo = _FakeRepo([7])
    svc = _FakeSvc()
    runners = [TaskRunner(repo=repo, resolve_service=svc, settings=_settings()) for _ in range(8)]

    await asyncio.gather(*(runner.run_once() for runner in runners))

    assert svc.pipelines_run == [7]


@pytest.mark.asyncio
async def test_run_once_routes_retry_claim_to_retry_pipeline():
    repo = _FakeRepo([])
    repo.claim_schedulable = AsyncMock(return_value=TaskClaim(9, TaskStatus.RETRY_PREPARING))
    svc = _FakeSvc()
    runner = TaskRunner(repo=repo, resolve_service=svc, settings=_settings())

    await runner.run_once()

    assert svc.pipelines_run == [("retry", 9)]


@pytest.mark.asyncio
async def test_start_stop_lifecycle():
    settings = Settings(
        env="dev",
        service_psm="x",
        nextcode_app_id="x",
        coco_api_base="x",
        coco_default_agent_id="x",
        lark_app_id="x",
        mysql_psm="x",
        worker_concurrency=2,
        worker_poll_interval_seconds=1,
    )  # type: ignore[arg-type]

    repo = _FakeRepo([])
    svc = _FakeSvc()
    runner = TaskRunner(repo=repo, resolve_service=svc, settings=settings)

    await runner.start()
    await asyncio.sleep(0.1)
    await runner.stop()
    # 未抛异常即通过
