import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Integer, create_engine
from sqlalchemy.orm import sessionmaker

from conciliator.domain.task import COCO_POLL_CLAIMS, TaskStatus
from conciliator.integrations.coco.client import CocoTaskResult
from conciliator.repositories.models import Base, ResolveTask
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.resolve_service import ResolveService
from conciliator.workers import task_runner
from conciliator.workers.task_runner import TaskRunner


@pytest.fixture
def storage(tmp_path, monkeypatch):
    for table in Base.metadata.tables.values():
        if "id" in table.c and isinstance(table.c.id.type, Integer):
            monkeypatch.setattr(table.c.id, "type", Integer())
    engine = create_engine(f"sqlite:///{tmp_path / 'claim.db'}")
    Base.metadata.create_all(engine)
    sf = sessionmaker(engine, expire_on_commit=False)
    yield sf, TaskRepository(sf)
    engine.dispose()


def make_task(sf, status, age_seconds=0):
    with sf.begin() as session:
        session.add(ResolveTask(
            id=7, repo_id="repo", source_branch="source", target_branch="main", resolve_branch="resolve/task7",
            original_mr_id=42, trigger_type="COMMENT", trigger_payload={}, active_key="active",
            status=status.value, coco_task_id="session", version=4,
            updated_at=datetime.now(UTC) - timedelta(seconds=age_seconds),
        ))


def service(repo):
    coco = SimpleNamespace(
        fetch_task_result=AsyncMock(return_value=CocoTaskResult(status="running")),
        fetch_retry_task_result=AsyncMock(return_value=CocoTaskResult(status="running")),
    )
    svc = ResolveService(SimpleNamespace(repo=repo, coco=coco))
    svc._reply_original = AsyncMock()
    return svc, coco


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting,processing", list(COCO_POLL_CLAIMS.items()))
async def test_claim_excludes_other_poller_and_releases_running_result(storage, waiting, processing):
    sf, repo = storage
    make_task(sf, waiting)
    first, coco = service(repo)
    second, other_coco = service(repo)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def fetch(*args, **kwargs):
        entered.set()
        await finish.wait()
        return CocoTaskResult(status="running")

    retry = waiting == TaskStatus.RETRY_COCO_RUNNING
    fetcher = coco.fetch_retry_task_result if retry else coco.fetch_task_result
    fetcher.side_effect = fetch
    advance = first.continue_after_retry_coco if retry else first.continue_after_coco
    other_advance = second.continue_after_retry_coco if retry else second.continue_after_coco
    work = asyncio.create_task(advance(7))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        mid = await repo.get(7)
        assert mid.status == processing and mid.version == 5
        await other_advance(7)
        other_coco.fetch_task_result.assert_not_awaited()
        other_coco.fetch_retry_task_result.assert_not_awaited()
    finally:
        finish.set()
        await work
    final = await repo.get(7)
    assert final.status == waiting and final.version == 6
    await other_advance(7)
    assert (other_coco.fetch_retry_task_result if retry else other_coco.fetch_task_result).await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting,processing", list(COCO_POLL_CLAIMS.items()))
async def test_timeout_stays_claimed_then_periodic_scan_fails_without_requeue(
    storage, monkeypatch, waiting, processing,
):
    sf, repo = storage
    make_task(sf, waiting)
    svc, coco = service(repo)
    interrupted = asyncio.Event()

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            interrupted.set()

    retry = waiting == TaskStatus.RETRY_COCO_RUNNING
    (coco.fetch_retry_task_result if retry else coco.fetch_task_result).side_effect = blocked
    runner = TaskRunner(repo=repo, resolve_service=svc, settings=SimpleNamespace(worker_concurrency=1))
    monkeypatch.setattr(task_runner, "_POLL_TASK_TIMEOUT_SECONDS", 0.1)
    advance = svc.continue_after_retry_coco if retry else svc.continue_after_coco
    await runner._poll_task(7, advance)
    assert interrupted.is_set()
    claimed = await repo.get(7)
    assert claimed.status == processing and claimed.version == 5
    await svc.fail_stuck_coco_poll(7)
    assert (await repo.get(7)).status == processing
    with sf.begin() as session:
        session.get(ResolveTask, 7).updated_at = datetime.now(UTC) - timedelta(minutes=20)
    await svc.fail_stuck_coco_poll(7)
    assert (await repo.get(7)).status == processing
    with sf.begin() as session:
        session.get(ResolveTask, 7).updated_at = datetime.now(UTC) - timedelta(seconds=1801)
    # Recovery runs on regular sweeps, without requiring a process restart.
    await runner._poll_coco_once()
    failed = await repo.get(7)
    assert failed.status == TaskStatus.FAILED and failed.active_key is None
    assert "30 分钟" in failed.last_error
    assert "/resolve" in svc._reply_original.await_args.args[1]
    assert not await repo.update_fields(task_id=7, from_version=claimed.version, lark_doc_url="stale")
    await advance(7)
    assert (coco.fetch_retry_task_result if retry else coco.fetch_task_result).await_count == 1


@pytest.mark.asyncio
async def test_cancel_during_poll_is_not_reverted(storage):
    sf, repo = storage
    make_task(sf, TaskStatus.COCO_RUNNING)
    svc, coco = service(repo)

    async def cancel(*args, **kwargs):
        current = await repo.get(7)
        assert (await repo.cancel_task(task_id=7, from_version=current.version,
                                      cancelled_by="user", webhook_event_id=None)).cancelled
        return CocoTaskResult(status="running")

    coco.fetch_task_result.side_effect = cancel
    # Returning the task to RUNNING must lose CAS, preserving the user decision.
    await svc.continue_after_coco(7)
    assert (await repo.get(7)).status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_claim_requires_both_status_and_version(storage):
    sf, repo = storage
    make_task(sf, TaskStatus.COCO_RUNNING)
    assert not await repo.update_status(task_id=7, from_version=4, from_status=TaskStatus.RETRY_COCO_RUNNING,
                                        new_status=TaskStatus.RETRY_COCO_PROCESSING)
    assert await repo.update_status(task_id=7, from_version=4, from_status=TaskStatus.COCO_RUNNING,
                                    new_status=TaskStatus.COCO_PROCESSING)
    assert not await repo.update_status(task_id=7, from_version=4, from_status=TaskStatus.COCO_RUNNING,
                                        new_status=TaskStatus.COCO_PROCESSING)


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting,processing", list(COCO_POLL_CLAIMS.items()))
async def test_timeout_during_validation_repair_does_not_release_claim(storage, monkeypatch, waiting, processing):
    sf, repo = storage
    make_task(sf, waiting)
    svc, coco = service(repo)
    retry = waiting == TaskStatus.RETRY_COCO_RUNNING
    (coco.fetch_retry_task_result if retry else coco.fetch_task_result).return_value = CocoTaskResult(
        status="invalid_result", error="invalid JSON",
    )
    entered = asyncio.Event()

    async def repair(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    svc._repair_or_fail = repair
    monkeypatch.setattr(task_runner, "_POLL_TASK_TIMEOUT_SECONDS", 0.1)
    runner = TaskRunner(repo=repo, resolve_service=svc, settings=SimpleNamespace(worker_concurrency=1))
    await runner._poll_task(7, svc.continue_after_retry_coco if retry else svc.continue_after_coco)
    assert entered.is_set()
    assert (await repo.get(7)).status == processing


@pytest.mark.asyncio
async def test_lost_claim_skips_coco_and_does_not_fail_task():
    task = SimpleNamespace(id=7, status="COCO_RUNNING", version=4, coco_task_id="session")
    repo = SimpleNamespace(get=AsyncMock(return_value=task), update_status=AsyncMock(return_value=False))
    svc, coco = service(repo)
    await svc.continue_after_coco(7)
    repo.update_status.assert_awaited_once_with(
        task_id=7, from_version=4, from_status=TaskStatus.COCO_RUNNING, new_status=TaskStatus.COCO_PROCESSING,
    )
    coco.fetch_task_result.assert_not_awaited()
    svc._reply_original.assert_not_awaited()
    assert task.status == "COCO_RUNNING" and task.version == 4
