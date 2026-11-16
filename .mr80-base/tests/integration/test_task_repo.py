import asyncio

import pytest

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.repositories._sync_repo import ActiveTaskExistsError
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.models import Base
from conciliator.repositories.task_repo import TaskRepository


def _settings() -> Settings:
    return Settings(
        env="dev",
        service_psm="x",
        nextcode_app_id="x",
        coco_api_base="x",
        coco_default_agent_id="x",
        lark_app_id="x",
        mysql_psm="x",
        mysql_db_name="test",
    )  # type: ignore[arg-type]


def _docker_available() -> bool:
    import subprocess

    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon unavailable (testcontainers needs it)",
)


@pytest.mark.asyncio
async def test_create_and_get(mysql_container, clean_db, monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={"body": "/resolve"},
    )
    fetched = await repo.get(task.id)
    assert fetched is not None
    assert fetched.status == TaskStatus.PENDING
    engine.dispose()


@pytest.mark.asyncio
async def test_optimistic_lock(mysql_container, clean_db, monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
    )
    ok1 = await repo.update_status(
        task_id=task.id,
        from_version=0,
        new_status=TaskStatus.PREPARING,
    )
    assert ok1 is True
    ok2 = await repo.update_status(
        task_id=task.id,
        from_version=0,
        new_status=TaskStatus.BRANCH_CREATED,
    )
    assert ok2 is False
    engine.dispose()


@pytest.mark.asyncio
async def test_claim_pending_is_atomic(mysql_container, clean_db, monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
    )

    claims = await asyncio.gather(*(repo.claim_pending(limit=1) for _ in range(8)))

    assert claims.count(task.id) == 1
    assert claims.count(None) == 7
    claimed = await repo.get(task.id)
    assert claimed is not None
    assert claimed.status == TaskStatus.PREPARING.value
    assert claimed.version == 1
    engine.dispose()


@pytest.mark.asyncio
async def test_find_active_for_mr(mysql_container, clean_db, monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    assert await repo.find_active_for_mr("r1", 42) is None
    task = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
    )
    found = await repo.find_active_for_mr("r1", 42)
    assert found is not None
    assert found.id == task.id
    engine.dispose()


@pytest.mark.asyncio
async def test_active_task_key_is_atomic_and_released_on_terminal_status(
    mysql_container,
    clean_db,
    monkeypatch,
):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
    )
    with pytest.raises(ActiveTaskExistsError):
        await repo.create(
            repo_id="r1",
            source_branch="feat",
            target_branch="main",
            resolve_branch="resolve/feat/y",
            original_mr_id=42,
            trigger_type="COMMENT",
            trigger_payload={},
        )

    assert await repo.update_status(
        task_id=task.id,
        from_version=0,
        new_status=TaskStatus.FAILED,
    )
    replacement = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/y",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
    )
    assert replacement.id != task.id
    engine.dispose()


@pytest.mark.asyncio
async def test_find_active_for_resolve_mr_uses_new_mr_id(
    mysql_container,
    clean_db,
    monkeypatch,
):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
    )
    assert await repo.update_status(
        task_id=task.id,
        from_version=0,
        new_status=TaskStatus.AWAITING_APPROVAL,
        new_mr_id=99,
    )

    assert await repo.find_active_for_resolve_mr("r1", 42) is None
    found = await repo.find_active_for_resolve_mr("r1", 99)
    assert found is not None
    assert found.id == task.id
    engine.dispose()


@pytest.mark.asyncio
async def test_add_event_dedup_by_coco_event_id(mysql_container, clean_db, monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
    )
    first = await repo.add_event(
        task_id=task.id,
        event_type="coco.started",
        payload={"x": 1},
        coco_event_id="evt-1",
    )
    second = await repo.add_event(
        task_id=task.id,
        event_type="coco.started",
        payload={"x": 1},
        coco_event_id="evt-1",
    )
    assert first is True
    assert second is False
    engine.dispose()


@pytest.mark.asyncio
async def test_nextcode_webhook_delivery_dedup_and_failed_retry(mysql_container, clean_db, monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    claim = {
        "delivery_key": "delivery:delivery-1",
        "event_id": "event-1",
        "delivery_id": "delivery-1",
        "event_type": "comment",
    }
    assert await repo.claim_nextcode_webhook(**claim) is True
    assert await repo.claim_nextcode_webhook(**claim) is False

    await repo.fail_nextcode_webhook(delivery_key=claim["delivery_key"], error="temporary failure")
    assert await repo.claim_nextcode_webhook(**claim) is True

    await repo.complete_nextcode_webhook(delivery_key=claim["delivery_key"], processed_count=1)
    assert await repo.claim_nextcode_webhook(**claim) is False
    engine.dispose()


@pytest.mark.asyncio
async def test_processing_version_key_allows_parallel_heads_and_deduplicates_races(
    mysql_container, clean_db, monkeypatch,
):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    engine = create_engine_from_settings(_settings())
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))
    kwargs = dict(
        repo_id="r1", original_mr_id=42, source_branch="source", target_branch="target",
        resolve_branch="", trigger_type="COMMENT", trigger_payload={}, mr_version=1,
        source_commit_sha="a" * 40, target_commit_sha="b" * 40, conflict_files_snapshot=[],
    )
    try:
        results = await asyncio.gather(*(repo.create(**kwargs) for _ in range(8)), return_exceptions=True)
        assert sum(isinstance(result, ActiveTaskExistsError) for result in results) == 7
        created = [result for result in results if not isinstance(result, BaseException)]
        assert len(created) == 1
        original = created[0]
        source_changed = await repo.create(**(kwargs | {"source_commit_sha": "c" * 40}))
        target_changed = await repo.create(**(kwargs | {"target_commit_sha": "d" * 40}))
        assert len(await repo.list_active_for_mr("r1", 42)) == 3
        assert (await repo.find_active_for_version("r1", 42, "a" * 40, "b" * 40)).id == original.id
        assert await repo.update_status(task_id=original.id, from_version=0, new_status=TaskStatus.FAILED)
        replacement = await repo.create(**kwargs)
        assert replacement.id != original.id
        assert (await repo.get(source_changed.id)).status == "PENDING"
        assert (await repo.get(target_changed.id)).status == "PENDING"
    finally:
        engine.dispose()
