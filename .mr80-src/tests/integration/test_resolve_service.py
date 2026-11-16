from unittest.mock import AsyncMock

import pytest

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.integrations.nextcode.mrs import MR, MRVersion
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.models import Base
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import resolve_service
from conciliator.services.resolve_service import ResolveDeps, ResolveService


def _settings() -> Settings:
    return Settings(
        env="dev", service_psm="x",
        nextcode_app_id="x",
        coco_api_base="x", coco_default_agent_id="agent-1",
        lark_app_id="x",
        mysql_psm="x", mysql_db_name="test",
    )  # type: ignore[arg-type]


def _docker_available() -> bool:
    import subprocess
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, check=True, timeout=5,
        )
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon unavailable (testcontainers needs it)",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_sha", ["b" * 40, "d" * 40])
async def test_handle_trigger_creates_pending_task(mysql_container, clean_db, monkeypatch, target_sha):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    mr = MR("id", 42, "title", "", "", "r1", "r1", "feat", "main",
            versions=(MRVersion("v", 1, "a" * 40, "b" * 40, "c" * 40),),
            source_head_sha="a" * 40, target_head_sha=target_sha)
    monkeypatch.setattr(resolve_service, "get_mr", AsyncMock(return_value=mr))
    monkeypatch.setattr(resolve_service, "list_conflict_files", AsyncMock(return_value=[]))

    deps = ResolveDeps(
        nc=None, coco=None, lark=None, repo=repo,  # type: ignore[arg-type]
        notification=None, settings=settings,  # type: ignore[arg-type]
    )
    svc = ResolveService(deps)

    task_id = await svc.handle_trigger(
        repo_id="r1", mr_iid=42, author_email="alice@x.com",
        payload={"source_branch": "feat", "target_branch": "main"},
        trigger_type="COMMENT",
    )
    assert task_id is not None
    task = await repo.get(task_id)
    assert task is not None
    assert task.status == TaskStatus.PENDING
    assert task.mr_author_email is None
    assert task.trigger_author_email == "alice@x.com"
    assert task.source_commit_sha == "a" * 40
    assert task.target_commit_sha == target_sha
    assert task.mr_version == 1

    # 同版本复用已创建的非终态任务。
    second = await svc.handle_trigger(
        repo_id="r1", mr_iid=42, author_email="alice@x.com",
        payload={"source_branch": "feat", "target_branch": "main"},
        trigger_type="COMMENT",
    )
    assert second == task_id
    engine.dispose()
