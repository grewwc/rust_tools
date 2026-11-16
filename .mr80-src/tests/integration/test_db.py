import pytest
from sqlalchemy import select

from conciliator.config import Settings
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.models import (
    Approval,
    Base,
    ConflictBlock,
    ResolveTask,
)


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


# 整个文件的测试都需要 Docker(testcontainers 起 MySQL 容器)
pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon unavailable (testcontainers needs it)",
)


def _settings_for_test() -> Settings:
    """测试环境(env=dev)用 MYSQL_DSN 走普通 mysql。"""
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


def test_create_task_with_blocks_and_approvals(mysql_container, clean_db, monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings_for_test()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    SessionLocal = get_session_factory(engine)

    with SessionLocal() as session:
        task = ResolveTask(
            repo_id="r1",
            source_branch="feat",
            target_branch="main",
            resolve_branch="resolve/feat/x",
            original_mr_id=42,
            trigger_type="COMMENT",
            trigger_payload={"body": "/resolve"},
            status="PENDING",
        )
        block = ConflictBlock(
            conflict_id="C0001",
            block_idx=1,
            file_path="a.py",
            start_line=10,
            end_line=20,
            ours_text="x",
            theirs_text="y",
        )
        task.blocks.append(block)
        session.add(task)
        session.commit()
        session.refresh(task)

        approval = Approval(block_id=task.blocks[0].id, status="PENDING")
        session.add(approval)
        session.commit()

    with SessionLocal() as session:
        fetched = session.execute(select(ConflictBlock).where(ConflictBlock.task_id == task.id)).scalar_one()
        assert fetched.approval.status == "PENDING"
        assert fetched.author_email is None

    engine.dispose()
