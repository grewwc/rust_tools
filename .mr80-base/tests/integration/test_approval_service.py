from types import SimpleNamespace

import pytest

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.models import (
    Base,
    ConflictBlock,
    LarkNotificationOutbox,
)
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.approval_service import ApprovalService


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
async def test_approve_block_flips_status(mysql_container, clean_db, monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", mysql_container.dsn)
    settings = _settings()
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    sf = get_session_factory(engine)
    repo = TaskRepository(sf)

    task = await repo.create(
        repo_id="r1",
        source_branch="feat",
        target_branch="main",
        resolve_branch="resolve/feat/x",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
        mr_author_email="mr-author@x.com",
        trigger_author_email="initiator@x.com",
    )
    await repo.update_status(
        task_id=task.id,
        from_version=0,
        new_status=TaskStatus.AWAITING_APPROVAL,
        new_mr_id=99,
        new_mr_url="https://code.test/mr/99",
    )
    block_ids = await repo.save_conflict_blocks(
        task.id,
        [
            {
                "conflict_id": "C0001",
                "block_idx": 0,
                "file_path": "a.py",
                "start_line": 10,
                "end_line": 20,
                "ours_text": "x",
                "theirs_text": "y",
                "author_email": "alice@x.com",
            }
        ],
    )
    with sf() as session:
        from sqlalchemy import update as sa_update

        session.execute(sa_update(ConflictBlock).where(ConflictBlock.id == block_ids[0]).values(discussion_id="disc-1"))
        session.commit()

    sent = []
    # patch service 模块里已 import 的引用(from ... import reply_discussion)
    from conciliator.services import approval_service as approval_mod

    async def fake_reply(client, pid, mr_iid, did, body):
        sent.append(("reply", did, body))

    async def fake_comment(client, pid, mr_iid, body):
        sent.append(("comment", body))

    monkeypatch.setattr(approval_mod, "reply_discussion", fake_reply)
    monkeypatch.setattr(approval_mod, "add_mr_comment", fake_comment)
    descriptions = []

    async def fake_update_description(client, pid, mr_iid, description):
        descriptions.append(description)

    async def fake_get_mr(client, pid, mr_iid):
        return SimpleNamespace(description=descriptions[-1])

    monkeypatch.setattr(
        approval_mod,
        "update_mr_description",
        fake_update_description,
    )
    monkeypatch.setattr(approval_mod, "get_mr", fake_get_mr)

    class _NoopNotif:
        async def notify_mr_comment(self, **kwargs):
            sent.append(("notif", kwargs.get("body")))

    svc = ApprovalService(
        nc=object(),
        repo=repo,
        notification=_NoopNotif(),
        session_factory=sf,
    )
    await svc.handle_comment(
        repo_id="r1",
        mr_iid=99,
        body="/approval",
        author_email="initiator@x.com",
        discussion_id="disc-1",
        webhook_event_id="event-approval-1",
    )

    full = await repo.get_with_blocks(task.id)
    assert full.blocks[0].approval.status == "APPROVED"
    assert full.status == TaskStatus.ALL_APPROVED.value
    assert full.active_key is None
    assert "全部确认完成（1/1）" in descriptions[-1]
    assert "- [x] #1（C0001）" in descriptions[-1]
    assert any("（1/1）" in s[2] for s in sent if s[0] == "reply")
    assert any(s[0] == "reply" for s in sent)
    with sf() as session:
        queued = session.query(LarkNotificationOutbox).filter_by(
            task_id=task.id, notification_type="ALL_APPROVED",
        ).one()
        assert queued.recipient_email == "initiator@x.com"
        assert queued.status == "PENDING"
    engine.dispose()
