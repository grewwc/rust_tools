from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conciliator.domain.task import TaskStatus
from conciliator.repositories.models import (
    Approval,
    Base,
    ConflictBlock,
    LarkNotificationOutbox,
    ResolveTask,
)
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import approval_service
from conciliator.services.approval_service import ApprovalService


def _task(*, approved: bool = False):
    block = SimpleNamespace(
        id=11,
        task_id=7,
        block_idx=0,
        conflict_id="C0001",
        file_path="src/example.py",
        start_line=10,
        end_line=20,
        source_author_email="source@example.com",
        target_author_email="target@example.com",
        author_email="source@example.com",
        resolved_path="src/example.py",
        resolved_start_line=12,
        resolved_end_line=16,
        discussion_id="thread-1",
        approval=SimpleNamespace(status="APPROVED" if approved else "PENDING"),
    )
    task = SimpleNamespace(
        id=7,
        repo_id="repo-1",
        original_mr_id=42,
        new_mr_id=99,
        source_branch="test/source",
        target_branch="test/target",
        resolve_branch="resolve/test/source-to-target",
        source_commit_sha="a" * 40,
        target_commit_sha="b" * 40,
        resolve_commit_sha="c" * 40,
        lark_doc_url="https://bytedance.larkoffice.com/file/token",
        coco_result_json={},
        mr_author_email="mr-author@example.com",
        trigger_author_email="trigger@example.com",
        new_mr_url="https://code.test/mr/99",
        status=TaskStatus.AWAITING_APPROVAL.value,
        version=8,
        blocks=[block],
    )
    return task, block


def _approval_task_db(
    approval_statuses: list[str],
    *,
    task_status: TaskStatus,
):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as session:
        session.add(
            ResolveTask(
                id=7,
                repo_id="repo-1",
                source_branch="test/source",
                target_branch="test/target",
                resolve_branch="resolve/test/source-to-target",
                original_mr_id=42,
                active_key="repo-1:42",
                new_mr_id=99,
                trigger_type="COMMENT",
                trigger_payload={},
                status=task_status.value,
                source_commit_sha="a" * 40,
                target_commit_sha="b" * 40,
                resolve_commit_sha="c" * 40,
                coco_result_schema_version="conciliator.resolve.result.v3",
                coco_result_json={},
                lark_doc_url="https://bytedance.larkoffice.com/file/token",
                mr_author_email="mr-author@example.com",
                trigger_author_email="trigger@example.com",
                new_mr_url="https://code.test/mr/99",
                version=0,
            )
        )
        for idx, status in enumerate(approval_statuses):
            block_id = idx + 1
            session.add(
                ConflictBlock(
                    id=block_id,
                    task_id=7,
                    conflict_id=f"C{block_id:04d}",
                    block_idx=idx,
                    file_path=f"src/file{block_id}.py",
                    start_line=10 + idx,
                    end_line=11 + idx,
                    resolved_path=f"src/file{block_id}.py",
                    resolved_start_line=20 + idx,
                    resolved_end_line=21 + idx,
                    source_author_email="source@example.com",
                    target_author_email="target@example.com",
                    discussion_id=f"thread-{block_id}",
                )
            )
            session.add(
                Approval(
                    id=block_id,
                    block_id=block_id,
                    status=status,
                )
            )
        session.commit()
    return engine, sf


@pytest.mark.asyncio
async def test_approval_comment_finds_task_by_resolver_mr(monkeypatch):
    task, block = _task()
    repo = SimpleNamespace(
        find_active_for_resolve_mr=AsyncMock(return_value=task),
        find_block_by_discussion=AsyncMock(return_value=(block, task)),
    )
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=None,
    )
    approve = AsyncMock()
    monkeypatch.setattr(service, "_handle_approve_block", approve)

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/approval",
        author_email="trigger@example.com",
        discussion_id="thread-1",
        webhook_event_id="event-1",
    )

    repo.find_active_for_resolve_mr.assert_awaited_once_with("repo-1", 99)
    approve.assert_awaited_once_with(
        task,
        block,
        "trigger@example.com",
        "thread-1",
        webhook_event_id="event-1",
    )


@pytest.mark.asyncio
async def test_mr_author_without_block_role_cannot_review_block(monkeypatch):
    task, block = _task()
    repo = SimpleNamespace(
        find_active_for_resolve_mr=AsyncMock(return_value=task),
        find_block_by_discussion=AsyncMock(return_value=(block, task)),
    )
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=None,
    )
    approve = AsyncMock()
    reply = AsyncMock()
    monkeypatch.setattr(service, "_handle_approve_block", approve)
    monkeypatch.setattr(approval_service, "reply_discussion", reply)

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/approval",
        author_email="mr-author@example.com",
        discussion_id="thread-1",
        webhook_event_id="event-2",
    )

    approve.assert_not_awaited()
    assert reply.await_args.args[4] == (
        "无权限，仅 block author、`/resolve` 发起人或超级管理员可操作"
    )


@pytest.mark.asyncio
async def test_mr_author_cannot_execute_global_write_command(monkeypatch):
    task, _ = _task()
    repo = SimpleNamespace(find_active_for_resolve_mr=AsyncMock(return_value=task))
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=None,
    )
    cancel = AsyncMock()
    add_comment = AsyncMock()
    monkeypatch.setattr(service, "_handle_cancel", cancel)
    monkeypatch.setattr(approval_service, "add_mr_comment", add_comment)

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/resolve cancel",
        author_email="mr-author@example.com",
        discussion_id=None,
        webhook_event_id="event-3",
    )

    cancel.assert_not_awaited()
    assert add_comment.await_args.args[3] == (
        "无权限，仅 `/resolve` 发起人或超级管理员可取消该任务。"
    )


@pytest.mark.asyncio
async def test_description_sync_renders_and_reads_back_approval_progress(
    monkeypatch,
):
    task, _ = _task(approved=True)
    repo = SimpleNamespace(get_with_blocks=AsyncMock(return_value=task))
    descriptions: list[str] = []

    async def fake_update(client, repo_id, mr_iid, description):
        descriptions.append(description)

    async def fake_get(client, repo_id, mr_iid):
        return SimpleNamespace(description=descriptions[-1])

    monkeypatch.setattr(approval_service, "update_mr_description", fake_update)
    monkeypatch.setattr(approval_service, "get_mr", fake_get)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=None,
    )

    synced = await service._sync_mr_description(task.id)

    assert synced is task
    assert len(descriptions) == 1
    assert "全部确认完成（1/1）" in descriptions[0]
    assert "- [x] #1（C0001）" in descriptions[0]


@pytest.mark.asyncio
async def test_four_individual_approvals_update_progress_then_complete(
    monkeypatch,
):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as session:
        task = ResolveTask(
            id=7,
            repo_id="repo-1",
            source_branch="test/source",
            target_branch="test/target",
            resolve_branch="resolve/test/source-to-target",
            original_mr_id=42,
            active_key="repo-1:42",
            new_mr_id=99,
            trigger_type="COMMENT",
            trigger_payload={},
            status=TaskStatus.AWAITING_APPROVAL.value,
            source_commit_sha="a" * 40,
            target_commit_sha="b" * 40,
            resolve_commit_sha="c" * 40,
            coco_result_schema_version="conciliator.resolve.result.v3",
            coco_result_json={},
            lark_doc_url="https://bytedance.larkoffice.com/file/token",
            mr_author_email="mr-author@example.com",
            trigger_author_email="trigger@example.com",
            new_mr_url="https://code.test/mr/99",
            version=0,
        )
        session.add(task)
        for idx in range(4):
            block_id = idx + 1
            session.add(
                ConflictBlock(
                    id=block_id,
                    task_id=task.id,
                    conflict_id=f"C{idx + 1:04d}",
                    block_idx=idx,
                    file_path=f"src/file{idx // 2 + 1}.py",
                    start_line=10 + idx,
                    end_line=11 + idx,
                    resolved_path=f"src/file{idx // 2 + 1}.py",
                    resolved_start_line=20 + idx,
                    resolved_end_line=21 + idx,
                    source_author_email="source@example.com",
                    target_author_email="target@example.com",
                    discussion_id=f"thread-{idx + 1}",
                )
            )
            session.add(
                Approval(
                    id=block_id,
                    block_id=block_id,
                    status="PENDING",
                )
            )
        session.commit()

    descriptions: list[str] = []
    replies: list[str] = []

    async def fake_update(client, repo_id, mr_iid, description):
        descriptions.append(description)

    async def fake_get(client, repo_id, mr_iid):
        return SimpleNamespace(description=descriptions[-1])

    async def fake_reply(client, repo_id, mr_iid, discussion_id, body):
        replies.append(body)

    monkeypatch.setattr(approval_service, "update_mr_description", fake_update)
    monkeypatch.setattr(approval_service, "get_mr", fake_get)
    monkeypatch.setattr(approval_service, "reply_discussion", fake_reply)
    notification = SimpleNamespace(notify_mr_comment=AsyncMock())
    repo = TaskRepository(sf)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=notification,
        session_factory=sf,
    )

    for idx in range(4):
        await service.handle_comment(
            repo_id="repo-1",
            mr_iid=99,
            body="/approval",
            author_email="trigger@example.com",
            discussion_id=f"thread-{idx + 1}",
            webhook_event_id=f"event-{idx + 1}",
        )
        assert f"（{idx + 1}/4）" in replies[-1]
        assert descriptions[-1].count("- [x]") == idx + 1

    full = await repo.get_with_blocks(7)
    assert full is not None
    assert full.status == TaskStatus.ALL_APPROVED.value
    assert full.active_key is None
    assert all(block.approval and block.approval.status == "APPROVED" for block in full.blocks)
    assert "全部确认完成（4/4）" in descriptions[-1]
    notification.notify_mr_comment.assert_awaited_once()
    with sf() as session:
        queued = session.execute(select(LarkNotificationOutbox)).scalar_one()
        assert queued.task_id == 7
        assert queued.recipient_email == "trigger@example.com"
        assert queued.mr_url == "https://code.test/mr/99"
        assert queued.status == "PENDING"
    assert notification.notify_mr_comment.await_args.kwargs["mention_emails"] == [
        "trigger@example.com"
    ]
    engine.dispose()


@pytest.mark.asyncio
async def test_reject_then_retry_persists_reason_and_queues_subset(
    monkeypatch,
):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as session:
        session.add(
            ResolveTask(
                id=7,
                repo_id="repo-1",
                source_branch="source",
                target_branch="target",
                resolve_branch="resolve/task-7",
                original_mr_id=42,
                active_key="repo-1:42",
                new_mr_id=99,
                trigger_type="COMMENT",
                trigger_payload={},
                status=TaskStatus.AWAITING_APPROVAL.value,
                resolve_commit_sha="a" * 40,
                lark_doc_url="https://lark.test/file/old",
                mr_author_email="author@example.com",
                trigger_author_email="trigger@example.com",
                coco_result_json={},
                version=0,
            )
        )
        for idx in range(2):
            block_id = idx + 1
            session.add(
                ConflictBlock(
                    id=block_id,
                    task_id=7,
                    conflict_id=f"C{block_id:04d}",
                    block_idx=idx,
                    file_path=f"src/file{block_id}.py",
                    start_line=1,
                    end_line=2,
                    discussion_id=f"thread-{block_id}",
                )
            )
            session.add(Approval(id=block_id, block_id=block_id, status="PENDING"))
        session.commit()

    descriptions: list[str] = []

    async def fake_update(client, repo_id, mr_iid, description):
        descriptions.append(description)

    async def fake_get(client, repo_id, mr_iid):
        return SimpleNamespace(description=descriptions[-1])

    monkeypatch.setattr(approval_service, "update_mr_description", fake_update)
    monkeypatch.setattr(approval_service, "get_mr", fake_get)
    monkeypatch.setattr(
        approval_service,
        "reply_discussion",
        AsyncMock(),
    )
    add_comment = AsyncMock()
    monkeypatch.setattr(approval_service, "add_mr_comment", add_comment)
    repo = TaskRepository(sf)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=sf,
    )

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/reject 需要合并两侧条件",
        author_email="trigger@example.com",
        discussion_id="thread-1",
        webhook_event_id="reject-event",
    )
    assert "0 approved / 1 rejected / 1 pending" in descriptions[-1]
    assert "需要合并两侧条件" in descriptions[-1]

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/resolve retry",
        author_email="trigger@example.com",
        discussion_id=None,
        webhook_event_id="retry-event",
    )

    full = await repo.get_with_blocks(7)
    assert full is not None
    assert full.status == TaskStatus.RETRYING.value
    assert full.retry_count == 1
    assert full.blocks[0].approval is not None
    assert full.blocks[0].approval.comment == "需要合并两侧条件"
    assert full.blocks[1].approval is not None
    assert full.blocks[1].approval.status == "PENDING"
    add_comment.assert_awaited_once()
    assert "仅处理 C0001" in add_comment.await_args.args[-1]
    events = await repo.list_events(7)
    assert [event.event_type for event in events] == [
        "BLOCK_REJECTED",
        "RETRY_REQUESTED",
    ]
    engine.dispose()


@pytest.mark.asyncio
async def test_approve_all_refuses_when_any_block_is_rejected(monkeypatch):
    engine, sf = _approval_task_db(
        ["REJECTED", "PENDING"],
        task_status=TaskStatus.PARTIAL_REJECTED,
    )
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    notification = SimpleNamespace(notify_mr_comment=AsyncMock())
    repo = TaskRepository(sf)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=notification,
        session_factory=sf,
    )

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/resolve approve all",
        author_email="trigger@example.com",
        discussion_id="command-thread",
        webhook_event_id="approve-all-rejected",
    )

    full = await repo.get_with_blocks(7)
    assert full is not None
    assert full.status == TaskStatus.PARTIAL_REJECTED.value
    assert [block.approval.status for block in full.blocks] == ["REJECTED", "PENDING"]
    reply.assert_awaited_once()
    assert reply.await_args.args[3] == "command-thread"
    assert "当前有 1 个 REJECTED 冲突块" in reply.await_args.args[4]
    assert "/resolve retry" in reply.await_args.args[4]
    notification.notify_mr_comment.assert_not_awaited()
    with sf() as session:
        assert session.query(LarkNotificationOutbox).count() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_approve_all_atomically_completes_and_queues_notification(monkeypatch):
    engine, sf = _approval_task_db(
        ["APPROVED", "PENDING", "PENDING"],
        task_status=TaskStatus.AWAITING_APPROVAL,
    )
    descriptions: list[str] = []

    async def fake_update(client, repo_id, mr_iid, description):
        descriptions.append(description)

    async def fake_get(client, repo_id, mr_iid):
        return SimpleNamespace(description=descriptions[-1])

    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "update_mr_description", fake_update)
    monkeypatch.setattr(approval_service, "get_mr", fake_get)
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    notification = SimpleNamespace(notify_mr_comment=AsyncMock())
    repo = TaskRepository(sf)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=notification,
        session_factory=sf,
    )

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/resolve approve all",
        author_email="trigger@example.com",
        discussion_id="command-thread",
        webhook_event_id="approve-all-success",
    )

    full = await repo.get_with_blocks(7)
    assert full is not None
    assert full.status == TaskStatus.ALL_APPROVED.value
    assert full.active_key is None
    assert all(block.approval.status == "APPROVED" for block in full.blocks)
    assert "全部确认完成（3/3）" in descriptions[-1]
    reply.assert_awaited_once()
    assert reply.await_args.args[3] == "command-thread"
    assert "已将 2 个 PENDING 冲突块设为 APPROVED" in reply.await_args.args[4]
    notification.notify_mr_comment.assert_awaited_once()
    assert notification.notify_mr_comment.await_args.kwargs["mention_emails"] == [
        "trigger@example.com"
    ]
    events = await repo.list_events(7)
    assert [event.event_type for event in events] == [
        "BLOCK_APPROVED",
        "BLOCK_APPROVED",
        "ALL_APPROVED",
        "LARK_NOTIFICATION_QUEUED",
    ]
    with sf() as session:
        queued = session.execute(select(LarkNotificationOutbox)).scalar_one()
        assert queued.recipient_email == "trigger@example.com"
        assert queued.status == "PENDING"
    engine.dispose()


@pytest.mark.asyncio
async def test_status_replies_in_command_thread_with_all_block_progress(monkeypatch):
    task, first_block = _task()
    first_block.file_path = "src/a|b.py"
    rejected_block = SimpleNamespace(
        id=12,
        task_id=7,
        block_idx=1,
        conflict_id="C0002",
        file_path="src/rejected.py",
        start_line=30,
        end_line=35,
        source_author_email="source-2@example.com",
        target_author_email="target-2@example.com",
        author_email="source-2@example.com",
        approval=SimpleNamespace(status="REJECTED"),
    )
    approved_block = SimpleNamespace(
        id=13,
        task_id=7,
        block_idx=2,
        conflict_id="C0003",
        file_path="src/approved.py",
        start_line=40,
        end_line=41,
        source_author_email=None,
        target_author_email=None,
        author_email="fallback@example.com",
        approval=SimpleNamespace(status="APPROVED"),
    )
    task.status = TaskStatus.PARTIAL_REJECTED.value
    task.blocks = [approved_block, first_block, rejected_block]
    repo = SimpleNamespace(
        find_for_resolve_mr=AsyncMock(return_value=task),
        get_with_blocks=AsyncMock(return_value=task),
    )
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=None,
    )

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/resolve status",
        author_email="anyone@example.com",
        discussion_id="command-thread",
    )

    reply.assert_awaited_once()
    assert reply.await_args.args[3] == "command-thread"
    body = reply.await_args.args[4]
    assert "任务状态：`PARTIAL_REJECTED`" in body
    assert "进度：1/3 APPROVED，1 REJECTED，1 PENDING" in body
    assert "src/a\\|b.py:10-20" in body
    assert body.index("| #1") < body.index("| #2") < body.index("| #3")


@pytest.mark.asyncio
@pytest.mark.parametrize("author_email", ["trigger@example.com", "ADMIN@example.com"])
@pytest.mark.parametrize("mr_iid", [42, 99])
@pytest.mark.parametrize("original_url", [None, "https://code.test/mr/42"])
async def test_cancel_marks_terminal_audits_and_replies_in_command_thread(
    monkeypatch, author_email, mr_iid, original_url,
):
    engine, sf = _approval_task_db(
        ["APPROVED", "PENDING"],
        task_status=TaskStatus.AWAITING_APPROVAL,
    )
    with sf() as session:
        session.get(ResolveTask, 7).original_mr_url = original_url
        session.commit()
    descriptions: list[str] = []

    async def fake_update(client, repo_id, mr_iid, description):
        descriptions.append(description)

    async def fake_get(client, repo_id, mr_iid):
        return SimpleNamespace(description=descriptions[-1])

    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "update_mr_description", fake_update)
    monkeypatch.setattr(approval_service, "get_mr", fake_get)
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    repo = TaskRepository(sf)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=sf,
        super_admin_emails=[" admin@example.com "],
    )

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=mr_iid,
        body="/resolve cancel",
        author_email=author_email,
        discussion_id="command-thread",
        webhook_event_id="cancel-event",
    )

    full = await repo.get_with_blocks(7)
    assert full is not None
    assert full.status == TaskStatus.CANCELLED.value
    assert full.active_key is None
    assert [block.approval.status for block in full.blocks] == ["APPROVED", "PENDING"]
    assert "任务状态**：`CANCELLED`" in descriptions[-1]
    assert "/resolve cancel" in descriptions[-1]
    assert await repo.find_active_for_resolve_mr("repo-1", 99) is None
    reply.assert_awaited_once()
    assert reply.await_args.args[2] == mr_iid
    assert reply.await_args.args[3] == "command-thread"
    body = reply.await_args.args[4]
    assert "任务 #7 已取消" in body and "已解除当前代码版本的任务占用" in body
    assert "再次评论 `/resolve`" in body and "当前分支版本" in body
    assert "已经发出的外部请求可能仍会完成" in body
    assert (f"[原始 MR]({original_url})" if original_url else "原始 MR !42") in body
    events = await repo.list_events(7)
    assert [event.event_type for event in events] == ["TASK_CANCELLED"]
    assert events[0].payload == {
        "from_status": "AWAITING_APPROVAL",
        "cancelled_by": author_email,
        "webhook_event_id": "cancel-event",
        "resolve_branch": "resolve/test/source-to-target",
        "new_mr_id": 99,
    }
    with sf() as session:
        assert session.query(LarkNotificationOutbox).count() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_approval_on_rejected_block_explains_retry_without_changing_state(monkeypatch):
    engine, sf = _approval_task_db(
        ["REJECTED"],
        task_status=TaskStatus.PARTIAL_REJECTED,
    )
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    repo = TaskRepository(sf)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=sf,
    )

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/approval",
        author_email="trigger@example.com",
        discussion_id="thread-1",
        webhook_event_id="approval-on-rejected",
    )

    full = await repo.get_with_blocks(7)
    assert full is not None
    assert full.status == TaskStatus.PARTIAL_REJECTED.value
    assert full.blocks[0].approval.status == "REJECTED"
    reply.assert_awaited_once()
    assert reply.await_args.args[1:] == (
        "repo-1",
        99,
        "thread-1",
        "当前冲突块处于 REJECTED，请先在 MR 主线执行 `/resolve retry`，"
        "重解完成后再回复 `/approval`。",
    )
    assert await repo.list_events(7) == []
    engine.dispose()


@pytest.mark.asyncio
async def test_reject_on_approved_block_explains_terminal_block_state(monkeypatch):
    engine, sf = _approval_task_db(
        ["APPROVED", "PENDING"],
        task_status=TaskStatus.AWAITING_APPROVAL,
    )
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    repo = TaskRepository(sf)
    service = ApprovalService(
        nc=object(),
        repo=repo,
        notification=object(),
        session_factory=sf,
    )

    await service.handle_comment(
        repo_id="repo-1",
        mr_iid=99,
        body="/reject 现在改为目标分支方案",
        author_email="trigger@example.com",
        discussion_id="thread-1",
        webhook_event_id="reject-on-approved",
    )

    full = await repo.get_with_blocks(7)
    assert full is not None
    assert full.status == TaskStatus.AWAITING_APPROVAL.value
    assert [block.approval.status for block in full.blocks] == ["APPROVED", "PENDING"]
    reply.assert_awaited_once()
    assert reply.await_args.args[1:] == (
        "repo-1",
        99,
        "thread-1",
        "当前冲突块处于 APPROVED，不能再回复 `/reject <reason>`。",
    )
    assert await repo.list_events(7) == []
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("body", "handler"), [
    ("/resolve approve all", "_handle_approve_all"),
    ("/resolve retry", "_handle_retry"),
    ("/resolve cancel", "_handle_cancel"),
    ("/approval", "_handle_approve_block"),
    ("/reject please revise", "_handle_reject_block"),
])
@pytest.mark.parametrize("author_email,allowed", [
    ("ADMIN@example.com", True),
    ("outsider@example.com", False),
    ("admin@example.com.evil", False),
    ("", False),
])
async def test_super_admin_can_execute_all_comment_write_commands(
    monkeypatch, body, handler, author_email, allowed,
):
    task, block = _task()
    repo = SimpleNamespace(
        find_active_for_resolve_mr=AsyncMock(return_value=task),
        find_block_by_discussion=AsyncMock(return_value=(block, task)),
    )
    service = ApprovalService(
        nc=object(), repo=repo, notification=object(), session_factory=None,
        super_admin_emails=[" admin@example.com ", ""],
    )
    action = AsyncMock()
    monkeypatch.setattr(service, handler, action)
    monkeypatch.setattr(approval_service, "reply_discussion", AsyncMock())
    monkeypatch.setattr(approval_service, "add_mr_comment", AsyncMock())
    await service.handle_comment(
        repo_id="repo-1", mr_iid=99, body=body,
        author_email=author_email, discussion_id="thread-1",
    )
    assert action.await_count == int(allowed)
    if allowed:
        assert author_email in action.await_args.args


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["discussion_task", "discussion_mr", "task_status"])
async def test_super_admin_still_requires_matching_discussion_and_reviewable_task(monkeypatch, mismatch):
    task, block = _task()
    block_task, _ = _task()
    if mismatch == "discussion_task":
        block_task.id = 8
    elif mismatch == "discussion_mr":
        block_task.new_mr_id = 100
    else:
        task.status = TaskStatus.COCO_RUNNING.value
    repo = SimpleNamespace(
        find_active_for_resolve_mr=AsyncMock(return_value=task),
        find_block_by_discussion=AsyncMock(return_value=(block, block_task)),
    )
    service = ApprovalService(
        nc=object(), repo=repo, notification=object(), session_factory=None,
        super_admin_emails=["admin@example.com"],
    )
    action = AsyncMock()
    monkeypatch.setattr(service, "_handle_approve_block", action)
    monkeypatch.setattr(approval_service, "reply_discussion", AsyncMock())
    await service.handle_comment(
        repo_id="repo-1", mr_iid=99, body="/approval",
        author_email="admin@example.com", discussion_id="thread-1",
    )
    action.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_super_admin_list_grants_no_extra_permissions():
    task, block = _task()
    task.trigger_author_email = None
    service = ApprovalService(nc=object(), repo=object(), notification=object(), session_factory=None)
    assert not await service._can_manage_task(task, "admin@example.com")
    assert not await service._can_manage_task(task, "")
    assert not await service._can_operate_block(block, task, "admin@example.com")


@pytest.mark.asyncio
@pytest.mark.parametrize("email", ["cuijiawei.2026@bytedance.com", "huaguojin@bytedance.com"])
async def test_repo_configured_super_admin_can_manage_another_users_task(monkeypatch, email):
    from conciliator.config import COMMENT_SUPER_ADMIN_EMAILS

    # An environment value must not override the repository-controlled list.
    monkeypatch.setenv("COMMENT_SUPER_ADMIN_EMAILS", '["outsider@example.com"]')
    task, block = _task()
    service = ApprovalService(
        nc=object(), repo=object(), notification=object(), session_factory=None,
        super_admin_emails=COMMENT_SUPER_ADMIN_EMAILS,
    )
    assert task.trigger_author_email != email
    assert await service._can_manage_task(task, email)
    assert await service._can_operate_block(block, task, email)
    assert not await service._can_manage_task(task, "outsider@example.com")


@pytest.mark.asyncio
async def test_cancel_conflict_explains_status_and_retry_commands(monkeypatch):
    task, _ = _task()
    repo = SimpleNamespace(cancel_task=AsyncMock(return_value=SimpleNamespace(cancelled=False)))
    service = ApprovalService(nc=object(), repo=repo, notification=None, session_factory=None)
    reply = AsyncMock()
    monkeypatch.setattr(service, "_reply_command", reply)
    await service._handle_cancel(
        task, "trigger@example.com", discussion_id="command-thread", webhook_event_id=None, reply_mr_iid=42,
    )
    body = reply.await_args.args[2]
    assert "本次取消未成功" in body
    assert "`/resolve status`" in body and "`/resolve cancel 7`" in body
    assert reply.await_args.kwargs["reply_mr_iid"] == 42
    assert task.status == TaskStatus.AWAITING_APPROVAL.value
