from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from conciliator.api.comment_handlers import (
    ApprovalCommentHandler,
    ResolveCommandHandler,
)
from conciliator.api.comment_router import CommentDispatcher, CommentRouter
from conciliator.api.webhooks import _handle_comment_event
from conciliator.domain.task import TERMINAL_STATUSES, TaskStatus
from conciliator.repositories.models import ResolveTask
from conciliator.services import approval_service
from conciliator.services.approval_service import ApprovalService
from conciliator.services.task_status import TASK_STATUS_LABELS, original_mr_status_reply
from tests.unit.test_approval_service import _approval_task_db
from tests.unit.test_nextcode_webhook_contract import FakeClient, _mr
from tests.unit.test_processing_version import create
from tests.unit.test_processing_version import repository as repository


def service(repo, nc=None):
    return ApprovalService(nc=nc, repo=repo, notification=None, session_factory=None)


async def query(svc, mr_iid=42, thread="query-thread"):
    await svc.handle_comment(
        repo_id="r1", mr_iid=mr_iid, body="/resolve status",
        author_email="unrelated-user@example.com", discussion_id=thread,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.PENDING, TaskStatus.COCO_RUNNING])
async def test_original_mr_status_before_resolver_mr_exists(repository, monkeypatch, status):
    repo, sf = repository
    task = await create(repo)
    stamp = datetime(2026, 9, 16, 6, 30)
    with sf() as session:
        row = session.get(ResolveTask, task.id)
        row.status = status.value
        row.updated_at = stamp
        row.trigger_payload = {"_trigger_discussion_id": "old-trigger-thread"}
        session.commit()
    before = await repo.get(task.id)
    before_events = [event.id for event in await repo.list_events(task.id)]
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    await query(service(repo))
    args = reply.await_args.args
    assert args[1:4] == ("r1", 42, "query-thread")
    body = args[4]
    assert (
        f"[任务 {task.id}](https://aeolus-conciliator.byted.org/trace?task_id={task.id}&order=desc)"
        f"：{TASK_STATUS_LABELS[status]}（`{status.value}`）"
    ) in body
    assert "source `aaaaaaaa` → target `bbbbbbbb`" in body
    assert "2026-09-16 14:30:00（北京时间）" in body
    assert "| Block" not in body and "最近一次" not in body
    after = await repo.get(task.id)
    assert (after.status, after.version, after.updated_at) == (before.status, before.version, before.updated_at)
    assert [event.id for event in await repo.list_events(task.id)] == before_events


@pytest.mark.asyncio
async def test_original_mr_lists_all_active_versions_not_newer_finished_task(repository, monkeypatch):
    repo, sf = repository
    older = await create(repo)
    newer = await create(repo, source="c" * 40)
    finished = await create(repo, source="d" * 40)
    other_mr = await create(repo, mr_iid=43)
    with sf() as session:
        session.get(ResolveTask, older.id).new_mr_id = 99
        session.get(ResolveTask, older.id).new_mr_url = "https://code.test/mr/99"
        session.get(ResolveTask, finished.id).status = "FAILED"
        # Same MR number in another repository must not leak into the response.
        session.get(ResolveTask, other_mr.id).repo_id = "other-repo"
        session.get(ResolveTask, other_mr.id).original_mr_id = 42
        session.commit()
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    await query(service(repo))
    body = reply.await_args.args[4]
    assert "2 个活动任务" in body
    for task in (newer, older):
        assert f"[任务 {task.id}](https://aeolus-conciliator.byted.org/trace?task_id={task.id}&order=desc)" in body
    assert body.index(f"[任务 {newer.id}]") < body.index(f"[任务 {older.id}]")
    assert f"[任务 {finished.id}]" not in body and f"[任务 {other_mr.id}]" not in body
    assert "source `cccccccc`" in body
    assert "[解决冲突的 MR](<https://code.test/mr/99>)" in body
    assert reply.await_args.args[2] == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("status", list(TERMINAL_STATUSES))
async def test_original_mr_falls_back_to_latest_terminal_task(repository, monkeypatch, status):
    repo, sf = repository
    older = await create(repo)
    newer = await create(repo, source="c" * 40)
    with sf() as session:
        session.get(ResolveTask, older.id).status = "FAILED"
        session.get(ResolveTask, newer.id).status = status.value
        session.commit()
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    await query(service(repo))
    body = reply.await_args.args[4]
    assert "当前没有进行中的任务。" in body
    assert (
        f"[最近一次任务 {newer.id}](https://aeolus-conciliator.byted.org/trace?task_id={newer.id}&order=desc)"
        f"：{TASK_STATUS_LABELS[status]}"
    ) in body
    assert f"[任务 {older.id}]" not in body
    assert "已合并" not in body


@pytest.mark.asyncio
async def test_original_mr_without_tasks_replies_in_query_thread(repository, monkeypatch):
    repo, _ = repository
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    await query(service(repo))
    reply.assert_awaited_once_with(None, "r1", 42, "query-thread", "该 MR 暂无解冲突任务，可评论 `/resolve` 发起。")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.AWAITING_APPROVAL, *TERMINAL_STATUSES])
async def test_resolver_status_includes_final_approval_records(monkeypatch, status):
    from conciliator.repositories.task_repo import TaskRepository

    engine, sf = _approval_task_db(["APPROVED", "REJECTED", "PENDING"], task_status=status)
    repo = TaskRepository(sf)
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    try:
        await service(repo).handle_comment(
            repo_id="repo-1", mr_iid=99, body="/resolve status",
            author_email="unrelated-user@example.com", discussion_id="new-query-thread",
        )
        assert reply.await_args.args[1:4] == ("repo-1", 99, "new-query-thread")
        body = reply.await_args.args[4]
        assert f"任务状态：`{status.value}`" in body
        assert "进度：1/3 APPROVED，1 REJECTED，1 PENDING" in body
        assert "| Block | File | Source author | Target author | Status |" in body
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_resolver_lookup_scopes_repository_and_includes_terminal(repository):
    repo, sf = repository
    task = await create(repo)
    with sf() as session:
        row = session.get(ResolveTask, task.id)
        row.status = "CANCELLED"
        row.new_mr_id = 99
        session.commit()
    assert (await repo.find_for_resolve_mr("r1", 99)).id == task.id
    assert await repo.find_for_resolve_mr("other-repo", 99) is None
    assert await repo.find_for_resolve_mr("r1", 42) is None
    assert await repo.find_active_for_resolve_mr("r1", 99) is None


@pytest.mark.asyncio
async def test_status_without_thread_never_creates_top_level_comment(monkeypatch):
    repo = SimpleNamespace(find_for_resolve_mr=AsyncMock())
    reply, top_level = AsyncMock(), AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    monkeypatch.setattr(approval_service, "add_mr_comment", top_level)
    with pytest.raises(ValueError, match="requires ThreadId"):
        await query(service(repo), thread=None)
    repo.find_for_resolve_mr.assert_not_awaited()
    reply.assert_not_awaited()
    top_level.assert_not_awaited()


@pytest.mark.asyncio
async def test_db_error_is_not_reported_as_no_task(monkeypatch):
    repo = SimpleNamespace(find_for_resolve_mr=AsyncMock(side_effect=RuntimeError("database unavailable")))
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    with pytest.raises(RuntimeError, match="database unavailable"):
        await query(service(repo))
    reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_error_does_not_change_task(repository, monkeypatch):
    repo, _ = repository
    task = await create(repo)
    before = await repo.get(task.id)
    before_events = [event.id for event in await repo.list_events(task.id)]
    monkeypatch.setattr(approval_service, "reply_discussion", AsyncMock(side_effect=RuntimeError("send failed")))
    with pytest.raises(RuntimeError, match="send failed"):
        await query(service(repo))
    after = await repo.get(task.id)
    assert (after.status, after.version, after.updated_at) == (before.status, before.version, before.updated_at)
    assert [event.id for event in await repo.list_events(task.id)] == before_events


def test_status_labels_cover_all_states_and_unknown_states_are_preserved():
    assert set(TASK_STATUS_LABELS) == set(TaskStatus)
    task = SimpleNamespace(
        id=123, status="FUTURE_STATE", source_commit_sha=None, target_commit_sha=None,
        updated_at=datetime(2026, 9, 16, 6, 30, tzinfo=UTC), new_mr_url=None,
    )
    text = original_mr_status_reply([task])
    assert "`FUTURE_STATE`" in text and "未记录" in text
    assert "2026-09-16 14:30:00（北京时间）" in text


def test_awaiting_approval_task_links_trace_and_resolver_mr_separately():
    task = SimpleNamespace(
        id=49, status="AWAITING_APPROVAL", source_commit_sha="a6984afe", target_commit_sha="45c29c65",
        updated_at=datetime(2026, 11, 2, 2, 22, 46, tzinfo=UTC),
        new_mr_url="https://code.test/mr/55",
    )
    text = original_mr_status_reply([task])
    assert (
        "[任务 49](https://aeolus-conciliator.byted.org/trace?task_id=49&order=desc)"
        "：等待冲突块审批（`AWAITING_APPROVAL`）"
    ) in text
    assert "[解决冲突的 MR](<https://code.test/mr/55>)" in text
    assert "2026-11-02 10:22:46（北京时间）" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("source_branch", ["feature/business", "fix/business", "fix/" + "long-name-" * 30])
@pytest.mark.parametrize("resolver", [False, True])
async def test_webhook_to_create_comment_keeps_query_mr_and_thread(repository, source_branch, resolver):
    repo, sf = repository
    task = await create(repo)
    mr_iid = 99 if resolver else 42
    with sf() as session:
        row = session.get(ResolveTask, task.id)
        row.new_mr_id = 99
        row.resolve_branch = "fix/generated-conflict-20260916-t1"
        row.trigger_payload = {"_trigger_discussion_id": "old-trigger-thread"}
        session.commit()
    mr = {**_mr(), "Number": mr_iid, "SourceBranchName": source_branch}
    client = FakeClient({
        "GetMergeRequest": {"MergeRequest": mr},
        "GetThread": {"Thread": {"Id": "new-query-thread", "Status": "open", "Positions": []}},
        "CreateComment": {"Comment": {"Id": "reply-id", "ThreadId": "new-query-thread"}},
    })
    state = SimpleNamespace(
        settings=SimpleNamespace(nextcode_app_id="app-1"),
        nextcode_client=client,
        comment_router=CommentRouter(
            resolve_service=object(), approval_service=service(repo, client), task_repo=repo,
        ),
    )
    state.comment_dispatcher = CommentDispatcher(
        handlers=[
            ResolveCommandHandler(state.comment_router),
            ApprovalCommentHandler(state.comment_router),
        ]
    )
    payload = {
        "Action": "created", "Repository": {"Id": "r1"},
        "Comment": {
            "Id": "query-comment", "CommentableType": "merge_request", "CommentableId": "mr-id",
            "RepoId": "r1", "Content": "/resolve status", "ThreadId": "new-query-thread",
            "CreatedBy": {"Email": "unrelated-user@example.com"},
        },
    }
    assert await _handle_comment_event(state, payload) == 1
    writes = [(action, body) for action, body, read_only in client.calls if not read_only]
    assert len(writes) == 1
    action, body = writes[0]
    assert action == "CreateComment"
    assert body["RepoId"] == "r1" and body["CommentableId"] == "mr-id"
    assert body["ThreadId"] == "new-query-thread"
    if resolver:
        assert "| Block |" in body["Content"]
    else:
        assert (
            f"[任务 {task.id}](https://aeolus-conciliator.byted.org/trace?task_id={task.id}&order=desc)"
            "：等待处理"
        ) in body["Content"]
        assert "| Block |" not in body["Content"]
    # Reply readback addresses the MR where the user just commented.
    mr_reads = [body for action, body, _ in client.calls if action == "GetMergeRequest"]
    assert mr_reads[-1]["Number"] == mr_iid
