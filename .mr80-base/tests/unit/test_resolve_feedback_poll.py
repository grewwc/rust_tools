import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from conciliator.api import webhooks
from conciliator.domain.task import TaskStatus
from conciliator.services import notification_service, resolve_service
from conciliator.services.resolve_service import ResolveService
from conciliator.workers import task_runner
from conciliator.workers.task_runner import TaskRunner


def service_mocks():
    return SimpleNamespace(**{name: AsyncMock() for name in (
        "notify_validation_failure", "fail_stuck_coco_poll", "continue_after_coco", "resume_blank_approval",
        "continue_after_retry_coco", "resume_retry_publication",
    )})


@pytest.mark.asyncio
async def test_poll_visits_tail_and_wraps_without_touching_business_timestamp():
    updated_at = datetime(2020, 1, 1, tzinfo=UTC)
    rows = [(i, updated_at) for i in range(1, 26)]

    async def candidates(*, status, cursor, **kwargs):
        if status != TaskStatus.COCO_RUNNING:
            return []
        return [row for row in rows if cursor is None or (row[1], row[0]) > cursor][:10]

    repo = SimpleNamespace(list_poll_candidates=AsyncMock(side_effect=candidates))
    svc = service_mocks()
    runner = TaskRunner(repo=repo, resolve_service=svc, settings=SimpleNamespace(worker_concurrency=2))
    for _ in range(4):
        await runner._poll_coco_once()
    assert [call.args[0] for call in svc.continue_after_coco.await_args_list] == [*range(1, 26), *range(1, 11)]
    assert rows == [(i, updated_at) for i in range(1, 26)]


@pytest.mark.asyncio
@pytest.mark.parametrize("query_failure", ["error", "timeout"])
async def test_poll_query_and_task_failure_do_not_block_other_tasks(monkeypatch, query_failure):
    monkeypatch.setattr(task_runner, "_POLL_QUERY_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(task_runner, "_POLL_TASK_TIMEOUT_SECONDS", 0.02)

    async def candidates(*, status, **kwargs):
        if status == TaskStatus.FAILED:
            if query_failure == "error":
                raise RuntimeError("database query failed")
            await asyncio.Event().wait()
        if status == TaskStatus.COCO_RUNNING:
            return [(1, datetime.now(UTC)), (2, datetime.now(UTC)), (3, datetime.now(UTC))]
        if status == TaskStatus.RETRY_COCO_RUNNING:
            return [(4, datetime.now(UTC))]
        return []

    completed = []

    async def advance(task_id):
        if task_id == 1:
            await asyncio.Event().wait()
        if task_id == 2:
            raise RuntimeError("one task failed")
        completed.append(task_id)

    svc = service_mocks()
    svc.continue_after_coco.side_effect = advance
    svc.continue_after_retry_coco.side_effect = advance
    runner = TaskRunner(
        repo=SimpleNamespace(list_poll_candidates=AsyncMock(side_effect=candidates)),
        resolve_service=svc, settings=SimpleNamespace(worker_concurrency=2),
    )
    await asyncio.wait_for(runner._poll_coco_once(), timeout=1)
    assert sorted(completed) == [3, 4]


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
async def test_running_coco_only_loads_metadata(retry):
    task = SimpleNamespace(
        id=1, status=(TaskStatus.RETRY_COCO_RUNNING if retry else TaskStatus.COCO_RUNNING).value,
        coco_task_id="session", trigger_payload={}, version=0,
    )
    repo = SimpleNamespace(get=AsyncMock(return_value=task), get_with_blocks=AsyncMock(),
                           update_status=AsyncMock(return_value=True))
    coco = SimpleNamespace(
        fetch_task_result=AsyncMock(return_value=SimpleNamespace(status="running")),
        fetch_retry_task_result=AsyncMock(return_value=SimpleNamespace(status="running")),
    )
    svc = ResolveService(SimpleNamespace(repo=repo, coco=coco))
    await (svc.continue_after_retry_coco(1) if retry else svc.continue_after_coco(1))
    repo.get.assert_awaited_once_with(1, include_bodies=False)
    repo.get_with_blocks.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_reply", [False, True])
async def test_fast_reply_precedes_mr_lookup_and_failure_does_not_stop_routing(monkeypatch, fail_reply):
    calls = []

    async def api(action, body):
        calls.append(action)
        assert body["CommentableId"] == "mr-internal"
        assert body["ThreadId"] == "thread"
        if fail_reply:
            raise RuntimeError("comment unavailable")
        return {}

    async def parse(*args):
        assert calls == ["CreateComment"]
        calls.append("GetMR")
        return object()

    route = AsyncMock()
    monkeypatch.setattr(webhooks, "_parse_comment", parse)
    monkeypatch.setattr(webhooks, "_route_comment", route)
    state = SimpleNamespace(
        settings=SimpleNamespace(nextcode_app_id="bot"), pipeline_enabled=True,
        nextcode_client=SimpleNamespace(call=api),
    )
    await webhooks._handle_comment_event(state, {"Action": "created", "Comment": {
        "Id": "comment", "CommentableType": "merge_request", "CommentableId": "mr-internal",
        "RepoId": "repo", "ThreadId": "thread", "Content": "/resolve",
    }})
    route.assert_awaited_once()
    assert calls == ["CreateComment", "GetMR"]


@pytest.mark.asyncio
async def test_feedback_timeout_is_best_effort(monkeypatch):
    monkeypatch.setattr(notification_service, "_FEEDBACK_TIMEOUT_SECONDS", 0.01)

    async def blocked(*args):
        await asyncio.Event().wait()

    await asyncio.wait_for(notification_service.send_resolve_feedback(
        SimpleNamespace(call=blocked), repo_id="repo", mr_id="mr", thread_id="thread", body="received",
    ), timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("won", [False, True])
async def test_failure_feedback_only_follows_successful_status_write(monkeypatch, won):
    feedback = AsyncMock()
    monkeypatch.setattr(resolve_service, "send_resolve_feedback", feedback)
    task = SimpleNamespace(
        id=7, version=0, status="PREPARING", repo_id="repo", original_mr_id=42,
        trigger_payload={"_trigger_discussion_id": "thread", "_trigger_mr_id": "mr-internal"},
    )
    repo = SimpleNamespace(update_status=AsyncMock(return_value=won))
    svc = ResolveService(SimpleNamespace(repo=repo, nc=object()))
    await svc._fail_task(task, "prepare failed")
    if won:
        body = feedback.await_args.kwargs["body"]
        assert "#7" in body and "PREPARING" in body and "prepare failed" in body
        assert feedback.await_args.kwargs["mr_id"] == "mr-internal"
    else:
        feedback.assert_not_awaited()


@pytest.mark.asyncio
async def test_result_feedback_links_original_thread_to_mr_and_report(monkeypatch):
    feedback = AsyncMock()
    monkeypatch.setattr(resolve_service, "send_resolve_feedback", feedback)
    task = SimpleNamespace(
        id=7, repo_id="repo", original_mr_id=42, new_mr_url="https://code.test/mr/43",
        lark_doc_url="https://docs.test/report",
        trigger_payload={"_trigger_discussion_id": "original-thread", "_trigger_mr_id": "original-mr"},
    )
    svc = ResolveService(SimpleNamespace(nc=object()))
    await svc._reply_result(task)
    args = feedback.await_args.kwargs
    assert args["thread_id"] == "original-thread" and args["mr_id"] == "original-mr"
    assert task.new_mr_url in args["body"] and task.lark_doc_url in args["body"]


@pytest.mark.asyncio
async def test_creation_error_is_reported_to_original_thread(monkeypatch):
    client = SimpleNamespace(call=AsyncMock(return_value={}))
    monkeypatch.setattr(webhooks, "get_mr", AsyncMock(side_effect=RuntimeError("MR lookup failed")))
    state = SimpleNamespace(settings=SimpleNamespace(nextcode_app_id="bot"), nextcode_client=client)
    with pytest.raises(RuntimeError, match="MR lookup failed"):
        await webhooks._handle_comment_event(state, {"Action": "created", "Comment": {
            "CommentableType": "merge_request", "CommentableId": "mr", "RepoId": "repo",
            "ThreadId": "original-thread", "Content": "/resolve",
        }})
    comments = [call.args[1] for call in client.call.await_args_list]
    assert len(comments) == 2
    assert "已收到" in comments[0]["Content"]
    assert "无法创建" in comments[1]["Content"] and "MR lookup failed" in comments[1]["Content"]
    assert all(comment["ThreadId"] == "original-thread" for comment in comments)
