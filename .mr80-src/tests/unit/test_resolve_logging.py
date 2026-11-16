import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from conciliator.domain.task import InvalidTransitionError, TaskStatus
from conciliator.services.resolve_service import ResolveService
from conciliator.utils.trace_context import context, observe_task, trace_scope
from conciliator.workers import task_runner
from conciliator.workers.task_runner import TaskRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "error", "cancelled"])
async def test_conflict_stage_logs_task_outcome_and_timing_without_source(failure):
    task = SimpleNamespace(
        id=7, status="BRANCH_CREATED", version=0,
        conflict_files_snapshot=[{
            "file_path": "a.py", "first_path": "a.py", "second_path": "a.py", "conflicts_id": "conflicts",
            "content": "<<<<<<< source\nprivate_source()\n=======\nprivate_target()\n>>>>>>> target\n",
        }],
    )
    save = AsyncMock()
    if failure == "error":
        save.side_effect = RuntimeError("private exception content")
    elif failure == "cancelled":
        save.side_effect = asyncio.CancelledError()
    repo = SimpleNamespace(save_conflict_blocks=save, update_status=AsyncMock(return_value=True), add_event=AsyncMock())
    svc = ResolveService(SimpleNamespace(repo=repo))
    with capture_logs() as logs:
        if failure:
            with pytest.raises(asyncio.CancelledError if failure == "cancelled" else RuntimeError):
                await svc._step2_identify_conflicts(task)
        else:
            await svc._step2_identify_conflicts(task)
    stage_logs = [record for record in logs if record["event"].startswith("resolve_stage_")]
    assert stage_logs[0]["event"] == "resolve_stage_started"
    expected = "failed" if failure == "error" else "interrupted" if failure == "cancelled" else "finished"
    assert stage_logs[-1]["event"] == f"resolve_stage_{expected}"
    assert stage_logs[-1]["duration_ms"] >= 0
    assert all(record["task_id"] == 7 for record in logs)
    assert "private_" not in str(logs) and "private exception" not in str(logs)
    if failure is None:
        transition = next(record for record in logs if record["event"] == "resolve_status_changed")
        assert (transition["from_status"], transition["to_status"]) == ("BRANCH_CREATED", "CONFLICTS_FETCHED")
        milestone = next(record for record in logs if record["event"] == "resolve_milestone")
        assert milestone["block_count"] == 1


@pytest.mark.asyncio
async def test_lost_status_update_does_not_log_success():
    task = SimpleNamespace(id=7, status="PREPARING", version=1)
    svc = ResolveService(SimpleNamespace(repo=SimpleNamespace(update_status=AsyncMock(return_value=False))))
    with capture_logs() as logs, pytest.raises(InvalidTransitionError):
        await svc._transition(task, TaskStatus.BRANCH_CREATED)
    assert [record["event"] for record in logs] == ["resolve_transition_conflict"]
    assert logs[0]["task_id"] == 7


@pytest.mark.asyncio
async def test_parallel_polls_keep_task_context_and_trace_ids_separate():
    contexts = []

    async def handler(task_id):
        await asyncio.sleep(0)
        contexts.append(dict(context.get()))
        assert context.get()["task_id"] == task_id

    runner = TaskRunner(repo=object(), resolve_service=object(), settings=SimpleNamespace(worker_concurrency=2))
    with trace_scope(99):
        parent = dict(context.get())
        with capture_logs() as logs:
            await asyncio.gather(runner._poll_task(1, handler), runner._poll_task(2, handler))
        assert dict(context.get()) == parent
    assert {ctx["task_id"] for ctx in contexts} == {1, 2}
    assert len({ctx["trace_id"] for ctx in contexts}) == 2
    assert len([record for record in logs if record["event"] == "task_poll_finished"]) == 2
    assert {record["task_id"] for record in logs} == {1, 2}


@pytest.mark.asyncio
async def test_poll_timeout_has_task_id(monkeypatch):
    monkeypatch.setattr(task_runner, "_POLL_TASK_TIMEOUT_SECONDS", 0.01)

    async def blocked(task_id):
        await asyncio.Event().wait()

    runner = TaskRunner(repo=object(), resolve_service=object(), settings=SimpleNamespace(worker_concurrency=1))
    with capture_logs() as logs:
        await runner._poll_task(7, blocked)
    assert logs[-1]["event"] == "task_poll_timed_out"
    assert logs[-1]["task_id"] == 7
    assert not any(record["event"] == "task_poll_finished" for record in logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("won", [False, True])
async def test_failure_refreshes_trace_context_before_feedback_only_after_success(won):
    task = SimpleNamespace(id=7, status="COCO_PROCESSING", version=5)
    svc = ResolveService(SimpleNamespace(repo=SimpleNamespace(update_status=AsyncMock(return_value=won))))

    async def reply(*args):
        assert context.get()["task_status"] == "FAILED"
        assert context.get()["task_version"] == 6

    svc._reply_original = AsyncMock(side_effect=reply)
    with trace_scope(7):
        observe_task(task)
        trace_id = context.get()["trace_id"]
        await svc._fail_task(task, "validation failed")
        assert context.get()["task_status"] == ("FAILED" if won else "COCO_PROCESSING")
        assert context.get()["task_version"] == (6 if won else 5)
        assert context.get()["task_id"] == 7 and context.get()["trace_id"] == trace_id
    if won:
        svc._reply_original.assert_awaited_once()
    else:
        svc._reply_original.assert_not_awaited()
