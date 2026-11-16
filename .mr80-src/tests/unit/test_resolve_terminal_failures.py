from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Integer, create_engine
from sqlalchemy.orm import sessionmaker

from conciliator.integrations.coco.client import (
    CocoResolveResult,
    CocoRetryResult,
    CocoTaskResult,
    ConflictAttribution,
)
from conciliator.integrations.nextcode.client import NextCodeError
from conciliator.repositories.models import Base, ResolveTask
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import resolve_service
from conciliator.services.resolve_service import ResolveService, UnsupportedConflictError, _check_supported_conflicts

OLD = "aeolus/libs/async/async.py"
NEW = "aeolus/libs/async_util/async_util.py"
COMMIT = "7aa851bbabc1997c4b56b667b3708df02809cb14"


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(ResolveTask.__table__.c.id, "type", Integer())
    engine = create_engine(f"sqlite:///{tmp_path / 'terminal.db'}")
    Base.metadata.create_all(engine)
    sf = sessionmaker(engine, expire_on_commit=False)
    yield TaskRepository(sf), sf
    engine.dispose()


async def make_task(database, *, renamed=False, status="PENDING"):
    repo, sf = database
    task = await repo.create(
        repo_id="9449", original_mr_id=42, source_branch="source", target_branch="target",
        resolve_branch="resolve/task", trigger_type="COMMENT",
        trigger_payload={"_trigger_discussion_id": "original-thread", "_trigger_mr_id": "original-mr"},
        source_commit_sha="a" * 40, target_commit_sha="b" * 40, mr_version=1,
        conflict_files_snapshot=[{
            "file_path": NEW, "first_path": OLD if renamed else NEW, "second_path": NEW,
            "content": "<<<<<<< source\nsource()\n=======\ntarget()\n>>>>>>> target\n", "conflicts_id": "c",
        }],
    )
    with sf.begin() as session:
        row = session.get(ResolveTask, task.id)
        row.status = status
        row.coco_task_id = "session"
        row.resolve_commit_sha = "c" * 40
    return await repo.get(task.id)


def service(repo, monkeypatch):
    coco = SimpleNamespace(
        submit_resolve_task=AsyncMock(), fetch_task_result=AsyncMock(), fetch_retry_task_result=AsyncMock(),
    )
    nc = SimpleNamespace(call=AsyncMock())
    svc = ResolveService(SimpleNamespace(repo=repo, nc=nc, coco=coco, lark=object()))
    reply, upload, repair = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(resolve_service, "send_resolve_feedback", reply)
    monkeypatch.setattr(resolve_service, "upload_markdown", upload)
    monkeypatch.setattr(svc, "_repair_or_fail", repair)
    return svc, coco, nc, reply, upload, repair


def result(task, retry=False):
    fields = dict(
        schema_version="conciliator.resolve.result.v3", task_id=task.id, status="succeeded", attempt=1,
        resolve_branch=task.resolve_branch, commit_sha=COMMIT, conflicts=(), unresolved_items=(),
    )
    if retry:
        return CocoTaskResult(status="succeeded", retry_result=CocoRetryResult(
            **fields, base_commit_sha="c" * 40, parent_commits=("c" * 40,),
        ))
    return CocoTaskResult(status="succeeded", resolve_result=CocoResolveResult(
        **fields, merge_parents=("a" * 40, "b" * 40),
    ))


@pytest.mark.parametrize("first,second", [(OLD, NEW), ("a.py", "b.py")])
def test_different_nonempty_paths_are_unsupported(first, second):
    with pytest.raises(UnsupportedConflictError, match="暂不支持文件重命名冲突"):
        _check_supported_conflicts(SimpleNamespace(
            conflict_files_snapshot=[{"first_path": first, "second_path": second}],
        ))


@pytest.mark.parametrize("first,second", [(NEW, NEW), ("", NEW), (OLD, ""), (None, None)])
def test_same_path_and_one_missing_side_are_not_treated_as_rename(first, second):
    _check_supported_conflicts(SimpleNamespace(conflict_files_snapshot=[{"first_path": first, "second_path": second}]))


@pytest.mark.asyncio
async def test_new_renamed_task_fails_before_branch_creation_or_coco(database, monkeypatch):
    repo, _ = database
    task = await make_task(database, renamed=True, status="PREPARING")
    svc, coco, nc, reply, upload, repair = service(repo, monkeypatch)
    await svc.run_pipeline(task.id)
    failed = await repo.get(task.id)
    assert failed.status == "FAILED" and failed.active_key is None
    assert OLD in failed.last_error and NEW in failed.last_error
    assert "暂不支持文件重命名冲突" in reply.await_args.kwargs["body"]
    nc.call.assert_not_awaited()
    coco.submit_resolve_task.assert_not_awaited()
    upload.assert_not_awaited()
    repair.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_legacy_renamed_task_preserves_result_and_stops(database, monkeypatch):
    repo, _ = database
    task = await make_task(database, renamed=True, status="COCO_RUNNING")
    svc, coco, nc, reply, upload, repair = service(repo, monkeypatch)
    coco.fetch_task_result.return_value = result(task)
    await svc.continue_after_coco(task.id)
    failed = await repo.get(task.id)
    assert failed.status == "FAILED" and failed.active_key is None
    assert failed.coco_result_json["commit_sha"] == COMMIT
    assert "暂不支持文件重命名冲突" in reply.await_args.kwargs["body"]
    nc.call.assert_not_awaited()
    upload.assert_not_awaited()
    repair.assert_not_awaited()
    await svc.continue_after_coco(task.id)
    coco.fetch_task_result.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("http_status", [422, 503])
async def test_permanent_blame_error_ends_task_but_transient_error_can_retry(database, monkeypatch, retry, http_status):
    repo, _ = database
    status = "RETRY_COCO_RUNNING" if retry else "COCO_RUNNING"
    task = await make_task(database, status=status)
    svc, coco, _nc, reply, upload, repair = service(repo, monkeypatch)
    fetch = coco.fetch_retry_task_result if retry else coco.fetch_task_result
    fetch.return_value = result(task, retry)
    error = NextCodeError(
        status=http_status, action="GetBlame", code="Unprocessable" if http_status == 422 else "InternalError",
        message="failed to parse blame: invalid blame format", request_id="request-29",
    )
    monkeypatch.setattr(resolve_service, "get_blame", AsyncMock(side_effect=error))

    async def verify(full_task, *_args, **_kwargs):
        await svc._verify_attribution(
            full_task, ConflictAttribution(NEW, 633, 642, "d" * 40, "author@example.com"),
            revision=full_task.source_commit_sha, side="source",
        )

    monkeypatch.setattr(svc, "_verify_retry_result" if retry else "_verify_coco_result", verify)
    if retry:
        monkeypatch.setattr(svc, "_retry_blocks_for_attempt", AsyncMock(return_value=([], [])))
    proceed = svc.continue_after_retry_coco if retry else svc.continue_after_coco
    await proceed(task.id)
    current = await repo.get(task.id)
    if http_status == 422:
        assert current.status == "FAILED" and current.active_key is None
        assert "request-29" in current.last_error
        assert "request-29" in reply.await_args.kwargs["body"]
        if not retry:
            assert current.coco_result_json["commit_sha"] == COMMIT
        await proceed(task.id)
        fetch.assert_awaited_once()
    else:
        assert current.status == status and current.active_key is not None
        reply.assert_not_awaited()
    upload.assert_not_awaited()
    repair.assert_not_awaited()
