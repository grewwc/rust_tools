import asyncio
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Integer, create_engine
from sqlalchemy.orm import sessionmaker

from conciliator.domain.processing_version import build_active_key
from conciliator.domain.task import TaskStatus
from conciliator.integrations.nextcode.branches import Branch
from conciliator.integrations.nextcode.merges import ConflictFile, ConflictSnapshotMismatchError
from conciliator.integrations.nextcode.mrs import MR, MRVersion
from conciliator.repositories._sync_repo import ActiveTaskExistsError
from conciliator.repositories.models import Base, ResolveTask
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import approval_service, resolve_service
from conciliator.services.approval_service import ApprovalService
from conciliator.services.resolve_service import ResolveDeps, ResolveService


@pytest.fixture
def repository(tmp_path, monkeypatch):
    # SQLite requires INTEGER PRIMARY KEY for server-generated IDs.
    monkeypatch.setattr(ResolveTask.__table__.c.id, "type", Integer())
    engine = create_engine(f"sqlite:///{tmp_path / 'tasks.db'}")
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    yield TaskRepository(sf), sf
    engine.dispose()


def mr(source="a" * 40, target="b" * 40, number=1):
    return MR(
        id="mr-id",
        iid=42,
        title="conflict",
        description="",
        web_url="",
        source_repo_id="r1",
        target_repo_id="r1",
        source_branch="source",
        target_branch="target",
        author_email="mr-author@example.com",
        versions=(MRVersion(f"v{number}", number, source, target, "f" * 40),),
        source_head_sha=source,
        target_head_sha=target,
    )


def conflict(text="old"):
    return ConflictFile(
        "a.py", f"<<<<<<< HEAD\n{text}\n=======\ntarget\n>>>>>>> target\n", "a.py", "a.py", "conflicts-id"
    )


def service(repo):
    return ResolveService(ResolveDeps(nc=object(), coco=None, lark=None, repo=repo, notification=None, settings=None))


def mock_snapshot(monkeypatch, state):
    async def get_mr(*_args):
        current = state["mr"]
        return replace(
            current,
            source_head_sha=state.get("source_sha", current.source_head_sha),
            target_head_sha=state.get("target_sha", current.target_head_sha),
        )

    monkeypatch.setattr(resolve_service, "get_mr", AsyncMock(side_effect=get_mr))
    monkeypatch.setattr(resolve_service, "get_branch", AsyncMock(side_effect=AssertionError("use MR branch HEADs")))
    monkeypatch.setattr(resolve_service, "list_conflict_files", AsyncMock(return_value=[conflict()]))


async def trigger(svc):
    return await svc.handle_trigger(
        repo_id="r1",
        mr_iid=42,
        author_email="user@example.com",
        payload={},
        trigger_type="COMMENT",
        discussion_id="trigger-thread",
    )


async def create(repo, source="a" * 40, target="b" * 40, mr_iid=42):
    return await repo.create(
        repo_id="r1",
        original_mr_id=mr_iid,
        source_branch="source",
        target_branch="target",
        resolve_branch="",
        trigger_type="COMMENT",
        trigger_payload={},
        trigger_author_email="user@example.com",
        mr_version=1,
        source_commit_sha=source,
        target_commit_sha=target,
        conflict_files_snapshot=[asdict(conflict())],
    )


@pytest.mark.asyncio
async def test_duplicate_trigger_race_reuses_one_task(repository, monkeypatch):
    repo, _ = repository
    mock_snapshot(monkeypatch, {"mr": mr(), "target_sha": "d" * 40})
    monkeypatch.setattr(resolve_service, "reply_discussion", AsyncMock())
    ids = await asyncio.gather(*(trigger(service(repo)) for _ in range(8)))
    assert len(set(ids)) == 1
    tasks = await repo.list_active_for_mr("r1", 42)
    assert len(tasks) == 1
    assert tasks[0].id == ids[0]
    assert tasks[0].target_commit_sha == "d" * 40
    events = await repo.list_events(ids[0])
    assert [event.event_type for event in events] == ["TASK_CREATED", "LARK_NOTIFICATION_QUEUED"]
    assert events[1].payload["notification_type"] == "TASK_RECEIVED"


@pytest.mark.asyncio
async def test_each_head_change_creates_parallel_version_and_preserves_old_input(repository, monkeypatch):
    repo, _ = repository
    state = {"mr": mr()}
    mock_snapshot(monkeypatch, state)
    svc = service(repo)
    ids = [await trigger(svc)]
    state["mr"] = mr(source="c" * 40, number=2)
    ids.append(await trigger(svc))
    state["mr"] = mr(source="c" * 40, target="d" * 40, number=3)
    ids.append(await trigger(svc))
    assert len(set(ids)) == 3
    tasks = await repo.list_active_for_mr("r1", 42)
    assert all(task.status == "PENDING" for task in tasks)
    old = await repo.get(ids[0])
    assert (old.source_commit_sha, old.target_commit_sha, old.mr_version) == ("a" * 40, "b" * 40, 1)
    assert old.conflict_files_snapshot == [asdict(conflict())]
    assert old.trigger_payload["_trigger_discussion_id"] == "trigger-thread"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.ALL_APPROVED])
async def test_terminal_version_releases_only_its_key(repository, status):
    repo, _ = repository
    old = await create(repo)
    newer = await create(repo, source="c" * 40)
    with pytest.raises(ActiveTaskExistsError):
        await create(repo)
    await repo.update_status(task_id=old.id, from_version=0, new_status=status)
    replacement = await create(repo)
    assert replacement.id != old.id
    assert (await repo.get(newer.id)).status == "PENDING"
    assert (await repo.find_active_for_version("r1", 42, "a" * 40, "b" * 40)).id == replacement.id
    assert await repo.find_active_for_version("other", 42, "a" * 40, "b" * 40) is None


@pytest.mark.asyncio
async def test_matching_snapshot_survives_source_head_advance(repository, monkeypatch):
    repo, _ = repository
    state = {"mr": mr()}
    mock_snapshot(monkeypatch, state)
    calls = 0

    async def moving_conflicts(*args, **kwargs):
        nonlocal calls
        calls += 1
        state["mr"] = mr(source="c" * 40, number=2)
        return [conflict(str(calls))]

    monkeypatch.setattr(resolve_service, "list_conflict_files", moving_conflicts)
    task = await repo.get(await trigger(service(repo)))
    assert calls == 1
    assert (task.source_commit_sha, task.mr_version) == ("a" * 40, 1)
    assert task.conflict_files_snapshot == [asdict(conflict("1"))]
    resolve_service.get_mr.assert_awaited_once()
    assert len(await repo.list_active_for_mr("r1", 42)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("source,target", [("a", "d"), ("c", "b"), ("c", "d")])
async def test_stale_mr_version_freezes_current_heads(repository, monkeypatch, source, target):
    repo, _ = repository
    mock_snapshot(monkeypatch, {"mr": mr(), "source_sha": source * 40, "target_sha": target * 40})
    svc = service(repo)
    task = await repo.get(await trigger(svc))
    assert (task.source_commit_sha, task.target_commit_sha, task.mr_version) == (source * 40, target * 40, 1)
    resolve_service.list_conflict_files.assert_awaited_once_with(
        svc._deps.nc, "r1", 42, expected_conflicts_id=f"{source * 40}..{target * 40}",
    )


@pytest.mark.asyncio
async def test_target_head_change_without_mr_version_creates_new_task(repository, monkeypatch):
    repo, _ = repository
    state = {"mr": mr()}
    mock_snapshot(monkeypatch, state)
    monkeypatch.setattr(resolve_service, "reply_discussion", AsyncMock())
    svc = service(repo)
    old_id = await trigger(svc)
    state["target_sha"] = "d" * 40
    new_id = await trigger(svc)
    assert new_id != old_id
    assert await trigger(svc) == new_id
    assert (await repo.get(old_id)).target_commit_sha == "b" * 40
    newer = await repo.get(new_id)
    assert (newer.target_commit_sha, newer.mr_version) == ("d" * 40, 1)
    assert newer.active_key == build_active_key("r1", 42, "a" * 40, "d" * 40)


@pytest.mark.asyncio
async def test_matching_snapshot_survives_target_head_advance(repository, monkeypatch):
    repo, _ = repository
    state = {"mr": mr()}
    mock_snapshot(monkeypatch, state)
    calls = 0

    async def moving_conflicts(*args, **kwargs):
        nonlocal calls
        calls += 1
        state["target_sha"] = "d" * 40
        return [conflict(str(calls))]

    monkeypatch.setattr(resolve_service, "list_conflict_files", moving_conflicts)
    task = await repo.get(await trigger(service(repo)))
    assert calls == 1
    assert (task.target_commit_sha, task.mr_version) == ("b" * 40, 1)
    assert task.conflict_files_snapshot == [asdict(conflict("1"))]
    resolve_service.get_mr.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_mismatch_recaptures_current_heads(repository, monkeypatch):
    repo, _ = repository
    state = {"mr": mr()}
    mock_snapshot(monkeypatch, state)

    async def moving_conflicts(*args, expected_conflicts_id):
        if state["mr"].latest_version.number == 1:
            state["mr"] = mr(source="c" * 40, target="d" * 40, number=2)
            raise ConflictSnapshotMismatchError("heads changed before the snapshot was captured")
        assert expected_conflicts_id == f"{'c' * 40}..{'d' * 40}"
        return [conflict("new")]

    monkeypatch.setattr(resolve_service, "list_conflict_files", moving_conflicts)
    task = await repo.get(await trigger(service(repo)))
    assert (task.source_commit_sha, task.target_commit_sha, task.mr_version) == ("c" * 40, "d" * 40, 2)
    assert task.conflict_files_snapshot == [asdict(conflict("new"))]
    assert resolve_service.get_mr.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("recovers", [False, True])
async def test_capture_retries_mismatched_conflict_snapshot(repository, monkeypatch, recovers):
    repo, _ = repository
    mock_snapshot(monkeypatch, {"mr": mr(), "target_sha": "d" * 40})
    mismatch = ConflictSnapshotMismatchError("wrong commits")
    conflicts = AsyncMock(side_effect=[mismatch, [conflict()]] if recovers else mismatch)
    monkeypatch.setattr(resolve_service, "list_conflict_files", conflicts)
    if recovers:
        task = await repo.get(await trigger(service(repo)))
        assert task.target_commit_sha == "d" * 40
        assert conflicts.await_count == 2
    else:
        with pytest.raises(RuntimeError, match="stable MR processing version"):
            await trigger(service(repo))
        assert conflicts.await_count == 3
        assert await repo.list_active_for_mr("r1", 42) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["source_sha", "target_sha"])
@pytest.mark.parametrize("head", [None, ""])
async def test_missing_branch_head_does_not_create_task(repository, monkeypatch, side, head):
    repo, _ = repository
    mock_snapshot(monkeypatch, {"mr": mr(), side: head})
    with pytest.raises(RuntimeError, match="stable MR processing version"):
        await trigger(service(repo))
    assert await repo.list_active_for_mr("r1", 42) == []
    resolve_service.list_conflict_files.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_task_uses_snapshot_after_heads_advance(repository, monkeypatch):
    repo, _ = repository
    old = await create(repo)
    await repo.update_status(task_id=old.id, from_version=0, new_status=TaskStatus.PREPARING)
    old = await repo.get(old.id)
    get_mr = AsyncMock(side_effect=AssertionError("must not read current MR"))
    get_conflicts = AsyncMock(side_effect=AssertionError("must not read current conflicts"))
    ensure = AsyncMock()
    monkeypatch.setattr(resolve_service, "get_mr", get_mr)
    monkeypatch.setattr(resolve_service, "list_conflict_files", get_conflicts)
    monkeypatch.setattr(resolve_service, "ensure_branch", ensure)
    svc = service(repo)
    # ConflictBlock IDs are BIGINT on SQLite; inspect the manifest at the repository boundary.
    save = AsyncMock()
    monkeypatch.setattr(repo, "save_conflict_blocks", save)
    await svc._step1_prepare(old)
    await svc._step2_identify_conflicts(old)
    assert ensure.await_args.kwargs["revision"] == "a" * 40
    assert save.await_args.args[1][0]["ours_text"] == "old\n"
    assert old.source_commit_sha == "a" * 40
    assert old.target_commit_sha == "b" * 40
    assert old.active_key == build_active_key("r1", 42, "a" * 40, "b" * 40)


@pytest.mark.asyncio
async def test_old_version_retry_checks_resolver_head_only(monkeypatch):
    task = SimpleNamespace(
        id=7,
        repo_id="r1",
        original_mr_id=42,
        new_mr_id=99,
        mr_version=1,
        source_commit_sha="a" * 40,
        target_commit_sha="b" * 40,
        source_branch="source",
        target_branch="target",
        resolve_branch="resolver",
        resolve_commit_sha="c" * 40,
    )
    get_branch = AsyncMock(return_value=Branch("resolver", "c" * 40))
    monkeypatch.setattr(resolve_service, "get_branch", get_branch)
    monkeypatch.setattr(resolve_service, "get_mr", AsyncMock(return_value=replace(mr(), source_branch="resolver")))
    svc = service(None)
    await svc._verify_retry_baseline(task)
    assert get_branch.await_args.args[2] == "resolver"
    get_branch.return_value = Branch("resolver", "d" * 40)
    with pytest.raises(RuntimeError, match="resolve branch moved"):
        await svc._verify_retry_baseline(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("targeted", [False, True])
async def test_original_mr_cancel_does_not_cancel_other_versions(repository, monkeypatch, targeted):
    repo, sf = repository
    old = await create(repo)
    newer = await create(repo, source="c" * 40)
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    svc = ApprovalService(nc=object(), repo=repo, notification=object(), session_factory=sf)
    await svc.handle_comment(
        repo_id="r1",
        mr_iid=42,
        body=f"/resolve cancel {old.id}" if targeted else "/resolve cancel",
        author_email="user@example.com",
        discussion_id="command",
    )
    assert (await repo.get(old.id)).status == ("CANCELLED" if targeted else "PENDING")
    assert (await repo.get(newer.id)).status == "PENDING"
    assert reply.await_args.args[2] == 42
    if not targeted:
        assert f"/resolve cancel {old.id}" in reply.await_args.args[4]
        assert f"/resolve cancel {newer.id}" in reply.await_args.args[4]


@pytest.mark.asyncio
@pytest.mark.parametrize("author,mr_iid", [("other@example.com", 42), ("user@example.com", 43)])
async def test_targeted_cancel_enforces_author_and_mr_scope(repository, monkeypatch, author, mr_iid):
    repo, sf = repository
    task = await create(repo)
    monkeypatch.setattr(approval_service, "reply_discussion", AsyncMock())
    svc = ApprovalService(nc=object(), repo=repo, notification=object(), session_factory=sf)
    await svc.handle_comment(
        repo_id="r1", mr_iid=mr_iid, body=f"/resolve cancel {task.id}", author_email=author, discussion_id="command"
    )
    assert (await repo.get(task.id)).status == "PENDING"


@pytest.mark.asyncio
async def test_cancelled_coco_task_is_not_resumed(repository):
    repo, _ = repository
    task = await create(repo)
    await repo.update_status(task_id=task.id, from_version=0, new_status=TaskStatus.CANCELLED, coco_task_id="session")
    svc = service(repo)
    svc._deps.coco = SimpleNamespace(fetch_task_result=AsyncMock())
    await svc.continue_after_coco(task.id)
    svc._deps.coco.fetch_task_result.assert_not_awaited()
    assert (await repo.get(task.id)).status == "CANCELLED"
