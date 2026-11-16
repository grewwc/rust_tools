from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Integer, create_engine, select
from sqlalchemy.orm import sessionmaker

from conciliator.domain.approval import ApprovalStatus, approval_counts, can_transition
from conciliator.domain.task import TaskStatus
from conciliator.integrations.coco.client import ConflictAttribution, ConflictResolution, ResolvedConflict
from conciliator.integrations.git.conflicts import is_blank_conflict, parse_conflict_file
from conciliator.integrations.nextcode.mrs import MR, MRVersion
from conciliator.repositories.models import (
    Approval,
    Base,
    LarkNotificationOutbox,
    ResolveTask,
    TaskEvent,
)
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import resolve_service, templates
from conciliator.services.blank_conflicts import verify_blank_files
from conciliator.services.resolve_service import (
    CocoValidationError,
    ResolveDeps,
    ResolveService,
    _mr_blocks,
    _report_blocks,
)
from conciliator.services.task_status import original_mr_status_reply


@pytest.mark.parametrize(
    "source,target,expected",
    [
        ("", "", True),
        ("\n", "", True),
        ("\n\n", "\n", True),
        (" \t\r\n", "\r\n", True),
        (None, "", False),
        ("", None, False),
        ("code\n", "", False),
        ("# comment\n", "", False),
        (" x()\n", "x()\n", False),
        ("\u00a0\n", "", False),
        ("\u200b", "", False),
    ],
)
def test_blank_classification_is_narrow_and_never_treats_missing_data_as_blank(source, target, expected):
    assert is_blank_conflict(source, target) is expected


@pytest.mark.parametrize("source,target,eol", [("\n", "", "\n"), (" \t\r\n\r\n", "\r\n", "\r\n")])
def test_parser_preserves_exact_whitespace_and_line_endings(source, target, eol):
    text = f"<<<<<<< source{eol}{source}======={eol}{target}>>>>>>> target{eol}"
    (block,) = parse_conflict_file("a.py", text)
    assert block.ours_text == source
    assert block.theirs_text == target


def fixture_blocks(sides=None):
    # Task 24's shape: C0002/C0003/C0004 are a blank source line versus zero target lines.
    sides = sides if sides is not None else [("source()\n", "target()\n"), *[("\n", "")] * 3, ("old()\n", "new()\n")]
    snapshot = "head\n"
    for index, (source, target) in enumerate(sides):
        snapshot += f"<<<<<<< source\n{source}=======\n{target}>>>>>>> target\ncontext{index}\n"
    blocks, results = [], {}
    text = "head\n"
    for index, parsed in enumerate(parse_conflict_file("a.py", snapshot)):
        exempt = is_blank_conflict(parsed.ours_text, parsed.theirs_text)
        resolution = parsed.ours_text if exempt else f"resolved{index}()\n"
        start = len(text.splitlines()) + 1
        end = start + max(len(resolution.splitlines()) - 1, 0)
        cid = f"C{index + 1:04d}"
        blocks.append(
            SimpleNamespace(
                id=index + 1,
                task_id=1,
                conflict_id=cid,
                block_idx=index,
                file_path="a.py",
                start_line=parsed.start_line,
                end_line=parsed.end_line,
                ours_text=parsed.ours_text,
                theirs_text=parsed.theirs_text,
                approval=SimpleNamespace(status="NOT_REQUIRED" if exempt else "PENDING", comment=None),
                ai_resolution=resolution,
                resolved_path="a.py",
                resolved_start_line=start,
                resolved_end_line=end,
                source_author_email=None if exempt else "source@example.com",
                target_author_email=None,
                author_email=None if exempt else "source@example.com",
                decision_summary="keep source",
                source_author_commit_sha=None,
                target_author_commit_sha=None,
                source_path="a.py",
                target_path="a.py",
                source_start_line=0,
                source_end_line=0,
                target_start_line=0,
                target_end_line=0,
            )
        )
        attr = ConflictAttribution("a.py", 0, 0, "", None)
        results[cid] = ConflictResolution(
            cid,
            "a.py",
            "keep source",
            attr,
            attr,
            ResolvedConflict("a.py", start, end, resolution),
        )
        text += resolution + f"context{index}\n"
    return (
        SimpleNamespace(blocks=blocks, conflict_files_snapshot=[{"file_path": "a.py", "content": snapshot}]),
        results,
        text,
    )


def test_exact_blank_readback_preserves_ids_and_does_not_accept_blank_lines_elsewhere():
    task, results, content = fixture_blocks()
    assert verify_blank_files(task, results, {"a.py": content}) == {
        cid: (results[cid].resolved.start_line, results[cid].resolved.end_line) for cid in ("C0002", "C0003", "C0004")
    }
    for bad_content in (content.replace("context1", "injected()"), content.replace("\n\ncontext1", "\ncontext1", 1)):
        with pytest.raises(ValueError, match="committed content differs"):
            verify_blank_files(task, results, {"a.py": bad_content})
    bad = {**results, "C0002": replace(results["C0002"], resolved=ResolvedConflict("a.py", 4, 4, "evil()\n"))}
    with pytest.raises(ValueError, match="preserve exact source"):
        verify_blank_files(task, bad, {"a.py": content})
    bad = {**results, "C0002": replace(results["C0002"], resolved=replace(results["C0002"].resolved, start_line=2))}
    with pytest.raises(ValueError, match="position differs"):
        verify_blank_files(task, bad, {"a.py": content})


def test_empty_file_requires_verified_automatic_classification_and_zero_anchor():
    task, results, _ = fixture_blocks([("", "\n")])
    task.conflict_files_snapshot[0]["content"] = "<<<<<<< source\n=======\n\n>>>>>>> target\n"
    (parsed,) = parse_conflict_file("a.py", task.conflict_files_snapshot[0]["content"])
    block = task.blocks[0]
    block.start_line, block.end_line = parsed.start_line, parsed.end_line
    result = replace(results["C0001"], resolved=ResolvedConflict("a.py", 0, 0, ""))
    assert verify_blank_files(task, {"C0001": result}, {"a.py": ""}) == {"C0001": (0, 0)}
    ResolveService._check_author_shape(block, result)
    block.approval.status = "PENDING"
    with pytest.raises(CocoValidationError, match="positive discussion anchor"):
        ResolveService._check_author_shape(block, result)


def test_retry_keeps_automatic_blocks_locked_and_recalculates_shifted_positions():
    task, results, content = fixture_blocks()
    changed = replace(results["C0001"], resolved=replace(results["C0001"].resolved, text="extra()\nresolved0()\n"))
    updated = content.replace("resolved0()", "extra()\nresolved0()")
    positions = verify_blank_files(task, {"C0001": changed}, {"a.py": updated})
    assert positions["C0002"][0] == results["C0002"].resolved.start_line + 1
    with pytest.raises(ValueError, match="committed content differs"):
        verify_blank_files(task, {"C0001": changed}, {"a.py": updated.replace("\n\ncontext2", "\ncontext2", 1)})


@pytest.fixture
def database(tmp_path, monkeypatch):
    # SQLite autoincrement requires INTEGER; production keeps BIGINT.
    for table in Base.metadata.tables.values():
        if "id" in table.c and isinstance(table.c.id.type, Integer):
            monkeypatch.setattr(table.c.id, "type", Integer())
    engine = create_engine(f"sqlite:///{tmp_path / 'blank.db'}")
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    yield TaskRepository(sf), sf
    engine.dispose()


async def persist_fixture(database, sides=None):
    repo, sf = database
    fixture, results, content = fixture_blocks(sides)
    task = await repo.create(
        repo_id="r",
        source_branch="source",
        target_branch="target",
        resolve_branch="resolver",
        original_mr_id=42,
        trigger_type="COMMENT",
        trigger_payload={},
        trigger_author_email="trigger@example.com",
        conflict_files_snapshot=fixture.conflict_files_snapshot,
    )
    keys = ("conflict_id", "block_idx", "file_path", "start_line", "end_line", "ours_text", "theirs_text")
    manifest = [{key: getattr(block, key) for key in keys} for block in fixture.blocks]
    ids = await repo.save_conflict_blocks(task.id, manifest)
    assert await repo.save_conflict_blocks(task.id, manifest) == ids
    for block_id, block in zip(ids, fixture.blocks, strict=True):
        await repo.update_block_resolution(
            block_id=block_id,
            fields={
                key: getattr(block, key)
                for key in ("ai_resolution", "resolved_path", "resolved_start_line", "resolved_end_line")
            },
        )
    with sf() as session:
        row = session.get(ResolveTask, task.id)
        row.status = "AWAITING_APPROVAL"
        row.new_mr_id = 99
        row.new_mr_url = "https://code.test/mr/99"
        row.resolve_commit_sha = "c" * 40
        row.active_key = "r:42"
        session.commit()
    return await repo.get_with_blocks(task.id), manifest


@pytest.mark.asyncio
async def test_mixed_task_keeps_five_ids_but_only_two_manual_approvals(database):
    repo, sf = database
    task, _ = await persist_fixture(database)
    counts = approval_counts(task.blocks)
    assert counts == {"PENDING": 2, "APPROVED": 0, "REJECTED": 0, "NOT_REQUIRED": 3, "UNKNOWN": 0}
    assert [b.conflict_id for b in task.blocks] == [f"C{i:04d}" for i in range(1, 6)]
    for block in task.blocks:
        if block.approval.status != "NOT_REQUIRED":
            assert (
                await repo.approve_block(
                    task_id=task.id,
                    block_id=block.id,
                    reviewer_email="reviewer@example.com",
                    discussion_id="t",
                    webhook_event_id=None,
                )
                == "approved"
            )
    full = await repo.get_with_blocks(task.id)
    assert await repo.complete_all_approved_and_enqueue(
        task_id=task.id,
        from_version=full.version,
        approved_count=2,
    ) == (True, True)
    with sf() as session:
        exempt = list(session.scalars(select(Approval).where(Approval.status == "NOT_REQUIRED")))
        assert len(exempt) == 3
        assert all(a.reviewer_email is None and a.reviewed_at is None for a in exempt)
        events = list(session.scalars(select(TaskEvent)))
        assert sum(e.event_type == "BLOCK_APPROVAL_NOT_REQUIRED" for e in events) == 3
        assert sum(e.event_type == "BLOCK_APPROVED" for e in events) == 2
        assert session.get(ResolveTask, task.id).active_key is None


@pytest.mark.asyncio
async def test_approve_all_skips_automatic_blocks(database):
    repo, sf = database
    task, _ = await persist_fixture(database)
    result = await repo.approve_all_pending_and_enqueue(
        task_id=task.id, reviewer_email="admin@example.com", webhook_event_id=None
    )
    assert (result.outcome, result.total_count, result.approved_now_count) == ("completed", 2, 2)
    full = await repo.get_with_blocks(task.id)
    assert approval_counts(full.blocks)["NOT_REQUIRED"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_status", [None, "UNKNOWN"])
async def test_missing_or_unknown_approval_cannot_be_ignored_by_completion(database, bad_status):
    repo, sf = database
    task, _ = await persist_fixture(database)
    with sf() as session:
        approvals = list(session.scalars(select(Approval)))
        for approval in approvals:
            if approval.status == "PENDING":
                approval.status = "APPROVED"
        if bad_status is None:
            session.delete(approvals[1])
        else:
            approvals[1].status = bad_status
        session.commit()
    assert await repo.complete_all_approved_and_enqueue(
        task_id=task.id,
        from_version=task.version,
        approved_count=2,
    ) == (False, False)
    result = await repo.approve_all_pending_and_enqueue(
        task_id=task.id, reviewer_email="admin@example.com", webhook_event_id=None
    )
    assert result.outcome == "invalid_task"
    assert (await repo.get(task.id)).status == "AWAITING_APPROVAL"


@pytest.mark.asyncio
async def test_retry_does_not_reset_or_operate_on_exempt_blocks(database):
    repo, sf = database
    task, _ = await persist_fixture(database)
    exempt = task.blocks[1]
    assert not can_transition(ApprovalStatus.NOT_REQUIRED, ApprovalStatus.APPROVED)
    for block in (task.blocks[0], exempt):
        await repo.reject_block_and_mark_partial(
            task_id=task.id,
            block_id=block.id,
            reviewer_email="admin@example.com",
            reason="fix",
            discussion_id="t",
            webhook_event_id=None,
        )
    await repo.approve_block(
        task_id=task.id,
        block_id=exempt.id,
        reviewer_email="admin@example.com",
        discussion_id="t",
        webhook_event_id=None,
    )
    request = await repo.request_retry(task_id=task.id, requested_by="admin@example.com", webhook_event_id=None)
    assert request.rejected_conflict_ids == ("C0001",)
    assert set(request.locked_conflict_ids) == {"C0002", "C0003", "C0004", "C0005"}
    assert await repo.reset_rejected_approvals(task_id=task.id, conflict_ids=("C0001", "C0002")) == 1
    assert approval_counts((await repo.get_with_blocks(task.id)).blocks)["NOT_REQUIRED"] == 3


@pytest.mark.asyncio
async def test_all_blank_completion_requires_publication_and_is_idempotent(database):
    repo, sf = database
    task, _ = await persist_fixture(database, [("\n", "")])
    with sf() as session:
        session.get(ResolveTask, task.id).status = "MR_CREATED"
        session.commit()
    args = dict(task_id=task.id, from_version=task.version, approved_count=0, no_manual_approval=True)
    assert await repo.complete_all_approved_and_enqueue(**args) == (False, False)
    with sf() as session:
        session.get(ResolveTask, task.id).trigger_payload = {"_blank_approval_publication_verified": True}
        session.commit()
    assert await repo.complete_all_approved_and_enqueue(**args) == (True, True)
    assert await repo.complete_all_approved_and_enqueue(**args) == (False, False)
    with sf() as session:
        notification = session.scalar(select(LarkNotificationOutbox).where(
            LarkNotificationOutbox.notification_type == "NO_MANUAL_APPROVAL_REQUIRED",
        ))
        assert notification.notification_type == "NO_MANUAL_APPROVAL_REQUIRED"
        completed = session.scalar(select(TaskEvent).where(TaskEvent.event_type == "ALL_APPROVED"))
        assert completed.payload["approved_count"] == 0
        assert completed.payload["not_required_count"] == 1
        assert completed.payload["completion_reason"] == "NO_MANUAL_APPROVAL_REQUIRED"
    task = await repo.get(task.id)
    assert "纯空行冲突已处理，无需人工审批" in original_mr_status_reply([task])


@pytest.mark.asyncio
async def test_cancel_wins_against_stale_automatic_completion(database):
    repo, sf = database
    task, _ = await persist_fixture(database, [("\n", "")])
    with sf() as session:
        row = session.get(ResolveTask, task.id)
        row.status = "MR_CREATED"
        row.trigger_payload = {"_blank_approval_publication_verified": True}
        session.commit()
    assert (
        await repo.cancel_task(
            task_id=task.id, from_version=task.version, cancelled_by="admin@example.com", webhook_event_id=None
        )
    ).cancelled
    assert await repo.complete_all_approved_and_enqueue(
        task_id=task.id,
        from_version=task.version,
        approved_count=0,
        no_manual_approval=True,
    ) == (False, False)
    assert (await repo.get(task.id)).status == "CANCELLED"
    with sf() as session:
        assert [item.notification_type for item in session.scalars(select(LarkNotificationOutbox))] == ["TASK_RECEIVED"]


@pytest.mark.asyncio
@pytest.mark.parametrize("all_blank,crash", [(False, False), (True, False), (True, True)])
async def test_publish_only_manual_discussions_or_complete_and_recover_all_blank(
    database, monkeypatch, all_blank, crash
):
    repo, sf = database
    task, _ = await persist_fixture(database, [("\n", "")] if all_blank else None)
    with sf() as session:
        session.get(ResolveTask, task.id).status = "DOC_CREATED"
        session.commit()
    task = await repo.get(task.id)
    mr = MR(
        id="mr",
        iid=99,
        title="resolve",
        description="",
        web_url="https://code.test/mr/99",
        source_repo_id="r",
        target_repo_id="r",
        source_branch="resolver",
        target_branch="target",
        versions=(MRVersion("v", 1, "c" * 40, "b" * 40, "b" * 40),),
    )
    notification = SimpleNamespace(notify_mr_comment=AsyncMock())
    svc = ResolveService(
        ResolveDeps(nc=object(), coco=None, lark=None, repo=repo, notification=notification, settings=None)
    )
    create = AsyncMock(return_value=mr)
    monkeypatch.setattr(svc, "_create_or_recover_resolve_mr", create)

    async def readback(*args):
        return replace(mr, description=create.await_args.kwargs["description"])

    monkeypatch.setattr(resolve_service, "get_mr", readback)
    discussion = AsyncMock(side_effect=[SimpleNamespace(id=f"thread-{i}", comment_id=f"comment-{i}") for i in (1, 5)])
    monkeypatch.setattr(resolve_service, "create_discussion", discussion)
    monkeypatch.setattr(resolve_service, "get_thread", AsyncMock(return_value=object()))
    monkeypatch.setattr(
        resolve_service,
        "find_position_for_version",
        lambda *args: {
            "Path": discussion.await_args.kwargs["file_path"],
            "StartLine": discussion.await_args.kwargs["line"],
            "EndLine": discussion.await_args.kwargs["end_line"],
        },
    )
    complete = svc._complete_blank_approval
    if crash:
        monkeypatch.setattr(svc, "_complete_blank_approval", AsyncMock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            await svc._step5_create_mr_and_discussions(task)
        assert (await repo.get(task.id)).status == "MR_CREATED"
        monkeypatch.setattr(svc, "_complete_blank_approval", complete)
        assert await svc.resume_blank_approval(task.id)
        assert not await svc.resume_blank_approval(task.id)
    else:
        await svc._step5_create_mr_and_discussions(task)
    full = await repo.get_with_blocks(task.id)
    if all_blank:
        assert full.status == "ALL_APPROVED"
        discussion.assert_not_awaited()
        notification.notify_mr_comment.assert_not_awaited()
        assert "纯空行冲突已处理，无需人工审批" in create.await_args.kwargs["description"]
    else:
        assert full.status == "AWAITING_APPROVAL"
        assert discussion.await_count == 2
        bodies = [call.kwargs["body"] for call in discussion.await_args_list]
        assert "block=C0001" in bodies[0] and "block=C0005" in bodies[1]
        description = templates.build_mr_description(
            source_branch="source",
            target_branch="target",
            resolve_branch="resolver",
            lark_doc_url="url",
            blocks=_mr_blocks(full),
        )
        assert "人工审批 0/2，另有 3 块无需人工审批" in description
        assert "- [ ] #2" not in description and "- [ ] #5" in description
        report = templates.build_lark_doc_markdown(
            original_mr_number=42,
            source_branch="source",
            source_commit="a",
            target_branch="target",
            target_commit="b",
            resolve_branch="resolver",
            resolve_commit="c",
            blocks=_report_blocks(full),
        )
        assert "Source：1 行空行" in report and "Target：无内容" in report


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [None, "text", "position", "file"])
async def test_automatic_validation_reads_commit_and_routes_failures_to_repair(database, monkeypatch, bad):
    from conciliator.integrations.coco.client import CocoResolveResult, CocoTaskResult
    from conciliator.integrations.nextcode.branches import Branch
    from conciliator.integrations.nextcode.commits import CommitDetails
    from conciliator.integrations.nextcode.files import RepositoryFile

    repo, sf = database
    task, _ = await persist_fixture(database, [("\n", "")])
    fixture, conflicts, content = fixture_blocks([("\n", "")])
    with sf() as session:
        row = session.get(ResolveTask, task.id)
        row.status = "COCO_RUNNING"
        row.source_commit_sha = "a" * 40
        row.target_commit_sha = "b" * 40
        row.coco_task_id = "session"
        session.commit()
    task = await repo.get_with_blocks(task.id)
    conflict = conflicts["C0001"]
    if bad == "text":
        conflict = replace(conflict, resolved=replace(conflict.resolved, text="injected()\n"))
    elif bad == "position":
        conflict = replace(conflict, resolved=replace(conflict.resolved, start_line=1, end_line=1))
    result = CocoResolveResult(
        schema_version="conciliator.resolve.result.v3",
        task_id=task.id,
        status="succeeded",
        attempt=1,
        resolve_branch="resolver",
        commit_sha="c" * 40,
        merge_parents=("a" * 40, "b" * 40),
        conflicts=(conflict,),
        unresolved_items=(),
    )
    monkeypatch.setattr(resolve_service, "get_branch", AsyncMock(return_value=Branch("resolver", "c" * 40)))
    monkeypatch.setattr(
        resolve_service,
        "get_commit",
        AsyncMock(
            return_value=CommitDetails(
                "c" * 40,
                "merge",
                ("a" * 40, "b" * 40),
            )
        ),
    )
    file_content = content.replace("\n\n", "\n") if bad == "file" else content
    monkeypatch.setattr(
        resolve_service, "get_file", AsyncMock(return_value=RepositoryFile("a.py", file_content, "text"))
    )
    blame = AsyncMock()
    monkeypatch.setattr(resolve_service, "get_blame", blame)
    coco = SimpleNamespace(
        fetch_task_result=AsyncMock(return_value=CocoTaskResult(status="succeeded", resolve_result=result))
    )
    svc = ResolveService(
        ResolveDeps(nc=object(), coco=coco, lark=object(), repo=repo, notification=None, settings=None)
    )
    if bad is None:
        assert await svc._verify_coco_result(task, result) == {"a.py": content}
        report, _ = await svc._prepare_report_data(task, CocoTaskResult(status="succeeded", resolve_result=result))
        assert report[0]["ai_resolution"] == "\n"
        assert report[0]["approval_status"] == "NOT_REQUIRED"
        blame.assert_not_awaited()
    else:
        repair = AsyncMock()
        monkeypatch.setattr(svc, "_repair_or_fail", repair)
        await svc.continue_after_coco(task.id)
        repair.assert_awaited_once()
        assert repair.await_args.kwargs == {"retry": False}
        assert (await repo.get(task.id)).status == "COCO_RUNNING"


@pytest.mark.asyncio
async def test_concurrent_cancel_and_automatic_completion_only_one_wins(database):
    import asyncio

    repo, sf = database
    task, _ = await persist_fixture(database, [("\n", "")])
    with sf() as session:
        row = session.get(ResolveTask, task.id)
        row.status = "MR_CREATED"
        row.trigger_payload = {"_blank_approval_publication_verified": True}
        session.commit()
    completed, cancelled = await asyncio.gather(
        repo.complete_all_approved_and_enqueue(
            task_id=task.id, from_version=task.version, approved_count=0, no_manual_approval=True
        ),
        repo.cancel_task(
            task_id=task.id, from_version=task.version, cancelled_by="admin@example.com", webhook_event_id=None
        ),
    )
    assert int(completed[0]) + int(cancelled.cancelled) == 1
    assert (await repo.get(task.id)).status == ("ALL_APPROVED" if completed[0] else "CANCELLED")


@pytest.mark.asyncio
async def test_worker_recovers_all_blank_publication_after_restart_and_periodically(database):
    from conciliator.workers.task_runner import TaskRunner

    repo, sf = database
    task, _ = await persist_fixture(database, [("\n", "")])
    task.trigger_payload = {"_blank_approval_publication_verified": True}
    task.status = "MR_CREATED"

    async def list_status(*, status, **kwargs):
        return [task] if status == TaskStatus.MR_CREATED else []

    fake_repo = SimpleNamespace(
        list_by_status_updated_before=AsyncMock(side_effect=list_status),
        list_poll_candidates=AsyncMock(side_effect=lambda **kw: (
            [(task.id, task.updated_at)] if kw["status"] == TaskStatus.MR_CREATED else []
        )),
        update_status=AsyncMock(),
    )
    svc = SimpleNamespace(**{name: AsyncMock() for name in (
        "resume_blank_approval", "notify_validation_failure", "fail_stuck_coco_poll", "continue_after_coco",
        "continue_after_retry_coco", "resume_retry_publication",
    )})
    runner = TaskRunner(repo=fake_repo, resolve_service=svc, settings=SimpleNamespace(worker_concurrency=1))
    await runner._recover_stuck_tasks()
    await runner._poll_coco_once()
    assert svc.resume_blank_approval.await_count == 2
    fake_repo.update_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_reply_and_last_manual_approval_use_two_block_total(database, monkeypatch):
    from conciliator.services import approval_service
    from conciliator.services.approval_service import ApprovalService

    repo, sf = database
    task, _ = await persist_fixture(database)
    svc = ApprovalService(nc=object(), repo=repo, notification=None, session_factory=sf)
    reply = AsyncMock()
    monkeypatch.setattr(approval_service, "reply_discussion", reply)
    monkeypatch.setattr(svc, "_sync_mr_description", repo.get_with_blocks)
    published = AsyncMock()
    monkeypatch.setattr(svc, "_publish_all_approved", published)
    await svc._reply_status(task, "command-thread", reply_mr_iid=99)
    assert "0/2 APPROVED" in reply.await_args.args[4]
    assert "3 块纯空行冲突无需人工审批" in reply.await_args.args[4]
    for index in (0, 4):
        await svc._handle_approve_block(
            task, task.blocks[index], "reviewer@example.com", "thread", webhook_event_id=None
        )
    assert "2/2" in reply.await_args.args[4]
    assert (await repo.get(task.id)).status == "ALL_APPROVED"
    published.assert_awaited_once()


def test_zero_anchor_wire_contract_still_rejects_nonempty_text():
    from conciliator.integrations.coco.client import CocoResultError, _parse_resolved

    assert _parse_resolved({"path": "a.py", "start_line": 0, "end_line": 0, "text": ""}, "resolved").start_line == 0
    for text in ("\n", "code"):
        with pytest.raises(CocoResultError, match="zero anchor"):
            _parse_resolved({"path": "a.py", "start_line": 0, "end_line": 0, "text": text}, "resolved")
