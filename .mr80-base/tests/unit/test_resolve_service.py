from dataclasses import asdict, replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from conciliator.domain.processing_version import build_active_key
from conciliator.domain.task import TaskStatus
from conciliator.integrations.coco.client import (
    CocoResolveResult,
    CocoRetryResult,
    CocoTaskRef,
    CocoTaskResult,
    ConflictAttribution,
    ConflictResolution,
    ResolvedConflict,
)
from conciliator.integrations.lark.drive import MarkdownUpload
from conciliator.integrations.nextcode.branches import Branch
from conciliator.integrations.nextcode.commits import CommitDetails
from conciliator.integrations.nextcode.discussions import Discussion, Thread
from conciliator.integrations.nextcode.files import RepositoryFile
from conciliator.integrations.nextcode.merges import ConflictFile
from conciliator.integrations.nextcode.mrs import MR, MRVersion
from conciliator.integrations.nextcode.repos import BlameSegment
from conciliator.services import resolve_service
from conciliator.services.resolve_service import ResolveDeps, ResolveService


def _trigger_mr(source_sha="a" * 40, target_sha="b" * 40, number=7):
    return MR(
        id="mr-id", iid=42, title="conflict", description="", web_url="",
        source_repo_id="repo-1", target_repo_id="repo-1",
        source_branch="source", target_branch="target",
        versions=(MRVersion("version-id", number, source_sha, target_sha, "f" * 40),),
        source_head_sha=source_sha, target_head_sha=target_sha,
    )


def _mock_trigger_snapshot(monkeypatch, mr=None):
    mr = mr or _trigger_mr()
    get_mr = AsyncMock(return_value=mr)
    monkeypatch.setattr(resolve_service, "get_mr", get_mr)
    monkeypatch.setattr(resolve_service, "get_branch", AsyncMock(side_effect=AssertionError("use MR branch HEADs")))
    conflicts = AsyncMock(return_value=[])
    monkeypatch.setattr(resolve_service, "list_conflict_files", conflicts)
    return get_mr, conflicts


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger_type", ["COMMENT", "AUTO_CONFLICT"])
async def test_new_task_receipt_links_to_trace_instead_of_an_mr(monkeypatch, trigger_type):
    mr = _trigger_mr()
    _mock_trigger_snapshot(monkeypatch, mr)
    repo = SimpleNamespace(
        find_active_for_version=AsyncMock(return_value=None),
        create=AsyncMock(return_value=SimpleNamespace(id=49)),
    )
    feedback = AsyncMock()
    monkeypatch.setattr(resolve_service, "send_resolve_feedback", feedback)
    nextcode = object()
    service = ResolveService(SimpleNamespace(repo=repo, nc=nextcode))

    task_id = await service.handle_trigger(
        repo_id="repo-1", mr_iid=42, author_email="user@example.com", payload={},
        trigger_type=trigger_type, discussion_id="resolve-thread-1",
    )

    assert task_id == 49
    repo.create.assert_awaited_once()
    if trigger_type == "COMMENT":
        feedback.assert_awaited_once_with(
            nextcode, repo_id="repo-1", mr_id=mr.id, thread_id="resolve-thread-1",
            body=(
                "解冲突任务 49 已受理，正在排队或处理中。\n"
                "[查看处理进度](https://aeolus-conciliator.byted.org/trace?task_id=49&order=desc)"
            ),
        )
    else:
        feedback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "new_mr_url", "mr_link"),
    [
        ("COCO_RUNNING", None, ""),
        ("PENDING", "", ""),
        (
            "AWAITING_APPROVAL",
            "https://code.byted.org/xiahaitao/Conciliator/merge_requests/29",
            "[查看解冲突 MR](https://code.byted.org/xiahaitao/Conciliator/merge_requests/29)。",
        ),
    ],
)
async def test_handle_trigger_replies_to_comment_when_active_task_exists(
    monkeypatch, status, new_mr_url, mr_link,
):
    _mock_trigger_snapshot(monkeypatch)
    active_task = SimpleNamespace(id=123, status=status, new_mr_url=new_mr_url)
    repo = SimpleNamespace(
        find_active_for_version=AsyncMock(return_value=active_task),
        create=AsyncMock(),
    )
    nextcode = object()
    reply = AsyncMock()
    monkeypatch.setattr(resolve_service, "reply_discussion", reply)
    service = ResolveService(
        ResolveDeps(
            nc=nextcode,
            coco=object(),
            lark=None,
            repo=repo,
            notification=None,
            settings=None,
        )
    )

    task_id = await service.handle_trigger(
        repo_id="repo-1",
        mr_iid=42,
        author_email="user@example.com",
        payload={"source_branch": "source", "target_branch": "target"},
        trigger_type="COMMENT",
        discussion_id="resolve-thread-1",
    )

    assert task_id == 123
    repo.create.assert_not_awaited()
    reply.assert_awaited_once_with(
        nextcode,
        "repo-1",
        42,
        "resolve-thread-1",
        f"当前代码版本已有解冲突任务 123（{status}），已复用该任务。" + mr_link,
    )


@pytest.mark.asyncio
async def test_handle_trigger_replies_when_active_task_is_created_concurrently(
    monkeypatch,
):
    _mock_trigger_snapshot(monkeypatch)
    repo = SimpleNamespace(
        find_active_for_version=AsyncMock(
            side_effect=[None, SimpleNamespace(id=123, status="PENDING", new_mr_url=None)],
        ),
        create=AsyncMock(
            side_effect=resolve_service.ActiveTaskExistsError("active task exists")
        ),
    )
    nextcode = object()
    reply = AsyncMock()
    monkeypatch.setattr(resolve_service, "reply_discussion", reply)
    service = ResolveService(
        ResolveDeps(
            nc=nextcode,
            coco=object(),
            lark=None,
            repo=repo,
            notification=None,
            settings=None,
        )
    )

    task_id = await service.handle_trigger(
        repo_id="repo-1",
        mr_iid=42,
        author_email="user@example.com",
        payload={"source_branch": "source", "target_branch": "target"},
        trigger_type="COMMENT",
        discussion_id="resolve-thread-1",
    )

    assert task_id == 123
    reply.assert_awaited_once_with(
        nextcode,
        "repo-1",
        42,
        "resolve-thread-1",
        "当前代码版本已有解冲突任务 123（PENDING），已复用该任务。",
    )


@pytest.mark.asyncio
async def test_handle_trigger_keeps_auto_conflict_duplicate_silent(monkeypatch):
    _mock_trigger_snapshot(monkeypatch)
    active_task = SimpleNamespace(id=123, status=TaskStatus.COCO_RUNNING.value)
    repo = SimpleNamespace(
        find_active_for_version=AsyncMock(return_value=active_task),
        create=AsyncMock(),
    )
    reply = AsyncMock()
    monkeypatch.setattr(resolve_service, "reply_discussion", reply)
    service = ResolveService(
        ResolveDeps(
            nc=object(),
            coco=object(),
            lark=None,
            repo=repo,
            notification=None,
            settings=None,
        )
    )

    task_id = await service.handle_trigger(
        repo_id="repo-1",
        mr_iid=42,
        author_email=None,
        payload={"source_branch": "source", "target_branch": "target"},
        trigger_type="AUTO_CONFLICT",
    )

    assert task_id == 123
    reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_pipeline_skips_task_without_pending_claim():
    task = SimpleNamespace(id=123, status=TaskStatus.PENDING.value)
    repo = SimpleNamespace(
        get=AsyncMock(return_value=task),
        update_status=AsyncMock(return_value=True),
    )
    service = ResolveService(
        ResolveDeps(
            nc=object(),
            coco=object(),
            lark=None,
            repo=repo,
            notification=None,
            settings=None,
        )
    )

    await service.run_pipeline(task.id)

    repo.update_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_pipeline_dispatches_only_rejected_blocks(monkeypatch):
    rejected = SimpleNamespace(
        id=1,
        block_idx=0,
        conflict_id="C0001",
        file_path="src/a.py",
        ours_text="source-a",
        theirs_text="target-a",
        ai_resolution="old-a",
        approval=SimpleNamespace(status="REJECTED", comment="keep target guard"),
    )
    locked = SimpleNamespace(
        id=2,
        block_idx=1,
        conflict_id="C0002",
        file_path="src/b.py",
        ours_text="source-b",
        theirs_text="target-b",
        ai_resolution="resolved-b",
        approval=SimpleNamespace(status="APPROVED", comment=None),
    )
    task = SimpleNamespace(
        id=123,
        repo_id="repo-1",
        original_mr_id=10,
        mr_version=1,
        source_branch="source",
        source_commit_sha="a" * 40,
        target_branch="target",
        target_commit_sha="b" * 40,
        resolve_branch="resolve/task-123",
        resolve_commit_sha="c" * 40,
        retry_count=1,
        trigger_payload={},
        status=TaskStatus.RETRY_PREPARING.value,
        version=4,
        blocks=[rejected, locked],
    )
    repo = SimpleNamespace(
        get_with_blocks=AsyncMock(return_value=task),
        update_status=AsyncMock(return_value=True),
        add_event=AsyncMock(return_value=True),
    )
    coco = SimpleNamespace(submit_retry_task=AsyncMock(return_value=CocoTaskRef("session-2", "run-2", "event-2")))
    service = ResolveService(
        ResolveDeps(
            nc=object(),
            coco=coco,
            lark=None,
            repo=repo,
            notification=None,
            settings=SimpleNamespace(
                coco_default_agent_id="agent-1",
                coco_workspace_id="workspace-1",
            ),
        )
    )
    monkeypatch.setattr(service, "_verify_retry_baseline", AsyncMock())

    await service.run_retry_pipeline(task.id)

    kwargs = coco.submit_retry_task.await_args.kwargs
    assert kwargs["attempt"] == 2
    assert kwargs["rejected_conflicts"] == [
        {
            "conflict_id": "C0001",
            "path": "src/a.py",
            "source_text": "source-a",
            "target_text": "target-a",
            "previous_resolution": "old-a",
            "reject_reason": "keep target guard",
        }
    ]
    assert kwargs["locked_conflict_ids"] == ["C0002"]
    assert [item.kwargs["new_status"] for item in repo.update_status.await_args_list] == [
        TaskStatus.RETRY_COCO_SUBMITTED,
        TaskStatus.RETRY_COCO_RUNNING,
    ]


@pytest.mark.asyncio
async def test_prepare_creates_resolver_from_source_branch(monkeypatch):
    source_sha = "a" * 40
    target_sha = "b" * 40
    get_mr = AsyncMock(
        return_value=MR(
            id="mr-id",
            iid=10,
            title="conflict",
            description="",
            web_url="https://code.test/mr/10",
            source_repo_id="repo-1",
            target_repo_id="repo-1",
            source_branch="feat/source",
            target_branch="main",
            author_email="author@example.com",
            versions=(MRVersion("v1", 7, source_sha, target_sha, "f" * 40),),
        )
    )
    resolve_branch = "fix/feat-source-to-main-conflict-20260828-t123"
    ensure_branch = AsyncMock(return_value=Branch(resolve_branch, source_sha))
    monkeypatch.setattr(resolve_service, "get_mr", get_mr)
    monkeypatch.setattr(resolve_service, "ensure_branch", ensure_branch)
    repo = SimpleNamespace(
        update_fields=AsyncMock(return_value=True),
        update_status=AsyncMock(return_value=True),
        add_event=AsyncMock(return_value=True),
    )
    nextcode = object()
    service = ResolveService(
        ResolveDeps(
            nc=nextcode,
            coco=None,
            lark=None,
            repo=repo,
            notification=None,
            settings=None,
        )
    )
    task = SimpleNamespace(
        id=123,
        repo_id="repo-1",
        original_mr_id=10,
        source_branch="feat/source",
        target_branch="main",
        resolve_branch="",
        retry_count=0,
        mr_author_email=None,
        status=TaskStatus.PREPARING.value,
        version=1,
        created_at=datetime(2026, 8, 28, 3, 0, tzinfo=UTC),
    )

    await service._step1_prepare(task)

    ensure_branch.assert_awaited_once_with(
        nextcode,
        "repo-1",
        revision=source_sha,
        name=resolve_branch,
    )
    repo.update_fields.assert_awaited_once_with(
        task_id=123,
        from_version=1,
        resolve_branch=resolve_branch,
        source_branch="feat/source",
        target_branch="main",
        mr_version=7,
        source_commit_sha=source_sha,
        target_commit_sha=target_sha,
        resolve_base_sha=source_sha,
        mr_author_email="author@example.com",
        active_key=build_active_key("repo-1", 10, source_sha, target_sha),
    )
    assert repo.update_status.await_args_list == [
        call(
            task_id=123,
            from_version=2,
            new_status=TaskStatus.BRANCH_CREATED,
        ),
    ]
    assert task.resolve_branch == resolve_branch
    assert task.source_commit_sha == source_sha
    assert task.target_commit_sha == target_sha
    assert task.status == TaskStatus.BRANCH_CREATED.value


@pytest.mark.asyncio
async def test_prepare_reuses_persisted_resolver_branch(monkeypatch):
    source_sha = "a" * 40
    target_sha = "b" * 40
    existing_resolve = "fix/persisted-conflict-20260828-t123"
    monkeypatch.setattr(
        resolve_service,
        "get_mr",
        AsyncMock(
            return_value=MR(
                id="mr-id",
                iid=10,
                title="conflict",
                description="",
                web_url="",
                source_repo_id="repo-1",
                target_repo_id="repo-1",
                source_branch="feat/source",
                target_branch="main",
                versions=(MRVersion("v1", 7, source_sha, target_sha, "f" * 40),),
            )
        ),
    )
    ensure_branch = AsyncMock(return_value=Branch(existing_resolve, source_sha))
    monkeypatch.setattr(resolve_service, "ensure_branch", ensure_branch)
    repo = SimpleNamespace(
        update_fields=AsyncMock(return_value=True),
        update_status=AsyncMock(return_value=True),
        add_event=AsyncMock(return_value=True),
    )
    task = SimpleNamespace(
        id=123,
        repo_id="repo-1",
        original_mr_id=10,
        source_branch="feat/source",
        target_branch="main",
        resolve_branch=existing_resolve,
        retry_count=0,
        mr_author_email=None,
        status=TaskStatus.PREPARING.value,
        version=1,
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
    )
    service = ResolveService(
        ResolveDeps(
            nc=object(),
            coco=None,
            lark=None,
            repo=repo,
            notification=None,
            settings=None,
        )
    )

    await service._step1_prepare(task)

    assert task.resolve_branch == existing_resolve
    assert ensure_branch.await_args.kwargs["name"] == existing_resolve


@pytest.mark.asyncio
async def test_identify_conflicts_sorts_files_and_assigns_stable_ids(monkeypatch):
    conflict_files = [
        ConflictFile(
            file_path="z.py",
            first_path="z.py",
            second_path="z.py",
            conflicts_id="conflicts-1",
            content="<<<<<<< z.py\nsource-z\n=======\ntarget-z\n>>>>>>> z.py\n",
        ),
        ConflictFile(
            file_path="a.py",
            first_path="a.py",
            second_path="a.py",
            conflicts_id="conflicts-1",
            content="<<<<<<< a.py\nsource-a\n=======\ntarget-a\n>>>>>>> a.py\n",
        ),
    ]
    monkeypatch.setattr(
        resolve_service,
        "list_conflict_files",
        AsyncMock(return_value=conflict_files),
    )
    repo = SimpleNamespace(
        save_conflict_blocks=AsyncMock(return_value=[1, 2]),
        update_status=AsyncMock(return_value=True),
        add_event=AsyncMock(return_value=True),
    )
    service = ResolveService(
        ResolveDeps(
            nc=object(),
            coco=None,
            lark=None,
            repo=repo,
            notification=None,
            settings=None,
        )
    )
    task = SimpleNamespace(
        id=123,
        repo_id="repo-1",
        original_mr_id=10,
        conflict_files_snapshot=[asdict(item) for item in conflict_files],
        status=TaskStatus.BRANCH_CREATED.value,
        version=0,
    )

    await service._step2_identify_conflicts(task)

    blocks = repo.save_conflict_blocks.await_args.args[1]
    assert [(block["conflict_id"], block["file_path"]) for block in blocks] == [
        ("C0001", "a.py"),
        ("C0002", "z.py"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("branches_moved", [False, True])
async def test_verify_coco_result_reads_back_branch_commit_file_and_blame(monkeypatch, branches_moved):
    source_sha = "a" * 40
    target_sha = "b" * 40
    resolve_sha = "c" * 40
    source_author_sha = "d" * 40
    target_author_sha = "e" * 40
    task = SimpleNamespace(
        id=123,
        repo_id="repo-1",
        source_branch="feat/source",
        target_branch="main",
        resolve_branch="resolve/mr-10/task-123",
        source_commit_sha=source_sha,
        target_commit_sha=target_sha,
        retry_count=0,
        blocks=[
            SimpleNamespace(
                conflict_id="C0001",
                file_path="src/example.py",
            )
        ],
    )
    result = CocoResolveResult(
        schema_version="conciliator.resolve.result.v3",
        task_id=123,
        status="succeeded",
        attempt=1,
        resolve_branch=task.resolve_branch,
        commit_sha=resolve_sha,
        merge_parents=(source_sha, target_sha),
        conflicts=(
            ConflictResolution(
                conflict_id="C0001",
                path="src/example.py",
                decision_summary="保留两侧有效逻辑",

                source_attribution=ConflictAttribution(
                    "src/example.py",
                    10,
                    12,
                    source_author_sha,
                    "source@example.com",
                ),
                target_attribution=ConflictAttribution(
                    "src/example.py",
                    8,
                    11,
                    target_author_sha,
                    "target@example.com",
                ),
                resolved=ResolvedConflict(
                    "src/example.py",
                    1,
                    1,
                    "resolved",
                ),
            ),
        ),

        unresolved_items=(),
    )

    async def fake_get_branch(_client, _repo_id, name):
        heads = {
            task.resolve_branch: resolve_sha,
            task.source_branch: "f" * 40 if branches_moved else source_sha,
            task.target_branch: "f" * 40 if branches_moved else target_sha,
        }
        return Branch(name, heads[name])

    async def fake_get_blame(
        _client,
        _repo_id,
        *,
        revision,
        path,
        start_line,
        end_line,
    ):
        del path
        if revision == source_sha:
            return [
                BlameSegment(
                    start_line,
                    end_line,
                    source_author_sha,
                    "Source",
                    "source@example.com",
                )
            ]
        return [
            BlameSegment(
                start_line,
                end_line,
                target_author_sha,
                "Target",
                "target@example.com",
            )
        ]

    monkeypatch.setattr(resolve_service, "get_branch", fake_get_branch)
    monkeypatch.setattr(
        resolve_service,
        "get_commit",
        AsyncMock(return_value=CommitDetails(resolve_sha, "merge target", (source_sha, target_sha))),
    )
    monkeypatch.setattr(
        resolve_service,
        "get_file",
        AsyncMock(return_value=RepositoryFile("src/example.py", "resolved\n", "text")),
    )
    monkeypatch.setattr(resolve_service, "get_blame", fake_get_blame)
    service = ResolveService(
        ResolveDeps(
            nc=object(),
            coco=None,
            lark=None,
            repo=None,
            notification=None,
            settings=None,
        )
    )

    resolved_files = await service._verify_coco_result(task, result)

    assert resolved_files == {"src/example.py": "resolved\n"}

    with pytest.raises(RuntimeError, match="parents"):
        await service._verify_coco_result(task, replace(result, merge_parents=(target_sha, source_sha)))

    invalid_conflict = replace(
        result.conflicts[0],
        resolved=replace(result.conflicts[0].resolved, end_line=2),
    )
    assert await service._verify_coco_result(
        task, replace(result, conflicts=(invalid_conflict,)),
    ) == resolved_files



@pytest.mark.asyncio
async def test_document_step_uploads_markdown_and_persists_canonical_url(
    monkeypatch,
):
    resolve_result = CocoResolveResult(
        schema_version="conciliator.resolve.result.v3",
        task_id=123,
        status="succeeded",
        attempt=1,
        resolve_branch="resolve/mr-10/task-123",
        commit_sha="c" * 40,
        merge_parents=("a" * 40, "b" * 40),
        conflicts=(),

        unresolved_items=(),
    )
    result = CocoTaskResult(status="succeeded", resolve_result=resolve_result)
    repo = SimpleNamespace(
        update_status=AsyncMock(return_value=True),
        add_event=AsyncMock(return_value=True),
    )
    service = ResolveService(
        ResolveDeps(
            nc=object(),
            coco=object(),
            lark=object(),
            repo=repo,
            notification=None,
            settings=SimpleNamespace(lark_markdown_folder_token="folder-token"),
        )
    )
    monkeypatch.setattr(
        service,
        "_prepare_report_data",
        AsyncMock(
            return_value=(
                [
                    {
                        "idx": 1,
                        "conflict_id": "C0001",
                        "file_path": "src/example.py",
                        "start_line": 1,
                        "end_line": 1,
                    }
                ],
                {"src/example.py": "resolved\n"},
            )
        ),
    )
    upload = AsyncMock(
        return_value=MarkdownUpload(
            file_token="file-token",
            file_name="conciliator-mr-10-task-123.md",
            size_bytes=123,
            web_url="https://bytedance.larkoffice.com/file/file-token",
        )
    )
    monkeypatch.setattr(resolve_service, "upload_markdown", upload)
    task = SimpleNamespace(
        id=123,
        original_mr_id=10,
        source_branch="feat/source",
        target_branch="main",
        resolve_branch="resolve/mr-10/task-123",
        source_commit_sha="a" * 40,
        target_commit_sha="b" * 40,
        status=TaskStatus.COCO_RUNNING.value,
        version=4,
    )

    await service._step4_document(task, result)

    assert upload.await_args.kwargs["file_name"] == ("conciliator-mr-10-task-123.md")
    assert upload.await_args.kwargs["folder_token"] == "folder-token"
    assert "# MR 自动解冲突报告" in (upload.await_args.kwargs["content"].decode("utf-8"))
    repo.update_status.assert_awaited_once_with(
        task_id=123,
        from_version=4,
        new_status=TaskStatus.DOC_CREATED,
        lark_doc_url="https://bytedance.larkoffice.com/file/file-token",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("long_comment", [False, True])
@pytest.mark.parametrize(
    ("block_order", "automatic_idx", "expected_order"),
    [
        ([0], None, [0]),
        ([0, 1, 2], None, [0, 1, 2]),
        ([2, 0, 1], None, [0, 1, 2]),
        ([2, 0, 3, 1], 1, [0, 2, 3]),
    ],
    ids=["single", "ordered", "unordered", "unordered-with-automatic-block"],
)
async def test_mr_step_publishes_verified_positions_without_waiting_for_merge_checks(
    monkeypatch, long_comment, block_order, automatic_idx, expected_order,
):
    version = MRVersion(
        "version-1",
        1,
        "c" * 40,
        "b" * 40,
        "b" * 40,
    )
    resolver_mr = MR(
        id="mr-internal",
        iid=99,
        title="[resolve task-123] conflict",
        description="",
        web_url="https://code.test/mr/99",
        source_repo_id="repo-1",
        target_repo_id="repo-1",
        source_branch="resolve/mr-10/task-123",
        target_branch="main",
        versions=(version,),
    )
    block = SimpleNamespace(
        id=11,
        block_idx=0,
        conflict_id="C0001",
        file_path="src/example.py",
        start_line=20,
        end_line=25,
        source_author_email="source@example.com",
        source_path="src/example.py",
        source_start_line=10,
        source_end_line=12,
        target_author_email="target@example.com",
        target_path="src/example.py",
        target_start_line=8,
        target_end_line=11,
        author_email="source@example.com",
        decision_summary="保留两侧逻辑",

        resolved_path="src/example.py",
        resolved_start_line=14,
        resolved_end_line=18,
        ours_text="source()",
        theirs_text="target()",
        ai_resolution="resolved()" * (1000 if long_comment else 1),
    )
    blocks = [
        SimpleNamespace(**{
            **vars(block),
            "id": 11 + idx,
            "block_idx": idx,
            "conflict_id": f"C{idx + 1:04d}",
            "approval": SimpleNamespace(status="NOT_REQUIRED" if idx == automatic_idx else "PENDING"),
        })
        for idx in block_order
    ]
    repo = SimpleNamespace(
        get_with_blocks=AsyncMock(return_value=SimpleNamespace(blocks=blocks)),
        update_status=AsyncMock(return_value=True),
        update_block_discussion_and_resolution=AsyncMock(),
        add_event=AsyncMock(return_value=True),
    )
    notification = SimpleNamespace(notify_mr_comment=AsyncMock())
    nextcode = SimpleNamespace(
        call=AsyncMock(return_value={
            "Mergeability": {"Mergeable": False, "UnmergeableReason": "checks_pending"}
        })
    )
    service = ResolveService(
        ResolveDeps(
            nc=nextcode,
            coco=object(),
            lark=object(),
            repo=repo,
            notification=notification,
            settings=SimpleNamespace(),
        )
    )
    task = SimpleNamespace(
        id=123,
        repo_id="repo-1",
        original_mr_id=10,
        source_branch="feat/source",
        target_branch="main",
        resolve_branch="resolve/mr-10/task-123",
        source_commit_sha="a" * 40,
        target_commit_sha="b" * 40,
        resolve_commit_sha="c" * 40,
        lark_doc_url="https://bytedance.larkoffice.com/file/file-token",
        coco_result_json={"tests": []},
        trigger_payload={"mr_title": "conflict"},
        trigger_type="COMMENT",
        new_mr_id=None,
        status=TaskStatus.DOC_CREATED.value,
        version=5,
    )
    create_or_recover = AsyncMock(return_value=resolver_mr)
    monkeypatch.setattr(
        service,
        "_create_or_recover_resolve_mr",
        create_or_recover,
    )

    async def fake_get_mr(client, repo_id, mr_iid):
        description = create_or_recover.await_args.kwargs["description"]
        return MR(
            **{
                **resolver_mr.__dict__,
                "description": description,
            }
        )

    monkeypatch.setattr(resolve_service, "get_mr", fake_get_mr)
    create_discussion = AsyncMock(return_value=Discussion("thread-1", "comment-1"))
    monkeypatch.setattr(
        resolve_service,
        "create_discussion",
        create_discussion,
    )
    monkeypatch.setattr(
        resolve_service,
        "get_thread",
        AsyncMock(
            return_value=Thread(
                "thread-1",
                "open",
                (
                    {
                        "OldCommitId": "b" * 40,
                        "NewCommitId": "c" * 40,
                        "Path": "src/example.py",
                        "StartLine": 14,
                        "EndLine": 18,
                    },
                ),
            )
        ),
    )

    await service._step5_create_mr_and_discussions(task)

    nextcode.call.assert_not_awaited()
    assert create_discussion.await_count == len(expected_order)
    for created, idx in zip(create_discussion.await_args_list, expected_order, strict=True):
        assert created.kwargs["file_path"] == "src/example.py"
        assert created.kwargs["line"] == 14
        assert created.kwargs["end_line"] == 18
        body = created.kwargs["body"]
        assert f"conciliator:task=123:block=C{idx + 1:04d}" in body
        assert len(body) <= 6000
        if long_comment:
            assert f"[飞书文档链接]({task.lark_doc_url})" in body
            assert "resolved()" not in body
    assert [
        updated.kwargs["block_id"]
        for updated in repo.update_block_discussion_and_resolution.await_args_list
    ] == [11 + idx for idx in expected_order]
    assert task.new_mr_url == "https://code.test/mr/99"
    assert task.status == TaskStatus.AWAITING_APPROVAL.value


@pytest.mark.asyncio
async def test_requires_action_fails_task_immediately():
    task = SimpleNamespace(
        id=123,
        coco_task_id="session-1",
        coco_last_event_id="event-1",
        status=TaskStatus.COCO_RUNNING.value,
        version=4,
    )
    repo = SimpleNamespace(
        get=AsyncMock(return_value=task),
        update_status=AsyncMock(return_value=True),
    )
    coco = SimpleNamespace(
        fetch_task_result=AsyncMock(
            return_value=CocoTaskResult(
                status="requires_action",
                error="Agent is waiting for tool confirmation",
                last_event_id="event-9",
            )
        )
    )
    service = ResolveService(
        ResolveDeps(
            nc=None,
            coco=coco,
            lark=None,
            repo=repo,
            notification=None,
            settings=None,
        )
    )

    result = await service.continue_after_coco(123)

    assert result is None
    repo.update_status.assert_has_awaits([call(
        task_id=123, from_version=4, from_status=TaskStatus.COCO_RUNNING, new_status=TaskStatus.COCO_PROCESSING,
    ), call(
        task_id=123,
        from_version=5,
        new_status=TaskStatus.FAILED,
        last_error="Coco Agent PermissionMode 配置错误",
        coco_last_event_id="event-9",
    )])
    assert task.status == TaskStatus.FAILED.value


@pytest.mark.asyncio
@pytest.mark.parametrize("long_comment", [False, True])
async def test_publish_retry_reuses_mr_and_discussion_then_resets_subset(
    monkeypatch, long_comment,
):
    rejected = SimpleNamespace(
        id=1,
        block_idx=0,
        conflict_id="C0001",
        file_path="src/a.py",
        start_line=10,
        end_line=12,
        source_author_email="source@example.com",
        target_author_email="target@example.com",
        author_email="source@example.com",
        resolved_path="src/a.py",
        resolved_start_line=20,
        resolved_end_line=22,
        discussion_id="thread-1",
        approval=SimpleNamespace(
            status="REJECTED",
            comment="保留 target guard",
        ),
    )
    approved = SimpleNamespace(
        id=2,
        block_idx=1,
        conflict_id="C0002",
        file_path="src/b.py",
        start_line=2,
        end_line=3,
        source_author_email="source@example.com",
        target_author_email="target@example.com",
        author_email="source@example.com",
        resolved_path="src/b.py",
        resolved_start_line=4,
        resolved_end_line=5,
        discussion_id="thread-2",
        approval=SimpleNamespace(status="APPROVED", comment=None),
    )
    task = SimpleNamespace(
        id=123,
        repo_id="repo-1",
        original_mr_id=10,
        new_mr_id=99,
        source_branch="source",
        source_commit_sha="a" * 40,
        target_branch="target",
        target_commit_sha="b" * 40,
        resolve_branch="resolve/task-123",
        resolve_commit_sha="d" * 40,
        lark_doc_url="https://lark.test/file/new",
        coco_result_json={"tests": [{"command": "pytest -q", "exit_code": 0, "summary": "ok"}]},
        retry_count=1,
        status=TaskStatus.RETRY_DOC_UPLOADED.value,
        version=8,
        blocks=[rejected, approved],
    )
    repo = SimpleNamespace(
        update_status=AsyncMock(return_value=True),
        get_with_blocks=AsyncMock(return_value=task),
        reset_rejected_approvals=AsyncMock(return_value=1),
        list_events=AsyncMock(return_value=[]),
        add_event=AsyncMock(return_value=True),
    )
    result = CocoRetryResult(
        schema_version="conciliator.resolve.retry.result.v2",
        task_id=123,
        status="succeeded",
        attempt=2,
        resolve_branch=task.resolve_branch,
        base_commit_sha="c" * 40,
        commit_sha="d" * 40,
        parent_commits=("c" * 40,),
        conflicts=(
            ConflictResolution(
                conflict_id="C0001",
                path="src/a.py",
                decision_summary="按反馈保留 guard",

                source_attribution=ConflictAttribution("src/a.py", 1, 1, "e" * 40, "source@example.com"),
                target_attribution=ConflictAttribution("src/a.py", 1, 1, "f" * 40, "target@example.com"),
                resolved=ResolvedConflict("src/a.py", 20, 22, "new-a" * (2000 if long_comment else 1)),
            ),
        ),

        unresolved_items=(),
    )
    reply = AsyncMock(return_value=Discussion("thread-1", "comment-retry-1"))
    descriptions: list[str] = []

    async def fake_update(client, repo_id, mr_iid, description):
        descriptions.append(description)

    async def fake_get(client, repo_id, mr_iid):
        return SimpleNamespace(
            description=descriptions[-1],
            source_branch=task.resolve_branch,
            target_branch=task.target_branch,
            latest_version=SimpleNamespace(source_commit_id=task.resolve_commit_sha),
        )

    monkeypatch.setattr(resolve_service, "reply_discussion", reply)
    monkeypatch.setattr(
        resolve_service,
        "get_thread",
        AsyncMock(return_value=Thread("thread-1", "open", (), ())),
    )
    monkeypatch.setattr(resolve_service, "update_mr_description", fake_update)
    monkeypatch.setattr(resolve_service, "get_mr", fake_get)
    service = ResolveService(
        ResolveDeps(
            nc=object(),
            coco=object(),
            lark=object(),
            repo=repo,
            notification=None,
            settings=SimpleNamespace(),
        )
    )

    await service._publish_retry(task, [rejected], result)

    assert reply.await_args.args[3] == "thread-1"
    body = reply.await_args.args[4]
    assert len(body) <= 6000
    if long_comment:
        assert f"[飞书文档链接]({task.lark_doc_url})" in body
        assert "source@example.com · `src/a.py:1-1`" in body
        assert "new-a" not in body
    else:
        assert "保留 target guard" in body
    assert "conciliator:task=123:block=C0001:attempt=2" in reply.await_args.args[4]
    assert "1 approved / 0 rejected / 1 pending" in descriptions[-1]
    repo.reset_rejected_approvals.assert_awaited_once_with(
        task_id=123,
        conflict_ids=("C0001",),
    )
    assert [item.kwargs["new_status"] for item in repo.update_status.await_args_list] == [
        TaskStatus.RETRY_PUBLISHING,
        TaskStatus.AWAITING_APPROVAL,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("create_fails", [False, True])
async def test_task_created_feedback_is_sent_only_after_database_success(monkeypatch, create_fails):
    _mock_trigger_snapshot(monkeypatch)
    feedback = AsyncMock()
    monkeypatch.setattr(resolve_service, "send_resolve_feedback", feedback)

    async def create(**kwargs):
        feedback.assert_not_awaited()
        assert kwargs["trigger_payload"]["_trigger_mr_id"] == "mr-id"
        if create_fails:
            raise RuntimeError("database unavailable")
        return SimpleNamespace(id=7)

    repo = SimpleNamespace(find_active_for_version=AsyncMock(return_value=None), create=AsyncMock(side_effect=create))
    service = ResolveService(SimpleNamespace(repo=repo, nc=object()))
    args = dict(
        repo_id="repo-1", mr_iid=42, author_email="user@example.com", payload={},
        trigger_type="COMMENT", discussion_id="original-thread",
    )
    if create_fails:
        with pytest.raises(RuntimeError, match="database unavailable"):
            await service.handle_trigger(**args)
        feedback.assert_not_awaited()
    else:
        assert await service.handle_trigger(**args) == 7
        assert "解冲突任务 7 已受理" in feedback.await_args.kwargs["body"]
        assert "https://aeolus-conciliator.byted.org/trace?task_id=7&order=desc" in feedback.await_args.kwargs["body"]
        assert feedback.await_args.kwargs["thread_id"] == "original-thread"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 422])
async def test_blame_error_logs_exact_request_and_preserves_error_handling(monkeypatch, status):
    from structlog.testing import capture_logs

    from conciliator.integrations.nextcode.client import NextCodeError

    error = NextCodeError(
        status=status, action="GetBlame", code="Unprocessable", request_id="request-123",
        message="invalid blame format; token=private-value",
    )
    get_blame = AsyncMock(side_effect=error)
    monkeypatch.setattr(resolve_service, "get_blame", get_blame)
    service = ResolveService(SimpleNamespace(nc=object()))
    task = SimpleNamespace(id=29, repo_id="repo-1")
    attribution = ConflictAttribution("aeolus/app.py", 164, 165, "attribution-commit", "author@example.com")
    expected = resolve_service.CocoValidationError if status == 404 else NextCodeError
    with capture_logs() as logs, pytest.raises(expected):
        await service._verify_attribution(task, attribution, revision="frozen-source-sha", side="source")
    record = next(record for record in logs if record["event"] == "resolve_blame_failed")
    assert record["task_id"] == 29 and record["side"] == "source"
    assert record["request_body"] == {
        "RepoId": "repo-1", "Revision": "frozen-source-sha", "Path": "aeolus/app.py",
        "StartLine": 164, "EndLine": 165,
    }
    assert record["request_id"] == "request-123" and record["http_status"] == status
    assert record["error_code"] == "Unprocessable"
    assert "private-value" not in str(record)
    assert "invalid blame format" in record["error_message"]
