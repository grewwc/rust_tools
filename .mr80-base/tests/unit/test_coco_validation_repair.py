import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Integer, create_engine
from sqlalchemy.orm import sessionmaker

from conciliator.domain.task import TaskStatus
from conciliator.integrations.coco.client import (
    CocoClient,
    CocoEvent,
    CocoEventPage,
    CocoResultError,
    CocoTaskResult,
    parse_resolve_result,
)
from conciliator.integrations.coco.prompts import repair_prompt
from conciliator.integrations.nextcode.branches import Branch
from conciliator.integrations.nextcode.commits import CommitDetails
from conciliator.integrations.nextcode.discussions import Discussion, Thread
from conciliator.integrations.nextcode.files import RepositoryFile
from conciliator.integrations.nextcode.repos import BlameSegment
from conciliator.repositories.models import Base
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import resolve_service
from conciliator.services.resolve_service import CocoValidationError, ResolveDeps, ResolveService
from tests.unit.test_coco_client import _resolve_result, _resolve_result_message, _secrets, _settings


@pytest.fixture
def repository(tmp_path, monkeypatch):
    for table in Base.metadata.tables.values():
        if "id" in table.c and isinstance(table.c.id.type, Integer):
            monkeypatch.setattr(table.c.id, "type", Integer())
    engine = create_engine(f"sqlite:///{tmp_path / 'repair.db'}")
    Base.metadata.create_all(engine)
    yield TaskRepository(sessionmaker(bind=engine, expire_on_commit=False))
    engine.dispose()


def event(event_id="u2", run_id="r2", kind="UserMessage", text=""):
    payload = {kind: {"Content": [{"Text": {"Text": text}}], "Delta": True}}
    return CocoEvent(event_id, "s1", run_id, payload, {})


async def setup_task(repo, monkeypatch, *, retry=False):
    task = await repo.create(
        repo_id="repo",
        original_mr_id=42,
        source_branch="source",
        target_branch="target",
        resolve_branch="resolver",
        trigger_type="COMMENT",
        trigger_author_email="user@example.com",
        trigger_payload={
            "_trigger_discussion_id": "original-resolve-thread",
            "_coco_round": {"session_id": "s1", "run_id": "r1", "event_id": "u1"},
        },
        mr_version=1,
        source_commit_sha="a" * 40,
        target_commit_sha="b" * 40,
    )
    await repo.update_status(
        task_id=task.id,
        from_version=task.version,
        new_status=TaskStatus.RETRY_COCO_RUNNING if retry else TaskStatus.COCO_RUNNING,
        coco_task_id="s1",
        coco_session_id="s1",
        coco_run_id="r1",
        coco_workspace_id="w1",
        retry_count=1 if retry else 0,
        resolve_commit_sha="f" * 40 if retry else None,
    )
    await repo.save_conflict_blocks(
        task.id,
        [
            {
                "conflict_id": "C0001",
                "block_idx": 0,
                "file_path": "src/example.py",
                "start_line": 1,
                "end_line": 7,
                "ours_text": "source",
                "theirs_text": "target",
                "author_email": None,
            }
        ],
    )
    parents = ("f" * 40,) if retry else ("a" * 40, "b" * 40)
    monkeypatch.setattr(resolve_service, "get_branch", AsyncMock(return_value=Branch("resolver", "c" * 40)))
    monkeypatch.setattr(
        resolve_service, "get_commit", AsyncMock(return_value=CommitDetails("c" * 40, "merge", parents))
    )
    monkeypatch.setattr(
        resolve_service, "get_file", AsyncMock(return_value=RepositoryFile("src/example.py", "resolved\n", "text"))
    )

    async def blame(*args, revision, start_line, end_line, **kwargs):
        source = revision == "a" * 40
        return [
            BlameSegment(
                start_line,
                end_line,
                ("d" if source else "e") * 40,
                "author",
                ("source" if source else "target") + "@example.com",
            )
        ]

    monkeypatch.setattr(resolve_service, "get_blame", blame)
    monkeypatch.setattr(
        resolve_service, "get_thread", AsyncMock(return_value=Thread("original-resolve-thread", "open", ()))
    )
    monkeypatch.setattr(
        resolve_service, "reply_discussion", AsyncMock(return_value=Discussion("original-resolve-thread", "notice1"))
    )
    coco = SimpleNamespace(
        send_user_message=AsyncMock(return_value=event()),
        fetch_task_result=AsyncMock(),
        fetch_retry_task_result=AsyncMock(),
        read_round_events=AsyncMock(return_value=[]),
    )
    svc = ResolveService(
        ResolveDeps(nc=object(), coco=coco, lark=None, repo=repo, notification=None, settings=SimpleNamespace())
    )
    return await repo.get(task.id), svc, coco


@pytest.mark.asyncio
async def test_invalid_result_repairs_once_in_same_session_then_replies_to_trigger(repository, monkeypatch):
    task, svc, coco = await setup_task(repository, monkeypatch)
    coco.fetch_task_result.return_value = CocoTaskResult(status="invalid_result", error="missing C0001")
    await svc.continue_after_coco(task.id)
    mid = await repository.get(task.id)
    assert mid.status == "COCO_RUNNING" and mid.coco_session_id == "s1" and mid.coco_run_id == "r2"
    assert mid.trigger_payload["_coco_round"]["event_id"] == "u2"
    send = coco.send_user_message.await_args.kwargs
    assert send["session_id"] == "s1" and "missing C0001" in send["text"]
    assert "唯一一次" in send["text"] and '"repair_base_sha": "cccc' in send["text"]
    await svc.continue_after_coco(task.id)
    final = await repository.get(task.id)
    assert final.status == "FAILED" and final.active_key is None
    assert coco.send_user_message.await_count == 1
    assert final.trigger_payload["_failure_notice"]["state"] == "sent"
    args = resolve_service.reply_discussion.await_args.args
    assert args[1:4] == ("repo", 42, "original-resolve-thread")
    assert "FAILED" in args[4] and "missing C0001" in args[4]
    assert coco.fetch_task_result.await_args.kwargs["after_event_id"] == "u2"
    await svc.continue_after_coco(task.id)
    await svc.notify_validation_failure(task.id)
    assert resolve_service.reply_discussion.await_count == 1


@pytest.mark.asyncio
async def test_repair_success_allows_report_text_mismatch_and_binds_metadata(repository, monkeypatch, tmp_path):
    task, svc, coco = await setup_task(repository, monkeypatch)
    wire = _resolve_result()
    wire["conflicts"][0]["resolved"].update(start_line=90, end_line=92, text="Coco report snippet")
    coco.fetch_task_result.side_effect = [
        CocoTaskResult(status="invalid_result", error="bad JSON"),
        CocoTaskResult(status="succeeded", resolve_result=parse_resolve_result(_resolve_result_message(wire))),
    ]
    await svc.continue_after_coco(task.id, local_report_dir=tmp_path)
    artifact = await svc.continue_after_coco(task.id, local_report_dir=tmp_path)
    full = await repository.get_with_blocks(task.id)
    assert artifact and "Coco report snippet" in artifact.preview_path.read_text()
    assert full.status == "DOC_CREATED" and full.blocks[0].resolved_start_line == 90
    assert full.coco_result_json["task_id"] == task.id
    assert full.coco_result_json["attempt"] == 1
    assert "tests" not in full.coco_result_json
    assert "introduced_fix" not in full.coco_result_json["conflicts"][0]
    assert "COCO_REPORT_WARNING" in [e.event_type for e in await repository.list_events(task.id)]
    resolve_service.reply_discussion.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_validation_failures_claim_only_one_send(repository, monkeypatch):
    task, svc, coco = await setup_task(repository, monkeypatch)
    copies = await asyncio.gather(*(repository.get(task.id) for _ in range(8)))
    await asyncio.gather(*(svc._repair_or_fail(t, "bad", retry=False) for t in copies))
    assert coco.send_user_message.await_count == 1


@pytest.mark.asyncio
async def test_uncertain_send_recovers_by_message_marker_without_resend(repository, monkeypatch):
    task, svc, coco = await setup_task(repository, monkeypatch)

    async def uncertain(**kwargs):
        coco.read_round_events.return_value = [event(text=kwargs["text"])]
        raise TimeoutError("response lost")

    coco.send_user_message.side_effect = uncertain
    await svc._repair_or_fail(task, "bad", retry=False)
    persisted = await repository.get(task.id)
    assert persisted.trigger_payload["_coco_validation_repair"]["phase"] == "sending"
    assert await svc._recover_repair(persisted)
    assert (await repository.get(task.id)).coco_run_id == "r2"
    assert coco.send_user_message.await_count == 1


@pytest.mark.asyncio
async def test_failed_notification_recovers_after_crash_without_duplicate(repository, monkeypatch):
    task, svc, _ = await setup_task(repository, monkeypatch)

    async def written_then_timeout(*args):
        monkeypatch.setattr(
            resolve_service,
            "get_thread",
            AsyncMock(
                return_value=Thread("original-resolve-thread", "open", (), ({"Id": "written", "Content": args[-1]},))
            ),
        )
        raise TimeoutError("reply written, response lost")

    monkeypatch.setattr(resolve_service, "reply_discussion", AsyncMock(side_effect=written_then_timeout))
    await svc._fail_validation(task, "still has markers")
    pending = await repository.list_pending_failure_notices()
    assert [t.id for t in pending] == [task.id]
    stale = pending[0]
    payload = {
        **stale.trigger_payload,
        "_failure_notice": {**stale.trigger_payload["_failure_notice"], "started_at": 0},
    }
    await repository.update_fields(task_id=task.id, from_version=stale.version, trigger_payload=payload)
    await svc.notify_validation_failure(task.id)
    assert resolve_service.reply_discussion.await_count == 1
    assert not await repository.list_pending_failure_notices()


@pytest.mark.asyncio
async def test_cancelled_task_does_not_send_or_fail(repository, monkeypatch):
    task, svc, coco = await setup_task(repository, monkeypatch)
    await repository.update_status(task_id=task.id, from_version=task.version, new_status=TaskStatus.CANCELLED)
    await svc._repair_or_fail(task, "bad", retry=False)
    await svc._fail_validation(task, "bad")
    assert (await repository.get(task.id)).status == "CANCELLED"
    coco.send_user_message.assert_not_awaited()
    resolve_service.reply_discussion.assert_not_awaited()


@pytest.mark.asyncio
async def test_repair_accepts_one_child_of_verified_base_but_not_unknown_parent(repository, monkeypatch):
    task, svc, _ = await setup_task(repository, monkeypatch)
    await svc._repair_or_fail(task, "markers", retry=False)
    full = await repository.get_with_blocks(task.id)
    parsed = replace(svc._bind_result(full, parse_resolve_result(_resolve_result_message())), commit_sha="9" * 40)
    monkeypatch.setattr(resolve_service, "get_branch", AsyncMock(return_value=Branch("resolver", "9" * 40)))
    monkeypatch.setattr(
        resolve_service, "get_commit", AsyncMock(return_value=CommitDetails("9" * 40, "fix", ("c" * 40,)))
    )
    await svc._verify_coco_result(full, parsed)
    monkeypatch.setattr(
        resolve_service, "get_commit", AsyncMock(return_value=CommitDetails("9" * 40, "unknown", ("e" * 40,)))
    )
    with pytest.raises(CocoValidationError, match="parents"):
        await svc._verify_coco_result(full, parsed)


@pytest.mark.asyncio
async def test_retry_validation_uses_same_session_and_single_budget(repository, monkeypatch):
    task, svc, coco = await setup_task(repository, monkeypatch, retry=True)
    monkeypatch.setattr(
        svc,
        "_retry_blocks_for_attempt",
        AsyncMock(return_value=((await repository.get_with_blocks(task.id)).blocks, [])),
    )
    coco.fetch_retry_task_result.return_value = CocoTaskResult(status="invalid_result", error="retry JSON invalid")
    await svc.continue_after_retry_coco(task.id)
    await svc.continue_after_retry_coco(task.id)
    assert coco.send_user_message.await_count == 1
    assert coco.send_user_message.await_args.kwargs["session_id"] == "s1"
    assert (await repository.get(task.id)).status == "FAILED"
    assert resolve_service.reply_discussion.await_count == 1


def test_empty_side_contract_and_repair_prompt():
    wire = _resolve_result()
    wire["conflicts"][0]["source_attribution"].update(author_email=None, commit_sha="", start_line=0, end_line=0)
    wire["conflicts"][0]["resolved"]["text"] = ""
    assert parse_resolve_result(_resolve_result_message(wire)).conflicts[0].source_attribution.author_email is None
    wire["tests"] = []
    with pytest.raises(CocoResultError, match="keys mismatch"):
        parse_resolve_result(_resolve_result_message(wire))
    text = repair_prompt(marker="repair-1", errors=["C0001 missing"], context={"source_commit": "a" * 40}, retry=False)
    assert "修正输出即可，不生成多余提交" in text
    assert "不得覆盖他人后续提交或 force push" in text
    assert '"tests":' not in text and '"introduced_fix":' not in text


@pytest.mark.asyncio
async def test_round_reader_ignores_prior_output_and_waits_for_current_idle():
    client = CocoClient(_settings(), _secrets())
    final = _resolve_result_message()
    client.read_round_events = AsyncMock(
        return_value=[
            event("old", "r1", "AgentMessage", "OLD JSON"),
            event("old-idle", "r1", "SessionStatusIdle"),
            event("new", "r2", "AgentMessage", final),
        ]
    )
    client.list_session_runs = AsyncMock(return_value=[{"Id": "r1", "Status": "idle"}, {"Id": "r2", "Status": "idle"}])
    result = await client.fetch_task_result("s1", run_id="r2", after_event_id="u2")
    assert result.status == "running"
    client.read_round_events.return_value.append(event("new-idle", "r2", "SessionStatusIdle"))
    result = await client.fetch_task_result("s1", run_id="r2", after_event_id="u2")
    assert result.status == "succeeded" and result.final_message == final
    await client.aclose()


@pytest.mark.asyncio
async def test_round_reader_drains_pages():
    client = CocoClient(_settings(), _secrets())
    client.list_session_event_page = AsyncMock(side_effect=[
        CocoEventPage([event(str(i)) for i in range(200)], "199"),
        CocoEventPage([event("200", kind="SessionStatusIdle")], "200"),
        CocoEventPage([], ""),
    ])
    events = await client.read_round_events(workspace_id="w", session_id="s1", after_event_id="u1")
    assert len(events) == 201
    assert client.list_session_event_page.await_args.kwargs["after_event_id"] == "200"
    await client.aclose()


@pytest.mark.asyncio
async def test_dependency_outage_does_not_consume_repair_budget(repository, monkeypatch):
    from conciliator.utils.retry import TransientError

    task, svc, coco = await setup_task(repository, monkeypatch)
    coco.fetch_task_result.side_effect = TransientError("read timeout")
    await svc.continue_after_coco(task.id)
    current = await repository.get(task.id)
    assert current.status == "COCO_RUNNING"
    assert "_coco_validation_repair" not in current.trigger_payload
    coco.send_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfirmed_send_fails_without_second_send(repository, monkeypatch):
    task, svc, coco = await setup_task(repository, monkeypatch)
    coco.send_user_message.side_effect = TimeoutError("uncertain")
    await svc._repair_or_fail(task, "bad JSON", retry=False)
    current = await repository.get(task.id)
    payload = {
        **current.trigger_payload,
        "_coco_validation_repair": {**current.trigger_payload["_coco_validation_repair"], "started_at": 0},
    }
    await repository.update_fields(task_id=task.id, from_version=current.version, trigger_payload=payload)
    await svc.continue_after_coco(task.id)
    assert (await repository.get(task.id)).status == "FAILED"
    assert coco.send_user_message.await_count == 1
    assert resolve_service.reply_discussion.await_count == 1


@pytest.mark.asyncio
async def test_actual_child_parents_are_persisted_after_repair(repository, monkeypatch, tmp_path):
    task, svc, coco = await setup_task(repository, monkeypatch)
    await svc._repair_or_fail(task, "markers", retry=False)
    parsed = replace(parse_resolve_result(_resolve_result_message()), commit_sha="9" * 40)
    coco.fetch_task_result.return_value = CocoTaskResult(status="succeeded", resolve_result=parsed)
    monkeypatch.setattr(resolve_service, "get_branch", AsyncMock(return_value=Branch("resolver", "9" * 40)))
    monkeypatch.setattr(
        resolve_service, "get_commit", AsyncMock(return_value=CommitDetails("9" * 40, "fix", ("c" * 40,)))
    )
    artifact = await svc.continue_after_coco(task.id, local_report_dir=tmp_path)
    stored = (await repository.get(task.id)).coco_result_json
    assert stored["merge_parents"] == ["c" * 40]
    assert json.loads(artifact.coco_result_path.read_text())["merge_parents"] == ["c" * 40]


@pytest.mark.asyncio
async def test_empty_source_skips_blame_and_retains_null_author(repository, monkeypatch):
    task, svc, _ = await setup_task(repository, monkeypatch)
    full = await repository.get_with_blocks(task.id)
    full.blocks[0].ours_text = ""
    wire = _resolve_result()
    wire["conflicts"][0]["source_attribution"].update(author_email=None, commit_sha="", start_line=0, end_line=0)
    parsed = svc._bind_result(full, parse_resolve_result(_resolve_result_message(wire)))
    spy = AsyncMock(return_value=[BlameSegment(8, 11, "e" * 40, "target", "target@example.com")])
    monkeypatch.setattr(resolve_service, "get_blame", spy)
    await svc._verify_coco_result(full, parsed)
    assert spy.await_count == 1
    assert spy.await_args.kwargs["revision"] == "b" * 40


@pytest.mark.asyncio
async def test_cancelled_current_run_is_not_hidden_by_idle_event():
    client = CocoClient(_settings(), _secrets())
    client.read_round_events = AsyncMock(
        return_value=[
            event("a", "r2", "AgentMessage", _resolve_result_message()),
            event("b", "r2", "SessionStatusIdle"),
        ]
    )
    client.list_session_runs = AsyncMock(return_value=[{"Id": "r2", "Status": "cancelled"}])
    result = await client.fetch_task_result("s1", run_id="r2", after_event_id="u2")
    assert result.status == "failed"
    await client.aclose()


def test_markers_are_detected_as_lines_and_old_protocol_rejected():
    ResolveService._check_conflict_markers({"a.py": 'pattern = "======="\n'})
    with pytest.raises(CocoValidationError, match="a.py:2"):
        ResolveService._check_conflict_markers({"a.py": "first\n<<<<<<< HEAD\n"})
    wire = _resolve_result()
    wire["schema_version"] = "conciliator.resolve.result.v2"
    with pytest.raises(CocoResultError, match="unsupported"):
        parse_resolve_result(_resolve_result_message(wire))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["markers", "head", "author", "missing_commit"])
async def test_blocking_readback_failure_dispatches_repair(repository, monkeypatch, tmp_path, failure):
    from conciliator.integrations.nextcode.client import NextCodeError

    task, svc, coco = await setup_task(repository, monkeypatch)
    coco.fetch_task_result.return_value = CocoTaskResult(
        status="succeeded", resolve_result=parse_resolve_result(_resolve_result_message())
    )
    if failure == "markers":
        monkeypatch.setattr(
            resolve_service,
            "get_file",
            AsyncMock(
                return_value=RepositoryFile(
                    "src/example.py", "<<<<<<< HEAD\nsource\n=======\ntarget\n>>>>>>> target\n", "text"
                )
            ),
        )
    elif failure == "head":
        monkeypatch.setattr(resolve_service, "get_branch", AsyncMock(return_value=Branch("resolver", "a" * 40)))
    elif failure == "author":
        monkeypatch.setattr(resolve_service, "get_blame", AsyncMock(return_value=[]))
    else:
        monkeypatch.setattr(
            resolve_service,
            "get_commit",
            AsyncMock(
                side_effect=NextCodeError(status=404, action="GetCommit", message="missing", code="NotFound.Commit")
            ),
        )
    assert await svc.continue_after_coco(task.id, local_report_dir=tmp_path) is None
    current = await repository.get(task.id)
    assert current.status == "COCO_RUNNING"
    assert current.trigger_payload["_coco_validation_repair"]["phase"] == "running"
    assert coco.send_user_message.await_count == 1
    assert not (tmp_path / f"task-{task.id}" / "lark-document-preview.md").exists()
