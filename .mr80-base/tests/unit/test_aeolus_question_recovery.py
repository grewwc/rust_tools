"""Question send/recovery boundaries using real registries, services and runner."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker

import conciliator.services.aeolus_registry as registry_module
from conciliator.domain.aeolus_task import AeolusRound, AeolusStartEvent, AeolusState
from conciliator.integrations.coco.aeolus import CocoAeolusRunner
from conciliator.integrations.coco.client import CocoEvent, CocoTaskResult
from conciliator.repositories.models import Base, MeegleAINodeCommentClaim, MeegleAINodeTask
from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL, AeolusCommentHandler
from conciliator.services.aeolus_registry import AeolusTaskRegistry, TaskLeaseLost, task_lease
from conciliator.services.aeolus_service import AeolusService
from conciliator.services.comment_events import CommentEvent, ThreadComment
from conciliator.utils.retry import TransientError

# Background tasks need scheduling slack on a loaded CI, and an over-tight
# bound turns that delay into a false failure.
_WAIT_SECONDS = 10


def _cursor(run_id="run-1"):
    return CocoEvent(
        event_id="event-1", session_id="session-1", run_id=run_id, payload={}, raw={}
    )


def _event():
    return CommentEvent(
        comment_id="q-1", repo_id="repo-1", mr_iid=7, body="/ask 为什么幂等？",
        author_email="dev@example.com", discussion_id="thread-1", position=None,
    )


@pytest.fixture
async def runtime(tmp_path):
    # Separate connections and registry instances model independent workers.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'questions.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(factory)
    await registry.create_from_event(AeolusStartEvent(
        task_id="task-1", project_key="space-1", work_item_id=42,
        state_key="state-1", node_uuid="node-1", user_key="user-1", props=(),
    ))
    await registry.update("task-1", repo_id="repo-1", mr_iid=7, coco_session_id="session-1")
    await registry.append_round("task-1", AeolusRound(round_no=1, status="review"))
    await registry.transition("task-1", {AeolusState.RECEIVED}, AeolusState.REVIEW)
    client = SimpleNamespace(
        send_user_message=AsyncMock(return_value=_cursor()),
        fetch_session_result=AsyncMock(return_value=CocoTaskResult(
            status="succeeded", final_message="幂等的答案", last_event_id="end"
        )),
        send_interrupt=AsyncMock(), send_tool_confirmation=AsyncMock(),
    )
    reply = AsyncMock()
    workers = []

    def worker():
        own_registry = AeolusTaskRegistry(factory)
        runner = CocoAeolusRunner(client, workspace_id="workspace", agent_id="agent", poll_interval=0.001)
        service = AeolusService(
            registry=own_registry, document_reader=AsyncMock(), coco=runner,
            code_host=AsyncMock(), meegle=AsyncMock(), minimum_processing_seconds=0,
        )
        tasks = []

        async def answerer(task, question, position, on_started=None):
            return await service.answer_question(
                task.task_id, question, position=position, on_started=on_started
            )

        handler = AeolusCommentHandler(
            registry=own_registry,
            comments=SimpleNamespace(spawn=lambda operation: tasks.append(asyncio.create_task(operation))),
            meegle=AsyncMock(), reply=reply, answerer=answerer, resumer=service.resume_answer,
        )
        result = SimpleNamespace(registry=own_registry, service=service, runner=runner, handler=handler, tasks=tasks)
        workers.append(result)
        return result

    yield SimpleNamespace(engine=engine, registry=registry, client=client, reply=reply, worker=worker)
    for item in workers:
        for task in item.tasks:
            task.cancel()
        await asyncio.gather(*item.tasks, return_exceptions=True)
    engine.dispose()


async def _drain(worker):
    await asyncio.wait_for(asyncio.gather(*worker.tasks), timeout=_WAIT_SECONDS)


async def _run_question(runtime, worker, entrypoint):
    if entrypoint == "normal":
        await worker.handler.handle(_event())
        await _drain(worker)
    else:
        token = await _sending(runtime.registry, with_cursor=True)
        await runtime.registry.release_comment("task-1", "mr-comment-answer:q-1", token)
        await worker.handler.resume_pending_questions()


async def _record(registry):
    event = _event()
    await registry.record_question(
        "task-1", event.comment_id, repo_id=event.repo_id, mr_iid=event.mr_iid,
        discussion_id=event.discussion_id, position=None, parent=None,
        question_text=event.body, author_email=event.author_email,
    )


async def _sending(registry, *, with_cursor=False):
    await _record(registry)
    token = await registry.claim_comment("task-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL)
    assert token is not None
    assert await registry.mark_question_sending(
        "task-1", "q-1", token=token, attempt=0, session_id="session-1"
    )
    if with_cursor:
        assert await registry.mark_question_answering(
            "task-1", "q-1", token=token, attempt=1,
            session_id="session-1", run_id="run-1", event_id="event-1",
        )
    return token


def _expire_claim(runtime):
    with runtime.engine.begin() as connection:
        connection.execute(update(MeegleAINodeCommentClaim).where(
            MeegleAINodeCommentClaim.task_id == "task-1",
            MeegleAINodeCommentClaim.request_id == "mr-comment-answer:q-1",
        ).values(expires_at=datetime(2000, 1, 1)))


async def test_recovery_does_not_close_a_live_send(runtime):
    started, finish = asyncio.Event(), asyncio.Event()

    async def send(**_kwargs):
        started.set()
        await finish.wait()
        return _cursor()

    runtime.client.send_user_message.side_effect = send
    first, second = runtime.worker(), runtime.worker()
    await first.handler.handle(_event())
    await asyncio.wait_for(started.wait(), timeout=_WAIT_SECONDS)
    await second.handler.resume_pending_questions()

    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "sending" and current["run_id"] is None
    runtime.reply.assert_not_awaited()
    finish.set()
    await _drain(first)
    assert runtime.client.send_user_message.await_count == 1
    assert runtime.reply.await_count == 1
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"


async def test_stale_pending_snapshot_is_rechecked_before_recovery(runtime):
    await _record(runtime.registry)
    snapshot = await runtime.registry.question_record("task-1", "q-1")
    token = await _sending(runtime.registry)
    await runtime.registry.release_comment("task-1", "mr-comment-answer:q-1", token)

    await runtime.worker().handler._resume_question(snapshot)

    runtime.client.send_user_message.assert_not_awaited()
    assert runtime.reply.await_count == 1
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "unknown" and current["answer_text"] is None


async def test_stale_sending_snapshot_resumes_the_persisted_cursor(runtime):
    token = await _sending(runtime.registry)
    snapshot = await runtime.registry.question_record("task-1", "q-1")
    assert await runtime.registry.mark_question_answering(
        "task-1", "q-1", token=token, attempt=1,
        session_id="session-1", run_id="run-1", event_id="event-1",
    )
    await runtime.registry.release_comment("task-1", "mr-comment-answer:q-1", token)

    await runtime.worker().handler._resume_question(snapshot)

    runtime.client.send_user_message.assert_not_awaited()
    assert runtime.client.fetch_session_result.await_args.kwargs["run_id"] == "run-1"
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"
    assert runtime.reply.await_count == 1
    assert "无法确认" not in runtime.reply.await_args.args[1]


@pytest.mark.parametrize("failure", ["reply", "status_write"])
async def test_unknown_notice_is_recoverable_without_resending(runtime, failure):
    token = await _sending(runtime.registry)
    await runtime.registry.release_comment("task-1", "mr-comment-answer:q-1", token)
    first, second = runtime.worker(), runtime.worker()
    if failure == "reply":
        runtime.reply.side_effect = [RuntimeError("reply unavailable"), None]
    else:
        first.registry.mark_question_published = AsyncMock(side_effect=RuntimeError("status write unavailable"))
    await first.handler.resume_pending_questions()

    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "unknown" and "重新提问" in current["answer_text"]
    if failure == "status_write":
        second.handler._thread_reader = AsyncMock(return_value=[ThreadComment(
            "notice-1", runtime.reply.await_args.args[1], "bot@example.com"
        )])
    await second.handler.resume_pending_questions()
    await second.handler.resume_pending_questions()

    runtime.client.send_user_message.assert_not_awaited()
    assert runtime.reply.await_count == (2 if failure == "reply" else 1)
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "unknown" and current["answer_text"] is None


async def test_waiting_for_task_lease_stays_pending_when_feedback_starts(runtime):
    first = runtime.worker()
    assert await runtime.registry.acquire_lease("task-1", "busy-worker")
    waiting = asyncio.Event()
    acquire = first.registry.acquire_lease

    async def observed_acquire(*args, **kwargs):
        acquired = await acquire(*args, **kwargs)
        if not acquired:
            waiting.set()
        return acquired

    first.registry.acquire_lease = observed_acquire
    await first.handler.handle(_event())
    await asyncio.wait_for(waiting.wait(), timeout=_WAIT_SECONDS)
    before = await runtime.registry.question_record("task-1", "q-1")
    assert before["status"] == "pending" and before["attempts"] == 0
    await runtime.registry.start_feedback("task-1", "请补测试", "feedback-1")
    await runtime.registry.release_lease("task-1", "busy-worker")
    await _drain(first)

    runtime.client.send_user_message.assert_not_awaited()
    runtime.reply.assert_not_awaited()
    assert (await runtime.registry.question_record("task-1", "q-1")) == before
    await runtime.registry.transition("task-1", {AeolusState.RECODING}, AeolusState.REVIEW)
    await runtime.worker().handler.resume_pending_questions()
    assert runtime.client.send_user_message.await_count == 1
    assert runtime.reply.await_count == 1
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"


async def test_resumed_answer_renews_task_lease_past_its_original_expiry(runtime, monkeypatch):
    base = datetime(2030, 1, 1)
    clock = {"now": base}
    monkeypatch.setattr(registry_module, "_database_now", lambda _session: clock["now"])
    monkeypatch.setattr("conciliator.services.aeolus_service._LEASE_RENEW_SECONDS", 0.01)
    started, finish, renewed, waiting = (asyncio.Event() for _ in range(4))
    first, second = runtime.worker(), runtime.worker()
    renew = first.registry.renew_lease

    async def observed_renew(*args, **kwargs):
        result = await renew(*args, **kwargs)
        if result and clock["now"] == base + timedelta(seconds=59):
            renewed.set()
        return result

    async def fetch(_session, **kwargs):
        if kwargs["run_id"] == "run-original":
            started.set()
            await finish.wait()
        return CocoTaskResult(status="succeeded", final_message="答案", last_event_id="end")

    acquire = second.registry.acquire_lease

    async def observed_acquire(*args, **kwargs):
        result = await acquire(*args, **kwargs)
        if not result:
            waiting.set()
        return result

    first.registry.renew_lease = AsyncMock(side_effect=observed_renew)
    second.registry.acquire_lease = observed_acquire
    runtime.client.fetch_session_result.side_effect = fetch
    resumed = asyncio.create_task(first.service.resume_answer("task-1", "session-1", "run-original", "start"))
    first.tasks.append(resumed)
    await asyncio.wait_for(started.wait(), timeout=_WAIT_SECONDS)
    clock["now"] = base + timedelta(seconds=59)
    await asyncio.wait_for(renewed.wait(), timeout=_WAIT_SECONDS)
    clock["now"] = base + timedelta(seconds=61)
    question = asyncio.create_task(second.service.answer_question("task-1", "新问题"))
    second.tasks.append(question)
    await asyncio.wait_for(waiting.wait(), timeout=_WAIT_SECONDS)

    assert first.registry.renew_lease.await_count > 0 and not resumed.done()
    runtime.client.send_user_message.assert_not_awaited()
    finish.set()
    assert await asyncio.wait_for(resumed, timeout=_WAIT_SECONDS) == "答案"
    assert await asyncio.wait_for(question, timeout=_WAIT_SECONDS) == "答案"
    assert runtime.client.send_user_message.await_count == 1


@pytest.mark.parametrize("renewal_failure", [False, RuntimeError("renewal unavailable")])
async def test_resumed_answer_cancels_collector_when_task_renewal_fails(runtime, monkeypatch, renewal_failure):
    monkeypatch.setattr("conciliator.services.aeolus_service._LEASE_RENEW_SECONDS", 0.01)
    started, cancelled = asyncio.Event(), asyncio.Event()
    worker = runtime.worker()

    async def fetch(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def fail_renewal(*_args, **_kwargs):
        await started.wait()
        if isinstance(renewal_failure, Exception):
            raise renewal_failure
        return renewal_failure

    runtime.client.fetch_session_result.side_effect = fetch
    worker.registry.renew_lease = AsyncMock(side_effect=fail_renewal)
    with pytest.raises(TaskLeaseLost) as error:
        await asyncio.wait_for(
            worker.service.resume_answer("task-1", "session-1", "run-1", "start"),
            timeout=_WAIT_SECONDS,
        )
    if isinstance(renewal_failure, Exception):
        assert error.value.__cause__ is renewal_failure

    assert cancelled.is_set()
    runtime.client.send_tool_confirmation.assert_not_awaited()
    runtime.client.send_interrupt.assert_not_awaited()
    assert await runtime.registry.acquire_lease("task-1", "next-worker")
    task = await runtime.registry.get("task-1")
    assert task.state == AeolusState.REVIEW and not task.last_error


async def test_lost_send_response_is_never_resent(runtime, monkeypatch):
    monkeypatch.setattr("conciliator.services.aeolus_comment_service._ANSWER_RETRY_DELAY_SECONDS", 0)
    # The substitute accepted run-1 before losing the response; no cursor is
    # returned to the process, just as with a response timeout after acceptance.
    accepted = []

    async def send(**kwargs):
        accepted.append(kwargs["text"])
        raise TransientError("accepted, response lost")

    runtime.client.send_user_message.side_effect = send
    first = runtime.worker()
    await first.handler.handle(_event())
    await _drain(first)
    await runtime.worker().handler.resume_pending_questions()
    await first.handler.handle(_event())
    await _drain(first)

    assert len(accepted) == 1 and runtime.client.send_user_message.await_count == 1
    runtime.client.fetch_session_result.assert_not_awaited()
    assert runtime.reply.await_count == 1
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "unknown" and current["attempts"] == 1


@pytest.mark.parametrize("entrypoint", ["normal", "resume"])
@pytest.mark.parametrize("takeover", [False, True], ids=["expired", "replacement-owner"])
async def test_finished_answer_with_lost_task_lease_is_not_saved_or_published(
    runtime, monkeypatch, entrypoint, takeover,
):
    # No heartbeat fires: the final ownership check must fence a successful
    # collector that loses its task lease immediately before returning.
    monkeypatch.setattr("conciliator.services.aeolus_service._LEASE_RENEW_SECONDS", 3600)
    worker = runtime.worker()
    collected = asyncio.Event()

    async def finish_after_losing_lease(*_args, **_kwargs):
        with runtime.engine.begin() as connection:
            connection.execute(update(MeegleAINodeTask).where(
                MeegleAINodeTask.task_id == "task-1",
            ).values(lease_expires_at=datetime(2000, 1, 1)))
        if takeover:
            assert await runtime.registry.acquire_lease("task-1", "replacement-owner")
        collected.set()
        return CocoTaskResult(status="succeeded", final_message="旧 worker 的答案", last_event_id="end")

    runtime.client.fetch_session_result.side_effect = finish_after_losing_lease
    await asyncio.wait_for(_run_question(runtime, worker, entrypoint), timeout=_WAIT_SECONDS)

    assert collected.is_set()
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "answering" and current["answer_text"] is None
    assert (current["session_id"], current["run_id"], current["event_id"], current["attempts"]) == (
        "session-1", "run-1", "event-1", 1,
    )
    runtime.reply.assert_not_awaited()
    sent = int(entrypoint == "normal")
    assert runtime.client.send_user_message.await_count == sent
    if takeover:
        # The stale worker's finally must not release its replacement's lease.
        assert not await runtime.registry.acquire_lease("task-1", "third-owner")
        await runtime.registry.release_lease("task-1", "replacement-owner")
    runtime.client.fetch_session_result.side_effect = None
    await runtime.worker().handler.resume_pending_questions()
    assert runtime.client.send_user_message.await_count == sent
    assert runtime.reply.await_count == 1
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"


@pytest.mark.parametrize("entrypoint", ["normal", "resume"])
async def test_task_heartbeat_exception_keeps_question_recoverable(runtime, monkeypatch, entrypoint):
    monkeypatch.setattr("conciliator.services.aeolus_service._LEASE_RENEW_SECONDS", 0.01)
    worker = runtime.worker()
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def collect(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def failed_renewal(*_args, **_kwargs):
        await started.wait()
        raise RuntimeError("task lease renewal unavailable")

    runtime.client.fetch_session_result.side_effect = collect
    worker.registry.renew_lease = AsyncMock(side_effect=failed_renewal)
    await asyncio.wait_for(_run_question(runtime, worker, entrypoint), timeout=_WAIT_SECONDS)

    assert cancelled.is_set() and worker.registry.renew_lease.await_count == 1
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "answering" and current["answer_text"] is None
    assert (current["session_id"], current["run_id"], current["event_id"]) == ("session-1", "run-1", "event-1")
    runtime.reply.assert_not_awaited()
    runtime.client.send_interrupt.assert_not_awaited()
    runtime.client.send_tool_confirmation.assert_not_awaited()
    runtime.client.fetch_session_result.side_effect = None
    await runtime.worker().handler.resume_pending_questions()
    assert runtime.client.send_user_message.await_count == int(entrypoint == "normal")
    assert runtime.reply.await_count == 1
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"


@pytest.mark.parametrize("entrypoint", ["normal", "resume"])
@pytest.mark.parametrize("failure", ["query", "heartbeat"])
async def test_final_task_lease_check_failure_keeps_question_recoverable(runtime, monkeypatch, entrypoint, failure):
    monkeypatch.setattr(
        "conciliator.services.aeolus_service._LEASE_RENEW_SECONDS", 0.01 if failure == "heartbeat" else 3600
    )
    worker = runtime.worker()
    collected, checked, renewal_failed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    get = worker.registry.get

    async def finish(*_args, **_kwargs):
        collected.set()
        return CocoTaskResult(status="succeeded", final_message="成功的答案", last_event_id="end")

    async def unavailable_check(*args, **kwargs):
        if collected.is_set() and registry_module._lease_owner.get() is not None:
            checked.set()
            if failure == "query":
                raise RuntimeError("task lease ownership query unavailable")
            # The task row still belongs to us, but heartbeat failure during
            # this read must not be silently discarded by helper cleanup.
            await renewal_failed.wait()
        return await get(*args, **kwargs)

    async def failed_renewal(*_args, **_kwargs):
        await checked.wait()
        try:
            raise RuntimeError("task lease renewal unavailable during final check")
        finally:
            renewal_failed.set()

    runtime.client.fetch_session_result.side_effect = finish
    worker.registry.get = unavailable_check
    if failure == "heartbeat":
        worker.registry.renew_lease = AsyncMock(side_effect=failed_renewal)
    await asyncio.wait_for(_run_question(runtime, worker, entrypoint), timeout=_WAIT_SECONDS)

    assert checked.is_set()
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "answering" and current["answer_text"] is None and current["run_id"] == "run-1"
    runtime.reply.assert_not_awaited()
    runtime.client.fetch_session_result.side_effect = None
    await runtime.worker().handler.resume_pending_questions()
    assert runtime.client.send_user_message.await_count == int(entrypoint == "normal")
    assert runtime.reply.await_count == 1
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"


@pytest.mark.parametrize("entrypoint", ["normal", "resume"])
async def test_queued_feedback_does_not_reject_a_finished_answer(runtime, entrypoint):
    worker = runtime.worker()

    async def finish_with_feedback(*_args, **_kwargs):
        await runtime.registry.start_feedback("task-1", "请补测试", "feedback-1")
        return CocoTaskResult(status="succeeded", final_message="排队反馈之前的答案", last_event_id="end")

    runtime.client.fetch_session_result.side_effect = finish_with_feedback
    await asyncio.wait_for(_run_question(runtime, worker, entrypoint), timeout=_WAIT_SECONDS)

    task = await runtime.registry.get("task-1")
    assert task.state == AeolusState.RECODING and task.rounds[-1].status == "queued"
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"
    assert runtime.client.send_user_message.await_count == int(entrypoint == "normal")
    assert runtime.reply.await_count == 1
    assert "排队反馈之前的答案" in runtime.reply.await_args.args[1]


async def test_confirmed_terminal_answer_failure_retries_with_a_new_attempt(runtime, monkeypatch):
    monkeypatch.setattr("conciliator.services.aeolus_comment_service._ANSWER_RETRY_DELAY_SECONDS", 0)
    runtime.client.send_user_message.side_effect = [_cursor("run-1"), _cursor("run-2")]
    runtime.client.fetch_session_result.side_effect = [
        CocoTaskResult(status="failed", final_message="", error="run ended without output"),
        CocoTaskResult(status="succeeded", final_message="重试答案", last_event_id="end"),
    ]
    worker = runtime.worker()
    await worker.handler.handle(_event())
    await _drain(worker)

    assert runtime.client.send_user_message.await_count == 2
    assert runtime.reply.await_count == 1
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "published" and current["attempts"] == 2 and current["run_id"] == "run-2"


async def test_transient_resume_failure_keeps_the_original_cursor(runtime):
    token = await _sending(runtime.registry, with_cursor=True)
    await runtime.registry.release_comment("task-1", "mr-comment-answer:q-1", token)
    worker = runtime.worker()
    original = await runtime.registry.question_record("task-1", "q-1")
    worker.handler._resumer = AsyncMock(side_effect=TransientError("read unavailable"))
    await worker.handler.resume_pending_questions()

    assert (await runtime.registry.question_record("task-1", "q-1")) == original
    runtime.client.send_user_message.assert_not_awaited()
    runtime.reply.assert_not_awaited()
    worker.handler._resumer = worker.service.resume_answer
    await worker.handler.resume_pending_questions()
    assert runtime.client.fetch_session_result.await_args.kwargs["run_id"] == "run-1"
    assert runtime.reply.await_count == 1
    runtime.client.send_user_message.assert_not_awaited()


async def test_question_claim_loss_cancels_recovery_and_keeps_its_cursor(runtime, monkeypatch):
    token = await _sending(runtime.registry, with_cursor=True)
    await runtime.registry.release_comment("task-1", "mr-comment-answer:q-1", token)
    original = await runtime.registry.question_record("task-1", "q-1")
    monkeypatch.setattr("conciliator.services.aeolus_comment_service.ANSWER_CLAIM_TTL", timedelta(seconds=0.3))
    started, cancelled = asyncio.Event(), asyncio.Event()
    worker = runtime.worker()

    async def fetch(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def fail_renewal(*_args, **_kwargs):
        await started.wait()
        return False

    runtime.client.fetch_session_result.side_effect = fetch
    worker.registry.renew_comment = AsyncMock(side_effect=fail_renewal)
    await asyncio.wait_for(worker.handler.resume_pending_questions(), timeout=_WAIT_SECONDS)

    assert cancelled.is_set()
    assert (await runtime.registry.question_record("task-1", "q-1")) == original
    runtime.client.send_user_message.assert_not_awaited()
    runtime.reply.assert_not_awaited()
    assert await runtime.registry.acquire_lease("task-1", "after-recovery")


@pytest.mark.parametrize("status", ["answered", "published", "failed", "unknown"])
async def test_late_cursor_callback_cannot_reopen_a_settled_question(runtime, status):
    old_token = await _sending(runtime.registry, with_cursor=status != "unknown")
    old_record = await runtime.registry.question_record("task-1", "q-1")
    _expire_claim(runtime)
    new_token = await runtime.registry.claim_comment("task-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL)
    assert new_token is not None and new_token != old_token
    if status in {"answered", "published"}:
        assert await runtime.registry.mark_question_answered(
            "task-1", "q-1", "新 owner 的答案", token=new_token, attempt=1
        )
        if status == "published":
            assert await runtime.registry.mark_question_published("task-1", "q-1", token=new_token, attempt=1)
    elif status == "failed":
        assert await runtime.registry.mark_question_failed("task-1", "q-1", token=new_token, attempt=1)
    else:
        assert await runtime.registry.mark_question_unknown(
            "task-1", "q-1", token=new_token, attempt=1, notice="无法确认"
        )
    settled = await runtime.registry.question_record("task-1", "q-1")
    handler = runtime.worker().handler

    for token in (old_token, new_token):
        with pytest.raises(TaskLeaseLost):
            await handler._track_answer_start(old_record, token, 1, "session-1", "late-run", "late-event")
        assert (await runtime.registry.question_record("task-1", "q-1")) == settled


async def test_previous_attempt_callback_cannot_replace_the_current_send(runtime):
    token = await _sending(runtime.registry, with_cursor=True)
    old_record = await runtime.registry.question_record("task-1", "q-1")
    # The old run was confirmed terminal before a safe replacement starts.
    assert await runtime.registry.mark_question_sending(
        "task-1", "q-1", token=token, attempt=1, session_id="session-1"
    )
    sending = await runtime.registry.question_record("task-1", "q-1")
    handler = runtime.worker().handler
    with pytest.raises(TaskLeaseLost):
        await handler._track_answer_start(old_record, token, 1, "session-1", "old-run", "old-event")
    assert (await runtime.registry.question_record("task-1", "q-1")) == sending
    await handler._track_answer_start(sending, token, 2, "session-1", "new-run", "new-event")
    assert (await runtime.registry.question_record("task-1", "q-1"))["run_id"] == "new-run"


@pytest.mark.parametrize("write", ["sending", "cursor", "answer", "failure", "unknown", "publish"])
async def test_expired_claim_cannot_write_a_question_even_without_takeover(runtime, write):
    token = await _sending(runtime.registry, with_cursor=write in {"sending", "answer", "failure", "publish"})
    registry = runtime.registry
    if write == "publish":
        assert await registry.mark_question_answered("task-1", "q-1", "已保存", token=token, attempt=1)
    before = await registry.question_record("task-1", "q-1")
    _expire_claim(runtime)
    if write == "sending":
        changed = await registry.mark_question_sending("task-1", "q-1", token=token, attempt=1, session_id="session-1")
    elif write == "cursor":
        changed = await registry.mark_question_answering(
            "task-1", "q-1", token=token, attempt=1, session_id="session-1", run_id="late", event_id="late"
        )
    elif write == "answer":
        changed = await registry.mark_question_answered("task-1", "q-1", "旧答案", token=token, attempt=1)
    elif write == "failure":
        changed = await registry.mark_question_failed("task-1", "q-1", token=token, attempt=1)
    elif write == "unknown":
        changed = await registry.mark_question_unknown("task-1", "q-1", token=token, attempt=1, notice="无法确认")
    else:
        changed = await registry.mark_question_published("task-1", "q-1", token=token, attempt=1)
    assert changed is False
    assert (await registry.question_record("task-1", "q-1")) == before


async def test_answer_cas_rejection_prevents_publication(runtime):
    worker = runtime.worker()
    persist = worker.registry.mark_question_answered
    replacement = []

    async def lose_claim_before_persist(*args, **kwargs):
        _expire_claim(runtime)
        replacement.append(await runtime.registry.claim_comment("task-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL))
        return await persist(*args, **kwargs)

    worker.registry.mark_question_answered = lose_claim_before_persist
    await worker.handler.handle(_event())
    await _drain(worker)

    runtime.reply.assert_not_awaited()
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "answering" and current["answer_text"] is None
    assert replacement[0] is not None
    await runtime.registry.release_comment("task-1", "mr-comment-answer:q-1", replacement[0])
    await runtime.worker().handler.resume_pending_questions()
    assert runtime.client.send_user_message.await_count == 1
    assert runtime.reply.await_count == 1


@pytest.mark.parametrize("entrypoint", ["normal", "resume"])
async def test_takeover_after_final_owner_snapshot_cannot_save_or_publish(runtime, monkeypatch, entrypoint):
    monkeypatch.setattr("conciliator.services.aeolus_service._LEASE_RENEW_SECONDS", 3600)
    worker = runtime.worker()
    collected, taken_over = asyncio.Event(), asyncio.Event()
    get = worker.registry.get

    async def finish(*_args, **_kwargs):
        collected.set()
        return CocoTaskResult(status="succeeded", final_message="快照后的旧答案", last_event_id="end")

    async def get_then_take_over(*args, **kwargs):
        # The real get has already run _require and validated the old owner.
        snapshot = await get(*args, **kwargs)
        if collected.is_set() and registry_module._lease_owner.get() is not None and not taken_over.is_set():
            taken_over.set()
            with runtime.engine.begin() as connection:
                connection.execute(update(MeegleAINodeTask).where(
                    MeegleAINodeTask.task_id == "task-1",
                ).values(lease_expires_at=datetime(2000, 1, 1)))
            assert await runtime.registry.acquire_lease("task-1", "replacement-owner")
        return snapshot

    runtime.client.fetch_session_result.side_effect = finish
    worker.registry.get = get_then_take_over
    await asyncio.wait_for(_run_question(runtime, worker, entrypoint), timeout=_WAIT_SECONDS)

    assert taken_over.is_set()
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "answering" and current["answer_text"] is None
    assert (current["session_id"], current["run_id"], current["event_id"]) == ("session-1", "run-1", "event-1")
    runtime.reply.assert_not_awaited()
    assert not await runtime.registry.acquire_lease("task-1", "third-owner")
    await runtime.registry.release_lease("task-1", "replacement-owner")
    runtime.client.fetch_session_result.side_effect = None
    await runtime.worker().handler.resume_pending_questions()
    assert runtime.client.send_user_message.await_count == int(entrypoint == "normal")
    assert runtime.reply.await_count == 1
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"


@pytest.mark.parametrize("takeover", [False, True], ids=["expired", "replacement-owner"])
async def test_answer_update_fences_lease_changes_after_registry_owner_snapshot(runtime, monkeypatch, takeover):
    token = await _sending(runtime.registry, with_cursor=True)
    assert await runtime.registry.acquire_lease("task-1", "answer-owner")
    require = runtime.registry._require
    snapshots = []

    def require_then_change_lease(session, task_id, *, lock=False):
        record = require(session, task_id, lock=lock)
        if lock and registry_module._lease_owner.get() == "answer-owner" and not snapshots:
            snapshots.append(record.lease_owner)
            with runtime.engine.begin() as connection:
                connection.execute(update(MeegleAINodeTask).where(
                    MeegleAINodeTask.task_id == task_id,
                ).values(lease_expires_at=datetime(2000, 1, 1)))
            if takeover:
                assert runtime.registry._change_lease(task_id, "replacement-owner", 60, False)
        return record

    monkeypatch.setattr(runtime.registry, "_require", require_then_change_lease)
    with task_lease("answer-owner"):
        changed = await runtime.registry.mark_question_answered(
            "task-1", "q-1", "过期快照的答案", token=token, attempt=1
        )

    assert snapshots == ["answer-owner"]
    assert changed is False
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "answering" and current["answer_text"] is None
    if takeover:
        assert not await runtime.registry.acquire_lease("task-1", "third-owner")
    else:
        assert await runtime.registry.acquire_lease("task-1", "next-owner")


@pytest.mark.parametrize("entrypoint", ["normal", "resume"])
async def test_answer_acceptance_write_exception_keeps_original_cursor(runtime, entrypoint):
    worker = runtime.worker()
    owners = []

    async def unavailable_write(*_args, **_kwargs):
        owners.append(registry_module._lease_owner.get())
        raise RuntimeError("answer acceptance write unavailable")

    worker.registry.mark_question_answered = AsyncMock(side_effect=unavailable_write)
    await asyncio.wait_for(_run_question(runtime, worker, entrypoint), timeout=_WAIT_SECONDS)

    assert owners and owners[0] is not None
    current = await runtime.registry.question_record("task-1", "q-1")
    assert current["status"] == "answering" and current["answer_text"] is None and current["run_id"] == "run-1"
    runtime.reply.assert_not_awaited()
    await runtime.worker().handler.resume_pending_questions()
    assert runtime.client.send_user_message.await_count == int(entrypoint == "normal")
    assert runtime.reply.await_count == 1
    assert (await runtime.registry.question_record("task-1", "q-1"))["status"] == "published"


async def test_start_returns_before_the_first_recovery_pass_finishes(runtime):
    worker = runtime.worker()
    worker.handler._recovery_worker = None
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def blocked_pass():
        entered.set()
        await gate.wait()

    # Application readiness must not wait for a recovery pass that can block on
    # a whole agent answer.
    worker.handler.resume_pending_questions = blocked_pass
    await asyncio.wait_for(worker.handler.start(), timeout=1)
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert worker.handler._recovery_worker is not None
    gate.set()
    await worker.handler.aclose()


async def test_failed_answer_notice_is_republished_after_a_reply_failure(runtime, monkeypatch):
    monkeypatch.setattr(
        "conciliator.services.aeolus_comment_service._ANSWER_RETRY_DELAY_SECONDS", 0
    )
    runtime.client.fetch_session_result.return_value = CocoTaskResult(
        status="failed", final_message="", error="run ended without output"
    )
    runtime.reply.side_effect = RuntimeError("reply interface unavailable")
    worker = runtime.worker()
    await worker.handler.handle(_event())
    await _drain(worker)

    # The failed answer must not terminate the record before its notice is
    # visible: a reply hiccup has to stay recoverable.
    pending = await runtime.registry.question_record("task-1", "q-1")
    assert pending["status"] == "failed_pending_publish"
    assert "回答失败" in pending["answer_text"]

    runtime.reply.side_effect = None
    await runtime.worker().handler.resume_pending_questions()

    settled = await runtime.registry.question_record("task-1", "q-1")
    assert settled["status"] == "failed" and settled["answer_text"] is None
    assert "回答失败" in runtime.reply.await_args.args[1]


async def test_unreadable_thread_keeps_a_saved_answer_publishable(runtime):
    worker = runtime.worker()
    await _record(runtime.registry)
    token = await runtime.registry.claim_comment(
        "task-1", "mr-comment-answer:q-1", ANSWER_CLAIM_TTL
    )
    assert token is not None
    assert await runtime.registry.mark_question_answered(
        "task-1", "q-1", "幂等的答案", token=token, attempt=0
    )
    await runtime.registry.release_comment("task-1", "mr-comment-answer:q-1", token)
    # A failed read cannot prove the reply is missing, so the scan must keep the
    # saved body instead of sending the author a second copy.
    worker.handler._thread_reader = AsyncMock(side_effect=RuntimeError("lark 500"))

    await worker.handler.resume_pending_questions()

    runtime.reply.assert_not_awaited()
    record = await runtime.registry.question_record("task-1", "q-1")
    assert record["status"] == "answered" and record["answer_text"] == "幂等的答案"

@pytest.mark.asyncio
async def test_closed_task_notice_is_persisted_when_the_reply_fails(runtime):
    worker = runtime.worker()
    stale = await worker.registry.get("task-1")

    async def read_then_finish(*_args, **_kwargs):
        # Completion commits after the delivery read the task, which is the
        # window this notice exists for.
        await worker.registry.transition(
            "task-1", {AeolusState.REVIEW}, AeolusState.COMPLETED
        )
        return stale

    worker.registry.find_active_for_mr = read_then_finish
    runtime.reply.side_effect = RuntimeError("reply unavailable")

    await worker.handler.handle(_event())
    await _drain(worker)

    record = await worker.registry.question_record("task-1", "q-1")
    assert record is not None
    assert record["status"] == "failed_pending_publish"
    assert "任务已结束" in (record["answer_text"] or "")

    runtime.reply.side_effect = None
    await worker.handler.resume_pending_questions()
    await _drain(worker)

    settled = await worker.registry.question_record("task-1", "q-1")
    assert settled["status"] == "failed"
    assert any(
        "任务已结束" in str(call.args[1]) for call in runtime.reply.await_args_list
    )

@pytest.mark.asyncio
async def test_question_delivered_mid_round_is_answered_after_the_round(runtime):
    worker = runtime.worker()
    await runtime.registry.transition(
        "task-1", {AeolusState.REVIEW}, AeolusState.RECODING
    )

    # The delivery is acknowledged while the round owns the session.
    await worker.handler.handle(_event())
    await _drain(worker)
    runtime.reply.assert_not_awaited()
    record = await runtime.registry.question_record("task-1", "q-1")
    assert record is not None and record["status"] == "pending"

    # Once the round releases the session the scan answers the question.
    await runtime.registry.transition(
        "task-1", {AeolusState.RECODING}, AeolusState.REVIEW
    )
    await worker.handler.resume_pending_questions()
    record = await runtime.registry.question_record("task-1", "q-1")
    assert record["status"] == "published"
    assert "幂等" in runtime.reply.await_args.args[1]

async def test_question_after_a_failed_evaluation_is_told_the_task_ended(runtime):
    worker = runtime.worker()
    await runtime.registry.transition(
        "task-1", {AeolusState.REVIEW}, AeolusState.VALIDATION_FAILED
    )

    await worker.handler.handle(_event())
    await _drain(worker)

    # The notice is stored as a publishable body and settled once visible.
    settled = await runtime.registry.question_record("task-1", "q-1")
    assert settled["status"] == "failed"
    assert any(
        "任务已结束" in str(call.args[1]) for call in runtime.reply.await_args_list
    )


async def test_question_after_the_task_finished_is_told_the_task_ended(runtime):
    worker = runtime.worker()
    await runtime.registry.transition(
        "task-1", {AeolusState.REVIEW}, AeolusState.COMPLETED
    )

    await worker.handler.handle(_event())
    await _drain(worker)

    settled = await runtime.registry.question_record("task-1", "q-1")
    assert settled["status"] == "failed"
    assert any(
        "任务已结束" in str(call.args[1]) for call in runtime.reply.await_args_list
    )


async def test_question_delivered_during_evaluation_is_queued(runtime):
    worker = runtime.worker()
    await runtime.registry.transition(
        "task-1", {AeolusState.REVIEW}, AeolusState.EVALUATING
    )

    # Evaluation owns the session, so the question is stored, not answered.
    await worker.handler.handle(_event())
    await _drain(worker)
    runtime.reply.assert_not_awaited()
    record = await runtime.registry.question_record("task-1", "q-1")
    assert record is not None and record["status"] == "pending"

    await runtime.registry.transition(
        "task-1", {AeolusState.EVALUATING}, AeolusState.REVIEW
    )
    await worker.handler.resume_pending_questions()
    record = await runtime.registry.question_record("task-1", "q-1")
    assert record["status"] == "published"
    assert "幂等" in runtime.reply.await_args.args[1]
