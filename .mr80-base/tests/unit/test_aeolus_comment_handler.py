import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conciliator.api.comment_handlers import (
    ApprovalCommentHandler,
    ResolveCommandHandler,
)
from conciliator.api.comment_router import CommentDispatcher
from conciliator.domain.aeolus_task import AeolusStartEvent, AeolusState, AeolusTask
from conciliator.integrations.coco.aeolus import CocoRunFailed
from conciliator.repositories.models import Base
from conciliator.services.aeolus_comment_service import (
    ACK_BODY,
    ANSWER_PREFIX,
    AeolusCommentHandler,
    AeolusCommentService,
    _answer_marker,
    feedback_text,
    quote_block,
    select_parent,
)
from conciliator.services.aeolus_registry import (
    AeolusTaskRegistry,
    InvalidTaskTransition,
)
from conciliator.services.comment_events import CommentEvent, ThreadComment


def _task(state: AeolusState = AeolusState.REVIEW) -> AeolusTask:
    return AeolusTask(
        task_id="task-1",
        project_key="space-1",
        work_item_id=42,
        state_key="state-1",
        node_uuid="node-1",
        user_key="user-1",
        props=(),
        action_token="token-1",
        state=state,
        repo_id="repo-1",
        mr_iid=7,
        coco_session_id="session-1",
    )


def _event(body: str = "改成批量化", **overrides) -> CommentEvent:
    values = {
        "comment_id": "comment-1",
        "repo_id": "repo-1",
        "mr_iid": 7,
        "body": body,
        "author_email": "wangwenchao.129@bytedance.com",
        "discussion_id": "thread-1",
        "position": {"Path": "aeolus/lod.py", "StartLine": 12},
    }
    values.update(overrides)
    return CommentEvent(**values)


class _Spawner:
    """Collects background work so tests can run it deterministically."""

    def __init__(self) -> None:
        self.pending: list = []

    def spawn(self, operation) -> None:
        self.pending.append(operation)

    async def run(self) -> None:
        while self.pending:
            await self.pending.pop(0)


def _handler(
    task: AeolusTask | None,
    *,
    email: str | None = "wangwenchao.129@bytedance.com",
    thread: list[ThreadComment] | None = None,
    thread_error: Exception | None = None,
    answer: str = "这段逻辑是幂等的，因为重复调用不会改变结果。",
    answer_error: Exception | None = None,
):
    records = {}

    async def record_question(task_id, comment_id, **payload):
        payload.pop("notice_text", None)
        key = (task_id, comment_id)
        if key not in records:
            records[key] = {
                "id": len(records) + 1,
                "task_id": task_id, "comment_id": comment_id,
                "status": "pending", "attempts": 0, "answer_text": None,
                "session_id": None, "run_id": None, "event_id": None, **payload,
            }
        return records[key]

    async def question_record(task_id, comment_id):
        return records.get((task_id, comment_id))

    registry = SimpleNamespace(
        find_active_for_mr=AsyncMock(return_value=task),
        get=AsyncMock(return_value=task),
        record_question=AsyncMock(side_effect=record_question),
        question_record=AsyncMock(side_effect=question_record),
        mark_question_sending=AsyncMock(return_value=True),
        mark_question_answering=AsyncMock(return_value=True),
        mark_question_answered=AsyncMock(return_value=True),
        mark_question_published=AsyncMock(return_value=True),
        mark_question_failed=AsyncMock(return_value=True),
        mark_question_failed_notice=AsyncMock(return_value=True),
        question_records_to_publish=AsyncMock(return_value=[]),
        question_records_to_resume=AsyncMock(return_value=[]),
    )
    meegle = SimpleNamespace(get_user_email=AsyncMock(return_value=email))
    spawner = _Spawner()
    comments = SimpleNamespace(enqueue=AsyncMock(), spawn=spawner.spawn)
    reply = AsyncMock()
    answerer = AsyncMock(side_effect=answer_error, return_value=answer)
    thread_reader = AsyncMock(side_effect=thread_error, return_value=list(thread or []))
    handler = AeolusCommentHandler(
        registry=registry,
        comments=comments,
        meegle=meegle,
        reply=reply,
        thread_reader=thread_reader,
        answerer=answerer,
    )
    comments.spawner = spawner
    comments.answerer = answerer
    return handler, registry, meegle, comments, reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "position", "expected"),
    [
        # Ordinary review prose is discussion: only explicit commands are ours.
        ("这里改成批量化", {"Path": "a.py", "StartLine": 3}, False),
        ("/code 这里改成批量化", {"Path": "a.py", "StartLine": 3}, True),
        ("/code 这里改成批量化", None, True),
        ("/CODE 请补测试", None, True),
        ("/ask 这里的改动合理吗？", {"Path": "a.py", "StartLine": 3}, True),
        ("/ask 这里的改动合理吗？", None, True),
        # Removed aliases are ordinary review prose, not commands.
        ("/aeolus 这里改成批量化", {"Path": "a.py", "StartLine": 3}, False),
        ("/AEOLUS 这里改成批量化", None, False),
        ("/codex 这里改成批量化", None, False),  # a command is a whole word
        ("LGTM", None, False),  # an approval word belongs to the resolve flow
        ("/resolve start", {"Path": "a.py", "StartLine": 3}, False),
        ("/resolve approve all", None, False),
        ("/RESOLVE start", {"Path": "a.py", "StartLine": 3}, False),
        ("approve", {"Path": "a.py", "StartLine": 3}, False),
        ("❌ 这里不对", {"Path": "a.py", "StartLine": 3}, False),
    ],
)
async def test_matches_takes_explicit_commands_only(body, position, expected):
    handler, *_ = _handler(_task())
    event = _event(body=body, position=position)

    assert await handler.matches(event) is expected


@pytest.mark.asyncio
async def test_matches_claims_a_command_before_looking_up_the_task():
    handler, *_ = _handler(None)

    assert await handler.matches(_event(body="/code 这里的改动合理吗？")) is True


@pytest.mark.asyncio
async def test_inline_comment_enqueues_feedback_for_the_initiator():
    handler, registry, meegle, comments, reply = _handler(_task())

    await handler.handle(_event(body="/code 这里改成批量化"))

    registry.find_active_for_mr.assert_awaited_once_with("repo-1", 7)
    meegle.get_user_email.assert_awaited_once_with("user-1")
    feedback_event = comments.enqueue.await_args.kwargs["event"]
    assert feedback_event.comment_id == "comment-1"
    reply.assert_awaited_once()
    ack = reply.await_args.args[1]
    assert ack.startswith(ACK_BODY)
    assert "aeolus/lod.py:12" in ack
    assert "> 这里改成批量化" in ack


@pytest.mark.asyncio
async def test_comment_from_another_author_under_the_old_name_is_ignored():
    handler, _registry, _meegle, comments, reply = _handler(_task())

    await handler.handle(
        _event(body="/aeolus 这里改成批量化", author_email="someone.else@bytedance.com")
    )

    comments.enqueue.assert_not_awaited()
    reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_removed_aeolus_command_does_not_start_a_round():
    handler, _registry, _meegle, comments, reply = _handler(_task())

    await handler.handle(_event(body="/aeolus 这里改成批量化"))

    comments.enqueue.assert_not_awaited()
    comments.answerer.assert_not_awaited()
    reply.assert_not_awaited()


def test_feedback_text_does_not_strip_the_removed_aeolus_command():
    text = feedback_text(_event(body="/aeolus 这里改成批量化"))

    assert text.endswith("/aeolus 这里改成批量化")
    assert "这里改成批量化" in text


@pytest.mark.asyncio
async def test_comment_from_another_author_is_ignored_without_a_reply():
    handler, _registry, _meegle, comments, reply = _handler(_task())

    await handler.handle(
        _event(body="/code 这里改成批量化", author_email="someone.else@bytedance.com")
    )

    comments.enqueue.assert_not_awaited()
    reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_comment_without_an_active_task_is_ignored():
    handler, _registry, _meegle, comments, reply = _handler(None)

    await handler.handle(_event())

    comments.enqueue.assert_not_awaited()
    reply.assert_not_awaited()


def _thread(*entries: tuple[str, str, str]) -> list[ThreadComment]:
    return [
        ThreadComment(comment_id=cid, body=body, author_email=email)
        for cid, body, email in entries
    ]


@pytest.mark.asyncio
async def test_reply_carries_the_body_of_the_comment_it_answers():
    thread = _thread(
        ("comment-0", "这里是为什么？", "reviewer@bytedance.com"),
        ("comment-1", "/code 按上面说的改", "wangwenchao.129@bytedance.com"),
    )
    handler, _registry, _meegle, comments, _reply = _handler(_task(), thread=thread)

    await handler.handle(_event(body="/code 按上面说的改"))

    feedback_event = comments.enqueue.await_args.kwargs["event"]
    assert feedback_event.parent is not None
    assert feedback_event.parent.body == "这里是为什么？"
    assert feedback_event.parent.author_email == "reviewer@bytedance.com"
    rendered = feedback_text(feedback_event)
    assert "被回复的评论（reviewer@bytedance.com）" in rendered
    assert "这里是为什么？" in rendered
    assert "按上面说的改" in rendered


@pytest.mark.asyncio
async def test_parent_lookup_skips_our_own_acknowledgement():
    thread = _thread(
        ("comment-0", "这段逻辑对吗？", "reviewer@bytedance.com"),
        ("comment-1", ACK_BODY, "wangwenchao.129@bytedance.com"),
        ("comment-2", "/code 再确认一下", "wangwenchao.129@bytedance.com"),
    )
    handler, _registry, _meegle, comments, _reply = _handler(_task(), thread=thread)

    await handler.handle(_event(comment_id="comment-2", body="/code 再确认一下"))

    feedback_event = comments.enqueue.await_args.kwargs["event"]
    assert feedback_event.parent is not None
    assert feedback_event.parent.body == "这段逻辑对吗？"


@pytest.mark.asyncio
async def test_parent_lookup_failure_still_starts_the_round():
    handler, _registry, _meegle, comments, reply = _handler(
        _task(), thread_error=RuntimeError("429 TooManyRequests")
    )

    await handler.handle(_event(body="/code 这里改一下"))

    comments.enqueue.assert_awaited_once()
    reply.assert_awaited_once()
    assert comments.enqueue.await_args.kwargs["event"].parent is None


def test_select_parent_returns_none_for_the_first_comment_in_a_thread():
    thread = _thread(("comment-1", "/code 新开一个话题", "a@bytedance.com"))

    assert select_parent(thread, "comment-1") is None


def test_select_parent_falls_back_when_the_trigger_is_missing_from_the_snapshot():
    thread = _thread(("comment-0", "先看这个", "a@bytedance.com"))

    assert select_parent(thread, "comment-9") == thread[0]


@pytest.mark.asyncio
async def test_allowlisted_author_skips_the_directory_lookup():
    registry = SimpleNamespace(find_active_for_mr=AsyncMock(return_value=_task()))
    meegle = SimpleNamespace(get_user_email=AsyncMock(return_value=None))
    comments = SimpleNamespace(enqueue=AsyncMock())
    handler = AeolusCommentHandler(
        registry=registry,
        comments=comments,
        meegle=meegle,
        allowlist=["reviewer@bytedance.com"],
    )

    await handler.handle(
        _event(body="/code 改成批量化", author_email="reviewer@bytedance.com")
    )

    comments.enqueue.assert_awaited_once()
    meegle.get_user_email.assert_not_awaited()


@pytest.mark.asyncio
async def test_comments_in_one_window_become_a_single_feedback_round():
    service = SimpleNamespace(submit_feedback=AsyncMock())
    comments = AeolusCommentService(service=service, window_seconds=0)

    await comments.enqueue(task=_task(), event=_event(comment_id="c-1"))
    await comments.enqueue(
        task=_task(), event=_event(comment_id="c-2", body="还有这里")
    )
    await comments.drain()

    service.submit_feedback.assert_awaited_once()
    task_id, feedback, request_ids = service.submit_feedback.await_args.args
    assert task_id == "task-1"
    assert "改成批量化" in feedback and "还有这里" in feedback
    assert request_ids == ["mr-comment:c-1", "mr-comment:c-2"]


@pytest.mark.asyncio
async def test_comment_is_durable_before_the_aggregation_window(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    task_id = await _task_in_state(registry, AeolusState.REVIEW)
    task = await registry.get(task_id)

    class DurableService:
        def __init__(self) -> None:
            self.submitted = []

        async def save_pending_feedback(self, task_id, entries, *, available_at):
            return await registry.save_pending_feedback(
                task_id, entries, available_at=available_at
            )

        async def discard_pending_feedback(self, task_id, comment_ids):
            await registry.discard_pending_feedback(task_id, comment_ids)

        async def submit_feedback(
            self, task_id, feedback, request_ids, *, entries=None
        ):
            self.submitted.append((task_id, feedback, request_ids))

    service = DurableService()
    comments = AeolusCommentService(
        service=service, window_seconds=60, defer_seconds=0, max_deferrals=2
    )
    await comments.enqueue(
        task=task, event=_event(comment_id="c-1", body="/code 改成批量化")
    )

    # A restarted process only sees this row; the in-memory batch is gone.
    pending = await registry.pending_feedback(task_id)
    assert len(pending) == 1
    assert pending[0][0] == "c-1"
    assert pending[0][1].endswith("改成批量化")

    await comments.aclose()

    assert len(service.submitted) == 1
    submitted_task_id, feedback, request_ids = service.submitted[0]
    assert submitted_task_id == task_id
    assert feedback.endswith("改成批量化")
    assert request_ids == ["mr-comment:c-1"]
    assert await registry.pending_feedback(task_id) == []


@pytest.mark.asyncio
async def test_every_comment_in_a_batch_is_persisted(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    task_id = await _task_in_state(registry, AeolusState.REVIEW)
    task = await registry.get(task_id)

    class DurableService:
        def __init__(self) -> None:
            self.submitted = []

        async def save_pending_feedback(self, task_id, entries, *, available_at):
            return await registry.save_pending_feedback(
                task_id, entries, available_at=available_at
            )

        async def discard_pending_feedback(self, task_id, comment_ids):
            await registry.discard_pending_feedback(task_id, comment_ids)

        async def submit_feedback(
            self, task_id, feedback, request_ids, *, entries=None
        ):
            self.submitted.append((task_id, feedback, request_ids))

    service = DurableService()
    comments = AeolusCommentService(service=service, window_seconds=60)
    await comments.enqueue(
        task=task, event=_event(comment_id="c-1", body="/code 第一处")
    )
    await comments.enqueue(
        task=task, event=_event(comment_id="c-2", body="/code 第二处")
    )

    # A restart during the window must recover every comment of the batch, not
    # only the first one that made it durable.
    pending = await registry.pending_feedback(task_id)
    assert [comment_id for comment_id, _ in pending] == ["c-1", "c-2"]

    await comments.aclose()

    assert len(service.submitted) == 1
    _, feedback, request_ids = service.submitted[0]
    assert "第一处" in feedback and "第二处" in feedback
    assert request_ids == ["mr-comment:c-1", "mr-comment:c-2"]
    assert await registry.pending_feedback(task_id) == []


@pytest.mark.asyncio
async def test_missing_pending_table_rejects_the_comment_before_it_is_batched():
    service = SimpleNamespace(
        save_pending_feedback=AsyncMock(
            side_effect=RuntimeError("meegle_ai_node_pending_feedback is missing")
        ),
        submit_feedback=AsyncMock(),
    )
    comments = AeolusCommentService(service=service, window_seconds=0)

    with pytest.raises(RuntimeError, match="pending_feedback is missing"):
        await comments.enqueue(task=_task(), event=_event(comment_id="c-1"))

    service.save_pending_feedback.assert_awaited_once()
    service.submit_feedback.assert_not_awaited()


@pytest.mark.asyncio
async def test_blocked_coding_accepts_review_feedback():
    handler, _registry, _meegle, comments, reply = _handler(
        _task(state=AeolusState.CODING_BLOCKED)
    )

    await handler.handle(_event(body="/code 阻塞已解决，请继续"))

    comments.enqueue.assert_awaited_once()
    reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_blocked_coding_answers_questions():
    handler, _registry, _meegle, comments, _reply = _handler(
        _task(state=AeolusState.CODING_BLOCKED)
    )

    await handler.handle(_event(body="/ask 这里为什么阻塞？"))
    await comments.spawner.run()

    comments.enqueue.assert_not_awaited()
    comments.answerer.assert_awaited_once()


@pytest.mark.asyncio
async def test_own_acknowledgement_is_never_read_back_as_feedback():
    handler, _registry, _meegle, comments, reply = _handler(_task())

    assert await handler.matches(_event(body=ACK_BODY)) is False
    await handler.handle(_event(body=ACK_BODY))

    comments.enqueue.assert_not_awaited()
    reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_window_starts_at_the_first_comment_of_a_burst():
    service = SimpleNamespace(submit_feedback=AsyncMock())
    comments = AeolusCommentService(service=service, window_seconds=0.4)

    await comments.enqueue(task=_task(), event=_event(comment_id="c-1"))
    await asyncio.sleep(0.25)
    await comments.enqueue(
        task=_task(), event=_event(comment_id="c-2", body="还有这里")
    )
    await asyncio.sleep(0.3)

    # A window re-armed by the second comment would not have fired yet.
    service.submit_feedback.assert_awaited_once()


@pytest.mark.asyncio
async def test_feedback_waits_while_the_previous_round_is_running():
    service = SimpleNamespace(
        submit_feedback=AsyncMock(side_effect=InvalidTaskTransition("busy"))
    )
    comments = AeolusCommentService(
        service=service, window_seconds=0, defer_seconds=0, max_deferrals=3
    )

    await comments.enqueue(task=_task(state=AeolusState.RECODING), event=_event())
    await comments.drain()

    assert service.submit_feedback.await_count == 3


@pytest.mark.asyncio
async def test_dispatcher_prefers_the_resolve_command_over_aeolus_feedback():
    router = SimpleNamespace(
        route_resolve_trigger=AsyncMock(),
        route_approval_command=AsyncMock(),
    )
    handler, _registry, _meegle, comments, _reply = _handler(_task())
    dispatcher = CommentDispatcher()
    dispatcher.register(ResolveCommandHandler(router))
    dispatcher.register(handler)
    dispatcher.register(ApprovalCommentHandler(router))

    await dispatcher.dispatch(_event(body="/resolve start"))

    router.route_resolve_trigger.assert_awaited_once()
    comments.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatcher_sends_inline_comments_to_aeolus_not_approval():
    router = SimpleNamespace(
        route_resolve_trigger=AsyncMock(),
        route_approval_command=AsyncMock(),
    )
    handler, _registry, _meegle, comments, _reply = _handler(_task())
    dispatcher = CommentDispatcher()
    dispatcher.register(ResolveCommandHandler(router))
    dispatcher.register(handler)
    dispatcher.register(ApprovalCommentHandler(router))

    await dispatcher.dispatch(_event(body="/code 这里改成批量化"))

    comments.enqueue.assert_awaited_once()
    router.route_approval_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_comment_stays_ordinary_review_discussion():
    router = SimpleNamespace(
        route_resolve_trigger=AsyncMock(),
        route_approval_command=AsyncMock(),
    )
    handler, _registry, _meegle, comments, reply = _handler(_task())
    dispatcher = CommentDispatcher()
    dispatcher.register(ResolveCommandHandler(router))
    dispatcher.register(handler)
    dispatcher.register(ApprovalCommentHandler(router))

    await dispatcher.dispatch(_event(body="这里改成批量化"))
    await comments.spawner.run()

    comments.enqueue.assert_not_awaited()
    comments.answerer.assert_not_awaited()
    reply.assert_not_awaited()
    # Falls through to the approval handler, whose parser drops plain prose.
    router.route_approval_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_question_is_answered_in_the_same_thread():
    handler, _registry, _meegle, comments, reply = _handler(_task())

    await handler.handle(_event(body="/ask 这里的改动合理吗？"))
    await comments.spawner.run()

    comments.enqueue.assert_not_awaited()
    _task_arg, question, position = comments.answerer.await_args.args
    assert "这里的改动合理吗？" in question
    assert position == "aeolus/lod.py:12"
    body = reply.await_args.args[1]
    assert body.startswith(ANSWER_PREFIX)
    assert "`aeolus/lod.py:12`" in body
    assert "> 这里的改动合理吗？" in body
    assert "幂等" in body


def test_quote_block_names_the_file_and_line_of_the_comment():
    quote = quote_block(_event(body="/ask 这里的改动合理吗？"))

    assert quote.splitlines()[0] == "> 引用 @ `aeolus/lod.py:12`"
    assert "> 这里的改动合理吗？" in quote


def test_quote_block_prefers_the_comment_being_answered():
    event = _event(
        body="/ask 按上面说的改",
        parent=ThreadComment("comment-0", "这里为什么要用 useRef？", "b@bytedance.com"),
    )

    quote = quote_block(event)

    assert "这里为什么要用 useRef？" in quote
    assert "/ask 按上面说的改" not in quote


@pytest.mark.asyncio
async def test_question_from_another_author_is_answered():
    handler, _registry, meegle, comments, reply = _handler(_task())

    await handler.handle(
        _event(body="/ask 这里为什么这么写？", author_email="someone.else@bytedance.com")
    )
    await comments.spawner.run()

    # A question only spends a session turn: no work item is touched, so it
    # needs neither the initiator lookup nor a feedback round.
    meegle.get_user_email.assert_not_awaited()
    comments.enqueue.assert_not_awaited()
    _task_arg, question, _position = comments.answerer.await_args.args
    assert "这里为什么这么写？" in question
    assert reply.await_args.args[1].startswith(ANSWER_PREFIX)


@pytest.mark.asyncio
async def test_question_is_queued_while_a_coding_round_owns_the_session():
    handler, registry, _meegle, comments, reply = _handler(_task(AeolusState.RECODING))

    await handler.handle(_event(body="/ask 这里的改动合理吗？"))
    await comments.spawner.run()

    # The delivery is acknowledged with the question on disk, so the scan can
    # answer it once the round releases the session.
    record = await registry.question_record("task-1", "comment-1")
    assert record is not None and record["status"] == "pending"
    comments.answerer.assert_not_awaited()
    reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_redelivered_question_is_answered_once():
    handler, _registry, _meegle, comments, _reply = _handler(_task())
    event = _event(body="/ask 这里的改动合理吗？")

    await handler.handle(event)
    await handler.handle(event)
    await comments.spawner.run()

    comments.answerer.assert_awaited_once()


@pytest.mark.asyncio
async def test_answer_failure_tells_the_author_to_ask_again():
    handler, _registry, _meegle, comments, reply = _handler(
        _task(), answer_error=RuntimeError("Coco run timed out")
    )

    await handler.handle(_event(body="/ask 这里的改动合理吗？"))
    await comments.spawner.run()

    # The failure must not be silent: the author is told to ask again instead
    # of losing the question with the acknowledged delivery.
    reply.assert_awaited_once()
    body = reply.await_args.args[1]
    assert body.startswith(ANSWER_PREFIX)
    assert "重新提问" in body


@pytest.mark.asyncio
async def test_our_own_answer_is_never_read_back_as_a_question():
    handler, _registry, _meegle, comments, reply = _handler(_task())

    await handler.handle(_event(body=f"{ANSWER_PREFIX}\n这段逻辑是幂等的。"))
    await comments.spawner.run()

    comments.answerer.assert_not_awaited()
    reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_retryable_answer_failure_is_retried_then_replies(monkeypatch):
    monkeypatch.setattr(
        "conciliator.services.aeolus_comment_service._ANSWER_RETRY_DELAY_SECONDS", 0
    )
    handler, _registry, _meegle, comments, reply = _handler(_task())
    comments.answerer.side_effect = [
        CocoRunFailed("Coco platform hiccup", retryable=True, replacement_safe=True),
        "第二次回答成功了",
    ]

    await handler.handle(_event(body="/ask 这里的改动合理吗？"))
    await comments.spawner.run()

    assert comments.answerer.await_count == 2
    assert "第二次回答成功了" in reply.await_args.args[1]


@pytest.mark.asyncio
async def test_two_questions_quoting_the_same_parent_are_not_confused():
    first = _event(comment_id="q-1", body="/ask 按上面说的改")
    # The thread already carries the answer to the first question; a different
    # question in the same thread must not look already-published.
    handler, _registry, _meegle, comments, reply = _handler(
        _task(),
        thread=[
            ThreadComment(
                "reply-1",
                f"{ANSWER_PREFIX}\n\n已回答\n\n{_answer_marker(first)}",
                "wangwenchao.129@bytedance.com",
            )
        ],
    )

    second = _event(comment_id="q-2", body="/ask 按上面说的改")
    await handler.handle(second)
    await comments.spawner.run()

    assert _answer_marker(first) != _answer_marker(second)
    comments.answerer.assert_awaited_once()
    assert _answer_marker(second) in reply.await_args.args[1]


@pytest.mark.asyncio
async def test_answer_reply_shows_no_id_but_keeps_a_hidden_marker():
    handler, _registry, _meegle, comments, reply = _handler(_task())
    question = _event(comment_id="q-1", body="/ask 这里的改动合理吗？")

    await handler.handle(question)
    await comments.spawner.run()

    body = reply.await_args.args[1]
    assert body.splitlines()[0] == ANSWER_PREFIX
    assert "#" not in body.splitlines()[0]
    assert _answer_marker(question) in body


@pytest.mark.asyncio
async def test_task_lookup_failure_propagates_to_the_delivery():
    handler, registry, _meegle, _comments, _reply = _handler(_task())
    registry.find_active_for_mr = AsyncMock(
        side_effect=RuntimeError("registry unavailable")
    )

    with pytest.raises(RuntimeError, match="registry unavailable"):
        await handler.handle(_event(body="/ask 这里的改动合理吗？"))


@pytest.mark.asyncio
async def test_dispatcher_keeps_threaded_commands_with_the_approval_flow():
    router = SimpleNamespace(
        route_resolve_trigger=AsyncMock(),
        route_approval_command=AsyncMock(),
    )
    handler, _registry, _meegle, comments, _reply = _handler(_task())
    dispatcher = CommentDispatcher()
    dispatcher.register(ResolveCommandHandler(router))
    dispatcher.register(handler)
    dispatcher.register(ApprovalCommentHandler(router))

    await dispatcher.dispatch(_event(body="approve"))

    router.route_approval_command.assert_awaited_once()
    comments.enqueue.assert_not_awaited()


@pytest.fixture
def registry_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


async def _task_in_state(registry: AeolusTaskRegistry, state: AeolusState) -> str:
    event = AeolusStartEvent(
        task_id=f"idem-{state.value}",
        project_key="space-1",
        work_item_id=42,
        state_key="state-1",
        node_uuid=f"node-{state.value}",
        user_key="user-1",
        props=(),
    )
    task, _created = await registry.create_from_event(event)
    await registry.update(task.task_id, repo_id="repo-1", mr_iid=7)
    await registry.transition(task.task_id, {AeolusState.RECEIVED}, state)
    return task.task_id


@pytest.mark.asyncio
async def test_find_active_for_mr_returns_tasks_that_take_feedback(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    coding_id = await _task_in_state(registry, AeolusState.CODING)

    # A working task still takes a question: the handler queues it instead of
    # dropping an acknowledged delivery.
    found = await registry.find_active_for_mr("repo-1", 7)
    assert found is not None and found.task_id == coding_id

    completed_id = await _task_in_state(registry, AeolusState.COMPLETED)
    # A finished task is still returned, so the question gets the termination
    # notice instead of being acknowledged with nothing stored.
    found = await registry.find_active_for_mr("repo-1", 7)
    assert found is not None and found.task_id == completed_id

    blocked_id = await _task_in_state(registry, AeolusState.CODING_BLOCKED)
    found = await registry.find_active_for_mr("repo-1", 7)

    assert found is not None and found.task_id == blocked_id

    review_id = await _task_in_state(registry, AeolusState.REVIEW)
    found = await registry.find_active_for_mr("repo-1", 7)

    assert found is not None and found.task_id == review_id
    assert await registry.find_active_for_mr("repo-1", 999) is None


@pytest.mark.asyncio
async def test_aclose_flushes_a_batch_still_in_the_aggregation_window():
    submit_feedback = AsyncMock()
    service = AeolusCommentService(
        service=SimpleNamespace(submit_feedback=submit_feedback),
        window_seconds=60,
    )
    await service.enqueue(task=_task(), event=_event())

    await asyncio.wait_for(service.aclose(), timeout=2)

    submit_feedback.assert_awaited_once()
    assert not service._tasks
    assert not service._pending


@pytest.mark.asyncio
async def test_aclose_limits_the_time_spent_draining_retries():
    submitted = asyncio.Event()

    async def submit_feedback(*_args, **_kwargs):
        submitted.set()
        raise InvalidTaskTransition("previous coding round is still running")

    service = AeolusCommentService(
        service=SimpleNamespace(submit_feedback=submit_feedback),
        window_seconds=0,
        defer_seconds=60,
        max_deferrals=5,
        shutdown_timeout_seconds=0,
    )
    await service.enqueue(task=_task(), event=_event())
    await asyncio.wait_for(submitted.wait(), timeout=2)

    await asyncio.wait_for(service.aclose(), timeout=2)

    assert not service._tasks
    assert not service._pending


@pytest.mark.asyncio
async def test_retry_budget_covers_the_configured_coco_timeout(monkeypatch):
    attempts: list[str] = []

    async def submit_feedback(_task_id, _feedback, request_id, **_kwargs):
        attempts.append(request_id)
        if len(attempts) < 32:
            raise InvalidTaskTransition("previous coding round is still running")

    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    service = AeolusCommentService(
        service=SimpleNamespace(submit_feedback=submit_feedback),
        window_seconds=0,
        defer_seconds=30,
        retry_timeout_seconds=900,
    )
    await service.enqueue(task=_task(), event=_event())

    await asyncio.wait_for(service.drain(), timeout=2)

    assert len(attempts) == 32
    assert sleep.await_count == 31


@pytest.mark.asyncio
async def test_transient_failure_retries_instead_of_dropping_feedback():
    attempts: list[str] = []

    async def submit_feedback(_task_id, feedback, request_id, **_kwargs):
        attempts.append(request_id)
        if len(attempts) == 1:
            raise RuntimeError("database is unavailable")

    service = AeolusCommentService(
        service=SimpleNamespace(submit_feedback=submit_feedback),
        window_seconds=0,
        defer_seconds=0,
        max_deferrals=3,
    )
    await service.enqueue(task=_task(), event=_event())
    await asyncio.wait_for(service.drain(), timeout=2)

    assert len(attempts) == 2
