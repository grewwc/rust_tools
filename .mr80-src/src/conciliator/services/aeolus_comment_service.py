"""MR comment driven re-coding for Aeolus tasks.

Comments never call the coding pipeline directly: they are aggregated per task
for a short window so a burst of review comments becomes one feedback round
instead of one round per comment. Submission reuses
:meth:`AeolusService.submit_feedback`, which already enforces the ``review``
state and de-duplicates by request id.
"""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from conciliator.domain.aeolus_task import AeolusState, AeolusTask
from conciliator.domain.commands import Ignored, parse_comment
from conciliator.integrations.coco.aeolus import CocoRunFailed, CocoSendOutcomeUnknown
from conciliator.services.aeolus_registry import (
    QUESTION_TERMINAL_STATES,
    AeolusTaskRegistry,
    InvalidTaskTransition,
    TaskLeaseLost,
    TaskNotFound,
)
from conciliator.services.aeolus_service import question_answer_acceptance
from conciliator.services.comment_events import CommentEvent, ThreadComment
from conciliator.utils.logging import get_logger
from conciliator.utils.retry import TransientError

# Both actions are opt-in commands. Ordinary review prose is not ours: a remark
# such as "这里为什么这么写？" is discussion, and silently spending a coding
# run or an agent call on it is the wrong default.
RECODE_COMMAND = "/code"
QUESTION_COMMAND = "/ask"
COMMANDS = (RECODE_COMMAND, QUESTION_COMMAND)

ACK_BODY = "已收到反馈，正在汇总本轮评审意见，稍后将开始新一轮 Coding。"
# Answers are free-form, so unlike the acknowledgement they cannot be matched by
# an exact body. The visible prefix is the loop guard: the reply lands in the
# same thread and, with a personal-token client, is authored by the user
# themselves, so without it every answer would read back as a new question.
ANSWER_PREFIX = "🤖 Conciliator 回答："
# Long enough to cover one agent answer, short enough that a worker killed
# mid-answer cannot keep the comment unanswered indefinitely.
# States in which a Coco run owns the session. A question may be answered in
# any other state: nothing is running that an extra turn could interleave with.
_SESSION_BUSY_STATES = frozenset(
    {AeolusState.EVALUATING, AeolusState.CODING, AeolusState.RECODING}
)

# Question-record states that still owe work. Terminal transitions are
# conditional on one of these, so a worker that lost its claim cannot
# overwrite an answer another worker produced or published.
_QUESTION_ACTIVE_STATUSES = ("pending", "answering", "sending")
# States that hold a body the author must still see: a real answer, an
# ambiguous-send notice, or the notice explaining a failed answer. All three
# publish through the same claim-fenced path, so a reply interface hiccup is
# retried by the recovery scan instead of silencing the question.
_QUESTION_PUBLISHABLE_STATUSES = ("answered", "unknown", "failed_pending_publish")

ANSWER_CLAIM_TTL = timedelta(minutes=30)
# A background answer failure is retried in-process while the claim is fresh;
# without a pending-question table a process restart still loses an unanswered
# question, so the retry stays bounded and best-effort.
_ANSWER_ATTEMPTS = 3
_ANSWER_RETRY_DELAY_SECONDS = 5.0
# How often the recovery scan looks for unanswered questions or saved answers
# that were never published.
_QUESTION_RECOVERY_INTERVAL_SECONDS = 30.0
# Recovery drives different tasks in parallel; one task's records stay serial
# because the step holds that task's answer lock. Each step is time-boxed so a
# wedged await cannot stall the scan: two claim leases is far longer than a
# healthy answer, which the runner bounds with its own stall window.
_QUESTION_RECOVERY_CONCURRENCY = 4
_QUESTION_RECOVERY_TIMEOUT_SECONDS = ANSWER_CLAIM_TTL.total_seconds() * 2


# Shared claim namespace, kept apart from feedback ids on the same task row.
ANSWER_CLAIM_PREFIX = "mr-comment-answer:"
# A long explanation is still one comment; cut it before the platform rejects it.
ANSWER_LIMIT = 6000
TRUNCATED_SUFFIX = "…（已截断）"
# Sent when the task ended between the delivery being read and its question
# being stored; the author still gets a reply for the question they asked.
_CLOSED_QUESTION_NOTICE = "任务已结束，这个问题未生成回答"
# Replies are posted into the same thread as the triggering comment. When the
# trigger was an inline comment the reply inherits that position, so without
# this filter the acknowledgement would be read back as fresh review feedback
# and answer itself indefinitely.
# No longer posted, but threads written before this rule still contain it; it is
# service output and must never be quoted back as review context.
LEGACY_REJECTED_BODY = "该评论作者不是当前任务的发起人，已忽略。"
SELF_REPLY_BODIES = (ACK_BODY, LEGACY_REJECTED_BODY, ANSWER_PREFIX)
# A quoted parent is context, not the instruction. Long ones (a whole design
# note pasted into a thread) are cut so the feedback stays readable.
PARENT_CONTEXT_LIMIT = 2000

log = get_logger(__name__)


def _position_label(position: dict[str, Any] | None) -> str:
    if not position:
        return ""
    path = position.get("Path") or position.get("NewPath") or position.get("OldPath") or ""
    line = position.get("StartLine") or position.get("NewLine") or position.get("Line") or ""
    if path and line:
        return f"{path}:{line}"
    return str(path or "")


def command_of(body: str) -> str | None:
    """Return the Aeolus command the body starts with, if any.

    The command has to be a whole word: ``/codex`` is prose, not ``/code``.
    """
    text = body.strip()
    lowered = text.lower()
    for command in COMMANDS:
        if lowered == command:
            return command
        if lowered.startswith(command) and text[len(command)].isspace():
            return command
    return None


def is_reserved_command(event: CommentEvent) -> bool:
    """True when the body is a command owned by another domain.

    ``/resolve`` variants and the approval vocabulary are recognised so the
    Ai node never steals an operator command aimed at the conflict-resolution
    flow.
    """
    if event.command.lower().startswith("/resolve"):
        return True
    parsed = parse_comment(event.body, in_discussion=event.discussion_id is not None)
    return not isinstance(parsed, Ignored)


def feedback_text(event: CommentEvent) -> str:
    """Render one comment as feedback lines the coding agent can act on."""
    body = event.command
    command = command_of(body)
    if command is not None:
        body = body[len(command) :].strip()
    header = f"MR 评论（{event.author_email or 'unknown'}）"
    label = _position_label(event.position)
    if label:
        header = f"{header} @ {label}"
    blocks = [f"{header}：\n{body}" if body else header]
    parent = _parent_block(event.parent)
    if parent:
        blocks.append(parent)
    return "\n\n".join(blocks)


def _parent_block(parent: ThreadComment | None) -> str:
    """Quote the answered comment so the agent sees the request behind the reply."""
    if parent is None:
        return ""
    body = parent.body.strip()
    if not body:
        return ""
    if len(body) > PARENT_CONTEXT_LIMIT:
        body = f"{body[:PARENT_CONTEXT_LIMIT]}…（已截断）"
    return f"被回复的评论（{parent.author_email or 'unknown'}）：\n{body}"


def quote_block(event: CommentEvent) -> str:
    """Render the commented location and text as a Markdown quote.

    A thread id keeps a reply inside the same discussion, but the platform has
    no back-reference of its own: in the comment list the reply sits next to the
    question with nothing saying which file and line it answers.
    """
    quoted = (event.parent.body if event.parent is not None else event.command).strip()
    if event.parent is None:
        command = command_of(quoted)
        if command is not None and quoted[len(command) :].strip():
            quoted = quoted[len(command) :].strip()
    if not quoted:
        return ""
    if len(quoted) > PARENT_CONTEXT_LIMIT:
        quoted = f"{quoted[:PARENT_CONTEXT_LIMIT]}…（已截断）"
    label = _position_label(event.position)
    lines = [f"> 引用 @ `{label}`" if label else "> 引用"]
    lines.extend(f"> {line}".rstrip() if line.strip() else ">" for line in quoted.splitlines())
    return "\n".join(lines)


def _answer_head(event: CommentEvent) -> str:
    """Visible opening of an answer: the loop guard plus the quoted question.

    It carries no id. A bare "#6" reads as a reference a reader can follow, yet
    a question record has no page to open, so the id would only invite a click
    that lands nowhere; it travels in the hidden marker below instead.
    """
    quote = quote_block(event)
    return f"{ANSWER_PREFIX}\n{quote}" if quote else ANSWER_PREFIX


def _answer_marker(event: CommentEvent) -> str:
    """Hidden identity tying a posted answer to the question that asked it.

    Two questions quoting the same parent text must not look already-published
    to each other, so the marker carries the question's own comment id. It is
    an HTML comment: the renderer drops it, while the stored markdown the
    publish probe reads back keeps it.
    """
    if not event.comment_id:
        return ""
    return f"<!-- conciliator:comment:{event.comment_id} -->"


def _answer_comment_body(event: CommentEvent, answer: str) -> str:
    """Fit the visible head, the answer, and the hidden marker into one comment.

    The head carries the quote, so its length is not known until the event is;
    budgeting here keeps a long quote plus a long answer from producing a body
    the platform rejects, which no retry could ever publish.
    """
    head = _answer_head(event)
    marker = _answer_marker(event)
    tail = f"\n\n{marker}" if marker else ""
    limit = ANSWER_LIMIT - len(tail)
    if len(head) > limit:
        keep = max(limit - len(TRUNCATED_SUFFIX), 0)
        head = f"{head[:keep]}{TRUNCATED_SUFFIX}"
    room = limit - len(head) - len("\n\n")
    body = answer.strip()
    if len(body) > room:
        # Only a truncated body pays for the suffix; reserving it up front
        # would shorten an answer that already fits.
        keep = room - len(TRUNCATED_SUFFIX)
        body = f"{body[:keep]}{TRUNCATED_SUFFIX}" if keep > 0 else ""
    return f"{head}\n\n{body}{tail}" if body else f"{head}{tail}"


def is_self_reply_body(body: str) -> bool:
    return body.strip().startswith(SELF_REPLY_BODIES)


def is_self_reply(event: CommentEvent) -> bool:
    """True for the acknowledgement notes this service posts itself."""
    return is_self_reply_body(event.body)


def select_parent(
    comments: Sequence[ThreadComment], comment_id: str
) -> ThreadComment | None:
    """Return the comment that ``comment_id`` answers.

    Codebase threads carry no explicit parent pointer, so the answer is the
    previous comment in creation order. Our own acknowledgements are skipped:
    answering one of them means the reviewer is talking to the service, and the
    context worth quoting is the human comment before it.
    """
    previous: ThreadComment | None = None
    for comment in comments:
        if comment.comment_id == comment_id:
            return previous
        if is_self_reply_body(comment.body):
            continue
        previous = comment
    # The trigger is normally the newest comment, but a slightly stale snapshot
    # may not contain it yet; the newest human comment is the best guess then.
    return previous


@dataclass
class _PendingBatch:
    # Keyed by comment id so a redelivery inside the window is folded into the
    # same round once. Every id is claimed atomically when the batch flushes.
    entries: dict[str, str] = field(default_factory=dict)
    flusher: asyncio.Task[None] | None = None
    durable: bool = False


class AeolusCommentService:
    def __init__(
        self,
        *,
        service: Any,
        window_seconds: float = 60.0,
        defer_seconds: float = 30.0,
        retry_timeout_seconds: float = 900.0,
        max_deferrals: int | None = None,
        shutdown_timeout_seconds: float = 60.0,
    ) -> None:
        self._service = service
        self._window_seconds = window_seconds
        self._defer_seconds = defer_seconds
        if max_deferrals is None:
            if defer_seconds <= 0:
                raise ValueError("defer_seconds must be positive when deriving retry budget")
            # Keep retrying beyond the longest Coco round. The extra attempt runs
            # one interval after the timeout boundary, avoiding a race with the
            # state transition that makes feedback admissible again.
            max_deferrals = math.ceil(retry_timeout_seconds / defer_seconds) + 2
        self._max_deferrals = max_deferrals
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._pending: dict[str, _PendingBatch] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._flushers: set[asyncio.Task[None]] = set()
        self._closing = asyncio.Event()

    async def enqueue(self, *, task: AeolusTask, event: CommentEvent) -> None:
        text = feedback_text(event)
        persist = getattr(self._service, "save_pending_feedback", None)
        if persist is not None:
            # Persist before creating the batch: the caller acknowledges the
            # comment only after this returns, so a missing table fails the
            # webhook instead of leaving an acknowledged comment in memory.
            await persist(
                task.task_id,
                [(event.comment_id, text)],
                available_at=datetime.now(UTC)
                + timedelta(seconds=self._window_seconds),
            )
        batch = self._pending.get(task.task_id)
        if batch is None:
            batch = _PendingBatch(durable=persist is not None)
            self._pending[task.task_id] = batch
        # Keyed by comment id: a redelivery inside the window must not append
        # the same feedback twice.
        batch.entries.setdefault(event.comment_id, text)
        # The window is anchored on the first comment of the batch. Re-arming it
        # on every comment would let a steady stream postpone the flush forever.
        if batch.flusher is None or batch.flusher.done():
            batch.flusher = self._spawn_flusher(self._flush_later(task.task_id))

    async def aclose(self) -> None:
        # Wake batches still inside the aggregation window and let feedback
        # flushers finish while AeolusService is alive. Cancelling them here
        # would discard comments for which the webhook already sent an ACK.
        self._closing.set()
        flushers = list(self._flushers)
        if flushers:
            _, unfinished = await asyncio.wait(
                flushers, timeout=self._shutdown_timeout_seconds
            )
            if unfinished:
                log.warning(
                    "aeolus_comment_shutdown_flush_timeout",
                    timeout_seconds=self._shutdown_timeout_seconds,
                    flushers=len(unfinished),
                )
                for task in unfinished:
                    task.cancel()
                await asyncio.gather(*unfinished, return_exceptions=True)
        pending = list(self._tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._pending.clear()

    async def drain(self) -> None:
        """Wait for the flushes scheduled so far without cancelling them."""
        pending = list(self._tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def spawn(self, operation: Awaitable[None]) -> asyncio.Task[None]:
        """Run a comment-driven operation after the webhook response returns.

        Answering a question is a full agent run; awaiting it in the handler
        would hold the webhook open for minutes and make the platform retry the
        delivery.
        """
        return self._spawn(operation)

    def _spawn(self, operation: Awaitable[None]) -> asyncio.Task[None]:
        task = asyncio.ensure_future(operation)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _spawn_flusher(self, operation: Awaitable[None]) -> asyncio.Task[None]:
        task = self._spawn(operation)
        self._flushers.add(task)
        task.add_done_callback(self._flushers.discard)
        return task

    async def _flush_later(self, task_id: str) -> None:
        with suppress(TimeoutError):
            await asyncio.wait_for(
                self._closing.wait(), timeout=self._window_seconds
            )
        batch = self._pending.pop(task_id, None)
        if batch is None:
            return
        entries: dict[str, str] = dict(batch.entries)
        due = getattr(self._service, "due_pending_feedback", None)
        if due is not None:
            # The shared table is the batch: comments that landed on another
            # worker inside the same window carry the same deadline, so reading
            # it here merges them into one round instead of racing two rounds.
            try:
                entries = dict(await due(task_id))
            except Exception as exc:
                log.warning(
                    "aeolus_comment_pending_read_failed",
                    task_id=task_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
        if not entries:
            return
        feedback = "\n\n".join(entries.values())
        request_ids = [f"mr-comment:{comment_id}" for comment_id in entries]
        for attempt in range(self._max_deferrals):
            try:
                await self._service.submit_feedback(
                    task_id, feedback, request_ids, entries=list(entries.items())
                )
                await self._discard_pending(task_id, list(entries))
                log.info(
                    "aeolus_comment_feedback_submitted",
                    task_id=task_id,
                    request_ids=request_ids,
                    comments=len(entries),
                    attempt=attempt + 1,
                )
                return
            except InvalidTaskTransition:
                # The previous round is still running; keep the feedback until
                # the task is back in review or the deferral budget runs out.
                if attempt + 1 < self._max_deferrals:
                    await asyncio.sleep(self._defer_seconds)
            except Exception as exc:
                # A transient failure must not silently discard the comments the
                # receipt already promised to process.
                log.warning(
                    "aeolus_comment_feedback_retry_failed",
                    task_id=task_id,
                    request_ids=request_ids,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if attempt + 1 < self._max_deferrals:
                    await asyncio.sleep(self._defer_seconds)
        if batch.durable:
            # The row is still in the database; the recovery scan retries it
            # after the task becomes reviewable instead of losing the comment.
            log.warning(
                "aeolus_comment_feedback_deferred_to_recovery",
                task_id=task_id,
                request_ids=request_ids,
                comments=len(entries),
            )
            return
        log.warning(
            "aeolus_comment_feedback_dropped",
            task_id=task_id,
            request_ids=request_ids,
            comments=len(entries),
        )

    async def _discard_pending(
        self, task_id: str, comment_ids: Sequence[str]
    ) -> None:
        discard = getattr(self._service, "discard_pending_feedback", None)
        if not comment_ids or discard is None:
            return
        try:
            await discard(task_id, list(comment_ids))
        except Exception as exc:
            # The round already exists; a leftover row is replayed idempotently
            # by the recovery scan, so it must not fail the successful submit.
            log.warning(
                "aeolus_comment_pending_discard_failed",
                task_id=task_id,
                error=f"{type(exc).__name__}: {exc}",
            )


class AeolusCommentHandler:
    """Route review comments on an Aeolus MR into a new coding round."""

    def __init__(
        self,
        *,
        registry: AeolusTaskRegistry,
        comments: AeolusCommentService,
        meegle: Any,
        allowlist: Iterable[str] = (),
        reply: Callable[[CommentEvent, str], Awaitable[None]] | None = None,
        thread_reader: (
            Callable[[str, str], Awaitable[Sequence[ThreadComment]]] | None
        ) = None,
        answerer: Callable[..., Awaitable[str]] | None = None,
        resumer: Callable[[str, str, str, str], Awaitable[str]] | None = None,
    ) -> None:
        self._registry = registry
        self._comments = comments
        self._meegle = meegle
        self._allowlist = {item.strip().lower() for item in allowlist if item.strip()}
        self._reply = reply
        self._thread_reader = thread_reader
        self._answerer = answerer
        self._resumer = resumer
        # Redelivery must not answer the same comment twice. Bounded so a long
        # lived process cannot grow this set without limit.
        self._answered: OrderedDict[str, None] = OrderedDict()
        # One session per task: two answers must not run concurrently on it.
        self._locks: dict[str, asyncio.Lock] = {}
        self._recovery_worker: asyncio.Task[None] | None = None

    async def matches(self, event: CommentEvent) -> bool:
        if not event.repo_id or not event.mr_iid:
            return False
        if is_self_reply(event):
            return False
        if is_reserved_command(event):
            return False
        # Both re-coding and answers are explicit commands. Anything else stays
        # ordinary review discussion, even on an MR this service created.
        return command_of(event.command) is not None

    async def handle(self, event: CommentEvent) -> None:
        if is_self_reply(event):
            return
        task = await self._task_for(event)
        command = command_of(event.command)
        # A question only needs a live task: it is stored and answered once the
        # task is answerable again. A coding command needs a reviewable state.
        if task is None or (
            command != QUESTION_COMMAND
            and task.state
            not in {
                AeolusState.REVIEW,
                AeolusState.CODING_BLOCKED,
                AeolusState.RECODING,
            }
        ):
            log.info(
                "aeolus_comment_no_active_task",
                repo_id=event.repo_id,
                mr_iid=event.mr_iid,
                comment_id=event.comment_id,
            )
            return
        if command == RECODE_COMMAND:
            # A coding round spends the initiator's work item, so only they (or
            # an allowlisted reviewer) may start one. Ignore silently: replying
            # would notify whoever commented on the MR even though this flow is
            # not theirs to drive.
            if not await self._author_allowed(task, event):
                log.info(
                    "aeolus_comment_author_not_allowed",
                    task_id=task.task_id,
                    author=event.author_email,
                )
                return
            await self._start_round(task, event)
            return
        if command == QUESTION_COMMAND:
            # Questions change nothing, so any MR participant may ask one; the
            # answer goes back in the same thread.
            await self._answer(task, event)

    async def _start_round(self, task: AeolusTask, event: CommentEvent) -> None:
        if not feedback_text(event).strip():
            return
        event = await self._with_parent(event)
        await self._comments.enqueue(task=task, event=event)
        quote = quote_block(event)
        await self._send_reply(event, f"{ACK_BODY}\n\n{quote}" if quote else ACK_BODY)

    async def _answer(self, task: AeolusTask, event: CommentEvent) -> None:
        """Answer a review question without touching code or task state."""
        if self._answerer is None:
            # Without an answerer a stored question could never be answered;
            # log so the misconfiguration is visible instead of silent.
            log.warning(
                "aeolus_comment_question_no_answerer",
                task_id=task.task_id, comment_id=event.comment_id,
            )
            return
        # A live run owns the Coco session. The question is recorded like any
        # other delivery and left to the recovery scan, which answers it once
        # the task is reviewable again: the delivery is acknowledged now, so a
        # question asked mid-round cannot depend on a platform redelivery.
        event = await self._with_parent(event)
        # Persist before the webhook acknowledges the delivery: an answer that
        # later fails (agent error, publish failure, worker crash) is recovered
        # from this record instead of being lost with the acknowledged delivery.
        # The record keeps the payload of the first delivery: a redelivery can
        # resolve a different parent comment, and rewriting it would change the
        # answer marker, breaking the "already published" thread check.
        record = await self._registry.question_record(task.task_id, event.comment_id)
        if record is None:
            record = await self._registry.record_question(
                task.task_id,
                event.comment_id,
                repo_id=event.repo_id,
                mr_iid=event.mr_iid,
                discussion_id=event.discussion_id,
                position=event.position,
                parent=asdict(event.parent) if event.parent is not None else None,
                # The raw command, not the agent-facing feedback rendering: the
                # record is rebuilt into an event for republishing, and the
                # answer marker must quote the same text the first publish did.
                question_text=event.command,
                author_email=event.author_email,
                # Completion can win the row lock first; the same transaction
                # then stores this notice so the acknowledged delivery still
                # has something the publish scan can deliver.
                notice_text=_CLOSED_QUESTION_NOTICE,
            )
        if record is None:
            return
        if record["status"] in _QUESTION_PUBLISHABLE_STATUSES:
            # The body is saved but was never published; republish it instead of
            # spending another agent run on the same question.
            self._comments.spawn(self._publish_question(record))
            return
        if record is not None and record["status"] not in _QUESTION_ACTIVE_STATUSES:
            return
        self._comments.spawn(self._answer_later(task, event))

    async def _answer_later(
        self, task: AeolusTask, event: CommentEvent
    ) -> None:
        # Normal delivery and recovery use the same claim and re-read path.
        # In particular, a redelivery must not resend an abandoned sending run.
        record = await self._registry.question_record(
            task.task_id, event.comment_id
        )
        if record is not None:
            await self._resume_question(record)

    async def _answer_with_retries(
        self, task: AeolusTask, event: CommentEvent, token: datetime | None, record: dict
    ) -> str:
        """Retry only failures whose previous run is confirmed terminal."""
        answerer = self._answerer
        if answerer is None:
            raise RuntimeError("no answerer configured")
        for attempt in range(_ANSWER_ATTEMPTS):
            send_attempt = record["attempts"] + 1
            try:
                return await self._question_with_claim(
                    record, token, answerer(
                        task,
                        feedback_text(event),
                        _position_label(event.position),
                        on_started=lambda session_id, run_id, event_id, ordinal=send_attempt: self._track_answer_start(
                            record, token, ordinal, session_id, run_id, event_id
                        ),
                    ),
                )
            except Exception as exc:
                retryable = (
                    isinstance(exc, CocoRunFailed)
                    and exc.retryable and exc.replacement_safe
                )
                log.warning(
                    "aeolus_comment_answer_failed",
                    task_id=task.task_id,
                    comment_id=event.comment_id,
                    attempt=attempt + 1,
                    retryable=retryable,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if not retryable or attempt + 1 >= _ANSWER_ATTEMPTS:
                    raise
                await asyncio.sleep(_ANSWER_RETRY_DELAY_SECONDS)
        # Every attempt returns or raises, so this is only reachable when the
        # attempt budget is configured as zero.
        raise RuntimeError("no answer attempt was made")

    async def _track_answer_start(
        self, record: dict, token: datetime | None, attempt: int,
        session_id: str, run_id: str, event_id: str,
    ) -> None:
        if not run_id:
            changed = await self._registry.mark_question_sending(
                record["task_id"], record["comment_id"], token=token,
                attempt=attempt - 1, session_id=session_id,
            )
        else:
            changed = await self._registry.mark_question_answering(
                record["task_id"], record["comment_id"], token=token, attempt=attempt,
                session_id=session_id, run_id=run_id, event_id=event_id,
            )
        if not changed:
            raise TaskLeaseLost("question claim or sending attempt was lost")
        # Mutate only after the CAS succeeds; the caller uses this same attempt
        # identity for answer persistence and all terminal writes.
        record.update(
            status="answering" if run_id else "sending", attempts=attempt,
            session_id=session_id, run_id=run_id or None, event_id=event_id or None,
        )

    async def _publish_answer_body(
        self, record: dict, event: CommentEvent, token: datetime | None, answer: str
    ) -> bool:
        """Persist and publish a generated answer; True when the question is settled.

        The answer is saved before the reply: if publishing fails, the recovery
        scan republishes the saved text without spending another agent run.
        """
        body = answer.strip() if answer else ""
        if not body:
            log.warning(
                "aeolus_comment_answer_empty",
                task_id=record["task_id"],
                comment_id=event.comment_id,
            )
            # Persist the notice instead of writing a terminal status: a reply
            # interface hiccup must not leave the author without an answer and
            # without the notice that would ask them to try again.
            return await self._close_failed(record, token, "回答为空，请重新提问")
        if record["status"] != "answered" and not await self._persist_answer_body(record, token, answer):
            return False
        # A real service has already committed the answer under its task lease.
        # Publishing consumes that durable result, including after a restart.
        return await self._publish_claimed_question(record, token)

    async def _persist_answer_body(
        self, record: dict, token: datetime | None, answer: str, *, require_task_lease: bool = False,
    ) -> bool:
        body = answer.strip() if answer else ""
        if not body:
            return False
        if len(body) > ANSWER_LIMIT:
            body = f"{body[: ANSWER_LIMIT - len(TRUNCATED_SUFFIX)]}{TRUNCATED_SUFFIX}"
        changed = await self._registry.mark_question_answered(
            record["task_id"], record["comment_id"], body, token=token,
            attempt=record["attempts"], require_task_lease=require_task_lease,
        )
        if not changed:
            if require_task_lease:
                raise TaskLeaseLost("task or question lease was lost before answer acceptance")
            return False
        record.update(status="answered", answer_text=body)
        return True

    async def _close_unknown(self, record: dict, token: datetime | None) -> bool:
        notice = "上一次回答的发送结果无法确认，请重新提问"
        if not await self._registry.mark_question_unknown(
            record["task_id"], record["comment_id"], token=token,
            attempt=record["attempts"], notice=notice,
        ):
            return False
        record.update(status="unknown", answer_text=notice)
        return await self._publish_claimed_question(record, token)

    async def _close_failed(
        self, record: dict, token: datetime | None, notice: str
    ) -> bool:
        """Save the failure notice as the publishable body, then publish it.

        Writing the terminal failed status before the notice is visible would
        silence the question for good: failed records are never scanned again.
        The notice therefore uses the same durable publish path as an answer,
        and only a confirmed publish moves the record to failed.
        """
        if not await self._registry.mark_question_failed_notice(
            record["task_id"], record["comment_id"], token=token,
            attempt=record["attempts"], notice=notice,
        ):
            return False
        record.update(status="failed_pending_publish", answer_text=notice)
        return await self._publish_claimed_question(record, token)

    async def _task_is_busy(self, task_id: str) -> bool:
        """Whether a coding round currently owns the Coco session.

        Re-read from the database rather than the in-memory task: the answer
        may have waited behind a lock while another worker started a round.
        A vanished task is not busy — the caller's terminal handling covers it.
        """
        get = getattr(self._registry, "get", None)
        if get is None:
            return False
        try:
            task = await get(task_id)
        except TaskNotFound:
            return False
        return task is not None and task.state in _SESSION_BUSY_STATES

    async def _task_for(self, event: CommentEvent) -> AeolusTask | None:
        # A lookup failure must fail the webhook delivery: an acknowledged
        # comment cannot be redelivered and would be lost forever.
        return await self._registry.find_active_for_mr(
            event.repo_id, event.mr_iid
        )

    async def _claim_question(
        self, task_id: str, comment_id: str
    ) -> datetime | None:
        if not comment_id:
            return datetime.now(UTC)
        claim = getattr(self._registry, "claim_comment", None)
        if claim is not None:
            # A redelivery can land on another worker; only the process that
            # wins the shared claim may spend an agent run on this comment.
            return await claim(
                task_id, f"{ANSWER_CLAIM_PREFIX}{comment_id}", ANSWER_CLAIM_TTL
            )
        if comment_id in self._answered:
            return None
        self._answered[comment_id] = None
        while len(self._answered) > 256:
            self._answered.popitem(last=False)
        return datetime.now(UTC)

    async def _release_question(
        self, task_id: str, comment_id: str, token: datetime | None
    ) -> None:
        if token is None:
            return
        release = getattr(self._registry, "release_comment", None)
        if release is None:
            self._answered.pop(comment_id, None)
            return
        try:
            await release(task_id, f"{ANSWER_CLAIM_PREFIX}{comment_id}", token)
        except Exception as exc:
            log.warning(
                "aeolus_comment_claim_release_failed",
                task_id=task_id,
                comment_id=comment_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _complete_question(
        self, task_id: str, comment_id: str, token: datetime | None
    ) -> bool:
        complete = getattr(self._registry, "complete_comment", None)
        if complete is None:
            return True
        try:
            return bool(
                await complete(task_id, f"{ANSWER_CLAIM_PREFIX}{comment_id}", token)
            )
        except Exception as exc:
            log.warning(
                "aeolus_comment_claim_complete_failed",
                task_id=task_id,
                comment_id=comment_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

    async def _holds_question(
        self, task_id: str, comment_id: str, token: datetime | None
    ) -> bool:
        """Prove this worker still owns the comment before publishing.

        A published reply cannot be withdrawn, so a worker that stalled past
        its lease must stay silent instead of answering over the new owner.
        """
        renew = getattr(self._registry, "renew_comment", None)
        if token is None:
            return False
        if renew is None:
            return True
        try:
            return bool(
                await renew(
                    task_id,
                    f"{ANSWER_CLAIM_PREFIX}{comment_id}",
                    ANSWER_CLAIM_TTL,
                    token,
                )
            )
        except Exception as exc:
            log.warning(
                "aeolus_comment_claim_renew_failed",
                task_id=task_id,
                comment_id=comment_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

    async def _answer_already_published(self, event: CommentEvent) -> bool | None:
        """Look for our own reply in the thread before publishing another one.

        The claim only outlives a crash as an expiring lease, so the thread is
        the durable evidence that an earlier attempt already answered. A failed
        read returns None: it proves nothing either way, so the caller keeps the
        saved body publishable instead of repeating the reply.
        """
        if self._thread_reader is None or not event.discussion_id:
            return False
        try:
            comments = await self._thread_reader(event.repo_id, event.discussion_id)
        except Exception as exc:
            log.warning(
                "aeolus_comment_thread_read_failed",
                repo_id=event.repo_id,
                discussion_id=event.discussion_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
        marker = _answer_marker(event)
        if not marker:
            # Nothing to match on, so a second reply beats reading every
            # comment in the thread as our own.
            return False
        return any(marker in comment.body for comment in comments)

    async def _renew_question(
        self, task_id: str, comment_id: str, token: datetime | None
    ) -> None:
        """Keep the answer lease alive while a slow agent is still working.

        Without this a question that takes longer than the lease lets another
        worker claim it and answer concurrently.
        """
        renew = getattr(self._registry, "renew_comment", None)
        if renew is None:
            return
        interval = ANSWER_CLAIM_TTL.total_seconds() / 3
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await renew(
                    task_id,
                    f"{ANSWER_CLAIM_PREFIX}{comment_id}",
                    ANSWER_CLAIM_TTL,
                    token,
                )
            except Exception as exc:
                log.warning(
                    "aeolus_comment_claim_renew_failed",
                    task_id=task_id,
                    comment_id=comment_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise TaskLeaseLost("question claim renewal failed") from exc
            if not renewed:
                raise TaskLeaseLost("question claim expired")

    async def _question_with_claim(
        self, record: dict, token: datetime | None, operation: Awaitable[str]
    ) -> str:
        """Stop driving a question when its shared claim can no longer renew."""
        if getattr(self._registry, "renew_comment", None) is None:
            return await operation
        worker = asyncio.ensure_future(operation)
        renewal = asyncio.create_task(
            self._renew_question(record["task_id"], record["comment_id"], token)
        )
        try:
            done, _ = await asyncio.wait({worker, renewal}, return_when=asyncio.FIRST_COMPLETED)
            if renewal in done:
                await renewal
                raise TaskLeaseLost("question claim renewal stopped")
            return await worker
        finally:
            worker.cancel()
            renewal.cancel()
            await asyncio.gather(worker, renewal, return_exceptions=True)

    async def _with_parent(self, event: CommentEvent) -> CommentEvent:
        parent = await self._parent_comment(event)
        return event if parent is None else replace(event, parent=parent)

    async def _parent_comment(self, event: CommentEvent) -> ThreadComment | None:
        if self._thread_reader is None or not event.discussion_id:
            return None
        try:
            comments = await self._thread_reader(event.repo_id, event.discussion_id)
        except Exception as exc:
            # Losing the quoted context is acceptable, losing the trigger is not;
            # this read can fail on rate limits or transient platform errors.
            log.warning(
                "aeolus_comment_parent_fetch_failed",
                repo_id=event.repo_id,
                discussion_id=event.discussion_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
        return select_parent(comments, event.comment_id)

    async def _author_allowed(self, task: AeolusTask, event: CommentEvent) -> bool:
        author = event.author_email.strip().lower()
        if not author:
            return False
        if author in self._allowlist:
            return True
        initiator = await self._initiator_email(task)
        return bool(initiator) and initiator == author

    async def _initiator_email(self, task: AeolusTask) -> str:
        try:
            email = await self._meegle.get_user_email(task.user_key)
        except Exception as exc:  # the directory is best-effort; deny on failure
            log.warning(
                "aeolus_comment_initiator_lookup_failed",
                task_id=task.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return ""
        return (email or "").strip().lower()

    async def _send_reply(self, event: CommentEvent, body: str) -> bool:
        if self._reply is None:
            return True
        try:
            await self._reply(event, body)
        except Exception as exc:
            log.warning(
                "aeolus_comment_reply_failed",
                repo_id=event.repo_id,
                mr_iid=event.mr_iid,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False
        return True

    def _record_event(self, record: dict) -> CommentEvent:
        """Rebuild the comment event a question record was created from."""
        parent = None
        if record.get("parent"):
            parent = ThreadComment(**record["parent"])
        return CommentEvent(
            comment_id=record["comment_id"],
            repo_id=record["repo_id"],
            mr_iid=record["mr_iid"],
            body=record["question_text"],
            author_email=record["author_email"],
            discussion_id=record["discussion_id"],
            position=record["position"],
            parent=parent,
        )

    async def _publish_question(self, record: dict) -> None:
        """Publish a saved answer that a previous attempt could not deliver.

        The claim, not the thread check, decides who may send: a worker that
        died mid-publish keeps its claim only until the lease expires, and the
        reading of the thread below happens after the claim is taken.  Without
        that, a worker still sending and a scan that cannot see the reply yet
        would both post the answer.
        """
        task_id = record["task_id"]
        comment_id = record["comment_id"]
        token = await self._claim_question(task_id, comment_id)
        if token is None:
            return
        published = False
        try:
            current = await self._registry.question_record(task_id, comment_id)
            if (
                current is None
                or current["status"] not in _QUESTION_PUBLISHABLE_STATUSES
                or current.get("answer_text") is None
            ):
                # Another worker already published, failed or resumed it.
                return
            published = await self._publish_claimed_question(current, token)
            # A failed publish keeps the record publishable; the next scan
            # retries it under a fresh claim.
        finally:
            if published:
                await self._complete_question(task_id, comment_id, token)
            else:
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(
                        self._release_question(task_id, comment_id, token)
                    )

    async def _publish_claimed_question(self, record: dict, token: datetime | None) -> bool:
        """Publish a durable answer or unknown-send notice under the same claim."""
        task_id, comment_id = record["task_id"], record["comment_id"]
        if not await self._holds_question(task_id, comment_id, token):
            return False
        event = self._record_event(record)
        already_published = await self._answer_already_published(event)
        # Reading the thread may have waited past our expiry.
        if not await self._holds_question(task_id, comment_id, token):
            return False
        if already_published is None:
            # An unreadable thread cannot prove the reply is missing, so keep
            # the saved body publishable: the next scan looks again instead of
            # sending the author a second copy.
            log.info(
                "aeolus_comment_publish_probe_unknown",
                task_id=task_id,
                comment_id=comment_id,
            )
            return False
        if not already_published and not await self._send_reply(
            event,
            _answer_comment_body(event, record["answer_text"] or ""),
        ):
            return False
        try:
            return bool(await self._registry.mark_question_published(
                task_id, comment_id, token=token,
                attempt=record["attempts"], status=record["status"],
            ))
        except Exception as exc:
            # Leave the saved body and release the claim. A scan can reconcile
            # the visible reply; completing the claim here would block recovery.
            log.warning(
                "aeolus_comment_question_publish_mark_failed",
                task_id=task_id, comment_id=comment_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

    async def resume_pending_questions(self) -> None:
        """Drive questions that still owe an answer or a publish.

        Called at startup and periodically: published and failed records are
        terminal and are never touched again.
        """
        if getattr(self._registry, "question_records_to_publish", None) is None:
            # A registry without the question-records table cannot persist
            # anything; the scan would only raise, so stay quiet instead.
            return
        slots = asyncio.Semaphore(_QUESTION_RECOVERY_CONCURRENCY)
        await self._drive_recovery(
            await self._registry.question_records_to_publish(),
            self._publish_question,
            "aeolus_comment_question_publish_recovery_failed",
            slots,
        )
        await self._drive_recovery(
            await self._registry.question_records_to_resume(),
            self._resume_question,
            "aeolus_comment_question_resume_failed",
            slots,
        )

    async def _drive_recovery(
        self,
        records: Sequence[dict],
        step: Callable[[dict], Awaitable[None]],
        event: str,
        slots: asyncio.Semaphore,
    ) -> None:
        """Run one recovery step per record with bounded concurrency.

        Different tasks overlap; one task's records stay serial because the
        step holds that task's answer lock. Each step is time-boxed so a wedged
        await cannot stall the scan.
        """
        if not records:
            return

        async def run(record: dict) -> None:
            async with slots:
                try:
                    async with asyncio.timeout(_QUESTION_RECOVERY_TIMEOUT_SECONDS):
                        await step(record)
                except Exception as exc:
                    log.warning(
                        event,
                        comment_id=record["comment_id"],
                        error=f"{type(exc).__name__}: {exc}",
                    )

        await asyncio.gather(*(run(record) for record in records))

    async def _resume_question(self, record: dict) -> None:
        """Continue a question a crashed worker left pending or answering."""
        task_id = record["task_id"]
        comment_id = record["comment_id"]
        event = self._record_event(record)
        lock = self._locks.setdefault(task_id, asyncio.Lock())
        answered = False
        token: datetime | None = None
        current = None
        try:
            try:
                async with lock:
                    token = await self._claim_question(task_id, comment_id)
                    if token is None:
                        # Another worker already took this question over.
                        return
                    # Re-read under the claim: the scan's view is stale by the
                    # time this worker owns the question, and a round that
                    # started meanwhile must defer it, not fail it for good.
                    current = await self._registry.question_record(task_id, comment_id)
                    if (
                        current is None
                        or current["status"] not in _QUESTION_ACTIVE_STATUSES
                    ):
                        return
                    event = self._record_event(current)
                    if current["status"] == "sending" and not current.get("run_id"):
                        # Only an owner of the re-read record can close an
                        # abandoned send. A live sender keeps its claim, and a
                        # stale pending snapshot cannot cause a second send.
                        answered = await self._close_unknown(current, token)
                        if answered:
                            await self._complete_question(task_id, comment_id, token)
                        return
                    try:
                        task = await self._registry.get(task_id)
                    except TaskNotFound:
                        # The question was already acknowledged to its author, so
                        # finishing the record without a visible reply would drop
                        # it silently.
                        answered = await self._close_failed(
                            current, token, "任务已结束，这个问题未生成回答"
                        )
                        if answered:
                            await self._complete_question(task_id, comment_id, token)
                        return
                    if task.state in _SESSION_BUSY_STATES:
                        log.info(
                            "aeolus_comment_question_deferred",
                            task_id=task_id,
                            comment_id=comment_id,
                            state=task.state.value,
                        )
                        return
                    if task.state.value in QUESTION_TERMINAL_STATES:
                        # A finished task never resumes its questions, so publish
                        # the termination notice here rather than closing the
                        # record into silence.
                        answered = await self._close_failed(
                            current, token, "任务已结束，这个问题未生成回答"
                        )
                        if answered:
                            await self._complete_question(task_id, comment_id, token)
                        return
                    if task.state not in {
                        AeolusState.REVIEW,
                        AeolusState.CODING_BLOCKED,
                    }:
                        # The task is alive but not answerable yet (reading the
                        # document, validating, ...): keep the question queued
                        # for a later scan instead of answering or closing it.
                        return
                    with question_answer_acceptance(
                        lambda body: self._persist_answer_body(current, token, body, require_task_lease=True)
                    ):
                        answer = await self._resume_or_reanswer(
                            current, task, event, token
                        )
                    answered = await self._publish_answer_body(
                        current, event, token, answer
                    )
                    if answered:
                        await self._complete_question(task_id, comment_id, token)
            except (TaskLeaseLost, TransientError) as exc:
                # Keep the same cursor recoverable. Neither loss of ownership
                # nor a transient read proves that a replacement run is safe.
                log.info(
                    "aeolus_comment_question_deferred",
                    task_id=task_id, comment_id=comment_id,
                    error_type=type(exc).__name__,
                )
                return
            except Exception as exc:
                if current is not None and (
                    isinstance(exc, CocoSendOutcomeUnknown)
                    or (current["status"] == "sending" and not current.get("run_id"))
                ):
                    answered = await self._close_unknown(current, token)
                    if answered:
                        await self._complete_question(task_id, comment_id, token)
                    return
                if await self._task_is_busy(task_id):
                    log.info(
                        "aeolus_comment_question_deferred",
                        task_id=task_id,
                        comment_id=comment_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    return
                log.warning(
                    "aeolus_comment_question_resume_failed",
                    task_id=task_id,
                    comment_id=comment_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                # Only the current claim owner may record the failure notice; a
                # stale worker stays silent. Saving the notice as the
                # publishable body first lets the scan retry it, instead of a
                # terminal status silencing a question whose notice never landed.
                if current is not None:
                    answered = await self._close_failed(
                        current, token, "回答失败，请重新提问"
                    )
                    if answered:
                        await self._complete_question(task_id, comment_id, token)
                return
        finally:
            if not answered:
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(
                        self._release_question(task_id, comment_id, token)
                    )

    async def _resume_or_reanswer(
        self, record: dict, task: AeolusTask, event: CommentEvent, token: datetime | None
    ) -> str:
        """Resume the run a crashed worker started; re-answer when it is terminal."""
        resumer = self._resumer
        if record.get("run_id") and record.get("session_id"):
            if resumer is None:
                raise CocoRunFailed("no answer resumer configured")
            try:
                return await self._question_with_claim(
                    record, token, resumer(
                        task.task_id,
                        record["session_id"],
                        record["run_id"],
                        record["event_id"] or "",
                    ),
                )
            except CocoRunFailed as exc:
                if not exc.replacement_safe:
                    # The run's state is unknown; starting a replacement would
                    # race it.
                    raise
                # A terminal run cannot race a replacement; re-answer.
        return await self._answer_with_retries(task, event, token, record)

    async def start(self) -> None:
        """Start the recovery scan for unanswered questions.

        The first pass runs inside the background task: a pending question can
        wait for a whole agent answer, and application readiness (HTTP and
        webhooks) must not depend on a remote agent finishing.
        """
        if self._recovery_worker is not None:
            return

        async def scan() -> None:
            while True:
                try:
                    await self.resume_pending_questions()
                except Exception as exc:
                    log.warning(
                        "aeolus_comment_recovery_scan_failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                await asyncio.sleep(_QUESTION_RECOVERY_INTERVAL_SECONDS)

        self._recovery_worker = asyncio.create_task(
            scan(), name="aeolus:comment-question-recovery"
        )

    async def aclose(self) -> None:
        if self._recovery_worker is not None:
            self._recovery_worker.cancel()
            await asyncio.gather(self._recovery_worker, return_exceptions=True)
            self._recovery_worker = None
