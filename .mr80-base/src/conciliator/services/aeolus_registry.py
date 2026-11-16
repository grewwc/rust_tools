from __future__ import annotations

import copy
import hashlib
import secrets
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.engine import Connection, CursorResult, Engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.elements import ColumnElement
from starlette.concurrency import run_in_threadpool

from conciliator.domain.aeolus_task import (
    AeolusAttempt,
    AeolusRound,
    AeolusStartEvent,
    AeolusState,
    AeolusTask,
    DeploymentVerification,
    can_retry_coding,
)
from conciliator.repositories.models import (
    MeegleAINodeAttempt,
    MeegleAINodeCommentClaim,
    MeegleAINodePendingFeedback,
    MeegleAINodeQuestionRecord,
    MeegleAINodeRound,
    MeegleAINodeTask,
)
from conciliator.utils.logging import get_logger

log = get_logger(__name__)


def _log_transition(record: MeegleAINodeTask, previous: str, **fields) -> None:
    log.info(
        "meegle_task_state_changed", workflow="meegle", task_id=record.task_id,
        previous_state=previous, state=record.status,
        evaluation_round_no=len(record.evaluation_rounds or []),
        **fields,
    )


class TaskNotFound(KeyError):
    pass


class InvalidTaskTransition(RuntimeError):
    pass


class TaskLeaseLost(RuntimeError):
    pass


def _normalize_request_ids(request_ids: str | Sequence[str]) -> list[str]:
    """Return the non-empty, de-duplicated ids of one feedback batch."""
    ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
    if not ids or any(not request_id.strip() for request_id in ids):
        raise ValueError("request_id must not be empty")
    return list(dict.fromkeys(ids))


def _missing_table(exc: Exception) -> bool:
    """True when the failure is the mandatory pending feedback table."""
    message = str(getattr(exc, "orig", exc)).lower()
    return "no such table" in message or "doesn't exist" in message or "1146" in message


# Creating a task is serialized per node through a MySQL named lock.  MySQL
# caps lock names at 64 characters, so the node identity is digested instead
# of embedded verbatim.
_NODE_LOCK_TIMEOUT_SECONDS = 10

# A handled comment must never be claimed again. The claim table carries no
# status column, so completion is recorded by pushing the expiry past any
# realistic clock instead of deleting the dedup row.
COMPLETED_CLAIM_EXPIRY = datetime(9999, 12, 31, tzinfo=UTC)

# Question states that still owe work. Terminal writes are conditional on the
# record still being in one of these states, so a worker that lost its claim
# cannot overwrite an answer another worker already produced or published.
_QUESTION_ACTIVE_STATUSES = ("pending", "answering", "sending")

# Question states that still owe the author a visible reply. A task that
# completes while one of these is left behind can never answer it again, so
# completion is refused until the record is published or failed.
#
# "unknown" is deliberately absent: it is terminal, and its notice is
# republished by the publish scan, which needs no task session. Keeping it
# here made a task that once lost a send impossible to complete even after
# the notice had reached the author.
UNSETTLED_QUESTION_STATUSES = (
    "pending",
    "answering",
    "sending",
    "answered",
    "failed_pending_publish",
)

# States in which a task can no longer answer a question. Everything else can
# still take the insert: a working task queues the question for the recovery
# scan, and only a closing task refuses it.
QUESTION_TERMINAL_STATES = frozenset(
    {
        AeolusState.COMPLETING.value,
        AeolusState.COMPLETED.value,
        AeolusState.FAILED.value,
        AeolusState.VALIDATION_FAILED.value,
    }
)
_QUESTION_STORABLE_STATES = tuple(
    state.value for state in AeolusState if state.value not in QUESTION_TERMINAL_STATES
)


def _node_lock_name(event: AeolusStartEvent) -> str:
    identity = f"{event.project_key}\x00{event.work_item_id}\x00{event.node_uuid}"
    return f"aeolus-node:{hashlib.sha1(identity.encode()).hexdigest()}"


_lease_owner: ContextVar[str | None] = ContextVar("aeolus_lease_owner", default=None)


# Terminal states a running round can fall into. Completion is guarded
# separately: _begin_completion refuses while feedback is still pending.
_PENDING_FEEDBACK_TERMINAL_STATES = {
    AeolusState.FAILED,
    AeolusState.VALIDATION_FAILED,
}


@contextmanager
def task_lease(owner: str):
    token = _lease_owner.set(owner)
    try:
        yield
    finally:
        _lease_owner.reset(token)


def _database_now(session: Session) -> datetime:
    clock = func.utc_timestamp() if session.get_bind().dialect.name == "mysql" else func.current_timestamp()
    return cast(datetime, session.scalar(select(clock))).replace(tzinfo=None)


def _task_lease_is_live(session: Session) -> ColumnElement[bool]:
    """Evaluate expiry at the fenced UPDATE, using the database's UTC clock."""
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        # SQLite stores DATETIME as text with fractional seconds; compare
        # timestamps numerically rather than against CURRENT_TIMESTAMP text.
        return func.julianday(MeegleAINodeTask.lease_expires_at) > func.julianday("now")
    clock = func.utc_timestamp() if dialect == "mysql" else func.current_timestamp()
    return MeegleAINodeTask.lease_expires_at > clock


_TASK_UPDATE_FIELDS = {
    "props",
    "requirement_url",
    "requirement_text",
    "evaluation",
    "evaluation_rounds",
    "repo_id",
    "work_branch",
    "coco_session_id",
    "coco_run_id",
    "coco_last_event_id",
    "mr_id",
    "mr_iid",
    "mr_url",
    "platform_status_done",
    "last_error",
}
_ROUND_UPDATE_FIELDS = {
    "status",
    "feedback",
    "trigger_type",
    "reviewed_round_no",
    "base_commit_sha",
    "summary",
    "commit_sha",
    "code_url",
    "tests",
    "unresolved_items",
    "failure_code",
    "error",
    "started_at",
    "finished_at",
}
_ATTEMPT_UPDATE_FIELDS = {"status", "error", "finished_at"}


class AeolusTaskRegistry:
    """MySQL-backed task registry with row-level transition serialization."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        deployments_enabled: bool = False,
        ppe_enabled: bool = False,
        lock_engine: Engine | None = None,
    ) -> None:
        self._sf = session_factory
        self._deployments_enabled = deployments_enabled
        self._ppe_enabled = ppe_enabled
        # Named locks are held on a connection the task sessions never use;
        # tests inject a fake engine to exercise the serialization.
        self._lock_engine = lock_engine or session_factory.kw.get("bind")

    async def ensure_pending_feedback_schema(self) -> None:
        await run_in_threadpool(self._ensure_pending_feedback_schema)

    def _ensure_pending_feedback_schema(self) -> None:
        with self._sf() as session:
            try:
                session.scalar(select(MeegleAINodePendingFeedback.id).limit(1))
            except (OperationalError, ProgrammingError) as exc:
                if not _missing_table(exc):
                    raise
                raise RuntimeError(
                    "meegle_ai_node_pending_feedback is missing; run "
                    "`alembic upgrade head` or apply "
                    "ddl/0020_pending_comment_feedback.sql"
                ) from exc

    async def create_from_event(
        self, event: AeolusStartEvent
    ) -> tuple[AeolusTask, bool]:
        return await run_in_threadpool(self._create_from_event, event)

    async def get(self, task_id: str) -> AeolusTask:
        return await run_in_threadpool(self._get, task_id)

    async def acquire_lease(self, task_id: str, owner: str, seconds: int = 60) -> bool:
        return await run_in_threadpool(self._change_lease, task_id, owner, seconds, False)

    async def renew_lease(self, task_id: str, owner: str, seconds: int = 60) -> bool:
        return await run_in_threadpool(self._change_lease, task_id, owner, seconds, True)

    async def release_lease(self, task_id: str, owner: str) -> None:
        await run_in_threadpool(self._release_lease, task_id, owner)

    async def recoverable_task_ids(self) -> list[str]:
        return await run_in_threadpool(self._recoverable_task_ids)

    def _change_lease(self, task_id: str, owner: str, seconds: int, renew: bool) -> bool:
        with self._sf() as session:
            now = _database_now(session)
            available = (
                (MeegleAINodeTask.lease_owner == owner)
                & _task_lease_is_live(session)
                if renew else or_(
                    MeegleAINodeTask.lease_owner.is_(None),
                    MeegleAINodeTask.lease_expires_at <= now,
                )
            )
            result = session.execute(update(MeegleAINodeTask).where(
                MeegleAINodeTask.task_id == task_id, available,
            ).values(lease_owner=owner, lease_expires_at=now + timedelta(seconds=seconds),
                     updated_at=MeegleAINodeTask.updated_at))
            session.commit()
            return cast(CursorResult[Any], result).rowcount == 1

    def _release_lease(self, task_id: str, owner: str) -> None:
        with self._sf() as session:
            session.execute(update(MeegleAINodeTask).where(
                MeegleAINodeTask.task_id == task_id,
                MeegleAINodeTask.lease_owner == owner,
            ).values(lease_owner=None, lease_expires_at=None,
                     updated_at=MeegleAINodeTask.updated_at))
            session.commit()

    def _recoverable_task_ids(self) -> list[str]:
        with self._sf() as session:
            return list(session.scalars(select(MeegleAINodeTask.task_id).where(
                or_(
                    MeegleAINodeTask.status.in_([
                        AeolusState.READING_DOCUMENT.value, AeolusState.EVALUATING.value,
                        AeolusState.COMPLETING.value,
                    ]),
                    MeegleAINodeTask.status.in_([AeolusState.CODING.value, AeolusState.RECODING.value])
                    & or_(
                        MeegleAINodeTask.coco_run_id != "",
                        select(MeegleAINodeRound.task_id).where(
                            MeegleAINodeRound.task_id == MeegleAINodeTask.task_id,
                            MeegleAINodeRound.status.in_(["retrying", "queued"]),
                        ).exists(),
                        # The initial coding round is committed before its first
                        # attempt. Feedback rounds use the explicit queued state
                        # below so a possibly-dispatched run is never replayed.
                        ~select(MeegleAINodeAttempt.id).where(
                            MeegleAINodeAttempt.task_id == MeegleAINodeTask.task_id
                        ).exists(),
                    ),
                ),
                or_(MeegleAINodeTask.lease_owner.is_(None),
                    MeegleAINodeTask.lease_expires_at <= _database_now(session)),
            )))

    async def find_by_node(
        self, project_key: str, work_item_id: int, node_id: str
    ) -> AeolusTask:
        return await run_in_threadpool(
            self._find_by_node, project_key, work_item_id, node_id
        )

    async def transition(
        self,
        task_id: str,
        expected: set[AeolusState],
        new_state: AeolusState,
        *,
        pending_note: str | None = None,
        **changes: Any,
    ) -> AeolusTask:
        return await run_in_threadpool(
            self._transition, task_id, expected, new_state, changes, pending_note
        )

    async def find_active_for_mr(
        self, repo_id: str, mr_iid: int
    ) -> AeolusTask | None:
        return await run_in_threadpool(self._find_active_for_mr, repo_id, mr_iid)

    async def update(self, task_id: str, **changes: Any) -> AeolusTask:
        return await run_in_threadpool(self._update, task_id, changes)

    async def claim_request(self, task_id: str, request_id: str) -> bool:
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        return await run_in_threadpool(self._claim_request, task_id, request_id)

    async def append_round(self, task_id: str, round_: AeolusRound) -> AeolusTask:
        return await run_in_threadpool(self._append_round, task_id, round_)

    async def start_coding_retry(self, task_id: str, request_id: str) -> tuple[AeolusTask, bool]:
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        return await run_in_threadpool(self._start_coding_retry, task_id, request_id)

    def _start_coding_retry(self, task_id: str, request_id: str) -> tuple[AeolusTask, bool]:
        with self._sf() as session:
            record = self._require(session, task_id, lock=True)
            previous_state = record.status
            claimed = list(record.claimed_request_ids or [])
            if request_id in claimed:
                return self._to_domain(session, record), False
            if not can_retry_coding(self._to_domain(session, record)):
                raise InvalidTaskTransition("coding retry is not available")
            round_record = cast(MeegleAINodeRound, self._last_round(session, task_id, lock=True))
            round_record.status = "retrying"
            round_record.error = None
            round_record.finished_at = None
            now = datetime.now(UTC)
            self._set_state(record, AeolusState.RECODING if record.mr_url else AeolusState.CODING, now)
            record.last_error = ""
            # The new attempt must establish its own authorization failure.
            evaluation = dict(record.evaluation or {})
            evaluation.pop("coding_authorization_required", None)
            record.evaluation = evaluation
            record.claimed_request_ids = [*claimed, request_id]
            record.updated_at = now
            session.commit()
            _log_transition(record, previous_state, trigger="coding_retry", round_no=round_record.round_no)
            return self._to_domain(session, record), True

    async def begin_completion(
        self, task_id: str, request_id: str, retrying: bool, verified_id: str = "", allow_release: bool = True,
        deployment_target: str = "devbox",
        *, verification_targets: tuple[DeploymentVerification, ...] = (),
    ):
        return await run_in_threadpool(
            self._begin_completion, task_id, request_id, retrying, verified_id, allow_release, deployment_target,
            verification_targets,
        )

    def _begin_completion(self, task_id: str, request_id: str, retrying: bool, verified_id: str, allow_release: bool,
                          deployment_target: str, verification_targets: tuple[DeploymentVerification, ...]):
        with self._sf() as session:
            record = self._require(session, task_id, lock=True)
            previous_state = record.status
            claimed = list(record.claimed_request_ids or [])
            if request_id in claimed or record.status == AeolusState.COMPLETED.value:
                return self._to_domain(session, record), False
            expected = {AeolusState.REVIEW.value}
            if retrying and (record.platform_status_done or record.completion_requested):
                expected.add(AeolusState.FAILED.value)
            if record.status not in expected:
                raise InvalidTaskTransition("code is not awaiting confirmation")
            # A comment in the aggregation window has already been acknowledged
            # to its author.  Locking the task row and checking the pending
            # comments here orders completion against save_pending_feedback:
            # either the comment is stored first and completion is refused, or
            # completion commits first and the comment is refused before ACK.
            pending = session.scalar(
                select(MeegleAINodePendingFeedback.id)
                .where(MeegleAINodePendingFeedback.task_id == task_id)
                .limit(1)
            )
            if pending is not None:
                raise InvalidTaskTransition(
                    "comments are still being processed; retry completion after they start a round"
                )
            # A question is acknowledged to its author as soon as it is stored,
            # then answered in the background. Completion ends the task for
            # good, so it must not overtake one that still owes a reply.
            awaiting_question = session.scalar(
                select(MeegleAINodeQuestionRecord.id)
                .where(
                    MeegleAINodeQuestionRecord.task_id == task_id,
                    MeegleAINodeQuestionRecord.status.in_(UNSETTLED_QUESTION_STATUSES),
                )
                .limit(1)
            )
            if awaiting_question is not None:
                raise InvalidTaskTransition(
                    "a question is still being answered; retry completion once it is published"
                )
            if self._ppe_enabled and not allow_release and not record.completion_requested:
                from conciliator.repositories.ppe_models import PpeEnvironment
                if session.scalar(select(PpeEnvironment.environment_id).where(
                    PpeEnvironment.task_id == task_id, PpeEnvironment.resource_state != "released",
                ).limit(1)):
                    raise InvalidTaskTransition("explicit release_resources confirmation required for PPE")
            if self._deployments_enabled and not allow_release and not record.completion_requested:
                from conciliator.repositories.models import Deployment
                from conciliator.services.deployment_counts import local_deployment_released
                owned = any(
                    item.data.get("protocol_version") == 2 and not local_deployment_released(item)
                    for item in session.scalars(select(Deployment).where(Deployment.task_id == task_id))
                )
                if owned:
                    raise InvalidTaskTransition(
                        "explicit release_resources confirmation required for the occupied environment"
                    )
            targets = list(verification_targets)
            if verified_id:
                if deployment_target not in {"devbox", "ppe"}:
                    raise InvalidTaskTransition("unknown verification target")
                targets.append(DeploymentVerification("ppe" if deployment_target == "ppe" else "devbox", verified_id))
            if len({item.target for item in targets}) != len(targets):
                raise InvalidTaskTransition("duplicate verification target")
            for target in targets:
                if target.target == "ppe":
                    if not self._ppe_enabled:
                        raise InvalidTaskTransition("PPE deployment is disabled")
                    from conciliator.services.ppe_deployment_service import verify_ppe
                    verify_ppe(session, task_id, target.deployment_id)
                    continue
                if target.target != "devbox" or not self._deployments_enabled:
                    raise InvalidTaskTransition("current deployment is not ready for verification")
                from conciliator.repositories.models import Deployment, DeploymentWorker
                deployment = session.get(Deployment, target.deployment_id, with_for_update=True)
                worker = session.get(DeploymentWorker, deployment.worker_id) if deployment else None
                latest = self._last_round(session, task_id, lock=True)
                from conciliator.services.deployment_counts import v2_entry_ready
                runtime_ready = bool(
                    deployment and deployment.data.get("protocol_version") == 2
                    and v2_entry_ready(deployment.data, _database_now(session))
                )
                if (deployment is None or deployment.task_id != task_id or deployment.status != "succeeded"
                        or worker is None or not runtime_ready
                        or latest is None
                        or (latest.round_no, latest.commit_sha) != (deployment.round_no, deployment.commit_sha)
                        or not worker.ready or (_database_now(session) - worker.last_seen_at).total_seconds() > 30):
                    raise InvalidTaskTransition("current deployment is not ready for verification")
                deployment.data = {**deployment.data, "verified": True}
            if self._deployments_enabled:
                from conciliator.services.deployment_service import invalidate_deployments
                invalidate_deployments(session, task_id)
            if self._ppe_enabled:
                from conciliator.services.ppe_deployment_service import invalidate_ppe
                invalidate_ppe(session, task_id, release=True)
            now = datetime.now(UTC)
            record.claimed_request_ids = [*claimed, request_id]
            record.completion_requested = True
            record.last_error = ""
            self._set_state(record, AeolusState.COMPLETING, now)
            session.commit()
            _log_transition(record, previous_state, trigger="completion_retry" if retrying else "completion")
            return self._to_domain(session, record), True

    async def start_feedback(
        self,
        task_id: str,
        feedback: str,
        request_ids: str | Sequence[str],
        *,
        entries: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[AeolusTask, bool]:
        ids = _normalize_request_ids(request_ids)
        return await run_in_threadpool(
            self._start_feedback, task_id, feedback, ids, entries
        )

    def _start_feedback(
        self,
        task_id: str,
        feedback: str,
        request_ids: Sequence[str],
        entries: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[AeolusTask, bool]:
        with self._sf() as session:
            record = self._require(session, task_id, lock=True)
            previous_state = record.status
            claimed = list(record.claimed_request_ids or [])
            new_ids = [request_id for request_id in request_ids if request_id not in claimed]
            if not new_ids:
                return self._to_domain(session, record), False
            if record.status not in {AeolusState.REVIEW.value, AeolusState.CODING_BLOCKED.value}:
                raise InvalidTaskTransition("coding is not awaiting review")
            now = datetime.now(UTC)
            text_by_request: dict[str, str] = (
                {
                    f"mr-comment:{comment_id}": text
                    for comment_id, text in entries
                }
                if entries is not None
                else {}
            )
            # Claim every feedback row whose window has closed in this same
            # transaction: a comment committed after the flusher's snapshot but
            # before this lock is still merged into this round instead of
            # becoming a second, redundant coding round.
            due = session.execute(
                select(
                    MeegleAINodePendingFeedback.comment_id,
                    MeegleAINodePendingFeedback.feedback,
                )
                .where(
                    MeegleAINodePendingFeedback.task_id == task_id,
                    MeegleAINodePendingFeedback.available_at
                    <= now.replace(tzinfo=None),
                )
                .order_by(
                    MeegleAINodePendingFeedback.available_at,
                    MeegleAINodePendingFeedback.id,
                )
            ).all()
            extra = [
                (comment_id, text)
                for comment_id, text in due
                if f"mr-comment:{comment_id}" not in claimed
                and f"mr-comment:{comment_id}" not in request_ids
            ]
            if extra:
                request_ids = [
                    *request_ids,
                    *(f"mr-comment:{comment_id}" for comment_id, _ in extra),
                ]
                new_ids = [
                    request_id for request_id in request_ids if request_id not in claimed
                ]
                if not new_ids:
                    return self._to_domain(session, record), False
                session.execute(
                    delete(MeegleAINodePendingFeedback).where(
                        MeegleAINodePendingFeedback.task_id == task_id,
                        MeegleAINodePendingFeedback.comment_id.in_(
                            [comment_id for comment_id, _ in extra]
                        ),
                    )
                )
                if entries is None:
                    feedback = "\n\n".join([feedback, *(text for _, text in extra)])
                text_by_request.update(
                    {
                        f"mr-comment:{comment_id}": text
                        for comment_id, text in extra
                    }
                )
            if entries is not None:
                # Only the text of comments claimed by this round may reach the
                # agent: a comment claimed by an earlier round must not repeat
                # its instruction just because its pending row outlived it.
                feedback = "\n\n".join(
                    text_by_request[request_id]
                    for request_id in new_ids
                    if request_id in text_by_request
                )
                if not feedback:
                    raise InvalidTaskTransition("feedback text is missing")
            previous = session.scalar(select(MeegleAINodeRound).where(
                MeegleAINodeRound.task_id == task_id,
            ).order_by(MeegleAINodeRound.round_no.desc()).limit(1))
            if previous is None:
                raise InvalidTaskTransition("coding review has no round")
            if self._ppe_enabled:
                from conciliator.services.ppe_deployment_service import invalidate_ppe
                invalidate_ppe(session, task_id, release=False)
            if self._deployments_enabled:
                from conciliator.services.deployment_service import invalidate_deployments
                invalidate_deployments(session, task_id)
            if previous.status != "blocked":
                previous.status = "revision_requested"
            session.add(MeegleAINodeRound(
                task_id=task_id, round_no=previous.round_no + 1, status="queued",
                feedback=feedback, trigger_type="review_feedback",
                reviewed_round_no=previous.round_no, base_commit_sha=previous.commit_sha or previous.base_commit_sha,
                tests=[], started_at=now,
            ))
            self._set_state(record, AeolusState.RECODING, now)
            self._start_round_timing(record, "coding", previous.round_no + 1, now)
            record.claimed_request_ids = [*claimed, *new_ids]
            record.updated_at = now
            session.commit()
            _log_transition(record, previous_state, trigger="coding_feedback", round_no=previous.round_no + 1)
            return self._to_domain(session, record), True

    async def claim_queued_feedback(self, task_id: str) -> AeolusTask | None:
        return await run_in_threadpool(self._claim_queued_feedback, task_id)

    def _claim_queued_feedback(self, task_id: str) -> AeolusTask | None:
        """Mark persisted feedback as dispatching while the caller holds the task lease."""
        with self._sf() as session:
            record = self._require(session, task_id, lock=True)
            if record.status != AeolusState.RECODING.value:
                return None
            round_record = self._last_round(session, task_id, lock=True)
            if (
                round_record is None
                or round_record.status != "queued"
                or round_record.trigger_type != "review_feedback"
            ):
                return None
            attempt_exists = session.scalar(
                select(MeegleAINodeAttempt.id).where(
                    MeegleAINodeAttempt.task_id == task_id,
                    MeegleAINodeAttempt.round_no == round_record.round_no,
                ).limit(1)
            )
            if attempt_exists is not None:
                raise InvalidTaskTransition("queued feedback already has a coding attempt")
            round_record.status = "coding"
            record.updated_at = datetime.now(UTC)
            session.commit()
            return self._to_domain(session, record)

    async def save_pending_feedback(
        self,
        task_id: str,
        entries: Sequence[tuple[str, str]],
        *,
        available_at: datetime,
    ) -> bool:
        return await run_in_threadpool(
            self._save_pending_feedback, task_id, entries, available_at
        )

    def _save_pending_feedback(
        self,
        task_id: str,
        entries: Sequence[tuple[str, str]],
        available_at: datetime,
    ) -> bool:
        """Store comments before their acknowledgement.

        The pending table is mandatory: a missing table raises instead of
        falling back to process memory, because completion may run on any
        process.
        """
        rows = [
            (comment_id, feedback)
            for comment_id, feedback in entries
            if comment_id and feedback
        ]
        if not rows:
            return True
        comment_ids = [comment_id for comment_id, _ in rows]
        for attempt in range(2):
            with self._sf() as session:
                try:
                    record = session.scalars(
                        select(MeegleAINodeTask)
                        .where(MeegleAINodeTask.task_id == task_id)
                        .with_for_update()
                    ).first()
                    if record is None:
                        raise TaskNotFound(task_id)
                    if record.status not in {
                        AeolusState.REVIEW.value,
                        AeolusState.RECODING.value,
                        AeolusState.CODING_BLOCKED.value,
                    }:
                        raise InvalidTaskTransition(
                            "comments are only accepted while review or coding is running"
                        )
                    existing = set(
                        session.scalars(
                            select(MeegleAINodePendingFeedback.comment_id).where(
                                MeegleAINodePendingFeedback.task_id == task_id,
                                MeegleAINodePendingFeedback.comment_id.in_(comment_ids),
                            )
                        )
                    )
                    # Anchor the aggregation window on the first pending
                    # comment: feedback arriving on another worker inside the
                    # window must share its deadline, so whichever worker
                    # flushes first submits the whole batch as one round.
                    earliest = session.scalar(
                        select(
                            func.min(MeegleAINodePendingFeedback.available_at)
                        ).where(MeegleAINodePendingFeedback.task_id == task_id)
                    )
                    if earliest is not None:
                        deadline = earliest.replace(tzinfo=None)
                        if deadline < available_at.replace(tzinfo=None):
                            available_at = deadline
                    for comment_id, feedback in rows:
                        if comment_id in existing:
                            continue
                        session.add(
                            MeegleAINodePendingFeedback(
                                task_id=task_id,
                                comment_id=comment_id,
                                feedback=feedback,
                                available_at=available_at,
                            )
                        )
                    session.commit()
                    return True
                except IntegrityError:
                    # A second instance stored the same comment first; re-read
                    # once so the batch still converges on one row per comment.
                    session.rollback()
                    if attempt == 1:
                        raise
        raise RuntimeError("pending feedback could not be stored")

    async def pending_feedback(self, task_id: str) -> list[tuple[str, str]]:
        return await run_in_threadpool(self._pending_feedback, task_id)

    def _pending_feedback(self, task_id: str) -> list[tuple[str, str]]:
        with self._sf() as session:
            rows = session.execute(
                select(
                    MeegleAINodePendingFeedback.comment_id,
                    MeegleAINodePendingFeedback.feedback,
                )
                .where(MeegleAINodePendingFeedback.task_id == task_id)
                .order_by(
                    MeegleAINodePendingFeedback.created_at,
                    MeegleAINodePendingFeedback.id,
                )
            ).all()
        return [(comment_id, feedback) for comment_id, feedback in rows]

    async def due_pending_feedback(self, task_id: str) -> list[tuple[str, str]]:
        return await run_in_threadpool(self._due_pending_feedback, task_id)

    def _due_pending_feedback(self, task_id: str) -> list[tuple[str, str]]:
        """Return every comment whose aggregation window has already closed."""
        with self._sf() as session:
            rows = session.execute(
                select(
                    MeegleAINodePendingFeedback.comment_id,
                    MeegleAINodePendingFeedback.feedback,
                )
                .where(
                    MeegleAINodePendingFeedback.task_id == task_id,
                    MeegleAINodePendingFeedback.available_at
                    <= datetime.now(UTC).replace(tzinfo=None),
                )
                .order_by(
                    MeegleAINodePendingFeedback.available_at,
                    MeegleAINodePendingFeedback.id,
                )
            ).all()
        return [(comment_id, feedback) for comment_id, feedback in rows]

    async def claim_comment(
        self, task_id: str, request_id: str, ttl: timedelta
    ) -> datetime | None:
        return await run_in_threadpool(self._claim_comment, task_id, request_id, ttl)

    def _claim_comment(
        self, task_id: str, request_id: str, ttl: timedelta
    ) -> datetime | None:
        """Claim a comment across workers; an expired claim can be retried.

        The unique key is the arbitration point, so this stays correct without
        relying on row locks (SQLite has none). A claim that is never released,
        because the worker died or was shut down mid-answer, expires instead of
        silencing the comment forever.

        The returned ``created_at`` is the fencing token of this lease. A worker
        that stalls past its expiry finds a different token once another worker
        re-claims the comment, and must not renew, complete, release or publish.
        """
        now = datetime.now(UTC)
        with self._sf() as session:
            session.execute(
                delete(MeegleAINodeCommentClaim).where(
                    MeegleAINodeCommentClaim.task_id == task_id,
                    MeegleAINodeCommentClaim.request_id == request_id,
                    MeegleAINodeCommentClaim.expires_at
                    <= now.replace(tzinfo=None),
                )
            )
            session.add(
                MeegleAINodeCommentClaim(
                    task_id=task_id,
                    request_id=request_id,
                    expires_at=now + ttl,
                    created_at=now,
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return None
            return now

    async def release_comment(
        self, task_id: str, request_id: str, token: datetime | None
    ) -> None:
        await run_in_threadpool(self._release_comment, task_id, request_id, token)

    def _release_comment(
        self, task_id: str, request_id: str, token: datetime | None
    ) -> None:
        with self._sf() as session:
            session.execute(
                delete(MeegleAINodeCommentClaim).where(
                    MeegleAINodeCommentClaim.task_id == task_id,
                    MeegleAINodeCommentClaim.request_id == request_id,
                    MeegleAINodeCommentClaim.created_at == token,
                )
            )
            session.commit()

    async def complete_comment(
        self, task_id: str, request_id: str, token: datetime | None
    ) -> bool:
        return await run_in_threadpool(
            self._complete_comment, task_id, request_id, token
        )

    def _complete_comment(
        self, task_id: str, request_id: str, token: datetime | None
    ) -> bool:
        """Keep the claim as the permanent dedup record of a handled comment.

        The table has no status column, so a claim that must never be retried
        is stored with the latest representable expiry instead of being deleted
        or released. A redelivery then loses the insert race and is dropped.

        The question record's terminal status is the primary dedup, so a
        redelivery returns before claiming. This claim is completed once that
        record's reply is confirmed visible, and its far-future expiry keeps a
        later insert from racing a record that is still active. ``False``
        means the lease moved to another worker, which then owns the answer.
        """
        with self._sf() as session:
            result = cast(
                CursorResult[Any],
                session.execute(
                    update(MeegleAINodeCommentClaim)
                    .where(
                        MeegleAINodeCommentClaim.task_id == task_id,
                        MeegleAINodeCommentClaim.request_id == request_id,
                        MeegleAINodeCommentClaim.created_at == token,
                    )
                    .values(expires_at=COMPLETED_CLAIM_EXPIRY)
                ),
            )
            session.commit()
            return bool(result.rowcount)

    async def renew_comment(
        self, task_id: str, request_id: str, ttl: timedelta, token: datetime | None
    ) -> bool:
        return await run_in_threadpool(
            self._renew_comment, task_id, request_id, ttl, token
        )

    def _renew_comment(
        self, task_id: str, request_id: str, ttl: timedelta, token: datetime | None
    ) -> bool:
        """Extend a claim that is still ours, so a slow answer keeps its lease."""
        now = datetime.now(UTC)
        with self._sf() as session:
            result = cast(
                CursorResult[Any],
                session.execute(
                    update(MeegleAINodeCommentClaim)
                    .where(
                        MeegleAINodeCommentClaim.task_id == task_id,
                        MeegleAINodeCommentClaim.request_id == request_id,
                        MeegleAINodeCommentClaim.created_at == token,
                        MeegleAINodeCommentClaim.expires_at > now.replace(tzinfo=None),
                    )
                    .values(expires_at=now + ttl)
                ),
            )
            session.commit()
            return bool(result.rowcount)

    async def record_question(
        self,
        task_id: str,
        comment_id: str,
        *,
        repo_id: str,
        mr_iid: int,
        discussion_id: str | None,
        position: dict | None,
        parent: dict | None,
        question_text: str,
        author_email: str,
        notice_text: str | None = None,
    ) -> dict | None:
        """Persist a question before the webhook acknowledges the delivery.

        The unique key (task, comment) is the arbitration point: redeliveries
        and recovery scans find the same row and do not start a second answer.
        Returns the stored record, or None when the task already ended and the
        caller passed no notice to store in its place.
        """
        return await run_in_threadpool(
            self._record_question,
            task_id,
            comment_id,
            repo_id,
            mr_iid,
            discussion_id,
            position,
            parent,
            question_text,
            author_email,
            notice_text,
        )

    def _record_question(
        self,
        task_id: str,
        comment_id: str,
        repo_id: str,
        mr_iid: int,
        discussion_id: str | None,
        position: dict | None,
        parent: dict | None,
        question_text: str,
        author_email: str,
        notice_text: str | None = None,
    ) -> dict | None:
        """Store the question before its delivery is acknowledged.

        Deciding whether the question can still be answered and writing the row
        happen under the same task row lock, so the insert either lands before
        completion or is replaced by the "task ended" notice in that same
        transaction. Splitting the two would let a crash in between leave an
        acknowledged delivery with nothing on disk to recover.
        """
        now = datetime.now(UTC).replace(tzinfo=None)
        status = "pending"
        answer_text: str | None = None
        with self._sf() as session:
            # Completion takes the same row lock, so either this insert lands
            # first and completion is refused, or completion commits first and
            # this insert stores the notice instead of a question.
            task = self._require(session, task_id, lock=True)
            if task.status not in _QUESTION_STORABLE_STATES:
                if notice_text is None:
                    return None
                status = "failed_pending_publish"
                answer_text = notice_text
            record = session.scalar(
                select(MeegleAINodeQuestionRecord).where(
                    MeegleAINodeQuestionRecord.task_id == task_id,
                    MeegleAINodeQuestionRecord.comment_id == comment_id,
                )
            )
            if record is None:
                record = MeegleAINodeQuestionRecord(
                    task_id=task_id,
                    comment_id=comment_id,
                    repo_id=repo_id,
                    mr_iid=mr_iid,
                    discussion_id=discussion_id,
                    position=position,
                    parent=parent,
                    question_text=question_text,
                    author_email=author_email,
                    status=status,
                    answer_text=answer_text,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
                try:
                    session.commit()
                except IntegrityError:
                    # Lost the insert race to a concurrent delivery; read the
                    # row the winner wrote instead of failing the delivery.
                    session.rollback()
                    record = session.scalar(
                        select(MeegleAINodeQuestionRecord).where(
                            MeegleAINodeQuestionRecord.task_id == task_id,
                            MeegleAINodeQuestionRecord.comment_id == comment_id,
                        )
                    )
                    return None if record is None else self._question_dict(record)
            else:
                # A redelivery refreshes the payload but never resets a terminal
                # status: a published answer stays published, a failed one stays
                # failed until the recovery scan drives it again.
                record.repo_id = repo_id
                record.mr_iid = mr_iid
                record.discussion_id = discussion_id
                record.position = position
                record.parent = parent
                record.question_text = question_text
                record.author_email = author_email
                record.updated_at = now
                session.commit()
            session.refresh(record)
            return self._question_dict(record)

    async def question_record(
        self, task_id: str, comment_id: str
    ) -> dict | None:
        return await run_in_threadpool(self._question_record, task_id, comment_id)

    def _question_record(self, task_id: str, comment_id: str) -> dict | None:
        with self._sf() as session:
            record = session.scalar(
                select(MeegleAINodeQuestionRecord).where(
                    MeegleAINodeQuestionRecord.task_id == task_id,
                    MeegleAINodeQuestionRecord.comment_id == comment_id,
                )
            )
            return None if record is None else self._question_dict(record)

    async def question_records_to_resume(self) -> list[dict]:
        """Questions that still owe an answer (pending or mid-answer)."""
        return await run_in_threadpool(self._question_records_to_resume)

    def _question_records_to_resume(self) -> list[dict]:
        with self._sf() as session:
            rows = session.scalars(
                select(MeegleAINodeQuestionRecord).where(
                    MeegleAINodeQuestionRecord.status.in_(_QUESTION_ACTIVE_STATUSES)
                )
            )
            return [self._question_dict(row) for row in rows]

    async def question_records_to_publish(self) -> list[dict]:
        """Questions whose answer is saved but not yet published."""
        return await run_in_threadpool(self._question_records_to_publish)

    def _question_records_to_publish(self) -> list[dict]:
        with self._sf() as session:
            rows = session.scalars(
                select(MeegleAINodeQuestionRecord).where(
                    or_(
                        MeegleAINodeQuestionRecord.status == "answered",
                        (MeegleAINodeQuestionRecord.status == "failed_pending_publish")
                        & MeegleAINodeQuestionRecord.answer_text.is_not(None),
                        (MeegleAINodeQuestionRecord.status == "unknown")
                        & MeegleAINodeQuestionRecord.answer_text.is_not(None),
                    )
                )
            )
            return [self._question_dict(row) for row in rows]

    def _question_dict(self, record: MeegleAINodeQuestionRecord) -> dict:
        return {
            "task_id": record.task_id,
            "comment_id": record.comment_id,
            "repo_id": record.repo_id,
            "mr_iid": record.mr_iid,
            "discussion_id": record.discussion_id,
            "position": record.position,
            "parent": record.parent,
            "question_text": record.question_text,
            "author_email": record.author_email,
            "status": record.status,
            "answer_text": record.answer_text,
            "session_id": record.session_id,
            "run_id": record.run_id,
            "event_id": record.event_id,
            "attempts": record.attempts,
        }

    async def mark_question_answering(
        self,
        task_id: str,
        comment_id: str,
        *,
        token: datetime | None,
        attempt: int,
        session_id: str,
        run_id: str,
        event_id: str,
    ) -> bool:
        if not run_id:
            return False
        return await run_in_threadpool(
            self._change_question,
            task_id, comment_id, token, attempt, ("sending",),
            {"status": "answering", "run_id": run_id, "event_id": event_id},
            True, session_id,
        )

    async def mark_question_sending(
        self, task_id: str, comment_id: str, *,
        token: datetime | None, attempt: int, session_id: str,
    ) -> bool:
        """Record that a prompt is about to leave the process.

        Sending is not transactional with the database, so the record has to
        say "a send may have happened" before the request goes out.  A crash
        in that window leaves no run cursor, and recovery must treat the
        outcome as unknown instead of assuming the prompt was never sent.
        """
        # An answering run may only be replaced after the caller has confirmed
        # it terminal. Sending without a cursor is never a safe retry state.
        return await run_in_threadpool(
            self._change_question,
            task_id, comment_id, token, attempt, ("pending", "answering"),
            {
                "status": "sending", "session_id": session_id,
                "run_id": None, "event_id": None, "attempts": attempt + 1,
            },
        )

    async def mark_question_answered(
        self, task_id: str, comment_id: str, answer_text: str, *,
        token: datetime | None, attempt: int, require_task_lease: bool = False,
    ) -> bool:
        return await run_in_threadpool(
            self._change_question,
            task_id, comment_id, token, attempt, ("pending", "answering"),
            {"status": "answered", "answer_text": answer_text},
            require_task_lease=require_task_lease,
        )

    async def mark_question_published(
        self, task_id: str, comment_id: str, *,
        token: datetime | None, attempt: int, status: str = "answered",
    ) -> bool:
        if status not in {"answered", "unknown", "failed_pending_publish"}:
            return False
        values: dict[str, Any] = {"status": "published"}
        if status == "unknown":
            # Keep the unknown outcome terminal, while clearing its durable
            # notice only after publication has been observed or succeeded.
            values = {"status": "unknown", "answer_text": None}
        elif status == "failed_pending_publish":
            # The notice that replaced the answer is now visible, so the record
            # can settle as failed without being republished by every scan.
            values = {"status": "failed", "answer_text": None}
        return await run_in_threadpool(
            self._change_question,
            task_id, comment_id, token, attempt, (status,), values,
        )

    async def mark_question_failed(
        self, task_id: str, comment_id: str, *,
        token: datetime | None, attempt: int,
    ) -> bool:
        return await run_in_threadpool(
            self._change_question,
            task_id, comment_id, token, attempt, _QUESTION_ACTIVE_STATUSES,
            {"status": "failed"},
        )

    async def mark_question_failed_notice(
        self, task_id: str, comment_id: str, *,
        token: datetime | None, attempt: int, notice: str,
    ) -> bool:
        """Fence the failure and save its notice for publish recovery.

        The record stays publishable until the notice is confirmed visible, so
        a failed reply cannot silence a question the author is still owed.
        """
        return await run_in_threadpool(
            self._change_question,
            task_id, comment_id, token, attempt, _QUESTION_ACTIVE_STATUSES,
            {"status": "failed_pending_publish", "answer_text": notice},
        )

    async def mark_question_unknown(
        self, task_id: str, comment_id: str, *,
        token: datetime | None, attempt: int, notice: str,
    ) -> bool:
        """Fence an ambiguous send and save its notice for publish recovery."""
        return await run_in_threadpool(
            self._change_question,
            task_id, comment_id, token, attempt, ("sending",),
            {"status": "unknown", "answer_text": notice}, True,
        )

    def _change_question(
        self, task_id: str, comment_id: str, token: datetime | None,
        attempt: int, statuses: Sequence[str], values: dict[str, Any],
        no_cursor: bool = False, session_id: str | None = None,
        require_task_lease: bool = False,
    ) -> bool:
        """Atomically fence every question write by owner, expiry and attempt.

        Lock the claim on MySQL so takeover cannot interleave with this write.
        The UPDATE also checks ownership, including on SQLite where FOR UPDATE
        is ignored. A late callback cannot revive a settled or newer attempt.
        """
        if token is None:
            return False
        owner = _lease_owner.get()
        if require_task_lease and owner is None:
            raise TaskLeaseLost("answer acceptance requires a task lease")
        with self._sf() as session:
            if owner is not None:
                self._require(session, task_id, lock=True)
            claim = select(MeegleAINodeCommentClaim.id).where(
                MeegleAINodeCommentClaim.task_id == task_id,
                MeegleAINodeCommentClaim.request_id == f"mr-comment-answer:{comment_id}",
                MeegleAINodeCommentClaim.created_at == token.replace(tzinfo=None),
            )
            if session.scalar(claim.with_for_update()) is None:
                return False
            now = datetime.now(UTC).replace(tzinfo=None)
            statement = update(MeegleAINodeQuestionRecord).where(
                MeegleAINodeQuestionRecord.task_id == task_id,
                MeegleAINodeQuestionRecord.comment_id == comment_id,
                MeegleAINodeQuestionRecord.status.in_(statuses),
                MeegleAINodeQuestionRecord.attempts == attempt,
                claim.where(MeegleAINodeCommentClaim.expires_at > now).exists(),
            )
            if owner is not None:
                # The earlier _require is a snapshot on SQLite, and even a
                # locked MySQL lease can expire while we wait for the claim.
                # Accept the answer only if owner/expiry still match at UPDATE.
                statement = statement.where(select(MeegleAINodeTask.task_id).where(
                    MeegleAINodeTask.task_id == task_id,
                    MeegleAINodeTask.lease_owner == owner,
                    _task_lease_is_live(session),
                ).exists())
            if no_cursor:
                statement = statement.where(MeegleAINodeQuestionRecord.run_id.is_(None))
            if session_id is not None:
                statement = statement.where(MeegleAINodeQuestionRecord.session_id == session_id)
            result = cast(CursorResult[Any], session.execute(statement.values(**values, updated_at=now)))
            session.commit()
            return bool(result.rowcount)

    async def pending_feedback_task_ids(self, due_before: datetime) -> list[str]:
        return await run_in_threadpool(self._pending_feedback_task_ids, due_before)

    def _pending_feedback_task_ids(self, due_before: datetime) -> list[str]:
        with self._sf() as session:
            rows = session.scalars(
                select(MeegleAINodePendingFeedback.task_id)
                .where(MeegleAINodePendingFeedback.available_at <= due_before)
                .distinct()
                .order_by(MeegleAINodePendingFeedback.task_id)
            )
            return list(rows)

    async def discard_pending_feedback(
        self, task_id: str, comment_ids: Sequence[str]
    ) -> None:
        await run_in_threadpool(
            self._discard_pending_feedback, task_id, list(comment_ids)
        )

    def _discard_pending_feedback(self, task_id: str, comment_ids: list[str]) -> None:
        ids = [comment_id for comment_id in comment_ids if comment_id]
        if not ids:
            return
        with self._sf() as session:
            session.execute(
                delete(MeegleAINodePendingFeedback).where(
                    MeegleAINodePendingFeedback.task_id == task_id,
                    MeegleAINodePendingFeedback.comment_id.in_(ids),
                )
            )
            session.commit()

    async def validation_failed_for_user(self, user_key: str) -> list[AeolusTask]:
        return await run_in_threadpool(self._validation_failed_for_user, user_key)

    def _validation_failed_for_user(self, user_key: str) -> list[AeolusTask]:
        with self._sf() as session:
            records = session.scalars(
                select(MeegleAINodeTask)
                .where(
                    MeegleAINodeTask.user_key == user_key,
                    MeegleAINodeTask.status == AeolusState.VALIDATION_FAILED.value,
                )
                .order_by(MeegleAINodeTask.updated_at.desc())
            )
            return [self._to_domain(session, record) for record in records]

    async def review_evaluation(
        self, task_id: str, request_id: str, *, feedback: str | None = None
    ) -> tuple[AeolusTask, bool]:
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        return await run_in_threadpool(
            self._review_evaluation, task_id, request_id, feedback
        )

    def _review_evaluation(
        self, task_id: str, request_id: str, feedback: str | None
    ) -> tuple[AeolusTask, bool]:
        with self._sf() as session:
            record = self._require(session, task_id, lock=True)
            previous_state = record.status
            claimed = list(record.claimed_request_ids or [])
            if request_id in claimed:
                return self._to_domain(session, record), False
            if record.status not in {
                AeolusState.EVALUATION_REVIEW.value,
                AeolusState.VALIDATION_FAILED.value,
            }:
                raise InvalidTaskTransition("input evaluation is not awaiting review")
            history = copy.deepcopy(record.evaluation_rounds or [])
            now = datetime.now(UTC)
            if feedback is None:
                if record.status != AeolusState.EVALUATION_REVIEW.value or not (
                    record.evaluation and record.evaluation.get("valid") is True
                ):
                    raise InvalidTaskTransition("input evaluation must pass before coding")
                if history:
                    history[-1]["status"] = "confirmed"
                    history[-1]["confirmed_at"] = now.isoformat()
                self._set_state(record, AeolusState.CODING, now)
                self._start_round_timing(record, "coding", 1, now)
                session.add(MeegleAINodeRound(
                    task_id=task_id, round_no=1, status="coding", tests=[],
                    trigger_type="initial", started_at=now,
                ))
            else:
                if history and history[-1]["status"] == "review":
                    history[-1]["status"] = "revision_requested"
                history.append({
                    "round_no": len(history) + 1,
                    "status": "evaluating",
                    "feedback": feedback,
                    "reviewed_round_no": history[-1]["round_no"] if history else None,
                    "result": None,
                    "started_at": now.isoformat(),
                })
                self._set_state(record, AeolusState.READING_DOCUMENT, now)
                self._start_round_timing(record, "evaluation", len(history), now)
                record.last_error = ""
            record.evaluation_rounds = history
            record.claimed_request_ids = [*claimed, request_id]
            record.updated_at = now
            session.commit()
            _log_transition(
                record, previous_state, trigger="evaluation_confirm" if feedback is None else "evaluation_retry",
            )
            return self._to_domain(session, record), True

    async def block_coding(
        self, task_id: str, *, round_changes: dict[str, Any], task_changes: dict[str, Any],
    ) -> AeolusTask:
        return await run_in_threadpool(self._block_coding, task_id, round_changes, task_changes)

    def _block_coding(self, task_id: str, round_changes: dict[str, Any], task_changes: dict[str, Any]) -> AeolusTask:
        """Save the returned result and enter human wait atomically, including recovery cursors."""
        with self._sf() as session:
            task = self._require(session, task_id, lock=True)
            previous_state = task.status
            if task.status not in {AeolusState.CODING.value, AeolusState.RECODING.value}:
                raise InvalidTaskTransition("coding is not running")
            round_ = self._last_round(session, task_id, lock=True)
            if round_ is None:
                raise InvalidTaskTransition("coding has no round")
            for name, value in round_changes.items():
                if name not in _ROUND_UPDATE_FIELDS:
                    raise ValueError(f"unsupported round field: {name}")
                setattr(round_, name, value)
            now = datetime.now(UTC)
            round_.status = "blocked"
            round_.finished_at = now
            attempt = session.scalar(select(MeegleAINodeAttempt).where(
                MeegleAINodeAttempt.task_id == task_id, MeegleAINodeAttempt.round_no == round_.round_no,
            ).order_by(MeegleAINodeAttempt.attempt_no.desc()).limit(1).with_for_update())
            if attempt is None:
                raise InvalidTaskTransition("coding has no attempt")
            attempt.status = "blocked"
            attempt.error = None
            attempt.finished_at = now
            self._apply_task(task, task_changes)
            task.last_error = None
            self._set_state(task, AeolusState.CODING_BLOCKED, now)
            task.updated_at = now
            session.commit()
            _log_transition(task, previous_state, trigger="coding_blocked", round_no=round_.round_no)
            return self._to_domain(session, task)

    async def update_last_round(
        self, task_id: str, **changes: Any
    ) -> AeolusTask:
        return await run_in_threadpool(self._update_last_round, task_id, changes)

    async def append_attempt(
        self,
        task_id: str,
        *,
        session_id: str,
        run_id: str,
        event_id: str,
    ) -> AeolusTask:
        return await run_in_threadpool(
            self._append_attempt, task_id, session_id, run_id, event_id
        )

    async def update_last_attempt(
        self, task_id: str, **changes: Any
    ) -> AeolusTask:
        return await run_in_threadpool(self._update_last_attempt, task_id, changes)

    def _create_from_event(
        self, event: AeolusStartEvent
    ) -> tuple[AeolusTask, bool]:
        existing = self._find_by_task_id(event.task_id)
        if existing is not None:
            return existing, False

        # Meegle issues a new idempotent key when a node is stopped and started
        # again.  One node owns one task: reuse the existing row so a restart
        # resumes the node instead of replaying the whole flow from document
        # evaluation.  The named lock makes the lookup and the insert atomic
        # per node, so two webhooks carrying different idempotent keys cannot
        # both create a task.
        with self._node_creation_lock(_node_lock_name(event)) as session:
            existing = self._find_by_task_id(event.task_id, session)
            if existing is not None:
                return existing, False
            existing = self._find_by_node_identity(event, session)
            if existing is not None:
                return self._adopt_trigger_user(existing, event, session), False
            return self._insert_task(event, session)

    def _adopt_trigger_user(
        self, task: AeolusTask, event: AeolusStartEvent, session: Session | None
    ) -> AeolusTask:
        """A restart can come from another user; later steps must act as them."""
        changes: dict[str, str] = {}
        if task.user_key != event.user_key:
            changes["user_key"] = event.user_key
        if event.work_item_type_key and event.work_item_type_key != task.work_item_type_key:
            changes["work_item_type_key"] = event.work_item_type_key
        if not changes:
            return task
        with self._read_session(session) as current:
            current.execute(
                update(MeegleAINodeTask)
                .where(MeegleAINodeTask.task_id == task.task_id)
                .values(**changes)
            )
            current.commit()
        for key, value in changes.items():
            setattr(task, key, value)
        return task

    @contextmanager
    def _read_session(self, session: Session | None) -> Iterator[Session]:
        """Use the caller's session, or a fresh one when none was supplied."""
        if session is not None:
            yield session
            return
        with self._sf() as owned:
            yield owned

    def _find_by_task_id(
        self, task_id: str, session: Session | None = None
    ) -> AeolusTask | None:
        with self._read_session(session) as current:
            record = current.scalar(
                select(MeegleAINodeTask).where(MeegleAINodeTask.task_id == task_id)
            )
            return None if record is None else self._to_domain(current, record)

    def _find_by_node_identity(
        self, event: AeolusStartEvent, session: Session | None = None
    ) -> AeolusTask | None:
        with self._read_session(session) as current:
            record = current.scalar(
                select(MeegleAINodeTask)
                .where(
                    MeegleAINodeTask.project_key == event.project_key,
                    MeegleAINodeTask.work_item_id == event.work_item_id,
                    MeegleAINodeTask.node_uuid == event.node_uuid,
                )
                .order_by(MeegleAINodeTask.id.desc())
                .limit(1)
            )
            return None if record is None else self._to_domain(current, record)

    @contextmanager
    def _node_creation_lock(self, name: str) -> Iterator[Session | None]:
        """Serialize task creation for one node across processes.

        MySQL named locks are connection scoped, so the lock lives on its own
        connection and the lookup and insert reuse that same connection.  A
        request therefore needs one connection, not one for the lock plus one
        for the session, and cannot starve itself when the pool is small.
        SQLite (tests) serializes writers itself and has no named locks.
        """
        engine = self._lock_engine
        if engine is None or engine.dialect.name != "mysql":
            yield None
            return
        with engine.connect() as conn:
            if self._acquire_named_lock(conn, name) != 1:
                raise RuntimeError(f"timed out acquiring node lock {name}")
            # GET_LOCK opened a transaction on this connection. End it so the
            # session owns the business transaction it commits; otherwise the
            # commit is deferred past RELEASE_LOCK and the next lock holder
            # cannot see the new row yet.
            conn.commit()
            session = Session(bind=conn)
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
                self._release_named_lock(conn, name)
                conn.commit()

    def _acquire_named_lock(self, conn: Connection, name: str) -> int:
        acquired = conn.execute(
            text("SELECT GET_LOCK(:name, :timeout)"),
            {"name": name, "timeout": _NODE_LOCK_TIMEOUT_SECONDS},
        ).scalar()
        return int(acquired or 0)

    def _release_named_lock(self, conn: Connection, name: str) -> None:
        conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": name})

    def _insert_task(
        self, event: AeolusStartEvent, session: Session | None = None
    ) -> tuple[AeolusTask, bool]:
        with self._read_session(session) as session:
            now = datetime.now(UTC)
            record = MeegleAINodeTask(
                task_id=event.task_id,
                project_key=event.project_key,
                work_item_id=event.work_item_id,
                work_item_type_key=event.work_item_type_key or None,
                state_key=event.state_key,
                node_uuid=event.node_uuid,
                user_key=event.user_key,
                props=list(event.props),
                action_token=secrets.token_urlsafe(24),
                status=AeolusState.RECEIVED.value,
                timing={
                    **self._new_timing(now),
                    "current_round": "evaluation:1",
                    "rounds": {"evaluation:1": {
                        **self._new_timing(now), "started_at": now.isoformat(),
                    }},
                },
                requirement_url="",
                requirement_text=None,
                repo_id="",
                work_branch="",
                coco_session_id="",
                coco_run_id="",
                coco_last_event_id="",
                mr_id="",
                mr_url="",
                platform_status_done=False,
                claimed_request_ids=[],
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                # Another creator won the race: converge on the surviving row.
                existing = self._find_by_node_identity(event, session)
                if existing is None:
                    existing = self._find_by_task_id(event.task_id, session)
                if existing is None:
                    raise
                return existing, False
            return self._to_domain(session, record), True

    def _get(self, task_id: str) -> AeolusTask:
        with self._sf() as session:
            return self._to_domain(session, self._require(session, task_id))

    def _find_by_node(
        self, project_key: str, work_item_id: int, node_id: str
    ) -> AeolusTask:
        with self._sf() as session:
            record = session.scalar(
                select(MeegleAINodeTask)
                .where(
                    MeegleAINodeTask.project_key == project_key,
                    MeegleAINodeTask.work_item_id == work_item_id,
                    or_(
                        MeegleAINodeTask.state_key == node_id,
                        MeegleAINodeTask.node_uuid == node_id,
                    ),
                )
                .order_by(MeegleAINodeTask.id.desc())
                .limit(1)
            )
            if record is None:
                raise TaskNotFound((project_key, work_item_id, node_id))
            return self._to_domain(session, record)

    def _find_active_for_mr(self, repo_id: str, mr_iid: int) -> AeolusTask | None:
        # The newest task of the MR, whatever its state: a question must reach
        # the row that records it even when the task already finished, so a
        # failed notice write can be retried instead of acknowledged silently.
        # A coding command is gated by the caller on a reviewable state.
        with self._sf() as session:
            record = session.scalar(
                select(MeegleAINodeTask)
                .where(
                    MeegleAINodeTask.repo_id == repo_id,
                    MeegleAINodeTask.mr_iid == mr_iid,
                )
                .order_by(MeegleAINodeTask.id.desc())
                .limit(1)
            )
            return None if record is None else self._to_domain(session, record)

    def _transition(
        self,
        task_id: str,
        expected: set[AeolusState],
        new_state: AeolusState,
        changes: dict[str, Any],
        pending_note: str | None = None,
    ) -> AeolusTask:
        with self._sf() as session:
            record = self._require(session, task_id, lock=True)
            current = AeolusState(record.status)
            if current not in expected:
                raise InvalidTaskTransition(
                    f"task {task_id} is {current.value}, expected "
                    + ", ".join(sorted(state.value for state in expected))
                )
            now = datetime.now(UTC)
            self._set_state(record, new_state, now)
            self._apply_task(record, changes)
            self._resolve_pending_feedback(session, record, new_state, pending_note)
            record.updated_at = now
            session.commit()
            _log_transition(
                record, current.value, mr_id=record.mr_id, mr_iid=record.mr_iid,
                repo_id=record.repo_id, branch=record.work_branch,
            )
            return self._to_domain(session, record)

    def _resolve_pending_feedback(
        self,
        session: Session,
        record: MeegleAINodeTask,
        new_state: AeolusState,
        note: str | None,
    ) -> int:
        """Delete feedback a terminal transition can no longer run.

        The caller already holds this task's row lock, and save_pending_feedback
        takes the same lock before inserting.  A comment therefore either
        commits before this transaction (and is counted here) or waits and then
        sees the terminal state and refuses the comment before it is
        acknowledged.  Resolving the rows in a later transaction would instead
        delete a comment acknowledged in between without telling its author.
        """
        if new_state not in _PENDING_FEEDBACK_TERMINAL_STATES:
            return 0
        pending = list(
            session.scalars(
                select(MeegleAINodePendingFeedback.id)
                .where(MeegleAINodePendingFeedback.task_id == record.task_id)
                .with_for_update()
            )
        )
        if not pending:
            return 0
        session.execute(
            delete(MeegleAINodePendingFeedback).where(
                MeegleAINodePendingFeedback.id.in_(pending)
            )
        )
        if note:
            message = note.format(count=len(pending))
            record.last_error = f"{record.last_error}{message}"
            self._append_round_note(session, record, new_state, message)
        return len(pending)

    def _append_round_note(
        self,
        session: Session,
        record: MeegleAINodeTask,
        new_state: AeolusState,
        message: str,
    ) -> None:
        """Repeat the note where the failure card reads its reason from."""
        if new_state is AeolusState.VALIDATION_FAILED:
            rounds = [dict(item) for item in (record.evaluation_rounds or [])]
            if rounds and rounds[-1].get("error"):
                rounds[-1]["error"] = f"{rounds[-1]['error']}{message}"
                record.evaluation_rounds = rounds
            return
        round_record = self._last_round(session, record.task_id, lock=True)
        if round_record is not None and round_record.error:
            round_record.error = f"{round_record.error}{message}"

    def _update(self, task_id: str, changes: dict[str, Any]) -> AeolusTask:
        with self._sf() as session:
            record = self._require(session, task_id, lock=True)
            self._apply_task(record, changes)
            record.updated_at = datetime.now(UTC)
            session.commit()
            return self._to_domain(session, record)

    @staticmethod
    def _new_timing(now: datetime) -> dict[str, Any]:
        return {
            "automatic_ms": 0, "human_wait_ms": 0, "retry_wait_ms": 0,
            "current_phase": "automatic", "phase_started_at": now.isoformat(),
            "completed_at": None,
        }

    @staticmethod
    def _advance_timing(timing: dict[str, Any], phase: str, now: datetime) -> None:
        if timing["current_phase"] == "completed":
            return
        elapsed = max(0, round((now - datetime.fromisoformat(timing["phase_started_at"])).total_seconds() * 1000))
        previous = timing["current_phase"]
        if previous == "automatic":
            timing["automatic_ms"] += elapsed
        elif previous in {"human_wait", "retry_wait"}:
            timing["human_wait_ms"] += elapsed
            if previous == "retry_wait":
                timing["retry_wait_ms"] += elapsed
        timing.update(current_phase=phase, phase_started_at=now.isoformat())
        if phase == "completed":
            timing["completed_at"] = now.isoformat()

    @classmethod
    def _start_round_timing(cls, record: MeegleAINodeTask, kind: str, round_no: int, now: datetime) -> None:
        if record.timing is None:
            return
        timing = copy.deepcopy(record.timing)
        rounds = timing.setdefault("rounds", {})
        previous = rounds.get(timing.get("current_round"))
        if previous is not None:
            cls._advance_timing(previous, "completed", now)
        key = f"{kind}:{round_no}"
        rounds[key] = {**cls._new_timing(now), "started_at": now.isoformat()}
        timing["current_round"] = key
        record.timing = timing

    @classmethod
    def _set_state(cls, record: MeegleAINodeTask, new_state: AeolusState, now: datetime) -> None:
        # Called under the task row lock; assigning a fresh dict persists JSON changes.
        if record.status == new_state.value:
            return
        if record.timing is not None:
            timing = copy.deepcopy(record.timing)
            phase = (
                "completed" if new_state == AeolusState.COMPLETED else
                "retry_wait" if new_state in {AeolusState.VALIDATION_FAILED, AeolusState.FAILED} else
                "human_wait" if new_state in {
                    AeolusState.EVALUATION_REVIEW, AeolusState.REVIEW, AeolusState.CODING_BLOCKED,
                } else
                "automatic"
            )
            cls._advance_timing(timing, phase, now)
            current_round = timing.get("rounds", {}).get(timing.get("current_round"))
            if current_round is not None:
                cls._advance_timing(current_round, phase, now)
            record.timing = timing
        record.status = new_state.value

    def _claim_request(self, task_id: str, request_id: str) -> bool:
        with self._sf() as session:
            record = self._require(session, task_id, lock=True)
            claimed = list(record.claimed_request_ids or [])
            if request_id in claimed:
                return False
            claimed.append(request_id)
            record.claimed_request_ids = claimed
            record.updated_at = datetime.now(UTC)
            session.commit()
            return True

    def _append_round(self, task_id: str, round_: AeolusRound) -> AeolusTask:
        with self._sf() as session:
            task = self._require(session, task_id, lock=True)
            session.add(
                MeegleAINodeRound(
                    task_id=task_id,
                    round_no=round_.round_no,
                    status=round_.status,
                    feedback=round_.feedback or None,
                    trigger_type=round_.trigger_type,
                    reviewed_round_no=round_.reviewed_round_no,
                    base_commit_sha=round_.base_commit_sha,
                    summary=round_.summary or None,
                    commit_sha=round_.commit_sha,
                    code_url=round_.code_url,
                    tests=list(round_.tests),
                    unresolved_items=list(round_.unresolved_items),
                    failure_code=round_.failure_code,
                    error=round_.error or None,
                    started_at=round_.started_at,
                    finished_at=round_.finished_at,
                )
            )
            task.updated_at = datetime.now(UTC)
            session.commit()
            log.info(
                "meegle_coding_round_created", workflow="meegle", task_id=task_id,
                round_no=round_.round_no, status=round_.status,
            )
            return self._to_domain(session, task)

    def _append_attempt(
        self,
        task_id: str,
        session_id: str,
        run_id: str,
        event_id: str,
    ) -> AeolusTask:
        with self._sf() as session:
            task = self._require(session, task_id, lock=True)
            round_record = self._last_round(session, task_id, lock=True)
            if round_record is None:
                raise InvalidTaskTransition(f"task {task_id} has no coding round")
            latest = session.scalar(
                select(MeegleAINodeAttempt)
                .where(
                    MeegleAINodeAttempt.task_id == task_id,
                    MeegleAINodeAttempt.round_no == round_record.round_no,
                )
                .order_by(MeegleAINodeAttempt.attempt_no.desc())
                .limit(1)
            )
            session.add(
                MeegleAINodeAttempt(
                    task_id=task_id,
                    round_no=round_record.round_no,
                    attempt_no=(latest.attempt_no + 1) if latest else 1,
                    status="running",
                    session_id=session_id,
                    run_id=run_id,
                    event_id=event_id,
                    started_at=datetime.now(UTC),
                )
            )
            task.coco_session_id = session_id
            task.coco_run_id = run_id
            task.coco_last_event_id = event_id
            task.updated_at = datetime.now(UTC)
            session.commit()
            log.info(
                "meegle_coco_attempt_started", workflow="meegle", task_id=task_id,
                round_no=round_record.round_no, attempt_no=(latest.attempt_no + 1) if latest else 1,
                session_id=session_id, run_id=run_id, event_id=event_id,
            )
            return self._to_domain(session, task)

    def _update_last_attempt(
        self, task_id: str, changes: dict[str, Any]
    ) -> AeolusTask:
        with self._sf() as session:
            task = self._require(session, task_id, lock=True)
            round_record = self._last_round(session, task_id, lock=True)
            if round_record is None:
                raise InvalidTaskTransition(f"task {task_id} has no coding round")
            attempt = session.scalar(
                select(MeegleAINodeAttempt)
                .where(
                    MeegleAINodeAttempt.task_id == task_id,
                    MeegleAINodeAttempt.round_no == round_record.round_no,
                )
                .order_by(MeegleAINodeAttempt.attempt_no.desc())
                .limit(1)
                .with_for_update()
            )
            if attempt is None:
                raise InvalidTaskTransition(
                    f"task {task_id} has no coding attempt"
                )
            previous_status = attempt.status
            for name, value in changes.items():
                if name not in _ATTEMPT_UPDATE_FIELDS:
                    raise ValueError(f"unsupported attempt field: {name}")
                setattr(attempt, name, value)
            task.updated_at = datetime.now(UTC)
            session.commit()
            if previous_status != attempt.status:
                log.info(
                    "meegle_coco_attempt_changed", workflow="meegle", task_id=task_id,
                    round_no=round_record.round_no, attempt_no=attempt.attempt_no,
                    session_id=attempt.session_id, run_id=attempt.run_id,
                    previous_status=previous_status, status=attempt.status,
                )
            return self._to_domain(session, task)

    def _update_last_round(
        self, task_id: str, changes: dict[str, Any]
    ) -> AeolusTask:
        with self._sf() as session:
            task = self._require(session, task_id, lock=True)
            round_record = self._last_round(session, task_id, lock=True)
            if round_record is None:
                raise InvalidTaskTransition(f"task {task_id} has no coding round")
            for name, value in changes.items():
                if name not in _ROUND_UPDATE_FIELDS:
                    raise ValueError(f"unsupported round field: {name}")
                setattr(
                    round_record,
                    name,
                    list(value) if name in {"tests", "unresolved_items"} else value,
                )
            task.updated_at = datetime.now(UTC)
            session.commit()
            return self._to_domain(session, task)

    @staticmethod
    def _last_round(
        session: Session, task_id: str, *, lock: bool = False
    ) -> MeegleAINodeRound | None:
        statement = (
            select(MeegleAINodeRound)
            .where(MeegleAINodeRound.task_id == task_id)
            .order_by(MeegleAINodeRound.round_no.desc())
            .limit(1)
        )
        if lock:
            statement = statement.with_for_update()
        return session.scalar(statement)

    @staticmethod
    def _require(
        session: Session, task_id: str, *, lock: bool = False
    ) -> MeegleAINodeTask:
        statement = select(MeegleAINodeTask).where(
            MeegleAINodeTask.task_id == task_id
        )
        if lock:
            statement = statement.with_for_update()
        record = session.scalar(statement)
        if record is None:
            raise TaskNotFound(task_id)
        owner = _lease_owner.get()
        if owner is not None and (
            record.lease_owner != owner or record.lease_expires_at is None
            or record.lease_expires_at.replace(tzinfo=None) <= _database_now(session)
        ):
            raise TaskLeaseLost("AI task processing lease was lost")
        return record

    @staticmethod
    def _apply_task(record: MeegleAINodeTask, changes: dict[str, Any]) -> None:
        for name, value in changes.items():
            if name not in _TASK_UPDATE_FIELDS:
                raise ValueError(f"unsupported task field: {name}")
            setattr(record, name, copy.deepcopy(value))

    @staticmethod
    def _to_domain(session: Session, record: MeegleAINodeTask) -> AeolusTask:
        rounds = session.scalars(
            select(MeegleAINodeRound)
            .where(MeegleAINodeRound.task_id == record.task_id)
            .order_by(MeegleAINodeRound.round_no)
        ).all()
        attempts = session.scalars(
            select(MeegleAINodeAttempt)
            .where(MeegleAINodeAttempt.task_id == record.task_id)
            .order_by(
                MeegleAINodeAttempt.round_no,
                MeegleAINodeAttempt.attempt_no,
            )
        ).all()
        attempts_by_round: dict[int, list[AeolusAttempt]] = {}
        for attempt in attempts:
            attempts_by_round.setdefault(attempt.round_no, []).append(
                AeolusAttempt(
                    attempt_no=attempt.attempt_no,
                    status=attempt.status,
                    session_id=attempt.session_id or "",
                    run_id=attempt.run_id or "",
                    event_id=attempt.event_id or "",
                    error=attempt.error or "",
                    started_at=_utc(attempt.started_at),
                    finished_at=_optional_utc(attempt.finished_at),
                )
            )
        return AeolusTask(
            task_id=record.task_id,
            project_key=record.project_key,
            work_item_id=record.work_item_id,
            work_item_type_key=record.work_item_type_key or "",
            state_key=record.state_key,
            node_uuid=record.node_uuid,
            user_key=record.user_key,
            props=tuple(copy.deepcopy(record.props or [])),
            action_token=record.action_token,
            state=AeolusState(record.status),
            requirement_url=record.requirement_url or "",
            requirement_text=record.requirement_text or "",
            evaluation=copy.deepcopy(record.evaluation),
            evaluation_rounds=copy.deepcopy(record.evaluation_rounds or []),
            repo_id=record.repo_id or "",
            work_branch=record.work_branch or "",
            coco_session_id=record.coco_session_id or "",
            coco_run_id=record.coco_run_id or "",
            coco_last_event_id=record.coco_last_event_id or "",
            mr_id=record.mr_id or "",
            mr_iid=record.mr_iid,
            mr_url=record.mr_url or "",
            platform_status_done=record.platform_status_done,
            completion_requested=bool(record.completion_requested),
            rounds=[
                AeolusRound(
                    round_no=item.round_no,
                    status=item.status,
                    feedback=item.feedback or "",
                    trigger_type=item.trigger_type or "initial",
                    reviewed_round_no=item.reviewed_round_no,
                    base_commit_sha=item.base_commit_sha or "",
                    summary=item.summary or "",
                    commit_sha=item.commit_sha or "",
                    code_url=item.code_url or "",
                    tests=list(item.tests or []),
                    unresolved_items=list(item.unresolved_items or []),
                    failure_code=item.failure_code or "",
                    error=item.error or "",
                    attempts=attempts_by_round.get(item.round_no, []),
                    started_at=_utc(item.started_at),
                    finished_at=_optional_utc(item.finished_at),
                )
                for item in rounds
            ],
            claimed_request_ids=set(record.claimed_request_ids or []),
            last_error=record.last_error or "",
            created_at=_utc(record.created_at),
            updated_at=_utc(record.updated_at),
            timing=copy.deepcopy(record.timing),
        )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _optional_utc(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None
