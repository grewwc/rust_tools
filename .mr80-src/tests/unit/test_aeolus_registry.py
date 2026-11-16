import asyncio
import threading
import types
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from conciliator.domain.aeolus_task import AeolusRound, AeolusStartEvent, AeolusState
from conciliator.repositories.models import (
    Base,
    MeegleAINodeQuestionRecord,
    MeegleAINodeTask,
)
from conciliator.services.aeolus_comment_service import ANSWER_CLAIM_TTL
from conciliator.services.aeolus_registry import (
    AeolusTaskRegistry,
    InvalidTaskTransition,
)


def _event(
    task_id: str = "idem-1",
    node_uuid: str = "node-uuid-1",
    user_key: str = "user-1",
) -> AeolusStartEvent:
    return AeolusStartEvent(
        task_id=task_id,
        project_key="space-1",
        work_item_id=42,
        state_key="state-1",
        node_uuid=node_uuid,
        user_key=user_key,
        props=(),
    )


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


@pytest.mark.asyncio
async def test_create_is_idempotent_and_find_accepts_state_key_or_uuid(
    registry_factory,
):
    registry = AeolusTaskRegistry(registry_factory)
    first, created = await registry.create_from_event(_event())
    duplicate, duplicate_created = await registry.create_from_event(_event())

    assert created is True
    assert duplicate_created is False
    assert duplicate.task_id == first.task_id
    assert duplicate.action_token == first.action_token
    assert (
        await registry.find_by_node("space-1", 42, "state-1")
    ).task_id == "idem-1"
    assert (
        await registry.find_by_node("space-1", 42, "node-uuid-1")
    ).task_id == "idem-1"


@pytest.mark.asyncio
async def test_restart_event_reuses_the_node_task(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    first, created = await registry.create_from_event(_event())
    restarted, restarted_created = await registry.create_from_event(
        _event(task_id="idem-2")
    )

    assert created is True
    assert restarted_created is False
    assert restarted.task_id == first.task_id
    assert restarted.action_token == first.action_token


@pytest.mark.asyncio
async def test_restart_retains_work_item_type_and_adopts_trigger_user(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    event = _event()
    first, _ = await registry.create_from_event(event)
    restarted, created = await registry.create_from_event(
        replace(event, task_id="idem-2", work_item_type_key="story", user_key="user-2"),
    )
    assert not created
    assert restarted.task_id == first.task_id
    saved = await AeolusTaskRegistry(registry_factory).get(first.task_id)
    assert saved.work_item_type_key == "story"
    assert saved.user_key == "user-2"
    await registry.create_from_event(_event(task_id="idem-3", user_key="user-3"))
    saved = await registry.get(first.task_id)
    assert saved.work_item_type_key == "story"
    assert saved.user_key == "user-3"


@pytest.mark.asyncio
async def test_transition_is_atomic_and_rejects_stale_action(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    await registry.create_from_event(_event())
    changed = await registry.transition(
        "idem-1", {AeolusState.RECEIVED}, AeolusState.READING_DOCUMENT
    )
    assert changed.state == AeolusState.READING_DOCUMENT

    with pytest.raises(InvalidTaskTransition, match="reading_document"):
        await registry.transition(
            "idem-1", {AeolusState.REVIEW}, AeolusState.RECODING
        )


@pytest.mark.asyncio
async def test_request_ids_are_claimed_once(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    await registry.create_from_event(_event())
    assert await registry.claim_request("idem-1", "request-1") is True


@pytest.mark.asyncio
async def test_completion_waits_for_a_question_that_still_owes_a_reply(
    registry_factory,
):
    registry = AeolusTaskRegistry(registry_factory)
    await registry.create_from_event(_event())
    await registry.transition(
        "idem-1", {AeolusState.RECEIVED}, AeolusState.REVIEW
    )
    await registry.record_question(
        "idem-1",
        "comment-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="thread-1",
        position=None,
        parent=None,
        question_text="/ask 为什么幂等？",
        author_email="dev@example.com",
    )

    # The question was acknowledged to its author and is answered in the
    # background; completing now would end the task that still owes a reply.
    with pytest.raises(InvalidTaskTransition, match="question is still being answered"):
        await registry.begin_completion("idem-1", "request-1", False)

    token = await registry.claim_comment(
        "idem-1", "mr-comment-answer:comment-1", ANSWER_CLAIM_TTL
    )
    assert token is not None
    assert await registry.mark_question_failed(
        "idem-1", "comment-1", token=token, attempt=0
    )

    _task, started = await registry.begin_completion("idem-1", "request-1", False)
    assert started is True
    assert await registry.claim_request("idem-1", "request-1") is False


@pytest.mark.asyncio
async def test_task_and_round_survive_registry_recreation(registry_factory):
    first = AeolusTaskRegistry(registry_factory)
    await first.create_from_event(_event())
    await first.update("idem-1", requirement_url="https://example/doc")
    await first.append_round("idem-1", AeolusRound(1, "coding"))
    await first.append_attempt(
        "idem-1",
        session_id="session-1",
        run_id="run-1",
        event_id="event-1",
    )
    await first.update_last_attempt("idem-1", status="succeeded")
    await first.update_last_round(
        "idem-1", status="review", commit_sha="a" * 40, tests=["pytest -q"]
    )

    restored = await AeolusTaskRegistry(registry_factory).get("idem-1")

    assert restored.requirement_url == "https://example/doc"
    assert restored.rounds[0].status == "review"
    assert restored.rounds[0].commit_sha == "a" * 40
    assert restored.rounds[0].tests == ["pytest -q"]
    assert restored.rounds[0].attempts[0].session_id == "session-1"
    assert restored.rounds[0].attempts[0].status == "succeeded"


@pytest.mark.asyncio
async def test_blocked_result_and_wait_state_roll_back_together_on_invalid_data(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    await registry.create_from_event(_event())
    await registry.append_round("idem-1", AeolusRound(1, "coding"))
    await registry.transition("idem-1", {AeolusState.RECEIVED}, AeolusState.CODING)
    await registry.append_attempt("idem-1", session_id="session", run_id="run", event_id="event")
    with pytest.raises(ValueError):
        await registry.block_coding("idem-1", round_changes={"summary": "must not persist"},
                                    task_changes={"unsupported_field": "invalid"})
    task = await registry.get("idem-1")
    assert task.state == AeolusState.CODING
    assert task.rounds[-1].summary == ""
    assert task.rounds[-1].status == "coding"
    assert task.rounds[-1].attempts[-1].status == "running"


def test_node_lock_name_follows_the_node_not_the_idempotent_key():
    from conciliator.services.aeolus_registry import _node_lock_name

    assert _node_lock_name(_event(task_id="idem-1")) == _node_lock_name(
        _event(task_id="idem-2")
    )
    assert _node_lock_name(_event()) != _node_lock_name(_event(node_uuid="node-uuid-2"))
    assert _node_lock_name(_event()).startswith("aeolus-node:")
    # MySQL rejects lock names longer than 64 characters.
    assert len(_node_lock_name(_event())) <= 64


class _MySQLDialectEngine:
    """Report a MySQL dialect while delegating to a real engine."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self.dialect = types.SimpleNamespace(name="mysql")

    def connect(self):
        return self._engine.connect()


@pytest.fixture
def file_engine(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'registry.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _spy_named_lock(monkeypatch, log: list[tuple[str, str]], acquired: int = 1) -> None:
    def acquire(self, conn, name):
        log.append(("GET_LOCK", name))
        return acquired

    def release(self, conn, name):
        log.append(("RELEASE_LOCK", name))

    monkeypatch.setattr(AeolusTaskRegistry, "_acquire_named_lock", acquire)
    monkeypatch.setattr(AeolusTaskRegistry, "_release_named_lock", release)


def _process_local_named_lock(monkeypatch) -> list[str]:
    """Emulate GET_LOCK with process-local locks for the concurrency tests."""
    locks: dict[str, threading.Lock] = {}
    held: list[str] = []
    guard = threading.Lock()

    def lock_for(name: str) -> threading.Lock:
        with guard:
            return locks.setdefault(name, threading.Lock())

    def acquire(self, conn, name):
        got = lock_for(name).acquire(timeout=10)
        if got:
            held.append(name)
        return 1 if got else 0

    def release(self, conn, name):
        if name in held:
            held.remove(name)
        lock_for(name).release()

    monkeypatch.setattr(AeolusTaskRegistry, "_acquire_named_lock", acquire)
    monkeypatch.setattr(AeolusTaskRegistry, "_release_named_lock", release)
    return held


def test_node_creation_lock_holds_and_releases_a_mysql_named_lock(
    file_engine, monkeypatch
):
    log: list[tuple[str, str]] = []
    _spy_named_lock(monkeypatch, log)
    factory = sessionmaker(bind=file_engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(
        factory, lock_engine=_MySQLDialectEngine(file_engine)
    )

    with registry._node_creation_lock("aeolus-node:test"):
        log.append(("body", ""))

    assert [entry[0] for entry in log] == ["GET_LOCK", "body", "RELEASE_LOCK"]


def test_node_creation_lock_fails_fast_when_the_lock_times_out(
    file_engine, monkeypatch
):
    log: list[tuple[str, str]] = []
    _spy_named_lock(monkeypatch, log, acquired=0)
    factory = sessionmaker(bind=file_engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(
        factory, lock_engine=_MySQLDialectEngine(file_engine)
    )

    with (
        pytest.raises(RuntimeError, match="node lock"),
        registry._node_creation_lock("aeolus-node:test"),
    ):
        raise AssertionError("the body must not run without the lock")

    assert [entry[0] for entry in log] == ["GET_LOCK"]


@pytest.mark.asyncio
async def test_create_converges_on_the_node_row_when_the_insert_loses_a_race(
    registry_factory, monkeypatch
):
    registry = AeolusTaskRegistry(registry_factory)
    first, created = await registry.create_from_event(_event(task_id="idem-1"))
    assert created is True

    # Both lookups miss, as they do when a concurrent creator commits between
    # the lookup and the insert; the duplicate-key handler must then converge
    # on the surviving row instead of surfacing the IntegrityError.
    task_id_calls = {"count": 0}
    original_task_id = AeolusTaskRegistry._find_by_task_id

    def stale_task_id(self, task_id, session=None):
        task_id_calls["count"] += 1
        if task_id_calls["count"] <= 2:
            return None
        return original_task_id(self, task_id, session)

    node_calls = {"count": 0}
    original_node = AeolusTaskRegistry._find_by_node_identity

    def stale_node(self, event, session=None):
        node_calls["count"] += 1
        if node_calls["count"] == 1:
            return None
        return original_node(self, event, session)

    monkeypatch.setattr(AeolusTaskRegistry, "_find_by_task_id", stale_task_id)
    monkeypatch.setattr(AeolusTaskRegistry, "_find_by_node_identity", stale_node)

    second, second_created = await registry.create_from_event(_event(task_id="idem-1"))

    assert second_created is False
    assert second.task_id == first.task_id
    assert node_calls["count"] == 2


@pytest.mark.asyncio
async def test_node_creation_holds_the_named_lock_across_the_insert(
    file_engine, monkeypatch
):
    from conciliator.services.aeolus_registry import _node_lock_name

    held = _process_local_named_lock(monkeypatch)
    factory = sessionmaker(bind=file_engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(
        factory, lock_engine=_MySQLDialectEngine(file_engine)
    )
    seen: dict[str, bool] = {}
    original_insert = AeolusTaskRegistry._insert_task

    def spy(self, event, session=None):
        seen["locked"] = _node_lock_name(event) in held
        return original_insert(self, event, session)

    monkeypatch.setattr(AeolusTaskRegistry, "_insert_task", spy)

    _, created = await registry.create_from_event(_event(task_id="idem-lock"))

    assert created is True
    assert seen["locked"] is True
    assert held == []


@pytest.mark.asyncio
async def test_node_creation_needs_a_single_pooled_connection(
    tmp_path, monkeypatch
):
    # The lock connection also carries the lookup and the insert.  With a
    # one-slot pool, taking a second connection inside the lock times out.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'pool.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=1,
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    log: list[tuple[str, str]] = []
    _spy_named_lock(monkeypatch, log)
    registry = AeolusTaskRegistry(
        factory, lock_engine=_MySQLDialectEngine(engine)
    )

    try:
        task, created = await registry.create_from_event(_event(task_id="idem-pool"))

        assert created is True
        assert [entry[0] for entry in log] == ["GET_LOCK", "RELEASE_LOCK"]
        found = await registry.find_by_node("space-1", 42, "node-uuid-1")
        assert found.task_id == task.task_id
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_creates_for_one_node_keep_a_single_task(
    tmp_path, monkeypatch
):
    # StaticPool shares one SQLite connection, which cannot model two
    # transactions racing, so this test uses a file database with a real pool.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'concurrent.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    _process_local_named_lock(monkeypatch)
    registry = AeolusTaskRegistry(
        factory, lock_engine=_MySQLDialectEngine(engine)
    )

    try:
        results = await asyncio.gather(
            registry.create_from_event(_event(task_id="idem-a")),
            registry.create_from_event(_event(task_id="idem-b")),
        )

        assert sorted(created for _, created in results) == [False, True]
        assert len({task.task_id for task, _ in results}) == 1
        task = await registry.find_by_node("space-1", 42, "node-uuid-1")
        assert task.task_id == results[0][0].task_id
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_insert_is_visible_before_the_node_lock_is_released(
    tmp_path, monkeypatch
):
    # The next webhook only sees the new row if the insert commits while the
    # node lock is still held; committing after the release reopens the race.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'visible.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(factory, lock_engine=_MySQLDialectEngine(engine))
    visible_at_release: list[int] = []

    def acquire(self, conn, name):
        # GET_LOCK opens the connection's transaction in production; keep that
        # effect so the test models the session joining an external one.
        conn.execute(text("SELECT 1"))
        return 1

    def release(self, conn, name):
        with engine.connect() as other:
            visible_at_release.append(
                other.scalar(select(func.count()).select_from(MeegleAINodeTask))
            )

    monkeypatch.setattr(AeolusTaskRegistry, "_acquire_named_lock", acquire)
    monkeypatch.setattr(AeolusTaskRegistry, "_release_named_lock", release)

    try:
        _, created = await registry.create_from_event(_event(task_id="idem-visible"))

        assert created is True
        assert visible_at_release == [1]
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_restart_by_another_user_adopts_the_new_identity(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    first, created = await registry.create_from_event(_event(task_id="idem-1"))
    assert created is True
    assert first.user_key == "user-1"

    restarted, restarted_created = await registry.create_from_event(
        _event(task_id="idem-2", user_key="user-2")
    )

    assert restarted_created is False
    assert restarted.task_id == first.task_id
    assert restarted.user_key == "user-2"
    assert (await registry.get(first.task_id)).user_key == "user-2"


@pytest.mark.asyncio
async def test_a_published_unknown_question_does_not_block_completion(registry_factory):
    """An unknown outcome is terminal, so it must not fence the task forever.

    Its notice is republished by the publish scan, which never needed the task
    session, so completing the task cannot lose it.
    """
    registry = AeolusTaskRegistry(registry_factory)
    await registry.create_from_event(_event())
    await registry.transition("idem-1", {AeolusState.RECEIVED}, AeolusState.REVIEW)
    await registry.record_question(
        "idem-1",
        "comment-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="discussion-1",
        position=None,
        parent=None,
        question_text="/ask 为什么？",
        author_email="a@example.com",
    )
    with registry_factory() as session:
        row = session.scalar(select(MeegleAINodeQuestionRecord))
        row.status = "unknown"
        row.answer_text = None
        session.commit()

    task, _ = await registry.begin_completion("idem-1", "request-1", False)

    assert task.task_id == "idem-1"


@pytest.mark.asyncio
async def test_question_is_refused_once_the_task_is_completing(registry_factory):
    registry = AeolusTaskRegistry(registry_factory)
    await registry.create_from_event(_event())
    await registry.transition("idem-1", {AeolusState.RECEIVED}, AeolusState.REVIEW)
    await registry.transition("idem-1", {AeolusState.REVIEW}, AeolusState.COMPLETING)

    stored = await registry.record_question(
        "idem-1",
        "comment-1",
        repo_id="repo-1",
        mr_iid=7,
        discussion_id="discussion-1",
        position=None,
        parent=None,
        question_text="/ask 为什么？",
        author_email="a@example.com",
    )

    assert stored is None
    assert await registry.question_record("idem-1", "comment-1") is None
