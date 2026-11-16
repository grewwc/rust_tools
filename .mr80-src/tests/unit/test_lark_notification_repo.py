import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from conciliator.domain.task import TaskStatus
from conciliator.repositories._sync_repo import ActiveTaskExistsError
from conciliator.repositories.models import (
    Approval,
    Base,
    ConflictBlock,
    LarkNotificationOutbox,
    ResolveTask,
    TaskEvent,
)
from conciliator.repositories.task_repo import TaskRepository
from tests.unit.test_processing_version import repository as repository


def received_task_args(**overrides):
    return {
        "repo_id": "repo-1", "source_branch": "source", "target_branch": "target", "resolve_branch": "",
        "original_mr_id": 54, "original_mr_url": "https://code.test/mr/54", "trigger_type": "COMMENT",
        "trigger_payload": {}, "trigger_author_email": " trigger@example.com ",
        "mr_author_email": "other@example.com", **overrides,
    }


@pytest.mark.asyncio
async def test_received_receipt_is_created_with_task_and_duplicate_trigger_cannot_enqueue_again(repository):
    repo, sf = repository
    task = await repo.create(**received_task_args())
    with pytest.raises(ActiveTaskExistsError):
        await repo.create(**received_task_args())
    with sf() as session:
        receipt = session.scalar(select(LarkNotificationOutbox))
        assert receipt.task_id == task.id and receipt.notification_type == "TASK_RECEIVED"
        assert receipt.recipient_email == "trigger@example.com"
        assert receipt.mr_url == "https://code.test/mr/54"
        assert receipt.status == "PENDING" and receipt.attempts == 0
        assert len(list(session.scalars(select(LarkNotificationOutbox)))) == 1
        assert len(list(session.scalars(select(ResolveTask)))) == 1
        queued = list(session.scalars(select(TaskEvent).where(TaskEvent.event_type == "LARK_NOTIFICATION_QUEUED")))
        assert len(queued) == 1

    # The same durable delivery identity survives a lost sender response and worker recovery.
    stale_before = datetime.now(UTC) - timedelta(minutes=5)
    first = await repo.claim_lark_notification(stale_before=stale_before)
    with sf() as session:
        session.get(LarkNotificationOutbox, first.id).updated_at = stale_before - timedelta(seconds=1)
        session.commit()
    recovered = await repo.claim_lark_notification(stale_before=stale_before)
    assert recovered.request_uuid == first.request_uuid and recovered.attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"trigger_type": "AUTO_CONFLICT"}, {"trigger_author_email": None}, {"trigger_author_email": "  "},
])
async def test_received_without_comment_sender_does_not_fallback_to_mr_author(repository, overrides):
    repo, sf = repository
    await repo.create(**received_task_args(**overrides))
    with sf() as session:
        assert session.scalar(select(LarkNotificationOutbox)) is None


@pytest.mark.asyncio
async def test_received_outbox_failure_rolls_back_task_and_creation_events(repository):
    from sqlalchemy import event

    repo, sf = repository

    def fail_insert(*args):
        raise RuntimeError("outbox insert failed")

    event.listen(LarkNotificationOutbox, "before_insert", fail_insert)
    try:
        with pytest.raises(RuntimeError, match="outbox insert failed"):
            await repo.create(**received_task_args())
    finally:
        event.remove(LarkNotificationOutbox, "before_insert", fail_insert)
    with sf() as session:
        assert session.scalar(select(ResolveTask)) is None
        assert session.scalar(select(LarkNotificationOutbox)) is None
        assert session.scalar(select(TaskEvent)) is None
    task = await repo.create(**received_task_args())
    assert await repo.get_lark_notification_for_task(task_id=task.id, notification_type="TASK_RECEIVED") is not None


def _add_approved_blocks(session, task_id: int, count: int) -> None:
    for index in range(count):
        block_id = task_id * 100 + index
        session.add(
            ConflictBlock(
                id=block_id,
                task_id=task_id,
                conflict_id=f"C{index + 1:04d}",
                block_idx=index,
                file_path=f"src/file{index}.py",
                start_line=1,
                end_line=1,
            )
        )
        session.add(
            Approval(
                id=block_id,
                block_id=block_id,
                status="APPROVED",
            )
        )


@pytest.mark.asyncio
async def test_all_approved_outbox_is_atomic_and_claimed_once(tmp_path):
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'outbox.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as session:
        session.add(
            ResolveTask(
                id=7,
                repo_id="repo-1",
                source_branch="source",
                target_branch="target",
                resolve_branch="resolve/source-to-target",
                original_mr_id=42,
                active_key="repo-1:42",
                trigger_type="COMMENT",
                trigger_payload={},
                trigger_author_email="trigger@example.com",
                new_mr_id=99,
                new_mr_url="https://code.test/mr/99",
                status=TaskStatus.AWAITING_APPROVAL.value,
                version=3,
            )
        )
        _add_approved_blocks(session, 7, 4)
        session.commit()
    repo = TaskRepository(sf)

    assert await repo.complete_all_approved_and_enqueue(
        task_id=7,
        from_version=3,
        approved_count=4,
    ) == (True, True)
    assert await repo.complete_all_approved_and_enqueue(
        task_id=7,
        from_version=3,
        approved_count=4,
    ) == (False, False)

    claimed = await asyncio.gather(
        repo.claim_lark_notification(stale_before=datetime.now(UTC) - timedelta(minutes=5)),
        repo.claim_lark_notification(stale_before=datetime.now(UTC) - timedelta(minutes=5)),
    )
    claimed_items = [item for item in claimed if item is not None]
    assert len(claimed_items) == 1
    item = claimed_items[0]
    assert item.attempts == 1
    assert item.status == "SENDING"

    assert await repo.mark_lark_notification_sent(
        notification_id=item.id,
        from_version=item.version,
        recipient_open_id="ou_trigger",
        message_id="om_message",
    )
    with sf() as session:
        task = session.get(ResolveTask, 7)
        assert task is not None
        assert task.status == TaskStatus.ALL_APPROVED.value
        assert task.active_key is None
        outbox = session.execute(select(LarkNotificationOutbox)).scalar_one()
        assert outbox.status == "SENT"
        assert outbox.message_id == "om_message"
        events = list(session.execute(select(TaskEvent).where(TaskEvent.task_id == 7)).scalars())
        assert [event.event_type for event in events] == [
            "ALL_APPROVED",
            "LARK_NOTIFICATION_QUEUED",
        ]
        assert events[0].payload["lark_notification_queued"] is True
    engine.dispose()


@pytest.mark.asyncio
async def test_auto_conflict_completion_does_not_fallback_to_mr_author(tmp_path):
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'auto.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as session:
        session.add(
            ResolveTask(
                id=8,
                repo_id="repo-1",
                source_branch="source",
                target_branch="target",
                resolve_branch="resolve/source-to-target",
                original_mr_id=43,
                active_key="repo-1:43",
                trigger_type="AUTO_CONFLICT",
                trigger_payload={},
                mr_author_email="author@example.com",
                new_mr_id=100,
                new_mr_url="https://code.test/mr/100",
                status=TaskStatus.AWAITING_APPROVAL.value,
                version=1,
            )
        )
        _add_approved_blocks(session, 8, 1)
        session.commit()
    repo = TaskRepository(sf)

    assert await repo.complete_all_approved_and_enqueue(
        task_id=8,
        from_version=1,
        approved_count=1,
    ) == (True, False)
    assert (
        await repo.get_lark_notification_for_task(
            task_id=8,
            notification_type="ALL_APPROVED",
        )
        is None
    )
    engine.dispose()


@pytest.fixture
def awaiting_repo(tmp_path):
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'awaiting.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as session:
        session.add(ResolveTask(
            id=7, repo_id="repo-1", source_branch="source", target_branch="target",
            resolve_branch="resolve/source-to-target", original_mr_id=42,
            trigger_type="COMMENT", trigger_payload={},
            trigger_author_email="trigger@example.com", mr_author_email="other@example.com",
            new_mr_url="https://code.test/mr/99", status=TaskStatus.MR_CREATED.value,
            version=0, retry_count=0,
        ))
        session.commit()
    yield TaskRepository(sf), sf
    engine.dispose()


@pytest.mark.asyncio
async def test_awaiting_notification_once_per_round_and_all_approved_coexists(awaiting_repo):
    repo, sf = awaiting_repo
    assert await repo.update_status(task_id=7, from_version=0, new_status=TaskStatus.AWAITING_APPROVAL)
    # A stale worker and a same-state update must not enqueue another notification.
    assert not await repo.update_status(task_id=7, from_version=0, new_status=TaskStatus.AWAITING_APPROVAL)
    assert await repo.update_status(task_id=7, from_version=1, new_status=TaskStatus.AWAITING_APPROVAL)
    assert await repo.update_status(task_id=7, from_version=2, new_status=TaskStatus.RETRY_PUBLISHING)
    assert await repo.update_status(task_id=7, from_version=3, new_status=TaskStatus.AWAITING_APPROVAL)
    with sf() as session:
        assert len(list(session.scalars(select(LarkNotificationOutbox)))) == 1
    assert await repo.update_status(
        task_id=7, from_version=4, new_status=TaskStatus.RETRY_PUBLISHING, retry_count=1,
    )
    assert await repo.update_status(task_id=7, from_version=5, new_status=TaskStatus.AWAITING_APPROVAL)
    with sf() as session:
        _add_approved_blocks(session, 7, 1)
        session.commit()
    assert await repo.complete_all_approved_and_enqueue(
        task_id=7, from_version=6, approved_count=1,
    ) == (True, True)
    with sf() as session:
        items = list(session.scalars(select(LarkNotificationOutbox).order_by(LarkNotificationOutbox.id)))
        assert [item.notification_type for item in items] == [
            "AWAITING_APPROVAL:0", "AWAITING_APPROVAL:1", "ALL_APPROVED",
        ]
        assert {item.recipient_email for item in items} == {"trigger@example.com"}
        assert {item.mr_url for item in items} == {"https://code.test/mr/99"}
        assert len({item.request_uuid for item in items}) == 3
        assert all(item.status == "PENDING" and len(item.request_uuid) <= 50 for item in items)
        events = list(session.scalars(select(TaskEvent).where(TaskEvent.event_type == "LARK_NOTIFICATION_QUEUED")))
        assert len(events) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"trigger_type": "AUTO_CONFLICT"},
    {"trigger_author_email": None},
    {"trigger_author_email": "  "},
    {"new_mr_url": None},
])
async def test_awaiting_without_comment_recipient_or_mr_skips_notification(awaiting_repo, overrides):
    repo, sf = awaiting_repo
    with sf() as session:
        task = session.get(ResolveTask, 7)
        for name, value in overrides.items():
            setattr(task, name, value)
        session.commit()
    assert await repo.update_status(task_id=7, from_version=0, new_status=TaskStatus.AWAITING_APPROVAL)
    with sf() as session:
        assert session.get(ResolveTask, 7).status == TaskStatus.AWAITING_APPROVAL.value
        assert session.scalar(select(LarkNotificationOutbox.id)) is None


@pytest.mark.asyncio
async def test_awaiting_enqueue_failure_rolls_back_status_and_event(awaiting_repo):
    from sqlalchemy import event

    repo, sf = awaiting_repo

    def fail_insert(*args):
        raise RuntimeError("outbox insert failed")

    event.listen(LarkNotificationOutbox, "before_insert", fail_insert)
    try:
        with pytest.raises(RuntimeError, match="outbox insert failed"):
            await repo.update_status(task_id=7, from_version=0, new_status=TaskStatus.AWAITING_APPROVAL)
    finally:
        event.remove(LarkNotificationOutbox, "before_insert", fail_insert)
    with sf() as session:
        task = session.get(ResolveTask, 7)
        assert task.status == TaskStatus.MR_CREATED.value
        assert task.version == 0
        assert session.scalar(select(LarkNotificationOutbox.id)) is None
        assert session.scalar(select(TaskEvent.id)) is None
    # A subsequent retry can commit both the transition and notification.
    assert await repo.update_status(task_id=7, from_version=0, new_status=TaskStatus.AWAITING_APPROVAL)
