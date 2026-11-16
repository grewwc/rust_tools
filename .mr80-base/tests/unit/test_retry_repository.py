import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conciliator.domain.task import TaskStatus
from conciliator.repositories.models import (
    Approval,
    Base,
    ConflictBlock,
    ResolveTask,
    TaskEvent,
)
from conciliator.repositories.task_repo import TaskRepository


def _repository() -> tuple[TaskRepository, sessionmaker]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
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
                resolve_branch="resolve/task-7",
                original_mr_id=42,
                active_key="repo-1:42",
                trigger_type="COMMENT",
                trigger_payload={},
                status=TaskStatus.AWAITING_APPROVAL.value,
                resolve_commit_sha="a" * 40,
                lark_doc_url="https://lark.test/file/old",
                version=3,
            )
        )
        for idx, status in enumerate(("PENDING", "APPROVED", "PENDING")):
            block_id = idx + 1
            session.add(
                ConflictBlock(
                    id=block_id,
                    task_id=7,
                    conflict_id=f"C{block_id:04d}",
                    block_idx=idx,
                    file_path=f"src/file{block_id}.py",
                    start_line=1,
                    end_line=2,
                )
            )
            session.add(
                Approval(
                    id=block_id,
                    block_id=block_id,
                    status=status,
                )
            )
        session.commit()
    return TaskRepository(sf), sf


@pytest.mark.asyncio
async def test_reject_request_and_claim_retry_are_atomic():
    repo, sf = _repository()

    for block_id, reason in ((1, "保留 source 兼容逻辑"), (3, "keep target guard")):
        result = await repo.reject_block_and_mark_partial(
            task_id=7,
            block_id=block_id,
            reviewer_email="author@example.com",
            reason=reason,
            discussion_id=f"thread-{block_id}",
            webhook_event_id=f"event-{block_id}",
        )
        assert result == "rejected"

    request = await repo.request_retry(
        task_id=7,
        requested_by="author@example.com",
        webhook_event_id="retry-event",
    )
    assert request is not None
    assert request.retry_count == 1
    assert request.rejected_conflict_ids == ("C0001", "C0003")
    assert request.locked_conflict_ids == ("C0002",)

    claims = [await repo.claim_schedulable() for _ in range(8)]
    claimed = [claim for claim in claims if claim is not None]
    assert len(claimed) == 1
    assert claimed[0].task_id == 7
    assert claimed[0].status is TaskStatus.RETRY_PREPARING

    with sf() as session:
        task = session.get(ResolveTask, 7)
        assert task is not None
        assert task.status == TaskStatus.RETRY_PREPARING.value
        assert task.retry_count == 1
        events = list(session.execute(select(TaskEvent).where(TaskEvent.task_id == 7)).scalars())
        assert [event.event_type for event in events].count("BLOCK_REJECTED") == 2
        requested = next(event for event in events if event.event_type == "RETRY_REQUESTED")
        assert requested.payload["attempt"] == 2
        assert requested.payload["rejected_conflict_ids"] == ["C0001", "C0003"]
        assert requested.payload["rejected"] == [
            {"conflict_id": "C0001", "reason": "保留 source 兼容逻辑"},
            {"conflict_id": "C0003", "reason": "keep target guard"},
        ]
        assert [event.event_type for event in events].count("RETRY_CLAIMED") == 1


@pytest.mark.asyncio
async def test_retry_reset_changes_only_rejected_snapshot():
    repo, sf = _repository()
    await repo.reject_block_and_mark_partial(
        task_id=7,
        block_id=1,
        reviewer_email="author@example.com",
        reason="重新组合两侧条件",
        discussion_id="thread-1",
        webhook_event_id="event-1",
    )

    reset = await repo.reset_rejected_approvals(
        task_id=7,
        conflict_ids=("C0001",),
    )

    assert reset == 1
    with sf() as session:
        approvals = list(session.execute(select(Approval).order_by(Approval.block_id)).scalars())
        assert [approval.status for approval in approvals] == [
            "PENDING",
            "APPROVED",
            "PENDING",
        ]
        assert approvals[0].comment is None
        assert approvals[0].reviewer_email is None
