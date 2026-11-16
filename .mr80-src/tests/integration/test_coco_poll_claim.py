import asyncio
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from conciliator.domain.task import COCO_POLL_CLAIMS, TaskStatus
from conciliator.repositories.models import ResolveTask
from conciliator.repositories.task_repo import TaskRepository


@pytest.mark.usefixtures("clean_db")
@pytest.mark.asyncio
@pytest.mark.parametrize("waiting,processing", list(COCO_POLL_CLAIMS.items()))
async def test_mysql_poll_claim_has_one_winner_and_stale_release_cannot_undo_cancel(
    mysql_container, waiting, processing,
):
    engine = create_engine(mysql_container.dsn)
    sf = sessionmaker(engine, expire_on_commit=False)
    try:
        with sf.begin() as session:
            session.add(ResolveTask(
                id=7, repo_id="repo", original_mr_id=42, source_branch="source", target_branch="main",
                resolve_branch="resolve/task7", trigger_type="COMMENT", trigger_payload={},
                status=waiting.value, active_key="active", version=4, updated_at=datetime(2020, 1, 1),
            ))
        repos = [TaskRepository(sf) for _ in range(8)]
        won = await asyncio.gather(*(repo.update_status(
            task_id=7, from_version=4, from_status=waiting, new_status=processing,
        ) for repo in repos))
        assert sum(won) == 1
        repo = repos[0]
        current = await repo.get(7)
        assert current.status == processing and current.version == 5
        assert await repo.update_status(task_id=7, from_version=5, from_status=processing, new_status=waiting)
        assert await repo.update_status(task_id=7, from_version=6, from_status=waiting, new_status=processing)
        result = await repo.cancel_task(task_id=7, from_version=7, cancelled_by="user", webhook_event_id=None)
        assert result.cancelled
        assert not await repo.update_status(task_id=7, from_version=7, from_status=processing, new_status=waiting)
        cancelled = await repo.get(7)
        assert cancelled.status == TaskStatus.CANCELLED and cancelled.active_key is None
    finally:
        engine.dispose()
