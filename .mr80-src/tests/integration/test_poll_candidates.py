from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import sessionmaker

from conciliator.domain.task import TaskStatus
from conciliator.repositories._sync_repo import SyncTaskRepository
from conciliator.repositories.models import ResolveTask


@pytest.mark.usefixtures("clean_db")
def test_poll_keyset_projection_and_json_filters(mysql_container):
    engine = create_engine(mysql_container.dsn)
    sf = sessionmaker(engine, expire_on_commit=False)
    old = datetime(2020, 1, 1)
    with sf() as session:
        for i in range(1, 30):
            status = "COCO_RUNNING" if i <= 25 else ("FAILED" if i <= 27 else "MR_CREATED")
            session.add(ResolveTask(
                repo_id="repo", original_mr_id=i, source_branch="source", target_branch="main",
                resolve_branch=f"resolve/{i}", trigger_type="COMMENT", status=status,
                trigger_payload=({"_failure_notice": {"state": "pending"}} if i == 26 else
                                 {"_blank_approval_publication_verified": True} if i == 28 else {}),
                conflict_files_snapshot=[{"content": "large snapshot"}], coco_result_json={"large": "result"},
                updated_at=old,
            ))
        session.commit()
    statements = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    repo = SyncTaskRepository(sf)
    cursor, ids = None, []
    for _ in range(3):
        batch = repo.list_poll_candidates(status=TaskStatus.COCO_RUNNING, threshold=datetime.now(UTC), cursor=cursor)
        ids.extend(row[0] for row in batch)
        cursor = (batch[-1][1], batch[-1][0])
    assert ids == list(range(1, 26))
    assert all("conflict_files_snapshot" not in sql and "coco_result_json" not in sql for sql in statements)
    for status, expected in [(TaskStatus.FAILED, [26]), (TaskStatus.MR_CREATED, [28])]:
        assert [row[0] for row in repo.list_poll_candidates(
            status=status, threshold=datetime.now(UTC) - timedelta(seconds=60),
        )] == expected
    task = repo.get(1, include_bodies=False)
    assert {"conflict_files_snapshot", "coco_result_json", "blocks"} <= inspect(task).unloaded
    assert task.updated_at == old and task.version == 0
    engine.dispose()
