"""Run with the existing disposable MySQL testcontainer, never the live RDS."""

from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from conciliator.domain.task import TaskStatus
from conciliator.repositories._sync_repo import SyncTaskRepository
from conciliator.repositories.models import TaskTraceEvent


def test_concurrent_mysql_cas_has_one_history_row(clean_db, mysql_container):
    engine = create_engine(mysql_container.dsn)
    sf = sessionmaker(engine, expire_on_commit=False)
    repo = SyncTaskRepository(sf)
    task = repo.create(
        repo_id="trace-test",
        original_mr_id=1,
        source_branch="source",
        target_branch="target",
        resolve_branch="resolve",
        trigger_type="COMMENT",
        trigger_payload={},
    )

    def update():
        return repo.update_status(task_id=task.id, from_version=0, new_status=TaskStatus.PREPARING)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: update(), range(2)))
    assert sorted(results) == [False, True]
    with sf() as s:
        events = s.scalars(
            select(TaskTraceEvent).where(TaskTraceEvent.task_id == task.id, TaskTraceEvent.to_status == "PREPARING")
        ).all()
        assert len(events) == 1
        assert events[0].from_status == "PENDING"
    engine.dispose()
