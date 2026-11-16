"""Exercise the destructive schema change against populated historical tables."""

import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, Table, create_engine, inspect, select

from conciliator.repositories.models import Base


def migration(name):
    path = Path(__file__).resolve().parents[2] / "migrations" / "versions" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_upgrade_preserves_history_and_downgrade_restores_empty_slots():
    engine = create_engine("sqlite://")
    old = migration("0013_aeolus_deployments")
    current = migration("0018_v2_only_deployments")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE meegle_ai_node_tasks (id INTEGER PRIMARY KEY)")
        with Operations.context(MigrationContext.configure(connection)):
            old.upgrade()
        metadata = MetaData()
        workers = Table("aeolus_deployment_workers", metadata, autoload_with=connection)
        deployments = Table("aeolus_deployments", metadata, autoload_with=connection)
        from datetime import datetime
        stamp = datetime(2026, 11, 4)
        connection.execute(workers.insert(), [
            {"worker_id": name, "vmid": name, "session_id": "session", "last_seen_at": stamp,
             "ready": True, "generation": 8, "reserved_task_id": "old-task", "active_deployment_id": "old",
             "releasing": True, "data": {"protocol_version": version, "audit": "retained"}}
            for name, version in (("old", 1), ("machine", 2))
        ])
        connection.execute(deployments.insert(), {
            "deployment_id": "old", "task_id": "old-task", "worker_id": "old", "request_id": "request",
            "round_no": 1, "commit_sha": "a" * 40, "generation": 8, "status": "succeeded",
            "stage": "ready", "created_at": stamp, "data": {"maat_env_id": "keep-me"},
        })
        with Operations.context(MigrationContext.configure(connection)):
            current.upgrade()
        inspector = inspect(connection)
        assert {c["name"] for c in inspector.get_columns("aeolus_deployment_workers")} == set(
            Base.metadata.tables["aeolus_deployment_workers"].columns.keys()
        ) - {"id"}  # Numeric primary key arrives in 0023.
        assert {i["name"] for i in inspector.get_indexes("aeolus_deployments")} == {
            "idx_aeolus_deploy_worker_status", "idx_aeolus_deploy_task",
        }
        after = MetaData()
        kept_workers = Table("aeolus_deployment_workers", after, autoload_with=connection)
        kept_deployments = Table("aeolus_deployments", after, autoload_with=connection)
        assert len(connection.execute(select(kept_workers)).all()) == 2
        assert all(row.data["audit"] == "retained" for row in connection.execute(select(kept_workers)))
        row = connection.execute(select(kept_deployments)).one()
        assert row.generation == 8 and row.data == {"maat_env_id": "keep-me"}
        with Operations.context(MigrationContext.configure(connection)):
            current.downgrade()
        restored = Table("aeolus_deployment_workers", MetaData(), autoload_with=connection)
        for row in connection.execute(select(restored)):
            assert row.generation == 0 and not row.releasing
            assert row.reserved_task_id is None and row.active_deployment_id is None
            assert row.data["audit"] == "retained"
    engine.dispose()
