import importlib.util
from datetime import datetime
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, Table, create_engine, inspect

from conciliator.repositories.models import Base


def test_historical_ppe_migration_and_downgrade_preserve_existing_tasks():
    path = Path(__file__).resolve().parents[2] / "migrations/versions/0019_ppe_deployments.py"
    spec = importlib.util.spec_from_file_location("ppe_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE existing_tasks (id INTEGER PRIMARY KEY, content TEXT)")
        connection.exec_driver_sql("INSERT INTO existing_tasks VALUES (1, 'keep')")
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()
        inspector = inspect(connection)
        for name in ("aeolus_ppe_environments", "aeolus_ppe_deployments"):
            assert {c["name"] for c in inspector.get_columns(name)} == (
                set(Base.metadata.tables[name].columns.keys()) - {"id"}
            )
            assert {tuple(c["column_names"]) for c in inspector.get_unique_constraints(name)} == {
                tuple(col.name for col in c.columns)
                for c in Base.metadata.tables[name].constraints
                if c.__class__.__name__ == "UniqueConstraint"
                # The unique business key and numeric primary key arrive in 0023.
                and tuple(col.name for col in c.columns) not in {("environment_id",), ("deployment_id",)}
            }
        environments = Table("aeolus_ppe_environments", MetaData(), autoload_with=connection)
        connection.execute(
            environments.insert(),
            {
                "environment_id": "a" * 32,
                "task_id": "task",
                "generation": 1,
                "name": "ppe_ai_ae_" + "a" * 16,
                "resource_state": "planned",
                "profile_revision": "b" * 64,
                "created_at": datetime(2026, 11, 4),
                "data": {},
            },
        )
        with Operations.context(MigrationContext.configure(connection)):
            module.downgrade()
        assert connection.exec_driver_sql("SELECT content FROM existing_tasks").scalar() == "keep"
        assert not inspect(connection).has_table("aeolus_ppe_environments")
    engine.dispose()
