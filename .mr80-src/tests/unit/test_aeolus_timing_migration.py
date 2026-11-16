from importlib import import_module

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def test_timing_migration_preserves_legacy_rows_and_is_reversible():
    migration = import_module("migrations.versions.0012_meegle_task_timing")
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE meegle_ai_node_tasks (task_id TEXT PRIMARY KEY)"))
        connection.execute(text("INSERT INTO meegle_ai_node_tasks VALUES ('legacy')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            assert connection.execute(text("SELECT timing FROM meegle_ai_node_tasks")).scalar() is None
            assert inspect(connection).get_columns("meegle_ai_node_tasks")[-1]["nullable"]
            migration.downgrade()
        assert connection.execute(text("SELECT task_id FROM meegle_ai_node_tasks")).scalar() == "legacy"
        assert [column["name"] for column in inspect(connection).get_columns("meegle_ai_node_tasks")] == ["task_id"]
