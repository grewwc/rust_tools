from importlib import import_module

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def test_session_context_migration_keeps_legacy_rows_and_is_reversible():
    migration = import_module("migrations.versions.0024_coco_session_context")
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE meegle_ai_node_tasks (task_id TEXT PRIMARY KEY)"))
        connection.execute(text("INSERT INTO meegle_ai_node_tasks VALUES ('legacy')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            assert connection.execute(
                text("SELECT work_item_type_key FROM meegle_ai_node_tasks"),
            ).one() == (None,)
            assert [column["name"] for column in inspect(connection).get_columns("meegle_ai_node_tasks")] == [
                "task_id", "work_item_type_key",
            ]
            migration.downgrade()
        assert connection.execute(text("SELECT task_id FROM meegle_ai_node_tasks")).scalar() == "legacy"
        assert [column["name"] for column in inspect(connection).get_columns("meegle_ai_node_tasks")] == ["task_id"]
    engine.dispose()
