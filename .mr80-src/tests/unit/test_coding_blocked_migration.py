from importlib import import_module

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def test_blocked_result_migration_preserves_legacy_results_and_is_reversible():
    migration = import_module("migrations.versions.0017_coding_blocked_results")
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE meegle_ai_node_rounds (id INTEGER PRIMARY KEY, summary TEXT)"))
        connection.execute(text("INSERT INTO meegle_ai_node_rounds VALUES (1, 'legacy result')"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            row = connection.execute(text(
                "SELECT summary, unresolved_items, failure_code FROM meegle_ai_node_rounds"
            )).one()
            assert tuple(row) == ("legacy result", None, None)
            migration.downgrade()
        columns = inspect(connection).get_columns("meegle_ai_node_rounds")
        assert [column["name"] for column in columns] == ["id", "summary"]
        assert connection.execute(text("SELECT summary FROM meegle_ai_node_rounds")).scalar() == "legacy result"
    engine.dispose()
