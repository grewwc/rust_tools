import re
from pathlib import Path

import pytest

from scripts.export_ddl import _inline_indexes, export_revision_sql, get_revisions


@pytest.mark.parametrize("stem", [
    "0008_meegle_ai_node_tasks",
    "0009_meegle_ai_node_attempts",
    "0010_meegle_evaluation_review",
    "0011_meegle_task_lease",
    "0012_meegle_task_timing",
    "0013_aeolus_deployments",
    "0018_v2_only_deployments",
    "0019_ppe_deployments",
    "0020_pending_comment_feedback",
    "0021_comment_answer_claim",
    "0024_coco_session_context",
])
def test_meegle_incremental_ddl_matches_offline_migration(stem, monkeypatch):
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("LARK_APP_ID", "cli_test_app")
    # --sql selects the MySQL dialect without connecting to a database.
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://localhost/ddl_render")
    revision, description, _, previous = next(item for item in get_revisions() if item[2] == stem)
    rendered = export_revision_sql(revision, description, stem, down_revision=previous)
    ddl_path = Path(__file__).resolve().parents[2] / "ddl" / f"{stem}.sql"
    recorded = ddl_path.read_text()

    def normalize(sql):
        sql = re.sub(r"^-- generated at:.*\n", "", sql, flags=re.M)
        # SQLAlchemy's equivalent MySQL table options can vary in order between processes.
        return re.sub(r"\)(COMMENT='[^']*')\s+(CHARSET=\w+);", r")\2 \1;", sql)

    assert normalize(recorded) == normalize(rendered)


def test_opted_in_incremental_ddl_is_rerunnable(monkeypatch):
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://localhost/ddl_render")
    revision, description, stem, previous = next(
        item for item in get_revisions() if item[2] == "0020_pending_comment_feedback"
    )

    rendered = export_revision_sql(revision, description, stem, down_revision=previous)

    assert "CREATE TABLE IF NOT EXISTS meegle_ai_node_pending_feedback" in rendered
    assert "CREATE INDEX" not in rendered
    assert "KEY idx_meegle_ai_pending_task" in rendered
    assert "本文件幂等" in rendered


def test_inline_indexes_preserves_mysql_table_comment():
    sql = """\
CREATE TABLE IF NOT EXISTS resolve_tasks (
    id BIGINT NOT NULL,
    new_mr_id BIGINT,
    PRIMARY KEY (id)
)COMMENT='解冲突任务主表';

CREATE INDEX idx_tasks_new_mr ON resolve_tasks (new_mr_id);
"""

    rendered = _inline_indexes(sql)

    assert "CREATE INDEX" not in rendered
    assert "KEY idx_tasks_new_mr (new_mr_id)" in rendered
    assert ")COMMENT='解冲突任务主表';" in rendered


def test_inline_indexes_preserves_mysql_options_and_unmatched_indexes():
    sql = """
CREATE TABLE task_trace_events (
  id BIGINT NOT NULL,
  occurred_at DATETIME(6),
  PRIMARY KEY (id)
)COMMENT='trace' CHARSET=utf8mb4;
CREATE INDEX idx_trace_time ON task_trace_events (occurred_at, id);
CREATE INDEX idx_existing ON existing_table (id);
"""
    rendered = _inline_indexes(sql)
    assert "KEY idx_trace_time (occurred_at, id)" in rendered
    assert ")COMMENT='trace' CHARSET=utf8mb4;" in rendered
    assert "CREATE INDEX idx_existing ON existing_table (id);" in rendered
