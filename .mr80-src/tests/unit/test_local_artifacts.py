import json
from pathlib import Path
from types import SimpleNamespace

from conciliator.integrations.coco.client import CocoResolveResult
from conciliator.services.local_artifacts import write_local_report_artifacts


def test_write_local_report_artifacts_are_private_and_repeatable(tmp_path: Path):
    task = SimpleNamespace(
        id=123,
        repo_id="repo-1",
        original_mr_id=10,
        source_branch="feat/source",
        source_commit_sha="a" * 40,
        target_branch="main",
        target_commit_sha="b" * 40,
        resolve_branch="fix/feat-source-to-main-conflict-20260828-t123",
        coco_session_id="session-1",
        coco_run_id="run-1",
    )
    result = CocoResolveResult(
        schema_version="conciliator.resolve.result.v3",
        task_id=123,
        status="succeeded",
        attempt=1,
        resolve_branch=task.resolve_branch,
        commit_sha="c" * 40,
        merge_parents=("a" * 40, "b" * 40),
        conflicts=(),

        unresolved_items=(),
    )

    artifacts = write_local_report_artifacts(
        output_root=tmp_path,
        task=task,
        blocks=[],
        resolve_result=result,
        resolved_files={"src/example.py": "resolved\n"},
    )

    assert artifacts.preview_path.exists()
    assert artifacts.preview_path.stat().st_mode & 0o777 == 0o600
    assert artifacts.directory.stat().st_mode & 0o777 == 0o700
    summary = json.loads(artifacts.summary_path.read_text())
    assert summary["status"] == "PASS"
    assert "source_branch_unchanged" not in summary["checks"]
    assert "tests" not in summary["checks"]
    assert summary["artifacts"]["lark-document-preview.md"]["sha256"] == artifacts.preview_sha256
