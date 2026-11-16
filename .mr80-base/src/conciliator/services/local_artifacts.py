import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conciliator.integrations.coco.client import CocoResolveResult, result_to_dict
from conciliator.services.templates import build_lark_doc_markdown


@dataclass(frozen=True)
class LocalReportArtifacts:
    directory: Path
    preview_path: Path
    conflicts_path: Path
    coco_result_path: Path
    summary_path: Path
    preview_sha256: str


def write_local_report_artifacts(
    *,
    output_root: Path,
    task: Any,
    blocks: list[dict],
    resolve_result: CocoResolveResult,
    resolved_files: dict[str, str],
) -> LocalReportArtifacts:
    directory = output_root / f"task-{task.id}"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)

    preview = build_lark_doc_markdown(
        original_mr_number=task.original_mr_id,
        source_branch=task.source_branch,
        source_commit=task.source_commit_sha or "",
        target_branch=task.target_branch,
        target_commit=task.target_commit_sha or "",
        resolve_branch=task.resolve_branch,
        resolve_commit=resolve_result.commit_sha,
        blocks=blocks,
        resolved_files=resolved_files,
    )
    preview_path = directory / "lark-document-preview.md"
    conflicts_path = directory / "conflicts.json"
    coco_result_path = directory / "coco-result.json"
    summary_path = directory / "validation-summary.json"

    preview_sha256 = _write_private(preview_path, preview)
    conflicts_sha256 = _write_private(conflicts_path, _json(blocks))
    coco_result_sha256 = _write_private(coco_result_path, _json(result_to_dict(resolve_result)))
    summary = {
        "schema_version": "conciliator.resolve.validation.v1",
        "status": "PASS",
        "task_id": task.id,
        "repo_id": task.repo_id,
        "original_mr_number": task.original_mr_id,
        "source_branch": task.source_branch,
        "source_commit": task.source_commit_sha,
        "target_branch": task.target_branch,
        "target_commit": task.target_commit_sha,
        "resolve_branch": task.resolve_branch,
        "resolve_commit": resolve_result.commit_sha,
        "coco_session_id": task.coco_session_id,
        "coco_run_id": task.coco_run_id,
        "checks": {
            "resolve_branch_head": "PASS",
            "merge_parents": "PASS",
            "conflict_ids_and_paths": "PASS",
            "conflict_markers_removed": "PASS",
            "source_target_blame": "PASS",
        },
        "artifacts": {
            preview_path.name: {"sha256": preview_sha256},
            conflicts_path.name: {"sha256": conflicts_sha256},
            coco_result_path.name: {"sha256": coco_result_sha256},
        },
    }
    _write_private(summary_path, _json(summary))
    return LocalReportArtifacts(
        directory=directory,
        preview_path=preview_path,
        conflicts_path=conflicts_path,
        coco_result_path=coco_result_path,
        summary_path=summary_path,
        preview_sha256=preview_sha256,
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_private(path: Path, content: str) -> str:
    encoded = content.encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
    path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()
