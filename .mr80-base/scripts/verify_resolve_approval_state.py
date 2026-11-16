#!/usr/bin/env python3
"""Read-only TCE/RDS/NextCode verification for one resolve approval task."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter

from conciliator.config import Settings
from conciliator.domain.approval import ApprovalStatus
from conciliator.domain.task import TaskStatus
from conciliator.integrations.nextcode.branches import get_branch
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.commits import get_commit
from conciliator.integrations.nextcode.discussions import (
    find_position_for_version,
    get_thread,
)
from conciliator.integrations.nextcode.mrs import get_mergeability, get_mr
from conciliator.repositories.db import (
    create_engine_from_settings,
    get_session_factory,
)
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import templates
from conciliator.utils.secrets import SecretsProvider, build_secrets_provider

LARK_FILE_URL = re.compile(r"^https://bytedance\.larkoffice\.com/file/(?P<token>[A-Za-z0-9_-]+)$")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="1106267")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task-id", type=int)
    selector.add_argument("--original-mr-number", type=int)
    parser.add_argument(
        "--expect-status",
        choices=(
            TaskStatus.AWAITING_APPROVAL.value,
            TaskStatus.ALL_APPROVED.value,
        ),
        required=True,
    )
    parser.add_argument("--expect-conflict-files", type=int, default=2)
    parser.add_argument("--expect-conflict-blocks", type=int, default=4)
    return parser.parse_args()


def _check(
    checks: dict[str, dict],
    failures: list[str],
    name: str,
    condition: bool,
    **evidence,
) -> None:
    checks[name] = {"ok": condition, **evidence}
    if not condition:
        failures.append(name)


def _description(task) -> str:
    blocks = [
        {
            "idx": block.block_idx + 1,
            "conflict_id": block.conflict_id,
            "file_path": block.file_path,
            "start_line": block.start_line,
            "end_line": block.end_line,
            "source_author_email": block.source_author_email,
            "target_author_email": block.target_author_email,
            "resolved_path": block.resolved_path,
            "resolved_start_line": block.resolved_start_line,
            "resolved_end_line": block.resolved_end_line,
            "approved": bool(block.approval and block.approval.status == ApprovalStatus.APPROVED.value),
        }
        for block in sorted(task.blocks, key=lambda item: item.block_idx)
    ]
    return templates.build_mr_description(
        original_mr_number=task.original_mr_id,
        source_branch=task.source_branch,
        source_commit=task.source_commit_sha or "",
        target_branch=task.target_branch,
        target_commit=task.target_commit_sha or "",
        resolve_branch=task.resolve_branch,
        resolve_commit=task.resolve_commit_sha or "",
        lark_doc_url=task.lark_doc_url or "",
        blocks=blocks,
    )


async def _run() -> int:
    args = _args()
    settings = Settings()
    result: dict = {
        "status": "running",
        "repo_id": args.repo_id,
        "task_id": args.task_id,
        "original_mr_number": args.original_mr_number,
        "expected_status": args.expect_status,
        "checks": {},
        "failures": [],
    }
    checks: dict[str, dict] = result["checks"]
    failures: list[str] = result["failures"]
    engine = None
    secrets: SecretsProvider | None = None
    client: NextCodeClient | None = None
    try:
        engine = create_engine_from_settings(settings)
        repo = TaskRepository(get_session_factory(engine))
        if args.task_id is not None:
            task = await repo.get_with_blocks(args.task_id)
        else:
            latest = await repo.find_latest_for_original_mr(
                args.repo_id,
                args.original_mr_number,
            )
            task = await repo.get_with_blocks(latest.id) if latest is not None else None
        if task is None:
            raise RuntimeError("resolve task was not found in RDS")
        result["task_id"] = task.id
        result["original_mr_number"] = task.original_mr_id
        result["resolver_mr_number"] = task.new_mr_id

        blocks = sorted(task.blocks, key=lambda item: item.block_idx)
        approved = sum(
            bool(block.approval and block.approval.status == ApprovalStatus.APPROVED.value) for block in blocks
        )
        expected_approved = len(blocks) if args.expect_status == TaskStatus.ALL_APPROVED.value else 0
        _check(
            checks,
            failures,
            "rds_task_status",
            task.status == args.expect_status,
            actual=task.status,
        )
        _check(
            checks,
            failures,
            "rds_active_key",
            bool(task.active_key) == (args.expect_status != TaskStatus.ALL_APPROVED.value),
            present=bool(task.active_key),
        )
        _check(
            checks,
            failures,
            "rds_coco_contract",
            task.coco_result_schema_version == "conciliator.resolve.result.v2"
            and bool(task.coco_session_id)
            and bool(task.coco_run_id)
            and bool(task.resolve_commit_sha),
            schema_version=task.coco_result_schema_version,
            session_id=task.coco_session_id,
            run_id=task.coco_run_id,
            resolve_commit=task.resolve_commit_sha,
        )
        lark_match = LARK_FILE_URL.fullmatch(task.lark_doc_url or "")
        _check(
            checks,
            failures,
            "rds_lark_file_url",
            lark_match is not None,
            url=task.lark_doc_url,
            file_token=lark_match.group("token") if lark_match else None,
        )
        _check(
            checks,
            failures,
            "rds_trigger_and_resolver_url",
            bool(task.trigger_author_email)
            and bool(task.new_mr_url)
            and task.trigger_type == "COMMENT",
            trigger_type=task.trigger_type,
            trigger_author_email=task.trigger_author_email,
            new_mr_url=task.new_mr_url,
        )
        conflict_files = sorted({block.file_path for block in blocks})
        expected_ids = [f"C{idx:04d}" for idx in range(1, args.expect_conflict_blocks + 1)]
        _check(
            checks,
            failures,
            "rds_conflict_shape",
            len(conflict_files) == args.expect_conflict_files
            and len(blocks) == args.expect_conflict_blocks
            and [block.conflict_id for block in blocks] == expected_ids,
            file_count=len(conflict_files),
            block_count=len(blocks),
            conflict_ids=[block.conflict_id for block in blocks],
        )
        block_contract_ok = all(
            block.resolved_path
            and block.resolved_start_line is not None
            and block.resolved_end_line is not None
            and block.resolved_start_line <= block.resolved_end_line
            and block.ai_resolution
            and block.attribution_status == "VERIFIED"
            and block.source_author_email
            and block.target_author_email
            and block.discussion_id
            and block.approval is not None
            for block in blocks
        )
        _check(
            checks,
            failures,
            "rds_block_contract",
            block_contract_ok,
            discussion_ids=[block.discussion_id for block in blocks],
        )
        _check(
            checks,
            failures,
            "rds_approval_progress",
            approved == expected_approved,
            approved=approved,
            total=len(blocks),
            statuses=[block.approval.status if block.approval else "MISSING" for block in blocks],
        )
        discussion_ids = [block.discussion_id for block in blocks]
        _check(
            checks,
            failures,
            "rds_discussion_uniqueness",
            all(discussion_ids) and len(set(discussion_ids)) == len(discussion_ids),
            unique_count=len(set(discussion_ids)),
        )

        events = await repo.list_events(task.id)
        event_counts = Counter(event.event_type for event in events)
        common_events = {
            "TASK_CREATED",
            "MR_FROZEN",
            "RESOLVE_BRANCH_CREATED",
            "CONFLICTS_PERSISTED",
            "COCO_SUBMITTED",
            "COCO_VERIFIED",
            "LARK_MARKDOWN_UPLOADED",
            "RESOLVE_MR_CREATED",
            "AWAITING_APPROVAL",
        }
        event_contract_ok = all(event_counts[name] >= 1 for name in common_events)
        event_contract_ok = event_contract_ok and (event_counts["DISCUSSION_CREATED"] == len(blocks))
        if args.expect_status == TaskStatus.ALL_APPROVED.value:
            event_contract_ok = event_contract_ok and (
                event_counts["BLOCK_APPROVED"] == len(blocks)
                and event_counts["ALL_APPROVED"] == 1
                and event_counts["LARK_NOTIFICATION_QUEUED"] == 1
                and event_counts["LARK_NOTIFICATION_SENT"] == 1
            )
        _check(
            checks,
            failures,
            "rds_task_events",
            event_contract_ok,
            counts=dict(sorted(event_counts.items())),
        )
        lark_notification = await repo.get_lark_notification_for_task(
            task_id=task.id,
            notification_type="ALL_APPROVED",
        )
        if args.expect_status == TaskStatus.ALL_APPROVED.value:
            notification_ok = bool(
                lark_notification is not None
                and lark_notification.status == "SENT"
                and lark_notification.recipient_email
                == task.trigger_author_email
                and lark_notification.recipient_open_id
                and lark_notification.mr_url == task.new_mr_url
                and lark_notification.message_id
            )
        else:
            notification_ok = lark_notification is None
        _check(
            checks,
            failures,
            "rds_lark_all_approved_notification",
            notification_ok,
            status=(
                lark_notification.status
                if lark_notification is not None
                else None
            ),
            message_id=(
                lark_notification.message_id
                if lark_notification is not None
                else None
            ),
            attempts=(
                lark_notification.attempts
                if lark_notification is not None
                else None
            ),
        )

        secrets = build_secrets_provider(settings)
        client = NextCodeClient(settings, secrets)
        source_branch, target_branch, resolve_branch = await asyncio.gather(
            get_branch(client, task.repo_id, task.source_branch),
            get_branch(client, task.repo_id, task.target_branch),
            get_branch(client, task.repo_id, task.resolve_branch),
        )
        _check(
            checks,
            failures,
            "nextcode_frozen_branches",
            source_branch is not None
            and source_branch.commit_sha == task.source_commit_sha
            and target_branch is not None
            and target_branch.commit_sha == task.target_commit_sha,
            source_head=(source_branch.commit_sha if source_branch is not None else None),
            target_head=(target_branch.commit_sha if target_branch is not None else None),
        )
        _check(
            checks,
            failures,
            "nextcode_resolve_branch",
            resolve_branch is not None and resolve_branch.commit_sha == task.resolve_commit_sha,
            resolve_head=(resolve_branch.commit_sha if resolve_branch is not None else None),
        )
        if not task.resolve_commit_sha:
            raise RuntimeError("resolve task has no verified resolve commit")
        commit = await get_commit(client, task.repo_id, task.resolve_commit_sha)
        _check(
            checks,
            failures,
            "nextcode_merge_commit",
            commit.id == task.resolve_commit_sha and commit.parents == (task.source_commit_sha, task.target_commit_sha),
            parents=list(commit.parents),
        )
        if task.new_mr_id is None:
            raise RuntimeError("resolve task has no resolver MR number")
        mr = await get_mr(client, task.repo_id, task.new_mr_id)
        expected_description = _description(task)
        _check(
            checks,
            failures,
            "nextcode_resolver_mr",
            mr.source_branch == task.resolve_branch
            and mr.target_branch == task.target_branch
            and mr.description == expected_description
            and mr.web_url == task.new_mr_url,
            source_branch=mr.source_branch,
            target_branch=mr.target_branch,
            description_exact=mr.description == expected_description,
            url=mr.web_url,
            persisted_url=task.new_mr_url,
        )
        mergeability = await get_mergeability(
            client,
            task.repo_id,
            task.new_mr_id,
        )
        _check(
            checks,
            failures,
            "nextcode_resolver_mr_mergeable",
            mergeability.mergeable,
            reason=mergeability.reason,
            detail=mergeability.detail,
        )
        version = mr.latest_version
        if version is None:
            raise RuntimeError("resolver MR has no version")
        positions = []
        positions_ok = True
        for block in blocks:
            if not block.discussion_id:
                raise RuntimeError(f"{block.conflict_id} has no persisted Discussion ID")
            thread = await get_thread(
                client,
                task.repo_id,
                block.discussion_id,
            )
            position = find_position_for_version(thread, version)
            current = {
                "conflict_id": block.conflict_id,
                "thread_id": thread.id,
                "path": position.get("Path") if position else None,
                "side": position.get("Side") if position else None,
                "start_line": position.get("StartLine") if position else None,
                "end_line": position.get("EndLine") if position else None,
            }
            positions.append(current)
            positions_ok = positions_ok and bool(
                thread.id == block.discussion_id
                and position is not None
                and position.get("Side") == "new"
                and position.get("Path") == block.resolved_path
                and int(position.get("StartLine") or 0) == block.resolved_start_line
                and int(position.get("EndLine") or 0) == block.resolved_end_line
            )
        _check(
            checks,
            failures,
            "nextcode_discussion_positions",
            positions_ok,
            positions=positions,
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        failures.append("verifier_exception")
    finally:
        if client is not None:
            await client.aclose()
        if secrets is not None:
            secrets.close()
        if engine is not None:
            engine.dispose()

    result["status"] = "ok" if not failures else "failed"
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
