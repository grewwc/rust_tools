#!/usr/bin/env python3
"""Read-only RDS and NextCode verification for one resolve retry task."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter

from conciliator.config import Settings
from conciliator.integrations.nextcode.branches import get_branch
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.commits import get_commit
from conciliator.integrations.nextcode.discussions import get_thread
from conciliator.integrations.nextcode.files import get_file
from conciliator.integrations.nextcode.mrs import get_mergeability, get_mr
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import templates
from conciliator.utils.secrets import build_secrets_provider

LARK_FILE_URL = re.compile(r"^https://bytedance\.larkoffice\.com/file/(?P<token>[A-Za-z0-9_-]+)$")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="1106267")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task-id", type=int)
    selector.add_argument("--original-mr-number", type=int)
    parser.add_argument("--expect-status", required=True)
    parser.add_argument("--expect-retry-count", type=int, default=0)
    parser.add_argument("--expect-approved", default="")
    parser.add_argument("--expect-rejected", default="")
    parser.add_argument("--expect-pending", default="")
    parser.add_argument("--expect-conflict-blocks", type=int, default=12)
    parser.add_argument("--expect-conflict-files", type=int, default=4)
    parser.add_argument("--expect-retry-replies", type=int, default=0)
    return parser.parse_args()


def _ids(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


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


def _mr_blocks(task) -> list[dict]:
    return [
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
            "approved": bool(block.approval and block.approval.status == "APPROVED"),
            "approval_status": (block.approval.status if block.approval else "PENDING"),
            "reject_reason": (block.approval.comment if block.approval else None),
        }
        for block in sorted(task.blocks, key=lambda item: item.block_idx)
    ]


async def _run() -> int:
    args = _args()
    expected = {
        "APPROVED": _ids(args.expect_approved),
        "REJECTED": _ids(args.expect_rejected),
        "PENDING": _ids(args.expect_pending),
    }
    result: dict = {"status": "running", "checks": {}, "failures": []}
    checks: dict[str, dict] = result["checks"]
    failures: list[str] = result["failures"]
    settings = Settings()
    engine = create_engine_from_settings(settings)
    client = None
    try:
        repo = TaskRepository(get_session_factory(engine))
        if args.task_id is not None:
            task = await repo.get_with_blocks(args.task_id)
        else:
            latest = await repo.find_latest_for_original_mr(
                args.repo_id,
                args.original_mr_number,
            )
            task = await repo.get_with_blocks(latest.id) if latest else None
        if task is None:
            raise RuntimeError("resolve task was not found in RDS")
        result.update(
            {
                "task_id": task.id,
                "original_mr_number": task.original_mr_id,
                "resolver_mr_number": task.new_mr_id,
            }
        )
        blocks = sorted(task.blocks, key=lambda item: item.block_idx)
        actual: dict[str, set[str]] = {
            "APPROVED": set(),
            "REJECTED": set(),
            "PENDING": set(),
        }
        for block in blocks:
            status = block.approval.status if block.approval else "PENDING"
            actual.setdefault(status, set()).add(block.conflict_id)
        _check(
            checks,
            failures,
            "rds_task_state",
            task.status == args.expect_status
            and task.retry_count == args.expect_retry_count
            and bool(task.active_key) == (args.expect_status != "ALL_APPROVED"),
            actual_status=task.status,
            retry_count=task.retry_count,
            active_key_present=bool(task.active_key),
        )
        _check(
            checks,
            failures,
            "rds_approval_sets",
            actual == expected,
            actual={key: sorted(value) for key, value in actual.items()},
            expected={key: sorted(value) for key, value in expected.items()},
        )
        conflict_files = sorted({block.file_path for block in blocks})
        _check(
            checks,
            failures,
            "rds_conflict_shape",
            len(blocks) == args.expect_conflict_blocks
            and len(conflict_files) == args.expect_conflict_files
            and [block.conflict_id for block in blocks] == [f"C{index:04d}" for index in range(1, len(blocks) + 1)],
            block_count=len(blocks),
            files=conflict_files,
        )
        _check(
            checks,
            failures,
            "rds_block_contract",
            all(
                block.discussion_id
                and block.ai_resolution
                and block.resolved_path
                and block.resolved_start_line
                and block.resolved_end_line
                and block.attribution_status == "VERIFIED"
                for block in blocks
            ),
            discussion_ids=[block.discussion_id for block in blocks],
        )
        lark_match = LARK_FILE_URL.fullmatch(task.lark_doc_url or "")
        _check(
            checks,
            failures,
            "rds_lark_url",
            lark_match is not None,
            url=task.lark_doc_url,
            file_token=lark_match.group("token") if lark_match else None,
        )

        events = await repo.list_events(task.id)
        counts = Counter(event.event_type for event in events)
        retry_events_ok = True
        if args.expect_retry_count:
            retry_events_ok = all(
                counts[name] >= 1
                for name in (
                    "RETRY_REQUESTED",
                    "RETRY_CLAIMED",
                    "RETRY_COCO_SUBMITTED",
                    "RETRY_COCO_VERIFIED",
                    "RETRY_LARK_MARKDOWN_UPLOADED",
                    "RETRY_PUBLISHED",
                )
            )
            if args.expect_retry_replies:
                retry_events_ok = retry_events_ok and (counts["RETRY_DISCUSSION_REPLIED"] == args.expect_retry_replies)
        _check(
            checks,
            failures,
            "rds_retry_events",
            retry_events_ok,
            counts=dict(sorted(counts.items())),
        )
        notification = await repo.get_lark_notification_for_task(
            task_id=task.id,
            notification_type="ALL_APPROVED",
        )
        if args.expect_status == "ALL_APPROVED":
            notification_ok = bool(
                notification is not None
                and notification.status == "SENT"
                and notification.recipient_email == task.trigger_author_email
                and notification.recipient_open_id
                and notification.message_id
                and notification.mr_url == task.new_mr_url
                and counts["LARK_NOTIFICATION_QUEUED"] == 1
                and counts["LARK_NOTIFICATION_SENT"] == 1
            )
        else:
            notification_ok = notification is None
        _check(
            checks,
            failures,
            "rds_lark_notification",
            notification_ok,
            status=notification.status if notification else None,
            attempts=notification.attempts if notification else None,
            message_id=notification.message_id if notification else None,
        )
        if args.expect_retry_count:
            upload_event = next(
                event for event in reversed(events) if event.event_type == "RETRY_LARK_MARKDOWN_UPLOADED"
            )
            _check(
                checks,
                failures,
                "retry_lark_link_changed",
                upload_event.payload.get("old_url") != upload_event.payload.get("new_url") == task.lark_doc_url,
                old_url=upload_event.payload.get("old_url"),
                new_url=upload_event.payload.get("new_url"),
            )

        secrets = build_secrets_provider(settings)
        client = NextCodeClient(settings, secrets)
        if not task.resolve_commit_sha or task.new_mr_id is None:
            raise RuntimeError("task has no resolver commit or MR")
        source, target, resolver = await asyncio.gather(
            get_branch(client, task.repo_id, task.source_branch),
            get_branch(client, task.repo_id, task.target_branch),
            get_branch(client, task.repo_id, task.resolve_branch),
        )
        _check(
            checks,
            failures,
            "nextcode_branch_heads",
            source is not None
            and source.commit_sha == task.source_commit_sha
            and target is not None
            and target.commit_sha == task.target_commit_sha
            and resolver is not None
            and resolver.commit_sha == task.resolve_commit_sha,
            source_head=source.commit_sha if source else None,
            target_head=target.commit_sha if target else None,
            resolver_head=resolver.commit_sha if resolver else None,
        )
        commit = await get_commit(client, task.repo_id, task.resolve_commit_sha)
        expected_parents = 1 if args.expect_retry_count else 2
        _check(
            checks,
            failures,
            "nextcode_resolve_commit_parents",
            len(commit.parents) == expected_parents,
            commit=commit.id,
            parents=list(commit.parents),
        )
        mr = await get_mr(client, task.repo_id, task.new_mr_id)
        expected_description = templates.build_mr_description(
            original_mr_number=task.original_mr_id,
            source_branch=task.source_branch,
            source_commit=task.source_commit_sha or "",
            target_branch=task.target_branch,
            target_commit=task.target_commit_sha or "",
            resolve_branch=task.resolve_branch,
            resolve_commit=task.resolve_commit_sha or "",
            lark_doc_url=task.lark_doc_url or "",
            blocks=_mr_blocks(task),
            retry_count=task.retry_count,
            task_status=task.status,
        )
        mergeability = await get_mergeability(client, task.repo_id, task.new_mr_id)
        _check(
            checks,
            failures,
            "nextcode_resolver_mr",
            mr.source_branch == task.resolve_branch
            and mr.target_branch == task.target_branch
            and mr.web_url == task.new_mr_url
            and mr.description == expected_description
            and mergeability.mergeable,
            web_url=mr.web_url,
            mergeable=mergeability.mergeable,
        )
        discussion_ids = []
        for block in blocks:
            if not block.discussion_id:
                raise RuntimeError(f"conflict {block.conflict_id} has no Discussion")
            discussion_ids.append(block.discussion_id)
        threads = await asyncio.gather(
            *[get_thread(client, task.repo_id, discussion_id) for discussion_id in discussion_ids]
        )
        _check(
            checks,
            failures,
            "nextcode_discussions_reused",
            all(
                thread.id == discussion_id
                for thread, discussion_id in zip(
                    threads,
                    discussion_ids,
                    strict=True,
                )
            ),
            discussion_count=len(threads),
        )
        source_only = await get_file(
            client,
            task.repo_id,
            revision=task.resolve_commit_sha,
            path="validation/resolve_retry_e2e/source_feature.py",
        )
        _check(
            checks,
            failures,
            "source_only_change_preserved",
            source_only is not None and "must-survive-resolution" in source_only.content,
        )
        result["status"] = "passed" if not failures else "failed"
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if not failures else 1
    finally:
        if client is not None:
            await client.aclose()
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
