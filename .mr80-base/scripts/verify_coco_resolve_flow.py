"""Run the real resolver through local report generation inside a TCE Pod."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.integrations.coco.client import CocoClient
from conciliator.integrations.git.conflicts import parse_conflict_file
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.merges import list_conflict_files
from conciliator.integrations.nextcode.mrs import get_mergeability, get_mr
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.notification_service import NotificationService
from conciliator.services.resolve_service import ResolveDeps, ResolveService
from conciliator.utils.secrets import SecretsProvider, build_secrets_provider


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="1106267")
    parser.add_argument("--mr-number", type=int)
    parser.add_argument("--resume-task-id", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/conciliator-validation"))
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--expect-conflict-files", type=int, default=2)
    parser.add_argument("--expect-conflict-blocks", type=int, default=4)
    parser.add_argument(
        "--fixture-root",
        default="validation/coco_complex_conflict",
    )
    args = parser.parse_args()
    if (args.mr_number is None) == (args.resume_task_id is None):
        parser.error("provide exactly one of --mr-number or --resume-task-id")
    return args


async def _run() -> int:
    args = _args()
    settings = Settings()  # type: ignore[call-arg]
    result: dict = {
        "status": "running",
        "repo_id": args.repo_id,
        "mr_number": args.mr_number,
        "resume_task_id": args.resume_task_id,
    }
    if not settings.coco_workspace_id.isdigit():
        return _fail(result, "COCO_WORKSPACE_ID must be the numeric Workspace ID")
    if not settings.coco_default_agent_id:
        return _fail(result, "COCO_DEFAULT_AGENT_ID is required")

    secrets: SecretsProvider | None = None
    nc: NextCodeClient | None = None
    coco: CocoClient | None = None
    engine = None
    try:
        secrets = build_secrets_provider(settings)
        nc = NextCodeClient(settings, secrets)
        coco = CocoClient(settings, secrets)
        engine = create_engine_from_settings(settings)
        repo = TaskRepository(get_session_factory(engine))
        service = ResolveService(
            ResolveDeps(
                nc=nc,
                coco=coco,
                lark=None,
                repo=repo,
                notification=NotificationService(nc),
                settings=settings,
            )
        )

        if args.resume_task_id is None:
            assert args.mr_number is not None
            mr = await get_mr(nc, args.repo_id, args.mr_number)
            mergeability = await get_mergeability(nc, args.repo_id, args.mr_number)
            if mergeability.reason != "conflict":
                return _fail(result, f"MR is not conflicted: {mergeability.reason or '<none>'}")
            conflict_files = await list_conflict_files(nc, args.repo_id, args.mr_number)
            parsed = [
                block
                for conflict_file in conflict_files
                for block in parse_conflict_file(conflict_file.file_path, conflict_file.content)
            ]
            if len(conflict_files) != args.expect_conflict_files or len(parsed) != args.expect_conflict_blocks:
                return _fail(
                    result,
                    "unexpected fixture shape: "
                    f"{len(conflict_files)} files/{len(parsed)} blocks",
                )
            task_id = await service.handle_trigger(
                repo_id=args.repo_id,
                mr_iid=args.mr_number,
                author_email=mr.author_email,
                payload={
                    "source_branch": mr.source_branch or "",
                    "target_branch": mr.target_branch,
                    "mr_title": mr.title,
                    "validation_mode": "local-report",
                },
                trigger_type="TCE_VALIDATION",
            )
            if task_id is None:
                return _fail(result, "an active task already exists for this MR; use --resume-task-id")
            instructions = (
                f"读取 {args.fixture_root}/EXPECTED.md 并逐条满足要求；"
                f"运行 python3 -m unittest discover -s {args.fixture_root}/tests -v；"
                "不得修改 EXPECTED.md，不得删除测试，只修改解决冲突和满足两侧测试所必需的文件。"
            )
            await service.run_pipeline(task_id, coco_instructions=instructions)
        else:
            task_id = args.resume_task_id

        result["task_id"] = task_id
        task = await repo.get(task_id)
        if task is None:
            return _fail(result, f"task {task_id} does not exist")
        if task.status == TaskStatus.FAILED.value:
            return _fail(result, task.last_error or "task failed before Coco polling")
        if task.status not in {TaskStatus.COCO_RUNNING.value, TaskStatus.DOC_CREATED.value}:
            return _fail(result, f"task is not ready for Coco polling: {task.status}")

        result.update(
            {
                "resolve_branch": task.resolve_branch,
                "source_commit": task.source_commit_sha,
                "target_commit": task.target_commit_sha,
                "coco_session_id": task.coco_session_id,
                "coco_run_id": task.coco_run_id,
            }
        )
        if task.status == TaskStatus.DOC_CREATED.value:
            directory = args.output_dir / f"task-{task_id}"
            result.update(
                {
                    "status": "ok",
                    "artifacts_dir": str(directory),
                    "note": "task already reached DOC_CREATED; existing local artifacts were not rewritten",
                }
            )
            return _emit(result, 0)

        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            artifacts = await service.continue_after_coco(
                task_id,
                local_report_dir=args.output_dir,
            )
            if artifacts is not None:
                result.update(
                    {
                        "status": "ok",
                        "artifacts_dir": str(artifacts.directory),
                        "preview_path": str(artifacts.preview_path),
                        "preview_sha256": artifacts.preview_sha256,
                    }
                )
                return _emit(result, 0)
            task = await repo.get(task_id)
            if task is None:
                return _fail(result, "task disappeared while polling Coco")
            if task.status == TaskStatus.FAILED.value:
                return _fail(result, task.last_error or "task failed while polling Coco")
            await asyncio.sleep(args.poll_interval)
        return _fail(result, f"Coco did not finish within {args.timeout} seconds")
    except Exception as exc:
        result["failure_type"] = type(exc).__name__
        return _fail(result, str(exc))
    finally:
        if coco is not None:
            await coco.aclose()
        if nc is not None:
            await nc.aclose()
        if secrets is not None:
            secrets.close()
        if engine is not None:
            engine.dispose()


def _fail(result: dict, message: str) -> int:
    result["status"] = "failed"
    result["error"] = message
    return _emit(result, 1)


def _emit(result: dict, code: int) -> int:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
