"""Repeatable NextCode App OpenAPI contract verification.

This script never prints the App Secret or ZTI token. It uses Conciliator's
configured SecretsProvider, so production reads both NextCode secrets from KMS
and development reads credentials from environment configuration.

Example:
  uv run python scripts/verify_nextcode_contracts.py \
    --provision-conflict-fixture --run-id 20260826-01
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime

from conciliator.config import Settings
from conciliator.integrations.nextcode.branches import (
    CommitFileAction,
    create_branch,
    create_commit,
)
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.discussions import (
    create_discussion,
    create_mr_comment,
    get_thread,
)
from conciliator.integrations.nextcode.merges import list_conflict_files
from conciliator.integrations.nextcode.mrs import (
    create_mr,
    get_mergeability,
    get_mr,
    list_repo_mrs,
    update_mr_description,
)
from conciliator.integrations.nextcode.repos import get_blame, get_repository
from conciliator.utils.secrets import SecretsProvider, build_secrets_provider

APP_ID = "792763146702616"
REPO_PATH = "xiahaitao/Conciliator"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default=REPO_PATH)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--provision-conflict-fixture", action="store_true")
    parser.add_argument(
        "--allow-missing-zti",
        action="store_true",
        help="local verification only; production must send the dynamic ZTI token",
    )
    return parser.parse_args()


def _client(*, zti_required: bool) -> tuple[NextCodeClient, SecretsProvider, Settings]:
    settings = Settings(
        service_psm="nextcode.contract.verifier",
        mysql_psm="unused",
        nextcode_zti_required=zti_required,
    )
    missing = [
        name
        for name, value in {
            "KMS_SECRET_NEXTCODE_APP_SECRET_ALIAS": (settings.kms_secret_nextcode_app_secret_alias),
            "KMS_SECRET_NEXTCODE_WEBHOOK_SECRET_ALIAS": (settings.kms_secret_nextcode_webhook_secret_alias),
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError("missing KMS aliases: " + ", ".join(missing))
    secrets = build_secrets_provider(settings)
    return NextCodeClient(settings, secrets), secrets, settings


async def _wait_for_conflict(client: NextCodeClient, repo_id: str, mr_number: int):
    result = None
    for _ in range(15):
        result = await get_mergeability(client, repo_id, mr_number)
        if result.reason == "conflict":
            return result
        await asyncio.sleep(2)
    return result


async def _run() -> int:
    args = _args()
    results: dict = {
        "app_id": APP_ID,
        "repo_path": args.repo_path,
        "run_id": args.run_id,
        "checks": {},
        "intentional_skips": {
            "git_clone": (
                "App service-account clone is documented as Bits-pipeline-only; "
                "ordinary TCE/local container clone is not executed."
            ),
            "repository_admins": ("The current App OpenAPI service does not expose ListRepoMembers."),
        },
    }
    client: NextCodeClient | None = None
    secrets: SecretsProvider | None = None
    try:
        client, secrets, settings = _client(zti_required=not args.allow_missing_zti)
        results["app_id"] = settings.nextcode_app_id
        repo = await get_repository(client, path=args.repo_path)
        results["checks"]["GetRepository"] = {
            "ok": True,
            "repo_id": repo.id,
            "path": repo.path,
        }
        if not args.provision_conflict_fixture:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0

        source = f"verify/nextcode-{args.run_id}-test1"
        target = f"verify/nextcode-{args.run_id}-test2"
        fixture_path = f"docs/nextcode-contract-fixtures/{args.run_id}.txt"

        source_branch = await create_branch(client, repo.id, args.base_branch, source)
        target_branch = await create_branch(client, repo.id, args.base_branch, target)
        results["checks"]["CreateBranch"] = {
            "ok": True,
            "source": source_branch.name,
            "target": target_branch.name,
        }

        source_commit = await create_commit(
            client,
            repo.id,
            source,
            "test: create NextCode source-side conflict fixture",
            [
                CommitFileAction(
                    "create",
                    fixture_path,
                    "source branch value\nshared trailing line\n",
                )
            ],
        )
        target_commit = await create_commit(
            client,
            repo.id,
            target,
            "test: create NextCode target-side conflict fixture",
            [
                CommitFileAction(
                    "create",
                    fixture_path,
                    "target branch value\nshared trailing line\n",
                )
            ],
        )
        results["checks"]["CreateCommit"] = {
            "ok": True,
            "source_commit": source_commit.id,
            "target_commit": target_commit.id,
        }

        description = (
            "NextCode App OpenAPI contract verification fixture.\n\n"
            f"Run ID: `{args.run_id}`\n"
            "Expected result: source -> target has one add/add conflict."
        )
        created = await create_mr(
            client,
            repo.id,
            source=source,
            target=target,
            title=f"[verification] NextCode contract {args.run_id}",
            description=description,
        )
        results["checks"]["CreateMergeRequest"] = {
            "ok": True,
            "number": created.iid,
            "id": created.id,
            "url": created.web_url,
        }

        fetched = await get_mr(client, repo.id, created.iid)
        results["checks"]["GetMergeRequest"] = {
            "ok": True,
            "source_repo_id": fetched.source_repo_id,
            "target_repo_id": fetched.target_repo_id,
            "source_branch": fetched.source_branch,
            "target_branch": fetched.target_branch,
            "author_email": fetched.author_email,
            "version_count": len(fetched.versions),
        }

        mergeability = await _wait_for_conflict(client, repo.id, created.iid)
        results["checks"]["GetMergeRequestMergeability"] = {
            "ok": mergeability is not None and mergeability.reason == "conflict",
            "mergeable": mergeability.mergeable if mergeability else None,
            "reason": mergeability.reason if mergeability else None,
            "detail": mergeability.detail if mergeability else None,
        }

        conflicts = await list_conflict_files(client, repo.id, created.iid)
        results["checks"]["ListMergeRequestConflicts/GetMergeRequestConflict"] = {
            "ok": bool(conflicts),
            "files": [
                {
                    "path": item.file_path,
                    "first_path": item.first_path,
                    "second_path": item.second_path,
                    "conflicts_id": item.conflicts_id,
                    "has_markers": all(marker in item.content for marker in ("<<<<<<<", "=======", ">>>>>>>")),
                }
                for item in conflicts
            ],
        }

        blame = await get_blame(
            client,
            repo.id,
            revision=source,
            path=fixture_path,
            start_line=1,
            end_line=2,
        )
        results["checks"]["GetBlame"] = {
            "ok": bool(blame),
            "segments": [
                {
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "commit_id": item.commit_id,
                    "author_email": item.author_email,
                }
                for item in blame
            ],
        }

        main_comment = await create_mr_comment(
            client,
            repo.id,
            created.iid,
            f"NextCode contract verification comment ({args.run_id})",
        )
        inline_comment = await create_discussion(
            client,
            repo.id,
            created.iid,
            file_path=fixture_path,
            line=1,
            body=f"NextCode inline contract verification ({args.run_id})",
        )
        thread = await get_thread(client, repo.id, inline_comment.id)
        results["checks"]["CreateComment/GetThread"] = {
            "ok": bool(thread.positions),
            "main_comment_id": main_comment.comment_id,
            "main_thread_id": main_comment.id,
            "inline_comment_id": inline_comment.comment_id,
            "inline_thread_id": inline_comment.id,
            "positions": list(thread.positions),
        }

        updated_description = description + "\n\nDescription update API: verified."
        await update_mr_description(client, repo.id, created.iid, updated_description)
        results["checks"]["UpdateMergeRequest"] = {"ok": True}

        listed = await list_repo_mrs(
            client,
            repo.id,
            source_branch=source,
            target_branch=target,
            status="open",
        )
        results["checks"]["ListRepoMergeRequests"] = {
            "ok": any(item.iid == created.iid for item in listed),
            "numbers": [item.iid for item in listed],
        }
    except Exception as exc:
        results["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 1
    finally:
        if client is not None:
            await client.aclose()
        if secrets is not None:
            secrets.close()

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
