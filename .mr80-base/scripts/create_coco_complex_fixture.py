"""Create the deterministic multi-block MR used by the Coco TCE validation."""

import argparse
import asyncio
import json
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from git import GitCommandError, Repo

from conciliator.config import Settings
from conciliator.integrations.git.conflicts import parse_conflict_file
from conciliator.integrations.nextcode.branches import (
    CommitFileAction,
    create_branch,
    create_commit,
    get_branch,
)
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.merges import list_conflict_files
from conciliator.integrations.nextcode.mrs import create_mr, get_mergeability
from conciliator.integrations.nextcode.repos import get_repository
from conciliator.utils.secrets import SecretsProvider, build_secrets_provider

REPO_PATH = "xiahaitao/Conciliator"
FIXTURE_ROOT = "validation/coco_complex_conflict"

BASE_POLICY = '''\
from dataclasses import dataclass


@dataclass(frozen=True)
class Change:
    risk: str
    owner_approved: bool = False
    security_review_required: bool = False
    urgent: bool = False
    bulk: bool = False


def choose_route(change: Change) -> str:
    if change.risk == "high":
        return "manual"
    return "auto"


# routing/timeout separator 1
# routing/timeout separator 2
# routing/timeout separator 3
# routing/timeout separator 4
# routing/timeout separator 5
# routing/timeout separator 6


def timeout_seconds(change: Change) -> int:
    return 30


# timeout/labels separator 1
# timeout/labels separator 2
# timeout/labels separator 3
# timeout/labels separator 4
# timeout/labels separator 5
# timeout/labels separator 6


def labels(change: Change) -> list[str]:
    del change
    return ["conciliator"]
'''

SOURCE_POLICY = BASE_POLICY.replace(
    '''def choose_route(change: Change) -> str:
    if change.risk == "high":
        return "manual"
    return "auto"
''',
    '''def choose_route(change: Change) -> str:
    if change.owner_approved:
        return "auto"
    if change.risk == "high":
        return "manual"
    return "auto"
''',
).replace(
    '''def timeout_seconds(change: Change) -> int:
    return 30
''',
    '''def timeout_seconds(change: Change) -> int:
    if change.urgent:
        return 10
    return 30
''',
).replace(
    '''def labels(change: Change) -> list[str]:
    del change
    return ["conciliator"]
''',
    '''def labels(change: Change) -> list[str]:
    del change
    return ["conciliator", "source-change"]
''',
)

TARGET_POLICY = BASE_POLICY.replace(
    '''def choose_route(change: Change) -> str:
    if change.risk == "high":
        return "manual"
    return "auto"
''',
    '''def choose_route(change: Change) -> str:
    if change.security_review_required:
        return "manual"
    if change.risk == "high":
        return "manual"
    return "auto"
''',
).replace(
    '''def timeout_seconds(change: Change) -> int:
    return 30
''',
    '''def timeout_seconds(change: Change) -> int:
    if change.bulk:
        return 60
    return 30
''',
).replace(
    '''def labels(change: Change) -> list[str]:
    del change
    return ["conciliator"]
''',
    '''def labels(change: Change) -> list[str]:
    del change
    return ["conciliator", "target-runtime"]
''',
)

RESOLVED_POLICY = BASE_POLICY.replace(
    '''def choose_route(change: Change) -> str:
    if change.risk == "high":
        return "manual"
    return "auto"
''',
    '''def choose_route(change: Change) -> str:
    if change.security_review_required:
        return "manual"
    if change.owner_approved:
        return "auto"
    if change.risk == "high":
        return "manual"
    return "auto"
''',
).replace(
    '''def timeout_seconds(change: Change) -> int:
    return 30
''',
    '''def timeout_seconds(change: Change) -> int:
    if change.urgent:
        return 10
    if change.bulk:
        return 60
    return 30
''',
).replace(
    '''def labels(change: Change) -> list[str]:
    del change
    return ["conciliator"]
''',
    '''def labels(change: Change) -> list[str]:
    del change
    return ["conciliator", "source-change", "target-runtime"]
''',
)

BASE_FORMATTER = '''\
def render_block(
    conflict_id: str,
    path: str,
    start_line: int,
    end_line: int,
    author_email: str,
) -> str:
    del author_email
    return f"{conflict_id}: {path}:{start_line}-{end_line}"
'''

SOURCE_FORMATTER = BASE_FORMATTER.replace(
    '''    del author_email
    return f"{conflict_id}: {path}:{start_line}-{end_line}"
''',
    '''    return f"{conflict_id}: {path}:{start_line}-{end_line} ({author_email})"
''',
)

TARGET_FORMATTER = BASE_FORMATTER.replace(
    '''    del author_email
    return f"{conflict_id}: {path}:{start_line}-{end_line}"
''',
    '''    del author_email
    return f"[conflict] {conflict_id} | {path} lines {start_line}-{end_line}"
''',
)

RESOLVED_FORMATTER = BASE_FORMATTER.replace(
    '''    del author_email
    return f"{conflict_id}: {path}:{start_line}-{end_line}"
''',
    '''    return f"[conflict] {conflict_id} | {path} lines {start_line}-{end_line} ({author_email})"
''',
)

EXPECTED = '''\
# Coco 复杂冲突验收要求

最终 resolver 分支必须同时满足以下要求：

- `security_review_required=True` 时始终返回 `manual`，优先级高于 owner approval。
- 没有 security review 时，`owner_approved=True` 可以把 high risk 路由为 `auto`。
- urgent timeout 为 10 秒，bulk timeout 为 60 秒，其他情况为 30 秒。
- labels 同时包含 `conciliator`、`source-change` 和 `target-runtime`。
- `render_block` 同时保留 `[conflict]` 前缀、固定字段格式和 author email。
- 运行 `python3 -m unittest discover -s validation/coco_complex_conflict/tests -v`，两侧测试必须全部通过。
- 最终提交必须是 `[source commit, target commit]` 顺序的双 parent merge commit。
- 只允许 push 任务指定的 resolve 分支。
'''

SOURCE_TEST = '''\
import unittest

from validation.coco_complex_conflict.formatter import render_block
from validation.coco_complex_conflict.policy import Change, labels, timeout_seconds, choose_route


class SourceRequirementsTest(unittest.TestCase):
    def test_owner_approval_and_urgent_timeout(self):
        change = Change(risk="high", owner_approved=True, urgent=True)
        self.assertEqual(choose_route(change), "auto")
        self.assertEqual(timeout_seconds(change), 10)

    def test_source_label_and_author_format(self):
        change = Change(risk="low")
        self.assertIn("source-change", labels(change))
        rendered = render_block("C0001", "policy.py", 10, 20, "source@example.com")
        self.assertIn("source@example.com", rendered)


if __name__ == "__main__":
    unittest.main()
'''

TARGET_TEST = '''\
import unittest

from validation.coco_complex_conflict.formatter import render_block
from validation.coco_complex_conflict.policy import Change, labels, timeout_seconds, choose_route


class TargetRequirementsTest(unittest.TestCase):
    def test_security_review_precedes_owner_approval(self):
        change = Change(
            risk="low",
            owner_approved=True,
            security_review_required=True,
        )
        self.assertEqual(choose_route(change), "manual")

    def test_bulk_timeout_and_target_label(self):
        change = Change(risk="low", bulk=True)
        self.assertEqual(timeout_seconds(change), 60)
        self.assertIn("target-runtime", labels(change))

    def test_target_report_format(self):
        rendered = render_block("C0001", "policy.py", 10, 20, "target@example.com")
        self.assertTrue(rendered.startswith("[conflict] C0001 | policy.py lines 10-20"))


if __name__ == "__main__":
    unittest.main()
'''


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default=REPO_PATH)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--output", type=Path)
    provision = parser.add_mutually_exclusive_group()
    provision.add_argument("--provision", action="store_true")
    provision.add_argument(
        "--git-provision",
        action="store_true",
        help="create and push fixture branches with the local repository's Git credentials",
    )
    parser.add_argument("--git-repo", type=Path, default=Path.cwd())
    parser.add_argument("--git-remote", default="origin")
    return parser.parse_args()


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _commit(repo: Repo, message: str) -> str:
    repo.git.add("--all")
    return repo.index.commit(message).hexsha


def _git_fixture_files() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    base = {
        "validation/__init__.py": "",
        f"{FIXTURE_ROOT}/__init__.py": "",
        f"{FIXTURE_ROOT}/tests/__init__.py": "",
        f"{FIXTURE_ROOT}/policy.py": BASE_POLICY,
        f"{FIXTURE_ROOT}/formatter.py": BASE_FORMATTER,
        f"{FIXTURE_ROOT}/EXPECTED.md": EXPECTED,
    }
    source = {
        f"{FIXTURE_ROOT}/policy.py": SOURCE_POLICY,
        f"{FIXTURE_ROOT}/formatter.py": SOURCE_FORMATTER,
        f"{FIXTURE_ROOT}/tests/test_source_requirements.py": SOURCE_TEST,
    }
    target = {
        f"{FIXTURE_ROOT}/policy.py": TARGET_POLICY,
        f"{FIXTURE_ROOT}/formatter.py": TARGET_FORMATTER,
        f"{FIXTURE_ROOT}/tests/test_target_requirements.py": TARGET_TEST,
    }
    return base, source, target


def _provision_with_git(args: argparse.Namespace, names: dict[str, str], result: dict) -> None:
    source_repo = Repo(args.git_repo, search_parent_directories=True)
    remote_url = source_repo.remotes[args.git_remote].url
    with tempfile.TemporaryDirectory(prefix="conciliator-coco-fixture-") as checkout:
        repo = Repo.clone_from(remote_url, checkout, branch=args.base_branch)
        collisions = [
            name
            for name in names.values()
            if str(
                repo.git.ls_remote("--heads", "origin", f"refs/heads/{name}")
            ).strip()
        ]
        if collisions:
            raise RuntimeError("fixture branches already exist: " + ", ".join(collisions))

        root = Path(checkout)
        base_files, source_files, target_files = _git_fixture_files()
        repo.git.checkout("-b", names["base"], f"origin/{args.base_branch}")
        _write_files(root, base_files)
        base_commit = _commit(repo, "test: add Coco complex conflict baseline")

        repo.git.checkout("-b", names["source"], base_commit)
        _write_files(root, source_files)
        source_commit = _commit(repo, "test: add source-side Coco conflict requirements")

        repo.git.checkout("-b", names["target"], base_commit)
        _write_files(root, target_files)
        target_commit = _commit(repo, "test: add target-side Coco runtime requirements")

        repo.git.checkout(names["source"])
        with suppress(GitCommandError):
            repo.git.merge(target_commit, "--no-commit", "--no-ff")
        unmerged_paths = sorted(
            str(path) for path in repo.index.unmerged_blobs()
        )
        parsed = [
            block
            for path in unmerged_paths
            for block in parse_conflict_file(path, (root / path).read_text(encoding="utf-8"))
        ]
        if len(unmerged_paths) != 2 or len(parsed) != 4:
            raise RuntimeError(
                "fixture shape mismatch: expected 2 files/4 blocks, "
                f"got {len(unmerged_paths)} files/{len(parsed)} blocks"
            )
        repo.git.merge("--abort")

        for branch in ("base", "source", "target"):
            repo.git.push("--set-upstream", "origin", names[branch])

        result.update(
            {
                "status": "branches-pushed",
                "base_commit": base_commit,
                "source_commit": source_commit,
                "target_commit": target_commit,
                "conflict_files": unmerged_paths,
                "conflict_blocks": [
                    {
                        "conflict_id": f"C{idx + 1:04d}",
                        "path": block.file_path,
                        "start_line": block.start_line,
                        "end_line": block.end_line,
                    }
                    for idx, block in enumerate(parsed)
                ],
            }
        )


def _clients() -> tuple[NextCodeClient, SecretsProvider]:
    settings = Settings(
        service_psm="coco.fixture.provisioner",
        mysql_psm="unused",
    )
    secrets = build_secrets_provider(settings)
    return NextCodeClient(settings, secrets), secrets


async def _wait_for_conflict(client: NextCodeClient, repo_id: str, mr_number: int):
    for _ in range(20):
        result = await get_mergeability(client, repo_id, mr_number)
        if result.reason == "conflict":
            return result
        await asyncio.sleep(2)
    return await get_mergeability(client, repo_id, mr_number)


async def _run() -> int:
    args = _args()
    names = {
        "base": f"verify/coco-complex-{args.run_id}-base",
        "source": f"verify/coco-complex-{args.run_id}-source",
        "target": f"verify/coco-complex-{args.run_id}-target",
    }
    result: dict = {"run_id": args.run_id, "repo_path": args.repo_path, "branches": names}
    if args.git_provision:
        try:
            _provision_with_git(args, names, result)
        except Exception as exc:
            result["status"] = "failed"
            result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            _emit(result, args.output)
            return 1
        _emit(result, args.output)
        return 0
    if not args.provision:
        result["status"] = "dry-run"
        _emit(result, args.output)
        return 0

    client: NextCodeClient | None = None
    secrets: SecretsProvider | None = None
    try:
        client, secrets = _clients()
        repo = await get_repository(client, path=args.repo_path)
        result["repo_id"] = repo.id
        collisions = [name for name in names.values() if await get_branch(client, repo.id, name) is not None]
        if collisions:
            raise RuntimeError("fixture branches already exist: " + ", ".join(collisions))

        await create_branch(client, repo.id, args.base_branch, names["base"])
        base_commit = await create_commit(
            client,
            repo.id,
            names["base"],
            "test: add Coco complex conflict baseline",
            [
                CommitFileAction("create", "validation/__init__.py", ""),
                CommitFileAction("create", f"{FIXTURE_ROOT}/__init__.py", ""),
                CommitFileAction("create", f"{FIXTURE_ROOT}/tests/__init__.py", ""),
                CommitFileAction("create", f"{FIXTURE_ROOT}/policy.py", BASE_POLICY),
                CommitFileAction("create", f"{FIXTURE_ROOT}/formatter.py", BASE_FORMATTER),
                CommitFileAction("create", f"{FIXTURE_ROOT}/EXPECTED.md", EXPECTED),
            ],
        )
        await create_branch(client, repo.id, base_commit.id, names["source"])
        await create_branch(client, repo.id, base_commit.id, names["target"])

        source_commit = await create_commit(
            client,
            repo.id,
            names["source"],
            "test: add source-side Coco conflict requirements",
            [
                CommitFileAction("update", f"{FIXTURE_ROOT}/policy.py", SOURCE_POLICY),
                CommitFileAction("update", f"{FIXTURE_ROOT}/formatter.py", SOURCE_FORMATTER),
                CommitFileAction("create", f"{FIXTURE_ROOT}/tests/test_source_requirements.py", SOURCE_TEST),
            ],
        )
        target_commit = await create_commit(
            client,
            repo.id,
            names["target"],
            "test: add target-side Coco runtime requirements",
            [
                CommitFileAction("update", f"{FIXTURE_ROOT}/policy.py", TARGET_POLICY),
                CommitFileAction("update", f"{FIXTURE_ROOT}/formatter.py", TARGET_FORMATTER),
                CommitFileAction("create", f"{FIXTURE_ROOT}/tests/test_target_requirements.py", TARGET_TEST),
            ],
        )
        mr = await create_mr(
            client,
            repo.id,
            source=names["source"],
            target=names["target"],
            title=f"[verification] Coco complex conflict {args.run_id}",
            description=(
                "Coco resolver TCE validation fixture.\n\n"
                f"Run ID: `{args.run_id}`\n"
                f"Acceptance rules: `{FIXTURE_ROOT}/EXPECTED.md`\n"
                "Expected conflicts: two files and four marker blocks."
            ),
        )
        mergeability = await _wait_for_conflict(client, repo.id, mr.iid)
        conflicts = await list_conflict_files(client, repo.id, mr.iid)
        parsed = [
            block
            for conflict_file in conflicts
            for block in parse_conflict_file(conflict_file.file_path, conflict_file.content)
        ]
        files = sorted({block.file_path for block in parsed})
        if mergeability.reason != "conflict":
            raise RuntimeError(f"fixture MR is not conflicted: {mergeability.reason or '<none>'}")
        if len(files) != 2 or len(parsed) != 4:
            raise RuntimeError(
                f"fixture shape mismatch: expected 2 files/4 blocks, got {len(files)} files/{len(parsed)} blocks"
            )

        result.update(
            {
                "status": "ok",
                "base_commit": base_commit.id,
                "source_commit": source_commit.id,
                "target_commit": target_commit.id,
                "mr_number": mr.iid,
                "mr_url": mr.web_url,
                "mergeable": mergeability.mergeable,
                "unmergeable_reason": mergeability.reason,
                "conflict_files": files,
                "conflict_blocks": [
                    {
                        "conflict_id": f"C{idx + 1:04d}",
                        "path": block.file_path,
                        "start_line": block.start_line,
                        "end_line": block.end_line,
                    }
                    for idx, block in enumerate(parsed)
                ],
            }
        )
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _emit(result, args.output)
        return 1
    finally:
        if client is not None:
            await client.aclose()
        if secrets is not None:
            secrets.close()

    _emit(result, args.output)
    return 0


def _emit(result: dict, output: Path | None) -> None:
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        output.chmod(0o600)
    print(rendered, end="")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
