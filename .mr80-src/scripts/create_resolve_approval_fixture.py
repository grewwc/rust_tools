#!/usr/bin/env python3
"""Create and push the two-branch fixture for the approval E2E flow.

The script creates exactly two remote branches.  The target branch contains the
common fixture baseline, while both branches then change two blocks in each of
two files.  Branch-specific padding makes every source/target block range
different without becoming part of the conflicts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from git import GitCommandError, Repo

from conciliator.integrations.git.conflicts import parse_conflict_file

FIXTURE_ROOT = "validation/resolve_approval_e2e"

BASE_POLICY = """\
from dataclasses import dataclass


@dataclass(frozen=True)
class Change:
    risk: str
    owner_approved: bool = False
    security_review_required: bool = False
    urgent: bool = False
    bulk: bool = False


# SOURCE-PADDING-POINT
# stable source/target padding separator 1
# stable source/target padding separator 2
# stable source/target padding separator 3
# stable source/target padding separator 4
# TARGET-PADDING-POINT
# stable padding/block separator 1
# stable padding/block separator 2
# stable padding/block separator 3
# stable padding/block separator 4


# ROUTE-BLOCK-BEGIN
def choose_route(change: Change) -> str:
    if change.risk == "high":
        return "manual"
    return "auto"
# ROUTE-BLOCK-END


# stable conflict separator 1
# stable conflict separator 2
# stable conflict separator 3
# stable conflict separator 4
# stable conflict separator 5
# stable conflict separator 6
# stable conflict separator 7
# stable conflict separator 8


# TIMEOUT-BLOCK-BEGIN
def timeout_seconds(change: Change) -> int:
    del change
    return 30
# TIMEOUT-BLOCK-END
"""

SOURCE_POLICY = (
    BASE_POLICY.replace(
        "# SOURCE-PADDING-POINT\n",
        "# SOURCE-PADDING-POINT\n# source physical line offset 1\n# source physical line offset 2\n",
    )
    .replace(
        """def choose_route(change: Change) -> str:
    if change.risk == "high":
        return "manual"
    return "auto"
""",
        """def choose_route(change: Change) -> str:
    if change.owner_approved:
        return "auto"
    if change.risk == "high":
        return "manual"
    return "auto"
""",
    )
    .replace(
        """def timeout_seconds(change: Change) -> int:
    del change
    return 30
""",
        """def timeout_seconds(change: Change) -> int:
    if change.urgent:
        return 10
    return 30
""",
    )
)

TARGET_POLICY = (
    BASE_POLICY.replace(
        "# TARGET-PADDING-POINT\n",
        "# TARGET-PADDING-POINT\n"
        "# target physical line offset 1\n"
        "# target physical line offset 2\n"
        "# target physical line offset 3\n"
        "# target physical line offset 4\n"
        "# target physical line offset 5\n",
    )
    .replace(
        """def choose_route(change: Change) -> str:
    if change.risk == "high":
        return "manual"
    return "auto"
""",
        """def choose_route(change: Change) -> str:
    if change.security_review_required:
        return "manual"
    if change.risk == "high":
        return "manual"
    return "auto"
""",
    )
    .replace(
        """def timeout_seconds(change: Change) -> int:
    del change
    return 30
""",
        """def timeout_seconds(change: Change) -> int:
    if change.bulk:
        return 60
    return 30
""",
    )
)

RESOLVED_POLICY = (
    BASE_POLICY.replace(
        "# SOURCE-PADDING-POINT\n",
        "# SOURCE-PADDING-POINT\n# source physical line offset 1\n# source physical line offset 2\n",
    )
    .replace(
        "# TARGET-PADDING-POINT\n",
        "# TARGET-PADDING-POINT\n"
        "# target physical line offset 1\n"
        "# target physical line offset 2\n"
        "# target physical line offset 3\n"
        "# target physical line offset 4\n"
        "# target physical line offset 5\n",
    )
    .replace(
        """def choose_route(change: Change) -> str:
    if change.risk == "high":
        return "manual"
    return "auto"
""",
        """def choose_route(change: Change) -> str:
    if change.security_review_required:
        return "manual"
    if change.owner_approved:
        return "auto"
    if change.risk == "high":
        return "manual"
    return "auto"
""",
    )
    .replace(
        """def timeout_seconds(change: Change) -> int:
    del change
    return 30
""",
        """def timeout_seconds(change: Change) -> int:
    if change.urgent:
        return 10
    if change.bulk:
        return 60
    return 30
""",
    )
)

BASE_FORMATTER = """\
# TARGET-PADDING-POINT
# stable target/source padding separator 1
# stable target/source padding separator 2
# stable target/source padding separator 3
# stable target/source padding separator 4
# SOURCE-PADDING-POINT
# stable padding/block separator 1
# stable padding/block separator 2
# stable padding/block separator 3
# stable padding/block separator 4


# LABELS-BLOCK-BEGIN
def labels() -> list[str]:
    return ["conciliator"]
# LABELS-BLOCK-END


# stable conflict separator 1
# stable conflict separator 2
# stable conflict separator 3
# stable conflict separator 4
# stable conflict separator 5
# stable conflict separator 6
# stable conflict separator 7
# stable conflict separator 8


# RENDER-BLOCK-BEGIN
def render_block(conflict_id: str, author_email: str) -> str:
    del author_email
    return conflict_id
# RENDER-BLOCK-END
"""

SOURCE_FORMATTER = (
    BASE_FORMATTER.replace(
        "# SOURCE-PADDING-POINT\n",
        "# SOURCE-PADDING-POINT\n"
        "# source physical line offset 1\n"
        "# source physical line offset 2\n"
        "# source physical line offset 3\n"
        "# source physical line offset 4\n",
    )
    .replace(
        'def labels() -> list[str]:\n    return ["conciliator"]\n',
        'def labels() -> list[str]:\n    return ["conciliator", "source-change"]\n',
    )
    .replace(
        """def render_block(conflict_id: str, author_email: str) -> str:
    del author_email
    return conflict_id
""",
        """def render_block(conflict_id: str, author_email: str) -> str:
    return f"{conflict_id} ({author_email})"
""",
    )
)

TARGET_FORMATTER = (
    BASE_FORMATTER.replace(
        "# TARGET-PADDING-POINT\n",
        "# TARGET-PADDING-POINT\n# target physical line offset 1\n",
    )
    .replace(
        'def labels() -> list[str]:\n    return ["conciliator"]\n',
        'def labels() -> list[str]:\n    return ["conciliator", "target-runtime"]\n',
    )
    .replace(
        """def render_block(conflict_id: str, author_email: str) -> str:
    del author_email
    return conflict_id
""",
        """def render_block(conflict_id: str, author_email: str) -> str:
    del author_email
    return f"[conflict] {conflict_id}"
""",
    )
)

RESOLVED_FORMATTER = (
    BASE_FORMATTER.replace(
        "# TARGET-PADDING-POINT\n",
        "# TARGET-PADDING-POINT\n# target physical line offset 1\n",
    )
    .replace(
        "# SOURCE-PADDING-POINT\n",
        "# SOURCE-PADDING-POINT\n"
        "# source physical line offset 1\n"
        "# source physical line offset 2\n"
        "# source physical line offset 3\n"
        "# source physical line offset 4\n",
    )
    .replace(
        'def labels() -> list[str]:\n    return ["conciliator"]\n',
        'def labels() -> list[str]:\n    return ["conciliator", "source-change", "target-runtime"]\n',
    )
    .replace(
        """def render_block(conflict_id: str, author_email: str) -> str:
    del author_email
    return conflict_id
""",
        """def render_block(conflict_id: str, author_email: str) -> str:
    return f"[conflict] {conflict_id} ({author_email})"
""",
    )
)

SOURCE_TEST = """\
import unittest

from validation.resolve_approval_e2e.formatter import labels, render_block
from validation.resolve_approval_e2e.policy import Change, choose_route, timeout_seconds


class SourceRequirementsTest(unittest.TestCase):
    def test_source_requirements(self):
        change = Change(risk="high", owner_approved=True, urgent=True)
        self.assertEqual(choose_route(change), "auto")
        self.assertEqual(timeout_seconds(change), 10)
        self.assertIn("source-change", labels())
        self.assertIn("owner@example.com", render_block("C0001", "owner@example.com"))
"""

TARGET_TEST = """\
import unittest

from validation.resolve_approval_e2e.formatter import labels, render_block
from validation.resolve_approval_e2e.policy import Change, choose_route, timeout_seconds


class TargetRequirementsTest(unittest.TestCase):
    def test_target_requirements(self):
        change = Change(risk="low", owner_approved=True, security_review_required=True, bulk=True)
        self.assertEqual(choose_route(change), "manual")
        self.assertEqual(timeout_seconds(change), 60)
        self.assertIn("target-runtime", labels())
        self.assertTrue(render_block("C0001", "owner@example.com").startswith("[conflict]"))
"""

EXPECTED = """\
# Resolve + approval E2E fixture

Resolve all four conflict blocks while preserving both branches' requirements.

- Security review takes priority over owner approval.
- Owner approval can route a high-risk change automatically when no security review is required.
- Urgent timeout is 10, bulk timeout is 60, and the default is 30.
- Labels contain `conciliator`, `source-change`, and `target-runtime`.
- Rendered blocks contain both the `[conflict]` prefix and the author email.
- Run `python3 -m unittest discover -s validation/resolve_approval_e2e/tests -v`.
- Create a two-parent merge commit in `[source commit, target commit]` order and push only the assigned resolve branch.
"""

BLOCK_MARKERS = (
    ("formatter.py", "# LABELS-BLOCK-BEGIN", "# LABELS-BLOCK-END"),
    ("formatter.py", "# RENDER-BLOCK-BEGIN", "# RENDER-BLOCK-END"),
    ("policy.py", "# ROUTE-BLOCK-BEGIN", "# ROUTE-BLOCK-END"),
    ("policy.py", "# TIMEOUT-BLOCK-BEGIN", "# TIMEOUT-BLOCK-END"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-branch", default="main")
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--git-repo", type=Path, default=Path.cwd())
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate locally without pushing either fixture branch",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"/tmp/conciliator-resolve-approval-{args.run_id}.json")
    return args


def _files(policy: str, formatter: str, test_name: str, test: str) -> dict[str, str]:
    return {
        "validation/__init__.py": "",
        f"{FIXTURE_ROOT}/__init__.py": "",
        f"{FIXTURE_ROOT}/policy.py": policy,
        f"{FIXTURE_ROOT}/formatter.py": formatter,
        f"{FIXTURE_ROOT}/tests/__init__.py": "",
        f"{FIXTURE_ROOT}/tests/{test_name}": test,
        f"{FIXTURE_ROOT}/EXPECTED.md": EXPECTED,
    }


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _commit(repo: Repo, message: str) -> str:
    repo.git.add("--all")
    return repo.index.commit(message).hexsha


def _range(content: str, begin: str, end: str) -> tuple[int, int]:
    lines = content.splitlines()
    return lines.index(begin) + 2, lines.index(end)


def _run_tests(root: Path) -> str:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        f"{FIXTURE_ROOT}/tests",
        "-v",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout + completed.stderr).strip()


def _main() -> int:
    args = _args()
    source_name = f"verify/resolve-approval-{args.run_id}-test1"
    target_name = f"verify/resolve-approval-{args.run_id}-test2"
    result: dict = {
        "run_id": args.run_id,
        "source_branch": source_name,
        "target_branch": target_name,
    }
    local_repo = Repo(args.git_repo, search_parent_directories=True)
    remote_url = local_repo.remotes[args.git_remote].url
    with tempfile.TemporaryDirectory(prefix="conciliator-resolve-approval-fixture-") as checkout:
        repo = Repo.clone_from(remote_url, checkout, branch=args.base_branch)
        collisions = [
            name
            for name in (source_name, target_name)
            if cast(
                str,
                repo.git.ls_remote(
                    "--heads",
                    "origin",
                    f"refs/heads/{name}",
                ),
            ).strip()
        ]
        if collisions:
            raise RuntimeError("fixture branches already exist: " + ", ".join(collisions))

        root = Path(checkout)
        base_files = _files(
            BASE_POLICY,
            BASE_FORMATTER,
            "test_placeholder.py",
            "",
        )
        repo.git.checkout("-b", target_name, f"origin/{args.base_branch}")
        _write_files(root, base_files)
        base_commit = _commit(repo, "test: add resolve approval fixture baseline")

        repo.git.checkout("-b", source_name, base_commit)
        placeholder = root / FIXTURE_ROOT / "tests/test_placeholder.py"
        placeholder.unlink()
        _write_files(
            root,
            _files(
                SOURCE_POLICY,
                SOURCE_FORMATTER,
                "test_source_requirements.py",
                SOURCE_TEST,
            ),
        )
        source_commit = _commit(
            repo,
            "test: add source-side resolve approval requirements",
        )
        source_test = _run_tests(root)

        repo.git.checkout(target_name)
        placeholder.unlink()
        _write_files(
            root,
            _files(
                TARGET_POLICY,
                TARGET_FORMATTER,
                "test_target_requirements.py",
                TARGET_TEST,
            ),
        )
        target_commit = _commit(
            repo,
            "test: add target-side resolve approval requirements",
        )
        target_test = _run_tests(root)

        ranges = []
        content_by_branch = {
            "source": {
                "policy.py": SOURCE_POLICY,
                "formatter.py": SOURCE_FORMATTER,
            },
            "target": {
                "policy.py": TARGET_POLICY,
                "formatter.py": TARGET_FORMATTER,
            },
        }
        for idx, (file_name, begin, end) in enumerate(BLOCK_MARKERS, start=1):
            source_range = _range(content_by_branch["source"][file_name], begin, end)
            target_range = _range(content_by_branch["target"][file_name], begin, end)
            if source_range == target_range:
                raise RuntimeError(f"C{idx:04d} has equal source/target ranges: {source_range}")
            ranges.append(
                {
                    "conflict_id": f"C{idx:04d}",
                    "path": f"{FIXTURE_ROOT}/{file_name}",
                    "source_range": source_range,
                    "target_range": target_range,
                }
            )

        repo.git.checkout(source_name)
        with suppress(GitCommandError):
            repo.git.merge(target_commit, "--no-commit", "--no-ff")
        unmerged = sorted(str(path) for path in repo.index.unmerged_blobs())
        conflict_blocks = [
            block
            for path in unmerged
            for block in parse_conflict_file(
                path,
                (root / path).read_text(encoding="utf-8"),
            )
        ]
        if len(unmerged) != 2 or len(conflict_blocks) != 4:
            raise RuntimeError(
                "fixture shape mismatch: expected 2 files/4 blocks, got "
                f"{len(unmerged)} files/{len(conflict_blocks)} blocks"
            )
        _write_files(
            root,
            {
                f"{FIXTURE_ROOT}/policy.py": RESOLVED_POLICY,
                f"{FIXTURE_ROOT}/formatter.py": RESOLVED_FORMATTER,
            },
        )
        repo.git.add(
            f"{FIXTURE_ROOT}/policy.py",
            f"{FIXTURE_ROOT}/formatter.py",
        )
        resolved_test = _run_tests(root)
        repo.git.merge("--abort")

        if not args.dry_run:
            repo.git.push(
                "--set-upstream",
                "origin",
                f"{source_name}:{source_name}",
            )
            repo.git.push(
                "--set-upstream",
                "origin",
                f"{target_name}:{target_name}",
            )
        result.update(
            {
                "status": ("fixture-validated" if args.dry_run else "branches-pushed"),
                "base_commit": base_commit,
                "source_commit": source_commit,
                "target_commit": target_commit,
                "conflict_files": unmerged,
                "conflict_block_count": len(conflict_blocks),
                "physical_ranges": ranges,
                "tests": {
                    "source": source_test,
                    "target": target_test,
                    "expected_resolution": resolved_test,
                },
            }
        )

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
