#!/usr/bin/env python3
"""Create two branches with four files and twelve stable conflict blocks."""

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

FIXTURE_ROOT = "validation/resolve_retry_e2e"
FILES = ("access.py", "formatting.py", "policy.py", "runtime.py")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--git-repo", type=Path, default=Path.cwd())
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"/tmp/conciliator-resolve-retry-{args.run_id}.json")
    return args


def _module(file_name: str, side: str) -> str:
    stem = file_name.removesuffix(".py")
    source_padding = "# source line offset 1\n# source line offset 2\n" if side == "source" else ""
    target_padding = (
        "# target line offset 1\n# target line offset 2\n"
        "# target line offset 3\n# target line offset 4\n"
        "# target line offset 5\n"
        if side == "target"
        else ""
    )
    separators = "".join(f"# stable separator {index}\n" for index in range(1, 17))
    blocks = []
    for index in range(1, 4):
        value = "base" if side == "base" else side
        blocks.append(
            f"# BLOCK-{index}-BEGIN\n"
            f"def {stem}_decision_{index}(value: str) -> str:\n"
            f'    return f"{value}-{stem}-{index}-{{value}}"\n'
            f"# BLOCK-{index}-END\n"
        )
    return (
        f'"""Resolve retry fixture module: {stem}."""\n\n'
        "# SOURCE-PADDING-POINT\n"
        f"{source_padding}"
        + separators
        + "# TARGET-PADDING-POINT\n"
        + target_padding
        + ("\n" + separators).join(blocks)
    )


def _files(side: str) -> dict[str, str]:
    files = {
        "validation/__init__.py": "",
        f"{FIXTURE_ROOT}/__init__.py": "",
    }
    files.update({f"{FIXTURE_ROOT}/{name}": _module(name, side) for name in FILES})
    files[f"{FIXTURE_ROOT}/EXPECTED.md"] = """\
# Resolve retry E2E fixture

Resolve all twelve conflict blocks and preserve both sides' intent.

- Do not modify `source_feature.py`; it is the source-only non-conflicting change.
- Run `python3 -m compileall -q validation/resolve_retry_e2e`.
- The initial result must be a two-parent merge commit.
- A retry must be a normal one-parent child commit and may only change rejected blocks.
"""
    if side == "source":
        files[f"{FIXTURE_ROOT}/source_feature.py"] = (
            'def source_only_requirement() -> str:\n    return "must-survive-resolution"\n'
        )
    return files


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _commit(repo: Repo, message: str) -> str:
    repo.git.add("--all")
    return repo.index.commit(message).hexsha


def _compile(root: Path) -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", FIXTURE_ROOT],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout + completed.stderr).strip() or "compileall passed"


def _block_ranges(content: str) -> list[tuple[int, int]]:
    lines = content.splitlines()
    return [
        (
            lines.index(f"# BLOCK-{index}-BEGIN") + 2,
            lines.index(f"# BLOCK-{index}-END"),
        )
        for index in range(1, 4)
    ]


def _main() -> int:
    args = _args()
    source_name = f"verify/resolve-retry-{args.run_id}-test1"
    target_name = f"verify/resolve-retry-{args.run_id}-test2"
    local_repo = Repo(args.git_repo, search_parent_directories=True)
    remote_url = local_repo.remotes[args.git_remote].url
    result: dict = {
        "run_id": args.run_id,
        "source_branch": source_name,
        "target_branch": target_name,
    }
    with tempfile.TemporaryDirectory(prefix="conciliator-resolve-retry-fixture-") as checkout:
        repo = Repo.clone_from(remote_url, checkout, branch=args.base_branch)
        collisions = [
            name
            for name in (source_name, target_name)
            if cast(
                str,
                repo.git.ls_remote("--heads", "origin", f"refs/heads/{name}"),
            ).strip()
        ]
        if collisions:
            raise RuntimeError("fixture branches already exist: " + ", ".join(collisions))
        root = Path(checkout)
        repo.git.checkout("-b", target_name, f"origin/{args.base_branch}")
        _write(root, _files("base"))
        base_commit = _commit(repo, "test: add resolve retry fixture baseline")

        repo.git.checkout("-b", source_name, base_commit)
        _write(root, _files("source"))
        source_commit = _commit(repo, "test: add source retry requirements")
        source_test = _compile(root)

        repo.git.checkout(target_name)
        _write(root, _files("target"))
        target_commit = _commit(repo, "test: add target retry requirements")
        target_test = _compile(root)

        ranges = []
        for file_name in FILES:
            source_ranges = _block_ranges(_module(file_name, "source"))
            target_ranges = _block_ranges(_module(file_name, "target"))
            for source_range, target_range in zip(
                source_ranges,
                target_ranges,
                strict=True,
            ):
                if source_range == target_range:
                    raise RuntimeError(f"equal source/target ranges in {file_name}")
                ranges.append(
                    {
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
        repo.git.merge("--abort")
        if len(unmerged) != 4 or len(conflict_blocks) != 12:
            raise RuntimeError(
                "fixture shape mismatch: expected 4 files/12 blocks, got "
                f"{len(unmerged)} files/{len(conflict_blocks)} blocks"
            )
        for conflict_id, item in zip(
            (f"C{index:04d}" for index in range(1, 13)),
            ranges,
            strict=True,
        ):
            item["conflict_id"] = conflict_id

        if not args.dry_run:
            repo.git.push("--set-upstream", "origin", source_name)
            repo.git.push("--set-upstream", "origin", target_name)
        result.update(
            {
                "status": "fixture-validated" if args.dry_run else "branches-pushed",
                "base_commit": base_commit,
                "source_commit": source_commit,
                "target_commit": target_commit,
                "conflict_files": unmerged,
                "conflict_block_count": len(conflict_blocks),
                "physical_ranges": ranges,
                "tests": {"source": source_test, "target": target_test},
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
