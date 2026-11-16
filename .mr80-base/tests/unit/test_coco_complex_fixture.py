import subprocess
from contextlib import suppress
from pathlib import Path

from git import GitCommandError, Repo

from conciliator.integrations.git.conflicts import parse_conflict_file
from scripts.create_coco_complex_fixture import (
    BASE_FORMATTER,
    BASE_POLICY,
    RESOLVED_FORMATTER,
    RESOLVED_POLICY,
    SOURCE_FORMATTER,
    SOURCE_POLICY,
    SOURCE_TEST,
    TARGET_FORMATTER,
    TARGET_POLICY,
    TARGET_TEST,
)


def test_complex_fixture_produces_four_blocks_and_expected_resolution_passes(tmp_path: Path):
    repo = Repo.init(tmp_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Fixture Test")
        config.set_value("user", "email", "fixture@example.com")

    root = tmp_path / "validation" / "coco_complex_conflict"
    tests = root / "tests"
    tests.mkdir(parents=True)
    _write(tmp_path / "validation" / "__init__.py", "")
    _write(root / "__init__.py", "")
    _write(tests / "__init__.py", "")
    _write(root / "policy.py", BASE_POLICY)
    _write(root / "formatter.py", BASE_FORMATTER)
    repo.git.add(A=True)
    repo.index.commit("base")
    base_sha = repo.head.commit.hexsha

    repo.git.checkout("-b", "source")
    _write(root / "policy.py", SOURCE_POLICY)
    _write(root / "formatter.py", SOURCE_FORMATTER)
    _write(tests / "test_source_requirements.py", SOURCE_TEST)
    repo.git.add(A=True)
    repo.index.commit("source")

    repo.git.checkout("-b", "target", base_sha)
    _write(root / "policy.py", TARGET_POLICY)
    _write(root / "formatter.py", TARGET_FORMATTER)
    _write(tests / "test_target_requirements.py", TARGET_TEST)
    repo.git.add(A=True)
    repo.index.commit("target")

    repo.git.checkout("source")
    with suppress(GitCommandError):
        repo.git.merge("target", no_edit=True)

    parsed = [
        *parse_conflict_file(str(root / "formatter.py"), (root / "formatter.py").read_text()),
        *parse_conflict_file(str(root / "policy.py"), (root / "policy.py").read_text()),
    ]
    assert len(parsed) == 4

    _write(root / "policy.py", RESOLVED_POLICY)
    _write(root / "formatter.py", RESOLVED_FORMATTER)
    repo.git.add(A=True)
    repo.index.commit("resolve")
    completed = subprocess.run(
        [
            "python3",
            "-m",
            "unittest",
            "discover",
            "-s",
            "validation/coco_complex_conflict/tests",
            "-v",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
