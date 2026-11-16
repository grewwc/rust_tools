from pathlib import Path

import git
import pytest

from conciliator.integrations.git.blame import blame_lines, load_mailmap


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "alice@example.com")
        cw.set_value("user", "name", "Alice")

    (repo_dir / "a.py").write_text("def foo():\n    return 1\n")
    repo.index.add(["a.py"])
    repo.index.commit("first commit")
    return repo_dir


def test_blame_returns_author(sample_repo: Path):
    result = blame_lines(sample_repo, "a.py", start_line=1, end_line=2)
    assert result is not None
    assert result.author_email == "alice@example.com"
    assert result.author_name == "Alice"


def test_blame_missing_file_returns_none(sample_repo: Path):
    assert blame_lines(sample_repo, "missing.py", 1, 1) is None


def test_load_mailmap(tmp_path: Path):
    (tmp_path / ".mailmap").write_text(
        "Alice <new@example.com> Alice <old@example.com>\n"
    )
    mapping = load_mailmap(tmp_path)
    assert mapping["old@example.com"] == ("Alice", "new@example.com")
