import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import git

_MAILMAP_RE = re.compile(
    r"^(?P<name>[^<]+)\s+<(?P<email>[^>]+)>\s*(?P<nname>[^<]*?)\s+<(?P<nemail>[^>]+)>"
)


@dataclass(frozen=True)
class BlameResult:
    author_email: str
    author_name: str
    commit_sha: str
    committed_at: str  # ISO 8601


def load_mailmap(repo_path: Path) -> dict[str, tuple[str, str]]:
    """读 .mailmap,返回 {(orig_email): (new_name, new_email)}。"""
    mailmap_file = repo_path / ".mailmap"
    if not mailmap_file.exists():
        return {}

    mapping: dict[str, tuple[str, str]] = {}
    for line in mailmap_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _MAILMAP_RE.match(line)
        if m:
            # mailmap 语义:Proper Name <proper@email> Commit Name <commit@email>
            # 意为 commit@email 在 blame 中应显示为 proper name/email
            # nname/nemail = commit 侧(作 key),name/email = proper 侧(作 value)
            commit_email = m.group("nemail").strip().lower()
            proper_name = m.group("nname").strip() or m.group("name").strip()
            proper_email = m.group("email").strip()
            mapping[commit_email] = (proper_name, proper_email)
    return mapping


def blame_lines(
    repo_path: Path,
    file_path: str,
    start_line: int,
    end_line: int,
) -> BlameResult | None:
    """对 [start_line, end_line] 区间(1-based)做 blame,取最后修改者。

    策略:取该区间内出现次数最多的 commit 作为代表,再用其 author。
    简化版:取区间最后一个 commit(= ours 侧最近改动)。
    """
    repo = git.Repo(str(repo_path))
    full_path = repo_path / file_path
    if not full_path.exists():
        return None

    try:
        # GitPython 的 blame 类型 stub 不准,实际返回 list[tuple[Commit, list[str]]]
        blame: list[tuple[Any, list[str]]] = repo.blame("HEAD", file_path)  # type: ignore[assignment]
    except Exception:
        return None

    # blame 返回 [(commit, [lines]), ...]
    # 需要把 1-based 行号映射到对应 commit
    current_line = 1
    chosen_commit: Any = None
    for commit, lines in blame:
        line_count = len(lines)
        if start_line <= current_line + line_count - 1 and current_line <= end_line:
            chosen_commit = commit  # 取区间内最后出现的 commit
        current_line += line_count
        if current_line > end_line:
            break

    if chosen_commit is None:
        return None

    mailmap = load_mailmap(repo_path)
    email = chosen_commit.author.email or ""
    name = chosen_commit.author.name or ""
    if email.lower() in mailmap:
        name, email = mailmap[email.lower()]

    return BlameResult(
        author_email=email,
        author_name=name,
        commit_sha=chosen_commit.hexsha,
        committed_at=chosen_commit.committed_datetime.isoformat(),
    )
