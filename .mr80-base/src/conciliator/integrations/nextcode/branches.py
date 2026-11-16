from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient, NextCodeError
from conciliator.utils.retry import TransientError


@dataclass(frozen=True)
class Branch:
    name: str
    commit_sha: str | None


async def get_branch(
    client: NextCodeClient,
    repo_id: str,
    name: str,
) -> Branch | None:
    try:
        result = await client.call(
            "GetBranch",
            {"RepoId": repo_id, "Name": name},
            read_only=True,
        )
    except NextCodeError as exc:
        if exc.code.startswith("NotFound"):
            return None
        raise
    data = result.get("Branch")
    if not isinstance(data, dict):
        return None
    return _parse_branch(data)


@dataclass(frozen=True)
class Commit:
    id: str
    title: str


@dataclass(frozen=True)
class CommitFileAction:
    action_type: str
    path: str
    content: str | None = None
    encoding: str = "text"
    previous_path: str | None = None


async def create_branch(
    client: NextCodeClient,
    repo_id: str,
    from_branch: str,
    new_branch: str,
) -> Branch:
    result = await client.call(
        "CreateBranch",
        {"RepoId": repo_id, "Name": new_branch, "Revision": from_branch},
    )
    return _parse_branch(result["Branch"])


async def ensure_branch(
    client: NextCodeClient,
    repo_id: str,
    *,
    revision: str,
    name: str,
) -> Branch:
    """Create a stable branch or recover a prior ambiguous create.

    ``CreateBranch`` has no documented idempotency key.  A deterministic name
    plus ``GetBranch`` readback makes worker retries safe without moving an
    existing branch.
    """
    existing = await get_branch(client, repo_id, name)
    if existing is not None:
        return _require_branch_head(existing, revision)

    try:
        created = await create_branch(client, repo_id, revision, name)
    except (NextCodeError, TransientError):
        existing = await get_branch(client, repo_id, name)
        if existing is None:
            raise
        return _require_branch_head(existing, revision)
    return _require_branch_head(created, revision)


async def create_commit(
    client: NextCodeClient,
    repo_id: str,
    branch: str,
    message: str,
    actions: list[CommitFileAction],
    *,
    start_branch: str | None = None,
) -> Commit:
    body: dict = {
        "RepoId": repo_id,
        "Branch": branch,
        "CommitMessage": message,
        "CommitFilesActions": [],
    }
    if start_branch is not None:
        body["StartBranch"] = start_branch
    for action in actions:
        item = {
            "ActionType": action.action_type,
            "Path": action.path,
            "Encoding": action.encoding,
        }
        if action.content is not None:
            item["Content"] = action.content
        if action.previous_path is not None:
            item["PreviousPath"] = action.previous_path
        body["CommitFilesActions"].append(item)

    result = await client.call("CreateCommit", body)
    commit = result["Commit"]
    return Commit(id=commit["Id"], title=commit["Title"])


def _parse_branch(data: dict) -> Branch:
    commit = data.get("Commit") or {}
    return Branch(name=data["Name"], commit_sha=commit.get("Id"))


def _require_branch_head(branch: Branch, expected_sha: str) -> Branch:
    if branch.commit_sha != expected_sha:
        raise RuntimeError(
            f"branch {branch.name} head mismatch: expected {expected_sha}, got {branch.commit_sha or '<missing>'}"
        )
    return branch
