from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient


@dataclass(frozen=True)
class CommitDetails:
    id: str
    title: str
    parents: tuple[str, ...]


async def get_commit(
    client: NextCodeClient,
    repo_id: str,
    revision: str,
) -> CommitDetails:
    result = await client.call(
        "GetCommit",
        {"RepoId": repo_id, "Revision": revision},
        read_only=True,
    )
    data = result["Commit"]
    return CommitDetails(
        id=data["Id"],
        title=data.get("Title", ""),
        parents=tuple(data.get("Parents") or ()),
    )
