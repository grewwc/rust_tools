from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.mrs import MRVersion, get_mr


@dataclass(frozen=True)
class Discussion:
    id: str
    comment_id: str


@dataclass(frozen=True)
class Thread:
    id: str
    status: str
    positions: tuple[dict, ...]
    comments: tuple[dict, ...] = ()


async def create_mr_comment(
    client: NextCodeClient,
    repo_id: str,
    mr_iid: int,
    content: str,
) -> Discussion:
    mr = await get_mr(client, repo_id, mr_iid)
    return await _create_comment(
        client,
        {
            "CommentableType": "merge_request",
            "CommentableId": mr.id,
            "RepoId": repo_id,
            "Content": content,
        },
    )


async def create_discussion(
    client: NextCodeClient,
    project_id: str,
    mr_iid: int,
    *,
    file_path: str,
    line: int,
    end_line: int | None = None,
    body: str,
) -> Discussion:
    mr = await get_mr(client, project_id, mr_iid)
    version = mr.latest_version
    if version is None:
        raise ValueError("merge request has no version for a diff position")
    return await _create_comment(
        client,
        {
            "CommentableType": "merge_request",
            "CommentableId": mr.id,
            "RepoId": project_id,
            "Content": body,
            "Position": {
                "Type": "content",
                "Side": "new",
                "OldCommitId": version.base_commit_id or version.target_commit_id,
                "NewCommitId": version.source_commit_id,
                "Path": file_path,
                "StartLine": line,
                "StartColumn": 1,
                "EndLine": end_line or line,
                "EndColumn": 1000,
            },
        },
    )


async def reply_discussion(
    client: NextCodeClient,
    project_id: str,
    mr_iid: int,
    discussion_id: str,
    body: str,
) -> Discussion:
    mr = await get_mr(client, project_id, mr_iid)
    return await _create_comment(
        client,
        {
            "CommentableType": "merge_request",
            "CommentableId": mr.id,
            "RepoId": project_id,
            "ThreadId": discussion_id,
            "Content": body,
        },
    )


async def get_thread(client: NextCodeClient, repo_id: str, thread_id: str) -> Thread:
    result = await client.call("GetThread", {"Id": thread_id, "RepoId": repo_id}, read_only=True)
    thread = result["Thread"]
    return Thread(
        id=thread["Id"],
        status=thread["Status"],
        positions=tuple(thread.get("Positions") or []),
        comments=tuple(thread.get("Comments") or []),
    )


def find_position_for_version(
    thread: Thread,
    version: MRVersion,
) -> dict | None:
    """Return the position belonging to the current MR source revision."""
    candidates = [
        position
        for position in thread.positions
        if position.get("NewCommitId") == version.source_commit_id
        and position.get("OldCommitId") in {version.base_commit_id, version.target_commit_id}
    ]
    return candidates[-1] if candidates else None


async def _create_comment(client: NextCodeClient, request: dict) -> Discussion:
    result = await client.call("CreateComment", request)
    comment = result["Comment"]
    return Discussion(id=comment["ThreadId"], comment_id=comment["Id"])
