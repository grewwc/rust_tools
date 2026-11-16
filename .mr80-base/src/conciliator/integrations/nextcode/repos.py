from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient


@dataclass(frozen=True)
class Repository:
    id: str
    path: str
    name: str
    url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    creator_email: str | None


@dataclass(frozen=True)
class BlameSegment:
    start_line: int
    end_line: int
    commit_id: str
    author_name: str
    author_email: str


class RepositoryMembersOpenAPIUnavailable(NotImplementedError):
    pass


async def get_repository(
    client: NextCodeClient,
    *,
    repo_id: str | None = None,
    path: str | None = None,
) -> Repository:
    if repo_id is None and path is None:
        raise ValueError("either repo_id or path is required")
    body = {"Id": repo_id} if repo_id is not None else {"Path": path}
    result = await client.call("GetRepository", body, read_only=True)
    repo = result["Repository"]
    creator = repo.get("Creator") or {}
    return Repository(
        id=repo["Id"],
        path=repo["Path"],
        name=repo["Name"],
        url=repo["URL"],
        clone_url=repo["CloneURL"],
        ssh_url=repo["SSHURL"],
        default_branch=repo["DefaultBranch"],
        creator_email=creator.get("Email"),
    )


async def get_blame(
    client: NextCodeClient,
    repo_id: str,
    *,
    revision: str,
    path: str,
    start_line: int,
    end_line: int,
) -> list[BlameSegment]:
    result = await client.call(
        "GetBlame",
        {
            "RepoId": repo_id,
            "Revision": revision,
            "Path": path,
            "StartLine": start_line,
            "EndLine": end_line,
        },
        read_only=True,
    )
    segments: list[BlameSegment] = []
    for item in result["Segments"]:
        commit = item["Commit"]
        author = commit["Author"]
        segments.append(
            BlameSegment(
                start_line=int(item["StartLine"]),
                end_line=int(item["EndLine"]),
                commit_id=commit["Id"],
                author_name=author["Name"],
                author_email=author["Email"],
            )
        )
    return segments


async def check_repo_member(
    client: NextCodeClient,
    project_id: str,
    email: str,
    *,
    role: str = "admin",
) -> bool:
    del client, project_id, email, role
    raise RepositoryMembersOpenAPIUnavailable(
        "The current Codebase App OpenAPI service does not expose ListRepoMembers; "
        "repository owner/master lookup cannot be implemented with granted App scopes."
    )
