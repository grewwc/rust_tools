from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient


@dataclass(frozen=True)
class MRVersion:
    id: str
    number: int
    source_commit_id: str
    target_commit_id: str
    base_commit_id: str


@dataclass(frozen=True)
class MR:
    id: str
    iid: int
    title: str
    description: str
    web_url: str
    source_repo_id: str | None
    target_repo_id: str
    source_branch: str | None
    target_branch: str
    author_id: str | None = None
    author_email: str | None = None
    author_username: str | None = None
    versions: tuple[MRVersion, ...] = ()
    source_head_sha: str | None = None
    target_head_sha: str | None = None

    @property
    def latest_version(self) -> MRVersion | None:
        return max(self.versions, key=lambda version: version.number) if self.versions else None


@dataclass(frozen=True)
class Mergeability:
    mergeable: bool
    reason: str | None
    detail: str | None


def _selector() -> dict[str, bool]:
    return {
        "Branch": True,
        "User": True,
        "Version": True,
        "URL": True,
        "Repository": True,
    }


async def get_mr(
    client: NextCodeClient,
    repo_id: str,
    mr_iid: int | None = None,
    *,
    mr_id: str | None = None,
) -> MR:
    body: dict = {"RepoId": repo_id, "Selector": _selector()}
    if mr_id is not None:
        body["Id"] = mr_id
    elif mr_iid is not None:
        body["Number"] = mr_iid
    else:
        raise ValueError("either mr_iid or mr_id is required")
    result = await client.call("GetMergeRequest", body, read_only=True)
    return _parse_mr(result["MergeRequest"])


async def create_mr(
    client: NextCodeClient,
    project_id: str,
    *,
    source: str,
    target: str,
    title: str,
    description: str,
) -> MR:
    result = await client.call(
        "CreateMergeRequest",
        {
            "SourceRepoId": project_id,
            "SourceBranch": source,
            "TargetRepoId": project_id,
            "TargetBranch": target,
            "Title": title,
            "Description": description,
            "SquashCommits": False,
            "RemoveSourceBranchAfterMerge": False,
        },
    )
    return _parse_mr(result["MergeRequest"])


async def update_mr_description(
    client: NextCodeClient,
    project_id: str,
    mr_iid: int,
    description: str,
) -> MR:
    result = await client.call(
        "UpdateMergeRequest",
        {"RepoId": project_id, "Number": mr_iid, "Description": description},
    )
    return _parse_mr(result["MergeRequest"])


async def get_mergeability(client: NextCodeClient, repo_id: str, mr_iid: int) -> Mergeability:
    result = await client.call(
        "GetMergeRequestMergeability",
        {"RepoId": repo_id, "Number": mr_iid},
        read_only=True,
    )
    data = result["Mergeability"]
    return Mergeability(
        mergeable=bool(data["Mergeable"]),
        reason=data.get("UnmergeableReason"),
        detail=data.get("UnmergeableReasonDetail"),
    )


async def list_repo_mrs(
    client: NextCodeClient,
    repo_id: str,
    *,
    source_branch: str | None = None,
    target_branch: str | None = None,
    status: str = "open",
    page_number: int = 1,
    page_size: int = 100,
) -> list[MR]:
    body: dict = {
        "TargetRepoId": repo_id,
        "Status": status,
        "Selector": _selector(),
        "PageNumber": page_number,
        "PageSize": page_size,
    }
    if source_branch is not None:
        body["SourceBranch"] = source_branch
    if target_branch is not None:
        body["TargetBranch"] = target_branch
    result = await client.call("ListRepoMergeRequests", body, read_only=True)
    return [_parse_mr(item) for item in result["MergeRequests"]]


async def add_mr_comment(
    client: NextCodeClient,
    project_id: str,
    mr_iid: int,
    body: str,
) -> None:
    from conciliator.integrations.nextcode.discussions import create_mr_comment

    await create_mr_comment(client, project_id, mr_iid, body)


def _parse_mr(data: dict) -> MR:
    author = data.get("CreatedBy") or {}
    versions = tuple(
        MRVersion(
            id=item["Id"],
            number=int(item["Number"]),
            source_commit_id=item["SourceCommitId"],
            target_commit_id=item["TargetCommitId"],
            base_commit_id=item["BaseCommitId"],
        )
        for item in data.get("Versions") or []
    )
    return MR(
        id=data["Id"],
        iid=int(data["Number"]),
        title=data["Title"],
        description=data.get("Description", ""),
        web_url=data.get("URL", ""),
        source_repo_id=data.get("SourceRepoId"),
        target_repo_id=data["TargetRepoId"],
        source_branch=data.get("SourceBranchName"),
        target_branch=data["TargetBranchName"],
        author_id=author.get("Id"),
        author_email=author.get("Email"),
        author_username=author.get("Username"),
        versions=versions,
        source_head_sha=((data.get("SourceBranch") or {}).get("Commit") or {}).get("Id"),
        target_head_sha=((data.get("TargetBranch") or {}).get("Commit") or {}).get("Id"),
    )
