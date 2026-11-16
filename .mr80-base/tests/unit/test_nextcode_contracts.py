import base64

import pytest

from conciliator.integrations.nextcode.branches import create_branch
from conciliator.integrations.nextcode.discussions import (
    create_discussion,
    create_mr_comment,
    find_position_for_version,
    get_thread,
    reply_discussion,
)
from conciliator.integrations.nextcode.merges import ConflictSnapshotMismatchError, list_conflict_files
from conciliator.integrations.nextcode.mrs import (
    create_mr,
    get_mergeability,
    get_mr,
    list_repo_mrs,
    update_mr_description,
)
from conciliator.integrations.nextcode.repos import get_blame, get_repository


def _mr(number: int = 42) -> dict:
    return {
        "Id": "mr-internal-id",
        "Number": number,
        "TargetRepoId": "repo-1",
        "SourceRepoId": "repo-1",
        "SourceBranchName": "test/source",
        "TargetBranchName": "test/target",
        "SourceBranch": {"Name": "test/source", "Commit": {"Id": "source-head"}},
        "TargetBranch": {"Name": "test/target", "Commit": {"Id": "target-head"}},
        "Title": "contract MR",
        "Description": "description",
        "URL": f"https://code.test/repo/merge_requests/{number}",
        "CreatedBy": {
            "Id": "user-1",
            "Username": "alice",
            "Email": "alice@example.com",
        },
        "Versions": [
            {
                "Id": "version-1",
                "Number": 1,
                "SourceCommitId": "source-sha",
                "TargetCommitId": "target-sha",
                "BaseCommitId": "base-sha",
            }
        ],
    }


class FakeClient:
    def __init__(self, responses: dict[str, list[dict] | dict]):
        self.responses = responses
        self.calls: list[tuple[str, dict, bool]] = []

    async def call(self, action: str, body: dict, *, read_only: bool = False) -> dict:
        self.calls.append((action, body, read_only))
        response = self.responses[action]
        if isinstance(response, list):
            return response.pop(0)
        return response


@pytest.mark.asyncio
async def test_repository_and_blame_contracts():
    client = FakeClient(
        {
            "GetRepository": {
                "Repository": {
                    "Id": "repo-1",
                    "Path": "xiahaitao/Conciliator",
                    "Name": "Conciliator",
                    "URL": "https://code.test/repo",
                    "CloneURL": "https://code.test/repo.git",
                    "SSHURL": "git@code.test:repo.git",
                    "DefaultBranch": "main",
                    "Creator": {"Email": "owner@example.com"},
                }
            },
            "GetBlame": {
                "Segments": [
                    {
                        "StartLine": 1,
                        "EndLine": 3,
                        "Commit": {
                            "Id": "commit-1",
                            "Author": {"Name": "Alice", "Email": "alice@example.com"},
                        },
                    }
                ]
            },
        }
    )
    repo = await get_repository(client, path="xiahaitao/Conciliator")
    blame = await get_blame(
        client,
        repo.id,
        revision="main",
        path="README.md",
        start_line=1,
        end_line=3,
    )
    assert repo.default_branch == "main"
    assert blame[0].author_email == "alice@example.com"
    assert client.calls == [
        ("GetRepository", {"Path": "xiahaitao/Conciliator"}, True),
        (
            "GetBlame",
            {
                "RepoId": "repo-1",
                "Revision": "main",
                "Path": "README.md",
                "StartLine": 1,
                "EndLine": 3,
            },
            True,
        ),
    ]


@pytest.mark.asyncio
async def test_merge_request_contracts():
    client = FakeClient(
        {
            "GetMergeRequest": {"MergeRequest": _mr()},
            "CreateMergeRequest": {"MergeRequest": _mr(43)},
            "UpdateMergeRequest": {"MergeRequest": _mr(43)},
            "GetMergeRequestMergeability": {
                "Mergeability": {
                    "Mergeable": False,
                    "UnmergeableReason": "conflict",
                    "UnmergeableReasonDetail": "content conflict",
                }
            },
            "ListRepoMergeRequests": {"MergeRequests": [_mr(43)]},
        }
    )
    mr = await get_mr(client, "repo-1", 42)
    created = await create_mr(
        client,
        "repo-1",
        source="test/source",
        target="test/target",
        title="contract MR",
        description="description",
    )
    await update_mr_description(client, "repo-1", 43, "updated")
    mergeability = await get_mergeability(client, "repo-1", 43)
    listed = await list_repo_mrs(client, "repo-1", target_branch="test/target", status="open")

    assert mr.author_email == "alice@example.com"
    assert mr.latest_version is not None
    assert mr.latest_version.base_commit_id == "base-sha"
    assert (mr.source_head_sha, mr.target_head_sha) == ("source-head", "target-head")
    assert mr.latest_version.source_commit_id == "source-sha"
    assert mr.latest_version.target_commit_id == "target-sha"
    assert client.calls[0][1]["Selector"]["Branch"] is True
    assert created.iid == 43
    assert mergeability.reason == "conflict"
    assert listed[0].iid == 43
    assert client.calls[1][1] == {
        "SourceRepoId": "repo-1",
        "SourceBranch": "test/source",
        "TargetRepoId": "repo-1",
        "TargetBranch": "test/target",
        "Title": "contract MR",
        "Description": "description",
        "SquashCommits": False,
        "RemoveSourceBranchAfterMerge": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", [None, {}, {"Commit": None}, {"Commit": {}}])
async def test_missing_mr_branch_heads_do_not_fall_back_to_version_shas(branch):
    data = {**_mr(), "SourceBranch": branch, "TargetBranch": branch}
    parsed = await get_mr(FakeClient({"GetMergeRequest": {"MergeRequest": data}}), "repo-1", 42)
    assert parsed.source_head_sha is None
    assert parsed.target_head_sha is None
    assert parsed.latest_version.source_commit_id == "source-sha"


@pytest.mark.asyncio
async def test_conflict_content_is_thrift_binary_base64():
    content = "before\n<<<<<<< file.txt\nours\n=======\ntheirs\n>>>>>>> file.txt\nafter\n"
    encoded = base64.b64encode(content.encode()).decode()
    client = FakeClient(
        {
            "ListMergeRequestConflicts": {
                "ConflictsId": "conflicts-1",
                "Conflicts": [
                    {
                        "ConflictHeader": {
                            "Base": {"Path": "file.txt", "Mode": 33188},
                            "First": {"Path": "file.txt", "Mode": 33188},
                            "Second": {"Path": "file.txt", "Mode": 33188},
                        },
                        "Content": encoded,
                    }
                ],
            }
        }
    )
    conflicts = await list_conflict_files(client, "repo-1", 42)
    assert conflicts[0].content == content
    assert client.calls[0] == (
        "ListMergeRequestConflicts",
        {"RepoId": "repo-1", "Number": 42, "WithContent": True},
        True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("has_conflicts", [False, True])
@pytest.mark.parametrize("snapshot_id", ["source..target", "source..old-target", "old-source..target", "unknown"])
async def test_conflict_snapshot_is_bound_to_captured_commits(has_conflicts, snapshot_id):
    client = FakeClient({"ListMergeRequestConflicts": {
        "ConflictsId": snapshot_id,
        "Conflicts": [{
            "ConflictHeader": {"First": {"Path": "a.py"}, "Second": {"Path": "a.py"}},
            "Content": base64.b64encode(b"conflict content").decode(),
        }] if has_conflicts else [],
    }})
    if snapshot_id == "source..target":
        files = await list_conflict_files(client, "repo-1", 42, expected_conflicts_id="source..target")
        assert len(files) == int(has_conflicts)
        if files:
            assert files[0].conflicts_id == snapshot_id
    else:
        with pytest.raises(ConflictSnapshotMismatchError):
            await list_conflict_files(client, "repo-1", 42, expected_conflicts_id="source..target")
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_comment_and_inline_thread_contracts():
    client = FakeClient(
        {
            "GetMergeRequest": [
                {"MergeRequest": _mr()},
                {"MergeRequest": _mr()},
                {"MergeRequest": _mr()},
                {"MergeRequest": _mr()},
            ],
            "CreateComment": [
                {"Comment": {"Id": "comment-1", "ThreadId": "thread-1"}},
                {"Comment": {"Id": "comment-2", "ThreadId": "thread-2"}},
                {"Comment": {"Id": "comment-3", "ThreadId": "thread-2"}},
            ],
            "GetThread": {
                "Thread": {
                    "Id": "thread-2",
                    "Status": "open",
                    "Positions": [
                        {
                            "OldCommitId": "base-sha",
                            "NewCommitId": "source-sha",
                            "Path": "file.txt",
                            "StartLine": 7,
                            "EndLine": 9,
                        }
                    ],
                    "Comments": [
                        {
                            "Id": "comment-3",
                            "Content": "<!-- conciliator:task=7:block=C0001:attempt=2 -->",
                        }
                    ],
                }
            },
        }
    )
    main = await create_mr_comment(client, "repo-1", 42, "main comment")
    inline = await create_discussion(
        client,
        "repo-1",
        42,
        file_path="file.txt",
        line=7,
        end_line=9,
        body="inline comment",
    )
    reply = await reply_discussion(client, "repo-1", 42, inline.id, "/approval")
    thread = await get_thread(client, "repo-1", inline.id)

    assert main.comment_id == "comment-1"
    assert reply.comment_id == "comment-3"
    assert thread.positions[0]["StartLine"] == 7
    assert "attempt=2" in thread.comments[0]["Content"]
    version = (await get_mr(client, "repo-1", 42)).latest_version
    assert version is not None
    assert find_position_for_version(thread, version) == thread.positions[0]
    create_calls = [call for call in client.calls if call[0] == "CreateComment"]
    assert create_calls[1][1]["Position"] == {
        "Type": "content",
        "Side": "new",
        "OldCommitId": "base-sha",
        "NewCommitId": "source-sha",
        "Path": "file.txt",
        "StartLine": 7,
        "StartColumn": 1,
        "EndLine": 9,
        "EndColumn": 1000,
    }
    assert create_calls[2][1]["ThreadId"] == "thread-2"


@pytest.mark.asyncio
async def test_create_branch_action_name_is_not_rest_path():
    client = FakeClient({"CreateBranch": {"Branch": {"Name": "test/source", "Commit": {"Id": "sha"}}}})
    await create_branch(client, "repo-1", "main", "test/source")
    assert client.calls == [
        (
            "CreateBranch",
            {"RepoId": "repo-1", "Name": "test/source", "Revision": "main"},
            False,
        )
    ]
