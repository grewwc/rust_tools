import base64

import pytest

from conciliator.integrations.nextcode.commits import get_commit
from conciliator.integrations.nextcode.files import get_file


class FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, dict, bool]] = []

    async def call(self, action: str, body: dict, *, read_only: bool = False):
        self.calls.append((action, body, read_only))
        if action == "GetCommit":
            return {
                "Commit": {
                    "Id": "c" * 40,
                    "Title": "merge target",
                    "Parents": ["a" * 40, "b" * 40],
                }
            }
        if action == "GetFile":
            return {
                "File": {
                    "Path": "src/example.py",
                    "Encoding": "base64",
                    "Content": base64.b64encode(b"resolved\n").decode(),
                }
            }
        raise AssertionError(action)


@pytest.mark.asyncio
async def test_commit_and_file_readback_contracts():
    client = FakeClient()

    commit = await get_commit(client, "repo-1", "c" * 40)
    file = await get_file(
        client,
        "repo-1",
        revision="c" * 40,
        path="src/example.py",
    )

    assert commit.parents == ("a" * 40, "b" * 40)
    assert file is not None
    assert file.content == "resolved\n"
    assert client.calls == [
        (
            "GetCommit",
            {"RepoId": "repo-1", "Revision": "c" * 40},
            True,
        ),
        (
            "GetFile",
            {
                "RepoId": "repo-1",
                "Revision": "c" * 40,
                "Path": "src/example.py",
            },
            True,
        ),
    ]
