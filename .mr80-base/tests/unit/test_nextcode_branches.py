import json

import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.nextcode.branches import (
    CommitFileAction,
    create_branch,
    create_commit,
    ensure_branch,
    get_branch,
)
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.utils.secrets import StaticSecretsProvider


class StaticZtiProvider:
    def get_token(self) -> str:
        return "zti"


def _client() -> NextCodeClient:
    settings = Settings(
        env="dev",
        service_psm="x",
        kms_secret_nextcode_app_secret_alias=("test/credentials/nextcode_app_secret"),
        kms_secret_nextcode_webhook_secret_alias=("test/credentials/nextcode_webhook_secret"),
        kms_secret_lark_app_secret_alias="test/credentials/lark_app_secret",
        nextcode_app_id="app1",
        coco_api_base="x",
        coco_workspace_id="x",
        coco_default_agent_id="x",
        lark_app_id="x",
        mysql_psm="x",
        nextcode_api_base="https://nc.test/v2/",
    )  # type: ignore[arg-type]
    return NextCodeClient(
        settings,
        StaticSecretsProvider(
            {
                "test/credentials/nextcode_app_secret": "s",
                "test/credentials/nextcode_webhook_secret": "w",
            }
        ),
        zti_provider=StaticZtiProvider(),
    )


@pytest.mark.asyncio
@respx.mock
async def test_create_branch_exact_contract():
    route = respx.post("https://nc.test/v2/?Action=CreateBranch").mock(
        return_value=httpx.Response(
            200,
            json={
                "ResponseMetadata": {},
                "Result": {"Branch": {"Name": "resolve/feat/x", "Commit": {"Id": "abc123"}}},
            },
        )
    )
    client = _client()
    branch = await create_branch(client, "r1", "feat", "resolve/feat/x")
    assert branch.name == "resolve/feat/x"
    assert branch.commit_sha == "abc123"
    assert json.loads(route.calls.last.request.content) == {
        "RepoId": "r1",
        "Name": "resolve/feat/x",
        "Revision": "feat",
    }
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_create_commit_exact_contract():
    route = respx.post("https://nc.test/v2/?Action=CreateCommit").mock(
        return_value=httpx.Response(
            200,
            json={
                "ResponseMetadata": {},
                "Result": {"Commit": {"Id": "c1", "Title": "test commit"}},
            },
        )
    )
    client = _client()
    commit = await create_commit(
        client,
        "r1",
        "test/one",
        "test commit",
        [CommitFileAction("create", "contract.txt", "first\n")],
    )
    assert commit.id == "c1"
    assert json.loads(route.calls.last.request.content)["CommitFilesActions"] == [
        {
            "ActionType": "create",
            "Path": "contract.txt",
            "Encoding": "text",
            "Content": "first\n",
        }
    ]
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_get_branch_and_ensure_existing_branch_exact_contract():
    route = respx.post("https://nc.test/v2/?Action=GetBranch").mock(
        return_value=httpx.Response(
            200,
            json={
                "ResponseMetadata": {},
                "Result": {
                    "Branch": {
                        "Name": "resolve/mr-10/task-123",
                        "Commit": {"Id": "a" * 40},
                    }
                },
            },
        )
    )
    client = _client()

    branch = await get_branch(client, "r1", "resolve/mr-10/task-123")
    ensured = await ensure_branch(
        client,
        "r1",
        revision="a" * 40,
        name="resolve/mr-10/task-123",
    )

    assert branch == ensured
    assert len(route.calls) == 2
    assert json.loads(route.calls[0].request.content) == {
        "RepoId": "r1",
        "Name": "resolve/mr-10/task-123",
    }
    await client.aclose()
