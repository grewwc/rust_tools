import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.lark.drive import (
    MAX_MARKDOWN_UPLOAD_BYTES,
    upload_markdown,
)
from conciliator.utils.secrets import StaticSecretsProvider


def _settings() -> Settings:
    return Settings(
        env="dev",
        service_psm="test",
        mysql_psm="unused",
        lark_app_id="cli_test",
        kms_secret_lark_app_secret_alias="test/credentials/lark_app_secret",
    )


@pytest.mark.asyncio
@respx.mock
async def test_upload_markdown_uses_tenant_token_and_multipart_contract():
    token_route = respx.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "tenant_access_token": "tenant-token",
                "expire": 7200,
            },
        )
    )

    def upload_response(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer tenant-token"
        assert request.headers["content-type"].startswith("multipart/form-data;")
        body = request.content
        for expected in (
            b'name="file_name"',
            b"report.md",
            b'name="parent_type"',
            b"explorer",
            b'name="parent_node"',
            b"folder-token",
            b'name="size"',
            b"9",
            b'name="file"; filename="report.md"',
            b"# report\n",
        ):
            assert expected in body
        return httpx.Response(
            200,
            json={"code": 0, "data": {"file_token": "file-token"}},
        )

    upload_route = respx.post(
        "https://open.feishu.cn/open-apis/drive/v1/files/upload_all"
    ).mock(side_effect=upload_response)
    client = LarkClient(
        _settings(),
        StaticSecretsProvider(
            {"test/credentials/lark_app_secret": "app-secret"}
        ),
    )
    try:
        uploaded = await upload_markdown(
            client,
            file_name="report.md",
            content=b"# report\n",
            folder_token="folder-token",
        )
    finally:
        await client.aclose()

    assert token_route.call_count == 1
    assert upload_route.call_count == 1
    assert uploaded.file_token == "file-token"
    assert uploaded.file_name == "report.md"
    assert uploaded.size_bytes == 9
    assert uploaded.web_url == "https://bytedance.larkoffice.com/file/file-token"


@pytest.mark.asyncio
@respx.mock
async def test_tenant_token_is_cached():
    token_route = respx.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "tenant_access_token": "tenant-token",
                "expire": 7200,
            },
        )
    )
    client = LarkClient(
        _settings(),
        StaticSecretsProvider(
            {"test/credentials/lark_app_secret": "app-secret"}
        ),
    )
    try:
        assert await client.get_tenant_access_token() == "tenant-token"
        assert await client.get_tenant_access_token() == "tenant-token"
    finally:
        await client.aclose()

    assert token_route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_injected_tenant_token_skips_app_secret_and_token_exchange():
    def upload_response(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer injected-token"
        return httpx.Response(
            200,
            json={"code": 0, "data": {"file_token": "file-token"}},
        )

    upload_route = respx.post(
        "https://open.feishu.cn/open-apis/drive/v1/files/upload_all"
    ).mock(side_effect=upload_response)
    client = LarkClient(
        _settings(),
        None,
        tenant_access_token="injected-token",
    )
    try:
        uploaded = await upload_markdown(
            client,
            file_name="report.md",
            content=b"# report\n",
            folder_token="folder-token",
        )
    finally:
        await client.aclose()

    assert upload_route.call_count == 1
    assert uploaded.file_token == "file-token"


@pytest.mark.parametrize(
    ("file_name", "content", "folder_token", "message"),
    [
        ("report.txt", b"x", "folder", "end with .md"),
        ("report.md", b"", "folder", "must not be empty"),
        ("report.md", b"x", "", "folder_token"),
        (
            "report.md",
            b"x" * (MAX_MARKDOWN_UPLOAD_BYTES + 1),
            "folder",
            "20 MiB",
        ),
    ],
)
@pytest.mark.asyncio
async def test_upload_markdown_rejects_invalid_input(
    file_name,
    content,
    folder_token,
    message,
):
    client = LarkClient(
        _settings(),
        StaticSecretsProvider(
            {"test/credentials/lark_app_secret": "app-secret"}
        ),
    )
    try:
        with pytest.raises(ValueError, match=message):
            await upload_markdown(
                client,
                file_name=file_name,
                content=content,
                folder_token=folder_token,
            )
    finally:
        await client.aclose()
