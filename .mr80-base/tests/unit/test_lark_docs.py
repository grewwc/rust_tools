import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.lark.docs import (
    create_doc,
    extract_document_ref,
    read_document_text,
)
from conciliator.utils.secrets import StaticSecretsProvider


def _settings() -> Settings:
    return Settings(
        env="dev", service_psm="x",
        kms_secret_nextcode_app_secret_alias="test/credentials/nextcode_app_secret",
        kms_secret_nextcode_webhook_secret_alias=(
            "test/credentials/nextcode_webhook_secret"
        ),
        kms_secret_lark_app_secret_alias="test/credentials/lark_app_secret",
        nextcode_app_id="x",
        coco_api_base="x", coco_workspace_id="x", coco_default_agent_id="x",
        lark_app_id="li", mysql_psm="x",
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
@respx.mock
async def test_create_doc_returns_token_and_url():
    respx.post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal").mock(
        return_value=httpx.Response(200, json={
            "code": 0, "tenant_access_token": "tok", "expire": 7200,
        })
    )
    respx.post("https://open.feishu.cn/open-apis/docx/v1/documents").mock(
        return_value=httpx.Response(200, json={
            "code": 0,
            "data": {"document": {"document_id": "doccnABC"}},
        })
    )
    client = LarkClient(
        _settings(),
        StaticSecretsProvider({"test/credentials/lark_app_secret": "ls"}),
    )
    ref = await create_doc(
        client,
        title="测试文档",
        content_xml="<p>hi</p>",
        folder_token="fld1",
    )
    assert ref.doc_token == "doccnABC"
    assert "doccnABC" in ref.url
    await client.aclose()


@pytest.mark.parametrize(
    ("url", "kind", "token"),
    [
        ("https://example.feishu.cn/docx/doccnABC?from=share", "docx", "doccnABC"),
        ("https://example.feishu.cn/wiki/wikcnABC", "wiki", "wikcnABC"),
    ],
)
def test_extract_document_ref(url, kind, token):
    assert extract_document_ref(url).kind == kind
    assert extract_document_ref(url).token == token


def test_extract_document_ref_rejects_unsupported_links():
    with pytest.raises(ValueError, match="docx or wiki"):
        extract_document_ref("https://example.feishu.cn/sheets/abc")


@pytest.mark.asyncio
@respx.mock
async def test_read_docx_raw_content():
    route = respx.get(
        "https://open.feishu.cn/open-apis/docx/v1/documents/doccnABC/raw_content"
    ).mock(
        return_value=httpx.Response(
            200, json={"code": 0, "data": {"content": "需求正文"}}
        )
    )
    client = LarkClient(_settings(), None, tenant_access_token="tenant-token")
    try:
        assert (
            await read_document_text(
                client, "https://example.feishu.cn/docx/doccnABC"
            )
            == "需求正文"
        )
    finally:
        await client.aclose()
    assert route.calls[0].request.headers["authorization"] == "Bearer tenant-token"


@pytest.mark.asyncio
@respx.mock
async def test_read_wiki_resolves_docx_object_first():
    node_route = respx.get(
        "https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node",
        params={"token": "wikcnABC"},
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "node": {"obj_type": "docx", "obj_token": "doccnResolved"}
                },
            },
        )
    )
    respx.get(
        "https://open.feishu.cn/open-apis/docx/v1/documents/doccnResolved/raw_content"
    ).mock(
        return_value=httpx.Response(
            200, json={"code": 0, "data": {"content": "Wiki 正文"}}
        )
    )
    client = LarkClient(_settings(), None, tenant_access_token="tenant-token")
    try:
        assert (
            await read_document_text(
                client, "https://example.feishu.cn/wiki/wikcnABC"
            )
            == "Wiki 正文"
        )
    finally:
        await client.aclose()
    assert node_route.call_count == 1
