import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.lark.contacts import (
    LarkUserNotFoundError,
    get_user_email_by_open_id,
    get_user_open_id_by_email,
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
async def test_get_user_email_by_open_id_passes_params():
    respx.post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal").mock(
        return_value=httpx.Response(200, json={
            "code": 0, "tenant_access_token": "tok", "expire": 7200,
        })
    )
    route = respx.get("https://open.feishu.cn/open-apis/contact/v3/users/ou_abc").mock(
        return_value=httpx.Response(200, json={
            "code": 0,
            "data": {"user": {"email": "alice@example.com"}},
        })
    )
    client = LarkClient(
        _settings(),
        StaticSecretsProvider({"test/credentials/lark_app_secret": "ls"}),
    )
    email = await get_user_email_by_open_id(client, "ou_abc")
    assert email == "alice@example.com"
    # 验证 params 正确传入
    assert route.calls.last.request.url.params["user_id_type"] == "open_id"
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_get_user_open_id_by_email_uses_batch_contract():
    route = respx.post(
        "https://open.feishu.cn/open-apis/contact/v3/users/batch_get_id"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "user_list": [
                        {
                            "email": "alice@example.com",
                            "user_id": "ou_alice",
                        }
                    ]
                },
            },
        )
    )
    client = LarkClient(_settings(), None, tenant_access_token="tok")

    open_id = await get_user_open_id_by_email(
        client,
        "Alice@Example.com",
    )

    assert open_id == "ou_alice"
    request = route.calls.last.request
    assert request.url.params["user_id_type"] == "open_id"
    assert request.content == (
        b'{"emails":["alice@example.com"],"include_resigned":false}'
    )
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_get_user_open_id_by_email_rejects_missing_user():
    respx.post(
        "https://open.feishu.cn/open-apis/contact/v3/users/batch_get_id"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"code": 0, "data": {"user_list": []}},
        )
    )
    client = LarkClient(_settings(), None, tenant_access_token="tok")

    with pytest.raises(LarkUserNotFoundError):
        await get_user_open_id_by_email(client, "missing@example.com")

    await client.aclose()
