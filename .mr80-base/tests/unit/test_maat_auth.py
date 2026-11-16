import json
import threading
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from conciliator.config import Settings
from conciliator.integrations.maat.auth import (
    MaatJWTAuth,
    SdkMaatTokenProvider,
    application_http_client,
    application_token,
)
from conciliator.integrations.maat.client import MAAT_GATEWAY_ORIGIN, MaatClient, MaatError
from conciliator.integrations.maat.control_plane import MaatControlPlaneClient
from conciliator.utils.secrets import StaticSecretsProvider


def _secrets(ak="test-ak", sk="test-sk"):
    return StaticSecretsProvider({
        "conciliator/credentials/bytecloud_credentials": json.dumps({"ak": ak, "sk": sk}),
    })


def test_application_credentials_load_without_entering_settings_representations(monkeypatch):
    monkeypatch.setenv("BYTECLOUD_AK", "test-application-ak")
    monkeypatch.setenv("BYTECLOUD_SK", "test-application-sk")
    settings = Settings(_env_file=None)
    assert settings.bytecloud_ak.get_secret_value() == "test-application-ak"
    assert settings.bytecloud_sk.get_secret_value() == "test-application-sk"
    for visible in (repr(settings), str(settings), settings.model_dump_json()):
        assert "test-application-ak" not in visible
        assert "test-application-sk" not in visible
    assert "bytecloud_ak" not in settings.model_dump()
    assert "bytecloud_sk" not in settings.model_dump()


@pytest.mark.asyncio
async def test_gateway_carries_application_jwt_and_explicit_operator():
    def handle(request):
        assert request.url.path == "/api/open/devEnv/list"
        body = json.loads(request.content)
        assert type(body["pageNo"]) is int and type(body["pageSize"]) is int
        assert request.headers["X-Jwt-Token"] == "test.jwt"
        assert request.headers["x-action-user"] == "configured-operator"
        assert request.headers["domain"] == "maat;v1"
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"code": 0, "data": {"total": 0, "rows": []}})

    async with httpx.AsyncClient(
        base_url=MAAT_GATEWAY_ORIGIN, transport=httpx.MockTransport(handle)
    ) as http:
        client = MaatClient(http, AsyncMock(return_value="test.jwt"), "configured-operator")
        with pytest.raises(MaatError, match="match exactly once"):
            await client.get_environment("aeolus", "CN", "test")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_console_uses_the_same_application_provider_without_personal_fallback(status):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["X-Jwt-Token"] == "test.jwt"
        assert "cookie" not in request.headers
        assert request.url.path == "/api/open/devEnv/create"
        return httpx.Response(status)

    provider = AsyncMock(return_value="test.jwt")
    async with application_http_client(
        provider, "https://maat.bytedance.net", transport=httpx.MockTransport(handle)
    ) as http:
        client = MaatControlPlaneClient(http, "configured-operator")
        with pytest.raises(MaatError, match=f"HTTP {status}"):
            await client.create_environment({"name": "coding-test", "projectId": "project"})
    assert len(requests) == 1
    provider.assert_awaited_once()


@pytest.mark.asyncio
async def test_each_request_uses_the_current_token_from_the_sdk_provider():
    tokens = []

    def handle(request):
        tokens.append(request.headers["X-Jwt-Token"])
        return httpx.Response(200)

    provider = AsyncMock(side_effect=["first.jwt", "refreshed.jwt"])
    async with application_http_client(
        provider, "https://maat.bytedance.net", transport=httpx.MockTransport(handle)
    ) as http:
        await http.post("/api/open/devEnv/list")
        await http.post("/api/open/devEnv/list")
    assert tokens == ["first.jwt", "refreshed.jwt"]


@pytest.mark.asyncio
async def test_sdk_failure_is_sanitized_and_no_request_is_sent():
    provider = AsyncMock(side_effect=RuntimeError("test-secret-in-sdk-error"))
    transport = httpx.MockTransport(lambda _: pytest.fail("authentication must fail before sending"))
    async with application_http_client(provider, "https://maat.bytedance.net", transport=transport) as http:
        with pytest.raises(MaatError, match="acquisition failed") as caught:
            await http.post("/api/open/devApi/create", json={})
    assert "test-secret-in-sdk-error" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, "", "bad\r\nheader", "bad token", "中文", "bad\x7f"])
async def test_invalid_sdk_token_is_rejected_without_echo(token):
    with pytest.raises(MaatError, match="JWT is invalid"):
        await application_token(AsyncMock(return_value=token))


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["https://other.invalid/path", "http://maat.bytedance.net/path"])
async def test_origin_mismatch_is_rejected_before_acquiring_jwt(target):
    provider = AsyncMock(return_value="test.jwt")
    async with application_http_client(provider, "https://maat.bytedance.net") as http:
        with pytest.raises(MaatError, match="origin does not match"):
            await http.post(target)
    provider.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_redirect_never_forwards_jwt_even_when_follow_redirects_is_requested(status):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, headers={"location": "https://other.invalid/path"})

    async with application_http_client(
        AsyncMock(return_value="test.jwt"), "https://maat.bytedance.net", transport=httpx.MockTransport(handle)
    ) as http:
        with pytest.raises(MaatError, match="redirects"):
            await http.post("/api/open/devApi/create", json={}, follow_redirects=True)
    assert len(requests) == 1


@pytest.mark.parametrize("origin", ["http://maat.bytedance.net", "https://user:pass@maat.bytedance.net",
                                     "https://maat.bytedance.net/path", "https://maat.bytedance.net/?query=1"])
def test_auth_requires_an_explicit_https_origin(origin):
    with pytest.raises(ValueError, match="HTTPS origin"):
        MaatJWTAuth(AsyncMock(), origin)


@pytest.mark.asyncio
async def test_sdk_client_is_reused_and_blocking_calls_run_off_the_event_loop(monkeypatch):
    import sys

    loop_thread = threading.get_ident()
    client = Mock()

    def generate(credential, site):
        assert threading.get_ident() != loop_thread
        assert site == "cn"
        return "sdk.jwt"

    client.generate_with_aksk.side_effect = generate
    client.userinfo.return_value = SimpleNamespace(username="real-app-operator")
    sdk = SimpleNamespace(Credential=Mock(return_value=object()), JWTClient=Mock(return_value=client))
    monkeypatch.setitem(sys.modules, "bytecloud_cloud_sdk", sdk)
    provider = SdkMaatTokenProvider(
        Settings(_env_file=None), _secrets(), proxy_url="http://proxy.invalid:8118",
    )
    assert await provider() == "sdk.jwt"
    assert await provider.operator() == "real-app-operator"
    sdk.JWTClient.assert_called_once_with(region="cn", timeout=10, proxy_url="http://proxy.invalid:8118")
    sdk.Credential.assert_called_once_with("test-ak", "test-sk")
    assert client.generate_with_aksk.call_count == 2
    client.userinfo.assert_called_once_with("sdk.jwt")


@pytest.mark.parametrize(("ak", "sk"), [
    ("", "test-sk"), ("test-ak", ""),
])
def test_sdk_provider_requires_both_credentials(ak, sk):
    with pytest.raises(RuntimeError, match="nonempty string ak and sk"):
        SdkMaatTokenProvider(Settings(_env_file=None), _secrets(ak, sk))


@pytest.mark.asyncio
async def test_sdk_exchange_error_does_not_expose_credentials(monkeypatch):
    import sys

    client = Mock()
    client.generate_with_aksk.side_effect = RuntimeError("test-secret-sdk-detail")
    monkeypatch.setitem(sys.modules, "bytecloud_cloud_sdk", SimpleNamespace(
        Credential=Mock(), JWTClient=Mock(return_value=client),
    ))
    provider = SdkMaatTokenProvider(Settings(_env_file=None), _secrets())
    with pytest.raises(MaatError, match="acquisition failed") as caught:
        await provider()
    assert "test-secret-sdk-detail" not in "".join(traceback.format_exception(caught.value))
