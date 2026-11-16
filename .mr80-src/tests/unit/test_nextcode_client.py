import asyncio
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.nextcode import client as client_module
from conciliator.integrations.nextcode.client import NextCodeClient, NextCodeError
from conciliator.utils.retry import TransientError
from conciliator.utils.secrets import StaticSecretsProvider


class StaticZtiProvider:
    def get_token(self) -> str:
        return "zti-token"


def _settings() -> Settings:
    return Settings(
        env="dev",
        service_psm="x",
        kms_secret_nextcode_app_secret_alias=(
            "test/credentials/nextcode_app_secret"
        ),
        kms_secret_nextcode_webhook_secret_alias=(
            "test/credentials/nextcode_webhook_secret"
        ),
        kms_secret_lark_app_secret_alias="test/credentials/lark_app_secret",
        nextcode_app_id="app1",
        coco_api_base="x",
        coco_workspace_id="x",
        coco_default_agent_id="x",
        lark_app_id="x",
        mysql_psm="x",
        nextcode_api_base="https://nc.test/v2/",
    )  # type: ignore[arg-type]


def _client() -> NextCodeClient:
    return NextCodeClient(
        _settings(),
        StaticSecretsProvider(
            {
                "test/credentials/nextcode_app_secret": "secret123",
                "test/credentials/nextcode_webhook_secret": "webhook123",
            }
        ),
        zti_provider=StaticZtiProvider(),
    )


@pytest.mark.asyncio
async def test_webhook_signature_matches_official_contract():
    client = _client()
    body = b'{"Action":"created"}'
    digest = hmac.new(b"webhook123", body, hashlib.sha256).hexdigest()
    assert client.verify_webhook_signature(body, f"sha256={digest}") is True
    assert client.verify_webhook_signature(body, digest) is False
    await client.aclose()


@pytest.fixture
def retry_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)
    return sleep


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("rejection", [
    httpx.Response(429, text="Too Many Requests"),
    httpx.Response(429, json=["Too Many Requests"]),
    httpx.Response(200, json={"ResponseMetadata": {
        "RequestId": "limited", "Error": {"Code": "TooManyRequests", "Message": "limited"},
    }}),
])
@respx.mock
async def test_rate_limit_recovers_with_same_body_and_fresh_auth(rejection, read_only, retry_sleep):
    route = respx.post("https://nc.test/v2/?Action=CreateComment").mock(side_effect=[
        rejection, httpx.Response(200, json={"Result": {"Comment": {"Id": "c1"}}}),
    ])
    client = _client()
    client._zti = Mock(get_token=Mock(side_effect=["first", "second"]))
    try:
        result = await client.call("CreateComment", {"Content": "hello"}, read_only=read_only)
        assert result == {"Comment": {"Id": "c1"}}
        assert route.call_count == 2
        assert all(json.loads(call.request.content) == {"Content": "hello"} for call in route.calls)
        assert [call.request.headers["X-Code-ZTI-Token"] for call in route.calls] == ["first", "second"]
        assert 1 <= retry_sleep.call_args.args[0] <= 2
    finally:
        await client.aclose()


@pytest.mark.parametrize("read_only", [False, True])
@respx.mock
async def test_rate_limit_exhausts_without_nested_retry_budget(read_only, retry_sleep):
    route = respx.post("https://nc.test/v2/?Action=CreateComment").mock(
        return_value=httpx.Response(429, json={"ResponseMetadata": {
            "RequestId": "limited", "Error": {"Code": "TooManyRequests"},
        }}),
    )
    client = _client()
    try:
        with pytest.raises(NextCodeError) as caught:
            await client.call("CreateComment", {}, read_only=read_only)
        assert caught.value.status == 429
        assert caught.value.request_id == "limited"
        assert route.call_count == 9
        assert retry_sleep.await_count == 8
        for call, base in zip(retry_sleep.call_args_list, [1, 2, 4, 8, 16, 16, 16, 16], strict=True):
            assert base <= call.args[0] <= base * 2
    finally:
        await client.aclose()


@pytest.mark.parametrize("use_http_date", [False, True])
@respx.mock
async def test_retry_after_is_respected(use_http_date, retry_sleep):
    header = format_datetime(datetime.now(UTC) + timedelta(seconds=60)) if use_http_date else "10"
    respx.post("https://nc.test/v2/?Action=CreateComment").mock(side_effect=[
        httpx.Response(429, text="limited", headers={"Retry-After": header}),
        httpx.Response(200, json={"Result": {}}),
    ])
    client = _client()
    try:
        assert await client.call("CreateComment", {}) == {}
        assert retry_sleep.call_args.args[0] >= 10
    finally:
        await client.aclose()


@pytest.mark.parametrize("value", [None, "", "invalid", "-1", "NaN", "inf"])
def test_invalid_retry_after_is_ignored(value):
    assert client_module._retry_after_seconds(value) == 0


@respx.mock
async def test_retry_after_beyond_deadline_does_not_retry_early(retry_sleep):
    route = respx.post("https://nc.test/v2/?Action=CreateComment").mock(
        return_value=httpx.Response(429, text="limited", headers={"Retry-After": "181"}),
    )
    client = _client()
    try:
        with pytest.raises(NextCodeError):
            await client.call("CreateComment", {})
        assert route.call_count == 1
        retry_sleep.assert_not_awaited()
    finally:
        await client.aclose()


@pytest.mark.parametrize("failure", [httpx.Response(503), httpx.ReadTimeout("timeout")])
@respx.mock
async def test_writes_do_not_retry_ambiguous_failures(failure, retry_sleep):
    route = respx.post("https://nc.test/v2/?Action=CreateComment").mock(side_effect=[failure])
    client = _client()
    try:
        with pytest.raises(TransientError):
            await client.call("CreateComment", {})
        assert route.call_count == 1
        retry_sleep.assert_not_awaited()
    finally:
        await client.aclose()


@respx.mock
async def test_read_transient_retry_is_preserved(retry_sleep):
    route = respx.post("https://nc.test/v2/?Action=GetRepository").mock(side_effect=[
        httpx.Response(503), httpx.Response(200, json={"Result": {"Id": "r1"}}),
    ])
    client = _client()
    try:
        assert await client.call("GetRepository", {}, read_only=True) == {"Id": "r1"}
        assert route.call_count == 2
    finally:
        await client.aclose()


@respx.mock
async def test_cancellation_during_backoff_propagates(retry_sleep):
    route = respx.post("https://nc.test/v2/?Action=CreateComment").mock(
        return_value=httpx.Response(429, text="limited"),
    )
    retry_sleep.side_effect = asyncio.CancelledError
    client = _client()
    try:
        with pytest.raises(asyncio.CancelledError):
            await client.call("CreateComment", {})
        assert route.call_count == 1
    finally:
        await client.aclose()


@respx.mock
async def test_deadline_bounds_inflight_retry(monkeypatch, retry_sleep):
    async def hanging_request(request):
        await asyncio.Event().wait()

    monkeypatch.setattr(client_module, "_CALL_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(client_module.random, "uniform", lambda *_: 0)
    respx.post("https://nc.test/v2/?Action=CreateComment").mock(side_effect=[
        httpx.Response(429, text="limited"), hanging_request,
    ])
    client = _client()
    try:
        with pytest.raises(NextCodeError) as caught:
            await client.call("CreateComment", {})
        assert caught.value.status == 429
        retry_sleep.assert_awaited_once()
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_call_uses_action_endpoint_headers_and_result_envelope():
    route = respx.post("https://nc.test/v2/?Action=GetRepository").mock(
        return_value=httpx.Response(
            200,
            json={
                "ResponseMetadata": {"RequestId": "req-1"},
                "Result": {"Repository": {"Id": "repo-1"}},
            },
        )
    )
    client = _client()
    result = await client.call(
        "GetRepository", {"Path": "xiahaitao/Conciliator"}, read_only=True
    )

    assert result == {"Repository": {"Id": "repo-1"}}
    request = route.calls.last.request
    assert request.headers["X-Code-App-ID"] == "app1"
    assert request.headers["X-Code-App-Secret"] == "secret123"
    assert request.headers["X-Code-ZTI-Token"] == "zti-token"
    assert json.loads(request.content) == {"Path": "xiahaitao/Conciliator"}
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_call_raises_structured_openapi_error():
    respx.post("https://nc.test/v2/?Action=GetRepository").mock(
        return_value=httpx.Response(
            200,
            json={
                "ResponseMetadata": {
                    "RequestId": "req-denied",
                    "Error": {"Code": "AccessDenied", "Message": "denied"},
                }
            },
        )
    )
    client = _client()
    with pytest.raises(NextCodeError) as caught:
        await client.call("GetRepository", {"Id": "repo-1"}, read_only=True)
    assert caught.value.code == "AccessDenied"
    assert caught.value.request_id == "req-denied"
    await client.aclose()
