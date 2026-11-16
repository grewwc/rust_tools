from unittest.mock import Mock

import httpx
import pytest

from conciliator.integrations.ppe import scm
from conciliator.integrations.ppe.bits import BitsScmClient
from conciliator.integrations.ppe.scm import PlatformError, request


@pytest.mark.parametrize("domain", ["env_platform;v1", "tce;v1", "scm;v1", "icm;v1", "bits"])
@pytest.mark.parametrize("status", [400, 200])
async def test_platform_errors_include_safe_diagnostics_in_card_and_log(monkeypatch, domain, status):
    log = Mock()
    monkeypatch.setattr(scm, "log", log)
    payload = {
        "code": "PARAM_INVALID",
        "request_id": "platform-request-123",
        "message": "Authorization: private-credential; env_value=private-config",
        "data": {"token": "private-token"},
    }
    calls = []

    def transport(req):
        calls.append(req)
        return httpx.Response(status, json=payload, headers={"X-Tt-Logid": "trace-456"})

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        with pytest.raises(PlatformError) as caught:
            path = "/api/status?token=private-query"
            if domain == "bits":
                await BitsScmClient(http)._request("GET", path, "operator")
            else:
                await request(http, domain, "GET", path, headers={"X-Jwt-Token": "private-auth"})
    message = str(caught.value)
    assert f"平台：{domain}" in message
    assert "接口：GET /api/status" in message
    assert "业务码：PARAM_INVALID" in message and "请求 ID：trace-456" in message
    assert not caught.value.uncertain and len(calls) == 1
    log.warning.assert_called_once()
    details = log.warning.call_args.kwargs
    assert details["path"] == "/api/status" and details["http_status"] == status
    assert details["code"] == "PARAM_INVALID" and details["request_id"] == "trace-456"
    for secret in ("private-credential", "private-config", "private-token", "private-query", "private-auth"):
        assert secret not in message and secret not in str(details)


@pytest.mark.parametrize("payload", [
    {"code": 10204, "request_id": "request-789"},
    {"error_code": -1, "log_id": "request-789"},
])
async def test_numeric_error_code_and_body_request_id_are_retained(payload):
    async with httpx.AsyncClient(
        base_url="https://platform.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(400, json=payload)),
    ) as http:
        with pytest.raises(PlatformError) as caught:
            await request(http, "env_platform;v1", "GET", "/api/v4/instance_meta")
    assert f"业务码：{payload.get('code', payload.get('error_code'))}" in str(caught.value)
    assert "请求 ID：request-789" in str(caught.value)


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="private-response-body"),
    httpx.Response(400, json={
        "code": {"token": "private-token"}, "request_id": "Authorization: private-token",
        "message": "private-response-body",
    }),
    httpx.Response(400, json={
        "code": "x" * 129, "request_id": ["private-token"], "logid": False,
    }),
])
async def test_unstructured_errors_stay_safe_and_writes_are_never_replayed(response, monkeypatch):
    log = Mock()
    monkeypatch.setattr(scm, "log", log)
    calls = []

    def transport(req):
        calls.append(req)
        return response

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        with pytest.raises(PlatformError) as caught:
            await request(http, "env_platform;v1", "PUT", "/api/v4/dsl/env/ppe_test", write=True)
    message = str(caught.value)
    assert caught.value.uncertain and len(calls) == 1
    assert "接口：PUT /api/v4/dsl/env/ppe_test" in message
    assert "业务码：" not in message and "请求 ID：" not in message
    assert "private-" not in message and "private-" not in str(log.warning.call_args)


@pytest.mark.parametrize("write", [False, True])
async def test_transport_failure_keeps_endpoint_without_exception_credentials(write, monkeypatch):
    log = Mock()
    monkeypatch.setattr(scm, "log", log)
    calls = []

    def transport(req):
        calls.append(req)
        raise httpx.ConnectError("private-signed-url", request=req)

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        with pytest.raises(PlatformError) as caught:
            await request(http, "tce;v1", "POST" if write else "GET", "/api/status", write=write)
    assert "平台请求未完成" in str(caught.value) and "/api/status" in str(caught.value)
    assert caught.value.uncertain == write and len(calls) == 1
    assert "private-signed-url" not in str(caught.value) + str(log.warning.call_args)


async def test_expected_absence_does_not_emit_an_error(monkeypatch):
    log = Mock()
    monkeypatch.setattr(scm, "log", log)
    async with httpx.AsyncClient(
        base_url="https://platform.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(400, json={"code": "NOT_EXIST"})),
    ) as http:
        assert await request(
            http, "env_platform;v1", "GET", "/api/v4/env/ppe_test", not_found_code="NOT_EXIST",
        ) is None
    log.warning.assert_not_called()
