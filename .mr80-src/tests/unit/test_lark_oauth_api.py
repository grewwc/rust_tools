from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from conciliator.api.oauth import callback, router
from conciliator.integrations.lark.client import LarkError
from conciliator.services.lark_user_token_service import InvalidOAuthState, OAuthIdentityMismatch


class _TokenService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.completed: list[tuple[str, str]] = []

    def authorize_url(self, user_key: str) -> str:
        return (
            "https://open.feishu.cn/oauth?state=signed-state"
            "&redirect_uri=https%3A%2F%2Ftest%2Foauth%2Flark%2Fcallback"
        )

    def parse_state(self, state: str) -> str:
        assert state == "signed-state"
        return "user-1"

    async def complete_authorization(self, *, code: str, state: str) -> None:
        self.completed.append((code, state))
        if self.error:
            raise self.error


class _AeolusService:
    def __init__(self) -> None:
        self.users: list[str] = []

    async def resume_authorization_blocked(self, user_key: str) -> int:
        self.users.append(user_key)
        return 1


def _request(tokens: _TokenService, aeolus: _AeolusService) -> SimpleNamespace:
    state = SimpleNamespace(
        lark_user_token_service=tokens,
        aeolus_service=aeolus,
    )
    return SimpleNamespace(
        app=SimpleNamespace(state=state), cookies={"lark_oauth_state": "signed-state"},
    )


@pytest.mark.asyncio
async def test_callback_resumes_authorization_blocked_tasks() -> None:
    tokens = _TokenService()
    aeolus = _AeolusService()

    response = await callback(
        _request(tokens, aeolus), code="oauth-code", state="signed-state"
    )

    assert response.status_code == 200
    assert tokens.completed == [("oauth-code", "signed-state")]
    assert aeolus.users == ["user-1"]
    assert "自动继续" in response.body.decode()


@pytest.mark.asyncio
async def test_callback_does_not_resume_when_token_exchange_fails() -> None:
    tokens = _TokenService(LarkError("invalid grant", http_status=400))
    aeolus = _AeolusService()

    response = await callback(
        _request(tokens, aeolus), code="bad-code", state="signed-state"
    )

    assert response.status_code == 200
    assert aeolus.users == []
    assert "授权失败" in response.body.decode()


@pytest.mark.asyncio
async def test_callback_escapes_exchange_error_html() -> None:
    tokens = _TokenService(LarkError('<script>alert("unsafe")</script>', http_status=400))
    aeolus = _AeolusService()

    response = await callback(
        _request(tokens, aeolus), code="bad-code", state="signed-state",
    )

    body = response.body.decode()
    assert "<script>" not in body
    assert "&lt;script&gt;" in body
    assert aeolus.users == []


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [
    {"error": "access_denied", "state": "signed-state"},
    {"state": "signed-state"},
    {"code": "unused-code", "error": "access_denied", "state": "signed-state"},
])
async def test_denied_or_incomplete_callback_returns_html_without_exchange(params) -> None:
    tokens, aeolus = _TokenService(), _AeolusService()
    app = FastAPI()
    app.include_router(router)
    app.state.lark_user_token_service = tokens
    app.state.aeolus_service = aeolus

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
        cookies={"lark_oauth_state": "signed-state"},
    ) as client:
        response = await client.get("/oauth/lark/callback", params=params)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "授权失败" in response.text
    assert tokens.completed == []
    assert aeolus.users == []


@pytest.mark.asyncio
async def test_denied_callback_still_rejects_invalid_state(monkeypatch) -> None:
    tokens, aeolus = _TokenService(), _AeolusService()

    def invalid_state(state: str) -> str:
        raise InvalidOAuthState("invalid state")

    monkeypatch.setattr(tokens, "parse_state", invalid_state)
    app = FastAPI()
    app.include_router(router)
    app.state.lark_user_token_service = tokens
    app.state.aeolus_service = aeolus
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
        cookies={"lark_oauth_state": "invalid"},
    ) as client:
        response = await client.get("/oauth/lark/callback", params={
            "error": "access_denied", "state": "invalid",
        })

    assert response.status_code == 400
    assert tokens.completed == []
    assert aeolus.users == []


@pytest.mark.asyncio
async def test_start_binds_callback_to_browser_and_clears_cookie_after_consent() -> None:
    tokens, aeolus = _TokenService(), _AeolusService()
    app = FastAPI()
    app.include_router(router)
    app.state.lark_user_token_service = tokens
    app.state.aeolus_service = aeolus
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        response = await client.get("/oauth/lark/start", params={"user_key": "user-1"})
        assert response.status_code == 302
        cookie = response.headers["set-cookie"]
        assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie
        assert response.headers["cache-control"] == "no-store"
        response = await client.get("/oauth/lark/callback", params={
            "state": "signed-state", "code": "code",
        })
        assert response.status_code == 200
        assert "Max-Age=0" in response.headers["set-cookie"]
        response = await client.get("/oauth/lark/callback", params={
            "state": "signed-state", "code": "replayed-code",
        })
        assert response.status_code == 400
    assert tokens.completed == [("code", "signed-state")]
    assert aeolus.users == ["user-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cookie", [None, "other-browser-state"])
async def test_callback_rejects_unbound_browser_without_exchange(cookie) -> None:
    tokens, aeolus = _TokenService(), _AeolusService()
    request = _request(tokens, aeolus)
    request.cookies = {} if cookie is None else {"lark_oauth_state": cookie}
    response = await callback(request, code="code", state="signed-state")
    assert response.status_code == 400
    assert tokens.completed == []
    assert aeolus.users == []


@pytest.mark.asyncio
async def test_callback_identity_mismatch_never_resumes_tasks() -> None:
    tokens = _TokenService(OAuthIdentityMismatch("wrong account"))
    aeolus = _AeolusService()
    response = await callback(_request(tokens, aeolus), code="code", state="signed-state")
    assert response.status_code == 403
    assert "wrong account" in response.body.decode()
    assert aeolus.users == []


@pytest.mark.asyncio
async def test_callback_without_state_returns_friendly_html() -> None:
    app = FastAPI()
    app.include_router(router)
    tokens, aeolus = _TokenService(), _AeolusService()
    app.state.lark_user_token_service = tokens
    app.state.aeolus_service = aeolus
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        response = await client.get("/oauth/lark/callback", params={"error": "access_denied"})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("text/html")
    assert tokens.completed == []
