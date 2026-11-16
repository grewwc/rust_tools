"""Unit tests for per-user Feishu OAuth (consent link, tokens, refresh, fallback)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from conciliator.config import Settings
from conciliator.integrations.lark import oauth
from conciliator.integrations.lark.client import LarkError
from conciliator.repositories.lark_user_token_repo import (
    LarkUserTokenRepository,
    StoredUserToken,
)
from conciliator.repositories.models import Base
from conciliator.services.lark_user_token_service import (
    InvalidOAuthState,
    LarkUserTokenService,
)
from conciliator.utils.retry import TransientError

USER_KEY = "7577212504225483993"
REDIRECT = "http://127.0.0.1:8000/oauth/lark/callback"

TOKEN_PAYLOAD = {
    "code": 0,
    "access_token": "u-access",
    "refresh_token": "ur-refresh",
    "expires_in": 7200,
    "refresh_token_expires_in": 2592000,
    "token_type": "Bearer",
    "scope": "wiki:node:read",
}


class _StaticSecrets:
    """Stands in for SecretsProvider; the service only calls .get(alias)."""

    def __init__(self, value: str = "app-secret") -> None:
        self._value = value

    def get(self, alias: str) -> str:
        return self._value


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"lark_app_id": "cli_test_app", "lark_oauth_redirect_uri": REDIRECT}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _repo(tmp_path) -> LarkUserTokenRepository:
    engine = create_engine(f"sqlite:///{tmp_path / 'lark_tokens.db'}")
    Base.metadata.create_all(engine)
    return LarkUserTokenRepository(sessionmaker(bind=engine))


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _service(tmp_path, handler, **overrides: object) -> LarkUserTokenService:
    async def identity_resolver(user_key: str) -> str:
        return "ou_user"

    return LarkUserTokenService(
        settings=_settings(**overrides),
        secrets=_StaticSecrets(),
        repo=_repo(tmp_path),
        http=_client(handler),
        identity_resolver=identity_resolver,
    )


def _expired_token(user_key: str = USER_KEY) -> StoredUserToken:
    now = datetime.now(UTC).replace(tzinfo=None)
    return StoredUserToken(
        open_id=f"ou_{user_key}", user_key=user_key,
        refresh_token=f"ur_{user_key}", access_token="u-expired",
        access_expires_at=now - timedelta(minutes=5),
        refresh_expires_at=now + timedelta(days=1),
    )


class _MemoryTokenRepo:
    def __init__(self, *tokens: StoredUserToken) -> None:
        self.tokens = {token.user_key: token for token in tokens}

    async def find_by_user_key(self, user_key: str) -> StoredUserToken | None:
        return self.tokens.get(user_key)

    async def save(self, token: StoredUserToken) -> None:
        self.tokens[token.user_key] = token

    async def replace_if_current(self, expected: StoredUserToken, token: StoredUserToken) -> bool:
        if self.tokens.get(expected.user_key) != expected:
            return False
        self.tokens[token.user_key] = token
        return True


def _memory_service(handler, *tokens: StoredUserToken) -> LarkUserTokenService:
    return LarkUserTokenService(
        settings=_settings(), secrets=_StaticSecrets(),
        repo=_MemoryTokenRepo(*tokens),  # type: ignore[arg-type]
        http=_client(handler),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [
    ("access", True), ("refresh", True), ("unknown_refresh_expiry", True),
    ("expired_refresh", False), ("no_refresh", False), ("missing", False),
])
async def test_authorization_available_is_read_only(mode, expected):
    now = datetime.now(UTC).replace(tzinfo=None)
    token = _expired_token()
    if mode == "access":
        token = replace(token, access_expires_at=now + timedelta(hours=1), refresh_token="")
    elif mode == "unknown_refresh_expiry":
        token = replace(token, refresh_expires_at=None)
    elif mode == "expired_refresh":
        token = replace(token, refresh_expires_at=now - timedelta(seconds=1))
    elif mode == "no_refresh":
        token = replace(token, refresh_token="")

    def unexpected_http(request):
        pytest.fail("Authorization polling must not contact Feishu")

    service = _memory_service(unexpected_http, *([] if mode == "missing" else [token]))

    async def unexpected_save(*args):
        pytest.fail("Authorization polling must not write tokens")

    service._repo.save = unexpected_save
    service._repo.replace_if_current = unexpected_save
    assert await service.authorization_available(USER_KEY) is expected
    assert await service.authorization_available("") is False


@pytest.mark.asyncio
async def test_authorization_available_surfaces_store_failure():
    service = _memory_service(lambda request: pytest.fail("must not contact Feishu"))

    async def broken_lookup(user_key):
        raise SQLAlchemyError("store unavailable")

    service._repo.find_by_user_key = broken_lookup
    with pytest.raises(TransientError):
        await service.authorization_available(USER_KEY)


def test_authorize_url_encodes_redirect_and_round_trips_state(tmp_path):
    service = _service(tmp_path, lambda request: httpx.Response(200, json={}))
    state = service.create_state(USER_KEY)
    url = service.authorize_url(USER_KEY)
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert parsed.path == "/open-apis/authen/v1/authorize"
    assert params["client_id"] == ["cli_test_app"]
    assert params["scope"] == [
        "offline_access docx:document:readonly wiki:node:read"
    ]
    assert service.parse_state(state) == USER_KEY


def test_authorize_url_requires_redirect_uri(tmp_path):
    service = _service(tmp_path, lambda request: httpx.Response(200, json={}), lark_oauth_redirect_uri="")
    assert service.enabled is False
    with pytest.raises(RuntimeError):
        service.authorize_url(USER_KEY)


def test_parse_state_rejects_tampered_and_malformed_values(tmp_path):
    service = _service(tmp_path, lambda request: httpx.Response(200, json={}))
    state = service.create_state(USER_KEY)
    user_key, issued_at, signature = state.rsplit(".", 2)
    tampered = f"{user_key}.{issued_at}.{'0' * len(signature)}"
    with pytest.raises(InvalidOAuthState):
        service.parse_state(tampered)
    with pytest.raises(InvalidOAuthState):
        service.parse_state("not-a-state")
    with pytest.raises(InvalidOAuthState):
        service.parse_state(f"{USER_KEY}.abc.{signature}")


def test_parse_state_rejects_expired_state(tmp_path):
    service = _service(tmp_path, lambda request: httpx.Response(200, json={}))
    stale = service.create_state(USER_KEY)
    user_key, _, nonce, _ = stale.rsplit(".", 3)
    issued_at = int(datetime.now(UTC).timestamp()) - 3600
    signature = service._signature(user_key, issued_at, nonce)
    expired = f"{user_key}.{issued_at}.{nonce}.{signature}"
    with pytest.raises(InvalidOAuthState):
        service.parse_state(expired)


async def test_complete_authorization_exchanges_code_and_stores_tokens(tmp_path):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/token"):
            seen.append(json.loads(request.content.decode()))
            return httpx.Response(200, json=TOKEN_PAYLOAD)
        return httpx.Response(200, json={"code": 0, "data": {"open_id": "ou_user"}})

    service = _service(tmp_path, handler)
    open_id = await service.complete_authorization(
        code="auth-code", state=service.create_state(USER_KEY)
    )
    assert open_id == "ou_user"
    assert seen[0]["grant_type"] == "authorization_code"
    assert seen[0]["client_id"] == "cli_test_app"
    assert seen[0]["code"] == "auth-code"
    assert seen[0]["redirect_uri"] == REDIRECT

    stored = await service._repo.find_by_user_key(USER_KEY)
    assert stored is not None
    assert stored.open_id == "ou_user"
    assert stored.refresh_token == "ur-refresh"
    assert stored.access_expires_at is not None


async def test_user_access_token_uses_cached_token_without_http(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError(f"unexpected call to {request.url}")

    service = _service(tmp_path, handler)
    now = datetime.now(UTC).replace(tzinfo=None)
    await service._repo.save(
        StoredUserToken(
            open_id="ou_user",
            user_key=USER_KEY,
            refresh_token="ur-refresh",
            access_token="u-cached",
            access_expires_at=now + timedelta(hours=1),
            refresh_expires_at=now + timedelta(days=30),
        )
    )
    assert await service.user_access_token(USER_KEY) == "u-cached"


async def test_user_access_token_refreshes_when_expired(tmp_path):
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "code": 0,
                "access_token": "u-new",
                "refresh_token": "ur-new",
                "expires_in": 7200,
                "refresh_token_expires_in": 2592000,
            },
        )

    service = _service(tmp_path, handler)
    now = datetime.now(UTC).replace(tzinfo=None)
    await service._repo.save(
        StoredUserToken(
            open_id="ou_user",
            user_key=USER_KEY,
            refresh_token="ur-old",
            access_token="u-stale",
            access_expires_at=now - timedelta(minutes=5),
            refresh_expires_at=now + timedelta(days=30),
        )
    )
    assert await service.user_access_token(USER_KEY) == "u-new"
    assert calls[0]["grant_type"] == "refresh_token"
    assert calls[0]["refresh_token"] == "ur-old"
    stored = await service._repo.find_by_user_key(USER_KEY)
    assert stored.refresh_token == "ur-new"


async def test_user_access_token_returns_none_without_or_with_spent_authorization(tmp_path):
    service = _service(tmp_path, lambda request: httpx.Response(200, json={}))
    assert await service.user_access_token(USER_KEY) is None

    now = datetime.now(UTC).replace(tzinfo=None)
    await service._repo.save(
        StoredUserToken(
            open_id="ou_user",
            user_key=USER_KEY,
            refresh_token="ur-spent",
            access_token="",
            access_expires_at=None,
            refresh_expires_at=now - timedelta(days=1),
        )
    )
    assert await service.user_access_token(USER_KEY) is None


async def test_user_access_token_returns_none_when_refresh_fails(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "The refresh token provided is invalid.",
                "code": 20026,
            },
        )

    service = _service(tmp_path, handler)
    now = datetime.now(UTC).replace(tzinfo=None)
    await service._repo.save(
        StoredUserToken(
            open_id="ou_user",
            user_key=USER_KEY,
            refresh_token="ur-revoked",
            access_token="",
            access_expires_at=None,
            refresh_expires_at=now + timedelta(days=1),
        )
    )
    assert await service.user_access_token(USER_KEY) is None


async def test_user_access_token_surfaces_temporary_refresh_failure(tmp_path):
    service = _service(tmp_path, lambda request: httpx.Response(503, text="unavailable"))
    now = datetime.now(UTC).replace(tzinfo=None)
    await service._repo.save(
        StoredUserToken(
            open_id="ou_user",
            user_key=USER_KEY,
            refresh_token="ur-valid",
            access_token="",
            access_expires_at=None,
            refresh_expires_at=now + timedelta(days=1),
        )
    )
    with pytest.raises(TransientError):
        await service.user_access_token(USER_KEY)


async def test_exchange_code_surfaces_oauth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": "invalid_grant", "error_description": "code used", "code": 20003},
        )

    with pytest.raises(LarkError) as excinfo:
        await oauth.exchange_code(
            app_id="cli_test_app",
            app_secret="secret",
            code="bad",
            redirect_uri=REDIRECT,
            http=_client(handler),
        )
    assert "invalid_grant" in str(excinfo.value)


def test_parse_token_bundle_accepts_nested_and_missing_refresh_token():
    nested = oauth.parse_token_bundle(
        {"code": 0, "data": {"access_token": "u", "refresh_token": "ur", "expires_in": 60}}
    )
    assert nested.access_token == "u"
    assert nested.expires_in == 60
    # The live v2 endpoint can issue an access_token without a refresh_token;
    # that must parse (refresh_token becomes ""), not raise.
    no_refresh = oauth.parse_token_bundle({"code": 0, "access_token": "u", "expires_in": 7200})
    assert no_refresh.access_token == "u"
    assert no_refresh.refresh_token == ""
    assert no_refresh.expires_in == 7200
    with pytest.raises(LarkError):
        oauth.parse_token_bundle({"code": 0, "refresh_token": "ur"})


async def test_user_access_token_returns_none_when_no_refresh_token_issued(tmp_path):
    """A token issued without a refresh_token can't rotate; fall back, don't call Feishu."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError(f"unexpected call to {request.url}")

    service = _service(tmp_path, handler)
    await service._repo.save(
        StoredUserToken(
            open_id="ou_user",
            user_key=USER_KEY,
            refresh_token="",
            access_token="",
            access_expires_at=None,
            refresh_expires_at=None,
        )
    )
    assert await service.user_access_token(USER_KEY) is None


async def test_user_access_token_surfaces_store_unavailability():

    class _BrokenRepo:
        async def find_by_user_key(self, user_key: str):
            raise SQLAlchemyError("table lark_user_tokens does not exist")

    service = LarkUserTokenService(
        settings=_settings(),
        secrets=_StaticSecrets(),
        repo=_BrokenRepo(),  # type: ignore[arg-type]
        http=_client(lambda request: httpx.Response(200, json={})),
    )
    with pytest.raises(TransientError):
        await service.user_access_token(USER_KEY)


async def test_concurrent_reads_share_one_rotating_refresh():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        entered.set()
        await release.wait()
        if len(calls) > 1:
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(200, json=TOKEN_PAYLOAD)

    service = _memory_service(handler, _expired_token())
    tasks = [asyncio.create_task(service.user_access_token(USER_KEY)) for _ in range(5)]
    await asyncio.wait_for(entered.wait(), timeout=2)
    release.set()
    assert await asyncio.gather(*tasks) == ["u-access"] * 5
    assert len(calls) == 1
    assert calls[0]["refresh_token"] == _expired_token().refresh_token
    stored = await service._repo.find_by_user_key(USER_KEY)
    assert stored.refresh_token == "ur-refresh"
    assert not service._token_requests


async def test_concurrent_refresh_failure_is_shared_and_later_request_can_retry():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        if calls == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=TOKEN_PAYLOAD)

    service = _memory_service(handler, _expired_token())
    tasks = [asyncio.create_task(service.user_access_token(USER_KEY)) for _ in range(5)]
    await asyncio.wait_for(entered.wait(), timeout=2)
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(result, TransientError) for result in results)
    assert calls == 1
    assert not service._token_requests
    assert await service.user_access_token(USER_KEY) == "u-access"
    assert calls == 2


async def test_different_users_can_refresh_independently():
    both_entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["refresh_token"])
        if len(calls) == 2:
            both_entered.set()
        await release.wait()
        return httpx.Response(200, json=TOKEN_PAYLOAD)

    service = _memory_service(handler, _expired_token("first"), _expired_token("second"))
    tasks = [
        asyncio.create_task(service.user_access_token(key))
        for key in ("first", "second")
    ]
    try:
        await asyncio.wait_for(both_entered.wait(), timeout=2)
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert sorted(calls) == ["ur_first", "ur_second"]


@pytest.mark.parametrize("known_expiry", [True, False])
async def test_refresh_without_replacement_preserves_old_token_and_deadline(tmp_path, known_expiry):
    payload = {
        "access_token": "u-refreshed", "expires_in": 7200,
        # Without a replacement token, this must not extend the old one's life.
        "refresh_token_expires_in": 2592000,
    }
    service = _service(tmp_path, lambda request: httpx.Response(200, json=payload))
    old = _expired_token()
    if not known_expiry:
        old = replace(old, refresh_expires_at=None)
    await service._repo.save(old)
    assert await service.user_access_token(USER_KEY) == "u-refreshed"
    stored = await service._repo.find_by_user_key(USER_KEY)
    assert stored.refresh_token == old.refresh_token
    assert stored.refresh_expires_at == old.refresh_expires_at


async def test_new_consent_without_refresh_does_not_inherit_old_credentials(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(200, json={"access_token": "u-consent", "expires_in": 7200})
        return httpx.Response(200, json={"data": {"open_id": "ou_user"}})

    service = _service(tmp_path, handler)
    await service._repo.save(replace(_expired_token(), open_id="ou_user"))
    await service.complete_authorization(code="new-code", state=service.create_state(USER_KEY))
    stored = await service._repo.find_by_user_key(USER_KEY)
    assert stored.refresh_token == ""
    assert stored.refresh_expires_at is None
    assert await service.user_access_token(USER_KEY) == "u-consent"
    await service._repo.save(replace(stored, access_expires_at=_expired_token().access_expires_at))
    assert await service.user_access_token(USER_KEY) is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": "invalid_client", "code": 20001}),
        httpx.Response(400, json={"code": 20026, "msg": "unclassified error"}),
        httpx.Response(200, json={"code": 0, "refresh_token": "ur-only"}),
        httpx.Response(200, text="not-json"),
    ],
    ids=["invalid-client", "unclassified-code", "missing-access-token", "invalid-json"],
)
async def test_non_grant_refresh_errors_are_not_reported_as_missing_consent(response):
    old = _expired_token()
    service = _memory_service(lambda request: response, old)
    with pytest.raises(LarkError):
        await service.user_access_token(USER_KEY)
    assert await service._repo.find_by_user_key(USER_KEY) == old
    assert not service._token_requests


async def test_invalid_grant_is_classified_without_parsing_error_description():
    service = _memory_service(
        lambda request: httpx.Response(
            400, json={"error": "invalid_client", "error_description": "not invalid_grant"},
        ),
        _expired_token(),
    )
    with pytest.raises(oauth.OAuthTokenError) as excinfo:
        await service.user_access_token(USER_KEY)
    assert excinfo.value.error == "invalid_client"


@pytest.mark.parametrize("failure", ["rate-limit", "network"])
async def test_transient_refresh_errors_preserve_consent(failure):
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectTimeout("test timeout", request=request)
        return httpx.Response(429, json={"error": "invalid_grant"})

    old = _expired_token()
    service = _memory_service(handler, old)
    with pytest.raises(TransientError):
        await service.user_access_token(USER_KEY)
    assert await service._repo.find_by_user_key(USER_KEY) == old


async def test_cancelling_a_reader_does_not_cancel_shared_refresh():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return httpx.Response(200, json=TOKEN_PAYLOAD)

    service = _memory_service(handler, _expired_token())
    reader = asyncio.create_task(service.user_access_token(USER_KEY))
    other = asyncio.create_task(service.user_access_token(USER_KEY))
    await asyncio.wait_for(entered.wait(), timeout=2)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    release.set()
    assert await other == "u-access"
    assert calls == 1
    assert not service._token_requests


@pytest.mark.parametrize("separate_service", [False, True])
@pytest.mark.parametrize("old_result", ["success", "invalid_grant", "transient"])
async def test_new_consent_supersedes_inflight_refresh(tmp_path, separate_service, old_result):
    entered, release = asyncio.Event(), asyncio.Event()
    refresh_calls = 0

    async def handler(request):
        nonlocal refresh_calls
        if request.url.path.endswith("/oauth/token"):
            if json.loads(request.content)["grant_type"] == "refresh_token":
                refresh_calls += 1
                entered.set()
                await release.wait()
                if old_result == "invalid_grant":
                    return httpx.Response(400, json={"error": "invalid_grant"})
                if old_result == "transient":
                    return httpx.Response(503)
                return httpx.Response(200, json=TOKEN_PAYLOAD)
            return httpx.Response(200, json={
                **TOKEN_PAYLOAD, "access_token": "new-consent", "refresh_token": "new-refresh",
            })
        return httpx.Response(200, json={"data": {"open_id": "ou_user"}})

    service = _service(tmp_path, handler)
    consent_service = _service(tmp_path, handler) if separate_service else service
    await service._repo.save(replace(_expired_token(), open_id="ou_user"))
    reader = asyncio.create_task(service.user_access_token(USER_KEY))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await consent_service.complete_authorization(
            code="new-code", state=consent_service.create_state(USER_KEY),
        )
        consent = await service._repo.find_by_user_key(USER_KEY)
        # Even a reader on the old service must not join the obsolete refresh.
        assert await asyncio.wait_for(service.user_access_token(USER_KEY), 2) == "new-consent"
    finally:
        release.set()
        result = await asyncio.gather(reader, return_exceptions=True)
    assert result == ["new-consent"]
    assert await service._repo.find_by_user_key(USER_KEY) == consent
    assert await service.authorization_available(USER_KEY) is True
    assert refresh_calls == 1
    assert not service._token_requests


@pytest.mark.parametrize("seconds", [-1, 0, 60, 120, 121])
async def test_authorization_status_and_reader_share_access_margin(monkeypatch, seconds):
    now = datetime(2030, 1, 1)
    monkeypatch.setattr(
        "conciliator.services.lark_user_token_service._utcnow_naive", lambda: now,
    )
    token = replace(
        _expired_token(), access_expires_at=now + timedelta(seconds=seconds), refresh_token="",
    )
    service = _memory_service(lambda request: pytest.fail("must not contact Feishu"), token)
    assert await service.authorization_available(USER_KEY) is (seconds > 120)
    assert await service.user_access_token(USER_KEY) == (token.access_token if seconds > 120 else None)
    assert await service._repo.find_by_user_key(USER_KEY) == token


async def test_invalid_grant_marks_only_current_credentials_unavailable(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "invalid_grant"})

    service = _service(tmp_path, handler)
    old = _expired_token()
    await service._repo.save(old)
    assert await service.authorization_available(USER_KEY) is True
    assert await service.user_access_token(USER_KEY) is None
    assert await service.authorization_available(USER_KEY) is False
    assert await service.user_access_token(USER_KEY) is None
    assert calls == 1
    stored = await service._repo.find_by_user_key(USER_KEY)
    assert stored is not None
    assert (stored.user_key, stored.open_id) == (old.user_key, old.open_id)
    assert stored.access_token == stored.refresh_token == ""
