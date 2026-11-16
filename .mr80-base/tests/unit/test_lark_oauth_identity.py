"""OAuth identity binding: never persist consent for a different Meegle user."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from conciliator.integrations.lark import oauth
from conciliator.integrations.lark.contacts import LarkUserNotFoundError
from conciliator.main import _resolve_lark_identity
from conciliator.services.lark_user_token_service import (
    InvalidOAuthState,
    LarkUserTokenService,
    OAuthIdentityMismatch,
)
from conciliator.utils.retry import TransientError


class _Secrets:
    def get(self, alias):
        return "test-only-secret"


def _service(repo, resolver):
    return LarkUserTokenService(
        settings=SimpleNamespace(
            lark_app_id="cli-test", kms_secret_lark_app_secret_alias="lark",
            lark_oauth_redirect_uri="https://test/oauth/lark/callback",
        ),
        secrets=_Secrets(), repo=repo, identity_resolver=resolver,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("consenting_id", ["ou-owner", "ou-attacker"])
async def test_consent_only_persists_for_server_resolved_identity(monkeypatch, consenting_id):
    repo = SimpleNamespace(save=AsyncMock())
    resolver = AsyncMock(return_value="ou-owner")
    service = _service(repo, resolver)
    exchange = AsyncMock(return_value=oauth.UserTokenBundle(
        access_token="access-test", refresh_token="refresh-test", expires_in=7200,
        refresh_expires_in=2592000,
    ))
    monkeypatch.setattr(oauth, "exchange_code", exchange)
    monkeypatch.setattr(oauth, "fetch_user_open_id", AsyncMock(return_value=consenting_id))
    state = service.create_state("meegle-owner")
    if consenting_id == "ou-owner":
        assert await service.complete_authorization(code="test-code", state=state) == consenting_id
        saved = repo.save.await_args.args[0]
        assert saved.user_key == "meegle-owner"
        assert saved.open_id == "ou-owner"
    else:
        with pytest.raises(OAuthIdentityMismatch):
            await service.complete_authorization(code="test-code", state=state)
        repo.save.assert_not_awaited()
    resolver.assert_awaited_once_with("meegle-owner")


@pytest.mark.asyncio
@pytest.mark.parametrize("resolver", [None, AsyncMock(return_value="")])
async def test_missing_identity_mapping_fails_closed_before_exchange(monkeypatch, resolver):
    repo = SimpleNamespace(save=AsyncMock())
    service = _service(repo, resolver)
    exchange = AsyncMock()
    monkeypatch.setattr(oauth, "exchange_code", exchange)
    with pytest.raises(OAuthIdentityMismatch):
        await service.complete_authorization(code="test-code", state=service.create_state("owner"))
    exchange.assert_not_awaited()
    repo.save.assert_not_awaited()


def test_state_is_random_per_consent_and_rejects_future_timestamp(monkeypatch):
    service = _service(SimpleNamespace(), None)
    monkeypatch.setattr("conciliator.services.lark_user_token_service.time.time", lambda: 1000)
    first, second = service.create_state("user.with.dots"), service.create_state("user.with.dots")
    assert first != second
    assert service.parse_state(first) == "user.with.dots"
    monkeypatch.setattr("conciliator.services.lark_user_token_service.time.time", lambda: 0)
    with pytest.raises(InvalidOAuthState):
        service.parse_state(first)


@pytest.mark.asyncio
async def test_identity_lookup_uses_server_email_not_browser_supplied_identity(monkeypatch):
    meegle = SimpleNamespace(get_user_email=AsyncMock(return_value="owner@example.com"))
    lark = object()
    lookup = AsyncMock(return_value="ou-owner")
    monkeypatch.setattr("conciliator.main.get_user_open_id_by_email", lookup)
    assert await _resolve_lark_identity(meegle, lark, "owner") == "ou-owner"
    meegle.get_user_email.assert_awaited_once_with("owner")
    lookup.assert_awaited_once_with(lark, "owner@example.com")


@pytest.mark.asyncio
@pytest.mark.parametrize("email", [None, ""])
async def test_identity_lookup_does_not_guess_missing_email(monkeypatch, email):
    meegle = SimpleNamespace(get_user_email=AsyncMock(return_value=email))
    lookup = AsyncMock()
    monkeypatch.setattr("conciliator.main.get_user_open_id_by_email", lookup)
    with pytest.raises(OAuthIdentityMismatch):
        await _resolve_lark_identity(meegle, object(), "owner")
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_identity_lookup_surfaces_temporary_meegle_failure():
    meegle = SimpleNamespace(get_user_email=AsyncMock(side_effect=httpx.ConnectError("offline")))
    with pytest.raises(TransientError):
        await _resolve_lark_identity(meegle, object(), "owner")


@pytest.mark.asyncio
async def test_identity_lookup_rejects_unmapped_lark_email(monkeypatch):
    meegle = SimpleNamespace(get_user_email=AsyncMock(return_value="owner@example.com"))
    monkeypatch.setattr("conciliator.main.get_user_open_id_by_email", AsyncMock(
        side_effect=LarkUserNotFoundError("not found"),
    ))
    with pytest.raises(OAuthIdentityMismatch):
        await _resolve_lark_identity(meegle, object(), "owner")
