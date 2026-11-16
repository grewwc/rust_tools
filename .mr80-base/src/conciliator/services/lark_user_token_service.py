"""Per-user Feishu tokens: consent link, callback handling, refresh, and lookup.

The service stores tokens keyed by Meegle user_key, because that is the only
identifier a task carries and it is known at authorization time (it travels in
the OAuth `state`), so no extra identity lookup is needed on the read path.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets as random_secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy.exc import SQLAlchemyError

from conciliator.config import Settings
from conciliator.integrations.lark import oauth
from conciliator.repositories.lark_user_token_repo import (
    LarkUserTokenRepository,
    StoredUserToken,
)
from conciliator.utils.logging import get_logger
from conciliator.utils.retry import TransientError
from conciliator.utils.secrets import SecretsProvider

log = get_logger(__name__)

# Consent state is short-lived; 15 minutes covers a slow user.
_STATE_MAX_AGE_SECONDS = 900
# Refresh slightly before real expiry so an in-flight read never sees an expired token.
_ACCESS_TOKEN_MARGIN_SECONDS = 120


class InvalidOAuthState(ValueError):
    """The OAuth state is malformed, tampered with, or expired."""


class OAuthIdentityMismatch(ValueError):
    """The consenting Feishu user does not match the requested Meegle identity."""


def _utcnow_naive() -> datetime:
    """Naive UTC, matching what MySQL DATETIME columns return on this database."""
    return datetime.now(UTC).replace(tzinfo=None)


class LarkUserTokenService:
    """Owns user authorization and the resulting tokens."""

    def __init__(
        self,
        *,
        settings: Settings,
        secrets: SecretsProvider,
        repo: LarkUserTokenRepository,
        http: httpx.AsyncClient | None = None,
        identity_resolver: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        self._repo = repo
        self._http = http
        self._app_id = settings.lark_app_id
        self._app_secret = secrets.get(settings.kms_secret_lark_app_secret_alias)
        self._redirect_uri = settings.lark_oauth_redirect_uri
        self._identity_resolver = identity_resolver
        self._token_requests: dict[StoredUserToken, asyncio.Task[None]] = {}

    @property
    def enabled(self) -> bool:
        """User-identity reading is opt-in: it needs a registered redirect URI."""
        return bool(self._redirect_uri.strip())

    def _signature(self, user_key: str, issued_at: int, nonce: str) -> str:
        message = f"{user_key}:{issued_at}:{nonce}".encode()
        return hmac.new(self._app_secret.encode(), message, hashlib.sha256).hexdigest()[:32]

    def create_state(self, user_key: str) -> str:
        issued_at = int(time.time())
        nonce = random_secrets.token_urlsafe(24)
        return f"{user_key}.{issued_at}.{nonce}.{self._signature(user_key, issued_at, nonce)}"

    def parse_state(self, state: str) -> str:
        """Return the user_key carried by a state we signed."""
        parts = state.rsplit(".", 3)
        if len(parts) != 4:
            raise InvalidOAuthState("OAuth state is malformed")
        user_key, issued_raw, nonce, signature = parts
        try:
            issued_at = int(issued_raw)
        except ValueError as exc:
            raise InvalidOAuthState("OAuth state is malformed") from exc
        if not user_key or not nonce or not hmac.compare_digest(
            self._signature(user_key, issued_at, nonce).encode(), signature.encode()
        ):
            raise InvalidOAuthState("OAuth state signature mismatch")
        if not 0 <= time.time() - issued_at <= _STATE_MAX_AGE_SECONDS:
            raise InvalidOAuthState("OAuth state has expired")
        return user_key

    def authorize_url(self, user_key: str) -> str:
        if not self.enabled:
            raise RuntimeError("lark_oauth_redirect_uri is not configured")
        return oauth.build_authorize_url(
            app_id=self._app_id,
            redirect_uri=self._redirect_uri,
            state=self.create_state(user_key),
        )

    async def complete_authorization(self, *, code: str, state: str) -> str:
        """Store consent only after verifying the server-side user mapping."""
        user_key = self.parse_state(state)
        if self._identity_resolver is None:
            raise OAuthIdentityMismatch("无法核验飞书账号身份，请联系管理员。")
        expected_open_id = await self._identity_resolver(user_key)
        if not expected_open_id:
            raise OAuthIdentityMismatch("无法核验飞书账号身份，请联系管理员。")
        bundle = await oauth.exchange_code(
            app_id=self._app_id,
            app_secret=self._app_secret,
            code=code,
            redirect_uri=self._redirect_uri,
            http=self._http,
        )
        open_id = await oauth.fetch_user_open_id(bundle.access_token, http=self._http)
        if not hmac.compare_digest(open_id, expected_open_id):
            raise OAuthIdentityMismatch("授权的飞书账号与任务操作者不一致，请使用本人账号。")
        await self._store(user_key=user_key, open_id=open_id, bundle=bundle)
        log.info("lark_user_authorized", user_key=user_key)
        return open_id

    async def _store(
        self, *, user_key: str, open_id: str, bundle: oauth.UserTokenBundle,
        refreshed_from: StoredUserToken | None = None,
    ) -> None:
        now = _utcnow_naive()
        refresh_token = bundle.refresh_token
        refresh_expires_at = (
            now + timedelta(seconds=bundle.refresh_expires_in)
            if bundle.refresh_expires_in
            else None
        )
        # Only refresh responses inherit credentials; a new consent replaces them.
        if not refresh_token and refreshed_from is not None:
            refresh_token = refreshed_from.refresh_token
            refresh_expires_at = refreshed_from.refresh_expires_at
        token = StoredUserToken(
            open_id=open_id,
            user_key=user_key,
            refresh_token=refresh_token,
            access_token=bundle.access_token,
            access_expires_at=now + timedelta(seconds=bundle.expires_in),
            refresh_expires_at=refresh_expires_at,
        )
        try:
            if refreshed_from is None:
                await self._repo.save(token)
            else:
                await self._repo.replace_if_current(refreshed_from, token)
        except SQLAlchemyError as exc:
            raise TransientError("Feishu user token store is unavailable") from exc

    @staticmethod
    def _access_usable(stored: StoredUserToken, now: datetime) -> bool:
        return bool(
            stored.access_token and stored.access_expires_at
            and stored.access_expires_at > now + timedelta(seconds=_ACCESS_TOKEN_MARGIN_SECONDS)
        )

    @staticmethod
    def _refresh_usable(stored: StoredUserToken, now: datetime) -> bool:
        return bool(stored.refresh_token and (
            stored.refresh_expires_at is None or stored.refresh_expires_at > now
        ))

    async def authorization_available(self, user_key: str) -> bool:
        """Inspect stored validity without refreshing; server revocation is unknown."""
        if not user_key:
            return False
        try:
            stored = await self._repo.find_by_user_key(user_key)
        except SQLAlchemyError as exc:
            raise TransientError("Feishu user token store is unavailable") from exc
        if stored is None:
            return False
        now = _utcnow_naive()
        return self._access_usable(stored, now) or self._refresh_usable(stored, now)

    async def user_access_token(self, user_key: str) -> str | None:
        """Return a usable user token, refreshing it when needed.

        Returns None when the user never authorized or the refresh token is spent.
        Temporary storage or Feishu failures are surfaced so callers do not
        misreport a short outage as expired user consent.
        """
        if not user_key:
            return None
        # Re-read after every refresh: another service may have replaced the
        # credentials while the request was in flight. Bound retries under churn.
        for attempt in range(3):
            stored = await self._find_token(user_key)
            if stored is None:
                return None
            now = _utcnow_naive()
            if self._access_usable(stored, now):
                return stored.access_token
            if not self._refresh_usable(stored, now):
                return None
            if attempt == 2:
                break
            try:
                await self._share_refresh(stored)
            except (TransientError, oauth.LarkError):
                # An old request's failure must not mask a newer authorization.
                if await self._find_token(user_key) == stored:
                    raise
        raise TransientError("Feishu user credentials changed during refresh; retry")

    async def _share_refresh(self, stored: StoredUserToken) -> None:
        # Coalesce only readers of the same credentials, not new consent for
        # the same user. The snapshot is immutable and removed on completion.
        task = self._token_requests.get(stored)
        if task is None:
            task = asyncio.create_task(self._refresh_user_access_token(stored))
            self._token_requests[stored] = task

            def finished(done: asyncio.Task[None]) -> None:
                if self._token_requests.get(stored) is done:
                    self._token_requests.pop(stored)
                # Observe failures even if every waiting document read was cancelled.
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished)
        # A cancelled document read must not abort refresh for the other readers.
        try:
            await asyncio.shield(task)
        except SQLAlchemyError as exc:
            raise TransientError("Feishu user token store is unavailable") from exc

    async def _find_token(self, user_key: str) -> StoredUserToken | None:
        try:
            return await self._repo.find_by_user_key(user_key)
        except SQLAlchemyError as exc:
            log.warning(
                "lark_user_token_lookup_failed",
                user_key=user_key,
                error_type=type(exc).__name__,
            )
            raise TransientError("Feishu user token store is unavailable") from exc

    async def _refresh_user_access_token(self, stored: StoredUserToken) -> None:
        user_key = stored.user_key
        try:
            bundle = await oauth.refresh_user_token(
                app_id=self._app_id,
                app_secret=self._app_secret,
                refresh_token=stored.refresh_token,
                http=self._http,
            )
        except TransientError as exc:
            log.warning(
                "lark_user_token_refresh_failed",
                user_key=user_key,
                error_type=type(exc).__name__,
            )
            raise
        except oauth.OAuthTokenError as exc:
            if exc.error != "invalid_grant":
                raise
            log.warning(
                "lark_user_token_refresh_rejected",
                user_key=user_key,
                error_type=type(exc).__name__,
            )
            # Keep the identity binding, but never invalidate newer consent.
            await self._repo.replace_if_current(stored, replace(
                stored, access_token="", refresh_token="",
                access_expires_at=None, refresh_expires_at=None,
            ))
            return
        await self._store(
            user_key=user_key, open_id=stored.open_id, bundle=bundle,
            refreshed_from=stored,
        )
