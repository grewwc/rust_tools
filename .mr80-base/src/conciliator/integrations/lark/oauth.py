"""Feishu user OAuth: consent link, code exchange, refresh, and user identity.

Calling Feishu with a user_access_token means the service reads documents with the
same permissions as the person who triggered the task, so no document has to be
shared with the app individually.

Endpoint contract verified against the live tenant:
- `POST /authen/v2/oauth/token` accepts `grant_type` + `client_id` + `client_secret`
  + `code` + `redirect_uri`, and returns the token fields at the top level.
- `POST /authen/v2/oauth/token` with `grant_type=refresh_token` accepts `refresh_token`.
- `GET /authen/v1/user_info` takes the user token as a bearer token.

Verified against the live tenant: the consent link has to use `/authen/v1/authorize`
with `client_id`. The older `/authen/v1/index` carried no scope list, so every token came
back without a refresh_token and user tokens lapsed after two hours.

Not yet verified: whether a completed authorization really returns a refresh_token.
Confirm it by reading `refresh_token` back from the token store after consent.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import httpx

from conciliator.integrations.lark.client import LarkError
from conciliator.utils.retry import TransientError

OPEN_API_BASE = "https://open.feishu.cn/open-apis"
AUTHORIZE_PATH = "/authen/v1/authorize"
TOKEN_PATH = "/authen/v2/oauth/token"
USER_INFO_PATH = "/authen/v1/user_info"

# Scopes to request on the consent link, space separated. `offline_access` is what makes
# Feishu hand back a refresh_token; the document scopes are listed as well because this
# endpoint grants exactly what is asked for, and a user token is only ever used to read
# requirement documents.
AUTHORIZE_SCOPE = "offline_access docx:document:readonly wiki:node:read"


def build_authorize_url(
    *,
    app_id: str,
    redirect_uri: str,
    state: str,
    scope: str = AUTHORIZE_SCOPE,
) -> str:
    """Consent link the user opens; Feishu redirects back to redirect_uri with a code.

    Values are percent-encoded here rather than by `httpx.QueryParams`, which turns the
    spaces between scopes into `+`; the consent endpoint expects `%20`.
    """
    query = "&".join(
        (
            f"client_id={quote(app_id, safe='')}",
            f"redirect_uri={quote(redirect_uri, safe='')}",
            f"state={quote(state, safe='')}",
            f"scope={quote(scope, safe='')}",
        )
    )
    return f"{OPEN_API_BASE}{AUTHORIZE_PATH}?{query}"


@dataclass(frozen=True)
class UserTokenBundle:
    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int


def _redact(payload: object) -> object:
    """Mask token-like values so diagnostics can include the raw response safely."""
    if isinstance(payload, dict):
        return {
            key: ("***" if "token" in str(key).lower() else _redact(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_redact(item) for item in payload]
    return payload


class OAuthTokenError(LarkError):
    """A token endpoint rejection with its machine-readable OAuth error."""

    def __init__(
        self, message: str, *, error: str,
        http_status: int | None = None, code: int | None = None,
    ) -> None:
        super().__init__(message, http_status=http_status, code=code)
        self.error = error


def parse_token_bundle(payload: dict) -> UserTokenBundle:
    """Read a token response; v2 returns fields at the top level, older shapes nest them.

    The live v2 endpoint issues a valid access_token without a refresh_token for this
    app, so refresh_token is optional here. Callers refresh only when one was issued;
    otherwise the token just expires and the user re-authorizes (one click).
    """
    nested = payload.get("data")
    data = nested if isinstance(nested, dict) else payload
    access_token = str(data.get("access_token") or "")
    if not access_token:
        raise LarkError(
            "Feishu OAuth token response is missing access_token; "
            f"got {_redact(payload)!r}"
        )
    return UserTokenBundle(
        access_token=access_token,
        refresh_token=str(data.get("refresh_token") or ""),
        expires_in=int(data.get("expires_in") or 7200),
        refresh_expires_in=int(data.get("refresh_token_expires_in") or 0),
    )


async def _post_token(
    payload: dict[str, str],
    *,
    http: httpx.AsyncClient | None,
    base_url: str,
) -> dict:
    client = http or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        try:
            resp = await client.post(f"{base_url}{TOKEN_PATH}", json=payload)
        except httpx.RequestError as exc:
            raise TransientError("Feishu OAuth token request failed") from exc
    finally:
        if http is None:
            await client.aclose()
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientError(
            f"Feishu OAuth token service unavailable (HTTP {resp.status_code})"
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise LarkError(
            f"Feishu OAuth token returned invalid JSON (HTTP {resp.status_code})"
        ) from exc
    error = data.get("error")
    code = int(data.get("code") or 0)
    if resp.status_code >= 400 or error or code != 0:
        raise OAuthTokenError(
            "Feishu OAuth token request failed "
            f"(HTTP {resp.status_code}, error={error or code}, "
            f"description={data.get('error_description') or data.get('msg', '')})",
            error=str(error or ""),
            http_status=resp.status_code,
            code=code or None,
        )
    return data


async def exchange_code(
    *,
    app_id: str,
    app_secret: str,
    code: str,
    redirect_uri: str,
    http: httpx.AsyncClient | None = None,
    base_url: str = OPEN_API_BASE,
) -> UserTokenBundle:
    """Trade the one-time authorization code from the callback for user tokens."""
    payload = await _post_token(
        {
            "grant_type": "authorization_code",
            "client_id": app_id,
            "client_secret": app_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        },
        http=http,
        base_url=base_url,
    )
    return parse_token_bundle(payload)


async def refresh_user_token(
    *,
    app_id: str,
    app_secret: str,
    refresh_token: str,
    http: httpx.AsyncClient | None = None,
    base_url: str = OPEN_API_BASE,
) -> UserTokenBundle:
    """Refresh an expired user token; preserve any refresh token returned by Feishu."""
    payload = await _post_token(
        {
            "grant_type": "refresh_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "refresh_token": refresh_token,
        },
        http=http,
        base_url=base_url,
    )
    return parse_token_bundle(payload)


async def fetch_user_open_id(
    user_access_token: str,
    *,
    http: httpx.AsyncClient | None = None,
    base_url: str = OPEN_API_BASE,
) -> str:
    """Resolve the open_id that owns this user token."""
    client = http or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        try:
            resp = await client.get(
                f"{base_url}{USER_INFO_PATH}",
                headers={"Authorization": f"Bearer {user_access_token}"},
            )
        except httpx.RequestError as exc:
            raise TransientError("Feishu user info request failed") from exc
    finally:
        if http is None:
            await client.aclose()
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientError(
            f"Feishu user info service unavailable (HTTP {resp.status_code})"
        )
    try:
        payload = resp.json()
    except ValueError as exc:
        raise LarkError(
            f"Feishu user info returned invalid JSON (HTTP {resp.status_code})"
        ) from exc
    code = int(payload.get("code") or 0)
    if resp.status_code >= 400 or code != 0:
        raise LarkError(
            f"Feishu user info failed (HTTP {resp.status_code}, code={code}, "
            f"msg={payload.get('msg', '')})",
            http_status=resp.status_code,
            code=code,
        )
    data = payload.get("data") or {}
    open_id = str(data.get("open_id") or "")
    if not open_id:
        raise LarkError("Feishu user info response has no open_id")
    return open_id
