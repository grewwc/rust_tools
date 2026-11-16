"""JWT transport for ByteCloud application identity.

The official exchange provider owns BC1 signing and token refresh. AK/SK must
never be attached to business requests or copied to deployment records.
"""

import asyncio

import httpx

from conciliator.config import Settings
from conciliator.integrations.maat.client import MaatError
from conciliator.utils.secrets import SecretsProvider, get_bytecloud_credentials


class SdkMaatTokenProvider:
    """Reuse the official CN SDK's JWT cache and concurrent refresh handling."""

    def __init__(self, settings: Settings, secrets: SecretsProvider, *, proxy_url=None):
        credentials = get_bytecloud_credentials(settings, secrets)
        try:
            from bytecloud_cloud_sdk import Credential, JWTClient

            self._credential = Credential(
                credentials.ak.get_secret_value(), credentials.sk.get_secret_value(),
            )
            self._client = JWTClient(region="cn", timeout=10, proxy_url=proxy_url)
        except Exception:
            raise MaatError("MAAT application SDK initialization failed") from None

    async def __call__(self):
        try:
            return await asyncio.to_thread(self._client.generate_with_aksk, self._credential, "cn")
        except Exception:
            raise MaatError("MAAT application JWT acquisition failed") from None

    async def operator(self):
        """Resolve the application's real identity; never substitute a personal operator."""
        token = await self()
        try:
            info = await asyncio.to_thread(self._client.userinfo, token)
            operator = info.username
        except Exception:
            raise MaatError("MAAT application identity lookup failed") from None
        if not isinstance(operator, str) or not operator or any(char.isspace() for char in operator):
            raise MaatError("MAAT application identity has no usable operator")
        return operator


async def application_token(token_provider):
    try:
        token = await token_provider()
    except Exception:
        # SDK exceptions can embed signed requests or credential material.
        raise MaatError("MAAT application JWT acquisition failed") from None
    if not isinstance(token, str) or not token or any(not 33 <= ord(char) <= 126 for char in token):
        raise MaatError("MAAT application JWT is invalid")
    return token


class MaatJWTAuth(httpx.Auth):
    """Use a refresh-capable provider, bound to one explicitly trusted HTTPS origin.

    Use application_http_client to also reject redirects before HTTPX follows
    them. A rejected write is never automatically retried on auth failure.
    """

    def __init__(self, token_provider, origin):
        url = httpx.URL(origin)
        if url.scheme != "https" or not url.host or url.userinfo or url.query or url.fragment or url.path != "/":
            raise ValueError("MAAT JWT requires one HTTPS origin")
        self._origin = (url.scheme, url.host, url.port)
        self._token_provider = token_provider

    async def async_auth_flow(self, request):
        if (request.url.scheme, request.url.host, request.url.port) != self._origin or request.url.userinfo:
            raise MaatError("MAAT JWT request origin does not match")
        request.headers["X-Jwt-Token"] = await application_token(self._token_provider)
        yield request


async def _reject_redirect(response):
    if response.is_redirect:
        raise MaatError("MAAT authenticated redirects are not supported")


def application_http_client(token_provider, origin, *, transport=None, trust_env=True):
    """Keep application JWTs on the configured origin, including console calls.

    Console permission must be verified separately; this factory does not grant
    it or acquire a personal session as a fallback.
    """
    return httpx.AsyncClient(
        base_url=origin,
        auth=MaatJWTAuth(token_provider, origin),
        timeout=httpx.Timeout(30, connect=10),
        follow_redirects=False,
        event_hooks={"response": [_reject_redirect]},
        transport=transport,
        trust_env=trust_env,
    )
