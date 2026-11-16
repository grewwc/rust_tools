import asyncio
import time
from typing import Any

import httpx

from conciliator.config import Settings
from conciliator.integrations.trace import traced_operation, traced_request
from conciliator.utils.logging import get_logger
from conciliator.utils.retry import TransientError, retry_async
from conciliator.utils.secrets import SecretsProvider

log = get_logger(__name__)


class LarkError(Exception):
    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        code: int | None = None,
    ):
        super().__init__(message)
        self.http_status = http_status
        self.code = code


_TRANSIENT_API_CODES = {
    230020,  # 当前操作触发频率限制
    230049,  # 消息正在发送中
}


class LarkClient:
    def __init__(
        self,
        settings: Settings,
        secrets: SecretsProvider | None,
        *,
        tenant_access_token: str | None = None,
    ):
        self._settings = settings
        self._app_id = settings.lark_app_id
        self._app_secret: str | None
        if tenant_access_token is None:
            if secrets is None:
                raise ValueError(
                    "secrets is required without an injected tenant access token"
                )
            self._app_secret = secrets.get(
                settings.kms_secret_lark_app_secret_alias
            )
        else:
            if not tenant_access_token.strip():
                raise ValueError("tenant_access_token must not be empty")
            self._app_secret = None
        self._http = httpx.AsyncClient(
            base_url="https://open.feishu.cn/open-apis",
            timeout=httpx.Timeout(30.0),
        )
        self._token_cache: tuple[str, float] | None = (
            (tenant_access_token, float("inf"))
            if tenant_access_token is not None
            else None
        )
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_tenant_access_token(self) -> str:
        if self._token_cache and self._token_cache[1] > time.time() + 60:
            return self._token_cache[0]
        async with self._token_lock:
            if self._token_cache and self._token_cache[1] > time.time() + 60:
                return self._token_cache[0]
            if self._app_secret is None:  # pragma: no cover - guarded by cache
                raise LarkError("injected Lark tenant token is unavailable")
            try:
                resp = await traced_request(self._http, "lark", "tenant_access_token", "POST",
                    "/auth/v3/tenant_access_token/internal",
                    json={"app_id": self._app_id, "app_secret": self._app_secret},
                )
            except httpx.RequestError as exc:
                raise TransientError("Lark tenant token request failed") from exc
            try:
                data = resp.json()
            except ValueError as exc:
                raise LarkError(
                    f"Lark tenant token returned invalid JSON (HTTP {resp.status_code})"
                ) from exc
            if resp.status_code >= 500:
                raise TransientError(
                    f"Lark tenant token server error {resp.status_code}"
                )
            if resp.status_code >= 400 or data.get("code") != 0:
                raise LarkError(
                    "Lark tenant token failed "
                    f"(HTTP {resp.status_code}, code={data.get('code')}, "
                    f"msg={data.get('msg', '')})"
                )
            token = str(data.get("tenant_access_token") or "")
            if not token:
                raise LarkError("Lark tenant token response has no token")
            expire = int(data.get("expire", 7200))
            self._token_cache = (token, time.time() + expire)
            return token

    async def request(
        self,
        method: str,
        path: str,
        *,
        access_token: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send one authorized request without retrying ambiguous writes.

        `access_token` overrides the tenant token, which is how a single
        request is sent with the identity of one Feishu user.
        """
        token = access_token or await self.get_tenant_access_token()
        headers = {**kwargs.pop("headers", {}), "Authorization": f"Bearer {token}"}
        try:
            resp = await traced_request(self._http, "lark", path,
                method,
                path,
                headers=headers,
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise TransientError("Lark request outcome is unknown") from exc
        if resp.status_code >= 500:
            raise TransientError(f"Lark server error {resp.status_code}")
        return resp

    @traced_operation
    @retry_async("lark_request", exceptions=(TransientError,))
    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        access_token: str | None = None,
    ) -> dict:
        resp = await self.request(
            method, path, json=json, params=params, access_token=access_token
        )
        try:
            data = resp.json()
        except ValueError as exc:
            raise LarkError(
                f"Lark API {path} returned invalid JSON (HTTP {resp.status_code})"
            ) from exc
        code = int(data.get("code") or 0)
        if resp.status_code == 429 or code in _TRANSIENT_API_CODES:
            raise TransientError(
                f"Lark API {path} returned a transient response "
                f"(HTTP {resp.status_code}, code={code})"
            )
        if resp.status_code >= 400 or code != 0:
            raise LarkError(
                f"Lark API {path} failed "
                f"(HTTP {resp.status_code}, code={code}, "
                f"msg={data.get('msg', '')})",
                http_status=resp.status_code,
                code=code,
            )
        return data.get("data", {})
