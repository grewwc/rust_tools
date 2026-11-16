import asyncio
import hashlib
import hmac
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx

from conciliator.config import Settings
from conciliator.integrations.trace import traced_operation, traced_request
from conciliator.utils.logging import get_logger
from conciliator.utils.retry import TransientError, retry_async
from conciliator.utils.secrets import SecretsProvider

log = get_logger(__name__)

_RATE_LIMIT_RETRIES = 8
_CALL_TIMEOUT_SECONDS = 180.0


def _retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    if value.strip().isascii() and value.strip().isdigit():
        return float(int(value.strip()))
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        return max(0.0, (date - datetime.now(UTC)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return 0.0


class ZtiTokenProvider(Protocol):
    def get_token(self) -> str: ...


class SdkZtiTokenProvider:
    """Use the official ZTI SDK; the SDK caches and refreshes SVIDs for us."""

    def __init__(self) -> None:
        self._helper: Any | None = None

    def get_token(self) -> str:
        if self._helper is None:
            try:
                from ztijwthelper import ZTIJwtHelper
            except ImportError as exc:  # pragma: no cover - internal SDK in TCE
                raise RuntimeError(
                    "bytedztijwthelper is required for NextCode App OpenAPI calls"
                ) from exc
            self._helper = ZTIJwtHelper()
        return str(self._helper.get_jwt_svid())


@dataclass(frozen=True)
class NextCodeError(Exception):
    status: int
    action: str
    message: str
    code: str = ""
    request_id: str = ""
    retry_after: float = 0.0

    def __post_init__(self) -> None:
        detail = f"NextCode {self.action} failed ({self.status})"
        if self.code:
            detail += f" [{self.code}]"
        detail += f": {self.message[:300]}"
        if self.request_id:
            detail += f" (request_id={self.request_id})"
        super().__init__(detail)


class NextCodeClient:
    """Client for the App OpenAPI contract documented at codebase-api /v2/."""

    def __init__(
        self,
        settings: Settings,
        secrets: SecretsProvider,
        *,
        zti_provider: ZtiTokenProvider | None = None,
    ):
        self._settings = settings
        self._base = settings.nextcode_api_base.rstrip("/") + "/"
        self._app_id = settings.nextcode_app_id
        self._secret = secrets.get(
            settings.kms_secret_nextcode_app_secret_alias
        )
        self._webhook_secret = secrets.get(
            settings.kms_secret_nextcode_webhook_secret_alias
        )
        self._zti = zti_provider or SdkZtiTokenProvider()
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def verify_webhook_signature(self, body: bytes, signature: str) -> bool:
        """Verify `sha256=<hex HMAC-SHA256(secret, raw body)>`."""
        expected = "sha256=" + hmac.new(
            self._webhook_secret.encode(), body, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    @traced_operation
    async def call(
        self,
        action: str,
        body: dict[str, Any],
        *,
        read_only: bool = False,
    ) -> dict[str, Any]:
        """Call one App OpenAPI Action.

        Retry explicit rate-limit rejections for all actions. Only read-only calls
        retry ambiguous network/server failures: writes have no idempotency key.
        The deadline and rate-limit retry budget cover the whole logical call.
        """
        request = self._call_read if read_only else self._call_once
        last_rate_limit: NextCodeError | None = None
        deadline = asyncio.get_running_loop().time() + _CALL_TIMEOUT_SECONDS
        try:
            async with asyncio.timeout(_CALL_TIMEOUT_SECONDS):
                for retry in range(_RATE_LIMIT_RETRIES + 1):
                    try:
                        result = await request(action, body)
                    except NextCodeError as exc:
                        if exc.status != 429 and exc.code != "TooManyRequests":
                            raise
                        last_rate_limit = exc
                        base = min(2 ** retry, 16)
                        delay = max(random.uniform(base, base * 2), exc.retry_after)
                        if retry == _RATE_LIMIT_RETRIES or (
                            asyncio.get_running_loop().time() + delay >= deadline
                        ):
                            log.warning(
                                "nextcode_rate_limit_exhausted", action=action,
                                retries=retry, request_id=exc.request_id,
                            )
                            raise
                        log.warning(
                            "nextcode_rate_limit_retry", action=action,
                            retry=retry + 1, delay_seconds=delay, request_id=exc.request_id,
                        )
                        await asyncio.sleep(delay)
                    else:
                        if retry:
                            log.info("nextcode_rate_limit_recovered", action=action, retries=retry)
                        return result
        except TimeoutError:
            log.warning("nextcode_call_deadline_exceeded", action=action)
            if last_rate_limit is not None:
                raise last_rate_limit from None
            raise
        raise RuntimeError("unreachable")  # pragma: no cover

    @retry_async("nextcode_request", exceptions=(TransientError,))
    async def _call_read(
        self, action: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._call_once(action, body)

    async def _call_once(
        self, action: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            response = await traced_request(self._http, "nextcode", action, "POST",
                self._base,
                params={"Action": action},
                json=body,
                headers=self._auth_headers(),
            )
        except httpx.RequestError as exc:
            raise TransientError(f"NextCode network error: {exc}") from exc

        if response.status_code >= 500:
            raise TransientError(
                f"NextCode server error {response.status_code}: {response.text[:300]}"
            )

        try:
            envelope = response.json()
        except ValueError as exc:
            raise NextCodeError(
                status=response.status_code,
                action=action,
                message=response.text,
                retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
            ) from exc

        # A gateway may return a JSON scalar/list rather than an OpenAPI envelope.
        if not isinstance(envelope, dict):
            raise NextCodeError(
                status=response.status_code, action=action, message=response.text,
                retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
            )
        metadata = envelope.get("ResponseMetadata") or {}
        request_id = str(metadata.get("RequestId") or "")
        error = metadata.get("Error") or {}
        if response.status_code >= 400 or error:
            raise NextCodeError(
                status=response.status_code,
                action=action,
                code=str(error.get("Code") or ""),
                message=str(error.get("Message") or response.text),
                request_id=request_id,
                retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
            )

        result = envelope.get("Result")
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise NextCodeError(
                status=response.status_code,
                action=action,
                message="response Result is not an object",
                request_id=request_id,
            )
        return result

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Code-App-ID": self._app_id,
            "X-Code-App-Secret": self._secret,
        }
        try:
            token = self._zti.get_token()
        except Exception:
            if self._settings.nextcode_zti_required:
                raise
            log.warning("nextcode_zti_token_unavailable")
        else:
            if token:
                headers["X-Code-ZTI-Token"] = token
        return headers
