from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import error_fields

log = get_logger(__name__)


@dataclass(frozen=True)
class AINodeRef:
    project_key: str
    work_item_id: int
    node_uuid: str
    user_key: str


class MeegleAINodeClient:
    def __init__(
        self,
        *,
        site_url: str,
        plugin_id: str,
        plugin_secret: str,
    ) -> None:
        if not all((site_url.strip(), plugin_id.strip(), plugin_secret.strip())):
            raise ValueError("site_url, plugin_id, and plugin_secret are required")
        self._plugin_id = plugin_id
        self._plugin_secret = plugin_secret
        self._http = httpx.AsyncClient(
            base_url=site_url.rstrip("/"), timeout=httpx.Timeout(30.0)
        )
        self._token: tuple[str, float] | None = None
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_plugin_token(self) -> str:
        if self._token and self._token[1] > time.time() + 60:
            return self._token[0]
        async with self._token_lock:
            if self._token and self._token[1] > time.time() + 60:
                return self._token[0]
            # source: lpm ai docs "/open_api/authen/plugin_access_token"
            response = await self._post(
                "/open_api/authen/plugin_token",
                json={
                    "plugin_id": self._plugin_id,
                    "plugin_secret": self._plugin_secret,
                    "type": 0,
                },
            )
            data = self._decode(response, "plugin token")
            token = str(data.get("token") or "")
            if not token:
                raise RuntimeError("Meegle plugin token response has no token")
            expire_time = int(data.get("expire_time") or 7200)
            self._token = (token, time.time() + expire_time)
            return token

    async def query(self, ref: AINodeRef) -> dict[str, Any]:
        # source: lpm ai docs "/open_api/ai_node/query"
        return await self._request("/open_api/ai_node/query", ref)

    async def get_node_owner_keys(self, ref: AINodeRef) -> list[str]:
        # source: 获取 AI 节点信息 https://meego-hc.larkoffice.com/b/helpcenter/1p8d7djs/pskw0p8q
        # Verified on the test space: node_fields.owner is a JSON-encoded multi-user list.
        data = await self.query(ref)
        field = next((item for item in data.get("node_fields", []) if item.get("field_key") == "owner"), None)
        if field is None:
            return []
        value = field.get("value")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError("invalid AI node owner field") from None
        if not isinstance(value, list) or any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError("AI node owner field must contain user keys")
        return list(dict.fromkeys(value))

    async def query_users(self, ref: AINodeRef, user_keys: list[str]) -> list[dict[str, Any]]:
        return await self._query_users(ref.user_key, user_keys)

    async def get_work_item_name(self, ref: AINodeRef, work_item_type_key: str) -> str:
        # source: lpm ai docs "获取工作项详情 work_item_ids 请求示例 返回 name"
        # https://meego-hc.larkoffice.com/b/helpcenter/1p8d7djs/r9jxqxnd
        path = f"/open_api/{quote(ref.project_key, safe='')}/work_item/{quote(work_item_type_key, safe='')}/query"
        items = await self._query_list(
            path, ref.user_key, {"work_item_ids": [ref.work_item_id], "fields": ["name"]},
            action="work item query",
        )
        matched = [item for item in items if item.get("id") == ref.work_item_id]
        if len(matched) != 1:
            raise RuntimeError("Meegle work item query has no unique matching item")
        name = matched[0].get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError("Meegle work item query has no name")
        return name.strip()

    async def get_user_email(self, user_key: str) -> str | None:
        """Resolve an exact Meegle user through the trusted plugin directory."""
        users = await self._query_users(user_key, [user_key])
        matched = [user for user in users if user.get("user_key") == user_key]
        if len(matched) != 1:
            return None
        email = matched[0].get("email")
        return email.strip() if isinstance(email, str) and email.strip() else None

    async def _query_users(self, operator_key: str, user_keys: list[str]) -> list[dict[str, Any]]:
        if not user_keys:
            return []
        # source: 获取用户详情 https://meego-hc.larkoffice.com/b/helpcenter/1p8d7djs/xqbea3rp
        # Verified with the existing AI plugin identity; data is a list (unlike AI node APIs).
        return await self._query_list(
            "/open_api/user/query", operator_key, {"user_keys": user_keys}, action="user query",
        )

    async def _query_list(
        self, path: str, operator_key: str, body: dict[str, Any], *, action: str
    ) -> list[dict[str, Any]]:
        token = await self.get_plugin_token()
        response = await self._post(
            path,
            headers={"X-PLUGIN-TOKEN": token, "X-USER-KEY": operator_key},
            json=body,
        )
        try:
            payload = response.json()
            error = payload.get("error") or payload.get("err") or {}
            code = int(payload.get("err_code") or error.get("code") or 0)
            users = payload.get("data")
        except (ValueError, TypeError, AttributeError):
            raise RuntimeError(f"Meegle {action} returned invalid JSON") from None
        if response.status_code >= 400 or code:
            raise RuntimeError(f"Meegle {action} failed (HTTP {response.status_code}, code={code})")
        if not isinstance(users, list) or any(not isinstance(user, dict) for user in users):
            raise RuntimeError(f"Meegle {action} response data must be a list")
        return users

    async def edit_status(
        self,
        ref: AINodeRef,
        status: str,
        *,
        idem_key: str,
    ) -> None:
        if status not in {"idle", "running", "done", "stopped"}:
            raise ValueError(f"unsupported AI node status: {status}")
        # source: lpm ai docs "修改 AI 节点数据 请求体参数 project_key work_item_id node_uuid status props"
        await self._request(
            "/open_api/ai_node/edit",
            ref,
            extra={"status": status},
            idem_key=idem_key,
        )

    async def confirm(self, ref: AINodeRef, *, idem_key: str) -> None:
        # source: lpm ai docs "完成 AI 节点 请求体参数 project_key work_item_id node_uuid"
        await self._request(
            "/open_api/ai_node/confirm", ref, idem_key=idem_key
        )

    async def _request(
        self,
        path: str,
        ref: AINodeRef,
        *,
        extra: dict[str, Any] | None = None,
        idem_key: str = "",
    ) -> dict[str, Any]:
        token = await self.get_plugin_token()
        # source: lpm ai docs "Open API 概述 Header X-PLUGIN-TOKEN X-USER-KEY AI 应用"
        headers = {
            "Content-Type": "application/json",
            "X-PLUGIN-TOKEN": token,
            "X-USER-KEY": ref.user_key,
        }
        if idem_key:
            headers["X-IDEM-UUID"] = idem_key
        body: dict[str, Any] = {
            "project_key": ref.project_key,
            "work_item_id": ref.work_item_id,
            "node_uuid": ref.node_uuid,
        }
        body.update(extra or {})
        response = await self._post(path, headers=headers, json=body)
        return self._decode(response, path)

    async def _post(self, path: str, **kwargs) -> httpx.Response:
        started = time.monotonic()
        try:
            response = await self._http.post(path, **kwargs)
        except Exception as exc:
            log.warning(
                "meegle_api_failed", workflow="meegle", provider="meegle", method="POST", path=path,
                duration_ms=int((time.monotonic() - started) * 1000), **error_fields(exc),
            )
            raise
        code = None
        try:
            payload = response.json()
            error = payload.get("error") or payload.get("err") or {}
            code = int(payload.get("err_code") or error.get("code") or 0)
        except (ValueError, TypeError, AttributeError):
            pass
        write = log.warning if response.status_code >= 400 or code is None or code else (
            log.info if path.endswith(("/edit", "/confirm", "/plugin_token")) else log.debug
        )
        write(
            "meegle_api_finished", workflow="meegle", provider="meegle", method="POST", path=path,
            http_status=response.status_code, code=code,
            external_request_id=response.headers.get("x-request-id") or response.headers.get("x-tt-logid"),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return response

    @staticmethod
    def _decode(response: httpx.Response, action: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Meegle {action} returned invalid JSON (HTTP {response.status_code})"
            ) from exc
        error = payload.get("error") or {}
        code = int(error.get("code") or 0)
        if response.status_code >= 400 or code != 0:
            raise RuntimeError(
                f"Meegle {action} failed (HTTP {response.status_code}, "
                f"code={code}, msg={error.get('msg', '')})"
            )
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"Meegle {action} response data is not an object")
        return data
