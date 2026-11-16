"""Feishu user-authorization endpoints.

`/oauth/lark/start` sends the operator to Feishu's consent page; `/oauth/lark/callback`
receives the code and stores the tokens. Both exist so a user can grant the service
permission to read documents with their own identity, once, instead of sharing each
document with the app.
"""

from __future__ import annotations

import secrets
from html import escape
from typing import Annotated
from urllib.parse import parse_qs, urlsplit

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from conciliator.integrations.lark.client import LarkError
from conciliator.services.lark_user_token_service import (
    InvalidOAuthState,
    LarkUserTokenService,
    OAuthIdentityMismatch,
)
from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import error_fields, logged_step
from conciliator.utils.retry import TransientError

router = APIRouter(tags=["lark-oauth"])

logger = get_logger("conciliator.lark_oauth")
_STATE_COOKIE = "lark_oauth_state"
_COOKIE_PATH = "/oauth/lark"

_SUCCESS_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>授权成功</title></head>
<body style="font-family:system-ui;padding:40px">
<h3>授权成功</h3>
<p>已获得以你的身份读取飞书文档的权限。系统会自动继续因授权中断的评估，请回到 Meegle 卡片查看进度。</p>
</body></html>"""

_ERROR_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>授权失败</title></head>
<body style="font-family:system-ui;padding:40px">
<h3>授权失败</h3>
<p>{message}</p>
<p>请关闭本页，回到 Meegle 卡片重新点击「去飞书授权」再试一次。</p>
</body></html>"""


def _service(request: Request) -> LarkUserTokenService:
    service = getattr(request.app.state, "lark_user_token_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lark user authorization is disabled",
        )
    return service


def _page(content: str, status_code: int = 200) -> HTMLResponse:
    response = HTMLResponse(content, status_code=status_code)
    response.delete_cookie(_STATE_COOKIE, path=_COOKIE_PATH, httponly=True, samesite="lax")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@router.get("/oauth/lark/start")
async def start(
    request: Request,
    user_key: str = Query(min_length=1, max_length=128),
) -> RedirectResponse:
    service = _service(request)
    try:
        url = service.authorize_url(user_key)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    response = RedirectResponse(url, status_code=status.HTTP_302_FOUND)
    params = parse_qs(urlsplit(url).query)
    response.set_cookie(
        _STATE_COOKIE, params["state"][0], max_age=900, path=_COOKIE_PATH,
        httponly=True, samesite="lax",
        secure=urlsplit(params["redirect_uri"][0]).scheme == "https",
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    logger.info("meegle_oauth_started", workflow="meegle", user_key=user_key)
    return response


@router.get("/oauth/lark/callback", response_class=HTMLResponse)
async def callback(
    request: Request,
    code: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
    state: Annotated[str | None, Query(min_length=1, max_length=1024)] = None,
    error: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
) -> HTMLResponse:
    service = _service(request)
    try:
        cookie_state = request.cookies.get(_STATE_COOKIE, "")
        if not state or not cookie_state or not secrets.compare_digest(
            state.encode(), cookie_state.encode()
        ):
            raise InvalidOAuthState("请在发起授权的浏览器中完成授权，或重新发起授权。")
        user_key = service.parse_state(state)
        if error is not None or code is None:
            logger.info(
                "meegle_oauth_incomplete", workflow="meegle", user_key=user_key,
                cancelled=error == "access_denied",
            )
            message = (
                "你已取消飞书授权，系统不会继续读取文档。"
                if error == "access_denied"
                else "飞书授权未完成，请重新发起授权。"
            )
            return _page(_ERROR_HTML.format(message=message))
        with logged_step("oauth_exchange", user_key=user_key):
            await service.complete_authorization(code=code, state=state)
    except InvalidOAuthState as exc:
        logger.warning("meegle_oauth_state_rejected", workflow="meegle", **error_fields(exc))
        return _page(_ERROR_HTML.format(message=escape(str(exc))), status_code=400)
    except OAuthIdentityMismatch as exc:
        logger.warning("meegle_oauth_identity_rejected", workflow="meegle", **error_fields(exc))
        return _page(_ERROR_HTML.format(message=escape(str(exc))), status_code=403)
    except (LarkError, TransientError) as exc:
        logger.error("lark_oauth_exchange_failed", workflow="meegle", **error_fields(exc))
        return _page(
            _ERROR_HTML.format(message=escape(str(exc))), status_code=status.HTTP_200_OK
        )
    aeolus_service = getattr(request.app.state, "aeolus_service", None)
    if aeolus_service is not None:
        try:
            resumed = await aeolus_service.resume_authorization_blocked(user_key)
            logger.info("meegle_oauth_tasks_resumed", workflow="meegle", user_key=user_key, count=resumed)
        except Exception as exc:
            logger.error(
                "lark_oauth_task_resume_failed", workflow="meegle", user_key=user_key, **error_fields(exc),
            )
    return _page(_SUCCESS_HTML)
