from __future__ import annotations

import base64
import hashlib
import hmac
import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote, urlsplit

from conciliator.domain.aeolus_task import AeolusState, AeolusTask
from conciliator.integrations.lark.client import LarkClient, LarkError
from conciliator.integrations.lark.docs import extract_document_ref, read_document_markdown
from conciliator.services.aeolus_registry import AeolusTaskRegistry
from conciliator.services.lark_user_token_service import LarkUserTokenService
from conciliator.utils.retry import TransientError

DOCUMENT_CAPABILITY_HEADER = "X-Aeolus-Document-Capability"
_CAPABILITY_VERSION = "v1"
_CAPABILITY_TTL_SECONDS = 6 * 60 * 60
_LARK_URL_RE = re.compile(
    r"https?://[A-Za-z0-9.-]+/(?:docx|wiki)/[A-Za-z0-9_-]+",
    re.IGNORECASE,
)
_DOCUMENT_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ACTIVE_STATES = {AeolusState.EVALUATING, AeolusState.CODING, AeolusState.RECODING}


class _CiteReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "cite":
            return
        values = {name: value or "" for name, value in attrs}
        kind = values.get("file-type", "").lower()
        token = values.get("doc-id", "")
        if kind in {"docx", "wiki"} and _DOCUMENT_TOKEN_RE.fullmatch(token):
            self.references.append((kind, token))


class InvalidDocumentCapability(ValueError):
    pass


class DocumentAccessExpired(InvalidDocumentCapability):
    pass


class DocumentNotAllowed(PermissionError):
    pass


@dataclass(frozen=True)
class LarkAuthorizationRequired(Exception):
    authorization_url: str


class DocumentAccessInactive(PermissionError):
    pass


def public_origin_from_redirect_uri(redirect_uri: str) -> str:
    parsed = urlsplit(redirect_uri.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LARK_OAUTH_REDIRECT_URI must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("LARK_OAUTH_REDIRECT_URI must not contain user info")
    return f"{parsed.scheme}://{parsed.netloc}"


def _document_key(url: str) -> str:
    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "").lower()
    if not (
        host == "larkoffice.com"
        or host.endswith(".larkoffice.com")
        or host == "feishu.cn"
        or host.endswith(".feishu.cn")
    ):
        raise ValueError("document URL must use a Feishu or Lark host")
    ref = extract_document_ref(url)
    return f"{ref.kind}:{ref.token}"


def _canonical_document_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    key = _document_key(url)
    kind, token = key.split(":", 1)
    return f"https://{parsed.hostname}/{kind}/{token}"


def _cite_document_urls(task: AeolusTask) -> list[str]:
    parsed = urlsplit(task.requirement_url.strip())
    try:
        _document_key(task.requirement_url)
    except ValueError:
        return []
    parser = _CiteReferenceParser()
    parser.feed(task.requirement_text)
    origin = f"{parsed.scheme}://{parsed.hostname}"
    return [f"{origin}/{kind}/{token}" for kind, token in parser.references]


def _allowed_documents(task: AeolusTask) -> dict[str, str]:
    allowed: dict[str, str] = {}
    document_urls = [
        *_LARK_URL_RE.findall(task.requirement_text),
        *_cite_document_urls(task),
    ]
    for url in document_urls:
        try:
            allowed.setdefault(_document_key(url), _canonical_document_url(url))
        except ValueError:
            continue
    # The backend supplies the primary document; even self-links are not tool inputs.
    with suppress(ValueError):
        allowed.pop(_document_key(task.requirement_url), None)
    return allowed


def referenced_document_urls(task: AeolusTask) -> tuple[str, ...]:
    """Return validated, de-duplicated references in source order."""
    return tuple(_allowed_documents(task).values())


class AeolusDocumentAccessService:
    def __init__(
        self,
        *,
        registry: AeolusTaskRegistry,
        user_tokens: LarkUserTokenService,
        lark_client: LarkClient,
        public_base_url: str,
        signing_secret: str,
        ttl_seconds: int = _CAPABILITY_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._registry = registry
        self._user_tokens = user_tokens
        self._lark_client = lark_client
        self._public_base_url = public_base_url.rstrip("/")
        if not signing_secret:
            raise ValueError("document capability signing secret is required")
        self._signing_key = signing_secret.encode()
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def instructions(self, task: AeolusTask) -> str:
        allowed = _allowed_documents(task)
        if not allowed:
            return ""
        capability = self.issue(task)
        endpoint = (
            f"{self._public_base_url}/api/aeolus/tasks/"
            f"{quote(task.task_id, safe='')}/lark-document"
        )
        allowed_lines = "\n".join(f"- {url}" for url in allowed.values())
        return f"""
## 飞书引用文档读取工具

主需求文档由后端读取，不支持通过此工具重复下载。
仅当需要核对下面列出的引用文档时，使用 Conciliator 的只读代理。不要尝试登录
`lark-cli`，也不要输出、解释或记录能力凭证。

允许读取的引用文档：
{allowed_lines}

调用方法（把 `$DOC_URL` 设置为上面某个完整 URL；沙箱内没有 `jq`/`rg`，请只用
`curl`/`python3`/`grep` 等基础命令）：
```bash
curl -sS -X POST '{endpoint}' \\
  -H 'Content-Type: application/json' \\
  -H '{DOCUMENT_CAPABILITY_HEADER}: {capability}' \\
  --data '{{"url":"'"$DOC_URL"'"}}'
```

响应是 JSON，直接读取字段，不需要 `jq`：
- 成功时读取 `markdown` 字段，即该文档的完整 Markdown 正文；
- `detail.code` 为 `lark_authorization_required` 时，说明应用无法读取且用户尚未授权或授权已失效；
- 其它错误把 `detail` 原文一并列出，不要猜测文档内容。

后端会先尝试应用身份，失败后再尝试当前操作者身份；任一身份可读即可返回正文。
若命令提示需要飞书授权，停止重试，在当前阶段的结构化结果中明确要求用户授权；
不要自行获取凭据或改用其他账号，所有文档读取必须通过此代理。

代理只返回 Markdown，凭证绑定当前任务和操作者，且会自动过期。不要读取列表外的文档。
""".strip()

    def issue(self, task: AeolusTask) -> str:
        return self._sign(task, int(self._clock()) + self._ttl_seconds)

    async def read_markdown(self, *, task_id: str, capability: str, url: str) -> str:
        task = await self._registry.get(task_id)
        self._verify(task, capability)
        if task.state not in _ACTIVE_STATES:
            raise DocumentAccessInactive("document access is not active for this task")
        try:
            key = _document_key(url)
        except ValueError as exc:
            raise DocumentNotAllowed("document is not allowed for this task") from exc
        if key not in _allowed_documents(task):
            raise DocumentNotAllowed("document is not allowed for this task")

        return await self.read_for_user(url, task.user_key)

    async def read_for_user(self, url: str, user_key: str) -> str:
        """Read with the application first, then the task operator if necessary."""
        _document_key(url)
        try:
            return await read_document_markdown(self._lark_client, url)
        except (LarkError, TransientError) as app_error:
            access_token = await self._user_tokens.user_access_token(user_key)
            if not access_token:
                # An absent user token cannot repair an app scope/configuration
                # error or an outage. Only document ACL denial calls for OAuth.
                if isinstance(app_error, LarkError) and app_error.code == 131006:
                    raise LarkAuthorizationRequired(
                        self._user_tokens.authorize_url(user_key)
                    ) from app_error
                raise
            try:
                return await read_document_markdown(
                    self._lark_client, url, access_token=access_token,
                )
            except LarkError:
                # Do not turn an application outage into a document ACL error
                # just because the other identity is also unable to read.
                if isinstance(app_error, TransientError):
                    raise app_error from None
                raise

    def _sign(self, task: AeolusTask, expires_at: int) -> str:
        payload = self._payload(task, expires_at)
        digest = hmac.new(self._signing_key, payload, hashlib.sha256).digest()
        signature = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return f"{_CAPABILITY_VERSION}.{expires_at}.{signature}"

    def _verify(self, task: AeolusTask, capability: str) -> None:
        try:
            version, expires_text, signature = capability.split(".", 2)
            expires_at = int(expires_text)
        except (TypeError, ValueError) as exc:
            raise InvalidDocumentCapability("invalid document capability") from exc
        if version != _CAPABILITY_VERSION or not signature:
            raise InvalidDocumentCapability("invalid document capability")
        if expires_at < int(self._clock()):
            raise DocumentAccessExpired("document capability has expired")
        expected = self._sign(task, expires_at).rsplit(".", 1)[-1]
        if not hmac.compare_digest(signature, expected):
            raise InvalidDocumentCapability("invalid document capability")

    @staticmethod
    def _payload(task: AeolusTask, expires_at: int) -> bytes:
        return (
            f"{_CAPABILITY_VERSION}\n{task.task_id}\n{task.user_key}\n"
            f"{expires_at}\nlark-document-read"
        ).encode()
