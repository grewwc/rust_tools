"""SCM OpenAPI for the server-selected PPE repository.

Sources: ByteCloud SCM v1 cicd_create, latest_version and version detail contracts.
No CLI or personal authentication is used at runtime.
"""

import re
from dataclasses import asdict, dataclass

import httpx

from conciliator.utils.logging import get_logger

SCM_REPOSITORY_ID = 572600
SCM_REPOSITORY = "data/dp/aeolus_ppe"
GATEWAY = "https://bc-cn-gw.bytedance.net"
log = get_logger(__name__)


class PlatformError(RuntimeError):
    """A credential-free error; write outcomes may require reconciliation."""

    def __init__(self, message, *, uncertain=False):
        super().__init__(message)
        self.uncertain = uncertain


def _diagnostic_id(value):
    if type(value) not in (str, int):
        return None
    value = str(value)
    return value if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else None


def platform_error(message, *, domain, method, path, response=None, uncertain=False):
    """Keep endpoint and correlation IDs, never platform prose or request config."""
    endpoint = httpx.URL(path).path
    details: dict[str, str | int] = {
        "platform": domain,
        "method": method,
        "path": endpoint,
    }
    labels = [f"平台：{domain}", f"接口：{method} {endpoint}"]
    if response is not None:
        details["http_status"] = response.status_code
        try:
            value = response.json()
        except ValueError:
            value = None
        payload = value if isinstance(value, dict) else {}
        code = next(
            (code for key in ("code", "error_code") if (code := _diagnostic_id(payload.get(key))) is not None),
            None,
        )
        request_id = next(
            (identity for candidate in (
                response.headers.get("X-Tt-Logid"), response.headers.get("X-Request-Id"),
                payload.get("request_id"), payload.get("log_id"), payload.get("logid"),
            ) if (identity := _diagnostic_id(candidate)) is not None),
            None,
        )
        if code is not None:
            details["code"] = code
            labels.append(f"业务码：{code}")
        if request_id is not None:
            details["request_id"] = request_id
            labels.append(f"请求 ID：{request_id}")
    log.warning("ppe_platform_request_failed", reason=message, **details)
    return PlatformError(f"{message}；" + "；".join(labels), uncertain=uncertain)


async def request(
    client,
    domain,
    method,
    path,
    *,
    body=None,
    params=None,
    headers=None,
    write=False,
    allow_list=False,
    not_found_code=None,
):
    try:
        response = await client.request(
            method, path, headers={"domain": domain, **(headers or {})}, json=body, params=params
        )
    except Exception:
        raise platform_error(
            "平台请求未完成，请查询原请求结果后再处理",
            domain=domain, method=method, path=path, uncertain=write,
        ) from None
    if response.status_code == 404 and not write:
        return None
    if response.status_code == 400 and not write and not_found_code:
        try:
            missing = response.json()
        except ValueError:
            missing = None
        if isinstance(missing, dict) and missing.get("code") == not_found_code:
            return None
    if response.status_code >= 400:
        raise platform_error(
            f"平台请求失败（HTTP {response.status_code}）",
            domain=domain, method=method, path=path, response=response, uncertain=write,
        ) from None
    try:
        value = response.json()
    except ValueError:
        raise platform_error(
            "平台返回格式无效", domain=domain, method=method, path=path, response=response, uncertain=write,
        ) from None
    if allow_list and isinstance(value, list):
        return value
    if not isinstance(value, dict):
        raise platform_error(
            "平台返回格式无效", domain=domain, method=method, path=path, response=response, uncertain=write,
        )
    if "code" in value:
        if value["code"] not in (0, "SUCCESS"):
            raise platform_error(
                "平台未接受请求，请核对权限与请求参数",
                domain=domain, method=method, path=path, response=response, uncertain=write,
            )
        return value.get("data")
    return value


@dataclass(frozen=True)
class ScmVersion:
    version: str
    version_id: str
    commit_sha: str
    status: str
    build_type: str
    architectures: tuple[str, ...]

    def snapshot(self):
        return asdict(self)

    def require_success(self, commit_sha, *, allowed_types=frozenset({"test"})):
        if (
            self.commit_sha != commit_sha
            or self.build_type not in allowed_types
            or "x86_64" not in self.architectures
        ):
            raise PlatformError("SCM 制品与固定提交、构建类型或架构不一致")
        if self.status != "build_ok":
            raise PlatformError("SCM 构建尚未成功，不能提交 PPE 部署")


class ScmClient:
    def __init__(self, http: httpx.AsyncClient, operator: str, repository_id=SCM_REPOSITORY_ID):
        self.http = http
        self.operator = operator
        self.repository_id = repository_id

    async def for_commit(self, commit_sha):
        """Reuse Codebase's exact-SHA build; never substitute the latest version."""
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise PlatformError("SCM 查询必须绑定完整 SHA")
        candidates = []
        for offset in range(0, 250, 50):
            value = await request(
                self.http,
                "scm;v1",
                "GET",
                "/api/v2/versions/",
                params={
                    "repos_id": self.repository_id,
                    "commit": commit_sha,
                    "type_list": "test",
                    "limit": 50,
                    "offset": offset,
                },
            )
            if not isinstance(value, dict) or not isinstance(value.get("results"), list):
                raise PlatformError("SCM 提交查询返回格式无效")
            for row in value["results"]:
                if row.get("repos") != self.repository_id or row.get("base_commit_hash") != commit_sha:
                    raise PlatformError("SCM 提交查询返回了其他仓库或提交")
                # CN PPE cannot consume offline artifacts, even when the SHA matches.
                if row.get("type") == "test" and "x86_64" in (row.get("arch") or []):
                    candidates.append(row)
            if not value.get("next"):
                break
        if not candidates:
            return None
        failed = {"build_failed", "build_fail", "failed", "canceled"}
        candidates.sort(
            key=lambda r: (r.get("status") == "build_ok", r.get("status") not in failed, int(r["id"])), reverse=True
        )
        chosen = candidates[0]
        detail = await self.version(chosen["version"])
        if detail is None or detail.version_id != str(chosen["id"]) or detail.commit_sha != commit_sha:
            raise PlatformError("SCM 制品详情与查询结果不一致")
        return detail

    async def version(self, version):
        if not re.fullmatch(r"\d+(?:\.\d+){3}", version):
            raise PlatformError("SCM 版本格式无效")
        # source: scm/v1/api/b86bb9678f90e2babe04d9bb52ed7c11e157d9e3.md
        value = await request(self.http, "scm;v1", "GET", f"/api/repos/{self.repository_id}/versions/{version}")
        if value is None:
            return None
        if not isinstance(value, dict) or value.get("version") != version or value.get("repos") != self.repository_id:
            raise PlatformError("SCM 版本身份不匹配")
        commit = value.get("base_commit_hash", "")
        arch = value.get("arch")
        if (
            not re.fullmatch(r"[0-9a-f]{40}", commit)
            or not isinstance(arch, list)
            or not all(isinstance(item, str) for item in arch)
            or not value.get("id")
        ):
            raise PlatformError("SCM 版本详情缺少提交或架构信息")
        return ScmVersion(
            version,
            str(value["id"]),
            commit,
            value.get("status", ""),
            value.get("type", ""),
            tuple(arch),
        )

    async def next_version(self):
        # source: scm/v1/api/4a330f9dab3d646be2357c76b2087512625e7ed7.md
        value = await request(
            self.http, "scm;v1", "GET", "/api/v2/versions/latest_version/", params={"repo_id": str(self.repository_id)}
        )
        version = value.get("version", "") if isinstance(value, dict) else ""
        if not re.fullmatch(r"\d+(?:\.\d+){3}", version):
            raise PlatformError("不能确定 SCM 预留版本；不会使用 latest 进行部署")
        prefix, number = version.rsplit(".", 1)
        return f"{prefix}.{int(number) + 1}"

    async def create(self, *, commit_sha, version, request_id):
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha) or not re.fullmatch(r"\d+(?:\.\d+){3}", version):
            raise PlatformError("SCM 构建必须绑定完整 SHA 和预留版本")
        if not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise PlatformError("SCM 构建标识无效")
        # source: scm/v1/api/ddd97cf76d1e1687d47448a7fce3b25c9283cf1e.md
        value = await request(
            self.http,
            "scm;v1",
            "POST",
            "/api/v2/versions/cicd_create",
            write=True,
            headers={"X-PLATFORM-PROXY-USER": self.operator},
            body={
                "repo_id": str(self.repository_id),
                "version": version,
                "type": "test",
                "arch": ["x86_64"],
                "commit_hash": commit_sha,
                "pub_base": "commit_base",
                "create_user": self.operator,
                "desc": f"Conciliator PPE {request_id}",
                "has_preonline": False,
                "sync_aws": False,
                "sync_oss": False,
                "sync_bvc": False,
                "skip_version_check": False,
            },
        )
        context = value.get("context") if isinstance(value, dict) else None
        if isinstance(value, dict) and value.get("state") == "failed" and not (context or {}).get("version_id"):
            raise PlatformError("SCM 未受理构建，请核对应用接入、操作人权限及构建参数")
        if (
            not isinstance(context, dict)
            or str(context.get("repo_id")) != str(self.repository_id)
            or context.get("version_version") != version
            or not context.get("version_id")
        ):
            raise PlatformError("SCM 构建响应缺少匹配的版本身份，请按预留版本对账", uncertain=True)
        return str(context["version_id"])
