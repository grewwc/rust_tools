"""Build a pinned PPE image with the application JWT and an authorized human operator.

ByteCloud icm/v1 build_multi_images returns version IDs; resolve those exact IDs
on ICM, then verify the precise version's recipe. Never infer identity from latest.
"""

import copy
import re

from conciliator.integrations.ppe.scm import PlatformError, request

IMAGE_ORIGIN = "https://image-manager.byted.org"
REGION = "China-North-LF"


def image_body(profile, scm_version, operator):
    if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,79}", operator) or operator.startswith("sa_"):
        raise PlatformError("镜像构建需要已授权的真实操作人")
    if not re.fullmatch(r"\d+(?:\.\d+){3}", scm_version):
        raise PlatformError("镜像构建必须使用精确 SCM 版本")
    body = copy.deepcopy(profile["image_recipe"])
    body.pop("image_id")
    body.pop("reference_version")
    body["username"] = operator
    body["repos"] = [
        {
            "remote_id": str(d["remote_id"]),
            "name": d["name"],
            "path": d["path"],
            "version": scm_version if d["main_repo"] else d["version"],
        }
        for d in profile["meta"]["scm_repos"]
    ]
    return body


def equivalent_recipe(expected, actual):
    scalars = (
        "username",
        "psm",
        "base_image",
        "load_path",
        "app_root_path",
        "runtime",
        "enable_sidecar",
        "region",
        "arch_map",
        "galaxy_node_id",
        "reuse",
        "user_tags",
    )
    if any(actual.get(key) != expected.get(key) for key in scalars):
        return False
    try:

        def deps(rows):
            return {(str(d["remote_id"]), d["name"], d["version"], d["path"]) for d in rows}

        def packages(rows):
            return sorted((d.get("package_type", d.get("type")), d["name"]) for d in rows)

        return (
            len(actual["repos"]) == len(expected["repos"])
            and deps(actual["repos"]) == deps(expected["repos"])
            and packages(actual["packages"]) == packages(expected["packages"])
        )
    except (KeyError, TypeError):
        return False


class IcmClient:
    def __init__(self, http, detail_http):
        self.http = http
        self.detail_http = detail_http

    async def create(self, body, image_id):
        # Official API aa872aea3e4ae8b2f5f8d14c5bef7e6f4f742b30; actual response is a list.
        data = await request(
            self.http,
            "icm;v1",
            "POST",
            "/api/v2/images/TCE/build_multi_images/",
            body=body,
            write=True,
            allow_list=True,
        )
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
            or data[0].get("region") != REGION
            or not str(data[0].get("id", "")).isdigit()
        ):
            raise PlatformError("镜像请求结果缺少精确版本 ID，需先对账", uncertain=True)
        return {"image_id": str(image_id), "image_version_id": str(data[0]["id"])}

    async def version(self, identity, expected):
        raw = await self._version_identity(identity)
        image_id = identity["image_id"]
        version = raw["version"]
        data = await request(
            self.http, "icm;v1", "GET", "/api/v2/images/TCE/version_detail/",
            params={"name": expected["psm"], "version": version}
        )
        if (
            not isinstance(data, dict)
            or str(data.get("image_id")) != str(image_id)
            or not equivalent_recipe(expected, data.get("build_params") or {})
        ):
            raise PlatformError("镜像配方与固定提交、依赖或操作人不一致")
        status = (data.get("regions", {}).get(REGION) or {}).get("status")
        # Completion requires agreement between the exact ID and version reads.
        succeeded = raw.get("status") == "ok" and status == "ok"
        failed = raw.get("status") in {"fail", "dropped", "timeout"} or status in {"fail", "dropped", "timeout"}
        resolved_base = (data.get("build_params", {}).get("region2base_image_version") or {}).get(REGION)
        return {
            **identity,
            "image_version": version,
            "image_status": "ok" if succeeded else "fail" if failed else "building",
            "resolved_base_image": resolved_base if isinstance(resolved_base, str) else None,
        }

    async def _version_identity(self, identity):
        image_id, version_id = identity["image_id"], identity["image_version_id"]
        if not str(image_id).isdigit() or not str(version_id).isdigit():
            raise PlatformError("镜像身份无效")
        # Same authenticated endpoint used by the official ICM CLI getVersion.
        raw = await request(self.detail_http, "icm;v1", "GET", f"/api/v1/images/{image_id}/versions/{version_id}/")
        if (
            not isinstance(raw, dict)
            or str(raw.get("id")) != str(version_id)
            or str((raw.get("image") or {}).get("id")) != str(image_id)
            or raw.get("region") != REGION
            or raw.get("arch_list") != ["x86_64"]
            or not re.fullmatch(r"\d+(?:\.\d+){3}", raw.get("version", ""))
        ):
            raise PlatformError("镜像版本身份、区域或架构不匹配")
        version = raw["version"]
        if identity.get("image_version") and identity["image_version"] != version:
            raise PlatformError("镜像版本 ID 与版本号不匹配")
        return raw

    async def baseline_version(self, baseline):
        """Verify the finished image itself; Atum and legacy builds share this contract."""
        raw = await self._version_identity(baseline)
        data = await request(
            self.http, "icm;v1", "GET", "/api/v2/images/TCE/version_detail/",
            params={"name": baseline["psm"], "version": baseline["image_version"]},
        )
        extras = raw.get("extras") or {}
        metadata = extras.get("Metadata") or {}
        main = [r for r in extras.get("Scm", []) if str(r.get("remote_id")) == str(baseline["scm_repo_id"])]
        repos = [
            {"remote_id": r["remote_id"], "name": r["repo_name"],
             "path": r["deploy_path"], "version": r["version"]}
            for r in extras.get("scm_repos", [])
        ]
        if (
            not isinstance(data, dict)
            or str(data.get("image_id")) != str(baseline["image_id"])
            or (data.get("build_params") or {}).get("psm") != baseline["psm"]
            or extras.get("image_digest") != baseline["image_digest"]
            or metadata.get("appPath") != baseline["app_path"]
            or metadata.get("baseImageVersion") != baseline["resolved_base_image"]
            or repos != baseline["scm_repos"]
            or len(main) != 1
            or main[0].get("name") != baseline["scm_repo_name"]
            or main[0].get("version") != baseline["version"]
            or main[0].get("commit_id") != baseline["commit_sha"]
        ):
            raise PlatformError("生产基准镜像的内容摘要、启动配置或制品不一致")
        status = (data.get("regions", {}).get(REGION) or {}).get("status")
        if raw.get("status") != "ok" or status != "ok":
            raise PlatformError("生产基准镜像已不可用，需更新基准配置")
        embedded = None
        if not repos:
            # Atum leaves replacement repos empty; Scm is the immutable in-image manifest.
            embedded = [
                {"name": r.get("name"), "version": r.get("version"), "path": r.get("path")}
                for r in extras.get("Scm", [])
            ]
            if (
                not embedded
                or len({r["name"] for r in embedded}) != len(embedded)
                or any(
                    not isinstance(r["name"], str) or not r["name"]
                    or not isinstance(r["version"], str) or not re.fullmatch(r"\d+(?:\.\d+){3}", r["version"])
                    or not isinstance(r["path"], str) or not r["path"].startswith("/opt/tiger/")
                    for r in embedded
                )
            ):
                raise PlatformError("生产基准镜像的内嵌依赖清单不完整")
        return {
            **{k: baseline[k] for k in ("image_id", "image_version_id", "image_version", "resolved_base_image")},
            "scm_version": baseline["version"], "scm_version_id": baseline["version_id"],
            "commit_sha": baseline["commit_sha"], "image_status": "ok", "artifact_source": "production_baseline",
            **({"embedded_scm_repos": embedded} if embedded is not None else {}),
        }
