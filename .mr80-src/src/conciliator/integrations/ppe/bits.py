"""SCM-only BITS OpenAPI with application JWT and the authorized operator.

Source: BITS OpenAPI examples 67d29ca7e7fd5a0511dd3bc6. Console APIs are not
used. A run ID must be saved before polling; an ambiguous write is not retried.
"""

import re

from conciliator.integrations.ppe.scm import SCM_REPOSITORY, SCM_REPOSITORY_ID, PlatformError, platform_error

ORIGIN = "https://bits.bytedance.net"


class BitsScmClient:
    def __init__(self, http, *, repository=SCM_REPOSITORY, repository_id=SCM_REPOSITORY_ID):
        self.http = http
        self.repository = repository
        self.repository_id = repository_id

    def for_profile(self, profile):
        return BitsScmClient(self.http, repository=profile["scm_repo_name"], repository_id=profile["scm_repo_id"])

    async def _request(self, method, path, operator, *, body=None):
        write = method == "POST"
        try:
            response = await self.http.request(method, path, headers={"Username": operator}, json=body)
        except Exception:
            raise platform_error(
                "BITS 请求未完成，请按原运行记录对账",
                domain="bits", method=method, path=path, uncertain=write,
            ) from None
        if response.status_code >= 400:
            raise platform_error(
                f"BITS 请求失败（HTTP {response.status_code}）",
                domain="bits", method=method, path=path, response=response, uncertain=write,
            )
        try:
            value = response.json()
        except ValueError:
            raise platform_error(
                "BITS 返回格式无效",
                domain="bits", method=method, path=path, response=response, uncertain=write,
            ) from None
        if not isinstance(value, dict) or value.get("code", 0) != 0:
            raise platform_error(
                "BITS 未接受请求，请核对权限与参数",
                domain="bits", method=method, path=path, response=response, uncertain=write,
            )
        return value

    def _validate_pipeline(self, detail, config):
        try:
            pipeline = detail["pipeline"]
            stages = pipeline["stages"]
            if (
                str(detail["pipeline_id"]) != str(config["pipeline_id"])
                or str(detail["space_id"]) != str(config["space_id"])
                or len(stages) != 1
                or len(stages[0]["jobs"]) != 1
            ):
                raise ValueError
            job = stages[0]["jobs"][0]
            builds = job["inputs"]["scm_configs"]
            if job["id"] != "scm_compile" or job["uses"] != "job_atoms/scm_compile_beta" or len(builds) != 1:
                raise ValueError
            build = builds[0]
            if (
                build["scm_repo_name"] != self.repository
                or build["pub_base"] != "commit_base"
                or build["revision"] != "{{custom.commit_hash}}"
                or build["scm_type"] != "test"
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError, IndexError):
            raise PlatformError("BITS 流水线必须只构建目标仓库固定提交，不能包含部署步骤") from None

    async def plan(self, config, commit_sha, operator):
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise PlatformError("BITS 构建必须绑定完整 SHA")
        value = await self._request("GET", f"/api/v1/pipelines_open/{config['pipeline_id']}", operator)
        detail = value.get("pipeline")
        self._validate_pipeline(detail, config)
        group = detail["pipeline"]["var_group"]
        variables = group.get("var_definitions", [])
        if not any(v.get("name") == "commit_hash" for v in variables):
            raise PlatformError("BITS 缺少已注册的 commit_hash 变量")
        return {
            "pipeline_id": int(config["pipeline_id"]),
            "custom_vars": [{
                "name": "custom.commit_hash",
                "value": {"text": commit_sha},
                "group_id": int(group["group_id"]),
                "group_version": int(group["version"]),
            }],
        }

    async def submit(self, body, operator):
        value = await self._request(
            "POST", f"/api/v1/pipelines/open/{body['pipeline_id']}/run", operator, body=body
        )
        run = value.get("pipeline_run") or {}
        if str(run.get("pipeline_id")) != str(body["pipeline_id"]) or not run.get("run_id"):
            raise PlatformError("BITS 写入响应缺少匹配的运行 ID，请对账后处理", uncertain=True)
        return {"bits_run_id": str(run["run_id"]), "bits_run_seq": str(run.get("run_seq", ""))}

    async def run(self, config, run_id, commit_sha, operator):
        value = await self._request("GET", f"/api/v1/pipelines_open/runs/{run_id}", operator)
        run = value.get("pipeline_run") or {}
        if str(run.get("pipeline_id")) != str(config["pipeline_id"]) or str(run.get("run_id")) != run_id:
            raise PlatformError("BITS 原运行身份不匹配")
        jobs = run.get("jobs") or []
        if len(jobs) > 1 or any(j.get("job_id") != "scm_compile" for j in jobs):
            raise PlatformError("BITS 原运行包含非 SCM 步骤")
        result = {"bits_run_status": run.get("run_status"), "bits_failed": run.get("run_status") in {9, 11}}
        if run.get("run_status") != 8:
            return result
        if len(jobs) != 1 or jobs[0].get("job_status") != 14:
            raise PlatformError("BITS 未完成成功的 SCM 构建")
        atom = jobs[0]["job_atom"]
        builds = atom["inputs"]["scm_configs"]
        if (
            len(builds) != 1
            or builds[0].get("revision") != commit_sha
            or builds[0].get("scm_repo_name") != self.repository
            or builds[0].get("pub_base") != "commit_base"
            or builds[0].get("scm_type") != "test"
        ):
            raise PlatformError("BITS 运行的构建输入与固定提交不一致")
        versions = [
            v for view in atom.get("views", [])
            for v in (view.get("value") or {}).get("versions", [])
        ]
        if (
            len(versions) != 1
            or versions[0].get("repo_id") != self.repository_id
            or versions[0].get("region") != "cn"
            or not re.fullmatch(r"\d+(?:\.\d+){3}", versions[0].get("version", ""))
            or not versions[0].get("version_id")
        ):
            raise PlatformError("BITS 原运行缺少唯一的目标仓库制品身份")
        result.update(
            scm_version=versions[0]["version"],
            scm_version_id=str(versions[0]["version_id"]),
            bits_success=True,
        )
        return result
