"""CN ENV v4 OpenAPI. All writes require a task-owned, exact resource identity.

Sources: ByteCloud env_platform/v1 catalog and the official ENV DSL guide
6655c52ab57cc702fdd19358. Never use high-level CLI dependency defaults here.
"""

import copy
import hashlib
import ipaddress
import json
import re
from pathlib import Path
from typing import Any

import httpx

from conciliator.integrations.ppe.scm import PlatformError, request

DOMAIN = "env_platform;v1"
PSM = "data.aeolus.api_v3"
REFERENCE = "ppe_aeolus_huaguojin_test"
TARGETS = {
    "dp/aeolus": (
        "aeolus_cn.json", PSM, 572600, "1251261261058", 217258, {"online", "query_online_mirror"}, "ppe_ai_ae_"
    ),
    "dp/VizQueryService": ("vqs_cn.json", "data.vqs.api", 53349, "1251819549442", 191197, {"default"}, "ppe_ai_vqs_"),
}
AEOLUS_PY27_TARGET = (
    "aeolus_py27_cn.json", PSM, 4863, "1254638941442", 217258, {"online", "query_online_mirror"}, "ppe_ai_ae_"
)


def environment_deleted(env):
    # ENV retains a readable tombstone after a successful destroy operation.
    stamp = (env or {}).get("deleted_at")
    return isinstance(stamp, (int, float)) and stamp > 0


def load_profile(path=None, *, repository="dp/aeolus", python_version=None):
    if repository not in TARGETS:
        raise PlatformError("PPE 仓库没有已核验的模板")
    if python_version is not None and (repository, python_version) != ("dp/aeolus", "2.7"):
        raise PlatformError("PPE Python 运行版本没有已核验的模板")
    selected = AEOLUS_PY27_TARGET if python_version == "2.7" else TARGETS[repository]
    value = json.loads(Path(path or Path(__file__).with_name(selected[0])).read_text())
    py27 = value.get("python_version") == "2.7"
    target = (
        AEOLUS_PY27_TARGET if py27 and value.get("repository") == "dp/aeolus"
        else TARGETS.get(value.get("repository"))
    )
    if target is None:
        raise PlatformError("PPE 模板目标不匹配")
    _, psm, repo_id, pipeline_id, image_id, cluster_names, prefix = target
    if (
        value.get("psm") != psm
        or value.get("standard_env") != "online_cn"
        or value.get("idc") != "LF"
        or value.get("scm_repo_id") != repo_id
        or value.get("operator") != "huaguojin"
        or value.get("environment_prefix", prefix) != prefix
    ):
        raise PlatformError("PPE 模板目标不匹配")
    bits = value.get("bits_scm")
    if bits is not None and bits != {"pipeline_id": pipeline_id, "space_id": "284273157378"}:
        raise PlatformError("PPE SCM 流水线目标不匹配")
    meta = value["meta"]
    clusters = meta.get("clusters", [])
    if {c.get("name") for c in clusters} != cluster_names or len(clusters) != len(cluster_names):
        raise PlatformError("PPE 模板集群集合不匹配")
    for cluster in clusters:
        if (
            cluster.get("zone") != "China-North-LF"
            or not cluster.get("base_cluster_id")
            or cluster.get("dc_infos") != [{"count": 1, "idc": "LF"}]
        ):
            raise PlatformError("PPE 集群来源或 LF 落点不匹配")
    dependencies = meta.get("scm_repos", [])
    main = [d for d in dependencies if d.get("main_repo")]
    if len(main) != 1 or main[0].get("remote_id") != repo_id or main[0].get("name") != value.get("scm_repo_name"):
        raise PlatformError("PPE 模板主仓不匹配")
    if any(not d.get("version") or not d.get("remote_id") for d in dependencies):
        raise PlatformError("PPE 依赖必须固定到明确仓库和版本")
    recipe = value.get("image_recipe", {})
    if (
        recipe.get("psm") != psm
        or recipe.get("region") != ["China-North-LF"]
        or recipe.get("arch_map") != {"China-North-LF": ["x86_64"]}
        or recipe.get("user_tags") != ["ppe"]
        or recipe.get("image_id") != image_id
        or not recipe.get("packages")
        or {p.get("package_type") for p in recipe["packages"]} != {"apt-get", "pip3" if repo_id == 572600 else "pip"}
        or any(not d.get("name") for d in dependencies)
    ):
        raise PlatformError("PPE 镜像配方、架构或完整依赖缺失")
    if repo_id == 53349 and (
        recipe.get("base_image") != "toutiao.debian.python3:latest"
        or recipe.get("load_path") != "toutiao/app/data/vqs/run/prod_run.sh"
        or len(dependencies) != 18
        or value.get("health_probe") != {"path": "/healthz", "json": {"code": "aeolus/ok"}}
    ):
        raise PlatformError("VQS PPE 必须使用线上普通 Python 3 配方和健康接口")
    if py27 and (
        value.get("base_release_max") != 104
        or recipe.get("base_image") != "debian.jessie.jdk8_yarn:latest"
        or recipe.get("load_path") != "toutiao/app/data/aeolus/run/prod_run.sh"
        or meta.get("app_path") != recipe.get("load_path")
        or meta.get("base_image") != recipe.get("base_image")
        or len(dependencies) != 26
        or not any(d["remote_id"] == 256483 and d["name"] == "data/dp/aeolus_expr_parse_py2" for d in dependencies)
        or not any(d["remote_id"] == 3177 and d["path"] == "ss_lib/python_package/lib/python2.7/site-packages/MySQLdb"
                   for d in dependencies)
        or not bits
    ):
        raise PlatformError("Aeolus Python 2.7 PPE 必须使用完整旧版配方与固定提交流水线")
    return value


def profile_revision(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def environment_name(identity, profile=None):
    if not re.fullmatch(r"[0-9a-f]{32}", identity):
        raise PlatformError("PPE 环境记录标识无效")
    return TARGETS[(profile or {}).get("repository", "dp/aeolus")][6] + identity[:16]


def deployment_body(
    profile, identity, version, *, image_version, cluster_ids=None, missing_names=None, operator=None,
    environment_profile=None,
):
    """Freeze dependencies; only replace the main SCM version, never select latest."""
    if not re.fullmatch(r"\d+(?:\.\d+){3}", version):
        raise PlatformError("PPE 部署版本必须固定")
    if not re.fullmatch(r"\d+(?:\.\d+){3}", image_version):
        raise PlatformError("PPE 部署必须固定已成功的镜像版本")
    name = environment_name(identity, environment_profile or profile)
    psm = profile["psm"]
    meta = copy.deepcopy(profile["meta"])
    for dependency in meta["scm_repos"]:
        if dependency.get("main_repo"):
            dependency["version"] = version
    for cluster in meta["clusters"]:
        cluster.update(image_tag="tce", image_version=image_version)
    if missing_names is not None:
        if not missing_names or not set(missing_names) <= {c["name"] for c in meta["clusters"]}:
            raise PlatformError("PPE 缺失集群集合无效")
        meta["clusters"] = [c for c in meta["clusters"] if c["name"] in missing_names]
    elif cluster_ids is not None:
        desired = {c["name"] for c in meta["clusters"]}
        if set(cluster_ids) != desired:
            raise PlatformError("更新必须指定本任务的全部集群")
        # ENV DSL requires numeric IDs, while resource observations store IDs as strings.
        if any(not isinstance(value, (int, str)) or not re.fullmatch(r"[0-9]+", str(value))
               for value in cluster_ids.values()):
            raise PlatformError("PPE 升级集群 ID 必须为正整数")
        cluster_ids = {name: int(value) for name, value in cluster_ids.items()}
        if any(value <= 0 for value in cluster_ids.values()) or len(set(cluster_ids.values())) != len(desired):
            raise PlatformError("更新必须指定本任务的不同集群")
        meta = {
            "scm_repos": meta["scm_repos"],
            # ENV version updates put the shared image on service meta.
            "image_tag": "tce",
            "image_version": image_version,
            "clusters": [{"id": cluster_ids[c["name"]]} for c in meta["clusters"]],
            "enable_dsd": False,
        }
    return {
        "env": {
            "name": name,
            "type": "ppe",
            "idc": "LF",
            "auth_type": 1,
            "env_managers": [operator] if operator else [],
            "env_collaborators": [],
            "desc": f"Conciliator PPE {identity}",
            "flow_base": "prod",
        },
        "deployment": {
            psm: {
                "psm": psm,
                "type": "tce",
                "action": "create_cluster" if missing_names is not None else "upgrade" if cluster_ids else "create",
                "meta": meta,
            }
        },
        "env_action": "create_env" if cluster_ids is None and missing_names is None else "",
        "failed_strategy": "WaitBrothers",
        "run_option": {"sync": False, "expire_time": 10},
        "comment": f"Conciliator PPE {identity}",
    }


class PpeClient:
    def __init__(self, http, profile=None, *, environment_profile=None, embedded_scm_repos=None):
        self.http = http
        self.profile = profile
        self.psm = profile["psm"] if profile else PSM
        self.environment_profile = environment_profile or profile
        self.embedded_scm_repos = embedded_scm_repos

    def for_profile(self, profile, *, companion_artifact=None):
        if profile.get("companion_profile"):
            from conciliator.integrations.ppe.environment import PpeEnvironmentClient
            return PpeEnvironmentClient(self.http, profile, companion_artifact)
        return PpeClient(self.http, profile)

    async def environment(self, name):
        if not re.fullmatch(r"ppe_[a-z0-9_]{1,26}", name):
            raise PlatformError("PPE 环境名称无效")
        # source: env_platform/v1/api/8ead444c063b7048f79f3cb3688071d31af22933.md
        return await request(
            self.http,
            DOMAIN,
            "GET",
            f"/api/v4/env/{name}",
            not_found_code="NOT_EXIST",
            params={"type": "ppe", "detail": True, "standard_env": "online_cn"},
        )

    async def service(self, name):
        # source: env_platform/v1/api/77af11dc8ed0f3db0c7c85f088b24f5eeb4adfe0.md
        return await request(
            self.http,
            DOMAIN,
            "GET",
            "/api/v4/instance_meta",
            params={
                "env": name,
                "env_type": "ppe",
                "service": self.psm,
                "service_type": "tce",
                "detail": True,
            },
        )

    async def precheck(self, body):
        # source: env_platform/v1/api/1bd1087649712ee2350e197b993a982cf53aa9ff.md
        result = await request(
            self.http, DOMAIN, "POST", "/api/v4/dsl/pre_check/common", body=body, params={"standard_env": "online_cn"}
        )
        if not isinstance(result, dict) or not {"pre_check_result", "err_res", "break_res"} & result.keys():
            raise PlatformError("PPE 预检没有返回可判断的结果")
        return result

    async def prepare(
        self,
        profile,
        identity,
        version,
        *,
        image_version,
        operator,
        cluster_ids=None,
        missing_names=None,
        existing=None,
    ):
        if profile["psm"] != self.psm:
            raise PlatformError("PPE 客户端与模板目标不匹配")
        if existing is not None:
            if (
                existing.get("name") != environment_name(identity, self.environment_profile)
                or existing.get("desc") != f"Conciliator PPE {identity}"
                or environment_deleted(existing)
            ):
                raise PlatformError("PPE 环境归属不匹配或已回收")
            if cluster_ids is None and missing_names is None:
                resources = await self.resources(existing["name"])
                actual = {c["name"]: c["id"] for c in resources["clusters"]}
                desired = {c["name"] for c in profile["meta"]["clusters"]}
                if (
                    set(actual) - desired
                    or len(actual) != len(resources["clusters"])
                    or set(actual.values()) & {str(c["base_cluster_id"]) for c in profile["meta"]["clusters"]}
                ):
                    raise PlatformError("PPE 目标集群集合或归属不匹配")
                if resources.get("service_id"):
                    missing_names = sorted(desired - set(actual))
                    if missing_names:
                        # Partial services still need their existing routing checked before repair.
                        for c in resources["clusters"]:
                            if not await self.routing_match(int(c["id"]), c["name"]):
                                raise PlatformError("VQS 必须配置 AEOLUS_SERVICE_DISCOVERY=0，请释放后重新部署")
                    else:
                        observed = await self.observe(
                            existing["name"], identity, str(existing["id"]), profile,
                            image_version=image_version, scm_version=version,
                        )
                        if not observed.get("routing_match"):
                            raise PlatformError("VQS 必须配置 AEOLUS_SERVICE_DISCOVERY=0，请释放后重新部署")
                        if observed.get("versions_match"):
                            return None
                        cluster_ids, missing_names = actual, None
        body = deployment_body(
            profile,
            identity,
            version,
            image_version=image_version,
            cluster_ids=cluster_ids,
            missing_names=missing_names,
            operator=operator,
            environment_profile=self.environment_profile,
        )
        if existing is not None:
            managers = existing.get("env_managers")
            if not managers or not isinstance(managers, list) or not all(isinstance(v, str) for v in managers):
                raise PlatformError("PPE 当前管理员信息无法保留")
            body["env"]["env_managers"] = managers
            body["env"]["env_collaborators"] = existing.get("env_collaborators", [])
            body["env_action"] = ""
        if cluster_ids is None or missing_names is not None:
            reference = await self.service(profile["reference_environment"])
            if not isinstance(reference, dict) or reference.get("meta", {}).get("psm") != self.psm:
                raise PlatformError("PPE 参考服务身份不匹配")
            source_ids = {item["name"]: item["id"] for item in reference.get("instances", [])}
            for cluster in body["deployment"][self.psm]["meta"]["clusters"]:
                # Fetch sensitive environment variables in memory from the frozen source cluster.
                # source: env_platform/v1/api/4bd1d480d3ecbf7f8a0311e83c2d35caa43d977d.md
                if source_ids.get(cluster["name"]) != cluster["base_cluster_id"]:
                    raise PlatformError("PPE 参考集群已变更，请重新核验模板")
                source = await request(
                    self.http,
                    DOMAIN,
                    "GET",
                    "/api/v4/suggest/cluster_by_id",
                    params={
                        "cluster_id": cluster["base_cluster_id"],
                        "env_type": "ppe",
                        "standard_env": "online_cn",
                    },
                )
                if (
                    not isinstance(source, dict)
                    or source.get("base_cluster", {}).get("id") != cluster["base_cluster_id"]
                    or source.get("name") != cluster["name"]
                    or source.get("psm") not in {None, "", self.psm}
                ):
                    raise PlatformError("PPE 来源集群身份发生变化")
                base = source.get("base_cluster", {})
                if source.get("new_env_vars_blocked") or base.get("new_env_vars_blocked"):
                    raise PlatformError("PPE 来源敏感配置无读取权限")
                variables = source.get("new_env_vars", base.get("new_env_vars"))
                overrides = profile.get("required_env_vars", {})
                if overrides:
                    if variables is not None and not isinstance(variables, dict):
                        raise PlatformError("PPE 来源环境变量格式无法覆盖")
                    variables = {**(variables or {}), **overrides}
                if variables is not None:
                    cluster["new_env_vars"] = variables
        return body

    async def submit(self, body, *, create):
        name = body["env"]["name"]
        prefix = TARGETS[(self.environment_profile or {}).get("repository", "dp/aeolus")][6]
        if not re.fullmatch(re.escape(prefix) + r"[0-9a-f]{16}", name):
            raise PlatformError("只能操作任务专属 PPE，不能覆盖参考环境")
        # source: env_platform/v1/api/0296fac6f887ccac1ea279f798e27e314591d2c3.md
        # source: env_platform/v1/api/a0637178c1b99fbdd87fb7bbdcc26132e679945e.md
        data = await request(
            self.http,
            DOMAIN,
            "POST" if create else "PUT",
            "/api/v4/dsl/env" + ("" if create else f"/{name}"),
            body=body,
            write=True,
            params={"standard_env": "online_cn", "type": "ppe"},
        )
        if not isinstance(data, dict) or not (data.get("id_str") or data.get("id")):
            raise PlatformError("PPE 部署已提交但没有工单身份，必须先对账", uncertain=True)
        return str(data.get("id_str") or data["id"])

    async def ticket(self, ticket_id):
        if not str(ticket_id).isdigit():
            raise PlatformError("PPE 工单标识无效")
        # source: env_platform/v1/api/2e214881f54b83421186f131ed8df6f196df6487.md
        return await request(self.http, DOMAIN, "GET", f"/api/v4/dsl/deployment/{ticket_id}")

    async def remove(self, name, *, expected_id, identity):
        if name != environment_name(identity, self.environment_profile):
            raise PlatformError("PPE 释放目标不属于环境记录")
        current = await self.environment(name)
        if (
            not isinstance(current, dict)
            or str(current.get("id")) != str(expected_id)
            or current.get("desc") != f"Conciliator PPE {identity}"
        ):
            raise PlatformError("PPE 环境身份或归属不匹配，禁止删除")
        # source: env_platform/v1/api/b56ccbaa7d3e32c989bfb0915d9697d9c7a1c1bc.md
        return await request(
            self.http,
            DOMAIN,
            "DELETE",
            f"/api/v4/dsl/env/{name}",
            write=True,
            params={"type": "ppe", "del_env": "true", "standard_env": "online_cn"},
        )

    async def observe(self, name, identity, platform_id, profile, *, image_version, scm_version):
        env = await self.environment(name)
        if env is None or environment_deleted(env):
            return {"absent": True, "ready": False, "clusters": []}
        if (
            str(env.get("id")) != str(platform_id)
            or name != environment_name(identity, self.environment_profile)
            or env.get("desc") != f"Conciliator PPE {identity}"
        ):
            raise PlatformError("PPE 环境身份不匹配")
        service = await self.service(name)
        if not service or service.get("meta", {}).get("psm") != self.psm:
            return {"ready": False, "clusters": [], "platform_id": str(env["id"])}
        service_id = str(service["meta"]["id"])
        expected = {
            str(d["remote_id"]): (scm_version if d["main_repo"] else d["version"], d["path"])
            for d in profile["meta"]["scm_repos"]
        }
        if profile.get("production_baseline") and not expected:
            if not self.embedded_scm_repos:
                raise PlatformError("生产基准镜像的内嵌依赖尚未核验")
            expected = {d["name"]: (d["version"], d["path"]) for d in self.embedded_scm_repos}
        desired = {c["name"]: c for c in profile["meta"]["clusters"]}
        clusters = []
        for cluster in service.get("instances", []):
            cluster_id, cluster_name = cluster["id"], cluster["name"]
            if cluster_name not in desired or cluster_id in {c["base_cluster_id"] for c in desired.values()}:
                raise PlatformError("PPE 目标集群集合或归属不匹配")
            detail = await request(
                self.http, "tce;v1", "GET", f"/open-apis/v1/services/{service_id}/clusters/{cluster_id}/"
            )
            if (
                not detail
                or detail.get("meta", {}).get("psm") != self.psm
                or detail["meta"].get("env") != name
                or detail["meta"].get("id") != cluster_id
            ):
                raise PlatformError("TCE 集群身份不匹配")
            runtime = detail.get("runtime", {})
            repo_info = cluster.get("repo_info", [])
            # Atum's manifest uses SCM names; auxiliary remote_id values there are zero.
            deps = {
                d.get("name") if self.embedded_scm_repos else str(d["scm_repo_id"]):
                    (d["version"], d.get("path", ""))
                for d in repo_info
            }
            dependencies_match = deps == expected and len(repo_info) == len(expected)
            groups = await request(
                self.http, "tce;v1", "GET", f"/open-apis/v1/clusters/{cluster_id}/instances/", allow_list=True
            )
            # TCE instances are a raw list. The request helper enables lists only on explicit read contracts.
            if not isinstance(groups, list):
                raise PlatformError("TCE 实例查询返回格式无效")
            pods = [p for group in groups for p in group.get("data", []) if not p.get("deleted")]
            versions_match = (
                runtime.get("image_tag") == "tce"
                and runtime.get("image_version") == image_version
                and dependencies_match
            )
            resource, wanted = detail.get("resource", {}), desired[cluster_name]
            dc_info = [d for d in runtime.get("replicas", {}).get("dc_info", []) if not d.get("deleted")]
            resources_match = (
                resource.get("zone") == wanted["zone"]
                and resource.get("physical_cluster") == "PPE"
                and resource.get("logical_cluster") == "ipv6"
                and resource.get("cpu") == wanted["cpu"]
                and resource.get("mem") == wanted["mem"]
                and len(dc_info) == 1
                and dc_info[0].get("idc") == "LF"
                and dc_info[0].get("count") == 1
            )
            routing_match = await self.routing_match(cluster_id, cluster_name, runtime=runtime)
            ready = (
                detail["meta"].get("status") == "running"
                and runtime.get("image_tag") == "tce"
                and runtime.get("image_version") == image_version
                and dependencies_match
                and len(pods) == 1
                and resources_match
                and routing_match
            )
            health = False
            if ready:
                pod = pods[0]
                ready = pod.get("podPhase") == "Running" and pod.get("podNotReady") is not True
                if ready:
                    health = await self.pod_health(pod["podName"], cluster_id)
            clusters.append(
                {
                    "id": str(cluster_id),
                    "name": cluster_name,
                    "status": detail["meta"].get("status"),
                    "image_version": runtime.get("image_version"),
                    "replicas": len(pods),
                    "ready": bool(ready and health),
                    "versions_match": versions_match,
                    "health_ok": health,
                    "resources_match": resources_match,
                    "routing_match": routing_match,
                    "tce_ticket_id": str(detail["meta"].get("last_deploy_ticket", "")),
                }
            )
        return {
            "ready": len(clusters) == len(desired) and all(c["ready"] for c in clusters),
            "versions_match": len(clusters) == len(desired) and all(c["versions_match"] for c in clusters),
            "routing_match": all(c["routing_match"] for c in clusters),
            "clusters": clusters,
            "service_id": service_id,
            "platform_id": str(env["id"]),
            "due_time": service.get("due_time") if isinstance(service.get("due_time"), (str, int)) else None,
            "recycle_rule": service.get("recycle_rule") if isinstance(service.get("recycle_rule"), str) else None,
        }

    async def routing_match(self, cluster_id, cluster_name, *, runtime=None):
        overrides = (self.profile or {}).get("required_env_vars", {})
        if not overrides:
            return True
        if runtime is not None:
            # Verified TCE runtime format: [{env_name, env_value}], not a name/value map.
            rows = runtime.get("env_list")
            if not isinstance(rows, list):
                return False
            variables = {row.get("env_name"): row.get("env_value") for row in rows if isinstance(row, dict)}
            return all(str(variables.get(k)) == v for k, v in overrides.items())
        source = await request(
            self.http, DOMAIN, "GET", "/api/v4/suggest/cluster_by_id",
            params={"cluster_id": cluster_id, "env_type": "ppe", "standard_env": "online_cn"},
        )
        if (
            not isinstance(source, dict)
            or source.get("name") != cluster_name
            or source.get("base_cluster", {}).get("id") != cluster_id
            or source.get("psm") not in {None, "", self.psm}
        ):
            raise PlatformError("PPE 路由配置的集群身份不匹配")
        base = source.get("base_cluster", {})
        if source.get("new_env_vars_blocked") or base.get("new_env_vars_blocked"):
            raise PlatformError("PPE 路由配置无读取权限")
        variables = source.get("new_env_vars", base.get("new_env_vars"))
        return isinstance(variables, dict) and all(str(variables.get(k)) == v for k, v in overrides.items())

    async def pod_health(self, pod_name, cluster_id):
        if not re.fullmatch(r"[a-z0-9-]+", pod_name):
            raise PlatformError("PPE 实例名称无效")
        pod = await request(
            self.http, "tce;v1", "GET", f"/open-apis/v1/pods/{pod_name}/", params={"cluster_id": str(cluster_id)}
        )
        if (
            not pod or pod.get("cluster_id") != cluster_id
            or pod.get("psm") != self.psm or pod.get("status") != "Running"
        ):
            return False
        primary: dict[str, Any] = next((c for c in pod.get("containers", []) if c.get("type") == "primary"), {})
        port_info: dict[str, Any] = next((p for p in primary.get("ports", []) if p.get("primary")), {})
        port = port_info.get("accessPort")
        if not isinstance(port, (str, int)):
            return False
        try:
            ip = ipaddress.ip_address(pod.get("ipv6") or pod.get("ip"))
            if ip.is_loopback or ip.is_link_local or ip.is_unspecified or not 0 < int(port) <= 65535:
                return False
            host = f"[{ip}]" if ip.version == 6 else str(ip)
            # No platform credentials are attached to the actual Pod's health request.
            async with httpx.AsyncClient(trust_env=False, timeout=10, follow_redirects=False) as probe:
                health = (self.profile or {}).get("health_probe", {
                    "path": "/aeolus/api/v3/misc/healthz", "text": "Hello Aeolus!"
                })
                response = await probe.get(f"http://{host}:{int(port)}{health['path']}")
                if response.status_code != 200:
                    return False
                if "json" in health:
                    value = response.json()
                    return isinstance(value, dict) and all(value.get(k) == v for k, v in health["json"].items())
                return health["text"] in response.text[:4096]
        except (ValueError, TypeError, httpx.HTTPError):
            return False

    async def resources(self, name):
        service = await self.service(name)
        if not service:
            return {"clusters": []}
        if service.get("meta", {}).get("psm") != self.psm:
            raise PlatformError("PPE 服务归属不匹配")
        return {
            "service_id": str(service["meta"]["id"]),
            "clusters": [{"id": str(c["id"]), "name": c["name"]} for c in service.get("instances", [])],
        }

    async def approvals(self, name, resources):
        pending: list[dict[str, str]] = []
        service_id = resources.get("service_id")
        if not service_id:
            return pending
        for cluster in resources.get("clusters", []):
            detail = await request(
                self.http, "tce;v1", "GET", f"/open-apis/v1/services/{service_id}/clusters/{cluster['id']}/"
            )
            meta = (detail or {}).get("meta", {})
            if meta.get("psm") != self.psm or meta.get("env") != name or str(meta.get("id")) != cluster["id"]:
                raise PlatformError("PPE 目标集群归属不匹配")
            ticket_id = meta.get("last_deploy_ticket")
            if not ticket_id:
                continue
            ticket = await request(self.http, "tce;v1", "GET", f"/open-apis/v1/deployment/{ticket_id}/")
            if not ticket or ticket.get("status") != "running":
                continue
            if any(
                s.get("step_type") == "ConfirmStep"
                and s.get("status") == "pending"
                and str(s.get("id")) == str(ticket.get("current_step"))
                for s in ticket.get("steps", [])
            ):
                pending.append({"cluster_name": cluster["name"], "tce_ticket_id": str(ticket_id)})
        return pending

    async def absent(self, name, service_id, clusters):
        env = await self.environment(name)
        if env is not None and not environment_deleted(env):
            return False
        if service_id:
            for cluster in clusters:
                if (
                    await request(
                        self.http, "tce;v1", "GET", f"/open-apis/v1/services/{service_id}/clusters/{cluster['id']}/"
                    )
                    is not None
                ):
                    return False
            if await request(self.http, "tce;v1", "GET", f"/open-apis/v1/services/{service_id}/") is not None:
                return False
        return True


def precheck_summary(result):
    checks, skipped, blocked = set(), set(), set()

    def walk(value):
        if isinstance(value, dict):
            name = value.get("name", "")
            if isinstance(name, str) and name.startswith("PreCheck"):
                checks.add(name)
                if value.get("is_break") is True:
                    blocked.add(name)
            if value.get("err_msg"):
                skipped.add("partial_checks")
            if value.get("is_break") is True:
                blocked.add(name if isinstance(name, str) and name.startswith("PreCheck") else "platform_check")
            for key, item in value.items():
                if key not in {"meta", "new_env_vars", "headers", "token", "authorization"}:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result)
    if not checks:
        raise PlatformError("PPE 预检没有有效检查项")
    if blocked:
        raise PlatformError("PPE 预检被平台拦截：" + ", ".join(sorted(blocked)))
    return {"checks": sorted(checks), "partial": bool(skipped)}
