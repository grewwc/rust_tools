"""A task environment builds changed code and reuses the companion's production image."""

import copy
import json
from pathlib import Path
from typing import Any

from conciliator.integrations.ppe.client import PpeClient, load_profile
from conciliator.integrations.ppe.scm import PlatformError


def load_environment_profile(*, repository="dp/aeolus", python_version=None):
    profile = copy.deepcopy(load_profile(repository=repository, python_version=python_version))
    companion = "dp/VizQueryService" if repository == "dp/aeolus" else "dp/aeolus"
    baseline = json.loads(Path(__file__).with_name("production_baselines.json").read_text())[companion]
    companion_profile = copy.deepcopy(load_profile(repository=companion))
    if baseline["psm"] != companion_profile["psm"] or baseline["source"]["environment"] != "prod":
        raise PlatformError("配套基准镜像的生产服务不匹配")
    companion_profile.update(
        production_baseline=baseline, baseline_artifact=baseline,
        scm_repo_id=baseline["scm_repo_id"], scm_repo_name=baseline["scm_repo_name"],
    )
    companion_profile["meta"].update(
        app_path=baseline["app_path"], scm_replace=False,
        scm_repos=[{**d, "main_repo": d["remote_id"] == baseline["scm_repo_id"],
                    "type": "scm", "sub_path": ""} for d in baseline["scm_repos"]],
    )
    companion_profile["image_recipe"]["reference_version"] = baseline["image_version"]
    profile["schema_version"] = 3
    profile["companion_profile"] = companion_profile
    return profile


def merge_owned_services(previous, current):
    """Keep every observed identity, including clusters lost from ENV during cleanup."""
    owned: dict[tuple[str, str], dict[str, Any]] = {}
    for service in [*previous, *current]:
        key = (service["psm"], service["service_id"])
        old = owned.get(key, {})
        clusters = {c["id"]: c for c in old.get("clusters", [])}
        clusters.update({c["id"]: c for c in service.get("clusters", [])})
        owned[key] = {**service, "clusters": list(clusters.values())}
    return list(owned.values())


class PpeEnvironmentClient(PpeClient):
    def __init__(self, http, profile, companion_artifact=None):
        super().__init__(http, profile)
        self.companion_artifact = companion_artifact
        self.profiles = [profile, profile["companion_profile"]]
        self.clients = {
            p["psm"]: PpeClient(
                http, p, environment_profile=profile,
                embedded_scm_repos=(companion_artifact or {}).get("embedded_scm_repos")
                if p.get("production_baseline") and not p["meta"]["scm_repos"] else None,
            ) for p in self.profiles
        }

    def artifact(self, profile, scm_version, image_version):
        if profile["psm"] == self.psm:
            return {"scm_version": scm_version, "image_version": image_version}
        if self.companion_artifact is not None:
            return self.companion_artifact
        # Preflight verifies both source configurations before any build/write.
        return {
            "scm_version": profile["baseline_artifact"]["version"],
            "image_version": profile["image_recipe"]["reference_version"],
        }

    async def prepare(
        self, profile, identity, version, *, image_version, operator, existing=None,
        cluster_ids=None, missing_names=None,
    ):
        if cluster_ids is not None or missing_names is not None:
            raise PlatformError("双服务部署必须分别核对集群集合")
        body = None
        deployments = {}
        for service_profile in self.profiles:
            client = self.clients[service_profile["psm"]]
            artifact = self.artifact(service_profile, version, image_version)
            service_body = await client.prepare(
                service_profile, identity, artifact["scm_version"],
                image_version=artifact["image_version"], operator=operator,
                existing=existing,
            )
            if service_body is None:
                continue
            body = body or service_body
            deployments.update(service_body["deployment"])
        if body is None:
            return None
        body["deployment"] = deployments
        return body

    async def resources(self, name):
        services = {}
        clusters: list[dict[str, Any]] = []
        owned = []
        for profile in self.profiles:
            psm = profile["psm"]
            result = await self.clients[psm].resources(name)
            services[psm] = result
            clusters.extend({**c, "psm": psm, "service_id": result["service_id"]} for c in result["clusters"])
            if result.get("service_id"):
                owned.append({"psm": psm, **result})
        return {"services": services, "clusters": clusters, "owned_services": owned}

    async def observe(self, name, identity, platform_id, profile, *, image_version, scm_version):
        services = {}
        clusters: list[dict[str, Any]] = []
        owned = []
        for service_profile in self.profiles:
            psm = service_profile["psm"]
            artifact = self.artifact(service_profile, scm_version, image_version)
            result = await self.clients[psm].observe(
                name, identity, platform_id, service_profile,
                image_version=artifact["image_version"], scm_version=artifact["scm_version"],
            )
            if result.get("absent"):
                return {"absent": True, "ready": False, "clusters": []}
            services[psm] = result
            clusters.extend({**c, "psm": psm, "service_id": result["service_id"]} for c in result["clusters"])
            if result.get("service_id"):
                owned.append({"psm": psm, "service_id": result["service_id"], "clusters": result["clusters"]})
        return {
            "services": services, "clusters": clusters, "owned_services": owned,
            "platform_id": str(platform_id),
            "ready": all(s["ready"] for s in services.values()),
            "versions_match": all(s.get("versions_match") for s in services.values()),
            "routing_match": all(s.get("routing_match") for s in services.values()),
        }

    async def approvals(self, name, resources):
        pending: list[dict[str, str]] = []
        for psm, service in resources.get("services", {}).items():
            pending.extend(
                {**approval, "psm": psm}
                for approval in await self.clients[psm].approvals(name, service)
            )
        return pending

    async def absent(self, name, owned_services, clusters):
        # Even an ENV tombstone is insufficient: each historical TCE identity must be gone.
        for service in owned_services or []:
            if not await self.clients[service["psm"]].absent(name, service["service_id"], service["clusters"]):
                return False
        # An environment that was never observed still needs an explicit absence check.
        return await super().absent(name, None, [])
