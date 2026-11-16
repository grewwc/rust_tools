"""MAAT control-plane contracts verified against platform callers and the current frontend.

The caller supplies an authenticated HTTP client. This module does not acquire credentials.
OpenAPI operations and console-only operations deliberately use different paths.
"""

import copy
import re
from urllib.parse import urlsplit

from conciliator.integrations.maat.client import (
    MaatClient,
    MaatError,
    MaatWriteUnknown,
    request_json,
    template_revision,
)

CONFIG_FIELDS = {
    "resourceType", "scmVersion", "branchName", "scmList", "apiProxyList", "devChildList",
    "packages", "templateConfig", "hasIgnoreManifestLogin", "kuraPackagesPrefetch", "volcengineEnvType",
}
RECORD_FIELDS = {
    "_id", "__v", "createdAt", "updatedAt", "name", "projectId", "projectName", "region",
    "createUser", "updateUser", "isDeleted", "status", "desc", "watchUserList", "larkGroupList", "maatV1Id",
}
DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def environment_body(template, plan, *, name, action_user):
    """Build a fixed-version environment payload, preserving reviewed opaque configuration in memory."""
    if not DNS_LABEL.fullmatch(name) or not action_user:
        raise MaatError("invalid environment name or operator")
    if plan.get("source_revision") != template_revision(template):
        raise MaatError("template changed after the route plan was reviewed")
    if set(template) - CONFIG_FIELDS - RECORD_FIELDS:
        raise MaatError("unreviewed template fields require an explicit contract decision")
    if not template.get("projectName") or template.get("isDeleted"):
        raise MaatError("template project or lifecycle is invalid")
    body = {key: copy.deepcopy(template[key]) for key in CONFIG_FIELDS if key in template}
    body.update(
        projectId=plan["project_id"], projectName=template["projectName"], region=plan["region"], name=name,
        scmList=copy.deepcopy(plan["scm_list"]), apiProxyList=copy.deepcopy(plan["api_proxy_list"]),
        createUser=action_user, watchUserList=[action_user], larkGroupList=[], desc=f"Conciliator: {name}",
    )
    return body


def environment_url(environment, project_config):
    """Verified CN generic-domain rule. Other regions/domain modes need their own reviewed mapping."""
    name = environment.get("name", "")
    if (
        not DNS_LABEL.fullmatch(name)
        or environment.get("region") != "CN"
        or project_config.get("domainType") != "generic"
    ):
        raise MaatError("environment domain mode or region is not verified")
    path = project_config.get("defaultOpenPath") or ""
    if path and (not path.startswith("/") or path.startswith("//") or urlsplit(path).scheme or "\\" in path):
        raise MaatError("invalid environment default path")
    return f"https://{name}.maat.bytedance.net{path}"


def check_environment_identity(environment, expected, action_user):
    for key in ("_id", "projectId", "projectName", "region", "name"):
        if not expected.get(key) or environment.get(key) != expected[key]:
            raise MaatError("environment ownership identity does not match")
    # updateUser alone is not proof of creation ownership.
    if environment.get("createUser") != action_user:
        raise MaatError("environment was not created by the configured operator")


def routes_match_endpoint(routes, bindings, *, domain, host, port, protocol="http"):
    """Compare effective gateway routes, not just the environment's stored apiProxyList."""
    if not bindings:
        return False
    for pathname, match_type, _old_api_id in bindings:
        matches = []
        for route in routes:
            incoming = route.get("incoming") or {}
            path = incoming.get("path") if isinstance(incoming, dict) else None
            if (
                route.get("domain") == domain and isinstance(path, dict)
                and path.get("name") == pathname and path.get("type") == match_type
            ):
                matches.append(route)
        if len(matches) != 1:
            return False
        try:
            target = urlsplit(matches[0].get("proxy", {}).get("target", ""))
            target_port = target.port or {"http": 80, "https": 443}.get(target.scheme)
            if (
                target.username or target.password or target.scheme != protocol
                or target.hostname != host.lower() or target_port != port
            ):
                return False
        except (TypeError, ValueError, AttributeError):
            return False
    return True


class MaatControlPlaneClient(MaatClient):
    def __init__(self, client, action_user):
        if not action_user:
            raise ValueError("MAAT operator is required")
        self.client = client
        self.action_user = action_user

    async def _post(self, operation, body, *, write=False):
        return await request_json(
            self.client, "/api/open/" + operation, body,
            headers={"x-action-user": self.action_user, "domain": "maat;v1"}, write=write,
        )

    async def _console(self, operation, body, *, write=False):
        return await request_json(
            self.client, "/api/" + operation, body, params={"__siteEnv__": "CN"}, write=write,
        )

    async def _cleanup_post(self, operation, body, *, write=False):
        return await self._console(operation, body, write=write)

    async def environment_detail(self, environment_id):
        data = await self._post("devEnv/detail", {"_id": environment_id})
        if not isinstance(data, dict) or data.get("_id") != environment_id:
            raise MaatError("environment detail has an invalid identity")
        return data

    async def environment_record(self, expected):
        return await self.environment_detail(expected["_id"])

    async def create_environment(self, body):
        # The full request must be recorded as a digest/intention by the coordinator before calling.
        if "_id" in body or not DNS_LABEL.fullmatch(body.get("name", "")):
            raise MaatError("new environment must have a fresh, valid name")
        data = await self._post("devEnv/create", body, write=True)
        if not isinstance(data, dict) or not isinstance(data.get("_id"), str) or not data["_id"]:
            raise MaatWriteUnknown("environment create returned no resource identity")
        return data["_id"]  # Persist this before attempting any subsequent read or update.

    async def apply_environment(self, expected, body):
        current = await self.environment_record(expected)
        check_environment_identity(current, expected, self.action_user)
        if current.get("isDeleted") or any(
            body.get(key) != expected[key] for key in ("projectId", "projectName", "region", "name")
        ):
            raise MaatError("cannot apply configuration to a deleted or different environment")
        return await self._post("devEnv/update", {**body, "_id": expected["_id"]}, write=True)

    async def deploy_gateway(self, expected):
        current = await self.environment_record(expected)
        check_environment_identity(current, expected, self.action_user)
        if current.get("isDeleted"):
            raise MaatError("cannot publish a deleted environment")
        return await self._console("devEnv/triggleDeployGateway", {"_id": expected["_id"]}, write=True)

    async def deploy_files(self, expected):
        current = await self.environment_record(expected)
        check_environment_identity(current, expected, self.action_user)
        if current.get("isDeleted"):
            raise MaatError("cannot publish files for a deleted environment")
        return await self._console("devEnv/triggleDeployFile", {"_id": expected["_id"]}, write=True)

    async def soft_delete_environment(self, expected):
        """Soft-delete metadata only. This is NOT evidence that gateway routes have been revoked."""
        current = await self.environment_record(expected)
        check_environment_identity(current, expected, self.action_user)
        if current.get("isDeleted"):
            return current
        return await self._post("devEnv/update", {"_id": expected["_id"], "isDeleted": True}, write=True)

    async def remove_environment(self, expected):
        """Soft-delete the owned environment record; this does not verify route withdrawal."""
        current = await self.environment_record(expected)
        check_environment_identity(current, expected, self.action_user)
        return await self._cleanup_post("devEnv/remove", {"_id": expected["_id"]}, write=True)

    async def domain_routes(self, domain):
        if not re.fullmatch(r"[a-z0-9-]+\.maat\.bytedance\.net", domain):
            raise MaatError("gateway inspection requires one exact CN environment domain")
        data = await self._console("maatPro/domainRoutesInfo", {"region": "CN", "domain": domain})
        if not isinstance(data, list) or any(not isinstance(route, dict) for route in data):
            raise MaatError("gateway route response schema is incomplete")
        return data

    async def proxy_references(self, proxy_id):
        data = await self._cleanup_post("devApi/getUsedEnvList", {"_id": proxy_id})
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise MaatError("proxy reference response schema is incomplete")
        return data

    async def find_proxy(self, *, project_id, name):
        found: list[dict] = []
        page = 1
        while True:
            data = await self._cleanup_post(
                "devApi/list", {"projectId": project_id, "keyword": name, "pageNo": page, "pageSize": 100},
            )
            if (
                not isinstance(data, dict) or type(data.get("total")) is not int
                or not isinstance(data.get("rows"), list)
                or any(not isinstance(row, dict) for row in data["rows"])
            ):
                raise MaatError("proxy listing schema is incomplete")
            found.extend(row for row in data["rows"] if row.get("projectId") == project_id and row.get("name") == name)
            if page * 100 >= data["total"]:
                break
            if not data["rows"]:
                raise MaatError("proxy listing ended before its reported total")
            page += 1
        if len(found) > 1:
            raise MaatError("proxy name must resolve to at most one resource")
        return found[0] if found else None

    async def remove_proxy(self, expected):
        current = await self.find_proxy(project_id=expected["projectId"], name=expected["name"])
        keys = ("_id", "projectId", "name", "ip", "port", "protocol", "type")
        if current is None or any(current.get(key) != expected.get(key) or not expected.get(key) for key in keys):
            raise MaatError("proxy ownership or endpoint does not match")
        if current.get("createUser") != self.action_user:
            raise MaatError("proxy was not created by the configured operator")
        if current["type"] != "dev" or await self.proxy_references(current["_id"]):
            raise MaatError("cannot remove a shared or referenced proxy")
        return await self._cleanup_post("devApi/remove", {"_id": current["_id"]}, write=True)


class MaatOpenAPIControlPlaneClient(MaatControlPlaneClient):
    """Public gateway operations, including record cleanup without route withdrawal."""

    async def _cleanup_post(self, operation, body, *, write=False):
        if operation == "devApi/list":
            body = {**body, "name": body["keyword"]}
            del body["keyword"]
        return await self._post(operation, body, write=write)

    async def environment_record(self, expected):
        record = await self.get_environment(expected["projectName"], expected["region"], expected["name"])
        if record.get("_id") != expected["_id"] or record.get("projectId") != expected["projectId"]:
            raise MaatError("environment listing identity does not match")
        return record

    async def project_configuration(self, project_id):
        config = await self._post("devConfig/get", {"projectId": project_id})
        if not isinstance(config, dict) or config.get("projectId") != project_id:
            raise MaatError("project configuration identity does not match")
        return config

    async def _console(self, operation, body, *, write=False):
        raise MaatError("MAAT console publishing and route inspection remain TODO")
