"""Documented MAAT operations. Writes are never automatically retried."""

import copy
import hashlib
import json

import httpx

MAAT_GATEWAY_ORIGIN = "https://bc-cn-gw.bytedance.net"


class MaatError(RuntimeError):
    pass


class MaatWriteUnknown(MaatError):
    """The remote side may have accepted a write; reconcile before any retry."""

    resource_id: str | None = None


class MaatClient:
    def __init__(self, client: httpx.AsyncClient, token_provider, action_user: str):
        self.client = client
        self.token_provider = token_provider
        self.action_user = action_user

    async def _post(self, operation, body, *, write=False):
        from conciliator.integrations.maat.auth import application_token

        token = await application_token(self.token_provider)
        return await request_json(
            self.client,
            "/api/open/" + operation,
            body,
            headers={"X-Jwt-Token": token, "x-action-user": self.action_user, "domain": "maat;v1"},
            write=write,
        )

    async def get_environment(self, project_name, region, name):
        return await find_environment(self._post, project_name, region, name)

    async def lookup_environment(self, project_name, region, name):
        return await find_environment(self._post, project_name, region, name, allow_missing=True)

    async def create_dev_api(
        self, *, project_id, name, host, port, protocol="http", proxy_timeout=None, is_socket=False
    ):
        if not 1024 <= port <= 65535 or protocol not in {"http", "https"}:
            raise ValueError("invalid backend endpoint")
        body = {
            "projectId": project_id,
            "name": name,
            "ip": host,
            "port": str(port),
            "protocol": protocol,
            "type": "dev",
            "isSocket": is_socket,
        }
        if proxy_timeout is not None:
            body["timeout"] = proxy_timeout
        data = await self._post("devApi/create", body, write=True)
        if not isinstance(data, dict) or not isinstance(data.get("_id"), str) or not data["_id"]:
            raise MaatWriteUnknown("MAAT create returned no resource identity")
        expected = {k: body[k] for k in ("projectId", "name", "ip", "port", "protocol", "type")}
        if any(data.get(k) != v for k, v in expected.items()):
            # Preserve only the resource ID for recovery; never repeat create to repair a response.
            error = MaatWriteUnknown("MAAT created resource could not be verified")
            error.resource_id = data["_id"]
            raise error
        return data


async def request_json(client, path, body, *, headers=None, params=None, write=False):
    """Shared transport for verified OpenAPI and console operations; never retries a write."""
    try:
        response = await client.post(path, headers=headers, params=params, json=body)
    except httpx.HTTPError as exc:
        error = MaatWriteUnknown if write else MaatError
        raise error("MAAT transport failed; no automatic write retry") from exc
    # Do not propagate raw headers, response bodies or arbitrary platform error strings.
    if response.status_code >= 500:
        error = MaatWriteUnknown if write else MaatError
        raise error("MAAT server result is unknown")
    if response.status_code >= 400:
        raise MaatError(f"MAAT request rejected with HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        error = MaatWriteUnknown if write else MaatError
        raise error("MAAT response is not JSON") from exc
    if not isinstance(payload, dict) or type(payload.get("code")) is not int:
        error = MaatWriteUnknown if write else MaatError
        raise error("MAAT response lacks a numeric business code")
    if payload["code"] != 0 or payload.get("success") is False:
        raise MaatError("MAAT rejected the operation")
    return payload.get("data")


async def find_environment(post, project_name, region, name, *, allow_missing=False):
    found: list[dict] = []
    page = 1
    while True:
        data = await post(
            "devEnv/list",
            {
                "projectName": project_name,
                "region": region,
                "name": name,
                "pageNo": page,
                "pageSize": 100,
            },
        )
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("rows"), list)
            or type(data.get("total")) is not int
        ):
            raise MaatError("MAAT environment listing schema is incomplete")
        rows = data["rows"]
        if any(not isinstance(row, dict) for row in rows):
            raise MaatError("MAAT environment listing contains invalid rows")
        found.extend(
            r
            for r in rows
            if r.get("name") == name
            and r.get("projectName") == project_name
            and r.get("region") == region
            and not r.get("isDeleted")
        )
        if page * 100 >= data["total"]:
            break
        if not rows:
            raise MaatError("MAAT listing ended before its reported total")
        page += 1
    if not found and allow_missing:
        return None
    if len(found) != 1:
        raise MaatError("MAAT environment must match exactly once")
    return found[0]


def template_revision(template):
    """Bind a reviewed full snapshot, including directives, without exposing its contents."""
    return hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()


def environment_plan(template, replacements):
    """Build an internal plan, not an undocumented devEnv/create request DTO.

    Replacements map (exact pathname, match type, expected apiId) to the new apiId.
    The input must be a complete server-side template, never CLI's redacted display projection.
    """
    if "[REDACTED]" in json.dumps(template):
        raise MaatError("redacted template cannot be used as writable configuration")
    if not template.get("_id") or not template.get("projectId") or not template.get("region"):
        raise MaatError("template identity is incomplete")
    if not template.get("scmList") or not isinstance(template.get("apiProxyList"), list):
        raise MaatError("template resource and proxy configuration is incomplete")
    resources = copy.deepcopy(template["scmList"])
    for resource in resources:
        if resource.get("resourceType") != "scm" or not resource.get("scmVersion"):
            raise MaatError("template must provide fixed SCM versions")
        # Verified in the current MAAT frontend: an absent persisted path is presented as '/'.
        resource.setdefault("path", "/")
        if not isinstance(resource["path"], str) or not resource["path"].startswith("/"):
            raise MaatError("template static resource path is invalid")
    rules = copy.deepcopy(template["apiProxyList"])
    if not replacements or any(not isinstance(value, str) or not value for value in replacements.values()):
        raise MaatError("replacement must reference explicit proxy rules")
    for binding in replacements:
        if (
            not isinstance(binding, tuple)
            or len(binding) != 3
            or not all(isinstance(value, str) and value for value in binding)
            or binding[1] not in {"prefix", "regexp"}
        ):
            raise MaatError("replacement requires pathname, match type and expected proxy identity")
        # Duplicate paths are ambiguous even when they currently point at different targets.
        matches = [r for r in rules if (r.get("pathname"), r.get("pathMatchType")) == binding[:2]]
        if len(matches) != 1 or matches[0].get("apiId") != binding[2]:
            raise MaatError("reviewed proxy binding no longer matches the template")
    for rule in rules:
        binding = (rule.get("pathname"), rule.get("pathMatchType"), rule.get("apiId"))
        rule["apiId"] = replacements.get(binding, rule.get("apiId"))
        # IDs belong to the old environment; their regeneration is the create adapter's responsibility.
        rule.pop("_id", None)
    plan = {
        "source_environment_id": template["_id"],
        "source_revision": template_revision(template),
        "project_id": template["projectId"],
        "region": template["region"],
        "scm_list": resources,
        "api_proxy_list": rules,
    }
    plan["revision"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def service_environment_plan(template, repository, new_api_id, *, bindings, expected_template_revision):
    """Use explicitly reviewed service routes; shared proxy IDs do not imply service ownership."""
    if template_revision(template) != expected_template_revision:
        raise MaatError("template revision changed; review the service bindings again")
    selected = bindings.get(repository)
    if not selected:
        raise MaatError("no reviewed proxy bindings for this repository")
    return environment_plan(template, {binding: new_api_id for binding in selected})
