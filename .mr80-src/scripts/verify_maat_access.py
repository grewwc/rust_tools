#!/usr/bin/env python3
"""Read-only application-identity preflight. Never prints credentials or raw responses."""

import argparse
import asyncio
import json
import os

from conciliator.config import Settings
from conciliator.integrations.maat.auth import SdkMaatTokenProvider, application_http_client
from conciliator.integrations.maat.client import MAAT_GATEWAY_ORIGIN, template_revision
from conciliator.integrations.maat.control_plane import MaatOpenAPIControlPlaneClient, environment_url
from conciliator.utils.secrets import build_secrets_provider


async def verify(auth_proxy=None, snapshot_path=None):
    result = {}
    secrets = None
    try:
        settings = Settings()
        secrets = build_secrets_provider(settings)
        provider = SdkMaatTokenProvider(settings, secrets, proxy_url=auth_proxy or settings.maat_auth_proxy)
        await provider()
        result["jwt_exchange"] = "ok"
        operator = settings.maat_action_user.strip()
        result["action_user_configured"] = bool(operator)
        if not operator:
            operator = await provider.operator()
            result["identity_lookup"] = "ok"
        result["write_operator_preflight"] = "configured" if settings.maat_action_user.strip() else "not_configured"
        # The relay works for JWT issuance; the OpenAPI gateway is reachable directly.
        async with application_http_client(provider, MAAT_GATEWAY_ORIGIN, trust_env=False) as http:
            response = await http.post(
                "/api/open/devEnv/list",
                headers={"x-action-user": operator, "domain": "maat;v1"},
                json={
                    "projectName": "aeolus", "region": "CN", "name": settings.maat_baseline_environment,
                    "pageNo": 1, "pageSize": 100,
                },
            )
        result["http_status"] = response.status_code
        result["request_log_id"] = response.headers.get("x-tt-logid", "")
        try:
            payload = response.json()
        except ValueError:
            result["error"] = "non_json_response"
            return result
        if not isinstance(payload, dict):
            result["error"] = "invalid_response"
            return result
        if type(payload.get("code")) is int:
            result["business_code"] = payload["code"]
        if response.status_code == 403:
            result["call_approval_required"] = "调用审批" in str(payload.get("error", ""))
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            matched = [
                row for row in data["rows"]
                if isinstance(row, dict) and row.get("name") == settings.maat_baseline_environment
                and row.get("projectName") == "aeolus" and row.get("region") == "CN"
                and not row.get("isDeleted")
            ]
            result["reference_environment_matches"] = len(matched)
            if len(matched) == 1:
                result["baseline_revision"] = template_revision(matched[0])
                if snapshot_path:
                    async with application_http_client(provider, MAAT_GATEWAY_ORIGIN, trust_env=False) as http:
                        control = MaatOpenAPIControlPlaneClient(http, operator)
                        config = await control.project_configuration(matched[0]["projectId"])
                    project_config = {
                        key: config[key] for key in ("domainType", "defaultOpenPath") if key in config
                    }
                    environment_url(matched[0], project_config)
                    # Exclusive creation prevents an implicit refresh of a reviewed snapshot.
                    fd = os.open(snapshot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "w") as output:
                        json.dump({"environment": matched[0], "project_config": project_config}, output, indent=2)
                        output.write("\n")
                    result["snapshot_file"] = str(snapshot_path)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    finally:
        if secrets is not None:
            secrets.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-proxy", help="Proxy for JWT exchange only; the MAAT gateway is contacted directly")
    parser.add_argument("--save-snapshot", help="Save a local baseline JSON to a new file; never overwrites")
    args = parser.parse_args()
    result = asyncio.run(verify(args.auth_proxy, args.save_snapshot))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if (
        not result.get("error_type") and result.get("http_status") == 200 and result.get("business_code") == 0
        and result.get("reference_environment_matches") == 1
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
