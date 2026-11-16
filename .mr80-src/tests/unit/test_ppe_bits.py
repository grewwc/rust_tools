import copy
import json

import httpx
import pytest

from conciliator.integrations.maat.auth import application_http_client
from conciliator.integrations.ppe.bits import ORIGIN, BitsScmClient
from conciliator.integrations.ppe.client import load_profile
from conciliator.integrations.ppe.scm import PlatformError

CONFIG = {"pipeline_id": "1251261261058", "space_id": "284273157378"}
SHA = "a" * 40


async def test_py27_formal_pipeline_binds_repo_and_rejects_python3_artifacts():
    profile = load_profile(python_version="2.7")
    config = profile["bits_scm"]
    detail = definition()
    detail["pipeline_id"] = int(config["pipeline_id"])
    detail["pipeline"]["stages"][0]["jobs"][0]["inputs"]["scm_configs"][0]["scm_repo_name"] = "data/dp/aeolus"
    run = successful_run()
    run["pipeline_id"] = int(config["pipeline_id"])
    run["jobs"][0]["job_atom"]["inputs"]["scm_configs"][0]["scm_repo_name"] = "data/dp/aeolus"
    artifact = run["jobs"][0]["job_atom"]["views"][0]["value"]["versions"][0]
    artifact["repo_id"] = 4863

    def transport(request):
        assert request.headers["username"] == "huaguojin"
        return httpx.Response(200, json={"pipeline_run": run} if request.url.path.endswith("/42")
                              else {"pipeline": detail})

    async with httpx.AsyncClient(base_url=ORIGIN, transport=httpx.MockTransport(transport)) as http:
        client = BitsScmClient(http).for_profile(profile)
        plan = await client.plan(config, SHA, "huaguojin")
        assert plan["pipeline_id"] == 1254638941442
        result = await client.run(config, "42", SHA, "huaguojin")
        assert result["bits_success"] and result["scm_version_id"] == str(artifact["version_id"])
        artifact["repo_id"] = 572600
        with pytest.raises(PlatformError, match="目标仓库制品身份"):
            await client.run(config, "42", SHA, "huaguojin")


def definition():
    return {
        "pipeline_id": 1251261261058, "space_id": 284273157378,
        "pipeline": {
            "stages": [{"jobs": [{
                "id": "scm_compile", "uses": "job_atoms/scm_compile_beta",
                "inputs": {"scm_configs": [{
                    "revision": "{{custom.commit_hash}}", "pub_base": "commit_base",
                    "scm_repo_name": "data/dp/aeolus_ppe", "scm_type": "test",
                }]},
            }]}],
            "var_group": {"group_id": 20660014939906, "version": 3,
                          "var_definitions": [{"name": "commit_hash"}]},
        },
    }


def successful_run():
    job = copy.deepcopy(definition()["pipeline"]["stages"][0]["jobs"][0])
    job["inputs"]["scm_configs"][0]["revision"] = SHA
    return {
        "pipeline_id": 1251261261058, "run_id": 42, "run_status": 8,
        "jobs": [{
            "job_id": "scm_compile", "job_status": 14,
            "job_atom": {**job, "views": [{"value": {"versions": [{
                "repo_id": 572600, "region": "cn", "version": "1.0.0.2139", "version_id": 173088716,
            }]}}]},
        }],
    }


@pytest.mark.asyncio
async def test_application_auth_uses_formal_openapi_and_live_variable_group():
    requests = []

    async def token():
        return "application-jwt"

    def transport(request):
        requests.append(request)
        assert request.headers["x-jwt-token"] == "application-jwt"
        assert request.headers["username"] == "huaguojin"
        assert "x-platform-proxy-user" not in request.headers
        if request.method == "GET":
            return httpx.Response(200, json={"pipeline": definition()})
        return httpx.Response(200, json={"pipeline_run": {
            "pipeline_id": 1251261261058, "run_id": 42, "run_seq": 2,
        }})

    async with application_http_client(token, ORIGIN, transport=httpx.MockTransport(transport)) as http:
        client = BitsScmClient(http)
        body = await client.plan(CONFIG, SHA, "huaguojin")
        assert body["custom_vars"] == [{
            "name": "custom.commit_hash", "value": {"text": SHA},
            "group_id": 20660014939906, "group_version": 3,
        }]
        assert await client.submit(body, "huaguojin") == {"bits_run_id": "42", "bits_run_seq": "2"}
    assert requests[0].url.path == "/api/v1/pipelines_open/1251261261058"
    assert requests[1].url.path == "/api/v1/pipelines/open/1251261261058/run"
    assert json.loads(requests[1].content) == body


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["deployment", "branch", "repository"])
async def test_pipeline_scope_drift_is_rejected_before_write(mutation):
    detail = definition()
    jobs = detail["pipeline"]["stages"][0]["jobs"]
    if mutation == "deployment":
        jobs.append({"uses": "job_atoms/tce_deploy"})
    elif mutation == "branch":
        jobs[0]["inputs"]["scm_configs"][0]["pub_base"] = "branch_base"
    else:
        jobs[0]["inputs"]["scm_configs"][0]["scm_repo_name"] = "data/dp/aeolus"
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(200, json={"pipeline": detail})

    async with httpx.AsyncClient(base_url=ORIGIN, transport=httpx.MockTransport(transport)) as http:
        with pytest.raises(PlatformError):
            await BitsScmClient(http).plan(CONFIG, SHA, "huaguojin")
    assert all(r.method == "GET" for r in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    httpx.Response(403, text="secret server error"),
    httpx.Response(200, json={"pipeline_run": {"pipeline_id": 1, "run_id": 42}}),
    httpx.Response(200, text="invalid"),
])
async def test_ambiguous_writes_are_quarantined_without_raw_error(response):
    async with httpx.AsyncClient(base_url=ORIGIN, transport=httpx.MockTransport(lambda r: response)) as http:
        with pytest.raises(PlatformError) as exc:
            await BitsScmClient(http).submit({"pipeline_id": 1251261261058}, "huaguojin")
    assert exc.value.uncertain
    assert "secret" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["none", "run_id", "sha", "version_repo", "skipped"])
async def test_original_run_requires_success_and_exact_artifact_identity(mutation):
    run = successful_run()
    if mutation == "run_id":
        run["run_id"] = 43
    elif mutation == "sha":
        run["jobs"][0]["job_atom"]["inputs"]["scm_configs"][0]["revision"] = "b" * 40
    elif mutation == "version_repo":
        run["jobs"][0]["job_atom"]["views"][0]["value"]["versions"][0]["repo_id"] = 4863
    elif mutation == "skipped":
        run["jobs"][0]["job_status"] = 10
    async with httpx.AsyncClient(
        base_url=ORIGIN, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"pipeline_run": run}))
    ) as http:
        client = BitsScmClient(http)
        if mutation == "none":
            result = await client.run(CONFIG, "42", SHA, "huaguojin")
            assert result["bits_success"] and result["scm_version_id"] == "173088716"
        else:
            with pytest.raises(PlatformError):
                await client.run(CONFIG, "42", SHA, "huaguojin")
