import copy
import json

import httpx
import pytest

from conciliator.integrations.ppe.bits import BitsScmClient
from conciliator.integrations.ppe.client import PpeClient, deployment_body, environment_name, load_profile
from conciliator.integrations.ppe.icm import image_body
from conciliator.integrations.ppe.scm import PlatformError, ScmClient

IDENTITY = "b" * 32
SHA = "d" * 40
VERSION = "1.0.0.6127"


def test_vqs_uses_online_recipe_and_separate_one_cluster_upgrade():
    profile = load_profile(repository="dp/VizQueryService")
    original = copy.deepcopy(profile)
    image = image_body(profile, VERSION, "huaguojin")
    assert image["psm"] == "data.vqs.api"
    assert image["base_image"] == "toutiao.debian.python3:latest"
    assert image["load_path"] == "toutiao/app/data/vqs/run/prod_run.sh"
    assert len(image["repos"]) == 18
    assert {p["package_type"] for p in image["packages"]} == {"apt-get", "pip"}
    assert next(d for d in image["repos"] if d["remote_id"] == "53349")["version"] == VERSION
    body = deployment_body(profile, IDENTITY, VERSION, image_version="1.0.0.2140")
    assert body["env"]["name"] == "ppe_ai_vqs_" + IDENTITY[:16]
    assert set(body["deployment"]) == {"data.vqs.api"}
    cluster = body["deployment"]["data.vqs.api"]["meta"]["clusters"][0]
    assert (cluster["cpu"], cluster["mem"], cluster["base_cluster_id"]) == (16, 32, 217131828)
    upgrade = deployment_body(
        profile, IDENTITY, VERSION, image_version="1.0.0.2140", cluster_ids={"default": "100"}
    )["deployment"]["data.vqs.api"]["meta"]
    assert upgrade["clusters"] == [{"id": 100}]
    assert upgrade["image_version"] == "1.0.0.2140"
    assert profile == original
    assert "py311" not in json.dumps(body)


@pytest.mark.asyncio
async def test_vqs_build_clients_reject_other_repository_artifacts():
    profile = load_profile(repository="dp/VizQueryService")
    run = {
        "pipeline_id": int(profile["bits_scm"]["pipeline_id"]), "run_id": 42, "run_status": 8,
        "jobs": [{
            "job_id": "scm_compile", "job_status": 14, "job_atom": {
                "inputs": {"scm_configs": [{
                    "revision": SHA, "scm_repo_name": "data/dp/vqs", "pub_base": "commit_base", "scm_type": "test"
                }]},
                "views": [{"value": {"versions": [{
                    "repo_id": 572600, "region": "cn", "version": VERSION, "version_id": 10
                }]}}],
            },
        }],
    }
    async with httpx.AsyncClient(
        base_url="https://platform.test",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"pipeline_run": run})),
    ) as http:
        client = BitsScmClient(http).for_profile(profile)
        with pytest.raises(PlatformError):
            await client.run(profile["bits_scm"], "42", SHA, "huaguojin")
        run["jobs"][0]["job_atom"]["views"][0]["value"]["versions"][0]["repo_id"] = 53349
        assert (await client.run(profile["bits_scm"], "42", SHA, "huaguojin"))["scm_version_id"] == "10"

    def transport(req):
        assert req.url.path == f"/api/repos/53349/versions/{VERSION}"
        return httpx.Response(200, json={
            "repos": 572600, "version": VERSION, "id": 10, "base_commit_hash": SHA,
            "status": "build_ok", "type": "test", "arch": ["x86_64"],
        })

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        with pytest.raises(PlatformError):
            await ScmClient(http, "huaguojin", 53349).version(VERSION)


@pytest.mark.asyncio
async def test_vqs_source_variables_stay_in_memory_and_reference_cannot_be_written():
    profile = load_profile(repository="dp/VizQueryService")
    calls = []

    def transport(req):
        calls.append(req)
        assert req.method == "GET"
        if req.url.path == "/api/v4/instance_meta":
            assert req.url.params["service"] == "data.vqs.api"
            return httpx.Response(200, json={
                "meta": {"psm": "data.vqs.api"}, "instances": [{"name": "default", "id": 217131828}]
            })
        return httpx.Response(200, json={
            "name": "default", "base_cluster": {
                "id": 217131828, "new_env_vars": {"SOURCE_CONFIG": "in-memory-only"}
            }
        })

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        client = PpeClient(http).for_profile(profile)
        body = await client.prepare(
            profile, IDENTITY, VERSION, image_version="1.0.0.2140", operator="huaguojin"
        )
        assert body["deployment"]["data.vqs.api"]["meta"]["clusters"][0]["new_env_vars"] == {
            "SOURCE_CONFIG": "in-memory-only", "AEOLUS_SERVICE_DISCOVERY": "0",
        }
        assert "in-memory-only" not in json.dumps(profile)
        body["env"]["name"] = profile["reference_environment"]
        with pytest.raises(PlatformError):
            await client.submit(body, create=False)
        with pytest.raises(PlatformError):
            await client.remove(environment_name(IDENTITY), expected_id="1", identity=IDENTITY)
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("code,healthy", [("aeolus/ok", True), ("aeolus/error", False)])
async def test_vqs_health_requires_application_json_without_platform_credentials(monkeypatch, code, healthy):
    profile = load_profile(repository="dp/VizQueryService")
    original_client = httpx.AsyncClient

    def platform(req):
        return httpx.Response(200, json={
            "cluster_id": 100, "psm": "data.vqs.api", "status": "Running", "ipv6": "fd00::123",
            "containers": [{"type": "primary", "ports": [{"primary": True, "accessPort": "12355"}]}],
        })

    def probe(req):
        assert req.url.path == "/healthz"
        assert "x-jwt-token" not in req.headers
        return httpx.Response(200, json={"code": code})

    async with original_client(
        base_url="https://platform.test", headers={"X-Jwt-Token": "application"},
        transport=httpx.MockTransport(platform),
    ) as http:
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
            **kwargs, transport=httpx.MockTransport(probe)
        ))
        assert await PpeClient(http, profile).pod_health("pod-one", 100) is healthy
