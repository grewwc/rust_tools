import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from conciliator.integrations.ppe.client import (
    PSM,
    PpeClient,
    deployment_body,
    environment_deleted,
    environment_name,
    load_profile,
    precheck_summary,
)
from conciliator.integrations.ppe.scm import PlatformError

IDENTITY = "a" * 32
VERSION = "1.0.0.1542"
IMAGE = "1.0.2.555"


@pytest.mark.parametrize("stamp,deleted", [(None, False), (0, False), ("1793781794", False), (1793781794, True)])
def test_environment_deleted_requires_positive_numeric_timestamp(stamp, deleted):
    assert environment_deleted({"deleted_at": stamp}) is deleted


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", ["cluster", "service", None])
async def test_deleted_environment_still_requires_all_tce_resources_absent(remaining):
    calls = []

    def transport(req):
        calls.append(req.url.path)
        if req.url.path.startswith("/api/v4/env/"):
            return httpx.Response(200, json={"code": 0, "data": {"id": 123, "deleted_at": 1793781794}})
        exists = (remaining == "cluster" and "/clusters/" in req.url.path) or (
            remaining == "service" and req.url.path == "/open-apis/v1/services/456/"
        )
        return httpx.Response(200, json={"id": 789}) if exists else httpx.Response(404)

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        client = PpeClient(http)
        assert await client.absent(environment_name(IDENTITY), "456", [{"id": "789"}]) is (remaining is None)
    assert "/open-apis/v1/services/456/clusters/789/" in calls


def test_template_freezes_dependencies_image_and_separates_cluster_repair():
    profile = load_profile()
    original = copy.deepcopy(profile)
    body = deployment_body(profile, IDENTITY, VERSION, image_version=IMAGE, operator="huaguojin")
    meta = body["deployment"][PSM]["meta"]
    assert len(meta["clusters"]) == 2
    assert meta["clusters"][0]["name"] == "online"
    assert meta["clusters"][0]["dc_infos"] == [{"count": 1, "idc": "LF"}]
    assert all(c["cpu"] == 8 and c["mem"] == 32 and c["dc_infos"] == [{"count": 1, "idc": "LF"}]
               for c in meta["clusters"])
    assert body["env"]["env_managers"] == ["huaguojin"]
    assert body["env"]["auth_type"] == 1
    assert len(meta["scm_repos"]) == 11
    for before, after in zip(profile["meta"]["scm_repos"], meta["scm_repos"], strict=True):
        assert after["version"] == (VERSION if before["main_repo"] else before["version"])
    assert all(c["image_tag"] == "tce" and c["image_version"] == IMAGE for c in meta["clusters"])
    assert all(c["virtual_cluster"] == "PPE/ipv6" and c["ipv6only_enable"] == "true" for c in meta["clusters"])
    assert profile == original
    repair = deployment_body(profile, IDENTITY, VERSION, image_version=IMAGE, missing_names=["online"])
    assert repair["deployment"][PSM]["action"] == "create_cluster"
    assert [c["name"] for c in repair["deployment"][PSM]["meta"]["clusters"]] == ["online"]
    upgrade = deployment_body(
        profile, IDENTITY, VERSION, image_version=IMAGE, cluster_ids={"online": "100", "query_online_mirror": "101"}
    )
    assert upgrade["deployment"][PSM]["action"] == "upgrade"
    upgrade_meta = upgrade["deployment"][PSM]["meta"]
    assert upgrade_meta["clusters"] == [{"id": 100}, {"id": 101}]
    assert all(type(c["id"]) is int for c in upgrade_meta["clusters"])
    assert upgrade_meta["image_tag"] == "tce"
    assert upgrade_meta["image_version"] == IMAGE
    assert upgrade_meta["scm_repos"] == meta["scm_repos"]


@pytest.mark.parametrize("online,mirror", [(100, 101), ("100", 101)])
def test_upgrade_accepts_numeric_resource_ids(online, mirror):
    body = deployment_body(
        load_profile(), IDENTITY, VERSION, image_version=IMAGE,
        cluster_ids={"online": online, "query_online_mirror": mirror},
    )
    assert [c["id"] for c in body["deployment"][PSM]["meta"]["clusters"]] == [100, 101]


@pytest.mark.parametrize("invalid", [None, True, 1.5, "1.5", "", "abc", -1, 0, "0", "0101"])
def test_upgrade_rejects_invalid_or_duplicate_numeric_cluster_ids(invalid):
    with pytest.raises(PlatformError):
        deployment_body(
            load_profile(), IDENTITY, VERSION, image_version=IMAGE,
            cluster_ids={"online": invalid, "query_online_mirror": "101"},
        )


@pytest.mark.parametrize(
    "raw,raises",
    [
        ({"pre_check_result": [{"name": "PreCheckQuota", "is_break": True}]}, True),
        ({"pre_check_result": []}, True),
        ({"pre_check_result": [{"name": "PreCheckQuota", "err_msg": "param invalid no need check"}]}, False),
    ],
)
def test_precheck_blocks_break_and_never_claims_skipped_checks_passed(raw, raises):
    if raises:
        with pytest.raises(PlatformError):
            precheck_summary(raw)
    else:
        assert precheck_summary(raw)["partial"]


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["dp/aeolus", "dp/VizQueryService"])
@pytest.mark.parametrize("change", ["none", "image", "dependencies", "pool", "size", "replicas", "health", "routing"])
async def test_observation_checks_exact_recipe_resources_and_health(change, repository):
    profile = load_profile(repository=repository)
    psm = profile["psm"]
    name = environment_name(IDENTITY, profile)
    repos = [
        {"scm_repo_id": d["remote_id"], "version": VERSION if d["main_repo"] else d["version"], "path": d["path"]}
        for d in profile["meta"]["scm_repos"]
    ]
    if change == "dependencies":
        repos = repos[:-1]

    def transport(req):
        path = req.url.path
        if path == f"/api/v4/env/{name}":
            return httpx.Response(200, json={"code": 0, "data": {"id": 123, "desc": f"Conciliator PPE {IDENTITY}"}})
        if path == "/api/v4/instance_meta":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "meta": {"id": 456, "psm": psm, "status": "need_cluster"},
                        "instances": [
                            {"id": 100 + i, "name": c["name"], "repo_info": repos}
                            for i, c in enumerate(profile["meta"]["clusters"])
                        ],
                    },
                },
            )
        if "/services/456/clusters/" in path:
            n = int(path.rstrip("/").split("/")[-1]) - 100
            c = profile["meta"]["clusters"][n]
            return httpx.Response(
                200,
                json={
                    "meta": {"id": 100 + n, "psm": psm, "env": name, "status": "running"},
                    "resource": {
                        "zone": c["zone"],
                        "physical_cluster": "PPE",
                        "logical_cluster": "amd" if change == "pool" else "ipv6",
                        "cpu": c["cpu"] + 1 if change == "size" else c["cpu"],
                        "mem": c["mem"],
                    },
                    "runtime": {
                        "image_tag": "tce",
                        "image_version": "1.0.2.1" if change == "image" else IMAGE,
                        "replicas": {"dc_info": [{"idc": "LF", "count": 1}]},
                        "env_list": [{
                            "env_name": "AEOLUS_SERVICE_DISCOVERY",
                            "env_value": "1" if change == "routing" else "0",
                        }],
                    },
                },
            )
        if path.endswith("/instances/"):
            pods = [{"podName": "pod-one", "podPhase": "Running", "podNotReady": None}]
            return httpx.Response(200, json=[{"data": pods * (2 if change == "replicas" else 1)}])
        if path == "/api/v4/suggest/cluster_by_id":
            return httpx.Response(200, json={
                "name": "default", "psm": psm,
                "base_cluster": {"id": int(req.url.params["cluster_id"])},
                "new_env_vars": {"AEOLUS_SERVICE_DISCOVERY": "1" if change == "routing" else "0"},
            })
        raise AssertionError(path)

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        client = PpeClient(http, profile)
        client.pod_health = AsyncMock(return_value=change != "health")
        result = await client.observe(name, IDENTITY, "123", profile, image_version=IMAGE, scm_version=VERSION)
    assert result["ready"] == (change == "none" or change == "routing" and repository == "dp/aeolus")
    assert len(result["clusters"]) == len(profile["meta"]["clusters"])
    assert "new_env_vars" not in json.dumps(result)


@pytest.mark.asyncio
async def test_delete_checks_name_id_and_ownership_before_any_write():
    requests = []

    def transport(req):
        requests.append(req)
        return httpx.Response(200, json={"code": 0, "data": {"id": 123, "desc": "other task"}})

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        client = PpeClient(http)
        with pytest.raises(PlatformError, match="禁止删除"):
            await client.remove(environment_name(IDENTITY), expected_id="123", identity=IDENTITY)
    assert [r.method for r in requests] == ["GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "http_status,code,missing", [(400, "NOT_EXIST", True), (400, "INVALID_PARAM", False), (403, "NOT_EXIST", False)]
)
async def test_environment_absence_is_explicit_platform_code_not_any_error(http_status, code, missing):
    def transport(req):
        return httpx.Response(http_status, json={"code": code, "message": "test error"})

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        client = PpeClient(http)
        if missing:
            assert await client.environment(environment_name(IDENTITY)) is None
        else:
            with pytest.raises(PlatformError):
                await client.environment(environment_name(IDENTITY))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "finished"])
async def test_tce_pending_resource_approval_is_reported_without_confirming(status):
    calls = []

    def transport(req):
        calls.append(req)
        assert req.method == "GET"
        if "/clusters/" in req.url.path:
            return httpx.Response(
                200,
                json={"meta": {"id": 100, "psm": PSM, "env": environment_name(IDENTITY), "last_deploy_ticket": 222}},
            )
        return httpx.Response(
            200,
            json={
                "status": "running",
                "current_step": 333,
                "steps": [
                    {"id": 333, "step_type": "ConfirmStep", "status": status, "detail": {"token": "must-not-copy"}}
                ],
            },
        )

    async with httpx.AsyncClient(base_url="https://test", transport=httpx.MockTransport(transport)) as http:
        result = await PpeClient(http).approvals(
            environment_name(IDENTITY),
            {"service_id": "456", "clusters": [{"id": "100", "name": "query_online_mirror"}]},
        )
    assert bool(result) == (status == "pending")
    assert "must-not-copy" not in json.dumps(result)
