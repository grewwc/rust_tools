import copy
import json
from pathlib import Path

import httpx
import pytest

from conciliator.integrations.ppe.client import load_profile
from conciliator.integrations.ppe.icm import IcmClient, equivalent_recipe, image_body
from conciliator.integrations.ppe.scm import PlatformError


def test_recipe_preserves_installers_and_all_dependencies():
    profile = load_profile()
    body = image_body(profile, "1.0.0.2000", "alice")
    assert body["username"] == "alice"
    assert body["arch_map"] == {"China-North-LF": ["x86_64"]}
    assert body["user_tags"] == ["ppe"]
    assert body["packages"] == profile["image_recipe"]["packages"]
    apt = next(p["name"] for p in body["packages"] if p["package_type"] == "apt-get")
    assert "python-dev-is-python3" in apt.splitlines()
    assert "python-dev" not in apt.splitlines()
    assert len(body["repos"]) == 11
    assert all(isinstance(d["remote_id"], str) for d in body["repos"])
    assert next(d for d in body["repos"] if d["remote_id"] == "572600")["version"] == "1.0.0.2000"
    assert all(
        d["version"]
        == next(x["version"] for x in profile["meta"]["scm_repos"] if str(x["remote_id"]) == d["remote_id"])
        for d in body["repos"]
        if d["remote_id"] != "572600"
    )


@pytest.mark.parametrize("operator", ["", "sa_g_123", "somebody@bytedance.com", "alice\nadmin"])
def test_application_or_unvalidated_user_cannot_be_build_operator(operator):
    with pytest.raises(PlatformError):
        image_body(load_profile(), "1.0.0.1541", operator)


@pytest.mark.parametrize("mutation", ["operator", "dependency", "installer", "architecture", "main", "duplicate"])
def test_recipe_mismatch_is_rejected(mutation):
    expected = image_body(load_profile(), "1.0.0.1541", "alice")
    actual = copy.deepcopy(expected)
    if mutation == "operator":
        actual["username"] = "bob"
    elif mutation == "dependency":
        actual["repos"].pop()
    elif mutation == "duplicate":
        actual["repos"].append(copy.deepcopy(actual["repos"][0]))
    elif mutation == "installer":
        actual["packages"][1]["package_type"] = "pip"
    elif mutation == "architecture":
        actual["arch_map"] = {"China-North-LF": ["aarch64"]}
    else:
        actual["repos"][0]["version"] = "1.0.0.9999"
    assert not equivalent_recipe(expected, actual)


async def test_actual_list_response_keeps_id_before_detail_queries():
    calls = []

    def handler(req):
        calls.append(req)
        assert req.headers["domain"] == "icm;v1"
        body = json.loads(req.content)
        assert body["username"] == "alice"
        return httpx.Response(200, json=[{"id": 178004962, "region": "China-North-LF", "image_name": ""}])

    async with httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(handler)) as http:
        value = await IcmClient(http, None).create(image_body(load_profile(), "1.0.0.1541", "alice"), 217258)
    assert value == {"image_id": "217258", "image_version_id": "178004962"}
    assert len(calls) == 1


async def test_exact_image_id_and_recipe_required_even_if_latest_image_succeeded():
    expected = image_body(load_profile(), "1.0.0.1541", "alice")

    def handler(req):
        if "/versions/" in req.url.path:
            assert req.url.path.endswith("/178004962/")
            return httpx.Response(
                200,
                json={
                    "id": 178004962,
                    "image": {"id": 217258},
                    "version": "1.0.2.554",
                    "region": "China-North-LF",
                    "arch_list": ["x86_64"],
                    "status": "ok",
                },
            )
        assert req.url.params["version"] == "1.0.2.554"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {"image_id": 217258, "build_params": expected, "regions": {"China-North-LF": {"status": "ok"}}},
            },
        )

    async with httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(handler)) as http:
        result = await IcmClient(http, http).version({"image_id": "217258", "image_version_id": "178004962"}, expected)
        assert result["image_status"] == "ok"
        other = copy.deepcopy(expected)
        other["repos"][0]["version"] = "1.0.0.2000"
        with pytest.raises(PlatformError, match="配方"):
            await IcmClient(http, http).version({"image_id": "217258", "image_version_id": "178004962"}, other)


async def test_unknown_write_is_not_repeated_or_error_content_exposed():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(502, text="sensitive credential")

    async with httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(PlatformError) as result:
            await IcmClient(http, http).create(image_body(load_profile(), "1.0.0.1541", "alice"), 217258)
        assert result.value.uncertain
        assert "sensitive" not in str(result.value)
        assert len(calls) == 1


@pytest.mark.parametrize("repository", ["dp/aeolus", "dp/VizQueryService"])
@pytest.mark.parametrize("mutation", [
    None, "digest", "id", "version", "architecture", "psm", "sha", "dependency", "startup", "status",
    "embedded_path", "embedded_duplicate",
])
async def test_production_baseline_requires_exact_identity_and_content(repository, mutation):
    from conciliator.integrations.ppe import icm

    baseline = json.loads(Path(icm.__file__).with_name("production_baselines.json").read_text())[repository]
    raw = {
        "id": int(baseline["image_version_id"]), "image": {"id": int(baseline["image_id"])},
        "version": baseline["image_version"], "region": "China-North-LF", "arch_list": ["x86_64"], "status": "ok",
        "extras": {
            "image_digest": baseline["image_digest"],
            "Metadata": {"appPath": baseline["app_path"], "baseImageVersion": baseline["resolved_base_image"]},
            "Scm": [{
                "remote_id": baseline["scm_repo_id"], "name": baseline["scm_repo_name"],
                "version": baseline["version"], "commit_id": baseline["commit_sha"],
                "path": "/opt/tiger/toutiao/app/data/aeolus",
            }],
            "scm_repos": [{
                "remote_id": d["remote_id"], "repo_name": d["name"], "deploy_path": d["path"], "version": d["version"],
            } for d in baseline["scm_repos"]],
        },
    }
    detail = {"image_id": int(baseline["image_id"]), "build_params": {"psm": baseline["psm"]},
              "regions": {"China-North-LF": {"status": "ok"}}}
    if mutation == "digest":
        raw["extras"]["image_digest"] = "sha256:other"
    elif mutation == "id":
        raw["id"] = 123
    elif mutation == "version":
        raw["version"] = "1.0.0.9999"
    elif mutation == "architecture":
        raw["arch_list"] = ["aarch64"]
    elif mutation == "psm":
        detail["build_params"]["psm"] = "other.service"
    elif mutation == "sha":
        raw["extras"]["Scm"][0]["commit_id"] = "b" * 40
    elif mutation == "dependency":
        raw["extras"]["scm_repos"].append(
            {"remote_id": 1, "repo_name": "other", "deploy_path": "other", "version": "1.0.0.1"}
        )
    elif mutation == "startup":
        raw["extras"]["Metadata"]["appPath"] = "other.sh"
    elif mutation == "status":
        detail["regions"]["China-North-LF"]["status"] = "building"
    elif mutation == "embedded_path":
        raw["extras"]["Scm"][0]["path"] = None
    elif mutation == "embedded_duplicate":
        raw["extras"]["Scm"].append(copy.deepcopy(raw["extras"]["Scm"][0]))

    def handler(req):
        assert req.method == "GET"
        return httpx.Response(200, json=raw if "/versions/" in req.url.path else detail)

    async with httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(handler)) as http:
        client = IcmClient(http, http)
        if mutation and (mutation != "embedded_path" or not baseline["scm_repos"]):
            with pytest.raises(PlatformError):
                await client.baseline_version(baseline)
        else:
            result = await client.baseline_version(baseline)
            assert result["image_version_id"] == baseline["image_version_id"]
            assert result["artifact_source"] == "production_baseline"
            if not baseline["scm_repos"]:
                assert result["embedded_scm_repos"] == [{
                    "name": baseline["scm_repo_name"], "version": baseline["version"],
                    "path": "/opt/tiger/toutiao/app/data/aeolus",
                }]
