import copy
import json

import httpx
import pytest

from conciliator.integrations.maat.client import (
    MaatClient,
    MaatError,
    MaatWriteUnknown,
    environment_plan,
    service_environment_plan,
    template_revision,
)


async def token():
    return "test-only-token"


@pytest.mark.asyncio
async def test_proxy_create_serializes_port_checks_business_code_and_never_retries():
    requests = []

    def handle(request):
        import json

        body = json.loads(request.content)
        requests.append(body)
        assert body["port"] == "43017" and body["type"] == "dev"
        return httpx.Response(200, json={"code": 12, "message": "do not expose private platform details"})

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        client = MaatClient(http, token, "operator")
        with pytest.raises(MaatError, match="rejected"):
            await client.create_dev_api(project_id="project", name="one", host="devbox.internal", port=43017)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_missing_proxy_id_is_unknown_and_not_recreated():
    async with httpx.AsyncClient(
        base_url="https://test.invalid",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"code": 0, "data": {}}),
        ),
    ) as http:
        client = MaatClient(http, token, "operator")
        with pytest.raises(MaatWriteUnknown, match="identity"):
            await client.create_dev_api(project_id="project", name="one", host="devbox.internal", port=43017)


@pytest.mark.asyncio
async def test_environment_lookup_uses_numeric_pages_and_finds_exact_match_after_first_page():
    pages = []
    expected = {"_id": "target", "projectName": "aeolus", "region": "CN", "name": "exact"}

    def handle(request):
        body = json.loads(request.content)
        assert type(body["pageNo"]) is int and body["pageSize"] == 100
        pages.append(body["pageNo"])
        rows = [{"name": "other"}] * 100 if body["pageNo"] == 1 else [expected]
        return httpx.Response(200, json={"code": 0, "data": {"total": 101, "rows": rows}})

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        found = await MaatClient(http, token, "operator").get_environment("aeolus", "CN", "exact")
    assert found == expected and pages == [1, 2]


@pytest.mark.asyncio
async def test_preflight_distinguishes_absent_environment_from_platform_failure():
    for status, payload in [(200, {"code": 0, "data": {"total": 0, "rows": []}}), (403, {"code": -1})]:
        async with httpx.AsyncClient(
            base_url="https://test.invalid",
            transport=httpx.MockTransport(lambda _, s=status, p=payload: httpx.Response(s, json=p)),
        ) as http:
            client = MaatClient(http, token, "operator")
            if status == 200:
                assert await client.lookup_environment("aeolus", "CN", "new") is None
            else:
                with pytest.raises(MaatError):
                    await client.lookup_environment("aeolus", "CN", "new")


def template():
    # Reference groups observed in xiaorui-aeolus-hua-integration; IDs anonymized in tests.
    return {
        "_id": "source",
        "projectId": "project",
        "region": "CN",
        "scmList": [{"scmId": "31336", "scmVersion": "1.0.16.6958", "resourceType": "scm", "path": "/"}],
        "apiProxyList": [
            {"_id": str(i), "pathname": path, "pathMatchType": "regexp", "apiId": api, "customDirectiveList": []}
            for i, (path, api) in enumerate(
                [
                    ("api/v3/misc", "aeolus"),
                    ("aeolus/api/v3", "aeolus"),
                    ("aeolus/openApi", "aeolus"),
                    ("aeolus/llm/api/v2", "aeolus"),
                    ("aeolus/vqs", "vqs"),
                    ("/auth", "auth"),
                ]
            )
        ],
    }


def test_replaces_all_aeolus_references_preserves_vqs_and_does_not_mutate_source():
    source = template()
    original = copy.deepcopy(source)
    bindings = {
        (rule["pathname"], rule["pathMatchType"], rule["apiId"]): "new-aeolus"
        for rule in source["apiProxyList"][:4]
    }
    plan = environment_plan(source, bindings)
    assert source == original
    assert [r["apiId"] for r in plan["api_proxy_list"]] == ["new-aeolus"] * 4 + ["vqs", "auth"]
    assert not any("_id" in r for r in plan["api_proxy_list"])
    assert plan["scm_list"] == original["scmList"]
    assert plan["revision"] == environment_plan(source, bindings)["revision"]
    vqs = environment_plan(source, {("aeolus/vqs", "regexp", "vqs"): "new-vqs"})
    assert [r["apiId"] for r in vqs["api_proxy_list"]] == ["aeolus"] * 4 + ["new-vqs", "auth"]


def test_redacted_or_incomplete_baseline_cannot_become_create_payload():
    source = template()
    source["apiProxyList"][0]["customDirectiveList"] = [{"name": "rewrite", "value": "[REDACTED]"}]
    with pytest.raises(MaatError, match="redacted"):
        environment_plan(source, {"aeolus": "new"})
    source = template()
    del source["scmList"][0]["scmVersion"]
    with pytest.raises(MaatError, match="fixed SCM versions"):
        environment_plan(source, {"aeolus": "new"})


def test_service_bindings_do_not_replace_other_services_sharing_the_same_proxy():
    source = template()
    for rule in source["apiProxyList"]:
        rule["apiId"] = "shared"
    bindings = {"dp/aeolus": [(rule["pathname"], "regexp", "shared") for rule in source["apiProxyList"][:4]]}
    plan = service_environment_plan(
        source, "dp/aeolus", "new", bindings=bindings, expected_template_revision=template_revision(source)
    )
    assert [r["apiId"] for r in plan["api_proxy_list"]] == ["new"] * 4 + ["shared", "shared"]
    assert source["apiProxyList"][0]["apiId"] == "shared"
    assert plan["source_revision"] == template_revision(source)


@pytest.mark.parametrize("change", ["target", "duplicate", "missing", "match_type", "leading_slash"])
def test_proxy_binding_changes_require_review(change):
    source = template()
    binding = ("aeolus/api/v3", "regexp", "aeolus")
    if change == "target":
        source["apiProxyList"][1]["apiId"] = "changed"
    elif change == "duplicate":
        source["apiProxyList"].append({**source["apiProxyList"][1], "apiId": "other"})
    elif change == "missing":
        source["apiProxyList"].pop(1)
    elif change == "match_type":
        source["apiProxyList"][1]["pathMatchType"] = "prefix"
    else:
        source["apiProxyList"][1]["pathname"] = "/aeolus/api/v3"
    with pytest.raises(MaatError, match="binding no longer matches"):
        environment_plan(source, {binding: "new"})


def test_template_revision_also_binds_unchanged_routes_and_custom_directives():
    source = template()
    revision = template_revision(source)
    source["apiProxyList"][-1]["customDirectiveList"] = [{"name": "proxy_set_header", "value": "x-ppe sample"}]
    with pytest.raises(MaatError, match="revision changed"):
        service_environment_plan(
            source,
            "dp/aeolus",
            "new",
            bindings={"dp/aeolus": [("aeolus/api/v3", "regexp", "aeolus")]},
            expected_template_revision=revision,
        )


def test_unreviewed_service_and_api_id_only_replacement_are_rejected():
    source = template()
    with pytest.raises(MaatError, match="no reviewed"):
        service_environment_plan(
            source, "dp/unknown", "new", bindings={}, expected_template_revision=template_revision(source)
        )
    with pytest.raises(MaatError, match="pathname, match type"):
        environment_plan(source, {"aeolus": "new"})
