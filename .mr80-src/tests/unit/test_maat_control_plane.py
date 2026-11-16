import copy
import json

import httpx
import pytest

from conciliator.integrations.maat.client import MaatError, MaatWriteUnknown, environment_plan
from conciliator.integrations.maat.control_plane import (
    MaatControlPlaneClient,
    MaatOpenAPIControlPlaneClient,
    environment_body,
    environment_url,
    routes_match_endpoint,
)
from tests.unit.test_maat_deployment import template

IDENTITY = {"_id": "new-env", "projectId": "project", "projectName": "aeolus", "region": "CN", "name": "coding-test"}
ENVIRONMENT = {**IDENTITY, "createUser": "operator", "isDeleted": False}
PROXY = {
    "_id": "new-proxy", "name": "coding-test", "projectId": "project", "ip": "devbox.internal",
    "port": "45001", "protocol": "http", "type": "dev", "createUser": "operator",
}
BINDINGS = [("aeolus/api/v3", "regexp", "baseline")]


@pytest.mark.asyncio
async def test_public_client_reads_full_record_from_list_and_refuses_console_operations():
    paths = []

    def handle(request):
        paths.append(request.url.path)
        assert request.url.path == "/api/open/devEnv/list"
        return httpx.Response(200, json={"code": 0, "data": {"total": 1, "rows": [ENVIRONMENT]}})

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        control = MaatOpenAPIControlPlaneClient(http, "operator")
        assert await control.environment_record(IDENTITY) == ENVIRONMENT
        with pytest.raises(MaatError, match="identity does not match"):
            await control.environment_record({**IDENTITY, "_id": "different"})
        with pytest.raises(MaatError, match="remain TODO"):
            await control.domain_routes("coding-test.maat.bytedance.net")
    assert paths == ["/api/open/devEnv/list"] * 2


def gateway_route(port=45001):
    return {
        "domain": "coding-test.maat.bytedance.net",
        "incoming": {"path": {"name": "aeolus/api/v3", "type": "regexp", "caseInsensitive": False}},
        "proxy": {"target": f"http://devbox.internal:{port}", "changeOrigin": True},
    }


def test_full_template_configuration_survives_without_copying_identity_or_subscribers():
    source = template()
    source.update(
        projectName="aeolus", resourceType="scm", scmVersion="1.0.16.6958", createUser="baseline-owner",
        watchUserList=["baseline-owner"], larkGroupList=["baseline-group"], templateConfig="opaque template",
        hasIgnoreManifestLogin=True, packages=[], devChildList=[], kuraPackagesPrefetch=True,
    )
    del source["scmList"][0]["path"]
    original = copy.deepcopy(source)
    plan = environment_plan(source, {("aeolus/vqs", "regexp", "vqs"): "new-vqs"})
    body = environment_body(source, plan, name="coding-test", action_user="operator")
    assert source == original
    assert body["name"] == "coding-test" and "_id" not in body and "status" not in body
    assert body["scmList"][0]["scmVersion"] == "1.0.16.6958" and body["scmList"][0]["path"] == "/"
    assert body["templateConfig"] == "opaque template" and body["hasIgnoreManifestLogin"] is True
    assert body["createUser"] == "operator" and body["watchUserList"] == ["operator"] and body["larkGroupList"] == []
    assert [r["apiId"] for r in body["apiProxyList"]] == ["aeolus"] * 4 + ["new-vqs", "auth"]
    source["unexpectedConfig"] = "new capability"
    fresh = environment_plan(source, {("aeolus/vqs", "regexp", "vqs"): "new-vqs"})
    with pytest.raises(MaatError, match="unreviewed"):
        environment_body(source, fresh, name="coding-test", action_user="operator")
    with pytest.raises(MaatError, match="changed"):
        environment_body(source, plan, name="coding-test", action_user="operator")


def test_verified_generic_url_preserves_default_open_path_and_rejects_other_modes():
    assert environment_url(IDENTITY, {"domainType": "generic"}) == "https://coding-test.maat.bytedance.net"
    assert environment_url(IDENTITY, {"domainType": "generic", "defaultOpenPath": "/app"}).endswith(".net/app")
    for config in ({"domainType": "path"}, {"domainType": "generic", "defaultOpenPath": "//other.invalid"}):
        with pytest.raises(MaatError):
            environment_url(IDENTITY, config)
    with pytest.raises(MaatError):
        environment_url({**IDENTITY, "region": "unknown"}, {"domainType": "generic"})


@pytest.mark.parametrize("change", ["port", "missing", "duplicate", "wrong_type", "other_domain", "missing_target"])
def test_effective_routes_must_uniquely_target_the_current_instance(change):
    route = gateway_route()
    routes = [route]
    kwargs = dict(domain=route["domain"], host="devbox.internal", port=45001)
    assert routes_match_endpoint(routes, BINDINGS, **kwargs)
    if change == "port":
        route["proxy"]["target"] = "http://devbox.internal:45002"
    elif change == "missing":
        routes = []
    elif change == "duplicate":
        routes.append(copy.deepcopy(route))
    elif change == "wrong_type":
        route["incoming"]["path"]["type"] = "prefix"
    elif change == "other_domain":
        route["domain"] = "another.maat.bytedance.net"
    else:
        route["proxy"] = {}
    assert not routes_match_endpoint(routes, BINDINGS, **kwargs)


@pytest.mark.asyncio
async def test_openapi_environment_create_keeps_the_request_identity_and_uses_no_console_alias():
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.path == "/api/open/devEnv/create"
        assert request.headers["x-action-user"] == "operator" and request.headers["domain"] == "maat;v1"
        body = json.loads(request.content)
        assert body["name"] == "coding-test" and body["apiProxyList"][0]["apiId"] == "new-proxy"
        return httpx.Response(200, json={"code": 0, "data": {"_id": "new-env"}})

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        client = MaatControlPlaneClient(http, "operator")
        assert await client.create_environment({
            "name": "coding-test", "projectId": "project", "projectName": "aeolus", "region": "CN",
            "apiProxyList": [{"apiId": "new-proxy"}],
        }) == "new-env"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [httpx.Response(500), httpx.Response(200, json={"code": 0, "data": {}})])
async def test_uncertain_environment_write_is_not_repeated(response):
    requests = []

    def handle(request):
        requests.append(request)
        return response

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(MaatWriteUnknown):
            await MaatControlPlaneClient(http, "operator").create_environment({"name": "coding-test"})
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_zero_code_with_success_false_does_not_become_success():
    async with httpx.AsyncClient(
        base_url="https://test.invalid", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"code": 0, "success": False, "data": {"_id": "new-env"}}),
        ),
    ) as http:
        with pytest.raises(MaatError, match="rejected"):
            await MaatControlPlaneClient(http, "operator").create_environment({"name": "coding-test"})


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign", ["_id", "projectId", "name", "createUser"])
@pytest.mark.parametrize("public", [False, True])
async def test_environment_updates_refuse_foreign_identity_before_any_write(foreign, public):
    requests = []

    def handle(request):
        requests.append(request)
        data = {**ENVIRONMENT, foreign: "other"}
        if public:
            assert request.url.path == "/api/open/devEnv/list"
            data = {"total": 1, "rows": [data]}
        else:
            assert request.url.path == "/api/open/devEnv/detail"
        return httpx.Response(200, json={"code": 0, "data": data})

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(MaatError):
            client_type = MaatOpenAPIControlPlaneClient if public else MaatControlPlaneClient
            await client_type(http, "operator").remove_environment(IDENTITY)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_soft_deleted_metadata_is_not_evidence_of_gateway_withdrawal():
    requests = []

    def handle(request):
        requests.append(request.url.path)
        if request.url.path == "/api/open/devEnv/detail":
            data = ENVIRONMENT
        elif request.url.path == "/api/open/devEnv/update":
            assert json.loads(request.content) == {"_id": "new-env", "isDeleted": True}
            data = {**ENVIRONMENT, "isDeleted": True}
        else:
            assert request.url.path == "/api/maatPro/domainRoutesInfo"
            assert request.url.params["__siteEnv__"] == "CN"
            data = [gateway_route()]  # Metadata was removed but the old route is still visible.
        return httpx.Response(200, json={"code": 0, "data": data})

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        client = MaatControlPlaneClient(http, "operator")
        assert (await client.soft_delete_environment(IDENTITY))["isDeleted"]
        routes = await client.domain_routes("coding-test.maat.bytedance.net")
        assert routes_match_endpoint(
            routes, BINDINGS, domain="coding-test.maat.bytedance.net", host="devbox.internal", port=45001,
        )
    assert requests == ["/api/open/devEnv/detail", "/api/open/devEnv/update", "/api/maatPro/domainRoutesInfo"]


@pytest.mark.asyncio
@pytest.mark.parametrize("referenced,foreign", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("public", [False, True])
async def test_proxy_removal_requires_owned_identity_and_no_other_environment_references(referenced, foreign, public):
    writes = []
    prefix = "/api/open/" if public else "/api/"

    def handle(request):
        if public:
            assert request.headers["domain"] == "maat;v1" and request.headers["x-action-user"] == "operator"
        if request.url.path == prefix + "devApi/list":
            body = json.loads(request.content)
            assert body["name" if public else "keyword"] == PROXY["name"]
            data = {"total": 1, "rows": [{**PROXY, "createUser": "other" if foreign else "operator"}]}
        elif request.url.path == prefix + "devApi/getUsedEnvList":
            data = [{"_id": "another-env", "name": "shared"}] if referenced else []
        else:
            assert request.url.path == prefix + "devApi/remove" and json.loads(request.content) == {"_id": "new-proxy"}
            writes.append(request)
            data = None
        return httpx.Response(200, json={"code": 0, "data": data})

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        client_type = MaatOpenAPIControlPlaneClient if public else MaatControlPlaneClient
        client = client_type(http, "operator")
        if referenced or foreign:
            with pytest.raises(MaatError):
                await client.remove_proxy(PROXY)
        else:
            await client.remove_proxy(PROXY)
    assert len(writes) == (0 if referenced or foreign else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    httpx.Response(200, json={"code": 0, "data": "删除成功"}),
    httpx.Response(200, json={"code": 100, "message": "rejected"}),
    httpx.Response(500),
])
async def test_public_environment_removal_checks_business_result_and_never_replays(response):
    paths = []

    def handle(request):
        paths.append(request.url.path)
        assert request.headers["domain"] == "maat;v1" and request.headers["x-action-user"] == "operator"
        if request.url.path == "/api/open/devEnv/list":
            return httpx.Response(200, json={"code": 0, "data": {"total": 1, "rows": [ENVIRONMENT]}})
        assert request.url.path == "/api/open/devEnv/remove"
        assert json.loads(request.content) == {"_id": IDENTITY["_id"]}
        return response

    async with httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle)) as http:
        client = MaatOpenAPIControlPlaneClient(http, "operator")
        if response.status_code == 200 and response.json()["code"] == 0:
            assert await client.remove_environment(IDENTITY) == "删除成功"
        else:
            with pytest.raises(MaatError):
                await client.remove_environment(IDENTITY)
    assert paths == ["/api/open/devEnv/list", "/api/open/devEnv/remove"]
