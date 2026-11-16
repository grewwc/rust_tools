import json

import httpx
import pytest

from conciliator.integrations.ppe.scm import PlatformError, ScmClient

SHA = "a" * 40


def version(**overrides):
    return {
        "version": "1.0.0.5",
        "id": 123,
        "repos": 572600,
        "base_commit_hash": SHA,
        "status": "build_ok",
        "type": "test",
        "arch": ["x86_64"],
        **overrides,
    }


@pytest.mark.asyncio
async def test_build_uses_pinned_sha_private_test_artifact_and_exact_version():
    def handle(req):
        body = json.loads(req.content)
        assert req.headers["domain"] == "scm;v1"
        assert req.url.path == "/api/v2/versions/cicd_create"
        assert body["commit_hash"] == SHA and body["pub_base"] == "commit_base"
        assert body["type"] == "test" and body["arch"] == ["x86_64"]
        assert body["repo_id"] == "572600" and body["version"] == "1.0.0.5"
        assert "branch_name" not in body and not body["skip_version_check"]
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "context": {
                        "repo_id": "572600",
                        "version_version": "1.0.0.5",
                        "version_id": 123,
                    }
                },
            },
        )

    async with httpx.AsyncClient(base_url="https://test", transport=httpx.MockTransport(handle)) as http:
        assert await ScmClient(http, "operator").create(commit_sha=SHA, version="1.0.0.5", request_id="b" * 32) == "123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override", [
        {"base_commit_hash": "b" * 40}, {"type": "online"}, {"type": "offline"},
        {"arch": ["aarch64"]}, {"status": "building"},
    ]
)
async def test_build_must_be_successful_for_exact_sha_type_and_arch(override):
    async with httpx.AsyncClient(
        base_url="https://test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=version(**override)),
        ),
    ) as http:
        artifact = await ScmClient(http, "operator").version("1.0.0.5")
        with pytest.raises(PlatformError):
            artifact.require_success(SHA)


@pytest.mark.asyncio
async def test_version_uses_exact_detail_and_not_embedded_latest_repo_version():
    async with httpx.AsyncClient(
        base_url="https://test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={**version(), "repo": {"version": "1.0.0.9999", "status": "building"}}),
        ),
    ) as http:
        artifact = await ScmClient(http, "operator").version("1.0.0.5")
        artifact.require_success(SHA)
        assert artifact.version == "1.0.0.5" and artifact.version_id == "123"


@pytest.mark.asyncio
async def test_unknown_write_is_not_retried_and_platform_errors_are_sanitized():
    calls = []

    def handle(req):
        calls.append(req)
        return httpx.Response(500, text="Authorization: private-credential")

    async with httpx.AsyncClient(base_url="https://test", transport=httpx.MockTransport(handle)) as http:
        with pytest.raises(PlatformError) as error:
            await ScmClient(http, "operator").create(commit_sha=SHA, version="1.0.0.5", request_id="b" * 32)
        assert error.value.uncertain and "private-credential" not in str(error.value)
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_explicit_failed_build_without_version_is_rejection_not_unknown_write():
    seen = []

    def transport(req):
        seen.append(req)
        return httpx.Response(
            200, json={"code": 0, "data": {"state": "failed", "context": {"error": "sensitive-platform-error"}}}
        )

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        with pytest.raises(PlatformError, match="未受理") as caught:
            await ScmClient(http, "huaguojin").create(commit_sha="a" * 40, version="1.0.0.1542", request_id="b" * 32)
    assert not caught.value.uncertain
    assert "sensitive-platform-error" not in str(caught.value)
    assert seen[0].headers["X-PLATFORM-PROXY-USER"] == "huaguojin"


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_sha", [False, True])
async def test_codebase_artifact_lookup_is_read_only_and_pins_sha(wrong_sha):
    calls = []

    def transport(req):
        calls.append(req)
        assert req.method == "GET"
        if req.url.path == "/api/v2/versions/":
            assert req.url.params["commit"] == SHA
            assert req.url.params["repos_id"] == "572600"
            assert req.url.params["type_list"] == "test"
            return httpx.Response(
                200,
                json={
                    "results": [version(base_commit_hash="b" * 40 if wrong_sha else SHA)],
                    "next": None,
                },
            )
        return httpx.Response(200, json=version())

    async with httpx.AsyncClient(base_url="https://test", transport=httpx.MockTransport(transport)) as http:
        if wrong_sha:
            with pytest.raises(PlatformError, match="其他仓库或提交"):
                await ScmClient(http, "huaguojin").for_commit(SHA)
        else:
            artifact = await ScmClient(http, "huaguojin").for_commit(SHA)
            artifact.require_success(SHA)
            assert artifact.build_type == "test" and artifact.version_id == "123"
            assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("has_test", [False, True])
async def test_cn_lookup_ignores_successful_offline_artifact_even_if_newer(has_test):
    calls = []

    def transport(req):
        calls.append(req)
        if req.url.path == "/api/v2/versions/":
            rows = [version(id=999, type="offline")]
            if has_test:
                rows.append(version())
            return httpx.Response(200, json={"results": rows, "next": None})
        return httpx.Response(200, json=version())

    async with httpx.AsyncClient(base_url="https://test", transport=httpx.MockTransport(transport)) as http:
        artifact = await ScmClient(http, "huaguojin").for_commit(SHA)
    if has_test:
        assert artifact.build_type == "test" and artifact.version_id == "123"
        assert len(calls) == 2
    else:
        assert artifact is None
        assert len(calls) == 1
