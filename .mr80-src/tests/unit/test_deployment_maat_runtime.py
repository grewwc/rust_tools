import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from conciliator.config import Settings
from conciliator.integrations.maat.client import MaatError, template_revision
from conciliator.services.deployment_maat_runtime import DeploymentMaatRuntime, build_maat_runtime
from tests.unit.test_maat_deployment import template


@pytest.mark.asyncio
async def test_runner_recovers_after_scan_error_and_closes_while_idle():
    recovered = asyncio.Event()
    calls = 0

    async def scan():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated scan failure")
        recovered.set()
        return []

    http = SimpleNamespace(aclose=AsyncMock())
    runtime = DeploymentMaatRuntime(SimpleNamespace(run_once=scan), http, interval=0.01)
    try:
        await runtime.start()
        worker = runtime._task
        await runtime.start()
        assert runtime._task is worker
        await asyncio.wait_for(recovered.wait(), timeout=1)
    finally:
        await runtime.aclose()
    assert worker.done()
    http.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("missing_path", [None, "aeolus/llm/api/v2", "aeolus/api/v3"])
async def test_runtime_build_uses_local_snapshot_and_rejects_changed_template(
    monkeypatch, tmp_path, changed, missing_path,
):
    import conciliator.services.deployment_maat_runtime as module

    source = {**template(), "projectName": "aeolus", "name": "baseline"}
    source["apiProxyList"] = [r for r in source["apiProxyList"] if r.get("pathname") != missing_path]
    revision = template_revision(source)
    if changed:
        source["scmList"][0]["scmVersion"] = "unreviewed"
    def handle(request):
        pytest.fail("the packaged baseline must not be queried remotely")

    http = httpx.AsyncClient(base_url="https://test.invalid", transport=httpx.MockTransport(handle))
    monkeypatch.setattr(module, "SdkMaatTokenProvider", lambda *a, **kw: SimpleNamespace(
        operator=AsyncMock(return_value="operator"),
    ))
    monkeypatch.setattr(module, "application_http_client", lambda *a, **kw: http)
    snapshot = tmp_path / "baseline.json"
    snapshot.write_text(json.dumps({
        "environment": source, "project_config": {"domainType": "generic"},
    }))
    monkeypatch.setattr(Settings, "maat_baseline_environment", "baseline")
    monkeypatch.setattr(Settings, "maat_baseline_revision", revision)
    monkeypatch.setattr(Settings, "maat_baseline_file", snapshot)
    monkeypatch.setattr(Settings, "maat_action_user", "project-operator")
    settings = Settings(_env_file=None)
    if changed:
        with pytest.raises(MaatError, match="baseline changed"):
            await build_maat_runtime(settings, SimpleNamespace(_sf=None), None)
        assert http.is_closed
    elif missing_path == "aeolus/api/v3":
        with pytest.raises(MaatError, match="binding is ambiguous"):
            await build_maat_runtime(settings, SimpleNamespace(_sf=None), None)
        assert http.is_closed
    else:
        runtime = await build_maat_runtime(settings, SimpleNamespace(_sf=None), None)
        try:
            assert runtime.coordinator.openapi_only
            assert len(runtime.coordinator.bindings["dp/aeolus"]) == (3 if missing_path else 4)
            assert len(runtime.coordinator.bindings["dp/VizQueryService"]) == 1
        finally:
            await runtime.aclose()


@pytest.mark.asyncio
async def test_checked_in_baseline_builds_without_remote_queries(monkeypatch):
    import conciliator.services.deployment_maat_runtime as module

    def forbidden(request):
        pytest.fail("the checked-in baseline must not require remote queries")

    http = httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
    monkeypatch.setattr(module, "SdkMaatTokenProvider", lambda *a, **kw: object())
    monkeypatch.setattr(module, "application_http_client", lambda *a, **kw: http)
    runtime = await build_maat_runtime(Settings(_env_file=None), SimpleNamespace(_sf=None), None)
    try:
        assert len(runtime.coordinator.bindings["dp/aeolus"]) == 3
        assert len(runtime.coordinator.bindings["dp/VizQueryService"]) == 1
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "invalid-json", "wrong-environment"])
async def test_local_snapshot_errors_never_fall_back_to_remote(monkeypatch, tmp_path, failure):
    import conciliator.services.deployment_maat_runtime as module

    def forbidden(request):
        pytest.fail("an invalid local snapshot must not fall back to remote queries")

    http = httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
    monkeypatch.setattr(module, "SdkMaatTokenProvider", lambda *a, **kw: object())
    monkeypatch.setattr(module, "application_http_client", lambda *a, **kw: http)
    snapshot = tmp_path / "baseline.json"
    if failure == "invalid-json":
        snapshot.write_text("{")
    elif failure == "wrong-environment":
        snapshot.write_text(json.dumps({"environment": {"name": "different"}, "project_config": {}}))
    monkeypatch.setattr(Settings, "maat_baseline_file", snapshot)
    settings = Settings(_env_file=None)
    expected = {"missing": FileNotFoundError, "invalid-json": json.JSONDecodeError, "wrong-environment": MaatError}
    with pytest.raises(expected[failure]):
        await build_maat_runtime(settings, SimpleNamespace(_sf=None), None)
    assert http.is_closed


@pytest.mark.asyncio
async def test_runtime_requires_explicit_operator_before_creating_a_client(monkeypatch):
    import conciliator.services.deployment_maat_runtime as module

    def forbidden(*_args, **_kwargs):
        pytest.fail("missing operator must fail before acquiring credentials")

    monkeypatch.setattr(module, "SdkMaatTokenProvider", forbidden)
    monkeypatch.setattr(Settings, "maat_action_user", "")
    with pytest.raises(MaatError, match="MAAT_ACTION_USER"):
        await build_maat_runtime(Settings(_env_file=None), SimpleNamespace(_sf=None), None)


@pytest.mark.asyncio
async def test_runtime_dispatches_maat_transition_notifications_without_stopping_scans():
    finished = asyncio.Event()
    calls = 0
    async def scan():
        nonlocal calls
        calls += 1
        if calls == 1:
            return [None, RuntimeError("one failed operation"), "task", "task"]
        finished.set()
        return []
    notifier = AsyncMock(side_effect=RuntimeError("notification unavailable"))
    runtime = DeploymentMaatRuntime(SimpleNamespace(run_once=scan), SimpleNamespace(aclose=AsyncMock()),
                                    interval=.001, notifier=notifier)
    await runtime.start()
    await asyncio.wait_for(finished.wait(), 1)
    await runtime.aclose()
    notifier.assert_awaited_once_with("task")
