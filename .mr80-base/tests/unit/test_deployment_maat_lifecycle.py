import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from conciliator.integrations.maat.client import MaatWriteUnknown, service_environment_plan, template_revision
from conciliator.repositories.models import Deployment
from conciliator.services.aeolus_lark_service import AeolusLarkService
from conciliator.services.deployment_counts import devbox_result_kind
from conciliator.services.deployment_maat import MaatDeploymentCoordinator
from tests.unit.test_deployment_multi import CAPABILITY, REGISTRATION, SHA, deploy, register, review
from tests.unit.test_deployment_multi import system as system
from tests.unit.test_maat_deployment import template


@pytest.fixture
async def lifecycle(system, request):
    sf, registry, service = system
    host = getattr(request, "param", REGISTRATION["advertised_host"])
    session = await register(service, {**REGISTRATION, "advertised_host": host})
    await review(registry, "one")
    record = await deploy(service, "one", session)
    endpoint = {
        "session_id": session, "generation": 1, "request_id": record["deployment_id"], "commit_sha": SHA,
        "runtime_profile_id": "py311", "profile_revision": CAPABILITY["profile_revision"],
        "runtime_revision": CAPABILITY["runtime_revision"], "endpoint_host": host,
        "endpoint_port": 45001, "runtime_attempt_id": "c" * 32, "log_path": "/tmp/deploy.log",
    }
    await service.run("endpoint_ready", "machine", record["deployment_id"], endpoint)
    source = {**template(), "projectName": "aeolus", "resourceType": "scm", "scmVersion": "1.0.16.6958"}
    name = f"hua-ai-{record['deployment_id'][:12]}-g1"
    state = {"routes": True, "proxy": True, "environment": False}

    async def create_environment(_body):
        state["environment"] = True
        return "new-env"

    async def routes(_domain):
        return [{
            "domain": f"{name}.maat.bytedance.net",
            "incoming": {"path": {"name": "aeolus/api/v3", "type": "regexp"}},
            "proxy": {"target": f"http://{host}:45001"},
        }] if state["routes"] else []

    async def remove_environment(_identity):
        state["routes"] = False

    async def remove_proxy(_identity):
        state["proxy"] = False

    control = SimpleNamespace(
        action_user="operator", create_environment=AsyncMock(side_effect=create_environment),
        apply_environment=AsyncMock(), deploy_files=AsyncMock(), deploy_gateway=AsyncMock(),
        environment_record=AsyncMock(return_value={
            "_id": "new-env", "projectId": "project", "projectName": "aeolus", "region": "CN",
            "name": name, "createUser": "operator", "status": "deploy_file_ok", "isDeleted": False,
        }),
        domain_routes=AsyncMock(side_effect=routes), remove_environment=AsyncMock(side_effect=remove_environment),
        remove_proxy=AsyncMock(side_effect=remove_proxy),
        find_proxy=AsyncMock(side_effect=lambda **_: {"_id": "new-proxy"} if state["proxy"] else None),
    )
    openapi = SimpleNamespace(create_dev_api=AsyncMock(return_value={"_id": "new-proxy"}))
    coordinator = MaatDeploymentCoordinator(
        service, openapi=openapi, control_plane=control,
        baseline_provider=AsyncMock(return_value=(source, {"domainType": "generic"})),
        baseline_revision=template_revision(source),
        service_bindings={"dp/aeolus": [("aeolus/api/v3", "regexp", "aeolus")]},
        proxy_options={"dp/aeolus": {"protocol": "http", "proxy_timeout": "30", "is_socket": False}},
        probe=AsyncMock(return_value=True),
    )
    return sf, registry, service, session, record["deployment_id"], coordinator, control, openapi, state


async def advance(coordinator, deployment_id, count):
    for _ in range(count):
        await coordinator.advance(deployment_id)


@pytest.mark.asyncio
async def test_maat_creation_progress_does_not_render_as_deployment_error(lifecycle):
    _sf, registry, service, _session, deployment_id, coordinator, _control, _openapi, _state = lifecycle
    cards = AeolusLarkService(AsyncMock(), app_id="app", meegle=AsyncMock(), verification_token="test")
    for step in range(7):
        snapshot = await service.run("snapshot", "one")
        deployment = snapshot["deployment"]
        # The same snapshot drives the web error notice and the Feishu task card.
        assert not deployment.get("error"), (step, deployment)
        if step == 0:
            assert deployment["status_label"] == "服务已启动，等待 MAAT 入口"
            assert "请稍候" in deployment["status_note"]
        task = await registry.get("one")
        task.deployment = deployment
        card = json.dumps(cards.card(task, "owner", "recipient"), ensure_ascii=False)
        assert "部署错误" not in card
        assert "联系维护者或释放环境" not in card
        if step == 0:
            assert deployment["status_note"] in card
        await coordinator.advance(deployment_id)
    assert deployment["current"]


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["devbox.internal", "10.88.24.19"], indirect=True)
async def test_create_apply_verify_then_cleanup_requires_both_gateway_and_worker_ack(lifecycle):
    sf, registry, service, session, deployment_id, coordinator, control, openapi, _state = lifecycle
    await advance(coordinator, deployment_id, 6)
    snapshot = (await service.run("snapshot", "one"))["deployment"]
    assert snapshot["status"] == "succeeded" and snapshot["current"]
    assert snapshot["maat_url"] == f"https://hua-ai-{deployment_id[:12]}-g1.maat.bytedance.net"
    openapi.create_dev_api.assert_awaited_once()
    with sf() as db:
        assert openapi.create_dev_api.await_args.kwargs["host"] == (
            db.get(Deployment, deployment_id).data["service_host"]
        )
    assert openapi.create_dev_api.await_args.kwargs["port"] == 45001
    assert openapi.create_dev_api.await_args.kwargs["proxy_timeout"] == "30"
    control.create_environment.assert_awaited_once()
    assert control.create_environment.await_args.args[0]["name"] == openapi.create_dev_api.await_args.kwargs["name"]
    control.apply_environment.assert_awaited_once()
    control.deploy_files.assert_awaited_once()
    control.deploy_gateway.assert_awaited_once()

    # Verification accepts v2's per-instance evidence, without a machine-wide reservation.
    await registry.begin_completion("one", "complete", False, verified_id=deployment_id, allow_release=True)
    ack = {"session_id": session, "deployment_id": deployment_id, "generation": 1}
    assert not (await service.run("stopped", "machine", ack))["port_released"]
    await advance(coordinator, deployment_id, 4)
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        assert row.data["maat_cleanup_verified"] and not row.data["released"]
        assert row.data["verified"]
    # The original stop ACK may have arrived before remote cleanup. Replaying it now releases the ledger.
    assert (await service.run("stopped", "machine", ack))["port_released"]
    assert (await service.run("snapshot", "one"))["deployment"]["released"]
    control.remove_environment.assert_awaited_once()
    control.remove_proxy.assert_awaited_once()


@pytest.mark.asyncio
async def test_gateway_match_without_end_to_end_probe_never_becomes_success(lifecycle):
    _sf, _registry, service, _session, deployment_id, coordinator, _control, _openapi, _state = lifecycle
    coordinator.probe = None
    await advance(coordinator, deployment_id, 8)
    snapshot = (await service.run("snapshot", "one"))["deployment"]
    assert snapshot["status"] == "running" and not snapshot["current"]


@pytest.mark.asyncio
async def test_cancel_during_create_preserves_late_resource_id_and_skips_configuration(lifecycle):
    sf, _registry, service, _session, deployment_id, coordinator, control, _openapi, _state = lifecycle
    await coordinator.advance(deployment_id)
    sent, finish = asyncio.Event(), asyncio.Event()

    async def delayed(_body):
        sent.set()
        await finish.wait()
        return "late-env"

    control.create_environment.side_effect = delayed
    pending = asyncio.create_task(coordinator.advance(deployment_id))
    try:
        await asyncio.wait_for(sent.wait(), 1)
        await service.run("release", "one", deployment_id, 1)
    finally:
        finish.set()
        await pending
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        assert row.data["maat_env_id"] == "late-env" and row.data["release_requested"]
    await advance(coordinator, deployment_id, 2)
    control.apply_environment.assert_not_awaited()
    assert control.remove_environment.await_args.args[0]["_id"] == "late-env"


@pytest.mark.asyncio
async def test_unknown_create_stops_service_but_quarantines_port_and_does_not_retry(lifecycle):
    sf, _registry, service, session, deployment_id, coordinator, _control, openapi, _state = lifecycle
    openapi.create_dev_api.side_effect = MaatWriteUnknown("response lost")
    await advance(coordinator, deployment_id, 3)
    openapi.create_dev_api.assert_awaited_once()
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        assert row.data["release_requested"] and row.data["maat_write_intent"]
    assert not (await service.run("stopped", "machine", {
        "session_id": session, "deployment_id": deployment_id, "generation": 1,
    }))["port_released"]
    snapshot = (await service.run("snapshot", "one"))["deployment"]
    assert snapshot["released"] and not snapshot["resource_releasing"] and not snapshot["resource_owned"]
    assert snapshot["status_label"] == "本机已释放，MAAT 待清理"
    assert not snapshot["error"]


@pytest.mark.asyncio
async def test_crash_after_persisting_intent_does_not_send_a_second_create(lifecycle):
    sf, _registry, _service, _session, deployment_id, coordinator, _control, openapi, _state = lifecycle
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        row.data = {**row.data, "maat_write_intent": {
            "id": "old-operation", "phase": "proxy", "at": (datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
        }}
        db.commit()
    await coordinator.advance(deployment_id)
    openapi.create_dev_api.assert_not_awaited()
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        assert row.data["release_requested"] and row.stage == "maat_write_unknown"


@pytest.mark.asyncio
async def test_removed_environment_with_stale_gateway_routes_does_not_remove_proxy_or_release(lifecycle):
    sf, _registry, service, _session, deployment_id, coordinator, control, _openapi, _state = lifecycle
    await advance(coordinator, deployment_id, 6)
    await service.run("release", "one", deployment_id, 1)
    control.remove_environment.side_effect = None  # Metadata removal does not withdraw the route in this scenario.
    await advance(coordinator, deployment_id, 5)
    control.remove_proxy.assert_not_awaited()
    with sf() as db:
        assert not db.get(Deployment, deployment_id).data.get("maat_cleanup_verified")


@pytest.mark.asyncio
async def test_endpoint_must_match_frozen_deployment_host_before_maat_creation(lifecycle):
    sf, _registry, _service, _session, deployment_id, coordinator, _control, openapi, _state = lifecycle
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        row.data = {**row.data, "endpoint": {**row.data["endpoint"], "endpoint_host": "different.internal"}}
        db.commit()
    with pytest.raises(RuntimeError, match="deployment host"):
        await coordinator.advance(deployment_id)
    openapi.create_dev_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_unready_environment_times_out_into_cleanup_instead_of_running_forever(lifecycle):
    sf, _registry, service, _session, deployment_id, coordinator, _control, _openapi, _state = lifecycle
    coordinator.probe = None
    await advance(coordinator, deployment_id, 5)
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        row.data = {**row.data, "maat_started_at": (datetime.now(UTC) - timedelta(minutes=11)).isoformat()}
        db.commit()
    await coordinator.advance(deployment_id)
    snapshot = (await service.run("snapshot", "one"))["deployment"]
    assert snapshot["resource_releasing"] and not snapshot["current"]


@pytest.mark.asyncio
async def test_public_api_creation_and_record_cleanup_preserve_port_without_console_calls(lifecycle):
    sf, _registry, service, session, deployment_id, coordinator, control, openapi, _state = lifecycle
    await public_lifecycle(lifecycle, status="deploy_file_ok")
    await advance(coordinator, deployment_id, 4)
    snapshot = (await service.run("snapshot", "one"))["deployment"]
    assert snapshot["stage"] == "maat_entry_pending" and not snapshot["current"]
    assert snapshot["maat_url"] == f"https://hua-ai-{deployment_id[:12]}-g1.maat.bytedance.net"
    assert snapshot["status"] != "succeeded" and "待业务验证" in snapshot["status_label"]
    # Pending human verification is not a failed create. Keep healthy instances alive after ten minutes.
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        row.data = {**row.data, "maat_started_at": (datetime.now(UTC) - timedelta(minutes=30)).isoformat()}
        db.commit()
    await coordinator.advance(deployment_id)
    assert not (await service.run("snapshot", "one"))["deployment"]["resource_releasing"]
    openapi.create_dev_api.assert_awaited_once()
    control.create_environment.assert_awaited_once()
    control.apply_environment.assert_not_awaited()
    control.deploy_files.assert_not_awaited()
    control.deploy_gateway.assert_not_awaited()
    await service.run("release", "one", deployment_id, 1)
    await coordinator.advance(deployment_id)
    control.remove_environment.assert_not_awaited()
    control.remove_proxy.assert_not_awaited()
    ack = {
        "session_id": session, "deployment_id": deployment_id, "generation": 1,
    }
    assert not (await service.run("stopped", "machine", ack))["port_released"]
    pending = (await service.run("snapshot", "one"))["deployment"]
    assert pending["cleanup_state"] == "pending"
    assert pending["status_label"] == "本机已释放，MAAT 清理中"
    assert "待人工" not in pending["status_note"]
    assert devbox_result_kind(pending) is None
    notifications = []
    for _ in range(6):
        notifications.extend(r for r in await coordinator.run_once() if isinstance(r, str))
    assert notifications == ["one"]
    control.remove_environment.assert_awaited_once()
    control.remove_proxy.assert_awaited_once()
    control.domain_routes.assert_not_awaited()
    assert _state["routes"]  # Soft deletion did not withdraw the gateway route.
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        assert row.data["cleanup_state"] == "succeeded"
        assert row.data["maat_phase"] == "cleanup_records_deleted"
        assert row.data["maat_api_id"] == "new-proxy" and row.data["maat_env_id"] == "new-env"
        assert not row.data.get("maat_cleanup_verified") and not row.data.get("port_released")
    assert not (await service.run("stopped", "machine", ack))["port_released"]
    snapshot = await service.run("snapshot", "one")
    assert snapshot["deployment"]["status_label"] == "环境已释放"
    assert devbox_result_kind(snapshot["deployment"]) == "released"
    assert "代理记录已清理" in snapshot["deployment"]["status_note"]
    assert snapshot["deployment_workers"][0]["deployment_counts"]["cleanup_pending"] == 0
    # Further scans and duplicate stop ACKs must not repeat deletes or free the old port.
    for _ in range(3):
        assert not [r for r in await coordinator.run_once() if isinstance(r, str)]
    control.remove_environment.assert_awaited_once()
    control.remove_proxy.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_public_environment_stops_before_any_resource_creation(lifecycle):
    sf, _registry, _service, _session, deployment_id, coordinator, control, openapi, _state = lifecycle
    coordinator.openapi_only = True
    control.lookup_environment = AsyncMock(return_value={"_id": "existing-environment"})
    await coordinator.advance(deployment_id)
    openapi.create_dev_api.assert_not_awaited()
    control.create_environment.assert_not_awaited()
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        assert row.stage == "maat_name_conflict" and row.data["release_requested"]
        assert not row.data.get("maat_env_id") and not row.data.get("maat_write_intent")


async def public_lifecycle(lifecycle, *, status="deploying_file"):
    sf, _registry, _service, _session, deployment_id, coordinator, control, _openapi, state = lifecycle
    coordinator.openapi_only = True
    control.lookup_environment = AsyncMock(side_effect=lambda *_: (
        control.environment_record.return_value if state["environment"] else None
    ))
    async def remove_environment(_identity):
        state["environment"] = False
    control.remove_environment.side_effect = remove_environment
    source, _config = await coordinator.baseline_provider()
    plan = service_environment_plan(
        source, "dp/aeolus", "new-proxy", bindings=coordinator.bindings,
        expected_template_revision=coordinator.baseline_revision,
    )
    control.environment_record.return_value.update(
        apiProxyList=plan["api_proxy_list"], scmList=plan["scm_list"], status=status,
    )
    return sf, deployment_id, coordinator, control


@pytest.mark.asyncio
async def test_public_cleanup_verifies_absence_before_deleting_proxy(lifecycle):
    sf, deployment_id, coordinator, control = await public_lifecycle(lifecycle)
    _sf, _registry, service, session, *_ = lifecycle
    await advance(coordinator, deployment_id, 3)
    await service.run("release", "one", deployment_id, 1)
    await service.run("stopped", "machine", {
        "session_id": session, "deployment_id": deployment_id, "generation": 1,
    })
    control.remove_environment.side_effect = None  # The record remains visible after the write.
    await advance(coordinator, deployment_id, 6)
    control.remove_environment.assert_awaited_once()
    control.remove_proxy.assert_not_awaited()
    with sf() as db:
        assert db.get(Deployment, deployment_id).data["maat_phase"] == "cleanup_environment_verify"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["environment", "proxy"])
async def test_public_unknown_delete_is_quarantined_without_retry(lifecycle, phase):
    sf, deployment_id, coordinator, control = await public_lifecycle(lifecycle)
    _sf, _registry, service, session, *_ = lifecycle
    await advance(coordinator, deployment_id, 3)
    await service.run("release", "one", deployment_id, 1)
    await service.run("stopped", "machine", {
        "session_id": session, "deployment_id": deployment_id, "generation": 1,
    })
    operation = control.remove_environment if phase == "environment" else control.remove_proxy
    operation.side_effect = MaatWriteUnknown("response lost")
    await advance(coordinator, deployment_id, 8)
    operation.assert_awaited_once()
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        assert row.data["maat_write_intent"]["phase"] == "cleanup_" + phase
        assert row.data["cleanup_state"] == "manual_required"
        assert row.stage == "maat_cleanup_unknown"
        assert not row.data.get("port_released")
    snapshot = (await service.run("snapshot", "one"))["deployment"]
    assert devbox_result_kind(snapshot) == "release_unknown"


@pytest.mark.asyncio
async def test_public_proxy_only_cleanup_skips_environment_delete(lifecycle):
    sf, deployment_id, coordinator, control = await public_lifecycle(lifecycle)
    _sf, _registry, service, session, *_ = lifecycle
    await coordinator.advance(deployment_id)  # Only the proxy exists when release is requested.
    await service.run("release", "one", deployment_id, 1)
    await service.run("stopped", "machine", {
        "session_id": session, "deployment_id": deployment_id, "generation": 1,
    })
    await advance(coordinator, deployment_id, 5)
    control.remove_environment.assert_not_awaited()
    control.remove_proxy.assert_awaited_once()
    with sf() as db:
        assert db.get(Deployment, deployment_id).data["cleanup_state"] == "succeeded"


@pytest.mark.asyncio
async def test_public_cleanup_accepts_already_deleted_records_without_replaying_deletes(lifecycle):
    sf, deployment_id, coordinator, control = await public_lifecycle(lifecycle)
    _sf, _registry, service, session, *_ = lifecycle
    await advance(coordinator, deployment_id, 3)
    await service.run("release", "one", deployment_id, 1)
    await service.run("stopped", "machine", {
        "session_id": session, "deployment_id": deployment_id, "generation": 1,
    })
    lifecycle[-1].update(environment=False, proxy=False)
    await advance(coordinator, deployment_id, 5)
    control.remove_environment.assert_not_awaited()
    control.remove_proxy.assert_not_awaited()
    with sf() as db:
        assert db.get(Deployment, deployment_id).data["cleanup_state"] == "succeeded"


@pytest.mark.asyncio
async def test_public_cleanup_waits_for_proxy_absence_without_repeating_delete(lifecycle):
    sf, deployment_id, coordinator, control = await public_lifecycle(lifecycle)
    _sf, _registry, service, session, *_ = lifecycle
    await advance(coordinator, deployment_id, 3)
    await service.run("release", "one", deployment_id, 1)
    await service.run("stopped", "machine", {
        "session_id": session, "deployment_id": deployment_id, "generation": 1,
    })
    control.remove_proxy.side_effect = None  # Successful response, but the record is still visible.
    await advance(coordinator, deployment_id, 8)
    control.remove_proxy.assert_awaited_once()
    with sf() as db:
        assert db.get(Deployment, deployment_id).data["maat_phase"] == "cleanup_proxy_verify"
        assert db.get(Deployment, deployment_id).data["cleanup_state"] != "succeeded"


@pytest.mark.asyncio
async def test_public_create_notifies_only_when_deployment_completes_without_redeploy(lifecycle):
    _sf, deployment_id, coordinator, control = await public_lifecycle(lifecycle)
    # Keep one write in flight while other scans observe the durable intent.
    entered, finish = asyncio.Event(), asyncio.Event()
    async def create_proxy(**_kwargs):
        entered.set()
        await finish.wait()
        return {"_id": "new-proxy"}
    coordinator.openapi.create_dev_api.side_effect = create_proxy
    first = asyncio.create_task(coordinator.advance(deployment_id))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        await asyncio.gather(*(coordinator.advance(deployment_id) for _ in range(3)))
    finally:
        finish.set()
        await first
    await coordinator.run_once()
    notices = [r for r in await coordinator.run_once() if isinstance(r, str)]
    assert notices == []
    for _ in range(3):
        assert not [r for r in await coordinator.run_once() if isinstance(r, str)]
    control.environment_record.return_value["status"] = "deploy_file_ok"
    assert [r for r in await coordinator.run_once() if isinstance(r, str)] == ["one"]
    assert not [r for r in await coordinator.run_once() if isinstance(r, str)]
    control.create_environment.assert_awaited_once()
    coordinator.openapi.create_dev_api.assert_awaited_once()
    control.apply_environment.assert_not_awaited()
    control.deploy_files.assert_not_awaited()
    control.deploy_gateway.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_public_apply_phase_reads_without_submitting_another_update(lifecycle):
    sf, deployment_id, coordinator, control = await public_lifecycle(lifecycle)
    await advance(coordinator, deployment_id, 2)
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        row.data = {**row.data, "maat_phase": "apply"}
        db.commit()
    await advance(coordinator, deployment_id, 2)
    control.apply_environment.assert_not_awaited()
    control.create_environment.assert_awaited_once()
    with sf() as db:
        assert db.get(Deployment, deployment_id).stage == "maat_entry_pending"


@pytest.mark.asyncio
async def test_public_deploy_failure_notifies_and_keeps_owned_service_until_manual_release(lifecycle):
    sf, deployment_id, coordinator, control = await public_lifecycle(lifecycle, status="deploy_file_failed")
    await advance(coordinator, deployment_id, 2)
    assert [r for r in await coordinator.run_once() if isinstance(r, str)] == ["one"]
    for _ in range(3):
        assert not [r for r in await coordinator.run_once() if isinstance(r, str)]
    with sf() as db:
        row = db.get(Deployment, deployment_id)
        assert row.status == "failed" and row.stage == "maat_deployment_failed"
        assert not row.data.get("release_requested")
    control.apply_environment.assert_not_awaited()
    control.remove_environment.assert_not_awaited()
