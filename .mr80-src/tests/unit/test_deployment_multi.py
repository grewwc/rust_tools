import asyncio
import json
import subprocess
import sys
from datetime import timedelta

import httpx
import pytest
from conciliator_deployment_worker.multi_worker import MachineWorker
from conciliator_deployment_worker.profiles import capabilities, runtime_profiles
from fastapi import FastAPI
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conciliator.api.deployments import router
from conciliator.domain.aeolus_task import AeolusRound, AeolusStartEvent, AeolusState
from conciliator.repositories.models import Base, Deployment, DeploymentWorker
from conciliator.services.aeolus_registry import AeolusTaskRegistry, InvalidTaskTransition
from conciliator.services.deployment_service import DeploymentService
from conciliator.services.runtime_selection import detect

SHA = "a" * 40
CAPABILITY = {
    "profile_revision": "b" * 64,
    "runtime_revision": "py311-r1",
    "python_version": "3.11",
    "base_branches": ["master"],
}
REGISTRATION = {
    "protocol_version": 2,
    "worker_id": "machine",
    "advertised_host": "devbox.internal",
    "runtime_profiles": {"dp/aeolus": {"py311": CAPABILITY}},
}


@pytest.fixture
def system():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(sf, deployments_enabled=True)
    service = DeploymentService(sf)
    yield sf, registry, service
    engine.dispose()


async def review(registry, task_id, sha=SHA):
    # Each helper call models an independent task, so give it its own AI node.
    await registry.create_from_event(AeolusStartEvent(
        task_id, "space", 42, "node", f"uuid-{task_id}", "user", ()
    ))
    await registry.append_round(task_id, AeolusRound(1, "review", commit_sha=sha))
    await registry.update(
        task_id,
        work_branch="work/generated",
        evaluation={
            "code_context": {"repository": "dp/aeolus", "base_branch": "master"},
            "_runtime_detection": detect("dp/aeolus", sha, {".python-version": "3.11"}),
        },
    )
    await registry.transition(task_id, {AeolusState.RECEIVED}, AeolusState.REVIEW)


async def register(service, body=REGISTRATION):
    reply = await service.run("register", body)
    await service.run("heartbeat", body["worker_id"], {"session_id": reply["session_id"], "reconciled": True})
    return reply["session_id"]


async def deploy(service, task_id, session, request_id=None):
    record = await service.run("enqueue", task_id, "machine", request_id or task_id, 1, SHA)
    return await service.run(
        "claim", "machine", {"session_id": session, "request_id": "claim-" + record["deployment_id"]}
    )


@pytest.mark.asyncio
async def test_registration_is_complete_before_clock_query_autoflush(system):
    sf, _registry, service = system

    def flush_before_execution(state):
        # SQLAlchemy 2.1 flushes before Core clock queries as well as ORM queries.
        if state.session.autoflush:
            state.session.flush()

    event.listen(sf, "do_orm_execute", flush_before_execution)
    try:
        first = await register(service)
        second = await register(service)
        assert first != second
        with sf() as db:
            worker = db.get(DeploymentWorker, "machine")
            assert worker.last_seen_at is not None and worker.session_id == second
            assert worker.data["runtime_profiles"] == REGISTRATION["runtime_profiles"]
    finally:
        event.remove(sf, "do_orm_execute", flush_before_execution)


@pytest.mark.asyncio
async def test_worker_host_can_change_only_after_deployments_are_released(system):
    sf, registry, service = system
    session = await register(service)
    await review(registry, "one")
    record = await deploy(service, "one", session)
    replacement = {**REGISTRATION, "advertised_host": "10.88.24.19"}
    with pytest.raises(InvalidTaskTransition, match="cannot change host"):
        await service.run("register", replacement)
    await service.run("release", "one", record["deployment_id"], 1)
    await service.run("stopped", "machine", {
        "session_id": session, "deployment_id": record["deployment_id"], "generation": 1,
    })
    await register(service, replacement)
    with sf() as db:
        assert db.get(DeploymentWorker, "machine").data["advertised_host"] == replacement["advertised_host"]
        assert db.get(Deployment, record["deployment_id"]).data["service_host"] == REGISTRATION["advertised_host"]


@pytest.mark.asyncio
async def test_same_repository_concurrent_deployments_have_independent_identity_and_cleanup(system):
    sf, registry, service = system
    session = await register(service)
    for task_id in ("one", "two"):
        await review(registry, task_id)
    one = await deploy(service, "one", session)
    two = await deploy(service, "two", session)
    assert one["deployment_id"] != two["deployment_id"]
    assert one["generation"] == two["generation"] == 1
    assert one["status"] == two["status"] == "running"
    snapshot = await service.run("snapshot", "one")
    worker = snapshot["deployment_workers"][0]
    assert "reserved_task_id" not in worker
    assert worker["deployment_counts"]["total"] == 2
    assert worker["recommended_runtime_profile_id"] == "py311"
    await service.run("release", "one", one["deployment_id"], one["generation"])
    heartbeat = await service.run("heartbeat", "machine", {"session_id": session})
    assert [c["deployment_id"] for c in heartbeat["commands"]] == [one["deployment_id"]]
    ack = {"session_id": session, "deployment_id": one["deployment_id"], "generation": 1}
    assert (await service.run("stopped", "machine", ack))["port_released"]
    assert (await service.run("stopped", "machine", ack))["accepted"]
    with sf() as db:
        assert db.get(Deployment, two["deployment_id"]).status == "running"
    assert (await service.run("snapshot", "two"))["deployment_workers"][0]["deployment_counts"]["total"] == 1


@pytest.mark.asyncio
async def test_profile_selection_is_exact_and_idempotency_binds_selection(system):
    _, registry, service = system
    await review(registry, "one")
    body = {
        **REGISTRATION,
        "runtime_profiles": {
            "dp/aeolus": {
                "py311": CAPABILITY,
                "py310": {**CAPABILITY, "base_branches": ["release/legacy"], "python_version": "3.10"},
            }
        },
    }
    session = await register(service, body)
    record = await deploy(service, "one", session)
    assert record["runtime_profile_id"] == "py311"
    with pytest.raises(InvalidTaskTransition, match="different deployment"):
        await service.run("enqueue", "one", "machine", "one", 1, SHA, "py310")
    await review(registry, "two")
    await registry.update("two", evaluation={"code_context": {"repository": "dp/aeolus", "base_branch": "other"}})
    with pytest.raises(ValueError, match="请选择运行配置"):
        await deploy(service, "two", session)
    selected = await service.run("enqueue", "two", "machine", "selected", 1, SHA, "py310")
    assert selected["runtime_profile_id"] == "py310"


@pytest.mark.parametrize("branch", [
    "release-aeolus-2.9.0", "release-aeolus-2.103.0", "release-aeolus-2.104.0", "release-aeolus-2.104.12",
])
def test_legacy_branch_does_not_default_runtime_without_code_evidence(branch):
    from conciliator.services.deployment_multi import select_runtime

    profiles = {"py27": {"base_release_max": 104, "python_version": "2.7"},
                "py310": {"base_branches": ["master"], "python_version": "3.10"}}
    with pytest.raises(ValueError, match="请选择运行配置"):
        select_runtime(profiles, branch)
    assert select_runtime(profiles, branch, "py27") == "py27"


@pytest.mark.parametrize("branch", [
    "release-aeolus-2.105.0", "release-aeolus-2.1040.0", "release-aeolus-3.104.0",
    "master", "aeolus/ai-node/task", "release-aeolus-2.104.0-extra",
])
def test_legacy_runtime_can_be_manually_selected_independently_of_branch(branch):
    from conciliator.services.deployment_multi import select_runtime

    profiles = {"py27": {"base_release_max": 104, "python_version": "2.7"}}
    with pytest.raises(ValueError, match="请选择运行配置"):
        select_runtime(profiles, branch)
    assert select_runtime(profiles, branch, "py27") == "py27"


def test_worker_registration_preserves_release_limit():
    from conciliator.api.deployments import Registration

    body = {
        **REGISTRATION,
        "runtime_profiles": {"dp/aeolus": {
            "py27": {**CAPABILITY, "python_version": "2.7", "base_release_max": 104},
            "py311": CAPABILITY,
        }},
    }
    advertised = Registration.model_validate(body).model_dump(exclude_none=True)["runtime_profiles"]["dp/aeolus"]
    assert advertised["py27"]["base_release_max"] == 104
    assert "base_release_max" not in advertised["py311"]


@pytest.mark.asyncio
async def test_legacy_runtime_is_recommended_and_enqueued_from_code_dependencies(system):
    _, registry, service = system
    await review(registry, "one")
    await registry.update("one", evaluation={
        "code_context": {"repository": "dp/aeolus", "base_branch": "master"},
        "_runtime_detection": detect("dp/aeolus", SHA, {"requirements.txt": "antlr4-python2-runtime==4.7.2"}),
    })
    body = {
        **REGISTRATION,
        "runtime_profiles": {"dp/aeolus": {
            "py311": CAPABILITY,
            "py27": {**CAPABILITY, "python_version": "2.7", "base_branches": [], "base_release_max": 104},
        }},
    }
    session = await register(service, body)
    snapshot = await service.run("snapshot", "one")
    assert snapshot["deployment_workers"][0]["recommended_runtime_profile_id"] == "py27"
    record = await deploy(service, "one", session)
    assert record["runtime_profile_id"] == "py27"
    assert record["python_version"] == "2.7"


@pytest.mark.asyncio
async def test_local_readiness_is_not_global_success_and_ports_are_unique(system):
    _, registry, service = system
    session = await register(service)
    records = []
    for task_id in ("one", "two"):
        await review(registry, task_id)
        records.append(await deploy(service, task_id, session))

    def endpoint(record):
        return {
            "session_id": session,
            "generation": 1,
            "request_id": record["deployment_id"],
            "commit_sha": SHA,
            "runtime_profile_id": "py311",
            "profile_revision": CAPABILITY["profile_revision"],
            "runtime_revision": CAPABILITY["runtime_revision"],
            "endpoint_host": "devbox.internal",
            "endpoint_port": 45001,
            "runtime_attempt_id": "c" * 32,
            "log_path": "/tmp/deploy.log",
        }

    one, two = records
    with pytest.raises(InvalidTaskTransition, match="runtime does not match"):
        await service.run("endpoint_ready", "machine", one["deployment_id"], {
            **endpoint(one), "endpoint_host": "different.internal",
        })
    assert (await service.run("endpoint_ready", "machine", one["deployment_id"], endpoint(one)))["accepted"]
    assert (await service.run("endpoint_ready", "machine", one["deployment_id"], endpoint(one)))["accepted"]
    snapshot = await service.run("snapshot", "one")
    assert snapshot["deployment"]["status"] == "running" and not snapshot["deployment"]["current"]
    assert snapshot["deployment"]["stage"] == "maat_contract_required"
    assert snapshot["deployment"]["log_path"] == "/tmp/deploy.log"
    with pytest.raises(InvalidTaskTransition, match="already allocated"):
        await service.run("endpoint_ready", "machine", two["deployment_id"], endpoint(two))
    with pytest.raises(InvalidTaskTransition, match="v2 success"):
        await service.run(
            "result", "machine", one["deployment_id"], {"session_id": session, "generation": 1, "status": "succeeded"}
        )


@pytest.mark.asyncio
async def test_feedback_cleans_all_task_instances_and_reconnect_fences_sessions(system):
    sf, registry, service = system
    session = await register(service)
    await review(registry, "one")
    record = await deploy(service, "one", session)
    from conciliator.services.deployment_service import invalidate_deployments

    with sf() as db:
        invalidate_deployments(db, "one")
        db.commit()
    assert (await service.run("heartbeat", "machine", {"session_id": session}))["commands"]
    replacement = await service.run("register", REGISTRATION)
    assert replacement["deployments"][0]["deployment_id"] == record["deployment_id"]
    with pytest.raises(InvalidTaskTransition, match="stale"):
        await service.run("heartbeat", "machine", {"session_id": session})
    with pytest.raises(InvalidTaskTransition, match="journals"):
        await service.run("heartbeat", "machine", {"session_id": replacement["session_id"], "reconciled": True})
    await service.run(
        "heartbeat",
        "machine",
        {
            "session_id": replacement["session_id"],
            "reconciled": True,
            "instances": [{"deployment_id": record["deployment_id"], "generation": 1, "service_status": "unknown"}],
        },
    )


@pytest.mark.asyncio
async def test_offline_counts_are_not_zero_and_stopped_route_pending_is_separate(system):
    sf, registry, service = system
    session = await register(service)
    await review(registry, "one")
    record = await deploy(service, "one", session)
    with sf() as db:
        worker = db.get(DeploymentWorker, "machine")
        worker.last_seen_at -= timedelta(seconds=60)
        deployment = db.get(Deployment, record["deployment_id"])
        deployment.data = {**deployment.data, "maat_write_intent": True}
        db.commit()
    snapshot = await service.run("snapshot", "one")
    assert not snapshot["deployment_workers"][0]["online"]
    assert snapshot["deployment_workers"][0]["deployment_counts"]["unknown"] == 1
    await service.run("release", "one", record["deployment_id"], 1)
    ack = await service.run(
        "stopped", "machine", {"session_id": session, "deployment_id": record["deployment_id"], "generation": 1}
    )
    assert not ack["port_released"]
    counts = (await service.run("snapshot", "one"))["deployment_workers"][0]["deployment_counts"]
    assert counts["total"] == 0 and counts["cleanup_pending"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["redeploy", "coding", "complete"])
async def test_pending_maat_cleanup_does_not_block_after_local_stop(system, action):
    sf, registry, service = system
    session = await register(service)
    await review(registry, "one")
    record = await deploy(service, "one", session)
    with sf() as db:
        old = db.get(Deployment, record["deployment_id"])
        old.data = {**old.data, "maat_env_id": "old-environment", "service_port": 45001}
        db.commit()
    await service.run("release", "one", record["deployment_id"], 1)
    with pytest.raises(InvalidTaskTransition, match="释放"):
        await deploy(service, "one", session, request_id="before-stop")
    assert not (await service.run("stopped", "machine", {
        "session_id": session, "deployment_id": record["deployment_id"], "generation": 1,
    }))["port_released"]
    snapshot = (await service.run("snapshot", "one"))["deployment"]
    assert snapshot["released"] and not snapshot["resource_releasing"] and not snapshot["can_release"]
    if action == "redeploy":
        new = await deploy(service, "one", session, request_id="after-stop")
        assert new["deployment_id"] != record["deployment_id"]
        endpoint = {
            "session_id": session, "generation": 1, "request_id": new["deployment_id"], "commit_sha": SHA,
            "runtime_profile_id": "py311", "profile_revision": CAPABILITY["profile_revision"],
            "runtime_revision": CAPABILITY["runtime_revision"], "endpoint_host": "devbox.internal",
            "endpoint_port": 45001, "runtime_attempt_id": "c" * 32, "log_path": "/tmp/deploy.log",
        }
        with pytest.raises(InvalidTaskTransition):
            await service.run("endpoint_ready", "machine", new["deployment_id"], endpoint)
        endpoint["endpoint_port"] = 45002
        assert (await service.run("endpoint_ready", "machine", new["deployment_id"], endpoint))["accepted"]
    elif action == "coding":
        task, created = await registry.start_feedback("one", "继续修改", "feedback")
        assert created and task.state == AeolusState.RECODING
    else:
        await registry.begin_completion("one", "complete", False, allow_release=False)
    # An old stop/health report cannot reoccupy the task or free the retained port.
    await service.run("result", "machine", record["deployment_id"], {
        "session_id": session, "generation": 1, "commit_sha": SHA, "status": "failed",
        "error": "service lost health or listener ownership after local readiness",
    })
    with sf() as db:
        old = db.get(Deployment, record["deployment_id"])
        assert old.data["cleanup_state"] == "manual_required" and not old.data["port_released"]
        assert old.data["maat_env_id"] == "old-environment" and not old.data.get("result")


@pytest.mark.asyncio
async def test_real_concurrent_services_release_and_worker_restore(system, tmp_path):
    _, registry, service = system
    source = tmp_path / "source"
    source.mkdir()
    for args in (
        ["git", "init"],
        ["git", "config", "user.name", "test"],
        ["git", "config", "user.email", "test@example.invalid"],
    ):
        await asyncio.to_thread(subprocess.run, args, cwd=source, check=True, capture_output=True)
    (source / "server.py").write_text(
        "import os\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        ' def do_GET(self):\n  self.send_response(200); self.end_headers(); self.wfile.write(b"ok")\n'
        'HTTPServer(("127.0.0.1", int(os.environ["PORT0"])), Handler).serve_forever()\n'
    )
    await asyncio.to_thread(subprocess.run, ["git", "add", "."], cwd=source, check=True)
    await asyncio.to_thread(
        subprocess.run, ["git", "commit", "-m", "fixture"], cwd=source, check=True, capture_output=True
    )
    sha = (
        await asyncio.to_thread(subprocess.check_output, ["git", "rev-parse", "HEAD"], cwd=source, text=True)
    ).strip()
    manifest = tmp_path / "runtime.json"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    manifest.write_text(json.dumps({"runtime_revision": "test-r1", "python_version": python_version, "files": {}}))
    profile = {
        "python_path": sys.executable,
        "python_version": python_version,
        "runtime_manifest": str(manifest),
        "runtime_revision": "test-r1",
        "source_path": str(source),
        "start_argv": [sys.executable, "server.py"],
        "base_branches": ["master"],
    }
    config = {
        "worker_id": "machine",
        "root_path": str(tmp_path / "machine"),
        "advertised_host": "devbox.internal",
        "services": {"aeolus": {"runtime_profiles": {"test": profile}}},
    }
    app = FastAPI()
    app.state.deployment_service = service
    app.include_router(router)
    asgi, network = httpx.ASGITransport(app=app), httpx.AsyncHTTPTransport()

    async def route(request):
        return await (asgi if request.url.host == "test" else network).handle_async_request(request)

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(route)) as client:
        manager = MachineWorker(config, client)
        session = await register(service, {**REGISTRATION, "runtime_profiles": capabilities(runtime_profiles(config))})
        manager.session_id = session
        for task_id in ("one", "two"):
            await review(registry, task_id, sha)
            await service.run("enqueue", task_id, "machine", task_id, 1, sha)
            claimed = await service.run("claim", "machine", {"session_id": session, "request_id": task_id})
            await manager.start_instance(claimed)
        try:
            await asyncio.wait_for(asyncio.gather(*manager.tasks.values()), 20)
            assert len(set(manager.ports.values())) == 2
            assert all(i.readiness_owned() for i in manager.instances.values())
            for task_id in ("one", "two"):
                snapshot = await service.run("snapshot", task_id)
                assert snapshot["deployment"]["stage"] == "maat_contract_required", snapshot["deployment"]
            original = manager
            manager = MachineWorker(config, client)
            replacement = await service.run(
                "register", {**REGISTRATION, "runtime_profiles": capabilities(manager.profiles)}
            )
            manager.session_id = replacement["session_id"]
            await manager.restore(replacement["deployments"])
            await manager.tick(reconciled=True)
            one = (await service.run("snapshot", "one"))["deployment"]
            await service.run("release", "one", one["deployment_id"], 1)
            await manager.tick()
            await asyncio.gather(*(entry[1] for entry in manager.maintenance_tasks.values()))
            assert len(manager.instances) == 1
            assert next(iter(manager.instances.values())).readiness_owned()
            assert (await service.run("snapshot", "one"))["deployment"]["released"]
            assert len(original.instances) == 2  # restoration did not restart or mutate another manager's records
        finally:
            for task_id in ("one", "two"):
                record = (await service.run("snapshot", task_id))["deployment"]
                if record and not record["released"]:
                    await service.run("release", task_id, record["deployment_id"], 1)
            await manager.tick()
            await asyncio.gather(*(entry[1] for entry in manager.maintenance_tasks.values()))
        assert not manager.instances and not manager.ports
    await network.aclose()


@pytest.mark.asyncio
async def test_missing_protocol_cannot_replace_registered_worker(system):
    _, _, service = system
    await register(service)
    with pytest.raises(InvalidTaskTransition, match="only deployment protocol v2"):
        await service.run(
            "register", {key: value for key, value in REGISTRATION.items() if key != "protocol_version"}
        )


@pytest.mark.asyncio
async def test_pending_maat_stop_replays_do_not_notify_or_repeat_release_commands(system):
    sf, registry, service = system
    session = await register(service)
    await review(registry, "one")
    record = await deploy(service, "one", session)
    with sf() as db:
        old = db.get(Deployment, record["deployment_id"])
        old.data = {**old.data, "maat_env_id": "old-environment"}
        db.commit()
    await service.run("release", "one", record["deployment_id"], 1)
    assert len((await service.run("heartbeat", "machine", {"session_id": session}))["commands"]) == 1
    body = {"session_id": session, "deployment_id": record["deployment_id"], "generation": 1}
    first = await service.run("stopped", "machine", body)
    assert first["notify_task_id"] == "one" and not first["port_released"]
    for _ in range(3):
        replay = await service.run("stopped", "machine", body)
        assert replay["accepted"] and not replay["port_released"] and "notify_task_id" not in replay
    assert not (await service.run("heartbeat", "machine", {"session_id": session}))["commands"]
    # A later confirmed MAAT cleanup can release the reserved port and notify once.
    with sf() as db:
        old = db.get(Deployment, record["deployment_id"])
        old.data = {**old.data, "maat_cleanup_verified": True}
        db.commit()
    assert len((await service.run("heartbeat", "machine", {"session_id": session}))["commands"]) == 1
    cleared = await service.run("stopped", "machine", body)
    assert cleared["port_released"] and cleared["notify_task_id"] == "one"
    assert "notify_task_id" not in await service.run("stopped", "machine", body)


@pytest.mark.asyncio
async def test_lost_stop_ack_retains_the_original_port_release_decision(system):
    _, registry, service = system
    session = await register(service)
    await review(registry, "one")
    record = await deploy(service, "one", session)
    await service.run("release", "one", record["deployment_id"], 1)
    body = {"session_id": session, "deployment_id": record["deployment_id"], "generation": 1}
    first = await service.run("stopped", "machine", body)
    replay = await service.run("stopped", "machine", body)
    assert first["port_released"] and replay["port_released"]


@pytest.mark.asyncio
async def test_completion_requires_release_confirmation_and_cleans_only_its_task(system):
    _, registry, service = system
    session = await register(service)
    for task_id in ("one", "two"):
        await review(registry, task_id)
        await deploy(service, task_id, session)
    with pytest.raises(InvalidTaskTransition, match="release_resources"):
        await registry.begin_completion("one", "complete", False, allow_release=False)
    await registry.begin_completion("one", "complete", False, allow_release=True)
    commands = (await service.run("heartbeat", "machine", {"session_id": session}))["commands"]
    one = (await service.run("snapshot", "one"))["deployment"]
    assert [c["deployment_id"] for c in commands] == [one["deployment_id"]]
    assert (await service.run("snapshot", "two"))["deployment"]["status"] == "running"


@pytest.mark.asyncio
async def test_cancelled_claim_replay_cannot_deliver_a_different_instance(system):
    _, registry, service = system
    session = await register(service)
    for task_id in ("one", "two"):
        await review(registry, task_id)
        await service.run("enqueue", task_id, "machine", task_id, 1, SHA)
    receipt = {"session_id": session, "request_id": "lost-response"}
    first = await service.run("claim", "machine", receipt)
    await service.run("release", first["task_id"], first["deployment_id"], 1)
    replay = await service.run("claim", "machine", receipt)
    assert replay["deployment_id"] == first["deployment_id"]
    assert replay["status"] == "cancel_requested"
    second = await service.run("claim", "machine", {"session_id": session, "request_id": "new-receipt"})
    assert second["deployment_id"] != first["deployment_id"]


@pytest.mark.asyncio
async def test_stale_card_runtime_revision_is_rejected_before_deployment(system):
    _, registry, service = system
    session = await register(service)
    await review(registry, "one")
    with pytest.raises(InvalidTaskTransition, match="配置已变更"):
        await service.run("enqueue", "one", "machine", "stale-config", 1, SHA, "py311", "c" * 64)
    assert (await service.run("snapshot", "one"))["deployment"] is None
    record = await service.run(
        "enqueue", "one", "machine", "fresh-config", 1, SHA, "py311", CAPABILITY["profile_revision"]
    )
    assert record["profile_revision"] == CAPABILITY["profile_revision"]
    claimed = await service.run("claim", "machine", {"session_id": session, "request_id": "fresh"})
    assert claimed["deployment_id"] == record["deployment_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 99])
async def test_registration_api_rejects_unsupported_protocol_without_mutating_worker(system, version):
    sf, _, service = system
    await register(service)
    app = FastAPI()
    app.include_router(router)
    app.state.deployment_service = service
    body = {key: value for key, value in REGISTRATION.items() if key != "protocol_version"}
    if version is not None:
        body["protocol_version"] = version
    with sf() as db:
        original = db.get(DeploymentWorker, "machine").session_id
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.post("/api/deployment-workers/register", json=body)).status_code == 422
    with sf() as db:
        assert db.get(DeploymentWorker, "machine").session_id == original


@pytest.mark.asyncio
async def test_unsupported_records_are_ignored_by_current_deployment_operations(system):
    from datetime import datetime

    from conciliator.services.aeolus_registry import TaskNotFound

    sf, registry, service = system
    session = await register(service)
    await review(registry, "one")
    with sf() as db:
        db.add(DeploymentWorker(
            worker_id="unsupported", session_id="unsupported-session", last_seen_at=datetime.now(), ready=True,
            data={"display_name": "unsupported", "protocol_version": 99},
        ))
        db.add(Deployment(
            deployment_id="unsupported", task_id="one", worker_id="unsupported", request_id="unsupported", round_no=1,
            commit_sha=SHA, generation=10, status="running", stage="dispatching", created_at=datetime.now(),
            data={"protocol_version": 99},
        ))
        db.commit()
    snapshot = await service.run("snapshot", "one")
    assert snapshot["deployment"] is None
    assert [w["worker_id"] for w in snapshot["deployment_workers"]] == ["machine"]
    with pytest.raises(TaskNotFound):
        await service.run("heartbeat", "unsupported", {"session_id": "unsupported-session", "reconciled": True})
    new = await deploy(service, "one", session)
    await registry.start_feedback("one", "继续修改", "feedback")
    with sf() as db:
        assert db.get(Deployment, "unsupported").status == "running"
        assert not db.get(Deployment, "unsupported").data.get("release_requested")
        assert db.get(Deployment, new["deployment_id"]).data["release_requested"]
