import copy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conciliator.api.aeolus import router
from conciliator.domain.aeolus_task import AeolusRound, AeolusStartEvent, AeolusState, DeploymentVerification
from conciliator.integrations.ppe.client import PpeClient, load_profile
from conciliator.integrations.ppe.scm import PlatformError, ScmClient, ScmVersion
from conciliator.repositories.models import Base, Deployment, DeploymentWorker
from conciliator.repositories.ppe_models import PpeDeployment, PpeEnvironment
from conciliator.services.aeolus_registry import AeolusTaskRegistry, InvalidTaskTransition, _database_now
from conciliator.services.aeolus_service import AeolusService
from conciliator.services.ppe_deployment_service import PpeDeploymentService, invalidate_ppe, task_lock, verify_ppe
from conciliator.services.ppe_runner import PpeRunner
from conciliator.services.runtime_selection import detect

SHA = "a" * 40


@pytest.fixture
async def system(request):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sf = sessionmaker(engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(sf, ppe_enabled=True)
    await registry.create_from_event(AeolusStartEvent("task", "space", 42, "node", "uuid", "user", ()))
    await registry.append_round("task", AeolusRound(1, "review", commit_sha=SHA))
    await registry.transition("task", {AeolusState.RECEIVED}, AeolusState.REVIEW)
    runtime = "2.7" if getattr(request, "param", "") == "py27-unrequested" else None
    profile = load_profile(python_version=runtime)
    if runtime:
        await registry.update("task", evaluation={"code_context": {
            "repository": "dp/aeolus", "base_branch": "release-aeolus-2.104.0",
        }})
    service = PpeDeploymentService(sf, profile)
    row = (
        (await service.create("task", "request", 1, SHA, service.revision))["ppe_deployment"]
        if getattr(request, "param", None) not in {"unrequested", "py27-unrequested"}
        else {"environment_name": "", "environment_id": ""}
    )
    env = {
        "name": row["environment_name"], "id": 123, "desc": "Conciliator PPE " + row["environment_id"],
        "env_managers": ["huaguojin"],
    }
    owned = {
        "service_id": "456",
        "clusters": [{"id": "789", "name": "online"}, {"id": "790", "name": "query_online_mirror"}],
    }
    observation = {**owned, "ready": True, "versions_match": True, "routing_match": True, "platform_id": "123"}
    ppe = SimpleNamespace(
        prepare=AsyncMock(return_value={"env": {"name": row["environment_name"]}}),
        environment=AsyncMock(return_value=env),
        resources=AsyncMock(return_value=owned),
        observe=AsyncMock(return_value=observation),
        precheck=AsyncMock(return_value={"pre_check_result": [{"name": "PreCheckQuota", "is_break": False}]}),
        submit=AsyncMock(return_value="111"),
        ticket=AsyncMock(return_value={"status": "Succeed"}),
        remove=AsyncMock(return_value={"id_str": "222"}),
        absent=AsyncMock(return_value=True),
        approvals=AsyncMock(return_value=[]),
    )
    writes = []

    def transport(request):
        if request.url.path == "/api/v4/suggest/cluster_by_id":
            cluster_id = int(request.url.params["cluster_id"])
            cluster = next(c for c in profile["meta"]["clusters"] if c["base_cluster_id"] == cluster_id)
            return httpx.Response(200, json={
                "name": cluster["name"], "base_cluster": {"id": cluster_id}, "new_env_vars": {},
            })
        if request.method == "POST":
            writes.append(request)
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {"context": {
                        "repo_id": profile["scm_repo_id"], "version_version": "1.0.0.1542", "version_id": 999,
                    }},
                },
            )
        if request.url.path.endswith("latest_version/"):
            return httpx.Response(200, json={"code": 0, "data": {"version": "1.0.0.1541"}})
        if request.url.path == "/api/v2/versions/":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "version": "1.0.0.1542",
                            "repos": profile["scm_repo_id"],
                            "base_commit_hash": SHA,
                            "arch": ["x86_64"],
                            "id": 999,
                            "status": "build_ok",
                            "type": "test",
                        }
                    ],
                    "next": None,
                },
            )
        return httpx.Response(
            200,
            json={
                "version": "1.0.0.1542",
                "repos": profile["scm_repo_id"],
                "base_commit_hash": SHA,
                "arch": ["x86_64"],
                "id": 999,
                "status": "build_ok",
                "type": "test",
            },
        )

    http = httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(transport))
    planner = PpeClient(http, profile)
    planner.resources = ppe.resources
    planner.observe = ppe.observe
    planner.service = AsyncMock(return_value={
        "meta": {"psm": profile["psm"]},
        "instances": [{"name": c["name"], "id": c["base_cluster_id"]} for c in profile["meta"]["clusters"]],
    })
    ppe.prepare.side_effect = planner.prepare
    icm = SimpleNamespace(
        create=AsyncMock(return_value={"image_id": 217258, "image_version_id": "1000"}),
        version=AsyncMock(
            return_value={
                "image_id": 217258,
                "image_version_id": "1000",
                "image_version": "1.0.2.555",
                "image_status": "ok",
                "resolved_base_image": "base:fixed",
            }
        ),
    )
    runner = PpeRunner(service, http, icm, ppe)
    yield SimpleNamespace(
        sf=sf, registry=registry, service=service, row=row, runner=runner, ppe=ppe, icm=icm, writes=writes,
        env=env, prepare=planner.prepare,
    )
    await http.aclose()
    engine.dispose()


@pytest.mark.parametrize("system", ["unrequested"], indirect=True)
async def test_vqs_task_selects_its_own_profile_revision_and_scm_repository(system):
    s = system
    vqs = load_profile(repository="dp/VizQueryService")
    service = PpeDeploymentService(s.sf, load_profile(), profiles=(vqs,))
    with s.sf() as db:
        task_lock(db, "task").evaluation = {
            "code_context": {"repository": "dp/VizQueryService"},
            "_runtime_detection": detect("dp/VizQueryService", SHA, {
                "requirements.txt": "antlr4-python3-runtime==4.7.2",
            }),
        }
        db.commit()
    capability = (await service.snapshot("task"))["ppe_capability"]
    assert capability["enabled"] and capability["repository"] == "dp/VizQueryService"
    with pytest.raises(InvalidTaskTransition, match="模板已变更"):
        await service.create("task", "wrong-profile", 1, SHA, service.revision)
    result = await service.create("task", "vqs-request", 1, SHA, capability["profile_revision"])
    row = result["ppe_deployment"]
    assert row["environment_name"].startswith("ppe_ai_vqs_")
    assert "/53349/" in row["scm_url"]
    with s.sf() as db:
        record = db.get(PpeDeployment, row["deployment_id"])
        assert record.data["profile"] == vqs
        record.stage = "artifact_wait"
        db.commit()

    calls = []

    def transport(req):
        calls.append(req)
        if req.url.path == "/api/v2/versions/":
            assert req.url.params["repos_id"] == "53349"
            assert req.url.params["commit"] == SHA
            return httpx.Response(200, json={"results": [{
                "repos": 53349, "base_commit_hash": SHA, "version": "1.0.0.6127",
                "id": 999, "arch": ["x86_64"], "status": "build_ok", "type": "test",
            }], "next": None})
        assert req.url.path == "/api/repos/53349/versions/1.0.0.6127"
        return httpx.Response(200, json={
            "repos": 53349, "base_commit_hash": SHA, "version": "1.0.0.6127",
            "id": 999, "arch": ["x86_64"], "status": "build_ok", "type": "test",
        })

    async with httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(transport)) as http:
        runner = PpeRunner(service, http, s.icm, s.ppe)
        await runner._advance(runner._claim(row["deployment_id"]))
    with s.sf() as db:
        record = db.get(PpeDeployment, row["deployment_id"])
        assert record.stage == "image_submit" and record.data["build_ok"]
    assert len(calls) == 2


@pytest.mark.parametrize("system", ["unrequested"], indirect=True)
async def test_unconfigured_repository_has_no_ppe_capability(system):
    with system.sf() as db:
        task_lock(db, "task").evaluation = {"code_context": {"repository": "other/repo"}}
        db.commit()
    assert not (await system.service.snapshot("task"))["ppe_capability"]["enabled"]
    with pytest.raises(InvalidTaskTransition):
        await system.service.create("task", "request", 1, SHA, system.service.revision)


def record(s):
    with s.sf() as db:
        r = db.get(PpeDeployment, s.row["deployment_id"])
        e = db.get(PpeEnvironment, s.row["environment_id"])
        return copy.deepcopy(r), copy.deepcopy(e)


async def tick(s):
    with s.sf() as db:
        r = db.get(PpeDeployment, s.row["deployment_id"])
        r.next_poll_at = _database_now(db) - timedelta(seconds=1)
        db.commit()
    await s.runner.run_once()


async def through(s, stage):
    for _ in range(20):
        if record(s)[0].stage == stage:
            return
        await tick(s)
    raise AssertionError(record(s)[0].stage)


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["unrequested", "py27-unrequested"], indirect=True)
async def test_card_api_deploy_update_confirm_and_release(system, monkeypatch):
    """Real task API, registry, ledger and runner; only remote platforms are faked."""
    s = system
    bits = bits_system(s, monkeypatch)
    s.ppe.environment.return_value = None

    versions = {
        "1.0.0.1542": ScmVersion("1.0.0.1542", "999", SHA, "build_ok", "test", ("x86_64",)),
        "1.0.0.1543": ScmVersion("1.0.0.1543", "1001", "b" * 40, "build_ok", "test", ("x86_64",)),
    }
    monkeypatch.setattr(ScmClient, "version", AsyncMock(side_effect=lambda version: versions[version]))
    meegle = AsyncMock()
    service = AeolusService(
        registry=s.registry, document_reader=AsyncMock(), coco=AsyncMock(),
        code_host=AsyncMock(), meegle=meegle, ppe_deployments=s.service,
        minimum_processing_seconds=0,
    )
    app = FastAPI()
    app.state.aeolus_service = service
    app.state.ppe_deployment_service = s.service
    app.include_router(router)
    path = "/api/aeolus/tasks/task"
    lookup = "/api/aeolus/tasks/by-node?space_id=space&work_item_id=42&node_id=node&work_object_id=story"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
        initial = (await api.get(lookup)).json()
        assert initial["ppe_capability"]["enabled"]
        assert initial["deployment_workers"] is None and initial["ppe_deployment"] is None
        headers = {"X-Aeolus-Task-Token": initial["action_token"]}
        body = {"request_id": "card-first", "round_no": 1, "commit_sha": SHA, "profile_revision": s.service.revision}
        response = await api.post(path + "/ppe-deployments", headers=headers, json=body)
        assert response.status_code == 202 and "action_token" not in response.json()
        s.row = response.json()["ppe_deployment"]
        first = copy.deepcopy(s.row)
        assert first["status"] == "pending" and first["stage"] == "preflight"
        assert first["current_deployment"] is None
        assert not first["can_view_environment"] and not first["can_access_environment"]
        assert not first["environment_url"] and not first["access_hint"]
        replay = await api.post(path + "/ppe-deployments", headers=headers, json=body)
        assert replay.json()["ppe_deployment"]["deployment_id"] == first["deployment_id"]
        s.env.update(name=first["environment_name"], desc="Conciliator PPE " + first["environment_id"])
        await through(s, "bits_wait")
        build = (await api.get(path + f"/ppe-deployments/{first['deployment_id']}", headers=headers)).json()
        assert build["ppe_deployment"]["bits_run_id"] == "42"
        assert "runId=42" in build["ppe_deployment"]["bits_run_url"]
        assert not build["ppe_deployment"]["can_view_environment"]
        await through(s, "ticket_wait")
        assert not (await s.service.snapshot("task"))["ppe_deployment"]["can_view_environment"]
        s.ppe.environment.return_value = s.env
        s.ppe.ticket.return_value = {"status": "Running"}
        await tick(s)
        deploying = (await s.service.snapshot("task"))["ppe_deployment"]
        assert deploying["can_view_environment"] and not deploying["can_access_environment"]
        assert first["environment_name"] in deploying["environment_url"] and not deploying["access_hint"]
        s.ppe.ticket.return_value = {"status": "Succeed"}
        await through(s, "entry_pending")
        ready = (await api.get(lookup)).json()["ppe_deployment"]
        assert ready["can_verify"] and not ready["verified"]
        assert ready["can_view_environment"] and ready["can_access_environment"]
        assert ready["status_label"] == "PPE 部署完成，待业务验证"
        assert ready["image_version"] == "1.0.2.555" and len(ready["clusters"]) == 2
        assert ready["current_deployment"]["deployment_id"] == first["deployment_id"]
        assert bits.submit.await_count == 1

        # Coding feedback creates a new round; deployment must reuse the same environment.
        await s.registry.start_feedback("task", "second commit", "card-feedback")
        assert not (await api.get(lookup)).json()["ppe_deployment"]["can_verify"]
        await s.registry.update_last_round("task", status="review", commit_sha="b" * 40)
        await s.registry.transition("task", {AeolusState.RECODING}, AeolusState.REVIEW)
        stale = await api.post(path + "/ppe-deployments", headers=headers, json={**body, "request_id": "stale"})
        assert stale.status_code == 409
        bits.submit.return_value = {"bits_run_id": "43", "bits_run_seq": "3"}
        bits.run.return_value = {
            "bits_run_status": 8, "bits_failed": False, "bits_success": True,
            "scm_version": "1.0.0.1543", "scm_version_id": "1001",
        }
        s.icm.version.return_value.update(image_version_id="1002", image_version="1.0.2.556")
        s.icm.create.return_value = {"image_id": 217258, "image_version_id": "1002"}
        s.ppe.observe.return_value["versions_match"] = False
        second_body = {**body, "request_id": "card-second", "round_no": 2, "commit_sha": "b" * 40}
        response = await api.post(path + "/ppe-deployments", headers=headers, json=second_body)
        assert response.status_code == 202
        s.row = response.json()["ppe_deployment"]
        assert (s.row["environment_id"], s.row["generation"]) == (first["environment_id"], first["generation"])
        previous = s.row["current_deployment"]
        assert previous == {
            "deployment_id": first["deployment_id"], "round_no": 1,
            "commit_sha": SHA, "image_version": "1.0.2.555",
        }
        assert not s.row["can_verify"]
        await through(s, "image_wait")
        building = (await api.get(lookup)).json()["ppe_deployment"]
        assert building["current_deployment"] == previous
        assert building["environment_name"] == first["environment_name"]
        assert not building["can_verify"]
        assert building["can_view_environment"] and building["can_access_environment"]
        assert building["access_hint"] == ready["access_hint"]
        await through(s, "ticket_wait")
        s.ppe.observe.return_value["versions_match"] = True
        await through(s, "entry_pending")
        ready = (await api.get(lookup)).json()["ppe_deployment"]
        assert ready["can_verify"] and ready["image_version"] == "1.0.2.556"
        assert ready["commit_sha"] == "b" * 40 and bits.submit.await_count == 2
        assert ready["current_deployment"]["deployment_id"] == ready["deployment_id"]
        assert ready["current_deployment"]["image_version"] == "1.0.2.556"
        upgrade = s.ppe.submit.call_args.args[0]["deployment"]["data.aeolus.api_v3"]["meta"]
        assert upgrade["image_version"] == "1.0.2.556"
        stale_confirm = await api.post(path + "/confirm", headers=headers, json={
            "request_id": "stale-confirm", "mode": "verified",
            "deployment_target": "ppe", "deployment_id": first["deployment_id"],
        })
        assert stale_confirm.status_code == 409
        s.ppe.absent.return_value = False
        response = await api.post(path + "/confirm", headers=headers, json={
            "request_id": "card-confirm", "mode": "verified",
            "deployment_target": "ppe", "deployment_id": ready["deployment_id"], "release_resources": True,
        })
        assert response.status_code == 202
        await service.wait_for_idle()
        completed = (await api.get(lookup)).json()
        assert completed["state"] == "completed"
        assert completed["ppe_deployment"]["verified"] and completed["ppe_deployment"]["release_requested"]
        assert not completed["ppe_deployment"]["can_view_environment"]
        assert not completed["ppe_deployment"]["can_access_environment"]
        meegle.edit_status.assert_awaited_once()
        meegle.confirm.assert_awaited_once()
        await through(s, "release_verify")
        await tick(s)
        assert not (await api.get(lookup)).json()["ppe_deployment"]["released"]
        s.ppe.absent.return_value = True
        await through(s, "released")
        released = (await api.get(lookup)).json()["ppe_deployment"]
        assert released["released"] and not released["can_verify"] and not released["can_release"]
        assert released["current_deployment"] is None
        s.ppe.remove.assert_awaited_once()
        repeat = await api.post(path + f"/ppe-deployments/{ready['deployment_id']}/release", headers=headers, json={
            "environment_id": ready["environment_id"], "generation": ready["generation"],
        })
        assert repeat.status_code == 202 and repeat.json()["ppe_deployment"]["released"]
        assert s.ppe.remove.await_count == 1
    await service.aclose()


@pytest.mark.asyncio
async def test_full_build_persists_image_identity_then_verifies_both_clusters(system):
    s = system
    s.ppe.environment.return_value = None
    await through(s, "image_wait")
    r, _ = record(s)
    assert r.data["operator"] == "huaguojin"
    assert r.data["image_version_id"] == "1000"
    assert not s.writes  # Codebase already built this SHA; do not submit another SCM build.
    assert s.icm.create.call_args.args[0]["username"] == "huaguojin"
    await through(s, "ticket_wait")
    assert record(s)[0].data["ticket_id"] == "111"
    s.ppe.environment.return_value = s.env
    await through(s, "entry_pending")
    assert record(s)[0].status == "deployed"
    assert record(s)[1].active_deployment_id is None
    ready = (await s.service.snapshot("task"))["ppe_deployment"]
    assert ready["can_verify"]
    assert not ready["can_deploy"]
    with s.sf() as db:
        verify_ppe(db, "task", s.row["deployment_id"])
        db.commit()
    assert record(s)[0].data["verified"]

@pytest.mark.asyncio
async def test_ppe_deploy_button_returns_for_new_round_or_released_environment(system):
    s = system
    await through(s, "entry_pending")
    assert not (await s.service.snapshot("task"))["ppe_deployment"]["can_deploy"]
    await s.registry.append_round("task", AeolusRound(2, "review", commit_sha="b" * 40))
    assert (await s.service.snapshot("task"))["ppe_deployment"]["can_deploy"]
    await s.service.release("task", s.row["deployment_id"], s.row["environment_id"], 1)
    assert not (await s.service.snapshot("task"))["ppe_deployment"]["can_deploy"]
    await through(s, "released")
    assert (await s.service.snapshot("task"))["ppe_deployment"]["can_deploy"]


@pytest.mark.asyncio
async def test_request_dedup_and_same_second_ordering(system):
    s = system
    result = await s.service.create("task", "request", 1, SHA, s.service.revision)
    assert result["ppe_deployment"]["deployment_id"] == s.row["deployment_id"]
    with pytest.raises(InvalidTaskTransition):
        await s.service.create("task", "request", 2, "b" * 40, s.service.revision)
    with pytest.raises(InvalidTaskTransition):
        await s.service.create("task", "another", 1, SHA, s.service.revision)
    await through(s, "entry_pending")
    second = (await s.service.create("task", "second", 1, SHA, s.service.revision))["ppe_deployment"]
    assert second["environment_id"] == s.row["environment_id"]
    assert (await s.service.snapshot("task"))["ppe_deployment"]["deployment_id"] == second["deployment_id"]
    with s.sf() as db:
        new = db.get(PpeDeployment, second["deployment_id"])
        assert new.sequence == 2
        assert new.data["image_version_id"] == "1000"


@pytest.mark.asyncio
async def test_unknown_write_is_not_replayed_after_restart(system):
    s = system
    await through(s, "image_submit")
    s.icm.create.side_effect = PlatformError("结果待确认", uncertain=True)
    await tick(s)
    assert record(s)[0].stage == "unknown"
    restarted = PpeRunner(s.service, s.runner.http, s.icm, s.ppe)
    s.runner = restarted
    for _ in range(3):
        await tick(s)
    assert s.icm.create.await_count == 1
    assert record(s)[1].active_deployment_id
    with pytest.raises(InvalidTaskTransition):
        await s.service.create("task", "new", 1, SHA, s.service.revision)


@pytest.mark.asyncio
async def test_expired_lease_with_intent_quarantines_but_accepts_late_identity(system):
    s = system
    item = s.runner._claim(s.row["deployment_id"])
    assert s.runner._claim(s.row["deployment_id"]) is None
    assert s.runner._intent(item, "icm", "nonce", {"pinned": "recipe"})
    with s.sf() as db:
        row = db.get(PpeDeployment, s.row["deployment_id"])
        row.lease_until = _database_now(db) - timedelta(seconds=1)
        db.commit()
    assert s.runner._claim(s.row["deployment_id"]) is None
    assert record(s)[0].stage == "unknown"
    item["intent"] = "nonce"
    assert s.runner._save(
        item, {"stage": "image_wait", "status": "running", "clear_intent": True, "data": {"image_version_id": "1000"}}
    )
    assert record(s)[0].data["image_version_id"] == "1000"
    assert "intent" not in record(s)[0].data


@pytest.mark.asyncio
async def test_feedback_stops_old_writes_and_preserves_environment(system):
    s = system
    await through(s, "image_submit")
    with s.sf() as db:
        invalidate_ppe(db, "task", release=False)
        db.commit()
    await s.registry.start_feedback("task", "change it", "feedback")
    await tick(s)
    assert record(s)[0].stage == "superseded"
    s.icm.create.assert_not_awaited()
    assert record(s)[1].resource_state != "released"


@pytest.mark.asyncio
async def test_release_waits_for_inflight_ticket_and_never_loops_failed_delete(system):
    s = system
    s.ppe.environment.return_value = None
    await through(s, "ticket_wait")
    s.ppe.environment.return_value = s.env
    s.ppe.ticket.return_value = {"status": "Running"}
    await s.service.release("task", s.row["deployment_id"], s.row["environment_id"], 1)
    await tick(s)
    s.ppe.remove.assert_not_awaited()
    assert record(s)[1].platform_id == "123"
    s.ppe.ticket.return_value = {"status": "Succeed"}
    await tick(s)
    await through(s, "release_wait")
    s.ppe.ticket.return_value = {"status": "Failed"}
    await tick(s)
    assert record(s)[0].status == "failed"
    assert record(s)[1].resource_state == "cleanup_failed"
    for _ in range(3):
        await tick(s)
    assert s.ppe.remove.await_count == 1
    assert (await s.service.snapshot("task"))["ppe_deployment"]["can_release"]


@pytest.mark.asyncio
@pytest.mark.parametrize("already_deleted", [False, True])
async def test_release_requires_actual_tce_absence_and_new_generation_is_fenced(system, already_deleted):
    s = system
    await through(s, "entry_pending")
    if already_deleted:
        s.ppe.environment.return_value = {**s.env, "deleted_at": 1793781794}
    await s.service.release("task", s.row["deployment_id"], s.row["environment_id"], 1)
    await through(s, "release_verify")
    assert s.ppe.remove.await_count == (0 if already_deleted else 1)
    s.ppe.absent.return_value = False
    await tick(s)
    assert record(s)[1].resource_state != "released"
    s.ppe.absent.return_value = True
    await tick(s)
    assert record(s)[1].resource_state == "released"
    new = (await s.service.create("task", "new", 1, SHA, s.service.revision))["ppe_deployment"]
    assert new["generation"] == 2 and new["environment_id"] != s.row["environment_id"]
    with pytest.raises(InvalidTaskTransition):
        await s.service.release("task", s.row["deployment_id"], s.row["environment_id"], 1)


@pytest.mark.asyncio
async def test_known_job_timeout_blocks_new_writes(system):
    s = system
    await through(s, "image_wait")
    with s.sf() as db:
        row = db.get(PpeDeployment, s.row["deployment_id"])
        row.data = {**row.data, "stage_started_at": (_database_now(db) - timedelta(hours=3)).isoformat()}
        db.commit()
    await tick(s)
    assert record(s)[0].stage == "unknown"
    assert record(s)[0].data["image_version_id"] == "1000"
    s.ppe.submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_cluster_repair_creates_only_missing_cluster(system):
    s = system
    s.ppe.resources.return_value = {"service_id": "456", "clusters": [{"id": "789", "name": "online"}]}
    await through(s, "ticket_wait")
    repair = s.ppe.submit.call_args.args[0]["deployment"]["data.aeolus.api_v3"]
    assert repair["action"] == "create_cluster"
    assert [c["name"] for c in repair["meta"]["clusters"]] == ["query_online_mirror"]
    assert record(s)[0].data["after_ticket"] == "deploy"
    assert record(s)[1].platform_id == "123"


async def test_legacy_single_service_does_not_repeat_finished_upgrade(system):
    s = system
    s.ppe.observe.return_value["versions_match"] = False
    await through(s, "ticket_wait")
    assert s.ppe.submit.await_count == 1
    await through(s, "failed")
    assert s.ppe.submit.await_count == 1
    assert record(s)[0].data["upgraded_services"] == [s.service.profile["psm"]]
    assert "不会重复提交" in record(s)[0].error


@pytest.mark.asyncio
async def test_health_observation_expires_and_does_not_spam_notifications(system):
    s = system
    s.runner.notifier = AsyncMock()
    await through(s, "entry_pending")
    count = s.runner.notifier.await_count
    assert count == 1
    s.runner.notifier.assert_awaited_once_with("task", s.row["deployment_id"])
    await tick(s)
    await tick(s)
    assert s.runner.notifier.await_count == count
    with s.sf() as db:
        env = db.get(PpeEnvironment, s.row["environment_id"])
        env.data = {**env.data, "observed_at": (_database_now(db) - timedelta(minutes=1)).isoformat()}
        db.commit()
    with s.sf() as db, pytest.raises(InvalidTaskTransition):
        verify_ppe(db, "task", s.row["deployment_id"])


async def test_ready_health_changes_keep_polling_and_restore_verification(system):
    s = system
    s.runner.notifier = AsyncMock()
    await through(s, "entry_pending")
    ready = copy.deepcopy(s.ppe.observe.return_value)
    _, env = record(s)
    writes = s.ppe.submit.await_count, s.ppe.remove.await_count, s.icm.create.await_count
    observations = s.ppe.observe.await_count
    s.ppe.observe.return_value = {**ready, "ready": False}

    for poll_no in range(1, 3):
        await tick(s)
        snapshot = (await s.service.snapshot("task"))["ppe_deployment"]
        assert (snapshot["status"], snapshot["stage"]) == ("deployed", "entry_pending")
        assert snapshot["status_label"] == "PPE 暂未就绪，正在重新检查"
        assert not snapshot["error"] and not snapshot["can_verify"]
        assert not snapshot["can_retry"] and not snapshot["can_deploy"]
        _, environment = record(s)
        assert environment.resource_state == "running" and not environment.data["clusters_ready"]
        assert environment.current_deployment_id == env.current_deployment_id
        assert s.ppe.observe.await_count == observations + poll_no
        with s.sf() as db, pytest.raises(InvalidTaskTransition, match="需要重新核验"):
            verify_ppe(db, "task", s.row["deployment_id"])

    s.ppe.observe.return_value = ready
    await tick(s)
    recovered = (await s.service.snapshot("task"))["ppe_deployment"]
    assert recovered["status_label"] == "PPE 部署完成，待业务验证"
    assert not recovered["error"] and recovered["can_verify"]
    with s.sf() as db:
        verify_ppe(db, "task", s.row["deployment_id"])
    assert s.runner.notifier.await_count == 1
    assert (s.ppe.submit.await_count, s.ppe.remove.await_count, s.icm.create.await_count) == writes


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["entry_pending", "released"])
async def test_notification_failure_retries_after_terminal_state(system, terminal):
    s = system
    if terminal == "released":
        await through(s, "entry_pending")
        await s.service.release("task", s.row["deployment_id"], s.row["environment_id"], 1)
    s.runner.notifier = AsyncMock(side_effect=RuntimeError("offline"))
    await through(s, terminal)
    assert record(s)[0].data["notification_pending"]
    s.runner.notifier.side_effect = None
    await tick(s)
    assert not record(s)[0].data["notification_pending"]
    calls = s.runner.notifier.await_count
    await tick(s)
    assert s.runner.notifier.await_count == calls


@pytest.mark.asyncio
async def test_ppe_notifications_skip_progress_and_report_failure_success_and_release(system):
    s = system
    s.runner.notifier = AsyncMock()
    await through(s, "image_wait")
    s.runner.notifier.assert_not_awaited()
    s.icm.version.side_effect = PlatformError("镜像构建失败")
    # image_wait reads can recover; only a final failure should send a result.
    await tick(s)
    s.runner.notifier.assert_not_awaited()
    s.icm.version.side_effect = None
    await through(s, "deploy")
    s.ppe.observe.return_value["versions_match"] = False
    s.ppe.prepare.side_effect = PlatformError("部署预检失败")
    await tick(s)
    assert record(s)[0].status == "failed"
    s.runner.notifier.assert_awaited_once_with("task", s.row["deployment_id"])
    s.ppe.prepare.side_effect = s.prepare
    s.ppe.observe.return_value["versions_match"] = True
    s.row = (await s.service.retry("task", s.row["deployment_id"], "retry-result"))["ppe_deployment"]
    await through(s, "entry_pending")
    assert s.runner.notifier.await_count == 2
    await s.service.release("task", s.row["deployment_id"], s.row["environment_id"], 1)
    await through(s, "released")
    assert s.runner.notifier.await_count == 3
    await tick(s)
    assert s.runner.notifier.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("superseded", [False, True])
async def test_ppe_release_notifies_only_after_resource_absence_even_for_old_code(system, superseded):
    s = system
    await through(s, "entry_pending")
    with s.sf() as db:
        row = db.get(PpeDeployment, s.row["deployment_id"])
        row.data = {**row.data, "superseded": superseded}
        db.commit()
    s.runner.notifier = AsyncMock()
    await s.service.release("task", s.row["deployment_id"], s.row["environment_id"], 1)
    await through(s, "release_verify")
    s.ppe.absent.return_value = False
    await tick(s)
    s.runner.notifier.assert_not_awaited()
    s.ppe.absent.return_value = True
    await tick(s)
    s.runner.notifier.assert_awaited_once_with("task", s.row["deployment_id"])
    assert record(s)[1].resource_state == "released"
    await tick(s)
    s.runner.notifier.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_legacy_progress_and_obsolete_results_are_discarded(system):
    s = system
    with s.sf() as db:
        row = db.get(PpeDeployment, s.row["deployment_id"])
        row.data = {**row.data, "notification_pending": True}
        db.commit()
    assert s.runner._notifications() == []
    assert not record(s)[0].data["notification_pending"]
    await through(s, "entry_pending")
    await s.service.create("task", "new-result", 1, SHA, s.service.revision)
    assert s.runner._notifications() == []
    assert not record(s)[0].data["notification_pending"]


@pytest.mark.asyncio
async def test_unknown_ppe_operation_notifies_once_and_notification_ack_is_versioned(system):
    s = system
    s.runner.notifier = AsyncMock()
    await through(s, "image_submit")
    s.icm.create.side_effect = PlatformError("结果待确认", uncertain=True)
    await tick(s)
    assert record(s)[0].status == "unknown"
    s.runner.notifier.assert_awaited_once_with("task", s.row["deployment_id"])
    await tick(s)
    assert s.runner.notifier.await_count == 1
    with s.sf() as db:
        row = db.get(PpeDeployment, s.row["deployment_id"])
        version = row.data["notification_version"]
        row.data = {**row.data, "notification_pending": True, "notification_version": version + 1}
        db.commit()
    s.runner._notification_sent(s.row["deployment_id"], version)
    assert record(s)[0].data["notification_pending"]


@pytest.mark.asyncio
async def test_completion_requires_release_consent_and_explicit_ppe_target(system):
    s = system
    await through(s, "entry_pending")
    with pytest.raises(InvalidTaskTransition, match="release_resources"):
        await s.registry.begin_completion("task", "complete", False, allow_release=False)
    with pytest.raises(InvalidTaskTransition, match="not ready"):
        await s.registry.begin_completion("task", "wrong-target", False, s.row["deployment_id"], True)
    task, started = await s.registry.begin_completion("task", "complete", False, s.row["deployment_id"], True, "ppe")
    assert started and task.state == AeolusState.COMPLETING
    assert record(s)[0].data["verified"]
    assert record(s)[1].data["release_requested"]
    assert record(s)[0].stage == "release"


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [None, "devbox", "ppe"])
async def test_combined_verification_is_atomic_and_never_partially_releases(system, stale):
    s = system
    await through(s, "entry_pending")
    registry = AeolusTaskRegistry(s.sf, ppe_enabled=True, deployments_enabled=True)
    with s.sf() as db:
        now = _database_now(db)
        db.add(DeploymentWorker(
            worker_id="box", session_id="session", ready=True, last_seen_at=now,
            data={"protocol_version": 2},
        ))
        db.add(Deployment(
            deployment_id="devbox", task_id="task", worker_id="box", request_id="box",
            round_no=1, commit_sha=SHA, generation=1, status="succeeded", stage="running", created_at=now,
            data={"protocol_version": 2, "maat_entry_verified": True, "claimed": True,
                  "observation": {"service_status": "healthy", "observed_at": (
                      now - timedelta(seconds=60) if stale == "devbox" else now
                  ).isoformat()}},
        ))
        if stale == "ppe":
            env = db.get(PpeEnvironment, s.row["environment_id"])
            env.data = {**env.data, "observed_at": (now - timedelta(seconds=60)).isoformat()}
        db.commit()
    targets = [DeploymentVerification("devbox", "devbox"), DeploymentVerification("ppe", s.row["deployment_id"])]
    if stale == "devbox":
        targets.reverse()  # PPE verifies first, then Devbox fails: neither change may commit.
    if stale:
        with pytest.raises(InvalidTaskTransition):
            await registry.begin_completion("task", "both", False, verification_targets=tuple(targets))
        assert (await registry.get("task")).state == AeolusState.REVIEW
    else:
        task, started = await registry.begin_completion("task", "both", False, verification_targets=tuple(targets))
        assert started and task.state == AeolusState.COMPLETING
        _, repeated = await registry.begin_completion("task", "both", False, verification_targets=tuple(targets))
        assert not repeated
    with s.sf() as db:
        devbox = db.get(Deployment, "devbox")
        ppe, env = record(s)
        assert bool(devbox.data.get("verified")) is (stale is None)
        assert bool(ppe.data.get("verified")) is (stale is None)
        assert bool(devbox.data.get("release_requested")) is (stale is None)
        assert bool(env.data.get("release_requested")) is (stale is None)


@pytest.mark.asyncio
async def test_registry_feedback_invalidates_ready_ppe_without_deleting_it(system):
    s = system
    await through(s, "entry_pending")
    await s.registry.start_feedback("task", "new version", "feedback")
    assert record(s)[0].data["superseded"]
    assert not record(s)[1].data.get("release_requested")
    with s.sf() as db, pytest.raises(InvalidTaskTransition):
        verify_ppe(db, "task", s.row["deployment_id"])


@pytest.mark.asyncio
async def test_malformed_read_fails_but_malformed_write_is_quarantined(system):
    s = system
    s.ppe.prepare.side_effect = KeyError("unexpected-contract")
    await tick(s)
    assert record(s)[0].status == "failed"
    assert record(s)[1].active_deployment_id is None
    s.ppe.prepare.side_effect = s.prepare
    retry = (await s.service.retry("task", s.row["deployment_id"], "retry"))["ppe_deployment"]
    s.row = retry
    await through(s, "image_submit")
    s.icm.create.side_effect = KeyError("unexpected-write-response")
    await tick(s)
    assert record(s)[0].status == "unknown"
    assert record(s)[1].active_deployment_id == retry["deployment_id"]


@pytest.mark.asyncio
async def test_deployment_rechecks_scm_sha_immediately_before_env_write(system, monkeypatch):
    from conciliator.integrations.ppe.scm import ScmClient, ScmVersion

    s = system
    await through(s, "deploy")
    monkeypatch.setattr(
        ScmClient,
        "version",
        AsyncMock(return_value=ScmVersion("1.0.0.1542", "999", "b" * 40, "build_ok", "test", ("x86_64",))),
    )
    await tick(s)
    assert record(s)[0].status == "failed"
    s.ppe.submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_older_deployed_rows_do_not_starve_active_update_scan(system):
    s = system
    await through(s, "entry_pending")
    new = (await s.service.create("task", "new", 1, SHA, s.service.revision))["ppe_deployment"]
    with s.sf() as db:
        old = db.get(PpeDeployment, s.row["deployment_id"])
        old.next_poll_at = _database_now(db) - timedelta(hours=1)
        db.commit()
    due = s.runner._due()
    assert new["deployment_id"] in due
    assert s.row["deployment_id"] not in due


def bits_system(s, monkeypatch):
    from conciliator.integrations.ppe.scm import ScmClient

    monkeypatch.setattr(ScmClient, "for_commit", AsyncMock(return_value=None))
    bits = SimpleNamespace(
        plan=AsyncMock(return_value={"pipeline_id": 1251261261058, "custom_vars": []}),
        submit=AsyncMock(return_value={"bits_run_id": "42", "bits_run_seq": "2"}),
        run=AsyncMock(return_value={
            "bits_run_status": 8, "bits_failed": False, "bits_success": True,
            "scm_version": "1.0.0.1542", "scm_version_id": "999",
        }),
    )
    s.runner.bits = bits
    return bits


@pytest.mark.asyncio
async def test_missing_scm_triggers_one_bits_run_then_resumes_after_restart(system, monkeypatch):
    s = system
    bits = bits_system(s, monkeypatch)
    await through(s, "bits_wait")
    assert bits.submit.await_count == 1
    assert record(s)[0].data["bits_run_id"] == "42"
    assert "intent" not in record(s)[0].data
    s.runner = PpeRunner(s.service, s.runner.http, s.icm, s.ppe, bits=bits)
    await through(s, "image_submit")
    assert record(s)[0].data["artifact_source"] == "bits_openapi"
    assert bits.run.call_args.args[1:3] == ("42", SHA)
    assert bits.submit.await_count == 1
    assert not s.writes


@pytest.mark.asyncio
async def test_offline_only_cn_artifact_starts_bits_instead_of_icm(system):
    s = system
    bits = SimpleNamespace(plan=AsyncMock(), submit=AsyncMock())
    s.runner.bits = bits
    await through(s, "artifact_wait")
    old_http = s.runner.http
    async with httpx.AsyncClient(
        base_url="https://test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
            "results": [{
                "version": "1.0.0.1542", "repos": 572600, "base_commit_hash": SHA,
                "arch": ["x86_64"], "id": 999, "status": "build_ok", "type": "offline",
            }], "next": None,
        })),
    ) as http:
        s.runner.http = http
        try:
            await tick(s)
        finally:
            s.runner.http = old_http
    assert record(s)[0].stage == "bits_plan"
    assert not record(s)[0].data.get("build_ok")
    s.icm.create.assert_not_awaited()
    assert not s.writes


@pytest.mark.asyncio
async def test_cached_offline_artifact_is_rejected_before_icm_write(system, monkeypatch):
    from conciliator.integrations.ppe.scm import ScmClient, ScmVersion

    s = system
    await through(s, "image_submit")
    monkeypatch.setattr(ScmClient, "version", AsyncMock(return_value=ScmVersion(
        "1.0.0.1542", "999", SHA, "build_ok", "offline", ("x86_64",),
    )))
    await tick(s)
    assert record(s)[0].status == "failed"
    assert "intent" not in record(s)[0].data
    s.icm.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_does_not_reuse_known_offline_artifact(system, monkeypatch):
    s = system
    await through(s, "image_submit")
    with s.sf() as db:
        previous = db.get(PpeDeployment, s.row["deployment_id"])
        previous.data = {**previous.data, "scm_build_type": "offline"}
        previous.status, previous.stage = "failed", "failed"
        env = db.get(PpeEnvironment, previous.environment_id)
        env.active_deployment_id = None
        db.commit()
    s.row = (await s.service.retry("task", s.row["deployment_id"], "retry-offline"))["ppe_deployment"]
    assert not record(s)[0].data.get("build_ok")
    bits_system(s, monkeypatch)
    await through(s, "bits_wait")
    s.icm.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_ambiguous_bits_write_cannot_be_replayed_or_retried(system, monkeypatch):
    s = system
    bits = bits_system(s, monkeypatch)
    bits.submit.side_effect = PlatformError("结果待确认", uncertain=True)
    await through(s, "unknown")
    for _ in range(3):
        await tick(s)
    assert bits.submit.await_count == 1
    assert record(s)[0].data["intent"]["kind"] == "bits"
    with pytest.raises(InvalidTaskTransition):
        await s.service.retry("task", s.row["deployment_id"], "retry")
    s.icm.create.assert_not_awaited()


@pytest.mark.parametrize("stage", ["ticket_wait", "release_wait"])
async def test_successful_poll_clears_transient_error_without_replaying_ticket(system, stage):
    s = system
    s.ppe.observe.return_value = {**s.ppe.observe.return_value, "versions_match": False}
    await through(s, "ticket_wait")
    if stage == "release_wait":
        with s.sf() as db:
            row = db.get(PpeDeployment, s.row["deployment_id"])
            row.stage = "release_wait"
            row.data = {**row.data, "release_ticket_id": "222"}
            db.commit()
    s.ppe.ticket.side_effect = PlatformError("平台请求失败（HTTP 400）")
    await tick(s)
    assert record(s)[0].error == "平台请求失败（HTTP 400）"
    assert record(s)[0].stage == stage
    writes = s.ppe.submit.await_count, s.ppe.remove.await_count
    s.ppe.ticket.side_effect = None
    s.ppe.ticket.return_value = {"status": "Running"}
    await tick(s)
    assert record(s)[0].stage == stage and not record(s)[0].error
    assert (s.ppe.submit.await_count, s.ppe.remove.await_count) == writes


async def test_ticket_read_success_does_not_clear_error_when_approval_read_still_fails(system):
    s = system
    s.ppe.observe.return_value = {**s.ppe.observe.return_value, "versions_match": False}
    await through(s, "ticket_wait")
    s.ppe.ticket.return_value = {"status": "Running"}
    s.ppe.approvals.side_effect = PlatformError("平台请求失败（HTTP 400）")
    await tick(s)
    assert record(s)[0].error == "平台请求失败（HTTP 400）"
    s.ppe.approvals.side_effect = None
    await tick(s)
    assert not record(s)[0].error


@pytest.mark.parametrize("failure", [
    PlatformError("平台请求失败（HTTP 400）；平台：tce;v1；接口：GET /open-apis/v1/pods/test/"),
    RuntimeError("private-platform-exception"),
])
async def test_ready_poll_recovers_without_stale_error_or_redeployment(system, failure):
    s = system
    s.runner.notifier = AsyncMock()
    await through(s, "entry_pending")
    deployment, env = record(s)
    observation = env.data["observed_at"]
    writes = s.ppe.submit.await_count, s.ppe.remove.await_count, s.icm.create.await_count
    s.ppe.observe.side_effect = failure
    await tick(s)
    failed_poll = (await s.service.snapshot("task"))["ppe_deployment"]
    assert failed_poll["status"] == "deployed"
    assert failed_poll["error"] and "private-platform-exception" not in failed_poll["error"]
    assert not failed_poll["can_verify"]
    with s.sf() as db, pytest.raises(InvalidTaskTransition, match="需要重新核验"):
        verify_ppe(db, "task", s.row["deployment_id"])
    assert record(s)[1].data["observed_at"] == observation
    s.ppe.observe.side_effect = None
    await tick(s)
    recovered = (await s.service.snapshot("task"))["ppe_deployment"]
    assert recovered["status"] == "deployed" and recovered["stage"] == "entry_pending"
    assert not recovered["error"] and recovered["can_verify"]
    assert record(s)[0].data["notification_version"] == deployment.data["notification_version"]
    assert s.runner.notifier.await_count == 1
    assert (s.ppe.submit.await_count, s.ppe.remove.await_count, s.icm.create.await_count) == writes


@pytest.mark.asyncio
async def test_bits_timeout_retry_resumes_original_run(system, monkeypatch):
    s = system
    bits = bits_system(s, monkeypatch)
    await through(s, "bits_wait")
    with s.sf() as db:
        row = db.get(PpeDeployment, s.row["deployment_id"])
        row.data = {**row.data, "stage_started_at": (_database_now(db) - timedelta(minutes=31)).isoformat()}
        db.commit()
    await tick(s)
    assert record(s)[0].status == "failed"
    s.row = (await s.service.retry("task", s.row["deployment_id"], "retry"))["ppe_deployment"]
    await through(s, "image_submit")
    assert bits.submit.await_count == 1
    assert bits.run.call_args.args[1] == "42"


@pytest.mark.asyncio
async def test_bits_success_does_not_allow_wrong_scm_sha_into_icm(system, monkeypatch):
    from conciliator.integrations.ppe.scm import ScmClient, ScmVersion

    s = system
    bits_system(s, monkeypatch)
    await through(s, "bits_wait")
    monkeypatch.setattr(ScmClient, "version", AsyncMock(return_value=ScmVersion(
        "1.0.0.1542", "999", "b" * 40, "build_ok", "test", ("x86_64",),
    )))
    await tick(s)
    assert record(s)[0].stage == "bits_wait"
    assert record(s)[0].error
    s.icm.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_observes_terminal_original_run_before_submitting_new_run(system, monkeypatch):
    s = system
    bits = bits_system(s, monkeypatch)
    await through(s, "bits_wait")
    with s.sf() as db:
        row = db.get(PpeDeployment, s.row["deployment_id"])
        row.data = {**row.data, "stage_started_at": (_database_now(db) - timedelta(minutes=31)).isoformat()}
        db.commit()
    await tick(s)
    s.row = (await s.service.retry("task", s.row["deployment_id"], "retry1"))["ppe_deployment"]
    bits.run.return_value = {"bits_run_status": 9, "bits_failed": True}
    await through(s, "failed")
    assert bits.submit.await_count == 1
    s.row = (await s.service.retry("task", s.row["deployment_id"], "retry2"))["ppe_deployment"]
    await through(s, "bits_wait")
    assert bits.submit.await_count == 2
