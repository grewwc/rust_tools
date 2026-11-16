import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conciliator.domain.aeolus_task import AeolusRound, AeolusStartEvent, AeolusState
from conciliator.integrations.ppe.client import PpeClient, load_profile, profile_revision
from conciliator.integrations.ppe.environment import load_environment_profile, merge_owned_services
from conciliator.integrations.ppe.icm import equivalent_recipe, image_body
from conciliator.integrations.ppe.scm import PlatformError
from conciliator.repositories.models import Base
from conciliator.services.aeolus_registry import AeolusTaskRegistry, InvalidTaskTransition
from conciliator.services.ppe_deployment_service import PpeDeploymentService, task_lock
from conciliator.services.ppe_runner import PpeRunner
from tests.unit.test_ppe_lifecycle import record, through, tick

SHA = "a" * 40
NEXT_SHA = "b" * 40
REPOSITORIES = ["dp/aeolus", "dp/VizQueryService", "py27"]
EMBEDDED = [
    {"name": "data/dp/aeolus", "version": "1.0.11.8884", "path": "/opt/tiger/toutiao/app/data/aeolus"},
    {"name": "toutiao/load", "version": "1.0.2.619", "path": "/opt/tiger/toutiao/load"},
]


class Platform:
    """ENV and TCE reads reflect writes, including partial creation and retained resources."""

    def __init__(self, profile):
        self.profiles = [profile, profile["companion_profile"]]
        self.by_psm = {p["psm"]: p for p in self.profiles}
        self.env = None
        self.services = {}
        self.writes = []
        self.images = {}
        self.image_builds = []
        self.remaining = None
        self.healthy = True
        self.task_sha = SHA
        self.ignore_upgrades = False

    def response(self, request):
        path = request.url.path
        if path == "/api/v2/versions/":
            repo = int(request.url.params["repos_id"])
            return httpx.Response(200, json={"results": [{
                "repos": repo, "base_commit_hash": self.task_sha, "arch": ["x86_64"],
                "id": 9001 if self.task_sha == SHA else 9002,
                "version": "1.0.0.9001" if self.task_sha == SHA else "1.0.0.9002",
                "type": "test", "status": "build_ok",
            }], "next": None})
        if path.startswith("/api/repos/"):
            repo, version = int(path.split("/")[3]), path.split("/")[-1]
            profile = next(p for p in self.profiles if p["scm_repo_id"] == repo)
            baseline = profile["baseline_artifact"]
            artifact = (
                baseline if version == baseline["version"] else {
                    "version": version, "version_id": version.split(".")[-1],
                    "commit_sha": SHA if version.endswith("9001") else NEXT_SHA, "build_type": "test",
                }
            )
            return httpx.Response(200, json={
                "repos": repo, "version": version, "id": int(artifact["version_id"]),
                "base_commit_hash": artifact["commit_sha"], "type": artifact["build_type"],
                "arch": ["x86_64"], "status": "build_ok",
            })
        if path.startswith("/api/v4/env/"):
            return httpx.Response(200, json=self.env) if self.env else httpx.Response(404)
        if path == "/api/v4/instance_meta":
            psm = request.url.params["service"]
            profile = self.by_psm[psm]
            if request.url.params["env"] == profile["reference_environment"]:
                return httpx.Response(200, json={
                    "meta": {"psm": psm}, "instances": [
                        {"id": c["base_cluster_id"], "name": c["name"]} for c in profile["meta"]["clusters"]
                    ],
                })
            return httpx.Response(200, json=self.services[psm]) if psm in self.services else httpx.Response(404)
        if path == "/api/v4/suggest/cluster_by_id":
            identity = int(request.url.params["cluster_id"])
            for profile in self.profiles:
                for c in profile["meta"]["clusters"]:
                    if c["base_cluster_id"] == identity:
                        return httpx.Response(200, json={
                            "name": c["name"], "psm": profile["psm"],
                            "base_cluster": {"id": identity},
                            "new_env_vars": {"SOURCE_CONFIG": "memory-only", "AEOLUS_SERVICE_DISCOVERY": "1"},
                        })
            for psm, service in self.services.items():
                for c in service["instances"]:
                    if c["id"] == identity:
                        return httpx.Response(200, json={
                            "name": c["name"], "psm": psm, "base_cluster": {"id": identity},
                            "new_env_vars": c.get("new_env_vars", {}),
                        })
        if path == "/api/v4/dsl/pre_check/common":
            return httpx.Response(200, json={"pre_check_result": [{"name": "PreCheckQuota", "is_break": False}]})
        if path.startswith("/api/v4/dsl/env") and request.method in {"POST", "PUT"}:
            body = json.loads(request.content)
            self.writes.append(body)
            if self.env is None:
                self.env = {**body["env"], "id": 123}
            for psm, action in body["deployment"].items():
                profile = self.by_psm[psm]
                service = self.services.setdefault(psm, {
                    "meta": {"id": 500 + self.profiles.index(profile), "psm": psm}, "instances": [],
                })
                meta = action["meta"]
                for cluster in meta["clusters"]:
                    if action["action"] == "upgrade":
                        current = next(c for c in service["instances"] if c["id"] == cluster["id"])
                        if self.ignore_upgrades:
                            continue
                        current["image_version"] = meta["image_version"]
                    else:
                        current = {
                            **cluster, "id": 100 + sum(len(s["instances"]) for s in self.services.values()),
                        }
                        service["instances"].append(current)
                    current["repo_info"] = [
                        {"scm_repo_id": d["remote_id"], "path": d["path"], "version": d["version"]}
                        for d in meta["scm_repos"]
                    ]
                    if profile.get("production_baseline") and not meta["scm_repos"]:
                        # ENV exposes Atum's embedded manifest even though the DSL
                        # contains no replacement dependencies.
                        current["repo_info"] = [
                            {**d, "scm_repo_id": 4863 if d["name"] == "data/dp/aeolus" else 667}
                            for d in EMBEDDED
                        ]
            return httpx.Response(200, json={"id_str": str(len(self.writes) + 1000)})
        if path.startswith("/api/v4/dsl/env") and request.method == "DELETE":
            self.env = {**self.env, "deleted_at": 1793781794}
            return httpx.Response(200, json={"id_str": "2222"})
        if path.startswith("/api/v4/dsl/deployment/"):
            return httpx.Response(200, json={"status": "Succeed"})
        if path.startswith("/open-apis/v1/services/"):
            if self.env and self.env.get("deleted_at"):
                if self.remaining and f"/services/{self.remaining}/" in path:
                    return httpx.Response(200, json={"id": self.remaining})
                return httpx.Response(404)
            service_id = int(path.split("/")[4])
            service = next(s for s in self.services.values() if s["meta"]["id"] == service_id)
            cluster_id = int(path.rstrip("/").split("/")[-1])
            c = next(c for c in service["instances"] if c["id"] == cluster_id)
            return httpx.Response(200, json={
                "meta": {"id": cluster_id, "env": self.env["name"], "psm": service["meta"]["psm"], "status": "running"},
                "resource": {
                    "zone": c["zone"], "physical_cluster": "PPE", "logical_cluster": "ipv6",
                    "cpu": c["cpu"], "mem": c["mem"],
                },
                "runtime": {
                    "image_tag": "tce", "image_version": c["image_version"],
                    "replicas": {"dc_info": c["dc_infos"]},
                    "env_list": [{"env_name": k, "env_value": v} for k, v in c.get("new_env_vars", {}).items()],
                },
            })
        if path.endswith("/instances/"):
            return httpx.Response(200, json=[{"data": [{"podName": "pod-one", "podPhase": "Running"}]}])
        raise AssertionError((request.method, path))

    async def create_image(self, body, image_id):
        identity = str(len(self.images) + 1)
        self.images[identity] = copy.deepcopy(body)
        self.image_builds.append(body["psm"])
        return {"image_id": str(image_id), "image_version_id": identity}

    async def image_version(self, identity, body):
        assert equivalent_recipe(body, self.images[identity["image_version_id"]])
        return {
            **{k: identity[k] for k in ("image_id", "image_version_id")},
            "image_version": "1.0.2." + identity["image_version_id"], "image_status": "ok",
        }

    async def baseline_version(self, baseline):
        return {
            **{k: baseline[k] for k in ("image_id", "image_version_id", "image_version", "resolved_base_image")},
            "scm_version": baseline["version"], "scm_version_id": baseline["version_id"],
            "commit_sha": baseline["commit_sha"], "image_status": "ok", "artifact_source": "production_baseline",
            **({"embedded_scm_repos": copy.deepcopy(EMBEDDED)} if not baseline["scm_repos"] else {}),
        }


@pytest.fixture
async def paired(request, monkeypatch):
    target = getattr(request, "param", REPOSITORIES[0])
    repository = "dp/aeolus" if target == "py27" else target
    profile = load_environment_profile(repository=repository, python_version="2.7" if target == "py27" else None)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sf = sessionmaker(engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(sf, ppe_enabled=True)
    await registry.create_from_event(AeolusStartEvent("task", "space", 42, "node", "uuid", "user", ()))
    await registry.append_round("task", AeolusRound(1, "review", commit_sha=SHA))
    await registry.transition("task", {AeolusState.RECEIVED}, AeolusState.REVIEW)
    with sf() as db:
        task_lock(db, "task").evaluation = {"code_context": {
            "repository": repository, "base_branch": "release-aeolus-2.104.0" if target == "py27" else "master",
        }}
        db.commit()
    service = PpeDeploymentService(sf, profile)
    row = (await service.create("task", "request", 1, SHA, service.revision))["ppe_deployment"]
    platform = Platform(profile)
    http = httpx.AsyncClient(base_url="https://platform.test", transport=httpx.MockTransport(platform.response))
    icm = SimpleNamespace(
        create=platform.create_image, version=platform.image_version, baseline_version=platform.baseline_version,
    )
    monkeypatch.setattr(PpeClient, "pod_health", AsyncMock(side_effect=lambda *args: platform.healthy))
    runner = PpeRunner(service, http, icm, PpeClient(http))
    yield SimpleNamespace(
        sf=sf, registry=registry, service=service, row=row, runner=runner, platform=platform, profile=profile,
    )
    await http.aclose()
    engine.dispose()


@pytest.mark.parametrize("paired", REPOSITORIES, indirect=True)
async def test_both_repositories_create_update_and_release_complete_environment(paired):
    s, p = paired, paired.platform
    assert not s.row["can_view_environment"] and not s.row["can_access_environment"]
    await through(s, "ticket_wait")
    assert not (await s.service.snapshot("task"))["ppe_deployment"]["can_access_environment"]
    await through(s, "entry_pending")
    ready = (await s.service.snapshot("task"))["ppe_deployment"]
    assert ready["can_view_environment"] and ready["can_access_environment"]
    initial = p.writes[0]
    assert set(initial["deployment"]) == {"data.aeolus.api_v3", "data.vqs.api"}
    assert initial["env_action"] == "create_env"
    assert len(record(s)[1].data["clusters"]) == 3
    assert len(record(s)[1].data["owned_services"]) == 2
    vqs = p.services["data.vqs.api"]["instances"][0]
    assert vqs["new_env_vars"]["AEOLUS_SERVICE_DISCOVERY"] == "0"
    assert "memory-only" not in json.dumps(record(s)[0].data)
    assert "memory-only" not in json.dumps(record(s)[1].data)
    for profile in p.profiles:
        repos = p.services[profile["psm"]]["instances"][0]["repo_info"]
        if profile.get("production_baseline"):
            baseline = profile["production_baseline"]
            assert initial["deployment"][profile["psm"]]["meta"]["scm_replace"] is False
            assert initial["deployment"][profile["psm"]]["meta"]["app_path"] == baseline["app_path"]
            assert p.services[profile["psm"]]["instances"][0]["image_version"] == baseline["image_version"]
            assert len(repos) == len(baseline["scm_repos"] or EMBEDDED)
        else:
            main = next(d for d in repos if d["scm_repo_id"] == profile["scm_repo_id"])
            assert main["version"] == "1.0.0.9001"
    ids = [(c["psm"], c["id"]) for c in record(s)[1].data["clusters"]]
    environment_id = s.row["environment_id"]
    await s.registry.append_round("task", AeolusRound(2, "review", commit_sha=NEXT_SHA))
    p.task_sha = NEXT_SHA
    s.row = (await s.service.create("task", "second", 2, NEXT_SHA, s.service.revision))["ppe_deployment"]
    await through(s, "entry_pending")
    assert s.row["environment_id"] == environment_id
    assert [(c["psm"], c["id"]) for c in record(s)[1].data["clusters"]] == ids
    assert len(p.writes) == 2 and set(p.writes[-1]["deployment"]) == {s.profile["psm"]}
    assert p.writes[-1]["deployment"][s.profile["psm"]]["action"] == "upgrade"
    assert p.image_builds == [s.profile["psm"], s.profile["psm"]]
    assert vqs["new_env_vars"]["AEOLUS_SERVICE_DISCOVERY"] == "0"
    await s.service.release("task", s.row["deployment_id"], s.row["environment_id"], 1)
    p.remaining = p.services["data.vqs.api"]["meta"]["id"]
    await through(s, "release_verify")
    await tick(s)
    assert record(s)[0].stage == "release_verify"
    p.remaining = None
    await through(s, "released")
    assert record(s)[1].resource_state == "released"


async def test_partial_service_creation_is_repaired_before_becoming_ready(paired):
    s, p = paired, paired.platform
    await through(s, "ticket_wait")
    # Simulate ENV completing only one Aeolus cluster and no VQS service.
    p.services["data.aeolus.api_v3"]["instances"].pop()
    del p.services["data.vqs.api"]
    await through(s, "entry_pending")
    assert len(p.writes) == 2
    repairs = p.writes[1]["deployment"]
    assert repairs["data.aeolus.api_v3"]["action"] == "create_cluster"
    assert len(repairs["data.aeolus.api_v3"]["meta"]["clusters"]) == 1
    assert repairs["data.vqs.api"]["action"] == "create"
    assert repairs["data.vqs.api"]["meta"]["clusters"][0]["new_env_vars"]["AEOLUS_SERVICE_DISCOVERY"] == "0"
    assert record(s)[1].data["clusters_ready"]


async def test_one_unhealthy_service_prevents_complete_environment_verification(paired):
    s = paired
    s.platform.healthy = False
    await through(s, "observe")
    await tick(s)
    assert record(s)[0].stage == "observe" and not record(s)[1].data["clusters_ready"]
    s.platform.healthy = True
    await through(s, "entry_pending")


async def test_vqs_override_drift_is_detected_without_replaying_deployment(paired):
    s, p = paired, paired.platform
    await through(s, "entry_pending")
    p.services["data.vqs.api"]["instances"][0]["new_env_vars"]["AEOLUS_SERVICE_DISCOVERY"] = "1"
    await tick(s)
    assert record(s)[0].status == "deployed"
    assert not record(s)[1].data["clusters_ready"]
    assert not (await s.service.snapshot("task"))["ppe_deployment"]["can_verify"]
    await tick(s)
    assert not record(s)[1].data["clusters_ready"]
    p.services["data.vqs.api"]["instances"][0]["new_env_vars"]["AEOLUS_SERVICE_DISCOVERY"] = "0"
    await tick(s)
    assert (await s.service.snapshot("task"))["ppe_deployment"]["can_verify"]
    assert len(p.writes) == 1


def restore_legacy_companion(s):
    """Frozen schema 2 rows must resume their accepted companion build."""
    old = load_profile(repository=s.profile["companion_profile"]["repository"])
    s.profile["schema_version"] = 2
    s.profile["companion_profile"] = old
    with s.sf() as db:
        row = db.get(type(record(s)[0]), s.row["deployment_id"])
        row.data = {**row.data, "profile": copy.deepcopy(s.profile)}
        db.commit()
    s.platform.profiles = [s.profile, old]
    s.platform.by_psm = {p["psm"]: p for p in s.platform.profiles}


async def test_legacy_companion_write_with_lost_response_is_never_replayed(paired):
    s = paired
    restore_legacy_companion(s)
    await through(s, "companion_image_submit")

    async def lost(*args):
        raise PlatformError("配套镜像响应丢失", uncertain=True)

    s.runner.icm.create = AsyncMock(side_effect=lost)
    await tick(s)
    await tick(s)
    assert record(s)[0].stage == "unknown"
    assert record(s)[0].data["intent"]["kind"] == "companion_icm"
    assert s.runner.icm.create.await_count == 1
    with pytest.raises(InvalidTaskTransition):
        await s.service.retry("task", s.row["deployment_id"], "unsafe-retry")


async def test_legacy_restart_polls_original_companion_image_identity(paired):
    s = paired
    restore_legacy_companion(s)
    await through(s, "companion_image_wait")
    builds = len(s.platform.image_builds)
    # Recreate the runner without process-local state.
    s.runner = PpeRunner(s.service, s.runner.http, s.runner.icm, PpeClient(s.runner.http))
    await through(s, "entry_pending")
    assert len(s.platform.image_builds) == builds


@pytest.mark.parametrize("paired", REPOSITORIES, indirect=True)
async def test_restart_reuses_frozen_production_baseline_without_building_companion(paired):
    s = paired
    await through(s, "companion_image_verify")
    baseline = copy.deepcopy(s.profile["companion_profile"]["production_baseline"])
    s.runner = PpeRunner(s.service, s.runner.http, s.runner.icm, PpeClient(s.runner.http))
    await through(s, "entry_pending")
    artifact = record(s)[0].data["companion_artifact"]
    assert artifact["image_version_id"] == baseline["image_version_id"]
    assert artifact["artifact_source"] == "production_baseline"
    assert s.platform.image_builds == [s.profile["psm"]]


@pytest.mark.parametrize("paired", ["py27"], indirect=True)
async def test_py27_recovery_uses_frozen_profile_when_loaded_template_changes(paired):
    s = paired
    await through(s, "companion_image_verify")
    frozen = copy.deepcopy(record(s)[0].data["profile"])
    changed = copy.deepcopy(frozen)
    changed["meta"]["health_check_path"] = "changed configuration"
    service = PpeDeploymentService(s.sf, load_environment_profile(), profiles=(changed,))
    s.runner = PpeRunner(service, s.runner.http, s.runner.icm, PpeClient(s.runner.http))
    await through(s, "entry_pending")
    assert record(s)[0].data["profile"] == frozen
    assert record(s)[0].profile_revision == profile_revision(frozen)
    snapshot = (await service.snapshot("task"))["ppe_deployment"]
    assert snapshot["python_version"] == "2.7" and snapshot["can_verify"]
    with pytest.raises(InvalidTaskTransition, match="先释放旧环境"):
        await service.create("task", "changed-template", 1, SHA, profile_revision(changed))


async def test_unavailable_production_baseline_prevents_task_build_and_environment_write(paired):
    s = paired
    s.runner.icm.baseline_version = AsyncMock(side_effect=PlatformError("生产基准镜像已不可用"))
    await through(s, "failed")
    assert not s.platform.image_builds and not s.platform.writes


async def test_modified_companion_identity_is_rejected_before_environment_write(paired):
    s = paired
    await through(s, "deploy")
    with s.sf() as db:
        row = db.get(type(record(s)[0]), s.row["deployment_id"])
        row.data = {**row.data, "companion_artifact": {**row.data["companion_artifact"], "image_version_id": "123"}}
        db.commit()
    await through(s, "failed")
    assert not s.platform.writes


@pytest.mark.parametrize("paired", ["dp/VizQueryService"], indirect=True)
async def test_old_frozen_atum_artifact_recovers_manifest_without_another_write(paired):
    s = paired
    await through(s, "ticket_wait")
    with s.sf() as db:
        row = db.get(type(record(s)[0]), s.row["deployment_id"])
        artifact = {k: v for k, v in row.data["companion_artifact"].items() if k != "embedded_scm_repos"}
        row.data = {**row.data, "companion_artifact": artifact}
        db.commit()
    revision = record(s)[0].profile_revision
    s.runner = PpeRunner(s.service, s.runner.http, s.runner.icm, PpeClient(s.runner.http))
    await through(s, "entry_pending")
    assert len(s.platform.writes) == 1
    assert record(s)[0].profile_revision == revision
    assert record(s)[0].data["companion_artifact"]["embedded_scm_repos"] == EMBEDDED
    assert all(c["health_ok"] for c in record(s)[1].data["clusters"])


@pytest.mark.parametrize("paired", ["dp/VizQueryService"], indirect=True)
@pytest.mark.parametrize("mutation", ["name", "version", "path", "duplicate"])
async def test_atum_embedded_dependency_drift_is_not_accepted(paired, mutation):
    s = paired
    await through(s, "entry_pending")
    repos = s.platform.services["data.aeolus.api_v3"]["instances"][0]["repo_info"]
    original = copy.deepcopy(repos)
    if mutation == "duplicate":
        repos.append(copy.deepcopy(repos[0]))
    else:
        repos[0][mutation] = "other"
    await tick(s)
    assert record(s)[0].stage == "entry_pending"
    assert not record(s)[1].data["clusters_ready"]
    assert not (await s.service.snapshot("task"))["ppe_deployment"]["can_verify"]
    repos[:] = original
    await tick(s)
    assert (await s.service.snapshot("task"))["ppe_deployment"]["can_verify"]
    assert len(s.platform.writes) == 1


async def test_finished_upgrade_with_unchanged_versions_is_not_repeated(paired):
    s = paired
    await through(s, "entry_pending")
    await s.registry.append_round("task", AeolusRound(2, "review", commit_sha=NEXT_SHA))
    s.platform.task_sha = NEXT_SHA
    s.platform.ignore_upgrades = True
    s.row = (await s.service.create("task", "second", 2, NEXT_SHA, s.service.revision))["ppe_deployment"]
    await through(s, "failed")
    assert len(s.platform.writes) == 2
    assert record(s)[0].data["upgraded_services"] == [s.profile["psm"]]
    assert "不会重复提交" in record(s)[0].error


def test_production_companion_does_not_change_task_build_repository():
    profile = load_environment_profile(repository="dp/VizQueryService")
    companion = profile["companion_profile"]
    assert companion["scm_repo_id"] == 4863
    assert companion["meta"]["scm_repos"] == []
    body = image_body(load_environment_profile(), "1.0.0.9999", "huaguojin")
    assert next(d for d in body["repos"] if d["remote_id"] == "572600")["version"] == "1.0.0.9999"


def test_owned_service_history_retains_partial_and_deleted_clusters():
    old = [{"psm": "data.vqs.api", "service_id": "1", "clusters": [{"id": "2", "name": "default"}]}]
    current = [{"psm": "data.vqs.api", "service_id": "1", "clusters": []}]
    assert merge_owned_services(old, current) == old
    new = [{"psm": "data.vqs.api", "service_id": "3", "clusters": [{"id": "4", "name": "default"}]}]
    assert len(merge_owned_services(old, new)) == 2
    profile = load_environment_profile()
    changed = copy.deepcopy(profile)
    changed["companion_profile"]["baseline_artifact"]["version_id"] = "other"
    assert profile_revision(changed) != profile_revision(profile)
