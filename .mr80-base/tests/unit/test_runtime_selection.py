from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from conciliator.domain.aeolus_task import AeolusRound, AeolusStartEvent, AeolusState
from conciliator.integrations.ppe.client import load_profile, profile_revision
from conciliator.repositories.models import Base, Deployment, MeegleAINodeTask
from conciliator.repositories.ppe_models import PpeDeployment, PpeEnvironment
from conciliator.services.aeolus_registry import AeolusTaskRegistry, InvalidTaskTransition
from conciliator.services.deployment_service import DeploymentService
from conciliator.services.ppe_deployment_service import PpeDeploymentService
from conciliator.services.runtime_selection import (
    FILES,
    RuntimeDetectionService,
    detect,
    runtime_choices,
    select_runtime,
    task_detection,
)

SHA = "a" * 40
PROFILES = {"py27": {"python_version": "2.7"}, "py310": {"python_version": "3.10"},
            "py311": {"python_version": "3.11"}}


@pytest.mark.parametrize(("files", "recommended", "compatible"), [
    ({".python-version": "3.10.14"}, "py310", ["py310"]),
    ({"pyproject.toml": '[project]\nrequires-python = ">=3.11,<3.12"'}, "py311", ["py311"]),
    ({"setup.cfg": "[options]\npython_requires = >=3.10.2,<3.11"}, "py310", ["py310"]),
    ({"setup.py": 'setup(python_requires=">=2.7,<3")'}, "py27", ["py27"]),
    ({"requirements.txt": "antlr4-python2-runtime==4.7.2"}, "py27", ["py27"]),
    ({"requirements.txt": "antlr4-python3-runtime==4.9.3"}, None, ["py310", "py311"]),
    ({"requirements.txt": "antlr4-python2-runtime==4.7.2\nantlr4-python3-runtime==4.9.3"}, None, []),
    ({".python-version": "3.11", "requirements.txt": "antlr4-python2-runtime==4.7.2"}, None, []),
    ({"requirements.txt": "# antlr4-python2-runtime\nsix==1.16\nFlask==0.12"}, None, list(PROFILES)),
    ({"requirements.txt": 'antlr4-python2-runtime; python_version < "3"'}, None, list(PROFILES)),
    ({"setup.py": 'setup(python_requires=get_version())'}, None, list(PROFILES)),
    ({"run/prod_run.sh": "python -m bytedunicorn aeolus.app:app"}, None, list(PROFILES)),
])
def test_static_evidence_and_conflicts(files, recommended, compatible):
    result = runtime_choices(PROFILES, detect("dp/aeolus", SHA, files), "release-aeolus-2.104.0")
    assert result["recommended_runtime_profile_id"] == recommended
    assert [name for name, p in result["runtime_profiles"].items() if p["compatible"]] == compatible


def test_manual_selection_can_override_hints_but_not_explicit_constraints():
    unknown = detect("dp/aeolus", SHA, {})
    assert select_runtime(PROFILES, "master", "py27", unknown) == "py27"
    with pytest.raises(ValueError, match="请选择运行配置"):
        select_runtime({"py27": PROFILES["py27"]}, "master", detection=unknown)
    strong = detect("dp/aeolus", SHA, {".python-version": "3.11"})
    with pytest.raises(ValueError, match="不兼容"):
        select_runtime(PROFILES, "release-aeolus-2.104.0", "py27", strong)


def test_python3_image_without_minor_version_does_not_claim_minor_compatibility():
    profiles = {"py3": {"python_version": "3"}}
    strong = detect("dp/VizQueryService", SHA, {".python-version": "3.11"})
    choices = runtime_choices(profiles, strong)
    assert choices["recommended_runtime_profile_id"] is None
    assert choices["runtime_profiles"]["py3"]["compatible"]
    assert not choices["runtime_profiles"]["py3"]["compatibility_known"]
    assert select_runtime(profiles, selected="py3", detection=strong) == "py3"
    major = detect("dp/VizQueryService", SHA, {"requirements.txt": "antlr4-python3-runtime==4.7.2"})
    assert runtime_choices(profiles, major)["recommended_runtime_profile_id"] == "py3"


@pytest.mark.parametrize(("spec", "compatible"), [
    ("==3.10.8", True),
    (">=3.10.2,<=3.10.7", True),
    (">3.10.2,<3.10.4", True),
    ("==3.10.8,!=3.10.8", False),
    (">3.10.2,<3.10.3", False),
    ("==2.7.18", False),
])
def test_major_only_profile_checks_patch_bounds_without_claiming_a_minor(spec, compatible):
    detection = detect("dp/VizQueryService", SHA, {
        "pyproject.toml": f'[project]\nrequires-python = "{spec}"',
    })
    choices = runtime_choices({"py3": {"python_version": "3"}}, detection)
    assert choices["runtime_profiles"]["py3"]["compatible"] is compatible
    assert choices["recommended_runtime_profile_id"] is None


def test_detection_cannot_carry_over_to_another_sha_or_repository():
    stored = detect("dp/aeolus", SHA, {".python-version": "2.7"})
    task = SimpleNamespace(evaluation={"_runtime_detection": stored})
    assert task_detection(task, SimpleNamespace(commit_sha=SHA)) == stored
    assert not task_detection(task, SimpleNamespace(commit_sha="b" * 40))["constraints"]
    task.evaluation["code_context"] = {"repository": "dp/VizQueryService"}
    assert not task_detection(task, SimpleNamespace(commit_sha=SHA))["constraints"]


@pytest.fixture
async def system():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sf = sessionmaker(engine, expire_on_commit=False)
    registry = AeolusTaskRegistry(sf, ppe_enabled=True, deployments_enabled=True)
    await registry.create_from_event(AeolusStartEvent("task", "space", 42, "node", "uuid", "user", ()))
    await registry.append_round("task", AeolusRound(1, "review", commit_sha=SHA))
    await registry.update("task", repo_id="repo", work_branch="work/one",
                          evaluation={"code_context": {"repository": "dp/aeolus", "base_branch": "master"}})
    await registry.transition("task", {AeolusState.RECEIVED}, AeolusState.REVIEW)
    yield sf, registry
    engine.dispose()


async def test_detector_reads_exact_sha_caches_and_retries_safe_failures(system, monkeypatch):
    sf, registry = system
    reader = AsyncMock(side_effect=lambda _client, _repo, **kw: SimpleNamespace(content=(
        "antlr4-python2-runtime==4.7.2" if kw["path"] == "requirements.txt" else ""
    )))
    monkeypatch.setattr("conciliator.services.runtime_selection.get_file", reader)
    detector = RuntimeDetectionService(sf, object())
    await detector.ensure("task")
    await detector.ensure("task")
    assert reader.call_count == len(FILES)
    assert all(call.kwargs["revision"] == SHA for call in reader.call_args_list)
    await registry.append_round("task", AeolusRound(2, "review", commit_sha="b" * 40))
    reader.side_effect = RuntimeError("Authorization: private-token")
    await detector.ensure("task")
    await detector.ensure("task")
    assert reader.call_count == 2 * len(FILES)
    with sf() as db:
        task = db.query(MeegleAINodeTask).filter_by(task_id="task").one()
        result = task.evaluation["_runtime_detection"]
        assert result["status"] == "unavailable" and "private-token" not in str(result)
        task.evaluation = {**task.evaluation, "_runtime_detection": {**result, "retry_after": 0}}
        db.commit()
    reader.side_effect = lambda _client, _repo, **kw: SimpleNamespace(content=(
        "3.11" if kw["path"] == ".python-version" else ""
    ))
    await detector.ensure("task")
    assert reader.call_count == 3 * len(FILES)


async def test_late_detection_does_not_overwrite_new_round(system, monkeypatch):
    sf, registry = system
    changed = False

    async def read(_client, _repo, **kw):
        nonlocal changed
        if not changed:
            changed = True
            await registry.append_round("task", AeolusRound(2, "review", commit_sha="b" * 40))
        return SimpleNamespace(content="")

    monkeypatch.setattr("conciliator.services.runtime_selection.get_file", read)
    await RuntimeDetectionService(sf, object()).ensure("task")
    with sf() as db:
        assert "_runtime_detection" not in db.query(MeegleAINodeTask).filter_by(task_id="task").one().evaluation


async def test_both_targets_share_evidence_and_manual_choice_is_frozen(system):
    sf, registry = system
    await registry.update("task", evaluation={
        "code_context": {"repository": "dp/aeolus", "base_branch": "master"},
        "_runtime_detection": detect("dp/aeolus", SHA, {"requirements.txt": "antlr4-python2-runtime==4.7.2"}),
    })
    py3, py27 = load_profile(), load_profile(python_version="2.7")
    ppe = PpeDeploymentService(sf, py3, profiles=(py27,))
    devbox = DeploymentService(sf)
    body = {"protocol_version": 2, "worker_id": "machine", "advertised_host": "devbox.internal",
            "runtime_profiles": {"dp/aeolus": {
                name: {**p, "profile_revision": "b" * 64, "runtime_revision": name, "base_branches": []}
                for name, p in PROFILES.items()
            }}}
    response = await devbox.run("register", body)
    await devbox.run("heartbeat", "machine", {"session_id": response["session_id"], "reconciled": True})
    capability = (await ppe.snapshot("task"))["ppe_capability"]
    worker = (await devbox.run("snapshot", "task"))["deployment_workers"][0]
    assert capability["recommended_runtime_profile_id"] == worker["recommended_runtime_profile_id"] == "py27"
    assert capability["runtime_detection"]["evidence"] == worker["runtime_detection"]["evidence"]
    with pytest.raises(ValueError, match="不兼容"):
        await ppe.create("task", "wrong", 1, SHA, profile_revision(py3), "py3")
    with pytest.raises(ValueError, match="不兼容"):
        await devbox.run("enqueue", "task", "machine", "wrong", 1, SHA, "py311")
    row = (await ppe.create("task", "ppe", 1, SHA, profile_revision(py27), "py27"))["ppe_deployment"]
    local = await devbox.run("enqueue", "task", "machine", "local", 1, SHA, "py27")
    with pytest.raises(InvalidTaskTransition, match="request_id"):
        await ppe.create("task", "ppe", 1, SHA, profile_revision(py3), "py3")
    with sf() as db:
        p = db.get(PpeDeployment, row["deployment_id"])
        assert p.data["runtime_profile_id"] == "py27"
        assert db.get(Deployment, local["deployment_id"]).data["runtime_detection"]["commit_sha"] == SHA
        p.status, p.stage = "failed", "failed"
        db.get(PpeEnvironment, row["environment_id"]).active_deployment_id = None
        db.commit()
    # Changing server templates and evidence does not alter an accepted retry.
    ppe.profiles = {("dp/aeolus", ""): py3}
    await registry.update("task", evaluation={"code_context": {"repository": "dp/aeolus"}})
    retried = (await ppe.retry("task", row["deployment_id"], "retry"))["ppe_deployment"]
    assert retried["profile_revision"] == row["profile_revision"]
    assert retried["python_version"] == "2.7"


async def test_unknown_ppe_requires_manual_profile_and_runtime_change_requires_release(system):
    sf, _ = system
    py3, py27 = load_profile(), load_profile(python_version="2.7")
    ppe = PpeDeploymentService(sf, py3, profiles=(py27,))
    capability = (await ppe.snapshot("task"))["ppe_capability"]
    assert capability["enabled"] and capability["recommended_runtime_profile_id"] is None
    row = (await ppe.create("task", "manual", 1, SHA, profile_revision(py27), "py27"))["ppe_deployment"]
    with sf() as db:
        db.get(PpeEnvironment, row["environment_id"]).active_deployment_id = None
        db.commit()
    with pytest.raises(InvalidTaskTransition, match="先释放旧环境"):
        await ppe.create("task", "switch", 1, SHA, profile_revision(py3), "py3")
