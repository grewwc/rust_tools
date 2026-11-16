import copy
import json
from types import SimpleNamespace

import pytest

from conciliator.integrations.ppe.client import deployment_body, load_profile, profile_revision
from conciliator.integrations.ppe.environment import load_environment_profile
from conciliator.integrations.ppe.icm import image_body
from conciliator.integrations.ppe.scm import PlatformError
from conciliator.services.ppe_deployment_service import PpeDeploymentService
from conciliator.services.runtime_selection import detect


@pytest.mark.parametrize("branch", [
    "release-aeolus-2.104.0", "release-aeolus-2.99.12", "release-aeolus-2.105.0", "master", "feature/py27-fix",
])
def test_code_dependencies_select_runtime_independently_of_base_branch(branch):
    service = PpeDeploymentService(None, load_environment_profile(), profiles=(
        load_environment_profile(python_version="2.7"),
        load_environment_profile(repository="dp/VizQueryService"),
    ))
    task = SimpleNamespace(evaluation={
        "code_context": {"repository": "dp/aeolus", "base_branch": branch, "target_branch": "master"},
        "_runtime_detection": detect("dp/aeolus", "a" * 40, {
            "requirements.txt": "antlr4-python2-runtime==4.7.2",
        }),
    })
    choices = service._choices(task, SimpleNamespace(commit_sha="a" * 40))
    assert choices["recommended_runtime_profile_id"] == "py27"
    assert not choices["runtime_profiles"]["py3"]["compatible"]
    profile = service._profiles(task)["py27"]
    assert profile["scm_repo_id"] == 4863
    assert profile["companion_profile"]["production_baseline"]["psm"] == "data.vqs.api"


def test_legacy_dependencies_cannot_silently_use_python3_when_py27_profile_is_missing():
    service = PpeDeploymentService(None, load_profile())
    task = SimpleNamespace(evaluation={
        "code_context": {"repository": "dp/aeolus", "base_branch": "release-aeolus-2.104.0"},
        "_runtime_detection": detect("dp/aeolus", "a" * 40, {
            "requirements.txt": "antlr4-python2-runtime==4.7.2",
        }),
    })
    choices = service._choices(task, SimpleNamespace(commit_sha="a" * 40))
    assert choices["recommended_runtime_profile_id"] is None
    assert not choices["runtime_profiles"]["py3"]["compatible"]


def test_py27_build_and_deployment_replace_only_main_artifact():
    profile = load_profile(python_version="2.7")
    original = copy.deepcopy(profile)
    image = image_body(profile, "1.0.11.9999", "huaguojin")
    assert len(image["repos"]) == 26
    assert image["base_image"] == "debian.jessie.jdk8_yarn:latest"
    assert image["load_path"].endswith("/run/prod_run.sh")
    assert {p["package_type"] for p in image["packages"]} == {"apt-get", "pip"}
    assert "antlr4-python2-runtime==4.7.2" in image["packages"][1]["name"]
    body = deployment_body(profile, "a" * 32, "1.0.11.9999", image_version="1.0.2.9999")
    meta = body["deployment"][profile["psm"]]["meta"]
    assert meta["app_path"] == image["load_path"]
    assert "exit 0" not in meta["health_check_path"]
    assert "Hello Aeolus!" in meta["health_check_path"]
    assert [c["base_cluster_id"] for c in meta["clusters"]] == [219217817, 219217814]
    for original_repo, image_repo, deployment_repo in zip(
        profile["meta"]["scm_repos"], image["repos"], meta["scm_repos"], strict=True,
    ):
        expected = "1.0.11.9999" if original_repo["main_repo"] else original_repo["version"]
        assert image_repo["version"] == deployment_repo["version"] == expected
    assert profile == original
    assert profile_revision(profile) != profile_revision(load_profile())


@pytest.mark.parametrize("mutation", ["parser", "mysql", "pipeline", "pip", "startup"])
def test_py27_rejects_mixed_runtime_recipe(tmp_path, mutation):
    profile = load_profile(python_version="2.7")
    if mutation == "parser":
        profile["meta"]["scm_repos"][-1]["remote_id"] = 520839
    elif mutation == "mysql":
        profile["meta"]["scm_repos"][13]["path"] = "lib/python3.10/MySQLdb"
    elif mutation == "pipeline":
        profile["bits_scm"]["pipeline_id"] = "1251261261058"
    elif mutation == "pip":
        profile["image_recipe"]["packages"][1]["package_type"] = "pip3"
    else:
        profile["meta"]["app_path"] = "toutiao/app/data/aeolus/dlv_control/bootstrap.sh"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(PlatformError):
        load_profile(path, python_version="2.7")
