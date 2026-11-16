import copy
import hashlib
import json
import sys

import pytest

from conciliator_deployment_worker.multi_worker import verify_runtime
from conciliator_deployment_worker.profiles import capabilities, runtime_profiles


def configuration(tmp_path, *, legacy_manifest=False):
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    one = {
        "python_path": sys.executable,
        "python_version": version,
        "runtime_revision": "one-r1",
        "base_branches": ["master"],
    }
    if legacy_manifest:
        manifest = tmp_path / "runtime.json"
        manifest.write_text(json.dumps({"runtime_revision": "one-r1", "python_version": version, "files": {}}))
        one["runtime_manifest"] = str(manifest)
    return {"services": {"aeolus": {"runtime_profiles": {"one": one, "two": {**one, "base_branches": ["next"]}}}}}


def test_multiple_profiles_are_distinct_without_fixed_ports_and_do_not_upload_paths(tmp_path):
    value = configuration(tmp_path)
    original = copy.deepcopy(value)
    profiles = runtime_profiles(value)
    assert value == original
    assert profiles["dp/aeolus"]["one"]["profile_revision"] != profiles["dp/aeolus"]["two"]["profile_revision"]
    advertised = capabilities(profiles)
    serialized = json.dumps(advertised)
    assert "python_path" not in serialized and str(tmp_path) not in serialized
    assert "service_port" not in json.dumps(profiles)


def test_omitted_runtime_revision_uses_profile_name_and_does_not_change_input(tmp_path):
    value = configuration(tmp_path)
    for profile in value["services"]["aeolus"]["runtime_profiles"].values():
        profile.pop("runtime_revision")
    original = copy.deepcopy(value)
    profiles = runtime_profiles(value)
    assert value == original
    for name, profile in profiles["dp/aeolus"].items():
        assert profile["runtime_revision"] == name
        assert capabilities(profiles)["dp/aeolus"][name]["runtime_revision"] == name
        assert "files" not in profile and "runtime_manifest" not in profile


@pytest.mark.parametrize("revision", [None, "", "invalid revision"])
def test_invalid_explicit_runtime_revision_is_not_replaced_by_default(tmp_path, revision):
    value = configuration(tmp_path)
    value["services"]["aeolus"]["runtime_profiles"]["one"]["runtime_revision"] = revision
    with pytest.raises(ValueError, match="runtime_revision"):
        runtime_profiles(value)


@pytest.mark.asyncio
async def test_runtime_manifest_and_actual_python_must_match(tmp_path):
    profile = runtime_profiles(configuration(tmp_path, legacy_manifest=True))["dp/aeolus"]["one"]
    assert "files" not in profile
    await verify_runtime(profile)
    with pytest.raises(RuntimeError, match="manifest differs"):
        await verify_runtime({**profile, "runtime_revision": "changed"})
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"runtime_revision": "one-r1", "python_version": "0.1", "files": {}}))
    with pytest.raises(RuntimeError, match="actual Python version"):
        await verify_runtime({**profile, "python_version": "0.1"})


@pytest.mark.asyncio
@pytest.mark.parametrize("files", [None, {}])
async def test_single_file_configuration_checks_python_without_manifest(tmp_path, files):
    value = configuration(tmp_path)
    if files is not None:
        value["services"]["aeolus"]["runtime_profiles"]["one"]["files"] = files
    profile = runtime_profiles(value)["dp/aeolus"]["one"]
    assert "runtime_manifest" not in profile
    assert not (tmp_path / "runtime.json").exists()
    await verify_runtime(profile)
    with pytest.raises(RuntimeError, match="actual Python version"):
        await verify_runtime({**profile, "python_version": "0.1"})


@pytest.mark.asyncio
async def test_inline_checksums_detect_modified_artifacts_and_are_not_uploaded(tmp_path):
    artifact = tmp_path / "parser.py"
    artifact.write_text("prepared parser")
    value = configuration(tmp_path)
    original = runtime_profiles(value)["dp/aeolus"]["one"]["profile_revision"]
    value["services"]["aeolus"]["runtime_profiles"]["one"]["files"] = {
        str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    profiles = runtime_profiles(value)
    profile = profiles["dp/aeolus"]["one"]
    assert profile["profile_revision"] != original
    advertised = json.dumps(capabilities(profiles))
    assert "files" not in advertised and str(artifact) not in advertised
    await verify_runtime(profile)
    artifact.write_text("unexpected change")
    with pytest.raises(RuntimeError, match="checksum differs"):
        await verify_runtime(profile)


@pytest.mark.parametrize("files", [
    None, [], "invalid", {"relative": "a" * 64}, {"/artifact": "bad"}, {"/artifact": 3}, {1: "a" * 64},
])
def test_invalid_inline_checksums_are_rejected(tmp_path, files):
    value = configuration(tmp_path)
    value["services"]["aeolus"]["runtime_profiles"]["one"]["files"] = files
    with pytest.raises(ValueError, match="absolute paths to SHA256"):
        runtime_profiles(value)


def test_legacy_manifest_cannot_be_combined_with_inline_checksums(tmp_path):
    value = configuration(tmp_path, legacy_manifest=True)
    value["services"]["aeolus"]["runtime_profiles"]["one"]["files"] = {}
    with pytest.raises(ValueError, match="not both"):
        runtime_profiles(value)


def test_duplicate_repository_and_fixed_ports_are_rejected(tmp_path):
    value = configuration(tmp_path)
    value["services"]["dp/aeolus"] = value["services"]["aeolus"]
    with pytest.raises(ValueError, match="duplicate repository"):
        runtime_profiles(value)
    value = configuration(tmp_path)
    value["services"]["aeolus"]["runtime_profiles"]["one"]["port"] = 12356
    with pytest.raises(ValueError, match="cannot specify port"):
        runtime_profiles(value)


def test_aeolus_release_limit_is_advertised_and_changes_profile_revision(tmp_path):
    value = configuration(tmp_path)
    original = runtime_profiles(value)["dp/aeolus"]["one"]["profile_revision"]
    value["services"]["aeolus"]["runtime_profiles"]["one"]["base_release_max"] = 104
    profiles = runtime_profiles(value)
    assert profiles["dp/aeolus"]["one"]["profile_revision"] != original
    assert capabilities(profiles)["dp/aeolus"]["one"]["base_release_max"] == 104


@pytest.mark.parametrize("limit", [None, -1, True, "104", 104.0])
def test_invalid_release_limit_is_rejected(tmp_path, limit):
    value = configuration(tmp_path)
    value["services"]["aeolus"]["runtime_profiles"]["one"]["base_release_max"] = limit
    with pytest.raises(ValueError, match="base_release_max"):
        runtime_profiles(value)
