"""Versioned runtime profiles for the machine-level deployment protocol."""

import hashlib
import json
import os
import re
from pathlib import Path

from conciliator_deployment_worker.config import ALIASES, SERVICE_DEFAULTS, VQS_DEPENDENCIES

PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def runtime_profiles(config):
    services = config.get("services")
    if not isinstance(services, dict) or not services:
        raise ValueError("v2 requires services")
    result = {}
    for name, service in services.items():
        repository = ALIASES.get(name, name)
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+", repository) or any(
            part in {".", ".."} for part in repository.split("/")
        ):
            raise ValueError("invalid repository")
        if repository in result:
            raise ValueError("duplicate repository")
        configured = service.get("runtime_profiles")
        if configured is None:
            configured = {"default": service}
        if not isinstance(configured, dict) or not configured:
            raise ValueError("runtime_profiles must contain at least one profile")
        profiles = {}
        for profile_id, value in configured.items():
            if not PROFILE_ID.fullmatch(profile_id) or not isinstance(value, dict):
                raise ValueError("invalid runtime profile")
            defaults = SERVICE_DEFAULTS.get(repository, {})
            profile = {"extra_python_paths": [], **defaults, **value}
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", profile.get("service_identity", "data.aeolus.api_v3")):
                raise ValueError("invalid service_identity")
            if any(k in profile for k in ("port", "service_port", "maat_url")):
                raise ValueError("v2 runtime profiles cannot specify port or maat_url")
            path_fields = ["python_path"]
            if "runtime_manifest" in profile:
                path_fields.append("runtime_manifest")
            for key in path_fields:
                if not isinstance(profile.get(key), str) or not profile[key]:
                    raise ValueError(f"{key} is required for {profile_id}")
                profile[key] = os.path.abspath(Path(profile[key]).expanduser())
            if "files" in profile:
                if "runtime_manifest" in profile:
                    raise ValueError("use files or legacy runtime_manifest, not both")
                files = profile["files"]
                if not isinstance(files, dict) or any(
                    not isinstance(filename, str) or not os.path.isabs(filename)
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    for filename, digest in files.items()
                ):
                    raise ValueError("files must map absolute paths to SHA256 checksums")
            runtime_revision = profile.setdefault("runtime_revision", profile_id)
            if not isinstance(runtime_revision, str) or not PROFILE_ID.fullmatch(runtime_revision):
                raise ValueError("invalid runtime_revision")
            if not re.fullmatch(r"\d+\.\d+", profile.get("python_version", "")):
                raise ValueError("python_version must be major.minor")
            paths = profile.pop("extra_paths", profile["extra_python_paths"])
            if not isinstance(paths, list) or not all(isinstance(p, str) and p for p in paths):
                raise ValueError("extra_paths must be a list of paths")
            profile["extra_python_paths"] = [os.path.abspath(Path(p).expanduser()) for p in paths]
            branches = profile.get("base_branches", [])
            if not isinstance(branches, list) or not all(isinstance(b, str) and b for b in branches):
                raise ValueError("base_branches must be a list of branch names")
            profile["base_branches"] = branches
            release_max = profile.get("base_release_max")
            if "base_release_max" in profile and (
                repository != "dp/aeolus" or type(release_max) is not int or release_max < 0
            ):
                raise ValueError("base_release_max must be a nonnegative Aeolus release number")
            dependencies = profile.get("dependency_paths", {})
            if repository == "dp/VizQueryService" and "dependency_root" in profile:
                root = Path(profile.pop("dependency_root")).expanduser()
                dependencies = {dep: str(root / dep) for dep in VQS_DEPENDENCIES}
            if not isinstance(dependencies, dict) or any(
                not re.fullmatch(r"[A-Za-z0-9_-]+", key) or not isinstance(path, str) or not path
                for key, path in dependencies.items()
            ):
                raise ValueError("invalid dependency_paths")
            profile["dependency_paths"] = {
                key: os.path.abspath(Path(path).expanduser()) for key, path in dependencies.items()
            }
            revision = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
            profiles[profile_id] = {**profile, "profile_revision": revision}
        result[repository] = profiles
    return result


def capabilities(profiles):
    """Never upload local paths or arbitrary profile settings."""
    return {
        repository: {
            name: {
                key: profile[key]
                for key in (
                    "profile_revision",
                    "runtime_revision",
                    "python_version",
                    "base_branches",
                    "base_release_max",
                )
                if key in profile
            }
            for name, profile in items.items()
        }
        for repository, items in profiles.items()
    }
