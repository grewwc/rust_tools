"""SHA-bound Python evidence shared by PPE and Devbox; never execute repository code."""

import ast
import asyncio
import configparser
import re
import time
import tomllib

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from conciliator.integrations.nextcode.files import get_file
from conciliator.repositories.models import MeegleAINodeRound, MeegleAINodeTask

RULE_VERSION = 1
FILES = (".python-version", "pyproject.toml", "setup.cfg", "setup.py", "requirements.txt", "run/prod_run.sh")


def detect(repository, sha, files):
    evidence, constraints, warnings = [], [], []

    def declaration(path, spec):
        try:
            parsed = str(SpecifierSet(spec))
        except InvalidSpecifier:
            warnings.append(f"{path} 的 Python 声明无法解析")
            return
        if parsed:
            constraints.append(parsed)
            evidence.append({"path": path, "kind": "declaration", "detail": f"Python {parsed}"})

    pin = files.get(".python-version", "").strip()
    if pin:
        # Profiles identify major.minor, not patch versions.
        match = re.fullmatch(r"(\d+\.\d+)(?:\.\d+)?", pin)
        if match:
            declaration(".python-version", f"=={match[1]}.*")
        else:
            warnings.append(".python-version 未声明单一 Python 版本")
    try:
        project = tomllib.loads(files.get("pyproject.toml", "")).get("project", {})
        if "requires-python" in project:
            declaration("pyproject.toml", project["requires-python"])
    except (ValueError, TypeError):
        warnings.append("pyproject.toml 无法解析")
    try:
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(files.get("setup.cfg", ""))
        if config.has_option("options", "python_requires"):
            declaration("setup.cfg", config.get("options", "python_requires"))
    except configparser.Error:
        warnings.append("setup.cfg 无法解析")
    try:
        for node in ast.walk(ast.parse(files.get("setup.py", ""))):
            if isinstance(node, ast.keyword) and node.arg == "python_requires":
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    declaration("setup.py", node.value.value)
                else:
                    warnings.append("setup.py 的 Python 声明不是静态值")
    except SyntaxError:
        # Python 2 setup scripts need not parse under the server's Python 3.
        pass
    for line in files.get("requirements.txt", "").splitlines():
        requirement = line.split("#", 1)[0].strip()
        # Conditional requirements cannot establish a universal constraint.
        match = re.match(r"^antlr4-python([23])-runtime(?:\s|[=<>!~]|$)", requirement, re.I)
        if match and ";" not in requirement:
            major = int(match[1])
            constraints.append(f">={major},<{major + 1}")
            evidence.append({"path": "requirements.txt", "kind": "dependency",
                             "detail": f"antlr4-python{major}-runtime 要求 Python {major}"})
    startup = files.get("run/prod_run.sh", "")
    explicit = sorted(set(re.findall(r"(?<![\w])python([23](?:\.\d+)?)\s+-m\s+bytedunicorn\b", startup)))
    for version in explicit:
        evidence.append({"path": "run/prod_run.sh", "kind": "hint",
                         "detail": f"启动脚本使用 python{version}"})
    return {"rule_version": RULE_VERSION, "repository": repository, "commit_sha": sha,
            "constraints": constraints, "evidence": evidence, "warnings": warnings,
            "status": "identified" if constraints else "unknown"}


def task_detection(task, round_):
    context = (task.evaluation or {}).get("code_context") or {}
    repository = context.get("repository", "dp/aeolus")
    sha = round_.commit_sha if round_ else ""
    stored = (task.evaluation or {}).get("_runtime_detection") or {}
    if (stored.get("repository"), stored.get("commit_sha"), stored.get("rule_version")) == (
        repository, sha, RULE_VERSION,
    ):
        return stored
    return detect(repository, sha, {})


def runtime_choices(profiles, detection, base_branch=""):
    constraints = [SpecifierSet(spec) for spec in detection["constraints"]]
    candidates = {}
    for name, profile in profiles.items():
        version = profile["python_version"]
        major_only = "." not in version
        minors = {0, 999999}
        if major_only:
            for spec in constraints:
                for bound in spec:
                    match = re.match(re.escape(version) + r"\.(\d+)", bound.version)
                    if match:
                        minor = int(match[1])
                        minors.update({max(0, minor - 1), minor, minor + 1})
        versions = [f"{version}.{minor}" for minor in minors] if major_only else [version]
        # A major.minor capability does not claim a particular patch release.
        patches = {0, 999999}
        patch_prefix = re.escape(version) + (r"\.\d+" if major_only else "")
        for spec in constraints:
            for bound in spec:
                match = re.fullmatch(patch_prefix + r"\.(\d+)", bound.version)
                if match:
                    patch = int(match[1])
                    patches.update({max(0, patch - 1), patch, patch + 1})
        is_compatible = any(
            all(Version(f"{candidate}.{patch}") in spec for spec in constraints)
            for candidate in versions for patch in patches
        )
        known = not major_only or not any(
            "." in bound.version.rstrip(".*") for spec in constraints for bound in spec
        )
        candidates[name] = {**profile, "compatible": is_compatible, "compatibility_known": known,
                            "incompatibility": "" if is_compatible else "与代码中的 Python 约束不兼容"}
    compatible = [name for name, p in candidates.items() if p["compatible"]]
    recommended = compatible[0] if constraints and len(compatible) == 1 and (
        candidates[compatible[0]]["compatibility_known"]
    ) else None
    warnings = list(detection["warnings"])
    if constraints and not compatible:
        warnings.append("代码约束冲突或当前环境没有兼容运行配置，请检查声明与依赖")
    elif not constraints:
        warnings.append("代码缺少明确运行时证据，请手动选择")
    elif len(compatible) > 1:
        warnings.append("存在多个兼容运行配置，无法确定 Python 小版本，请手动选择")
    if any(not candidates[name]["compatibility_known"] for name in compatible):
        warnings.append("运行配置未声明 Python 小版本，无法核实小版本约束，请手动确认")
    # Branch names remain visible hints, never compatibility gates or defaults.
    branch_hints = []
    release = re.fullmatch(r"release-aeolus-2\.(\d+)\.\d+", base_branch)
    for name, profile in profiles.items():
        if base_branch in profile.get("base_branches", []) or (
            release and "base_release_max" in profile and int(release[1]) <= profile["base_release_max"]
        ):
            branch_hints.append(name)
    return {"runtime_profiles": candidates, "recommended_runtime_profile_id": recommended,
            "runtime_detection": {**detection, "warnings": warnings, "branch_hints": branch_hints}}


def select_runtime(profiles, base_branch="", selected=None, detection=None):
    detection = detection or detect("", "", {})
    choices = runtime_choices(profiles, detection, base_branch)
    name = selected or choices["recommended_runtime_profile_id"]
    if not name:
        raise ValueError("请选择运行配置；代码没有唯一推荐")
    if name not in choices["runtime_profiles"]:
        raise ValueError("运行配置不存在，请重新选择")
    if not choices["runtime_profiles"][name]["compatible"]:
        raise ValueError("运行配置与代码中的 Python 约束不兼容，请重新选择")
    return name


class RuntimeDetectionService:
    def __init__(self, session_factory, client):
        self.sf, self.client = session_factory, client

    async def ensure(self, task_id):
        target = await run_in_threadpool(self._target, task_id)
        if target is None:
            return
        repository, repo_id, sha = target
        try:
            async with asyncio.timeout(15):
                found = await asyncio.gather(*(
                    get_file(self.client, repo_id, revision=sha, path=path) for path in FILES
                ))
            if any(file and len(file.content) > 256_000 for file in found):
                raise ValueError("runtime evidence file too large")
            result = detect(repository, sha, {
                path: file.content for path, file in zip(FILES, found, strict=True) if file is not None
            })
        except Exception:
            # Never expose repository contents, URLs or client credentials in errors.
            result = detect(repository, sha, {})
            result.update(status="unavailable", warnings=["运行时声明读取失败，请手动选择或稍后刷新"],
                          retry_after=time.time() + 60)
        await run_in_threadpool(self._save, task_id, repo_id, result)

    def _target(self, task_id):
        with self.sf() as session:
            task = session.scalar(select(MeegleAINodeTask).where(MeegleAINodeTask.task_id == task_id))
            if task is None or task.status != "review" or not task.repo_id:
                return None
            round_ = session.scalar(select(MeegleAINodeRound).where(
                MeegleAINodeRound.task_id == task_id,
            ).order_by(MeegleAINodeRound.round_no.desc()).limit(1))
            if round_ is None or not re.fullmatch(r"[0-9a-f]{40}", round_.commit_sha):
                return None
            result = task_detection(task, round_)
            stored = (task.evaluation or {}).get("_runtime_detection")
            if stored == result and (
                result["status"] != "unavailable" or time.time() < result.get("retry_after", 0)
            ):
                return None
            return result["repository"], task.repo_id, round_.commit_sha

    def _save(self, task_id, repo_id, result):
        with self.sf() as session:
            task = session.scalar(select(MeegleAINodeTask).where(
                MeegleAINodeTask.task_id == task_id,
            ).with_for_update())
            if task is None:
                return
            round_ = session.scalar(select(MeegleAINodeRound).where(
                MeegleAINodeRound.task_id == task_id,
            ).order_by(MeegleAINodeRound.round_no.desc()).limit(1))
            current = task_detection(task, round_)
            if task.status != "review" or task.repo_id != repo_id or (
                current["repository"], current["commit_sha"]
            ) != (result["repository"], result["commit_sha"]):
                return
            task.evaluation = {**(task.evaluation or {}), "_runtime_detection": result}
            session.commit()
