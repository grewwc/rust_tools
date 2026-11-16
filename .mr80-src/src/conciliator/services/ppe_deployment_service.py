"""Task-owned PPE request ledger. Remote operations are driven by the durable runner."""

import copy
import re
import uuid
from datetime import datetime, timedelta

from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from conciliator.integrations.ppe.client import environment_name, profile_revision
from conciliator.repositories.models import MeegleAINodeRound, MeegleAINodeTask
from conciliator.repositories.ppe_models import PpeDeployment, PpeEnvironment
from conciliator.services.aeolus_registry import InvalidTaskTransition, TaskNotFound, _database_now
from conciliator.services.runtime_selection import runtime_choices, select_runtime, task_detection
from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import workflow_step

log = get_logger(__name__)

ACTIVE = {"pending", "running", "releasing", "unknown"}
PPE_RESULT_STATES = {
    ("deployed", "entry_pending"), ("failed", "failed"), ("unknown", "unknown"), ("released", "released"),
}


def ppe_result_kind(deployment):
    if not deployment or (deployment.get("status"), deployment.get("stage")) not in PPE_RESULT_STATES:
        return None
    if deployment["status"] == "released":
        return "released" if deployment.get("resource_state") == "released" else None
    releasing = deployment.get("release_requested") or deployment.get("resource_state") == "cleanup_failed"
    if deployment.get("superseded") and not releasing:
        return None
    if deployment["status"] == "unknown":
        return "release_unknown" if releasing else "unknown"
    if deployment["status"] == "failed":
        return "release_failure" if releasing else "failure"
    return "success" if not releasing else None


LABELS = {
    "artifact_wait": "查询本提交的 SCM 制品",
    "bits_plan": "检查 SCM 构建流水线",
    "bits_submit": "触发 SCM 构建流水线",
    "bits_wait": "等待 SCM 构建流水线",
    "image_submit": "提交 ICM 镜像构建",
    "image_wait": "等待 ICM 镜像构建",
    "companion_image_submit": "构建配套服务镜像",
    "companion_image_wait": "等待配套服务镜像",
    "companion_image_verify": "核验配套服务生产镜像",
    "build_plan": "固定 SCM 构建版本",
    "preflight": "检查 PPE 配置与权限",
    "build_submit": "提交 SCM 构建",
    "build_wait": "等待 SCM 构建",
    "deploy": "检查并提交 PPE 部署",
    "ticket_wait": "等待 PPE 部署工单",
    "observe": "核对集群版本与实例",
    "entry_pending": "PPE 部署完成，待业务验证",
    "superseded": "旧轮次部署",
    "failed": "PPE 部署失败",
    "unknown": "PPE 操作结果待确认",
    "release": "释放 PPE",
    "release_wait": "等待 PPE 释放工单",
    "release_verify": "确认 PPE 资源已回收",
    "released": "PPE 已释放",
}


def task_lock(session, task_id):
    task = session.scalar(select(MeegleAINodeTask).where(MeegleAINodeTask.task_id == task_id).with_for_update())
    if task is None:
        raise TaskNotFound(task_id)
    return task


def latest_round(session, task_id):
    return session.scalar(
        select(MeegleAINodeRound)
        .where(MeegleAINodeRound.task_id == task_id)
        .order_by(MeegleAINodeRound.round_no.desc())
        .limit(1)
    )


def latest_environment(session, task_id):
    return session.scalar(
        select(PpeEnvironment)
        .where(PpeEnvironment.task_id == task_id)
        .order_by(PpeEnvironment.generation.desc())
        .limit(1)
        .with_for_update()
    )


def latest_deployment(session, task_id):
    return session.scalar(
        select(PpeDeployment).where(PpeDeployment.task_id == task_id).order_by(PpeDeployment.sequence.desc()).limit(1)
    )


def current_round(session, deployment):
    task = task_lock(session, deployment.task_id)
    round_ = latest_round(session, deployment.task_id)
    return bool(
        task.status == "review"
        and round_
        and (round_.round_no, round_.commit_sha) == (deployment.round_no, deployment.commit_sha)
    )


def invalidate_ppe(session, task_id, *, release):
    """Caller holds the task lock. Feedback keeps resources; completion requests their release."""
    environments = list(
        session.scalars(select(PpeEnvironment).where(PpeEnvironment.task_id == task_id).with_for_update())
    )
    for env in environments:
        if env.resource_state == "released":
            continue
        if release:
            env.data = {**env.data, "release_requested": True}
            if not env.active_deployment_id:
                deployment = session.scalar(
                    select(PpeDeployment)
                    .where(
                        PpeDeployment.environment_id == env.environment_id,
                    )
                    .order_by(PpeDeployment.sequence.desc())
                    .limit(1)
                )
                if deployment:
                    deployment.status, deployment.stage = "releasing", "release"
                    deployment.next_poll_at = _database_now(session)
                    env.active_deployment_id = deployment.deployment_id
        if not release:
            for record in session.scalars(
                select(PpeDeployment).where(PpeDeployment.environment_id == env.environment_id)
            ):
                record.data = {**record.data, "superseded": True, "verified": False}


def verify_ppe(session, task_id, deployment_id):
    """A human verification must target the currently running, latest-round deployment."""
    record = session.get(PpeDeployment, deployment_id)
    env = latest_environment(session, task_id)
    now = _database_now(session)
    if (
        record is None
        or record.task_id != task_id
        or env is None
        or record.environment_id != env.environment_id
        or env.current_deployment_id != deployment_id
        or env.resource_state != "running"
        or record.status != "deployed"
        or record.stage != "entry_pending"
        or env.active_deployment_id
        or env.data.get("release_requested")
        or record.data.get("superseded")
        or not current_round(session, record)
    ):
        raise InvalidTaskTransition("PPE 已变更或未完成部署，不能确认验证通过")
    try:
        age = (now - datetime.fromisoformat(env.data["observed_at"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        age = 999
    if not 0 <= age <= 30 or not env.data.get("clusters_ready"):
        raise InvalidTaskTransition("PPE 实例状态需要重新核验，请刷新后再确认")
    record.data = {**record.data, "verified": True, "verification_source": "user", "verified_at": now.isoformat()}


class PpeDeploymentService:
    def __init__(self, session_factory, profile, *, profiles=(), runtime_detector=None):
        self.sf = session_factory
        self.runtime_detector = runtime_detector
        self.profile = copy.deepcopy(profile)
        self.revision = profile_revision(profile)
        self.profiles = {(p["repository"], p.get("python_version", "")): copy.deepcopy(p) for p in (profile, *profiles)}

    def _profiles(self, task):
        context = (task.evaluation or {}).get("code_context") or {}
        repository = context.get("repository", "dp/aeolus")
        return {
            "py27" if version == "2.7" else "py3": profile
            for (repo, version), profile in self.profiles.items() if repo == repository
        }

    def _choices(self, task, round_):
        profiles = {
            name: {
                "profile_revision": profile_revision(profile),
                "python_version": profile.get("python_version") or (
                    "3.10" if profile["repository"] == "dp/aeolus" else "3"
                ),
            }
            for name, profile in self._profiles(task).items()
        }
        return runtime_choices(profiles, task_detection(task, round_),
                               ((task.evaluation or {}).get("code_context") or {}).get("base_branch", ""))

    @workflow_step("ppe_create")
    async def create(self, task_id, request_id, round_no, commit_sha, revision, runtime_profile_id=None):
        if self.runtime_detector is not None:
            await self.runtime_detector.ensure(task_id)
        return await run_in_threadpool(
            self._create, task_id, request_id, round_no, commit_sha, revision, runtime_profile_id,
        )

    def _create(self, task_id, request_id, round_no, commit_sha, revision, runtime_profile_id=None, *, retry_of=None):
        if not request_id or len(request_id) > 128 or not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            raise ValueError("PPE 请求必须提供请求标识和完整 SHA")
        with self.sf() as session:
            task = task_lock(session, task_id)
            existing = session.scalar(
                select(PpeDeployment).where(
                    PpeDeployment.task_id == task_id,
                    PpeDeployment.request_id == request_id,
                )
            )
            if existing:
                if (
                    existing.round_no,
                    existing.commit_sha,
                    existing.profile_revision,
                    existing.data.get("retry_of"),
                ) != (
                    round_no,
                    commit_sha,
                    revision,
                    retry_of,
                ) or runtime_profile_id and existing.data.get("runtime_profile_id") != runtime_profile_id:
                    raise InvalidTaskTransition("request_id 已用于其他 PPE 请求")
                return self._snapshot(session, task_id, existing)
            latest = latest_round(session, task_id)
            choices = self._choices(task, latest)
            profiles = self._profiles(task)
            if task.status != "review" or latest is None or not profiles or (
                latest.status, latest.round_no, latest.commit_sha
            ) != ("review", round_no, commit_sha):
                raise InvalidTaskTransition("仅支持已配置仓库最新成功轮次的 PPE 部署")
            # Older clients already select a template by its exact revision.
            selected = runtime_profile_id or next((
                name for name, p in choices["runtime_profiles"].items() if p["profile_revision"] == revision
            ), None)
            original = session.get(PpeDeployment, retry_of) if retry_of else None
            if original is not None and original.task_id == task_id:
                profile = copy.deepcopy(original.data["profile"])
                selected = original.data.get("runtime_profile_id") or (
                    "py27" if profile.get("python_version") == "2.7" else "py3"
                )
                detection = original.data.get("runtime_detection", task_detection(task, latest))
            else:
                if selected is None:
                    raise InvalidTaskTransition("PPE 模板已变更，请刷新后重新提交")
                selected = select_runtime(
                    choices["runtime_profiles"], selected=selected, detection=task_detection(task, latest),
                )
                profile = profiles[selected]
                detection = task_detection(task, latest)
            if revision != profile_revision(profile):
                raise InvalidTaskTransition("PPE 模板已变更，请刷新后重新提交")
            env = latest_environment(session, task_id)
            if env and env.resource_state != "released":
                if env.active_deployment_id or env.data.get("release_requested"):
                    raise InvalidTaskTransition("前一次 PPE 操作尚未结束或结果待确认")
                if env.profile_revision != revision:
                    raise InvalidTaskTransition("PPE 配置已变更，请先释放旧环境，再重新部署")
            now = _database_now(session)
            if env is None or env.resource_state == "released":
                identity = uuid.uuid4().hex
                env = PpeEnvironment(
                    environment_id=identity,
                    task_id=task_id,
                    generation=env.generation + 1 if env else 1,
                    name=environment_name(identity, profile),
                    resource_state="planned",
                    profile_revision=revision,
                    created_at=now,
                    data={},
                )
                session.add(env)
            identity = uuid.uuid4().hex
            data = {"profile": copy.deepcopy(profile), "retry_of": retry_of, "operator": profile["operator"],
                    "runtime_profile_id": selected, "runtime_detection": detection}
            previous_artifacts = list(
                session.scalars(
                    select(PpeDeployment)
                    .where(
                        PpeDeployment.task_id == task_id,
                        PpeDeployment.commit_sha == commit_sha,
                        PpeDeployment.profile_revision == revision,
                    )
                    .order_by(PpeDeployment.sequence.desc())
                )
            )
            reusable = next((
                r for r in previous_artifacts
                if r.data.get("build_ok") and r.data.get("scm_build_type") in {None, "test"}
            ), None)
            original_runs: dict[str, PpeDeployment] = {}
            for previous in previous_artifacts:
                if previous.data.get("bits_run_id"):
                    # Newer retries may have observed the same run's terminal result.
                    original_runs.setdefault(previous.data["bits_run_id"], previous)
            unfinished_run = next(
                (r for r in original_runs.values() if r.data.get("bits_run_status") not in {8, 9, 11}), None
            )
            if unfinished_run:
                data.update({
                    k: unfinished_run.data[k] for k in ("bits_run_id", "bits_run_seq")
                    if k in unfinished_run.data
                })
            if reusable:
                keys = (
                    "scm_version",
                    "scm_version_id",
                    "scm_build_type",
                    "build_ok",
                    "image_id",
                    "image_version_id",
                    "image_version",
                    "image_status",
                    "resolved_base_image",
                )
                data.update({k: reusable.data[k] for k in keys if k in reusable.data})
            if profile.get("companion_profile"):
                companion_build = session.scalars(
                    select(PpeDeployment).where(
                        PpeDeployment.task_id == task_id, PpeDeployment.profile_revision == revision,
                    ).order_by(PpeDeployment.sequence.desc())
                )
                for previous in companion_build:
                    artifact = previous.data.get("companion_artifact", {})
                    if artifact.get("image_version_id") and artifact.get("image_status") != "fail":
                        data["companion_artifact"] = copy.deepcopy(artifact)
                        break
            if retry_of:
                previous = session.get(PpeDeployment, retry_of)
                if previous is None or previous.task_id != task_id or previous.status not in {"failed", "deployed"}:
                    raise InvalidTaskTransition("该 PPE 请求不可重试；结果未知时必须先对账")
                if (
                    previous.commit_sha == commit_sha and previous.data.get("build_ok")
                    and previous.data.get("scm_build_type") in {None, "test"}
                ):
                    data.update({
                        k: previous.data[k] for k in ("scm_version", "scm_version_id", "scm_build_type", "build_ok")
                        if k in previous.data
                    })
            previous_request = latest_deployment(session, task_id)
            record = PpeDeployment(
                deployment_id=identity,
                task_id=task_id,
                environment_id=env.environment_id,
                request_id=request_id,
                sequence=previous_request.sequence + 1 if previous_request else 1,
                round_no=round_no,
                commit_sha=commit_sha,
                profile_revision=revision,
                status="pending",
                stage="preflight",
                created_at=now,
                next_poll_at=now,
                error="",
                data={**data, "notification_pending": False, "stage_started_at": now.isoformat()},
            )
            session.add(record)
            env.active_deployment_id = identity
            session.flush()
            result = self._snapshot(session, task_id, record)
            session.commit()
            log.info(
                "meegle_ppe_created", workflow="meegle", task_id=task_id,
                deployment_id=identity, environment_id=env.environment_id, round_no=round_no,
                commit_sha=commit_sha, retry_of=retry_of,
            )
            return result

    @workflow_step("ppe_retry")
    async def retry(self, task_id, deployment_id, request_id):
        return await run_in_threadpool(self._retry, task_id, deployment_id, request_id)

    def _retry(self, task_id, deployment_id, request_id):
        with self.sf() as session:
            row = session.get(PpeDeployment, deployment_id)
            if row is None or row.task_id != task_id:
                raise TaskNotFound(deployment_id)
            args = row.round_no, row.commit_sha, row.profile_revision
        return self._create(task_id, request_id, *args, retry_of=deployment_id)

    @workflow_step("ppe_release")
    async def release(self, task_id, deployment_id, environment_id, generation):
        return await run_in_threadpool(self._release, task_id, deployment_id, environment_id, generation)

    def _release(self, task_id, deployment_id, environment_id, generation):
        with self.sf() as session:
            task_lock(session, task_id)
            env = latest_environment(session, task_id)
            record = latest_deployment(session, task_id)
            if (
                env is None
                or record is None
                or record.deployment_id != deployment_id
                or env.environment_id != environment_id
                or env.generation != generation
            ):
                raise InvalidTaskTransition("PPE 释放目标已变更，请刷新后重试")
            if env.resource_state == "released":
                return self._snapshot(session, task_id, record)
            env.data = {**env.data, "release_requested": True}
            if not env.active_deployment_id:
                env.active_deployment_id = record.deployment_id
                record.status, record.stage = "releasing", "release"
                record.next_poll_at = _database_now(session)
            result = self._snapshot(session, task_id, record)
            session.commit()
            log.info(
                "meegle_ppe_release_requested", workflow="meegle", task_id=task_id,
                deployment_id=deployment_id, environment_id=environment_id, generation=generation,
            )
            return result

    async def snapshot(self, task_id, deployment_id=None):
        if self.runtime_detector is not None:
            await self.runtime_detector.ensure(task_id)
        return await run_in_threadpool(self._snapshot_query, task_id, deployment_id)

    def _snapshot_query(self, task_id, deployment_id=None):
        with self.sf() as session:
            row = session.get(PpeDeployment, deployment_id) if deployment_id else latest_deployment(session, task_id)
            if deployment_id and (row is None or row.task_id != task_id):
                raise TaskNotFound(deployment_id)
            return self._snapshot(session, task_id, row)

    def _snapshot(self, session, task_id, row):
        task = task_lock(session, task_id)
        choices = self._choices(task, latest_round(session, task_id))
        recommended = choices["recommended_runtime_profile_id"]
        profile = self._profiles(task).get(recommended)
        capability = {
            **choices,
            "enabled": bool(choices["runtime_profiles"]),
            "profile_revision": profile_revision(profile) if profile else "",
            "repository": choices["runtime_detection"]["repository"],
            "python_version": choices["runtime_profiles"][recommended]["python_version"] if recommended else None,
        }
        if row is None:
            return {"ppe_deployment": None, "ppe_capability": capability}
        env = session.get(PpeEnvironment, row.environment_id)
        now = _database_now(session)
        recent = False
        if env and env.data.get("observed_at"):
            recent = timedelta(0) <= now - datetime.fromisoformat(env.data["observed_at"]) <= timedelta(seconds=30)
        data = row.data
        is_current_round = current_round(session, row)
        current = session.get(PpeDeployment, env.current_deployment_id) if env.current_deployment_id else None
        can_view_environment = bool(
            env.platform_id and env.resource_state != "released" and not env.data.get("release_requested")
        )
        can_access_environment = can_view_environment and current is not None
        status_label = LABELS.get(row.stage, row.stage)
        if row.stage == "ticket_wait" and env.data.get("approvals"):
            status_label = "等待 TCE 资源审批"
        elif row.stage == "entry_pending" and not env.data.get("clusters_ready"):
            status_label = "PPE 暂未就绪，正在重新检查"
        snapshot = {
            "deployment_id": row.deployment_id,
            "environment_id": row.environment_id,
            "generation": env.generation,
            "environment_name": env.name,
            "environment_platform_id": env.platform_id,
            "round_no": row.round_no,
            "commit_sha": row.commit_sha,
            "profile_revision": row.profile_revision,
            "runtime_profile_id": data.get("runtime_profile_id"),
            "python_version": data["profile"].get("python_version") or (
                "3.10" if data["profile"]["repository"] == "dp/aeolus" else "3"
            ),
            "status": row.status,
            "stage": row.stage,
            "status_label": status_label,
            "error": row.error,
            "scm_version": data.get("scm_version"),
            "scm_version_id": data.get("scm_version_id"),
            "bits_run_id": data.get("bits_run_id"),
            "bits_run_seq": data.get("bits_run_seq"),
            "bits_run_status": data.get("bits_run_status"),
            "bits_run_url": (
                f"https://bits.bytedance.net/devops/{data['profile']['bits_scm']['space_id']}"
                f"/pipeline/detail/{data['profile']['bits_scm']['pipeline_id']}?runId={data['bits_run_id']}"
                if data.get("bits_run_id") and data["profile"].get("bits_scm")
                else ""
            ),
            "image_version": data.get("image_version"),
            "image_status": data.get("image_status"),
            "image_url": (
                f"https://cloud.bytedance.net/icm/detail/{data['image_id']}/versions?version={data['image_version']}"
                if data.get("image_id") and data.get("image_version")
                else ""
            ),
            "operator": data.get("operator"),
            "ticket_id": data.get("ticket_id"),
            "release_ticket_id": data.get("release_ticket_id"),
            "resource_state": env.resource_state,
            "release_requested": bool(env.data.get("release_requested")),
            "released": env.resource_state == "released",
            "verified": bool(data.get("verified")),
            "superseded": bool(data.get("superseded")),
            "current_deployment": {
                "deployment_id": current.deployment_id,
                "round_no": current.round_no,
                "commit_sha": current.commit_sha,
                "image_version": current.data.get("image_version"),
            } if current and env.resource_state != "released" else None,
            "clusters": env.data.get("clusters", []),
            "approvals": env.data.get("approvals", []),
            "due_time": env.data.get("due_time"),
            "recycle_rule": env.data.get("recycle_rule"),
            "can_retry": row.status == "failed"
            and not env.active_deployment_id
            and not env.data.get("release_requested")
            and is_current_round,
            "can_deploy": (
                not env.active_deployment_id and not env.data.get("release_requested")
                or env.resource_state == "released"
            ) and not (
                row.status == "deployed"
                and is_current_round
                and env.current_deployment_id == row.deployment_id
                and env.resource_state != "released"
            ),
            "can_release": env.resource_state != "released"
            and (not env.data.get("release_requested") or env.resource_state == "cleanup_failed"),
            "can_view_environment": can_view_environment,
            "can_access_environment": can_access_environment,
            "can_verify": bool(
                row.status == "deployed"
                and row.stage == "entry_pending"
                and recent
                and env.current_deployment_id == row.deployment_id
                and env.data.get("clusters_ready")
                and not env.data.get("release_requested")
                and not env.active_deployment_id
                and not data.get("superseded")
                and is_current_round
            ),
            "environment_url": (
                f"https://bits.bytedance.net/env/life/ppe/online_cn/detail/{env.name}/service"
                if can_view_environment else ""
            ),
            "scm_url": f"https://cloud.bytedance.net/scm/detail/{data['profile']['scm_repo_id']}/versions/",
            "access_hint": (
                f"使用现有 Aeolus 前端入口，确认 API 请求携带 X-TT-ENV: {env.name}；仅使用测试账号与数据。"
                if can_access_environment else ""
            ),
        }
        return {"ppe_deployment": snapshot, "ppe_capability": capability}
