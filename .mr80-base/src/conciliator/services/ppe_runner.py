"""Durable PPE orchestration. A remote ticket, not a local lease, serializes changes."""

import asyncio
import contextlib
import copy
import hashlib
import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy import or_, select
from starlette.concurrency import run_in_threadpool

from conciliator.integrations.ppe.client import environment_deleted, precheck_summary
from conciliator.integrations.ppe.environment import merge_owned_services
from conciliator.integrations.ppe.icm import image_body
from conciliator.integrations.ppe.scm import PlatformError, ScmClient
from conciliator.repositories.ppe_models import PpeDeployment, PpeEnvironment
from conciliator.services.aeolus_registry import _database_now
from conciliator.services.ppe_deployment_service import (
    PPE_RESULT_STATES,
    current_round,
    latest_deployment,
    ppe_result_kind,
    task_lock,
)
from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import error_fields, logged_step, workflow_step

log = get_logger(__name__)
EXTERNAL_WAIT = {
    "artifact_wait", "bits_wait", "build_wait", "image_wait", "companion_image_wait", "ticket_wait", "release_wait",
}


class PpeRunner:
    def __init__(self, service, http, icm, ppe, *, bits=None, notifier=None, interval=5):
        self.service, self.sf = service, service.sf
        self.http, self.icm, self.ppe = http, icm, ppe
        self.bits = bits
        self.notifier, self.interval = notifier, interval
        self._task = None

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="ppe:deployments")

    async def aclose(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self.http.aclose()
        await self.icm.detail_http.aclose()
        if self.bits:
            await self.bits.http.aclose()

    async def _loop(self):
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                log.warning("ppe_scan_failed", workflow="meegle", **error_fields(exc))
            await asyncio.sleep(self.interval)

    async def run_once(self):
        ids = await run_in_threadpool(self._due)
        for identity in ids:
            item = await run_in_threadpool(self._claim, identity)
            if item is None:
                continue
            try:
                elapsed = (await self._now() - datetime.fromisoformat(item["data"]["stage_started_at"])).total_seconds()
                if item["stage"] == "artifact_wait" and elapsed > 1800:
                    await self._fail(item, "30 分钟内未获得本提交的成功 SCM 制品，请检查 Codebase 流水线触发及构建状态")
                    continue
                if item["stage"] == "bits_wait" and elapsed > 1800:
                    await self._fail(item, "SCM 流水线等待超过三十分钟，请核对原运行；重试仍查询原运行")
                    continue
                if item["stage"] in EXTERNAL_WAIT | {"release_verify"} and elapsed > 7200:
                    raise PlatformError("平台工单等待超过两小时，请核对原工单；不会重复提交", uncertain=True)
                if item["stage"] == "observe" and elapsed > 1200:
                    await self._fail(item, "PPE 实例在二十分钟内未就绪，请查看集群及探针状态")
                    continue
                await self._advance(item)
            except PlatformError as exc:
                log.warning(
                    "ppe_step_failed", workflow="meegle", task_id=item["task_id"], deployment_id=identity,
                    stage=item["stage"], **error_fields(exc),
                )
                if exc.uncertain:
                    await self._finish(item, status="unknown", stage="unknown", error=str(exc))
                elif item["stage"] in EXTERNAL_WAIT or item["stage"] in {"observe", "entry_pending", "release_verify"}:
                    # Reads never replay the operation. Retain external identity and retry polling.
                    await self._finish(
                        item, error=str(exc), delay=10,
                        env_data={"clusters_ready": False} if item["stage"] == "entry_pending" else {},
                    )
                else:
                    await self._fail(item, str(exc))
            except Exception as exc:
                # Do not copy raw platform or SDK exceptions into persistent state or notifications.
                log.warning(
                    "ppe_step_failed", workflow="meegle", task_id=item["task_id"], deployment_id=identity,
                    stage=item["stage"], **error_fields(exc),
                )
                if item.get("intent"):
                    await self._finish(
                        item, status="unknown", stage="unknown", error="PPE 写入结果待确认，请核对原请求"
                    )
                elif item["stage"] in EXTERNAL_WAIT | {"release_verify", "entry_pending"}:
                    await self._finish(
                        item, error="PPE 状态查询暂不可用，请稍后刷新", delay=10,
                        env_data={"clusters_ready": False} if item["stage"] == "entry_pending" else {},
                    )
                else:
                    await self._fail(item, "PPE 平台响应无法解析，请维护者核对接口契约后重试")
        if self.notifier:
            for task_id, identity, version in await run_in_threadpool(self._notifications):
                try:
                    await asyncio.wait_for(self.notifier(task_id, identity), 10)
                    await run_in_threadpool(self._notification_sent, identity, version)
                except Exception as exc:
                    log.warning(
                        "ppe_notification_failed", workflow="meegle", task_id=task_id, deployment_id=identity,
                        **error_fields(exc),
                    )

    def _notifications(self):
        with self.sf() as session:
            notifications = []
            for row in session.scalars(
                select(PpeDeployment)
                .where(PpeDeployment.data["notification_pending"].as_boolean().is_(True))
                .limit(20)
            ):
                latest = latest_deployment(session, row.task_id)
                env = session.get(PpeEnvironment, row.environment_id)
                if env is None:
                    continue
                if (
                    ppe_result_kind({
                        **row.data, "status": row.status, "stage": row.stage,
                        "release_requested": env.data.get("release_requested"), "resource_state": env.resource_state,
                    }) is not None
                    and latest.deployment_id == row.deployment_id
                ):
                    notifications.append((row.task_id, row.deployment_id, row.data.get("notification_version", 0)))
                else:
                    # Discard progress and obsolete results queued by older versions.
                    row.data = {**row.data, "notification_pending": False}
            session.commit()
            return notifications

    def _notification_sent(self, identity, version):
        with self.sf() as session:
            row = session.get(PpeDeployment, identity, with_for_update=True)
            if row and row.data.get("notification_version", 0) == version:
                row.data = {**row.data, "notification_pending": False}
                session.commit()

    def _due(self):
        with self.sf() as session:
            return list(
                session.scalars(
                    select(PpeDeployment.deployment_id)
                    .join(PpeEnvironment, PpeEnvironment.environment_id == PpeDeployment.environment_id)
                    .where(
                        PpeDeployment.status.in_({"pending", "running", "releasing", "unknown", "deployed"}),
                        PpeDeployment.next_poll_at <= _database_now(session),
                        or_(PpeDeployment.lease_until.is_(None), PpeDeployment.lease_until <= _database_now(session)),
                        or_(
                            PpeEnvironment.active_deployment_id == PpeDeployment.deployment_id,
                            (PpeDeployment.status == "deployed")
                            & (PpeEnvironment.current_deployment_id == PpeDeployment.deployment_id)
                            & PpeEnvironment.active_deployment_id.is_(None),
                        ),
                    )
                    .order_by(PpeDeployment.next_poll_at)
                    .limit(20)
                )
            )

    def _claim(self, identity):
        with self.sf() as session:
            record = session.get(PpeDeployment, identity)
            if record is None:
                return None
            task_lock(session, record.task_id)
            env = session.scalar(
                select(PpeEnvironment).where(PpeEnvironment.environment_id == record.environment_id).with_for_update()
            )
            session.refresh(record, with_for_update=True)
            previous = record.status, record.stage
            now = _database_now(session)
            if (
                record.status not in {"pending", "running", "releasing", "unknown", "deployed"}
                or record.next_poll_at > now
                or (record.lease_until and record.lease_until > now)
            ):
                return None
            if record.status == "deployed":
                if (
                    env.active_deployment_id
                    or env.current_deployment_id != identity
                    or env.data.get("release_requested")
                ):
                    return None
            elif env.active_deployment_id != identity:
                return None
            if record.data.get("intent") or record.stage == "unknown":
                if record.stage != "unknown":
                    record.data = {
                        **record.data,
                        "notification_pending": True,
                        "notification_version": record.data.get("notification_version", 0) + 1,
                    }
                record.status, record.stage = "unknown", "unknown"
                record.error = "远程操作结果待确认；禁止重复提交，请按请求记录对账"
                record.next_poll_at = now + timedelta(minutes=5)
                session.commit()
                if previous != (record.status, record.stage):
                    log.warning(
                        "meegle_ppe_quarantined", workflow="meegle", task_id=record.task_id,
                        deployment_id=identity, previous_status=previous[0], previous_stage=previous[1],
                        status=record.status, stage=record.stage,
                    )
                return None
            lease = uuid.uuid4().hex
            record.lease_owner, record.lease_until = lease, now + timedelta(seconds=180)
            if record.status == "pending":
                record.status = "running"
            item = {
                "deployment_id": identity,
                "task_id": record.task_id,
                "environment_id": env.environment_id,
                "name": env.name,
                "platform_id": env.platform_id,
                "env_data": copy.deepcopy(env.data),
                "stage": record.stage,
                "lease": lease,
                "data": copy.deepcopy(record.data),
                "commit_sha": record.commit_sha,
                "current": current_round(session, record),
                "release": bool(env.data.get("release_requested")),
                "status": record.status,
            }
            session.commit()
            if previous[0] != record.status:
                log.info(
                    "meegle_ppe_claimed", workflow="meegle", task_id=record.task_id, deployment_id=identity,
                    previous_status=previous[0], status=record.status, stage=record.stage,
                )
            return item

    async def _finish(self, item, **changes):
        return await run_in_threadpool(self._save, item, changes)

    def _save(self, item, changes):
        with self.sf() as session:
            task_lock(session, item["task_id"])
            env = session.scalar(
                select(PpeEnvironment).where(PpeEnvironment.environment_id == item["environment_id"]).with_for_update()
            )
            record = session.scalar(
                select(PpeDeployment).where(PpeDeployment.deployment_id == item["deployment_id"]).with_for_update()
            )
            intent = record.data.get("intent")
            late_write = bool(item.get("intent") and intent and intent["id"] == item["intent"])
            if record.lease_owner != item["lease"] and not late_write:
                return False
            now = _database_now(session)
            previous = record.status, record.stage
            had_error = bool(record.error)
            data = {**record.data, **changes.get("data", {})}
            if changes.get("clear_intent"):
                data.pop("intent", None)
            record.data = data
            if (
                changes.get("stage")
                and changes["stage"] != record.stage
                and changes["stage"] not in {"failed", "unknown"}
            ):
                record.error = ""
            for field in ("status", "stage", "error"):
                if field in changes:
                    setattr(record, field, changes[field])
            if changes.get("stage") and changes["stage"] != item["stage"]:
                record.data = {**record.data, "stage_started_at": now.isoformat()}
            if not changes.get("keep_lease"):
                record.lease_owner, record.lease_until = None, None
            record.next_poll_at = now + timedelta(seconds=changes.get("delay", 0))
            env_changes = changes.get("env_data", {})
            if "owned_services" in env_changes:
                env_changes = {
                    **env_changes,
                    "owned_services": merge_owned_services(
                        env.data.get("owned_services", []), env_changes["owned_services"],
                    ),
                }
            env.data = {**env.data, **env_changes}
            if "platform_id" in changes:
                if env.platform_id and env.platform_id != changes["platform_id"]:
                    raise PlatformError("PPE 资源身份改变，需人工对账")
                env.platform_id = changes["platform_id"]
            if "resource_state" in changes:
                env.resource_state = changes["resource_state"]
            if changes.get("current"):
                env.current_deployment_id = record.deployment_id
                if not current_round(session, record):
                    record.status, record.stage = "superseded", "superseded"
                    record.data = {**record.data, "superseded": True, "verified": False}
            if changes.get("terminal") and env.active_deployment_id == record.deployment_id:
                if (
                    env.data.get("release_requested")
                    and env.resource_state != "released"
                    and record.stage != "release"
                    and not changes.get("release_failed")
                ):
                    record.status, record.stage = "releasing", "release"
                    record.next_poll_at = now
                else:
                    env.active_deployment_id = None
            if previous != (record.status, record.stage):
                record.data = {
                    **record.data,
                    "notification_pending": (record.status, record.stage) in PPE_RESULT_STATES,
                    "notification_version": record.data.get("notification_version", 0) + 1,
                }
            session.commit()
            if previous != (record.status, record.stage) or had_error and not record.error:
                log.info(
                    "meegle_ppe_state_changed", workflow="meegle", task_id=record.task_id,
                    deployment_id=record.deployment_id, environment_id=record.environment_id,
                    round_no=record.round_no, previous_status=previous[0], previous_stage=previous[1],
                    status=record.status, stage=record.stage, resource_state=env.resource_state,
                    error_cleared=had_error and not bool(record.error),
                    external_environment_id=env.platform_id,
                    external_ids={
                        key: record.data[key] for key in (
                            "scm_version_id", "image_version_id", "ticket_id", "release_ticket_id", "bits_run_id",
                        ) if key in record.data
                    },
                )
            return True

    async def _write(self, item, kind, body, operation):
        intent = uuid.uuid4().hex
        if not await run_in_threadpool(self._intent, item, kind, intent, body):
            return None
        item["intent"] = intent
        with logged_step(
            "ppe_write", task_id=item["task_id"], deployment_id=item["deployment_id"],
            stage=item["stage"], operation=kind, intent_id=intent,
        ):
            return await operation()

    def _intent(self, item, kind, intent, body):
        with self.sf() as session:
            task_lock(session, item["task_id"])
            env = session.scalar(
                select(PpeEnvironment).where(PpeEnvironment.environment_id == item["environment_id"]).with_for_update()
            )
            record = session.get(PpeDeployment, item["deployment_id"], with_for_update=True)
            if (
                record.lease_owner != item["lease"]
                or env.active_deployment_id != record.deployment_id
                or record.data.get("intent")
            ):
                return False
            releasing = kind == "release"
            if not releasing and (env.data.get("release_requested") or not current_round(session, record)):
                record.lease_owner, record.lease_until = None, None
                record.next_poll_at = _database_now(session)
                session.commit()
                return False
            record.data = {
                **record.data,
                "intent": {
                    "id": intent,
                    "kind": kind,
                    "digest": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                },
            }
            session.commit()
            return True

    async def _fail(self, item, message):
        releasing = item["stage"] in {"release", "release_wait", "release_verify"}
        await self._finish(
            item,
            status="failed",
            stage="failed",
            error=message,
            terminal=True,
            clear_intent=True,
            release_failed=releasing,
            **({"resource_state": "cleanup_failed"} if releasing else {}),
        )

    def _client(self, data):
        if not hasattr(self.ppe, "for_profile"):
            return self.ppe
        if data["profile"].get("companion_profile"):
            return self.ppe.for_profile(data["profile"], companion_artifact=data.get("companion_artifact"))
        return self.ppe.for_profile(data["profile"])

    async def _absent(self, ppe, item):
        resources = item["env_data"]
        identity = (
            resources.get("owned_services", [])
            if item["data"]["profile"].get("companion_profile") else resources.get("service_id")
        )
        return await ppe.absent(item["name"], identity, resources.get("clusters", []))

    async def _companion_version(self, data):
        profile = data["profile"]["companion_profile"]
        baseline = profile["baseline_artifact"]
        version = await ScmClient(self.http, data["operator"], profile["scm_repo_id"]).version(baseline["version"])
        if version is None or version.version_id != baseline["version_id"]:
            raise PlatformError("配套服务 SCM 基线身份已改变")
        version.require_success(baseline["commit_sha"], allowed_types={baseline["build_type"]})
        return version

    @staticmethod
    def _after_image(data):
        if (data["profile"].get("companion_profile") or {}).get("production_baseline"):
            return "companion_image_verify"
        return "companion_image_submit" if data["profile"].get("companion_profile") else "deploy"

    @workflow_step("ppe_advance", level="debug")
    async def _advance(self, item):
        data, stage = item["data"], item["stage"]
        profile = data["profile"]
        # A remote ENV write must finish before another update or deletion starts.
        if item["release"] and stage not in {"ticket_wait", "release", "release_wait", "release_verify"}:
            await self._finish(item, status="releasing", stage="release")
            return
        if not item["current"] and not item["release"] and stage not in {"ticket_wait", "observe", "entry_pending"}:
            await self._finish(item, status="superseded", stage="superseded", data={"superseded": True}, terminal=True)
            return
        baseline = (profile.get("companion_profile") or {}).get("production_baseline")
        companion = data.get("companion_artifact", {})
        if (
            stage in {"deploy", "observe", "entry_pending"} and baseline and not baseline["scm_repos"]
            and not companion.get("embedded_scm_repos")
        ):
            # Old schema 3 artifacts omitted Atum's embedded manifest. Read the same
            # immutable image, without changing its frozen profile or redeploying it.
            image = await self.icm.baseline_version(baseline)
            if any(companion.get(k) != image[k] for k in ("image_id", "image_version_id", "image_version")):
                raise PlatformError("配套服务镜像与冻结的生产基准不一致")
            companion = {**companion, **image}
            if not await self._finish(item, data={"companion_artifact": companion}, keep_lease=True):
                return
            data["companion_artifact"] = companion
        scm = ScmClient(self.http, data["operator"], profile["scm_repo_id"])
        ppe = self._client(data)
        bits = self.bits.for_profile(profile) if hasattr(self.bits, "for_profile") else self.bits
        if stage == "preflight":
            if profile.get("companion_profile"):
                await self._companion_version(data)
                baseline = profile["companion_profile"].get("production_baseline")
                if baseline:
                    await self.icm.baseline_version(baseline)
            # Preparing the template also verifies source-cluster ownership and sensitive config access.
            baseline_version = next(d["version"] for d in profile["meta"]["scm_repos"] if d["main_repo"])
            await ppe.prepare(
                profile,
                item["environment_id"],
                baseline_version,
                image_version=profile["image_recipe"]["reference_version"],
                operator=data["operator"],
            )
            await self._finish(item, stage="image_submit" if data.get("build_ok") else "artifact_wait")
        elif stage == "artifact_wait":
            artifact = await scm.for_commit(item["commit_sha"])
            if profile.get("bits_scm") and (artifact is None or artifact.status != "build_ok"):
                await self._finish(item, stage="bits_plan")
            elif artifact is None:
                await self._finish(item, delay=10)
            elif artifact.status in {"build_failed", "build_fail", "failed", "canceled"}:
                await self._finish(
                    item,
                    status="failed",
                    stage="failed",
                    terminal=True,
                    data={"scm_version": artifact.version, "scm_version_id": artifact.version_id},
                    error="Codebase 的 SCM 构建失败，请查看对应制品日志后重试",
                )
            elif artifact.status != "build_ok":
                await self._finish(
                    item, delay=10, data={"scm_version": artifact.version, "scm_version_id": artifact.version_id}
                )
            else:
                artifact.require_success(item["commit_sha"])
                await self._finish(
                    item,
                    stage="image_submit",
                    data={
                        "build_ok": True,
                        "scm_version": artifact.version,
                        "scm_version_id": artifact.version_id,
                        "scm_build_type": artifact.build_type,
                        "artifact_source": "existing_scm_build",
                    },
                )
        elif stage == "bits_plan":
            if bits is None:
                raise PlatformError("PPE SCM 流水线客户端未配置")
            # A failed deployment retry may resume its original BITS run.
            if data.get("bits_run_id"):
                await self._finish(item, stage="bits_wait")
            else:
                body = await bits.plan(profile["bits_scm"], item["commit_sha"], data["operator"])
                await self._finish(item, stage="bits_submit", data={"bits_request": body})
        elif stage == "bits_submit":
            body = await bits.plan(profile["bits_scm"], item["commit_sha"], data["operator"])
            if body != data["bits_request"]:
                raise PlatformError("SCM 流水线配置已改变，请重新检查后重试")
            result = await self._write(
                item, "bits", body, lambda: bits.submit(body, data["operator"])
            )
            if result:
                await self._finish(item, stage="bits_wait", data=result, clear_intent=True)
        elif stage == "bits_wait":
            result = await bits.run(
                profile["bits_scm"], data["bits_run_id"], item["commit_sha"], data["operator"]
            )
            if result["bits_failed"]:
                await self._finish(
                    item, status="failed", stage="failed", terminal=True, data=result,
                    error="SCM 流水线构建失败，请查看原运行日志",
                )
            elif not result.get("bits_success"):
                await self._finish(item, data=result, delay=10)
            else:
                version = await scm.version(result["scm_version"])
                if version is None or version.version_id != result["scm_version_id"]:
                    raise PlatformError("BITS 产出的 SCM 版本身份不匹配")
                version.require_success(item["commit_sha"])
                await self._finish(
                    item, stage="image_submit",
                    data={**result, "build_ok": True, "scm_build_type": version.build_type,
                          "artifact_source": "bits_openapi"},
                )
        elif stage == "build_plan":
            version = await scm.next_version()
            await self._finish(item, stage="build_submit", data={"scm_version": version})
        elif stage == "build_submit":
            version_id = await self._write(
                item,
                "scm",
                {"version": data["scm_version"], "sha": item["commit_sha"]},
                lambda: scm.create(
                    commit_sha=item["commit_sha"], version=data["scm_version"], request_id=item["deployment_id"]
                ),
            )
            if version_id:
                await self._finish(item, stage="build_wait", data={"scm_version_id": version_id}, clear_intent=True)
        elif stage == "build_wait":
            version = await scm.version(data["scm_version"])
            if version is None or version.status not in {
                "build_ok",
                "build_failed",
                "build_fail",
                "failed",
                "canceled",
            }:
                await self._finish(item, delay=10)
            elif version.status != "build_ok":
                await self._fail(item, "SCM 制品构建失败，请查看精确版本日志")
            else:
                version.require_success(item["commit_sha"])
                if version.version_id != data["scm_version_id"]:
                    raise PlatformError("SCM 版本 ID 已改变")
                await self._finish(item, stage="image_submit", data={"build_ok": True})
        elif stage == "image_submit":
            version = await scm.version(data["scm_version"])
            if version is None:
                raise PlatformError("SCM 制品已不可用")
            version.require_success(item["commit_sha"])
            if version.version_id != data["scm_version_id"]:
                raise PlatformError("SCM 版本 ID 已改变，不能使用缓存制品")
            if data.get("image_status") == "ok":
                result = await self.icm.version(data, image_body(profile, data["scm_version"], data["operator"]))
                if result["image_status"] != "ok":
                    raise PlatformError("缓存镜像已不可用")
                await self._finish(item, stage=self._after_image(data))
                return
            body = image_body(profile, data["scm_version"], data["operator"])
            result = await self._write(
                item, "icm", body, lambda: self.icm.create(body, profile["image_recipe"]["image_id"])
            )
            if result:
                await self._finish(item, stage="image_wait", data=result, clear_intent=True)
        elif stage == "image_wait":
            result = await self.icm.version(data, image_body(profile, data["scm_version"], data["operator"]))
            if result["image_status"] == "fail":
                await self._finish(
                    item,
                    data=result,
                    status="failed",
                    stage="failed",
                    error="ICM 镜像构建失败，请查看精确镜像版本日志",
                    terminal=True,
                )
            else:
                await self._finish(
                    item, data=result, stage=self._after_image(data) if result["image_status"] == "ok" else stage,
                    delay=10,
                )
        elif stage == "companion_image_verify":
            await self._companion_version(data)
            artifact = await self.icm.baseline_version(profile["companion_profile"]["production_baseline"])
            await self._finish(item, stage="deploy", data={"companion_artifact": artifact})
        elif stage == "companion_image_submit":
            version = await self._companion_version(data)
            companion = profile["companion_profile"]
            body = image_body(companion, version.version, data["operator"])
            artifact = data.get("companion_artifact", {})
            if artifact.get("image_version_id"):
                await self._finish(item, stage="companion_image_wait")
                return
            result = await self._write(
                item, "companion_icm", body,
                lambda: self.icm.create(body, companion["image_recipe"]["image_id"]),
            )
            if result:
                await self._finish(
                    item, stage="companion_image_wait", clear_intent=True,
                    data={"companion_artifact": {
                        **result, "scm_version": version.version, "scm_version_id": version.version_id,
                        "commit_sha": version.commit_sha,
                    }},
                )
        elif stage == "companion_image_wait":
            version = await self._companion_version(data)
            artifact = data["companion_artifact"]
            result = await self.icm.version(
                artifact, image_body(profile["companion_profile"], version.version, data["operator"]),
            )
            if result["image_status"] == "fail":
                await self._finish(
                    item, status="failed", stage="failed", terminal=True,
                    error="配套服务 ICM 镜像构建失败，请查看精确版本",
                    data={"companion_artifact": {**artifact, **result}},
                )
            else:
                await self._finish(
                    item, stage="deploy" if result["image_status"] == "ok" else stage, delay=10,
                    data={"companion_artifact": {**artifact, **result}},
                )
        elif stage == "deploy":
            await self._deploy(item)
        elif stage in {"ticket_wait", "release_wait"}:
            resource_changes = {}
            if stage == "ticket_wait":
                env = await ppe.environment(item["name"])
                if env is not None:
                    if env.get("desc") != f"Conciliator PPE {item['environment_id']}":
                        raise PlatformError("PPE 环境归属不匹配", uncertain=True)
                    if not await self._finish(item, platform_id=str(env["id"]), keep_lease=True):
                        return
                    resource_changes = {
                        "platform_id": str(env["id"]),
                        "env_data": await ppe.resources(item["name"]),
                    }
            ticket_id = data["ticket_id" if stage == "ticket_wait" else "release_ticket_id"]
            ticket = await ppe.ticket(ticket_id)
            if ticket is None or ticket.get("status") not in {"Succeed", "Failed"}:
                if stage == "ticket_wait" and resource_changes.get("env_data"):
                    resource_changes["env_data"]["approvals"] = await ppe.approvals(
                        item["name"], resource_changes["env_data"]
                    )
                await self._finish(
                    item, delay=10, **resource_changes,
                    **({"error": ""} if ticket and ticket.get("status") else {}),
                )
            elif ticket["status"] == "Failed":
                await self._finish(
                    item,
                    status="failed",
                    stage="failed",
                    terminal=True,
                    error="PPE 工单失败，请查看原工单；重试前先核对实际资源",
                    release_failed=stage == "release_wait",
                    **({"resource_state": "cleanup_failed"} if stage == "release_wait" else {}),
                    **resource_changes,
                )
            else:
                if resource_changes.get("env_data"):
                    resource_changes["env_data"]["approvals"] = []
                next_stage = "release_verify" if stage == "release_wait" else data.get("after_ticket", "observe")
                await self._finish(item, stage=next_stage, **resource_changes)
        elif stage in {"observe", "entry_pending"}:
            result = await ppe.observe(
                item["name"],
                item["environment_id"],
                item["platform_id"],
                profile,
                image_version=data["image_version"],
                scm_version=data["scm_version"],
            )
            if result.get("absent"):
                if not await self._absent(ppe, item):
                    raise PlatformError("PPE 环境不可见但 TCE 资源仍存在，需先对账", uncertain=True)
                await self._finish(
                    item,
                    resource_state="released",
                    status="failed",
                    stage="failed",
                    error="PPE 已被平台回收，请重新部署",
                    terminal=True,
                    env_data={"clusters_ready": False},
                )
            elif result["ready"]:
                current = item["current"] and not data.get("superseded")
                await self._finish(
                    item,
                    stage="entry_pending" if current else "superseded",
                    status="deployed" if current else "superseded",
                    error="",
                    current=True,
                    terminal=True,
                    resource_state="running",
                    env_data={**result, "clusters_ready": True, "observed_at": await self._observed_now()},
                    delay=10,
                )
            elif stage == "entry_pending":
                await self._finish(
                    item,
                    error="",
                    env_data={**result, "clusters_ready": False, "observed_at": await self._observed_now()},
                    delay=10,
                )
            else:
                await self._finish(
                    item, status="running", stage="observe", env_data={**result, "clusters_ready": False}, delay=10
                )
        elif stage == "release":
            env = await ppe.environment(item["name"])
            if env is None or environment_deleted(env):
                await self._finish(item, stage="release_verify")
            else:
                if (
                    env.get("desc") != f"Conciliator PPE {item['environment_id']}"
                    or item["platform_id"]
                    and str(env["id"]) != item["platform_id"]
                ):
                    raise PlatformError("PPE 资源身份不匹配，不能释放", uncertain=True)
                resources = await ppe.resources(item["name"])
                if not await self._finish(item, platform_id=str(env["id"]), env_data=resources, keep_lease=True):
                    return
                item["platform_id"] = str(env["id"])
                result = await self._write(
                    item,
                    "release",
                    {"platform_id": item["platform_id"]},
                    lambda: ppe.remove(
                        item["name"], expected_id=item["platform_id"], identity=item["environment_id"]
                    ),
                )
                ticket_id = str((result or {}).get("id_str") or (result or {}).get("id") or "")
                if result is None:
                    return
                if not ticket_id.isdigit():
                    raise PlatformError("释放响应没有工单身份，需先对账", uncertain=True)
                await self._finish(item, stage="release_wait", data={"release_ticket_id": ticket_id}, clear_intent=True)
        elif stage == "release_verify":
            absent = await self._absent(ppe, item)
            if absent:
                await self._finish(
                    item,
                    status="released",
                    stage="released",
                    terminal=True,
                    resource_state="released",
                    env_data={"clusters_ready": False},
                )
            else:
                await self._finish(item, delay=10)

    async def _observed_now(self):
        return (await self._now()).isoformat()

    async def _now(self):
        def now():
            with self.sf() as session:
                return _database_now(session)

        return await run_in_threadpool(now)

    async def _deploy(self, item):
        data, profile = item["data"], item["data"]["profile"]
        ppe = self._client(data)
        artifact = await ScmClient(self.http, data["operator"], profile["scm_repo_id"]).version(data["scm_version"])
        if artifact is None or artifact.version_id != data["scm_version_id"]:
            raise PlatformError("SCM 制品身份已改变，不能部署")
        artifact.require_success(item["commit_sha"])
        version = await self.icm.version(data, image_body(profile, data["scm_version"], data["operator"]))
        if version["image_status"] != "ok":
            raise PlatformError("镜像尚未成功，不能部署")
        if profile.get("companion_profile"):
            companion_version = await self._companion_version(data)
            companion = data.get("companion_artifact", {})
            if (
                companion.get("scm_version") != companion_version.version
                or companion.get("scm_version_id") != companion_version.version_id
                or companion.get("commit_sha") != companion_version.commit_sha
            ):
                raise PlatformError("配套服务制品与固定基线不一致")
            baseline = profile["companion_profile"].get("production_baseline")
            if baseline:
                image = await self.icm.baseline_version(baseline)
                if any(companion.get(k) != image[k] for k in ("image_id", "image_version_id", "image_version")):
                    raise PlatformError("配套服务镜像与冻结的生产基准不一致")
            else:
                image = await self.icm.version(
                    companion, image_body(profile["companion_profile"], companion_version.version, data["operator"]),
                )
            if image["image_status"] != "ok":
                raise PlatformError("配套服务镜像尚未成功，不能部署")
        await self._deploy_environment(item, ppe)

    async def _deploy_environment(self, item, ppe):
        data = item["data"]
        env = await ppe.environment(item["name"])
        if env is not None:
            if (
                env.get("desc") != f"Conciliator PPE {item['environment_id']}"
                or item["platform_id"] and str(env["id"]) != item["platform_id"]
                or environment_deleted(env)
            ):
                raise PlatformError("PPE 环境身份已改变或已回收，请先对账")
            resources = await ppe.resources(item["name"])
            if not await self._finish(item, platform_id=str(env["id"]), env_data=resources, keep_lease=True):
                return
        body = await ppe.prepare(
            data["profile"], item["environment_id"], data["scm_version"],
            image_version=data["image_version"], operator=data["operator"], existing=env,
        )
        if body is None:
            await self._finish(item, stage="observe")
            return
        upgrades = {psm for psm, action in body["deployment"].items() if action["action"] == "upgrade"}
        completed_upgrades = set(data.get("upgraded_services", []))
        if upgrades & completed_upgrades:
            raise PlatformError("PPE 已完成本版本升级，但实际版本仍不匹配，请核对原工单和依赖；不会重复提交")
        check = precheck_summary(await ppe.precheck(body))
        ticket = await self._write(item, "environment", body, lambda: ppe.submit(body, create=env is None))
        if ticket:
            await self._finish(
                item, stage="ticket_wait", clear_intent=True, resource_state="deploying",
                data={
                    "ticket_id": ticket, "precheck": check, "after_ticket": "deploy",
                    "upgraded_services": sorted(completed_upgrades | upgrades),
                },
            )


def build_ppe_runtime(settings, service, secrets):
    from conciliator.integrations.maat.auth import SdkMaatTokenProvider, application_http_client
    from conciliator.integrations.ppe.bits import ORIGIN, BitsScmClient
    from conciliator.integrations.ppe.client import PpeClient
    from conciliator.integrations.ppe.icm import IcmClient
    from conciliator.integrations.ppe.scm import GATEWAY

    provider = SdkMaatTokenProvider(settings, secrets, proxy_url=settings.maat_auth_proxy)
    http = application_http_client(provider, GATEWAY, trust_env=False)
    detail_http = application_http_client(provider, "https://image-manager.byted.org", trust_env=False)
    bits_http = application_http_client(provider, ORIGIN, trust_env=False)
    return PpeRunner(service, http, IcmClient(http, detail_http), PpeClient(http), bits=BitsScmClient(bits_http))
