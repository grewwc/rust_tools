"""Small pull-runner POC. Worker identity authentication is intentionally deferred."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from conciliator.repositories.models import Deployment, DeploymentWorker, MeegleAINodeRound, MeegleAINodeTask
from conciliator.services.aeolus_registry import InvalidTaskTransition, TaskNotFound, _database_now
from conciliator.services.deployment_counts import deployment_counts, local_deployment_released, v2_entry_ready
from conciliator.services.deployment_multi import MachineDeployments, request_cleanup
from conciliator.services.runtime_selection import runtime_choices, task_detection

DEFAULT_REPOSITORY = "dp/aeolus"
OFFLINE_SECONDS = 30


def invalidate_deployments(session, task_id: str) -> None:
    """Called under the task lock, in the same transaction as code feedback/completion."""
    records = list(session.scalars(select(Deployment).where(Deployment.task_id == task_id).with_for_update()))
    for record in records:
        if record.data.get("protocol_version") == 2:
            request_cleanup(record)


class DeploymentService:
    def __init__(self, session_factory, *, runtime_detector=None):
        self._sf = session_factory
        self.runtime_detector = runtime_detector
        self.multi = MachineDeployments(self)

    async def run(self, operation: str, *args):
        if self.runtime_detector is not None and operation in {"snapshot", "enqueue"}:
            await self.runtime_detector.ensure(args[0])
        if operation == "snapshot":
            return await run_in_threadpool(self._snapshot, *args)
        return await run_in_threadpool(getattr(self.multi, operation), *args)

    @staticmethod
    def _worker(session, worker_id, session_id=None):
        worker = session.scalar(
            select(DeploymentWorker)
            .where(
                DeploymentWorker.worker_id == worker_id,
            )
            .with_for_update()
        )
        if worker is None or worker.data.get("protocol_version") != 2:
            raise TaskNotFound("worker")
        if session_id is not None and worker.session_id != session_id:
            raise InvalidTaskTransition("stale worker session")
        return worker

    @staticmethod
    def _task(session, task_id):
        task = session.scalar(
            select(MeegleAINodeTask)
            .where(
                MeegleAINodeTask.task_id == task_id,
            )
            .with_for_update()
        )
        if task is None:
            raise TaskNotFound(task_id)
        return task

    @staticmethod
    def _round(session, task_id):
        return session.scalar(
            select(MeegleAINodeRound)
            .where(
                MeegleAINodeRound.task_id == task_id,
            )
            .order_by(MeegleAINodeRound.round_no.desc())
            .limit(1)
        )

    @staticmethod
    def _online(worker, now):
        return worker.ready and now - worker.last_seen_at <= timedelta(seconds=OFFLINE_SECONDS)


    def _snapshot(self, task_id):
        with self._sf() as session:
            task = session.scalar(select(MeegleAINodeTask).where(MeegleAINodeTask.task_id == task_id))
            if task is None:
                raise TaskNotFound(task_id)
            now = _database_now(session)
            repository = ((task.evaluation or {}).get("code_context") or {}).get("repository", DEFAULT_REPOSITORY)
            detection = task_detection(task, self._round(session, task_id))
            workers = []
            for worker in session.scalars(select(DeploymentWorker).order_by(DeploymentWorker.worker_id)):
                if worker.data.get("protocol_version") != 2:
                    continue
                profiles = worker.data.get("runtime_profiles", {}).get(repository, {})
                if not profiles:
                    continue
                item = {
                    "worker_id": worker.worker_id,
                    "display_name": worker.data["display_name"],
                    "vmid": worker.vmid,
                    "online": self._online(worker, now),
                    "supports_repository": True,
                    "protocol_version": 2,
                    "runtime_profiles": profiles,
                }
                records = list(session.scalars(select(Deployment).where(Deployment.worker_id == worker.worker_id)))
                item["deployment_counts"] = deployment_counts(
                    records, observations_fresh=self._online(worker, now), now=now,
                )
                item["counts_observed_at"] = worker.last_seen_at.isoformat()
                item.update(runtime_choices(
                    profiles, detection, ((task.evaluation or {}).get("code_context") or {}).get("base_branch", ""),
                ))
                workers.append(item)
            latest = session.scalar(
                select(Deployment)
                .where(
                    Deployment.task_id == task_id,
                    Deployment.data["protocol_version"].as_integer() == 2,
                )
                .order_by(Deployment.created_at.desc(), Deployment.deployment_id.desc())
                .limit(1)
            )
            deployment: dict[str, Any] | None = None
            if latest:
                deployment = self._serialize(latest)
                worker = session.get(DeploymentWorker, latest.worker_id)
                if worker and not self._online(worker, now) and latest.status == "running":
                    deployment["status"] = "unknown"
                deployment["resource_owned"] = not local_deployment_released(latest)
                deployment["resource_releasing"] = bool(
                    latest.data.get("release_requested") and not local_deployment_released(latest)
                )
                deployment["released"] = local_deployment_released(latest)
                deployment["can_release"] = deployment["resource_owned"] and not deployment["resource_releasing"]
                latest_round = self._round(session, task_id)
                deployment["current"] = bool(
                    worker and self._online(worker, now) and latest.status == "succeeded"
                    and task.status == "review" and latest_round
                    and (latest_round.round_no, latest_round.commit_sha) == (latest.round_no, latest.commit_sha)
                    and v2_entry_ready(latest.data, now)
                )
            if deployment:
                labels = {"running": "部署中", "unknown": "部署状态待确认",
                          "cancel_requested": "停止中", "cancelled": "部署已取消", "failed": "部署失败",
                          "succeeded": "部署成功"}
                deployment["status_label"] = labels.get(deployment["status"], deployment["status"])
                deployment["status_note"] = ""
                if latest.stage == "maat_contract_required":
                    deployment["status_label"] = "服务已启动，等待 MAAT 入口"
                    deployment["status_note"] = "本机服务已就绪，等待 MAAT 入口配置完成，请稍候。"
                if latest.stage == "maat_entry_pending":
                    deployment["status_label"] = "MAAT 环境已创建，入口待验证"
                    if latest.data.get("maat_environment_status") in {"deploy_file_ok", "deploy_static_ok"}:
                        deployment["status_label"] = "MAAT 环境部署完成，待业务验证"
                        deployment["status_note"] = "产物已部署，可打开入口进行业务验证；尚未验证通过。"
                    else:
                        deployment["status_note"] = "可打开入口检查；产物部署或实际路由尚未确认，暂不能标记验证通过。"
                if latest.stage == "maat_deployment_failed":
                    deployment["status_label"] = "MAAT 构建或产物部署失败"
                    deployment["status_note"] = "本机服务保留；请检查 MAAT 环境，或释放本次部署后重试。"
                if deployment["resource_releasing"]:
                    deployment["status_label"] = "释放中"
                    deployment["status_note"] = "等待 worker 停服确认；离线时须等待恢复，占用暂不清除。"
                    deployment["status_note"] = latest.data.get("error") or deployment["status_note"]
                elif deployment["released"]:
                    deployment["status_label"] = "环境已释放"
                    deployment["status_note"] = "部署日志保留；可在代码 Review 阶段重新选择 Devbox 部署。"
                    if latest.data.get("maat_phase") == "cleanup_records_deleted":
                        deployment["status_note"] = (
                            "服务和实例代码、MAAT 环境及代理记录已清理；"
                            "旧端口保留预留，可重新部署或继续 Coding。"
                        )
                    elif latest.data.get("cleanup_state") in {"pending", "running"}:
                        deployment["status_label"] = "本机已释放，MAAT 清理中"
                        deployment["status_note"] = (
                            "服务和实例代码已清理；正在自动清理 MAAT 环境和代理记录，"
                            "旧端口保留预留，可重新部署或继续 Coding。"
                        )
                    elif latest.data.get("cleanup_state") == "manual_required":
                        deployment["status_label"] = "本机已释放，MAAT 待清理"
                        deployment["status_note"] = (
                            "服务和实例代码已清理；MAAT 资源待人工清理，"
                            "旧端口保留预留，可重新部署或继续 Coding。"
                        )
                        if not latest.data.get("result"):
                            deployment["error"] = ""
                elif deployment.get("superseded"):
                    deployment["status_label"] = "旧版本部署"
                    deployment["status_note"] = (
                        "旧服务仍可能运行，不能验证最新代码；可手动释放或在新代码 Review 后重新部署。"
                    )
                elif deployment["status"] == "succeeded" and not deployment["current"]:
                    deployment["status_note"] = "当前环境不可用于验证，请检查 worker 在线状态与服务健康。"
                elif deployment["status"] == "failed" and deployment["resource_owned"]:
                    deployment["status_note"] = "部署失败但 Devbox 仍被保留；可重试部署或手动释放。"
            return {"deployment": deployment, "deployment_workers": workers}

    @staticmethod
    def _serialize(record) -> dict[str, Any]:
        return {
            "deployment_id": record.deployment_id,
            "task_id": record.task_id,
            "worker_id": record.worker_id,
            "round_no": record.round_no,
            "commit_sha": record.commit_sha,
            "generation": record.generation,
            "status": record.status,
            "stage": record.stage,
            "deployment_failed": bool(record.data.get("result")),
            "created_at": record.created_at.isoformat(),
            "log_path": (
                (record.data.get("result") or {}).get("log_path")
                or (record.data.get("endpoint") or {}).get("log_path", "")
            ),
            **{
                key: record.data.get(key)
                for key in (
                    "repository",
                    "profile_revision",
                    "branch",
                    "maat_url",
                    "service_port",
                    "display_name",
                    "vmid",
                    "error",
                    "released",
                    "verified",
                    "superseded",
                    "protocol_version",
                    "runtime_profile_id",
                    "runtime_revision",
                    "python_version",
                    "profile_revision",
                    "service_host",
                    "cleanup_state",
                    "maat_environment_status",
                )
            },
        }
