"""Machine-level multi-instance deployment protocol."""

import re
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from conciliator.repositories.models import Deployment, DeploymentWorker
from conciliator.services.aeolus_registry import InvalidTaskTransition, TaskNotFound, _database_now
from conciliator.services.deployment_counts import local_deployment_released
from conciliator.services.runtime_selection import select_runtime, task_detection
from conciliator.utils.logging import get_logger
from conciliator.utils.redaction import safe_text

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
log = get_logger(__name__)


def _log_deployment(event, record, **fields):
    log.info(
        event, workflow="meegle", task_id=record.task_id, deployment_id=record.deployment_id,
        worker_id=record.worker_id, generation=record.generation, round_no=record.round_no,
        status=record.status, stage=record.stage, **fields,
    )


def request_cleanup(record):
    if local_deployment_released(record):
        return
    record.data = {**record.data, "release_requested": "environment", "cleanup_state": "pending"}
    if not record.data.get("claimed"):
        record.status = "cancelled"
        record.data = {**record.data, "released": True, "cleanup_state": "succeeded"}
    else:
        record.status = "cancel_requested"


class MachineDeployments:
    def __init__(self, owner):
        self.owner = owner
        self.sf = owner._sf

    def register(self, body):
        if body.get("protocol_version") != 2:
            raise InvalidTaskTransition("only deployment protocol v2 is supported")
        profiles = body.get("runtime_profiles") or {}
        if not profiles:
            raise ValueError("runtime_profiles is required")
        for repository, items in profiles.items():
            if (
                not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+", repository)
                or any(part in {".", ".."} for part in repository.split("/"))
                or not items
            ):
                raise ValueError("invalid runtime repositories")
            for name, profile in items.items():
                if not ID.fullmatch(name) or not re.fullmatch(r"[0-9a-f]{64}", profile.get("profile_revision", "")):
                    raise ValueError("invalid runtime profile revision")
                if not ID.fullmatch(profile.get("runtime_revision", "")) or not re.fullmatch(
                    r"\d+\.\d+", profile.get("python_version", "")
                ):
                    raise ValueError("invalid runtime version")
                if not isinstance(profile.get("base_branches", []), list) or not all(
                    isinstance(branch, str) and branch for branch in profile.get("base_branches", [])
                ):
                    raise ValueError("invalid base branches")
                release_max = profile.get("base_release_max")
                if "base_release_max" in profile and (
                    repository != "dp/aeolus" or type(release_max) is not int or release_max < 0
                ):
                    raise ValueError("invalid base release maximum")
        host = body.get("advertised_host", "")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}", host)
            or host
            in {
                "localhost",
                "0.0.0.0",
                "::",
                "::1",
            }
            or host.startswith("127.")
        ):
            raise ValueError("advertised_host must be a routable host")
        with self.sf() as session:
            worker = session.scalar(
                select(DeploymentWorker)
                .where(
                    DeploymentWorker.worker_id == body["worker_id"],
                )
                .with_for_update()
            )
            if worker and (worker.data or {}).get("protocol_version") != 2:
                raise InvalidTaskTransition("registered worker has an unsupported deployment protocol")
            if worker and worker.data.get("advertised_host") != host:
                active = session.scalars(select(Deployment).where(Deployment.worker_id == worker.worker_id)).all()
                if any(not r.data.get("released") for r in active):
                    raise InvalidTaskTransition("cannot change host while deployments are active")
            if body.get("vmid") and session.scalar(
                select(DeploymentWorker).where(
                    DeploymentWorker.vmid == body["vmid"],
                    DeploymentWorker.worker_id != body["worker_id"],
                )
            ):
                raise InvalidTaskTransition("Devbox already registered with another worker_id")
            now = _database_now(session)
            if worker is None:
                worker = DeploymentWorker(worker_id=body["worker_id"])
                session.add(worker)
            worker.vmid = body.get("vmid")
            worker.session_id = uuid.uuid4().hex
            worker.ready = False
            worker.last_seen_at = now
            worker.data = {
                **(worker.data or {}),
                "protocol_version": 2,
                "advertised_host": host,
                "display_name": body.get("display_name") or worker.worker_id,
                "vmid": body.get("vmid"),
                "repositories": {repository: "multi" for repository in profiles},
                "runtime_profiles": profiles,
            }
            records = session.scalars(select(Deployment).where(Deployment.worker_id == worker.worker_id)).all()
            # Include acknowledged releases so a worker can recover an ACK response lost before local commit.
            expected = [self.owner._serialize(r) for r in records if r.data.get("claimed")]
            session.commit()
            log.info(
                "meegle_worker_registered", workflow="meegle", worker_id=worker.worker_id,
                deployment_count=len(expected), repository_count=len(profiles),
            )
            return {
                "session_id": worker.session_id,
                "heartbeat_seconds": 10,
                "poll_seconds": 2,
                "deployments": expected,
                "protocol_version": 2,
            }

    def enqueue(
        self,
        task_id,
        worker_id,
        request_id,
        round_no,
        commit_sha,
        runtime_profile_id=None,
        expected_profile_revision=None,
    ):
        with self.sf() as session:
            task = self.owner._task(session, task_id)
            worker = self.owner._worker(session, worker_id)
            prior = session.scalar(
                select(Deployment).where(
                    Deployment.task_id == task_id,
                    Deployment.request_id == request_id,
                )
            )
            if prior:
                if (
                    (prior.worker_id, prior.round_no, prior.commit_sha) != (worker_id, round_no, commit_sha)
                    or runtime_profile_id
                    and prior.data.get("runtime_profile_id") != runtime_profile_id
                    or expected_profile_revision
                    and prior.data.get("profile_revision") != expected_profile_revision
                ):
                    raise InvalidTaskTransition("request_id reused with different deployment parameters")
                return self.owner._serialize(prior)
            if not self.owner._online(worker, _database_now(session)):
                raise InvalidTaskTransition("worker is offline or reconciling")
            latest = self.owner._round(session, task_id)
            if (
                task.status != "review"
                or latest is None
                or (latest.round_no, latest.commit_sha) != (round_no, commit_sha)
            ):
                raise InvalidTaskTransition("code review round changed")
            if not re.fullmatch(r"[0-9a-f]{40}", commit_sha) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._/-]*",
                task.work_branch,
            ):
                raise ValueError("invalid deployment SHA or branch")
            context = (task.evaluation or {}).get("code_context") or {}
            repository = context.get("repository", "dp/aeolus")
            profiles = worker.data["runtime_profiles"].get(repository, {})
            if not profiles:
                raise ValueError("worker does not support this repository")
            detection = task_detection(task, latest)
            name = select_runtime(profiles, context.get("base_branch", ""), runtime_profile_id, detection)
            profile = profiles[name]
            if expected_profile_revision and profile["profile_revision"] != expected_profile_revision:
                raise InvalidTaskTransition("运行配置已变更，请刷新后重新选择")
            old = session.scalars(select(Deployment).where(Deployment.task_id == task_id).with_for_update()).all()
            if any(r.data.get("protocol_version") == 2 and not local_deployment_released(r) for r in old):
                raise InvalidTaskTransition("请先释放本任务的旧部署；同机其他任务不受影响")
            record = Deployment(
                deployment_id=uuid.uuid4().hex,
                task_id=task_id,
                worker_id=worker_id,
                request_id=request_id,
                round_no=round_no,
                commit_sha=commit_sha,
                generation=1,
                status="running",
                stage="dispatching",
                created_at=datetime.now(UTC).replace(tzinfo=None),
                data={
                    "protocol_version": 2,
                    "repository": repository,
                    "branch": task.work_branch,
                    "runtime_profile_id": name,
                    "runtime_detection": detection,
                    **profile,
                    "display_name": worker.data["display_name"],
                    "vmid": worker.vmid,
                    "service_host": worker.data["advertised_host"],
                    "maat_url": "",
                    "cleanup_state": "none",
                },
            )
            session.add(record)
            session.commit()
            _log_deployment("meegle_devbox_enqueued", record, commit_sha=commit_sha)
            return self.owner._serialize(record)

    def claim(self, worker_id, body):
        # Do not hold the worker lock while acquiring a task lock.
        with self.sf() as session:
            worker = session.get(DeploymentWorker, worker_id)
            if not worker or worker.data.get("protocol_version") != 2 or worker.session_id != body["session_id"]:
                raise InvalidTaskTransition("stale worker session")
            rows = session.scalars(
                select(Deployment)
                .where(
                    Deployment.worker_id == worker_id,
                )
                .order_by(Deployment.created_at, Deployment.deployment_id)
            ).all()
            candidate = next((r for r in rows if r.data.get("claim_request") == body["request_id"]), None)
            candidate = candidate or next(
                (r for r in rows if r.status == "running" and not r.data.get("claimed")), None
            )
            if candidate is None:
                return None
            task = self.owner._task(session, candidate.task_id)
            record = session.scalar(
                select(Deployment)
                .where(
                    Deployment.deployment_id == candidate.deployment_id,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            worker = self.owner._worker(session, worker_id, body["session_id"])
            if not self.owner._online(worker, _database_now(session)):
                return None
            if record.data.get("claimed"):
                return self.owner._serialize(record) if record.data.get("claim_request") == body["request_id"] else None
            latest = self.owner._round(session, task.task_id)
            profile = (
                worker.data["runtime_profiles"]
                .get(record.data["repository"], {})
                .get(record.data["runtime_profile_id"], {})
            )
            if (
                record.status != "running"
                or task.status != "review"
                or latest is None
                or (latest.round_no, latest.commit_sha) != (record.round_no, record.commit_sha)
                or profile.get("profile_revision") != record.data["profile_revision"]
            ):
                request_cleanup(record)
                session.commit()
                return None
            record.stage = "checkout"
            record.data = {**record.data, "claimed": True, "claim_request": body["request_id"]}
            session.commit()
            _log_deployment("meegle_devbox_claimed", record)
            return self.owner._serialize(record)

    def _record(self, session, worker_id, deployment_id, body):
        initial = session.get(Deployment, deployment_id)
        if initial is None:
            raise TaskNotFound(deployment_id)
        task = self.owner._task(session, initial.task_id)
        record = session.scalar(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
            .with_for_update().execution_options(populate_existing=True)
        )
        worker = self.owner._worker(session, worker_id, body["session_id"])
        if (
            record.data.get("protocol_version") != 2
            or worker.data.get("protocol_version") != 2
            or record.worker_id != worker_id
            or record.generation != body["generation"]
        ):
            raise InvalidTaskTransition("stale deployment identity")
        if not record.data.get("claimed"):
            raise InvalidTaskTransition("deployment has not been claimed")
        return task, record, worker

    def heartbeat(self, worker_id, body):
        observations = body.get("instances", [])
        # Each observation updates only its instance. No missing-entry inference.
        for item in observations:
            with self.sf() as session:
                _, record, _ = self._record(
                    session, worker_id, item["deployment_id"], {**item, "session_id": body["session_id"]}
                )
                previous = record.stage, (record.data.get("observation") or {}).get("service_status")
                if not record.data.get("released"):
                    record.data = {
                        **record.data,
                        "observation": {
                            "service_status": item.get("service_status", "unknown"),
                            "stage": item.get("stage", ""),
                            "observed_at": _database_now(session).isoformat(),
                        },
                    }
                    if record.status == "running" and not record.data.get("endpoint"):
                        record.stage = item.get("stage") or record.stage
                session.commit()
                health = (record.data.get("observation") or {}).get("service_status")
                if previous != (record.stage, health):
                    _log_deployment(
                        "meegle_devbox_observation_changed", record, previous_stage=previous[0],
                        previous_service_status=previous[1], service_status=health,
                    )
        with self.sf() as session:
            worker = self.owner._worker(session, worker_id, body["session_id"])
            was_ready = worker.ready
            worker.last_seen_at = _database_now(session)
            records = session.scalars(select(Deployment).where(Deployment.worker_id == worker_id)).all()
            if body.get("reconciled"):
                expected = {
                    (r.deployment_id, r.generation)
                    for r in records
                    if r.data.get("claimed") and not r.data.get("released")
                }
                observed = {(i["deployment_id"], i["generation"]) for i in observations}
                if expected != observed:
                    raise InvalidTaskTransition("worker journals do not match active deployments")
                worker.ready = True
            commands = [
                {"deployment_id": r.deployment_id, "generation": r.generation, "command": "release"}
                for r in records
                if r.data.get("claimed") and r.data.get("release_requested") and not r.data.get("released")
                and (r.data.get("stop_ack") != r.generation or r.data.get("maat_cleanup_verified"))
            ]
            session.commit()
            if worker.ready != was_ready:
                log.info("meegle_worker_ready", workflow="meegle", worker_id=worker_id, ready=worker.ready)
            return {"ready": worker.ready, "commands": commands}

    def endpoint_ready(self, worker_id, deployment_id, body):
        with self.sf() as session:
            task, record, worker = self._record(session, worker_id, deployment_id, body)
            endpoint = {
                k: body[k]
                for k in (
                    "commit_sha",
                    "runtime_profile_id",
                    "profile_revision",
                    "runtime_revision",
                    "endpoint_host",
                    "endpoint_port",
                    "runtime_attempt_id",
                    "request_id",
                    "log_path",
                )
            }
            prior = record.data.get("endpoint")
            if prior:
                if prior != endpoint:
                    raise InvalidTaskTransition("conflicting endpoint-ready replay")
                return {"accepted": not bool(record.data.get("release_requested"))}
            if (
                body["commit_sha"] != record.commit_sha
                or any(
                    body[k] != record.data[k] for k in ("runtime_profile_id", "profile_revision", "runtime_revision")
                )
                or body["endpoint_host"] != worker.data["advertised_host"]
            ):
                raise InvalidTaskTransition("runtime does not match deployment")
            others = session.scalars(select(Deployment).where(Deployment.worker_id == worker_id)).all()
            if any(
                r.deployment_id != deployment_id
                and not r.data.get("port_released")
                and r.data.get("service_port") == body["endpoint_port"]
                for r in others
            ):
                raise InvalidTaskTransition("endpoint port is already allocated")
            latest = self.owner._round(session, task.task_id)
            valid = (
                task.status == "review"
                and latest is not None
                and (latest.round_no, latest.commit_sha) == (record.round_no, record.commit_sha)
                and not record.data.get("release_requested")
            )
            record.data = {
                **record.data,
                "endpoint": endpoint,
                "service_port": body["endpoint_port"],
                "observation": {
                    "service_status": "healthy",
                    "stage": "local_ready",
                    "observed_at": _database_now(session).isoformat(),
                },
            }
            if valid:
                # No undocumented devEnv/cleanup calls: local readiness cannot become global success.
                record.stage = "maat_contract_required"
                record.data = {
                    **record.data,
                    "error": "",
                }
            else:
                request_cleanup(record)
            session.commit()
            _log_deployment("meegle_devbox_endpoint_ready", record, accepted=valid, port=body["endpoint_port"])
            return {"accepted": valid}

    def result(self, worker_id, deployment_id, body):
        if body["status"] != "failed":
            raise InvalidTaskTransition("v2 success requires endpoint-ready and verified MAAT provisioning")
        with self.sf() as session:
            task, record, _ = self._record(session, worker_id, deployment_id, body)
            if local_deployment_released(record):
                return {"accepted": True, "notify": False, "task_id": task.task_id}
            payload = {k: body.get(k) for k in ("status", "commit_sha", "log_path")}
            payload["error"] = safe_text(body.get("error", ""))[:1000]
            if record.data.get("result"):
                if record.data["result"] != payload:
                    raise InvalidTaskTransition("conflicting result")
                return {"accepted": True, "notify": False, "task_id": task.task_id}
            record.data = {**record.data, "result": payload, "error": payload["error"]}
            request_cleanup(record)
            session.commit()
            _log_deployment("meegle_devbox_result", record, result_status=body["status"])
            return {"accepted": True, "notify": True, "task_id": task.task_id}

    def release(self, task_id, deployment_id, generation):
        with self.sf() as session:
            self.owner._task(session, task_id)
            record = session.scalar(
                select(Deployment).where(Deployment.deployment_id == deployment_id).with_for_update()
            )
            if (record is None or record.data.get("protocol_version") != 2
                    or record.task_id != task_id or record.generation != generation):
                raise InvalidTaskTransition("stale deployment identity")
            requested = bool(record.data.get("release_requested"))
            request_cleanup(record)
            session.commit()
            if not requested:
                _log_deployment("meegle_devbox_release_requested", record)
            return self.owner._serialize(record)

    def stopped(self, worker_id, body):
        with self.sf() as session:
            task, record, _ = self._record(session, worker_id, body["deployment_id"], body)
            if record.data.get("stop_ack") == body["generation"] and record.data.get("port_released"):
                return {"accepted": True, "port_released": bool(record.data.get("port_released"))}
            if not record.data.get("release_requested"):
                raise InvalidTaskTransition("no stop requested")
            route_clear = not record.data.get("maat_write_intent") and (
                record.data.get("maat_cleanup_verified")
                or not record.data.get("maat_api_id") and not record.data.get("maat_env_id")
            )
            if record.data.get("stop_ack") == body["generation"] and not route_clear:
                return {"accepted": True, "port_released": False}
            record.data = {
                **record.data,
                "stop_ack": body["generation"],
                "port_released": route_clear,
                "released": route_clear,
                "cleanup_state": "succeeded" if route_clear else (
                    "manual_required" if record.data.get("maat_write_intent") or not record.data.get("maat_context")
                    else "pending"
                ),
            }
            record.status = "failed" if record.data.get("result") else "cancelled"
            record.stage = "released" if route_clear else "cleanup_pending"
            session.commit()
            _log_deployment(
                "meegle_devbox_stopped", record, port_released=route_clear,
                cleanup_state=record.data["cleanup_state"],
            )
            return {"accepted": True, "notify_task_id": task.task_id, "port_released": route_clear}
