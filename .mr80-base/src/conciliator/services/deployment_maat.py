"""Durable, one-operation-at-a-time MAAT orchestration for v2 deployments.

Remote writes run outside database transactions. An intent without a recorded response is
quarantined rather than replayed, including after a server restart. Authentication and the
application-specific end-to-end probe are supplied by the operator's integration.
"""

import asyncio
import copy
import hashlib
import json
import uuid
from datetime import UTC, datetime
from urllib.parse import urlsplit

from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from conciliator.integrations.maat.client import MaatError, service_environment_plan, template_revision
from conciliator.integrations.maat.control_plane import (
    check_environment_identity,
    environment_body,
    environment_url,
    routes_match_endpoint,
)
from conciliator.repositories.models import Deployment
from conciliator.services.aeolus_registry import _database_now
from conciliator.services.deployment_counts import devbox_result_kind, v2_entry_ready
from conciliator.services.deployment_multi import request_cleanup
from conciliator.utils.logging import get_logger
from conciliator.utils.meegle_logging import logged_step, workflow_step

log = get_logger(__name__)


def _log_maat(row, event, **fields):
    log.info(
        event, workflow="meegle", task_id=row.task_id, deployment_id=row.deployment_id,
        worker_id=row.worker_id, status=row.status, stage=row.stage,
        phase=row.data.get("maat_phase"), **fields,
    )


class MaatDeploymentCoordinator:
    def __init__(
        self, service, *, openapi, control_plane, baseline_provider, baseline_revision,
        service_bindings, proxy_options, probe=None, openapi_only=False,
    ):
        if not baseline_revision:
            raise ValueError("MAAT requires a reviewed baseline")
        self.service = service
        self.sf = service._sf
        self.openapi = openapi
        self.control = control_plane
        self.baseline_provider = baseline_provider
        self.baseline_revision = baseline_revision
        self.bindings = service_bindings
        self.proxy_options = proxy_options
        self.probe = probe
        self.openapi_only = openapi_only

    def _load(self, deployment_id=None):
        with self.sf() as session:
            query = select(Deployment)
            if deployment_id:
                query = query.where(Deployment.deployment_id == deployment_id)
            return [
                {
                    "deployment_id": row.deployment_id, "worker_id": row.worker_id, "generation": row.generation,
                    "task_id": row.task_id, "commit_sha": row.commit_sha, "status": row.status,
                    "stage": row.stage, "data": copy.deepcopy(row.data),
                }
                for row in session.scalars(query)
                if row.data.get("protocol_version") == 2 and row.data.get("endpoint") and not row.data.get("released")
            ]

    def _locked(self, session, deployment_id):
        initial = session.get(Deployment, deployment_id)
        if initial is None:
            raise MaatError("MAAT deployment identity is unavailable")
        # Match worker callbacks/feedback lock ordering: task, deployment, worker.
        self.service._task(session, initial.task_id)
        row = session.scalar(
            select(Deployment).where(Deployment.deployment_id == deployment_id)
            .with_for_update().execution_options(populate_existing=True)
        )
        if row is None or row.data.get("protocol_version") != 2:
            raise MaatError("MAAT deployment identity is unavailable")
        endpoint = row.data.get("endpoint") or {}
        if not endpoint.get("endpoint_host") or endpoint["endpoint_host"] != row.data.get("service_host"):
            raise MaatError("worker endpoint does not match the deployment host")
        return row

    def _begin(self, deployment_id, phase, body, context):
        with self.sf() as session:
            row = self._locked(session, deployment_id)
            data = row.data
            if data.get("released") or data.get("maat_write_intent") or data.get("maat_phase", "proxy") != phase:
                return None
            if not phase.startswith("cleanup"):
                task = self.service._task(session, row.task_id)
                latest = self.service._round(session, row.task_id)
                if (
                    data.get("release_requested") or task.status != "review" or latest is None
                    or (latest.round_no, latest.commit_sha) != (row.round_no, row.commit_sha)
                ):
                    request_cleanup(row)
                    session.commit()
                    return None
                now = _database_now(session)
                worker = self.service._worker(session, row.worker_id)
                if not self.service._online(worker, now) or not v2_entry_ready(
                    {**data, "maat_entry_verified": True}, now,
                ):
                    return None
            if data.get("maat_context") and data["maat_context"] != context:
                raise MaatError("MAAT deployment context changed")
            token = uuid.uuid4().hex
            row.data = {
                **data, "maat_context": context,
                "maat_started_at": data.get("maat_started_at") or _database_now(session).isoformat(),
                "maat_write_intent": {
                    "id": token, "phase": phase, "at": _database_now(session).isoformat(),
                    "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                },
            }
            row.stage = "maat_" + phase
            session.commit()
            _log_maat(row, "meegle_maat_write_intent", operation=phase, intent_id=token)
            return token

    def _finish(self, deployment_id, token, phase, *, field=None, resource_id=None, unknown=False):
        with self.sf() as session:
            row = self._locked(session, deployment_id)
            intent = row.data.get("maat_write_intent") or {}
            data = {**row.data}
            if field and resource_id:
                if data.get(field) and data[field] != resource_id:
                    raise MaatError("conflicting MAAT resource identities require reconciliation")
                data[field] = resource_id
            if intent.get("id") != token:
                unknown = True
            if unknown:
                data["cleanup_state"] = "manual_required"
                data["error"] = (
                    "MAAT 清理结果待确认，旧端口继续保留预留，请维护者核对原请求。"
                    if phase.startswith("cleanup")
                    else "MAAT 写入结果待确认，服务将停止；入口与端口保留，等待对账。"
                )
                row.data = data
                request_cleanup(row)
                row.stage = "maat_cleanup_unknown" if phase.startswith("cleanup") else "maat_write_unknown"
            else:
                row.data = {**data, "maat_phase": phase, "maat_write_intent": None, "error": ""}
                row.stage = "maat_" + phase
            session.commit()
            _log_maat(row, "meegle_maat_write_recorded", unknown=unknown, resource_id=resource_id)

    async def _write(self, record, phase, body, context, call, next_phase, *, field=None):
        deployment_id = record["deployment_id"]
        token = await run_in_threadpool(self._begin, deployment_id, phase, body, context)
        if token is None:
            return
        try:
            with logged_step(
                "maat_write", task_id=record["task_id"], deployment_id=deployment_id,
                operation=phase, intent_id=token,
            ):
                result = await call()
        except Exception as exc:
            await run_in_threadpool(
                self._finish, deployment_id, token, phase, field=field,
                resource_id=getattr(exc, "resource_id", None), unknown=True,
            )
            return
        await run_in_threadpool(
            self._finish, deployment_id, token, next_phase, field=field,
            resource_id=(result.get("_id") if isinstance(result, dict) else result) if field else None,
        )

    def _phase(
        self, deployment_id, phase, *, ready_url=None, pending_url=None, cleanup_verified=False,
        environment_status=None,
    ):
        with self.sf() as session:
            row = self._locked(session, deployment_id)
            previous = row.status, row.stage, row.data.get("maat_phase"), row.data.get("maat_environment_status")
            if row.data.get("released") or row.data.get("maat_write_intent"):
                return
            if ready_url or pending_url:
                task = self.service._task(session, row.task_id)
                latest = self.service._round(session, row.task_id)
                if (
                    row.data.get("release_requested") or task.status != "review" or latest is None
                    or (latest.round_no, latest.commit_sha) != (row.round_no, row.commit_sha)
                ):
                    request_cleanup(row)
                    session.commit()
                    return
                worker = self.service._worker(session, row.worker_id)
                now = _database_now(session)
                if not self.service._online(worker, now) or not v2_entry_ready(
                    {**row.data, "maat_entry_verified": True}, now,
                ):
                    return
                if ready_url:
                    row.status = "succeeded"
                    row.stage = "ready"
            row.data = {
                **row.data, "maat_phase": phase,
                **({"maat_url": ready_url, "maat_entry_verified": True, "error": ""} if ready_url else {}),
                **({"maat_url": pending_url, "error": ""} if pending_url else {}),
                **({"maat_cleanup_verified": True, "cleanup_state": "pending"} if cleanup_verified else {}),
                **({"maat_environment_status": environment_status} if environment_status else {}),
            }
            if phase == "cleanup_records_deleted":
                # Soft deletion does not verify route withdrawal or release the worker's port ledger.
                row.data = {**row.data, "cleanup_state": "succeeded", "error": ""}
                row.stage = "maat_cleanup_records_deleted"
            if pending_url:
                row.stage = "maat_entry_pending"
                if environment_status in {
                    "trigger_build_failed", "build_failed", "deploy_static_failed", "deploy_file_failed",
                }:
                    row.status = "failed"
                    row.stage = "maat_deployment_failed"
                    row.data = {
                        **row.data, "maat_phase": "failed",
                        "error": "MAAT 构建或产物部署失败，请检查 MAAT 环境；本机服务保留，可手动释放。",
                    }
            # Only the worker's subsequent stopped ACK can release the local port ledger.
            session.commit()
            if previous != (
                row.status, row.stage, row.data.get("maat_phase"), row.data.get("maat_environment_status")
            ):
                _log_maat(
                    row, "meegle_maat_state_changed", previous_status=previous[0], previous_stage=previous[1],
                    environment_status=environment_status, cleanup_verified=cleanup_verified,
                )

    async def run_once(self):
        records = await run_in_threadpool(self._load)
        results = await asyncio.gather(*(self.advance(row["deployment_id"]) for row in records), return_exceptions=True)
        before = {row["deployment_id"]: row for row in records}
        after = await run_in_threadpool(self._load)
        def state(row):
            return devbox_result_kind({**row["data"], "status": row["status"], "stage": row["stage"]})
        # Only completed deployments and actionable failures need a message.
        task_ids = {
            row["task_id"] for row in after
            if row["deployment_id"] in before and state(row) != state(before[row["deployment_id"]])
            and state(row) is not None
        }
        return [*results, *sorted(task_ids)]

    def _expire(self, deployment_id):
        with self.sf() as session:
            row = self._locked(session, deployment_id)
            if row.data.get("maat_phase") != "ready" and not row.data.get("released"):
                requested = bool(row.data.get("release_requested"))
                row.data = {**row.data, "error": "MAAT 入口验证超时，正在释放本次部署。"}
                request_cleanup(row)
                session.commit()
                if not requested:
                    _log_maat(row, "meegle_maat_expired")

    @staticmethod
    def _age(stamp):
        time = datetime.fromisoformat(stamp)
        time = time.replace(tzinfo=UTC) if time.tzinfo is None else time.astimezone(UTC)
        return (datetime.now(UTC) - time).total_seconds()

    async def advance(self, deployment_id):
        records = await run_in_threadpool(self._load, deployment_id)
        if not records:
            return
        await self._advance_record(records[0])

    @workflow_step("maat_advance", level="debug")
    async def _advance_record(self, record):
        deployment_id = record["deployment_id"]
        data = record["data"]
        if data.get("maat_write_intent"):
            # Another process may still be sending, or may have died after sending. Never replay.
            intent = data["maat_write_intent"]
            if self._age(intent["at"]) > 60:
                await run_in_threadpool(
                    self._finish, deployment_id, intent["id"], intent["phase"], unknown=True,
                )
            return
        phase = data.get("maat_phase", "proxy")
        if data.get("release_requested"):
            if self.openapi_only:
                return await self._cleanup_public(record, phase)
            return await self._cleanup(record, phase)
        if phase in {"ready", "failed"}:
            return
        waiting_for_verification = self.openapi_only and phase == "verify" and bool(data.get("maat_url"))
        if not waiting_for_verification and data.get("maat_started_at") and self._age(data["maat_started_at"]) > 600:
            await run_in_threadpool(self._expire, deployment_id)
            return
        template, project_config = await self.baseline_provider()
        if template_revision(template) != self.baseline_revision:
            raise MaatError("MAAT baseline changed; review before continuing")
        endpoint = data["endpoint"]
        name = f"hua-ai-{deployment_id[:12]}-g{record['generation']}"
        context = {
            "name": name, "projectId": template["projectId"], "projectName": template["projectName"],
            "region": template["region"], "baseline_revision": self.baseline_revision,
            "proxy_options": self.proxy_options[data["repository"]],
        }
        expected_env = {**{k: context[k] for k in ("name", "projectId", "projectName", "region")},
                        "_id": data.get("maat_env_id")}
        if phase == "proxy":
            if self.openapi_only and await self.control.lookup_environment(
                template["projectName"], template["region"], name,
            ) is not None:
                await run_in_threadpool(self._name_conflict, deployment_id)
                return
            body = {
                "project_id": context["projectId"], "name": name,
                "host": endpoint["endpoint_host"], "port": endpoint["endpoint_port"],
                **context["proxy_options"],
            }
            return await self._write(
                record, phase, body, context, lambda: self.openapi.create_dev_api(**body), "environment",
                field="maat_api_id",
            )
        plan = service_environment_plan(
            template, data["repository"], data["maat_api_id"], bindings=self.bindings,
            expected_template_revision=self.baseline_revision,
        )
        body = environment_body(template, plan, name=name, action_user=self.control.action_user)
        if phase == "environment":
            return await self._write(
                record, phase, body, context, lambda: self.control.create_environment(body),
                "verify" if self.openapi_only else "apply",
                field="maat_env_id",
            )
        if self.openapi_only and phase == "apply":
            # Older records may resume here. Creation already submitted the complete configuration.
            # Updating the same body triggers another deploy and can collide with the first one.
            await run_in_threadpool(self._phase, deployment_id, "verify")
            return
        operations = {
            "apply": (lambda: self.control.apply_environment(expected_env, body), "files", body),
            "files": (lambda: self.control.deploy_files(expected_env), "gateway", {"_id": expected_env["_id"]}),
            "gateway": (lambda: self.control.deploy_gateway(expected_env), "verify", {"_id": expected_env["_id"]}),
        }
        if phase in operations:
            call, next_phase, request = operations[phase]
            return await self._write(record, phase, request, context, call, next_phase)
        if phase == "verify":
            current = await self.control.environment_record(expected_env)
            check_environment_identity(current, expected_env, self.control.action_user)
            if self.openapi_only:
                if current.get("isDeleted"):
                    raise MaatError("created environment has been deleted")
                expected_rules = {
                    (r["pathname"], r["pathMatchType"]): r["apiId"] for r in plan["api_proxy_list"]
                }
                observed_rules = {
                    (r.get("pathname"), r.get("pathMatchType")): r.get("apiId")
                    for r in current.get("apiProxyList", [])
                }
                expected_scm = [
                    (r.get("scmId"), r.get("scmVersion"), r.get("path") or "/") for r in plan["scm_list"]
                ]
                observed_scm = [
                    (r.get("scmId"), r.get("scmVersion"), r.get("path") or "/")
                    for r in current.get("scmList", [])
                ]
                if (
                    expected_rules != observed_rules or len(current.get("apiProxyList", [])) != len(expected_rules)
                    or expected_scm != observed_scm
                ):
                    return
                await run_in_threadpool(
                    self._phase, deployment_id, "verify", pending_url=environment_url(current, project_config),
                    environment_status=current.get("status"),
                )
                return
            if current.get("isDeleted") or current.get("status") != "deploy_file_ok":
                return
            url = environment_url(current, project_config)
            domain = urlsplit(url).hostname
            routes = await self.control.domain_routes(domain)
            if not routes_match_endpoint(
                routes, self.bindings[data["repository"]], domain=domain,
                host=endpoint["endpoint_host"], port=endpoint["endpoint_port"],
                protocol=context["proxy_options"]["protocol"],
            ):
                return
            if self.probe is not None and await self.probe(url, copy.deepcopy(record)) is True:
                await run_in_threadpool(self._phase, deployment_id, "ready", ready_url=url)

    async def _cleanup_public(self, record, phase):
        data, deployment_id = record["data"], record["deployment_id"]
        context = data.get("maat_context")
        if (
            not context or data.get("stop_ack") != record["generation"]
            or phase == "cleanup_records_deleted"
        ):
            return
        if not phase.startswith("cleanup"):
            await run_in_threadpool(self._phase, deployment_id, "cleanup_environment")
            return
        expected = {**{k: context[k] for k in ("name", "projectId", "projectName", "region")},
                    "_id": data.get("maat_env_id")}
        if phase in {"cleanup_environment", "cleanup_environment_verify"}:
            current = await self.control.lookup_environment(
                context["projectName"], context["region"], context["name"],
            ) if expected["_id"] else None
            if current is not None:
                check_environment_identity(current, expected, self.control.action_user)
                if phase == "cleanup_environment":
                    return await self._write(
                        record, phase, {"_id": expected["_id"]}, context,
                        lambda: self.control.remove_environment(expected), "cleanup_environment_verify",
                    )
                return
            await run_in_threadpool(self._phase, deployment_id, "cleanup_proxy")
            return
        if phase not in {"cleanup_proxy", "cleanup_proxy_verify"}:
            return
        proxy = await self.control.find_proxy(project_id=context["projectId"], name=context["name"])
        if proxy is None:
            await run_in_threadpool(self._phase, deployment_id, "cleanup_records_deleted")
            return
        if phase == "cleanup_proxy" and data.get("maat_api_id"):
            endpoint = data["endpoint"]
            expected_proxy = {
                "_id": data["maat_api_id"], "projectId": context["projectId"], "name": context["name"],
                "ip": endpoint["endpoint_host"], "port": str(endpoint["endpoint_port"]),
                "type": "dev", "protocol": context["proxy_options"]["protocol"],
            }
            return await self._write(
                record, phase, {"_id": expected_proxy["_id"]}, context,
                lambda: self.control.remove_proxy(expected_proxy), "cleanup_proxy_verify",
            )

    def _name_conflict(self, deployment_id):
        with self.sf() as session:
            row = self._locked(session, deployment_id)
            if row.data.get("maat_write_intent") or row.data.get("maat_api_id") or row.data.get("maat_env_id"):
                return
            request_cleanup(row)
            row.data = {**row.data, "error": "MAAT 环境名称已存在，本次创建已停止；不会复用或覆盖已有环境。"}
            row.stage = "maat_name_conflict"
            session.commit()
            _log_maat(row, "meegle_maat_name_conflict")

    async def _cleanup(self, record, phase):
        data, deployment_id = record["data"], record["deployment_id"]
        context = data.get("maat_context")
        if not context or data.get("maat_cleanup_verified"):
            return
        if not phase.startswith("cleanup"):
            await run_in_threadpool(self._phase, deployment_id, "cleanup_environment")
            return
        expected = {**{k: context[k] for k in ("name", "projectId", "projectName", "region")},
                    "_id": data.get("maat_env_id")}
        if phase == "cleanup_environment":
            if expected["_id"]:
                return await self._write(
                    record, phase, {"_id": expected["_id"]}, context,
                    lambda: self.control.remove_environment(expected), "cleanup_routes",
                )
            await run_in_threadpool(self._phase, deployment_id, "cleanup_routes")
            return
        domain = environment_url({**expected, "name": context["name"]}, {"domainType": "generic"})
        if await self.control.domain_routes(urlsplit(domain).hostname):
            return
        if phase == "cleanup_routes" and data.get("maat_api_id"):
            endpoint = data["endpoint"]
            proxy = {
                "_id": data["maat_api_id"], "projectId": context["projectId"], "name": context["name"],
                "ip": endpoint["endpoint_host"], "port": str(endpoint["endpoint_port"]),
                "type": "dev", "protocol": context["proxy_options"]["protocol"],
            }
            return await self._write(
                record, phase, {"_id": proxy["_id"]}, context,
                lambda: self.control.remove_proxy(proxy), "cleanup_verify",
            )
        proxy = await self.control.find_proxy(project_id=context["projectId"], name=context["name"])
        if proxy is None:
            await run_in_threadpool(self._phase, deployment_id, "cleanup_verified", cleanup_verified=True)
