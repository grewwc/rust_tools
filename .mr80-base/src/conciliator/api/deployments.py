"""Credential-free worker endpoints for the explicitly enabled deployment POC."""

import asyncio
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.exc import IntegrityError

from conciliator.services.aeolus_registry import InvalidTaskTransition, TaskNotFound
from conciliator.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/deployment-workers", tags=["deployment-workers"])


class RuntimeCapability(BaseModel):
    profile_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    python_version: str = Field(pattern=r"^\d+\.\d+$")
    base_branches: list[str] = Field(default_factory=list, max_length=128)
    base_release_max: int | None = Field(default=None, ge=0, strict=True)


class Registration(BaseModel):
    worker_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    display_name: str = Field(default="", max_length=128)
    vmid: str | None = Field(default=None, min_length=1, max_length=128)
    protocol_version: Literal[2]
    advertised_host: str = Field(default="", max_length=253)
    runtime_profiles: dict[str, dict[str, RuntimeCapability]] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def check_protocol(self):
        if not self.runtime_profiles or not self.advertised_host:
            raise ValueError("v2 requires runtime_profiles and advertised_host")
        return self


class WorkerRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


class InstanceObservation(BaseModel):
    deployment_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    generation: int = Field(ge=1)
    service_status: Literal["healthy", "unhealthy", "starting", "stopped", "unknown"] = "unknown"
    stage: str = Field(default="", max_length=32)


class Heartbeat(WorkerRequest):
    reconciled: bool = False
    stage: str = Field(default="", max_length=32)
    instances: list[InstanceObservation] = Field(default_factory=list, max_length=1000)


class EndpointReady(WorkerRequest):
    generation: int = Field(ge=1)
    request_id: str = Field(min_length=1, max_length=128)
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    profile_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    endpoint_host: str = Field(min_length=1, max_length=253)
    endpoint_port: int = Field(ge=1024, le=65535)
    runtime_attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    log_path: str = Field(default="", max_length=1024)


class Claim(WorkerRequest):
    request_id: str = Field(min_length=1, max_length=128)


class Result(WorkerRequest):
    generation: int
    status: Literal["succeeded", "failed"]
    commit_sha: str = ""
    error: str = Field(default="", max_length=4000)
    log_path: str = Field(default="", max_length=1024)


class Stopped(WorkerRequest):
    deployment_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    generation: int = Field(ge=1)


def _deployment_service(request):
    service = getattr(request.app.state, "deployment_service", None)
    if service is None:
        raise HTTPException(503, "Deployment POC is disabled")
    return service


async def call(request, operation, *args):
    try:
        return await _deployment_service(request).run(operation, *args)
    except IntegrityError as exc:
        raise HTTPException(409, "Worker or request already registered; retry with its stable identity") from exc
    except TaskNotFound as exc:
        raise HTTPException(404, "Deployment or worker not found") from exc
    except InvalidTaskTransition as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/register")
async def register(body: Registration, request: Request):
    return await call(request, "register", body.model_dump(exclude_none=True))


@router.post("/{worker_id}/deployments/{deployment_id}/endpoint-ready")
async def endpoint_ready(worker_id: str, deployment_id: str, body: EndpointReady, request: Request):
    return await call(request, "endpoint_ready", worker_id, deployment_id, body.model_dump())


@router.post("/{worker_id}/heartbeat")
async def heartbeat(worker_id: str, body: Heartbeat, request: Request):
    return await call(request, "heartbeat", worker_id, body.model_dump())


@router.post("/{worker_id}/claim")
async def claim(worker_id: str, body: Claim, request: Request, background: BackgroundTasks):
    deployment = await call(request, "claim", worker_id, body.model_dump())
    if deployment:
        background.add_task(notify_result, request, deployment["task_id"])
    return {"deployment": deployment}


async def notify_result(request, task_id):
    cards = getattr(request.app.state, "aeolus_lark_service", None)
    service = getattr(request.app.state, "aeolus_service", None)
    if cards is not None and service is not None:
        try:
            task = await service.get(task_id)
            await asyncio.wait_for(cards.notify(task, devbox_only=True), 5)
        except Exception as exc:
            log.warning("deployment_notification_failed", task_id=task_id, error_type=type(exc).__name__)


@router.post("/{worker_id}/deployments/{deployment_id}/result")
async def result(
    worker_id: str,
    deployment_id: str,
    body: Result,
    request: Request,
    background: BackgroundTasks,
):
    reply = await call(request, "result", worker_id, deployment_id, body.model_dump())
    if reply.pop("notify", False):
        background.add_task(notify_result, request, reply["task_id"])
    return reply


@router.post("/{worker_id}/stopped")
async def stopped(worker_id: str, body: Stopped, request: Request, background: BackgroundTasks):
    reply = await call(request, "stopped", worker_id, body.model_dump())
    task_id = reply.pop("notify_task_id", None)
    if task_id:
        background.add_task(notify_result, request, task_id)
    return reply
