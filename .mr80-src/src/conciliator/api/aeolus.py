from __future__ import annotations

import hmac
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal, cast

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from conciliator.domain.aeolus_task import (
    DeploymentVerification,
    InvalidAeolusWebhook,
    can_retry_coding,
    can_retry_completion,
    current_verification_targets,
)
from conciliator.integrations.lark.client import LarkError
from conciliator.services.aeolus_document_access import (
    DOCUMENT_CAPABILITY_HEADER,
    DocumentAccessInactive,
    DocumentNotAllowed,
    InvalidDocumentCapability,
    LarkAuthorizationRequired,
)
from conciliator.services.aeolus_lark_service import InvalidCardCallback
from conciliator.services.aeolus_registry import (
    InvalidTaskTransition,
    TaskNotFound,
)
from conciliator.services.aeolus_service import authorization_blocked
from conciliator.utils.logging import get_logger
from conciliator.utils.retry import TransientError

router = APIRouter(tags=["aeolus"])
log = get_logger(__name__)


class FeedbackBody(BaseModel):
    feedback: str = Field(min_length=1, max_length=4000)
    request_id: str = Field(min_length=1, max_length=128)


class DeployBody(BaseModel):
    worker_id: str = Field(min_length=1, max_length=64)
    request_id: str = Field(min_length=1, max_length=128)
    round_no: int = Field(gt=0)
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_profile_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    expected_profile_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ReleaseDeploymentBody(BaseModel):
    deployment_id: str = Field(min_length=1, max_length=64)
    generation: int | None = Field(default=None, ge=0)


class VerificationBody(BaseModel):
    target: Literal["devbox", "ppe"]
    deployment_id: str = Field(min_length=1, max_length=64)


class ActionBody(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    mode: Literal["code_only", "verified"] = "code_only"
    deployment_id: str = ""
    release_resources: bool = False
    deployment_target: Literal["devbox", "ppe"] = "devbox"
    verification_targets: list[VerificationBody] = Field(default_factory=list, max_length=2)


class PpeDeployBody(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    round_no: int = Field(gt=0)
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    profile_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_profile_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class PpeReleaseBody(BaseModel):
    environment_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    generation: int = Field(gt=0)


class LarkDocumentBody(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


def _service(request: Request):
    return request.app.state.aeolus_service


def _document_access_service(request: Request):
    service = getattr(request.app.state, "aeolus_document_access_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Aeolus document access is disabled")
    return service


@router.post("/hooks/lark/aeolus-card")
async def lark_card_callback(request: Request) -> dict[str, Any]:
    service = _service(request)
    cards = getattr(request.app.state, "aeolus_lark_service", None)
    if cards is None:
        raise HTTPException(status_code=503, detail="Aeolus Lark cards are disabled")
    try:
        return await cards.callback(service, await request.body(), request.headers)
    except InvalidCardCallback as exc:
        log.warning("aeolus_lark_callback_rejected", reason=str(exc))
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/hooks/meegle/ai-node")
async def meegle_ai_node_webhook(request: Request) -> dict[str, Any]:
    service = _service(request)
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("webhook body must be an object")
        task, started = await service.accept_webhook(payload)
    except InvalidAeolusWebhook as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "started": started,
        "task_id": task.task_id if task is not None else None,
    }


@router.get("/api/aeolus/tasks/by-node")
async def get_task_by_node(
    request: Request,
    space_id: str = Query(min_length=1),
    work_item_id: int = Query(gt=0),
    node_id: str = Query(min_length=1),
    work_object_id: str = Query(default=""),
) -> dict[str, Any]:
    del work_object_id
    try:
        task = await _service(request).find_by_node(
            space_id, work_item_id, node_id
        )
    except TaskNotFound as exc:
        raise HTTPException(status_code=404, detail="Aeolus task not found") from exc
    snapshot = _snapshot(task, include_action_token=True)
    # The reader tries the application identity before the operator's token, so a
    # lapsed user token alone must not ask for consent. Only a round that was
    # blocked on both identities sets this flag. Coding runs record their own
    # marker because they finish in FAILED rather than VALIDATION_FAILED.
    evaluation = task.evaluation or {}
    snapshot["authorization_required"] = (
        authorization_blocked(task.evaluation)
        or bool(evaluation.get("coding_authorization_required"))
    )
    tokens = getattr(request.app.state, "lark_user_token_service", None)
    if tokens is not None:
        # Polling only inspects stored validity; actual document access refreshes
        # credentials. Revocation cannot be detected without contacting Feishu.
        try:
            snapshot["lark_authorized"] = await tokens.authorization_available(task.user_key)
        except TransientError:
            log.warning(
                "aeolus_lark_authorization_status_unavailable",
                task_id=task.task_id,
            )
            snapshot["lark_authorized"] = None
    return snapshot


@router.post("/api/aeolus/tasks/{task_id}/lark-document")
async def read_lark_document(
    task_id: str,
    body: LarkDocumentBody,
    request: Request,
    capability: str = Header(..., alias=DOCUMENT_CAPABILITY_HEADER),
) -> dict[str, str]:
    try:
        markdown = await _document_access_service(request).read_markdown(
            task_id=task_id,
            capability=capability,
            url=body.url,
        )
    except TaskNotFound as exc:
        raise HTTPException(status_code=404, detail="Aeolus task not found") from exc
    except InvalidDocumentCapability as exc:
        raise HTTPException(status_code=401, detail="Invalid document capability") from exc
    except (DocumentAccessInactive, DocumentNotAllowed) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except LarkAuthorizationRequired as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "lark_authorization_required",
                "authorization_url": exc.authorization_url,
            },
        ) from exc
    except TransientError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "lark_temporarily_unavailable"},
        ) from exc
    except (LarkError, ValueError) as exc:
        log.warning(
            "aeolus_agent_lark_document_read_failed",
            task_id=task_id,
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail="Feishu document read failed") from exc
    return {"url": body.url, "markdown": markdown}


@router.post(
    "/api/aeolus/tasks/{task_id}/retry-coding",
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_coding(
    task_id: str, body: ActionBody, request: Request,
    action_token: str = Header("", alias="X-Aeolus-Task-Token"),
) -> dict[str, Any]:
    service = _service(request)
    await _authorize(service, task_id, action_token)
    return _snapshot(await _call(service.retry_coding, task_id, body.request_id))


@router.post(
    "/api/aeolus/tasks/{task_id}/retry-validation",
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_validation(
    task_id: str,
    request: Request,
    action_token: str = Header("", alias="X-Aeolus-Task-Token"),
) -> dict[str, Any]:
    service = _service(request)
    await _authorize(service, task_id, action_token)
    return _snapshot(await _call(service.retry_validation, task_id))


@router.post(
    "/api/aeolus/tasks/{task_id}/evaluation-feedback",
    status_code=status.HTTP_202_ACCEPTED,
)
async def evaluation_feedback(
    task_id: str, body: FeedbackBody, request: Request,
    action_token: str = Header("", alias="X-Aeolus-Task-Token"),
) -> dict[str, Any]:
    service = _service(request)
    await _authorize(service, task_id, action_token)
    return _snapshot(await _call(
        service.evaluation_feedback, task_id, body.feedback, body.request_id
    ))


@router.post(
    "/api/aeolus/tasks/{task_id}/confirm-evaluation",
    status_code=status.HTTP_202_ACCEPTED,
)
async def confirm_evaluation(
    task_id: str, body: ActionBody, request: Request,
    action_token: str = Header("", alias="X-Aeolus-Task-Token"),
) -> dict[str, Any]:
    service = _service(request)
    await _authorize(service, task_id, action_token)
    return _snapshot(await _call(service.confirm_evaluation, task_id, body.request_id))


@router.post(
    "/api/aeolus/tasks/{task_id}/feedback",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_feedback(
    task_id: str,
    body: FeedbackBody,
    request: Request,
    action_token: str = Header("", alias="X-Aeolus-Task-Token"),
) -> dict[str, Any]:
    service = _service(request)
    await _authorize(service, task_id, action_token)
    task = await _call(
        service.submit_feedback, task_id, body.feedback, body.request_id
    )
    return _snapshot(task)


@router.post("/api/aeolus/tasks/{task_id}/retry-completion", status_code=status.HTTP_202_ACCEPTED)
@router.post(
    "/api/aeolus/tasks/{task_id}/confirm",
    status_code=status.HTTP_202_ACCEPTED,
)
async def confirm(
    task_id: str,
    body: ActionBody,
    request: Request,
    action_token: str = Header("", alias="X-Aeolus-Task-Token"),
) -> dict[str, Any]:
    service = _service(request)
    task = await _authorize(service, task_id, action_token)
    if body.mode == "verified":
        if body.verification_targets:
            targets = tuple(
                DeploymentVerification(item.target, item.deployment_id) for item in body.verification_targets
            )
            if len(set(targets)) != len(targets) or set(targets) != set(current_verification_targets(task)):
                raise HTTPException(409, "Verification environments changed; refresh before confirming")
            return _snapshot(await _call(
                service.confirm, task_id, body.request_id, release_authorized=True, verification_targets=targets,
            ))
        if body.deployment_target == "ppe":
            if not body.deployment_id:
                raise HTTPException(400, "verification requires a deployment_id")
            return _snapshot(await _call(service.confirm, task_id, body.request_id, body.deployment_id, True, "ppe"))
        if not getattr(task, "deployment", None) or not task.deployment.get("current"):
            raise HTTPException(409, "Current deployment is not accessible or ready for verification")
        if not body.deployment_id:
            raise HTTPException(400, "verification requires a deployment_id")
        return _snapshot(await _call(service.confirm, task_id, body.request_id, body.deployment_id))
    if body.release_resources:
        return _snapshot(await _call(service.confirm, task_id, body.request_id, "", True))
    return _snapshot(await _call(service.confirm, task_id, body.request_id))


def _ppe(request):
    service = getattr(request.app.state, "ppe_deployment_service", None)
    if service is None:
        raise HTTPException(503, "PPE deployment is disabled")
    return service


@router.post("/api/aeolus/tasks/{task_id}/ppe-deployments", status_code=202)
async def ppe_deploy(task_id: str, body: PpeDeployBody, request: Request,
                     action_token: str = Header("", alias="X-Aeolus-Task-Token")):
    service = _service(request)
    await _authorize(service, task_id, action_token)
    args = (task_id, body.request_id, body.round_no, body.commit_sha, body.profile_revision)
    await _call(_ppe(request).create, *args, **(
        {"runtime_profile_id": body.runtime_profile_id} if body.runtime_profile_id else {}
    ))
    return _snapshot(await service.get(task_id))


@router.get("/api/aeolus/tasks/{task_id}/ppe-deployments/{deployment_id}")
async def ppe_get(task_id: str, deployment_id: str, request: Request,
                  action_token: str = Header("", alias="X-Aeolus-Task-Token")):
    await _authorize(_service(request), task_id, action_token)
    return await _call(_ppe(request).snapshot, task_id, deployment_id)


@router.post("/api/aeolus/tasks/{task_id}/ppe-deployments/{deployment_id}/retry", status_code=202)
async def ppe_retry(task_id: str, deployment_id: str, body: ActionBody, request: Request,
                    action_token: str = Header("", alias="X-Aeolus-Task-Token")):
    service = _service(request)
    await _authorize(service, task_id, action_token)
    await _call(_ppe(request).retry, task_id, deployment_id, body.request_id)
    return _snapshot(await service.get(task_id))


@router.post("/api/aeolus/tasks/{task_id}/ppe-deployments/{deployment_id}/release", status_code=202)
async def ppe_release(task_id: str, deployment_id: str, body: PpeReleaseBody, request: Request,
                      action_token: str = Header("", alias="X-Aeolus-Task-Token")):
    service = _service(request)
    await _authorize(service, task_id, action_token)
    await _call(_ppe(request).release, task_id, deployment_id, body.environment_id, body.generation)
    return _snapshot(await service.get(task_id))


@router.post("/api/aeolus/tasks/{task_id}/deploy", status_code=status.HTTP_202_ACCEPTED)
async def deploy(
    task_id: str, body: DeployBody, request: Request, background: BackgroundTasks,
    action_token: str = Header("", alias="X-Aeolus-Task-Token"),
):
    service = _service(request)
    await _authorize(service, task_id, action_token)
    from conciliator.api.deployments import call, notify_result
    await call(request, "enqueue", task_id, body.worker_id, body.request_id, body.round_no, body.commit_sha,
               body.runtime_profile_id, body.expected_profile_revision)
    background.add_task(notify_result, request, task_id)
    return _snapshot(await service.get(task_id))


@router.post("/api/aeolus/tasks/{task_id}/release-environment", status_code=status.HTTP_202_ACCEPTED)
async def release_environment(
    task_id: str, body: ReleaseDeploymentBody, request: Request, background: BackgroundTasks,
    action_token: str = Header("", alias="X-Aeolus-Task-Token"),
):
    service = _service(request)
    await _authorize(service, task_id, action_token)
    from conciliator.api.deployments import call, notify_result
    await call(request, "release", task_id, body.deployment_id, body.generation)
    background.add_task(notify_result, request, task_id)
    return _snapshot(await service.get(task_id))


async def _authorize(service, task_id: str, supplied: str):
    try:
        task = await service.get(task_id)
    except TaskNotFound as exc:
        raise HTTPException(status_code=404, detail="Aeolus task not found") from exc
    if not supplied or not hmac.compare_digest(task.action_token, supplied):
        raise HTTPException(status_code=401, detail="invalid action token")
    return task


async def _call(operation, *args, **kwargs):
    try:
        return await operation(*args, **kwargs)
    except TaskNotFound as exc:
        raise HTTPException(status_code=404, detail="Aeolus task not found") from exc
    except InvalidTaskTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _snapshot(task, *, include_action_token: bool = False) -> dict[str, Any]:
    allowed = (
        "task_id",
        "project_key",
        "work_item_id",
        "state_key",
        "node_uuid",
        "user_key",
        "state",
        "requirement_url",
        "evaluation",
        "evaluation_rounds",
        "mr_url",
        "coco_session_id",
        "coco_run_id",
        "coco_last_event_id",
        "rounds",
        "last_error",
        "created_at",
        "updated_at",
        "timing",
        "deployment",
        "deployment_workers",
        "ppe_deployment",
        "ppe_capability",
    )
    result = {name: _jsonable(getattr(task, name, None)) for name in allowed}
    for round_ in result["rounds"] or []:
        if round_.get("status") == "queued":
            round_["status"] = "coding"
    result["can_retry_coding"] = can_retry_coding(task)
    result["can_retry_completion"] = can_retry_completion(task)
    if include_action_token:
        result["action_token"] = task.action_token
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
