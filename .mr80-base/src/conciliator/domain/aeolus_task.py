from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal


class InvalidAeolusWebhook(ValueError):
    pass


class AeolusState(StrEnum):
    RECEIVED = "received"
    READING_DOCUMENT = "reading_document"
    EVALUATING = "evaluating"
    EVALUATION_REVIEW = "evaluation_review"
    VALIDATION_FAILED = "validation_failed"
    CODING = "coding"
    REVIEW = "review"
    CODING_BLOCKED = "coding_blocked"
    RECODING = "recoding"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class DeploymentVerification:
    target: Literal["devbox", "ppe"]
    deployment_id: str


def current_verification_targets(task: AeolusTask) -> tuple[DeploymentVerification, ...]:
    targets = []
    deployment = getattr(task, "deployment", None)
    ppe = getattr(task, "ppe_deployment", None)
    if deployment and deployment.get("current"):
        targets.append(DeploymentVerification("devbox", deployment["deployment_id"]))
    if ppe and ppe.get("can_verify"):
        targets.append(DeploymentVerification("ppe", ppe["deployment_id"]))
    return tuple(targets)


@dataclass(frozen=True)
class CodeContext:
    repository: str = "dp/aeolus"
    base_branch: str = "master"
    base_commit: str = ""
    target_branch: str = ""

    def __post_init__(self) -> None:
        if not self.target_branch:
            object.__setattr__(self, "target_branch", self.base_branch)

    @classmethod
    def from_dict(cls, value: Any) -> CodeContext:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("code_context must be an object")
        defaults = cls()
        fields: dict[str, str] = {}
        for name in ("repository", "base_branch", "base_commit", "target_branch"):
            default = fields["base_branch"] if name == "target_branch" else getattr(defaults, name)
            item = value.get(name, default)
            if not isinstance(item, str):
                raise ValueError(f"code_context.{name} must be a string")
            fields[name] = item.strip() or default
        repository = fields["repository"]
        if repository.startswith("https://code.byted.org/"):
            repository = repository.removeprefix("https://code.byted.org/").rstrip("/").removesuffix(".git")
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+", repository):
            raise ValueError("code_context.repository must be a Codebase repository path")
        if any(part in {".", ".."} for part in repository.split("/")):
            raise ValueError("invalid repository path")
        fields["repository"] = repository
        for name in ("base_branch", "target_branch"):
            branch = fields[name]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch) or any(
                part in branch for part in ("..", "//")
            ) or branch.endswith(("/", ".", ".lock")):
                raise ValueError(f"invalid code_context.{name}")
        if fields["base_commit"] and not re.fullmatch(r"[0-9a-fA-F]{7,40}", fields["base_commit"]):
            raise ValueError("code_context.base_commit must be a commit SHA")
        return cls(**fields)


@dataclass(frozen=True)
class AeolusStartEvent:
    task_id: str
    project_key: str
    work_item_id: int
    state_key: str
    node_uuid: str
    user_key: str
    props: tuple[dict[str, Any], ...]
    work_item_type_key: str = ""


@dataclass
class AeolusAttempt:
    attempt_no: int
    status: str
    session_id: str = ""
    run_id: str = ""
    event_id: str = ""
    error: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None


@dataclass
class AeolusRound:
    round_no: int
    status: str
    feedback: str = ""
    trigger_type: str = "initial"
    reviewed_round_no: int | None = None
    base_commit_sha: str = ""
    summary: str = ""
    commit_sha: str = ""
    code_url: str = ""
    tests: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    failure_code: str = ""
    error: str = ""
    attempts: list[AeolusAttempt] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None


@dataclass
class AeolusTask:
    task_id: str
    project_key: str
    work_item_id: int
    state_key: str
    node_uuid: str
    user_key: str
    props: tuple[dict[str, Any], ...]
    action_token: str
    state: AeolusState = AeolusState.RECEIVED
    requirement_url: str = ""
    requirement_text: str = ""
    evaluation: dict[str, Any] | None = None
    evaluation_rounds: list[dict[str, Any]] = field(default_factory=list)
    repo_id: str = ""
    work_branch: str = ""
    coco_session_id: str = ""
    coco_run_id: str = ""
    coco_last_event_id: str = ""
    mr_id: str = ""
    mr_iid: int | None = None
    mr_url: str = ""
    platform_status_done: bool = False
    completion_requested: bool = False
    rounds: list[AeolusRound] = field(default_factory=list)
    claimed_request_ids: set[str] = field(default_factory=set)
    last_error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    timing: dict[str, Any] | None = None
    deployment: dict[str, Any] | None = None
    deployment_workers: list[dict[str, Any]] | None = None
    ppe_deployment: dict[str, Any] | None = None
    ppe_capability: dict[str, Any] | None = None
    work_item_type_key: str = ""


def verify_webhook_signature(payload: dict[str, Any], webhook_token: str) -> bool:
    plugin_id = str(payload.get("source_plugin_id") or "")
    request_time = str(payload.get("request_time") or "")
    signature = str(payload.get("signature") or "").lower()
    if not plugin_id or not request_time or not signature or not webhook_token:
        return False
    # source: lpm ai docs "兼容性说明与验签说明 webhook 请求体 signature request_time"
    expected = hashlib.sha256(
        f"{plugin_id}{request_time}{webhook_token}".encode()
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def parse_start_event(
    payload: dict[str, Any],
    *,
    expected_plugin_id: str,
    webhook_token: str,
) -> AeolusStartEvent | None:
    if str(payload.get("source_plugin_id") or "") != expected_plugin_id:
        raise InvalidAeolusWebhook("unexpected source plugin id")
    if not verify_webhook_signature(payload, webhook_token):
        raise InvalidAeolusWebhook("invalid webhook signature")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise InvalidAeolusWebhook("webhook data must be an object")
    change = data.get("change_ai_node_info") or {}
    after = change.get("after") or {}
    if data.get("event_type") != 8001 or after.get("status") != "running":
        return None

    work_items = data.get("work_item_info") or []
    ai_node_info = data.get("ai_node_info") or {}
    ai_info = ai_node_info.get("ai_info") or {}
    try:
        task_id = str(data["idempotent_key"])
        project_key = str(data["project_key"])
        work_item_id = int(work_items[0]["work_item_id"])
        state_key = str(ai_node_info["state_key"])
        node_uuid = str(after["entity_id"])
        user_key = str((data.get("user_info") or {})["user_key"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise InvalidAeolusWebhook("start event is missing required fields") from exc
    if not all((task_id, project_key, state_key, node_uuid, user_key)):
        raise InvalidAeolusWebhook("start event contains empty required fields")
    props = ai_info.get("props") or []
    if not isinstance(props, list) or any(not isinstance(item, dict) for item in props):
        raise InvalidAeolusWebhook("AI node props must be an object list")
    # source: AI 节点事件 https://meego-hc.larkoffice.com/b/helpcenter/1p8d7djs/dl7cjf41
    work_item = work_items[0]
    work_item_type = work_item.get("work_item_type_key") or data.get("work_item_type_key")
    return AeolusStartEvent(
        task_id=task_id,
        project_key=project_key,
        work_item_id=work_item_id,
        state_key=state_key,
        node_uuid=node_uuid,
        user_key=user_key,
        props=tuple(props),
        work_item_type_key=work_item_type.strip() if isinstance(work_item_type, str) else "",
    )


def can_retry_coding(task: AeolusTask) -> bool:
    return (
        task.state == AeolusState.FAILED
        and bool(task.rounds)
        and (task.evaluation or {}).get("valid") is True
        and not getattr(task, "platform_status_done", False)
        and not getattr(task, "completion_requested", False)
    )


def can_retry_completion(task: AeolusTask) -> bool:
    return task.state == AeolusState.FAILED and (
        getattr(task, "completion_requested", False) or getattr(task, "platform_status_done", False)
    )


def extract_requirement_url(
    event: AeolusStartEvent,
    *,
    prop_key: str = "design_doc",
) -> str:
    prop = next((item for item in event.props if item.get("prop_key") == prop_key), None)
    if prop is None:
        raise ValueError(f"AI node property {prop_key!r} is missing")
    fields = (prop.get("prop_value") or {}).get("field") or []
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"AI node property {prop_key!r} has no mapped field value")
    value = fields[0].get("field_value", fields[0].get("value"))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"AI node property {prop_key!r} is not a document link") from exc
        else:
            value = stripped
    if isinstance(value, list):
        value = value[0] if value else ""
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        raise ValueError(f"AI node property {prop_key!r} is not a document link")
    return value
