from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from conciliator.integrations.nextcode.discussions import get_thread
from conciliator.integrations.nextcode.mrs import (
    MR,
    get_mergeability,
    get_mr,
    list_repo_mrs,
)
from conciliator.services.comment_events import CommentEvent
from conciliator.services.notification_service import send_resolve_feedback
from conciliator.utils.logging import get_logger
from conciliator.utils.redaction import safe_text
from conciliator.utils.trace_context import trace_scope, traced_child_work, traced_work

router = APIRouter(tags=["webhooks"])
log = get_logger(__name__)


def _nextcode_delivery_key(delivery_id: str, event_id: str) -> str | None:
    if event_id:
        if len(event_id) > 128:
            raise HTTPException(status_code=400, detail="webhook event id is too long")
        return f"event:{event_id}"
    if delivery_id:
        if len(delivery_id) > 128:
            raise HTTPException(status_code=400, detail="webhook delivery id is too long")
        return f"delivery:{delivery_id}"
    return None


@dataclass(frozen=True)
class ParsedComment:
    comment_id: str
    repo_id: str
    mr_iid: int
    body: str
    author_email: str
    discussion_id: str | None
    position: dict | None
    mr: MR


@router.post("/hooks/codebase")
async def codebase_webhook(
    request: Request,
    x_vecode_event: str = Header("", alias="X-Vecode-Event"),
    x_vecode_signature: str = Header("", alias="X-Vecode-Signature"),
    x_vecode_app_id: str = Header("", alias="X-Vecode-App-Id"),
    x_vecode_event_id: str = Header("", alias="X-Vecode-Event-Id"),
    x_vecode_delivery: str = Header("", alias="X-Vecode-Delivery"),
) -> dict:
    body = await request.body()
    state = request.app.state

    if x_vecode_app_id and x_vecode_app_id != state.settings.nextcode_app_id:
        raise HTTPException(status_code=401, detail="unexpected app id")
    if not state.nextcode_client.verify_webhook_signature(body, x_vecode_signature):
        raise HTTPException(status_code=401, detail="invalid signature")
    if not x_vecode_event:
        raise HTTPException(status_code=400, detail="missing X-Vecode-Event")

    delivery_key = _nextcode_delivery_key(x_vecode_delivery, x_vecode_event_id)
    if delivery_key is None:
        raise HTTPException(
            status_code=400,
            detail="missing webhook event and delivery ids",
        )
    if delivery_key is not None:
        claimed = await state.task_repo.claim_nextcode_webhook(
            delivery_key=delivery_key,
            event_id=x_vecode_event_id or None,
            delivery_id=x_vecode_delivery or None,
            event_type=x_vecode_event,
        )
        if not claimed:
            log.info(
                "codebase_webhook_deduplicated",
                webhook_event=x_vecode_event,
                event_id=x_vecode_event_id,
                delivery=x_vecode_delivery,
            )
            return {"ok": True, "processed": 0, "deduplicated": True}

    log.info(
        "codebase_webhook_received",
        webhook_event=x_vecode_event,
        event_id=x_vecode_event_id,
        delivery=x_vecode_delivery,
    )

    try:
        payload: dict[str, Any] = await request.json()
        payload["_webhook"] = {
            "event_id": x_vecode_event_id or None,
            "delivery_id": x_vecode_delivery or None,
            "event_type": x_vecode_event,
        }
        processed = 0
        if x_vecode_event == "comment":
            processed = await _handle_comment_event(state, payload)
        elif x_vecode_event == "merge_request_review":
            log.info(
                "webhook_ignored_event",
                event_type=x_vecode_event,
                action=payload.get("Action"),
            )
        elif x_vecode_event == "merge_request":
            if payload.get("Action") in {
                "opened",
                "reopened",
                "edited",
                "pushed",
            }:
                processed = int(await _handle_mr_conflict(state, payload))
        elif x_vecode_event == "push":
            processed = await _handle_target_branch_push(state, payload)
        else:
            log.info("webhook_ignored_event", event_type=x_vecode_event)
    except Exception as exc:
        if delivery_key is not None:
            try:
                await state.task_repo.fail_nextcode_webhook(
                    delivery_key=delivery_key,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                log.exception(
                    "codebase_webhook_failure_record_failed",
                    delivery=delivery_key,
                )
        raise

    if delivery_key is not None:
        await state.task_repo.complete_nextcode_webhook(
            delivery_key=delivery_key,
            processed_count=processed,
        )

    return {"ok": True, "processed": processed}


async def _handle_comment_event(state, payload: dict) -> int:
    if payload.get("Action") != "created":
        return 0
    comments: list[dict] = []
    if payload.get("Comment"):
        comments.append(payload["Comment"])
    comments.extend(payload.get("Comments") or [])

    processed = 0
    seen: set[str] = set()
    for comment in comments:
        comment_id = str(comment.get("Id") or "")
        if comment_id and comment_id in seen:
            continue
        seen.add(comment_id)
        if comment.get("CommentableType") != "merge_request":
            continue
        if _is_comment_created_by_current_app(state, payload, comment):
            log.info(
                "comment_created_by_current_app_ignored",
                comment_id=comment_id,
            )
            continue
        tokens = str(comment.get("Content") or "").strip().split()
        is_resolve = (
            bool(tokens) and tokens[0] == "/resolve"
            and (len(tokens) == 1 or tokens[1].lower() in {"start", "go"})
            and getattr(state, "pipeline_enabled", True)
        )
        repo_id = str(comment.get("RepoId") or (payload.get("Repository") or {}).get("Id") or "")
        mr_id = str(comment.get("CommentableId") or "")
        thread_id = comment.get("ThreadId")
        with trace_scope():
            if is_resolve and mr_id:
                await send_resolve_feedback(
                    state.nextcode_client, repo_id=repo_id, mr_id=mr_id, thread_id=thread_id,
                    body="已收到解冲突指令，正在检查并创建任务。",
                )
            try:
                parsed = await _parse_comment(state, payload, comment)
                await _route_comment(state, payload, parsed)
            except Exception as exc:
                if is_resolve and mr_id:
                    await send_resolve_feedback(
                        state.nextcode_client, repo_id=repo_id, mr_id=mr_id, thread_id=thread_id,
                        body=f"无法创建解冲突任务：{safe_text(str(exc))[:1000]}",
                    )
                raise
        processed += 1
    return processed


def _is_comment_created_by_current_app(state, payload: dict, comment: dict) -> bool:
    """Return whether a comment was authored by this NextCode app account."""
    app_id = str(state.settings.nextcode_app_id or "").strip()
    if not app_id:
        return False

    creator = comment.get("CreatedBy") or {}
    sender = payload.get("Sender") or {}
    creator_id = str(creator.get("Id") or "").strip()
    sender_id = str(sender.get("Id") or "").strip()
    return app_id in {creator_id, sender_id}


async def _parse_comment(state, payload: dict, comment: dict) -> ParsedComment:
    repository = payload.get("Repository") or {}
    repo_id = str(comment.get("RepoId") or repository.get("Id") or "")
    mr_id = str(comment.get("CommentableId") or "")
    if not repo_id or not mr_id:
        raise HTTPException(status_code=400, detail="comment webhook lacks RepoId/CommentableId")
    mr = await get_mr(state.nextcode_client, repo_id, mr_id=mr_id)
    creator = comment.get("CreatedBy") or {}
    sender = payload.get("Sender") or {}
    discussion_id = comment.get("ThreadId")
    position = comment.get("Position")
    if position is None and discussion_id:
        thread = await get_thread(state.nextcode_client, repo_id, discussion_id)
        position = thread.positions[0] if thread.positions else None
    return ParsedComment(
        comment_id=str(comment.get("Id") or ""),
        repo_id=repo_id,
        mr_iid=mr.iid,
        body=str(comment.get("Content") or ""),
        author_email=str(creator.get("Email") or sender.get("Email") or ""),
        discussion_id=discussion_id,
        position=position,
        mr=mr,
    )


async def _route_comment(state, payload: dict, comment: ParsedComment) -> None:
    log.info(
        "comment_received",
        repo_id=comment.repo_id,
        mr_iid=comment.mr_iid,
        body=comment.body[:100],
        author=comment.author_email,
        discussion_id=comment.discussion_id,
        position=comment.position,
    )

    raw_payload = _normalize_trigger_payload(payload, comment.mr)
    raw_payload["comment_position"] = comment.position
    raw_payload["comment_author_email"] = comment.author_email
    dispatcher = getattr(state, "comment_dispatcher", None)
    if dispatcher is None:
        # No domain registered an interest in comments during startup.
        return
    await dispatcher.dispatch(
        CommentEvent(
            comment_id=comment.comment_id,
            repo_id=comment.repo_id,
            mr_iid=comment.mr_iid,
            body=comment.body,
            author_email=comment.author_email,
            discussion_id=comment.discussion_id,
            position=comment.position,
            raw_payload=raw_payload,
        )
    )


async def _handle_mr_conflict(state, payload: dict) -> bool:
    repository = payload.get("Repository") or {}
    mr_data = payload.get("MergeRequest") or {}
    repo_id = str(repository.get("Id") or mr_data.get("TargetRepoId") or "")
    mr_iid = int(mr_data.get("Number") or 0)
    if not repo_id or not mr_iid:
        raise HTTPException(status_code=400, detail="invalid merge_request webhook")
    return await _check_and_route_conflict(
        state,
        repo_id=repo_id,
        mr_iid=mr_iid,
        target_branch=str(mr_data.get("TargetBranchName") or ""),
        raw_payload=_normalize_trigger_payload(payload),
    )


@traced_work
async def _handle_target_branch_push(state, payload: dict) -> int:
    repository = payload.get("Repository") or {}
    repo_id = str(repository.get("Id") or "")
    ref = str(payload.get("Ref") or "")
    prefix = "refs/heads/"
    if not repo_id or not ref.startswith(prefix) or payload.get("Deleted"):
        return 0
    target_branch = ref[len(prefix) :]
    mrs = await list_repo_mrs(
        state.nextcode_client,
        repo_id,
        target_branch=target_branch,
        status="open",
    )
    processed = 0
    for mr in mrs:
        normalized = _normalize_trigger_payload(payload, mr)
        if await _check_and_route_conflict(
            state,
            repo_id=repo_id,
            mr_iid=mr.iid,
            target_branch=mr.target_branch,
            raw_payload=normalized,
        ):
            processed += 1
    return processed


@traced_child_work
async def _check_and_route_conflict(
    state,
    *,
    repo_id: str,
    mr_iid: int,
    target_branch: str,
    raw_payload: dict,
) -> bool:
    mergeability = await get_mergeability(state.nextcode_client, repo_id, mr_iid)
    log.info(
        "mr_mergeability_checked",
        repo_id=repo_id,
        mr_iid=mr_iid,
        mergeable=mergeability.mergeable,
        reason=mergeability.reason,
        detail=mergeability.detail,
    )
    if mergeability.mergeable or mergeability.reason != "conflict":
        return False

    repo_config = await state.task_repo.get_repo_config(repo_id)
    if repo_config is None or not repo_config.auto_resolve_enabled:
        log.info("auto_resolve_skipped_not_enabled", repo_id=repo_id, mr_iid=mr_iid)
        return False
    if repo_config.allowed_target_branches and target_branch:
        allowed = repo_config.allowed_target_branches
        if not any(target_branch == pattern or target_branch.startswith(pattern.rstrip("*")) for pattern in allowed):
            log.info(
                "auto_resolve_skipped_target_not_allowed",
                repo_id=repo_id,
                target=target_branch,
                allowed=allowed,
            )
            return False

    await state.comment_router.route_auto_conflict(
        repo_id=repo_id,
        mr_iid=mr_iid,
        raw_payload=raw_payload,
    )
    return True


def _normalize_trigger_payload(payload: dict, mr: MR | None = None) -> dict:
    mr_data = payload.get("MergeRequest") or {}
    repository = payload.get("Repository") or {}
    if mr is None:
        source_branch = str(mr_data.get("SourceBranchName") or "")
        target_branch = str(mr_data.get("TargetBranchName") or "")
        title = str(mr_data.get("Title") or "")
        mr_iid = int(mr_data.get("Number") or 0)
        repo_id = str(repository.get("Id") or mr_data.get("TargetRepoId") or "")
        repo_url = str(repository.get("CloneURL") or repository.get("URL") or "")
    else:
        source_branch = mr.source_branch or ""
        target_branch = mr.target_branch
        title = mr.title
        mr_iid = mr.iid
        repo_id = mr.target_repo_id
        target_repo = mr_data.get("TargetRepository") or {}
        repo_url = str(repository.get("CloneURL") or target_repo.get("CloneURL") or repository.get("URL") or mr.web_url)
    return {
        "source_branch": source_branch,
        "target_branch": target_branch,
        "repo_url": repo_url,
        "mr_title": title,
        "mr_iid": mr_iid,
        "repo_id": repo_id,
        "_webhook": payload.get("_webhook") or {},
        "_raw": payload,
    }
