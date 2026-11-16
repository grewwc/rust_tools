from dataclasses import dataclass, field
from typing import Protocol

from conciliator.services.comment_events import CommentEvent


class CommentHandler(Protocol):
    """One business domain that reacts to merge-request comments.

    Handlers are consulted in registration order and the first match wins, so
    the order encodes precedence between domains.
    """

    async def matches(self, event: CommentEvent) -> bool:
        ...

    async def handle(self, event: CommentEvent) -> None:
        ...


@dataclass
class CommentDispatcher:
    handlers: list[CommentHandler] = field(default_factory=list)

    def register(self, handler: CommentHandler) -> None:
        self.handlers.append(handler)

    async def dispatch(self, event: CommentEvent) -> None:
        for handler in self.handlers:
            if await handler.matches(event):
                await handler.handle(event)
                return


@dataclass
class CommentRouter:
    """路由评论到 resolve_service / approval_service。

    服务实际实例在 Task 17 (main.py lifespan) 注入。这里用 object 占位类型,
    实际类型由 ResolveService / ApprovalService 提供。
    """
    resolve_service: object
    approval_service: object
    task_repo: object

    async def route_resolve_trigger(
        self, *, repo_id, mr_iid, author_email, body, discussion_id, raw_payload,
    ):
        from conciliator.services.resolve_service import ResolveService
        svc: ResolveService = self.resolve_service  # type: ignore[assignment]
        await svc.handle_trigger(
            repo_id=repo_id, mr_iid=mr_iid, author_email=author_email,
            payload=raw_payload, trigger_type="COMMENT",
            discussion_id=discussion_id,
        )

    async def route_approval_command(
        self, *, repo_id, mr_iid, body, author_email, discussion_id, raw_payload,
    ):
        from conciliator.services.approval_service import ApprovalService
        svc: ApprovalService = self.approval_service  # type: ignore[assignment]
        await svc.handle_comment(
            repo_id=repo_id, mr_iid=mr_iid, body=body,
            author_email=author_email, discussion_id=discussion_id,
            webhook_event_id=str(
                (raw_payload.get("_webhook") or {}).get("event_id") or ""
            )
            or None,
        )

    async def route_auto_conflict(self, *, repo_id, mr_iid, raw_payload):
        from conciliator.services.resolve_service import ResolveService
        svc: ResolveService = self.resolve_service  # type: ignore[assignment]
        await svc.handle_trigger(
            repo_id=repo_id, mr_iid=mr_iid, author_email=None,
            payload=raw_payload, trigger_type="AUTO_CONFLICT",
        )
