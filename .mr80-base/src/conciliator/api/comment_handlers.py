"""Comment handlers for the conflict-resolution / approval domains.

The bodies of these handlers are the previous ``_route_comment`` branches moved
verbatim: the transport layer used to hard-code the domain order, and moving
each branch behind :class:`~conciliator.api.comment_router.CommentHandler`
keeps the behaviour identical while letting other domains slot in between.
"""

from __future__ import annotations

from dataclasses import dataclass

from conciliator.api.comment_router import CommentRouter
from conciliator.services.comment_events import CommentEvent
from conciliator.utils.logging import get_logger

log = get_logger(__name__)


def _log_skipped(event: CommentEvent) -> None:
    log.info(
        "comment_pipeline_skipped",
        repo_id=event.repo_id,
        mr_iid=event.mr_iid,
    )


@dataclass
class ResolveCommandHandler:
    """Owns the explicit ``/resolve ...`` command family."""

    router: CommentRouter
    pipeline_enabled: bool = True

    async def matches(self, event: CommentEvent) -> bool:
        return event.command.startswith("/resolve")

    async def handle(self, event: CommentEvent) -> None:
        if not self.pipeline_enabled:
            _log_skipped(event)
            return
        tokens = event.command.split()
        if len(tokens) == 1 or tokens[1].lower() in {"start", "go"}:
            await self.router.route_resolve_trigger(
                repo_id=event.repo_id,
                mr_iid=event.mr_iid,
                author_email=event.author_email,
                body=event.body,
                discussion_id=event.discussion_id,
                raw_payload=event.raw_payload,
            )
        else:
            await self.router.route_approval_command(
                repo_id=event.repo_id,
                mr_iid=event.mr_iid,
                body=event.body,
                author_email=event.author_email,
                discussion_id=event.discussion_id,
                raw_payload=event.raw_payload,
            )


@dataclass
class ApprovalCommentHandler:
    """Threaded comments that are not commands of any other domain."""

    router: CommentRouter
    pipeline_enabled: bool = True

    async def matches(self, event: CommentEvent) -> bool:
        return event.discussion_id is not None

    async def handle(self, event: CommentEvent) -> None:
        if not self.pipeline_enabled:
            _log_skipped(event)
            return
        await self.router.route_approval_command(
            repo_id=event.repo_id,
            mr_iid=event.mr_iid,
            body=event.body,
            author_email=event.author_email,
            discussion_id=event.discussion_id,
            raw_payload=event.raw_payload,
        )
