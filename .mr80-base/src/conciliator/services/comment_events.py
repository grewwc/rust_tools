"""Domain-neutral description of a merge-request comment.

The webhook layer translates a Codebase payload into :class:`CommentEvent`
once, then hands it to :class:`~conciliator.api.comment_router.CommentDispatcher`.
Handlers decide on their own whether an event belongs to them, so the transport
layer never has to know which business domains exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ThreadComment:
    """One comment inside an MR discussion, in the order the platform returns it."""

    comment_id: str
    body: str
    author_email: str


@dataclass(frozen=True)
class CommentEvent:
    comment_id: str
    repo_id: str
    mr_iid: int
    body: str
    author_email: str
    discussion_id: str | None
    position: dict[str, Any] | None
    raw_payload: dict[str, Any] = field(default_factory=dict)
    # The comment this one answers, when the caller had to read the thread to
    # find out. ``None`` means either "first comment in the thread" or "not
    # checked"; both render the same way.
    parent: ThreadComment | None = None

    @property
    def command(self) -> str:
        return self.body.strip()

    @property
    def is_inline(self) -> bool:
        return self.position is not None
