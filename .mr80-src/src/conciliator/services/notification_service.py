import asyncio

from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.mrs import add_mr_comment, get_mr
from conciliator.utils.logging import get_logger
from conciliator.utils.redaction import safe_text

log = get_logger(__name__)
_FEEDBACK_TIMEOUT_SECONDS = 5


async def send_resolve_feedback(
    nc: NextCodeClient, *, repo_id: str, thread_id: str | None, body: str,
    mr_id: str | None = None, mr_iid: int | None = None,
) -> None:
    if not thread_id or not repo_id:
        return
    try:
        async with asyncio.timeout(_FEEDBACK_TIMEOUT_SECONDS):
            if not mr_id:
                if mr_iid is None:
                    return
                mr_id = (await get_mr(nc, repo_id, mr_iid)).id
            await nc.call("CreateComment", {
                "CommentableType": "merge_request", "CommentableId": mr_id,
                "RepoId": repo_id, "ThreadId": thread_id, "Content": safe_text(body),
            })
        log.info("resolve_feedback_sent", repo_id=repo_id, mr_id=mr_id, thread_id=thread_id)
    except Exception:
        # Feedback failure must not turn a successfully enqueued task into a failure.
        log.exception("resolve_feedback_failed", repo_id=repo_id, thread_id=thread_id)


class NotificationService:
    def __init__(self, nc_client: NextCodeClient):
        self._nc = nc_client

    async def notify_mr_comment(
        self,
        *,
        project_id: str,
        mr_iid: int,
        body: str,
        mention_emails: list[str] | None = None,
    ) -> None:
        final_body = body
        if mention_emails:
            mentions = " ".join(f"@{e}" for e in mention_emails)
            final_body = f"{mentions}\n{body}"
        await add_mr_comment(self._nc, project_id, mr_iid, final_body)
