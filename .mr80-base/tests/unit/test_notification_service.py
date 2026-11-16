import pytest

from conciliator.services.notification_service import NotificationService


class FakeNextCodeClient:
    def __init__(self):
        self.calls: list[tuple[str, dict, bool]] = []

    async def call(self, action: str, body: dict, *, read_only: bool = False) -> dict:
        self.calls.append((action, body, read_only))
        if action == "GetMergeRequest":
            return {
                "MergeRequest": {
                    "Id": "mr-id",
                    "Number": 42,
                    "TargetRepoId": "p1",
                    "TargetBranchName": "main",
                    "Title": "MR",
                }
            }
        if action == "CreateComment":
            return {"Comment": {"Id": "comment-id", "ThreadId": "thread-id"}}
        raise AssertionError(action)


@pytest.mark.asyncio
async def test_notify_with_mentions():
    nc = FakeNextCodeClient()
    svc = NotificationService(nc)  # type: ignore[arg-type]
    await svc.notify_mr_comment(
        project_id="p1",
        mr_iid=42,
        body="可以合入",
        mention_emails=["alice@x.com", "bob@x.com"],
    )

    create = nc.calls[1]
    assert create[0] == "CreateComment"
    assert create[1]["CommentableType"] == "merge_request"
    assert create[1]["CommentableId"] == "mr-id"
    assert "@alice@x.com" in create[1]["Content"]
    assert "@bob@x.com" in create[1]["Content"]
    assert "可以合入" in create[1]["Content"]
