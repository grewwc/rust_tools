from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from conciliator.api.comment_handlers import (
    ApprovalCommentHandler,
    ResolveCommandHandler,
)
from conciliator.api.comment_router import CommentDispatcher
from conciliator.api.webhooks import (
    _handle_comment_event,
    _handle_mr_conflict,
    _handle_target_branch_push,
    _nextcode_delivery_key,
    router,
)


def _mr() -> dict:
    return {
        "Id": "mr-id",
        "Number": 42,
        "TargetRepoId": "repo-1",
        "SourceRepoId": "repo-1",
        "SourceBranchName": "test/source",
        "TargetBranchName": "test/target",
        "Title": "conflict test",
        "Description": "",
        "URL": "https://code.test/mr/42",
        "CreatedBy": {"Email": "mr-author@example.com"},
        "Versions": [],
    }


class FakeClient:
    def __init__(self, responses: dict[str, dict]):
        self.responses = responses
        self.calls: list[tuple[str, dict, bool]] = []

    async def call(self, action: str, body: dict, *, read_only: bool = False) -> dict:
        self.calls.append((action, body, read_only))
        return self.responses[action]


class FakeWebhookClient:
    @staticmethod
    def verify_webhook_signature(body: bytes, signature: str) -> bool:
        return signature == "valid"


def _webhook_app(task_repo) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.settings = SimpleNamespace(nextcode_app_id="app-1")
    app.state.nextcode_client = FakeWebhookClient()
    app.state.task_repo = task_repo
    return app


def test_delivery_key_prefers_event_id_and_falls_back_to_delivery_id():
    assert _nextcode_delivery_key("delivery-1", "event-1") == "event:event-1"
    assert _nextcode_delivery_key("delivery-1", "") == "delivery:delivery-1"
    assert _nextcode_delivery_key("", "") is None


def test_delivery_key_rejects_values_that_do_not_fit_persistent_columns():
    with pytest.raises(HTTPException) as error:
        _nextcode_delivery_key("x" * 129, "")
    assert error.value.status_code == 400

    with pytest.raises(HTTPException) as error:
        _nextcode_delivery_key("", "x" * 129)
    assert error.value.status_code == 400


def test_same_webhook_event_with_different_delivery_ids_is_processed_only_once():
    task_repo = SimpleNamespace(
        claim_nextcode_webhook=AsyncMock(side_effect=[True, False]),
        complete_nextcode_webhook=AsyncMock(),
        fail_nextcode_webhook=AsyncMock(),
    )
    headers = {
        "X-Vecode-Event": "repository",
        "X-Vecode-App-Id": "app-1",
        "X-Vecode-Signature": "valid",
        "X-Vecode-Event-Id": "event-1",
        "X-Vecode-Delivery": "delivery-1",
    }

    with TestClient(_webhook_app(task_repo)) as client:
        first = client.post("/hooks/codebase", json={"Action": "updated"}, headers=headers)
        redelivery_headers = {**headers, "X-Vecode-Delivery": "delivery-2"}
        duplicate = client.post(
            "/hooks/codebase",
            json={"Action": "updated"},
            headers=redelivery_headers,
        )

    assert first.status_code == 200
    assert first.json() == {"ok": True, "processed": 0}
    assert duplicate.status_code == 200
    assert duplicate.json() == {
        "ok": True,
        "processed": 0,
        "deduplicated": True,
    }
    assert task_repo.claim_nextcode_webhook.await_count == 2
    task_repo.complete_nextcode_webhook.assert_awaited_once_with(
        delivery_key="event:event-1",
        processed_count=0,
    )
    claims = task_repo.claim_nextcode_webhook.await_args_list
    assert [call.kwargs["delivery_key"] for call in claims] == [
        "event:event-1",
        "event:event-1",
    ]
    assert [call.kwargs["delivery_id"] for call in claims] == [
        "delivery-1",
        "delivery-2",
    ]
    task_repo.fail_nextcode_webhook.assert_not_awaited()


def test_comment_updated_is_ignored_and_returns_200():
    task_repo = SimpleNamespace(
        claim_nextcode_webhook=AsyncMock(return_value=True),
        complete_nextcode_webhook=AsyncMock(),
        fail_nextcode_webhook=AsyncMock(),
    )
    headers = {
        "X-Vecode-Event": "comment",
        "X-Vecode-App-Id": "app-1",
        "X-Vecode-Signature": "valid",
        "X-Vecode-Event-Id": "event-comment-updated",
        "X-Vecode-Delivery": "delivery-comment-updated",
    }
    payload = {
        "Action": "updated",
        "Comment": {
            "Id": "comment-1",
            "CommentableType": "merge_request",
            "Content": "/resolve",
        },
    }

    with TestClient(_webhook_app(task_repo)) as client:
        response = client.post("/hooks/codebase", json=payload, headers=headers)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "processed": 0}
    task_repo.complete_nextcode_webhook.assert_awaited_once_with(
        delivery_key="event:event-comment-updated",
        processed_count=0,
    )


def test_merge_request_review_is_ignored_and_returns_200():
    task_repo = SimpleNamespace(
        claim_nextcode_webhook=AsyncMock(return_value=True),
        complete_nextcode_webhook=AsyncMock(),
        fail_nextcode_webhook=AsyncMock(),
    )
    headers = {
        "X-Vecode-Event": "merge_request_review",
        "X-Vecode-App-Id": "app-1",
        "X-Vecode-Signature": "valid",
        "X-Vecode-Event-Id": "event-review-commented",
        "X-Vecode-Delivery": "delivery-review-commented",
    }
    payload = {
        "Action": "commented",
        "Review": {"Id": "comment-1", "Content": "/resolve"},
        "Comments": [
            {
                "Id": "comment-1",
                "CommentableType": "merge_request",
                "Content": "/resolve",
            }
        ],
    }

    with TestClient(_webhook_app(task_repo)) as client:
        response = client.post("/hooks/codebase", json=payload, headers=headers)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "processed": 0}
    task_repo.complete_nextcode_webhook.assert_awaited_once_with(
        delivery_key="event:event-review-commented",
        processed_count=0,
    )


def test_webhook_processing_failure_is_recorded_for_retry():
    task_repo = SimpleNamespace(
        claim_nextcode_webhook=AsyncMock(return_value=True),
        complete_nextcode_webhook=AsyncMock(),
        fail_nextcode_webhook=AsyncMock(),
    )
    headers = {
        "X-Vecode-Event": "repository",
        "X-Vecode-Signature": "valid",
        "X-Vecode-Delivery": "delivery-bad-json",
    }

    with TestClient(_webhook_app(task_repo), raise_server_exceptions=False) as client:
        response = client.post("/hooks/codebase", content=b"not-json", headers=headers)

    assert response.status_code == 500
    task_repo.fail_nextcode_webhook.assert_awaited_once()
    assert task_repo.fail_nextcode_webhook.await_args.kwargs["delivery_key"] == "delivery:delivery-bad-json"
    task_repo.complete_nextcode_webhook.assert_not_awaited()


def _state(
    client: FakeClient,
    *,
    auto_resolve: bool = True,
):
    config = SimpleNamespace(
        auto_resolve_enabled=auto_resolve,
        allowed_target_branches=["test/*"],
    )
    comment_router = SimpleNamespace(
        route_resolve_trigger=AsyncMock(),
        route_approval_command=AsyncMock(),
        route_auto_conflict=AsyncMock(),
    )
    dispatcher = CommentDispatcher()
    dispatcher.register(ResolveCommandHandler(comment_router))
    dispatcher.register(ApprovalCommentHandler(comment_router))
    return SimpleNamespace(
        nextcode_client=client,
        settings=SimpleNamespace(nextcode_app_id="app-1"),
        comment_router=comment_router,
        comment_dispatcher=dispatcher,
        task_repo=SimpleNamespace(get_repo_config=AsyncMock(return_value=config)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("creator_id", ["app-1", ""])
async def test_comment_created_by_current_app_is_ignored_before_contract_reads(
    creator_id,
):
    client = FakeClient({})
    state = _state(client)
    payload = {
        "Action": "created",
        "Sender": {
            "Id": "app-1",
            "Username": "aeolus_conciliator_test",
            "Email": "aeolus_conciliator_test@bytedance.com",
        },
        "Repository": {"Id": "repo-1"},
        "Comment": {
            "Id": "bot-comment-1",
            "CommentableType": "merge_request",
            "CommentableId": "mr-id",
            "RepoId": "repo-1",
            "Content": "✅ Block #1 已 approve（1/4）",
            "ThreadId": "thread-1",
            "CreatedBy": {"Id": creator_id, "Email": ""},
        },
    }

    processed = await _handle_comment_event(state, payload)

    assert processed == 0
    assert client.calls == []
    state.comment_router.route_resolve_trigger.assert_not_awaited()
    state.comment_router.route_approval_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_comment_event_reads_text_author_thread_position_and_mr_number():
    client = FakeClient(
        {
            "GetMergeRequest": {"MergeRequest": _mr()},
            "GetThread": {
                "Thread": {
                    "Id": "thread-1",
                    "Status": "open",
                    "Positions": [{"Type": "content", "Path": "file.txt", "StartLine": 9}],
                }
            },
        }
    )
    state = _state(client)
    payload = {
        "Action": "created",
        "Sender": {"Email": "sender@example.com"},
        "Repository": {"Id": "repo-1", "CloneURL": "https://code.test/r.git"},
        "Comments": [
            {
                "Id": "comment-1",
                "CommentableType": "merge_request",
                "CommentableId": "mr-id",
                "RepoId": "repo-1",
                "Content": "/reject keep target behavior",
                "ThreadId": "thread-1",
                "CreatedBy": {"Email": "reviewer@example.com"},
            }
        ],
    }

    processed = await _handle_comment_event(state, payload)

    assert processed == 1
    state.comment_router.route_approval_command.assert_awaited_once()
    args = state.comment_router.route_approval_command.await_args.kwargs
    assert args["repo_id"] == "repo-1"
    assert args["mr_iid"] == 42
    assert args["body"] == "/reject keep target behavior"
    assert args["author_email"] == "reviewer@example.com"
    assert args["discussion_id"] == "thread-1"
    assert args["raw_payload"]["comment_position"]["StartLine"] == 9


@pytest.mark.asyncio
async def test_resolve_comment_routes_its_thread_for_user_feedback():
    client = FakeClient(
        {
            "GetMergeRequest": {"MergeRequest": _mr()},
            "GetThread": {
                "Thread": {
                    "Id": "resolve-thread-1",
                    "Status": "open",
                    "Positions": [],
                }
            },
        }
    )
    state = _state(client)
    payload = {
        "Action": "created",
        "Sender": {"Email": "sender@example.com"},
        "Repository": {"Id": "repo-1"},
        "Comment": {
            "Id": "comment-1",
            "CommentableType": "merge_request",
            "CommentableId": "mr-id",
            "RepoId": "repo-1",
            "Content": "/resolve",
            "ThreadId": "resolve-thread-1",
            "CreatedBy": {"Id": "user-1", "Email": "user@example.com"},
        },
    }

    processed = await _handle_comment_event(state, payload)

    assert processed == 1
    state.comment_router.route_resolve_trigger.assert_awaited_once()
    args = state.comment_router.route_resolve_trigger.await_args.kwargs
    assert args["repo_id"] == "repo-1"
    assert args["mr_iid"] == 42
    assert args["discussion_id"] == "resolve-thread-1"


@pytest.mark.asyncio
async def test_comment_event_routes_after_reading_mr():
    client = FakeClient({"GetMergeRequest": {"MergeRequest": _mr()}})
    state = _state(client)
    payload = {
        "Action": "created",
        "Sender": {"Email": "sender@example.com"},
        "Repository": {"Id": "repo-1"},
        "Comment": {
            "Id": "comment-1",
            "CommentableType": "merge_request",
            "CommentableId": "mr-id",
            "RepoId": "repo-1",
            "Content": "/resolve",
        },
    }

    processed = await _handle_comment_event(state, payload)

    assert processed == 1
    assert len(client.calls) == 1
    action, body, read_only = client.calls[0]
    assert action == "GetMergeRequest"
    assert body["RepoId"] == "repo-1"
    assert body["Id"] == "mr-id"
    assert read_only is True
    state.comment_router.route_resolve_trigger.assert_awaited_once()
    state.comment_router.route_approval_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_comment_author_falls_back_to_sender_when_created_by_is_sparse():
    client = FakeClient(
        {
            "GetMergeRequest": {"MergeRequest": _mr()},
            "GetThread": {"Thread": {"Id": "thread-1", "Status": "open", "Positions": []}},
        }
    )
    state = _state(client)
    payload = {
        "Action": "created",
        "Sender": {"Email": "sender@example.com"},
        "Repository": {"Id": "repo-1"},
        "Comment": {
            "Id": "comment-1",
            "CommentableType": "merge_request",
            "CommentableId": "mr-id",
            "RepoId": "repo-1",
            "Content": "/approval",
            "ThreadId": "thread-1",
            "CreatedBy": {"Id": "user-1", "Email": ""},
        },
    }
    await _handle_comment_event(state, payload)
    args = state.comment_router.route_approval_command.await_args.kwargs
    assert args["author_email"] == "sender@example.com"


@pytest.mark.asyncio
async def test_merge_request_event_queries_mergeability_instead_of_payload_flag():
    client = FakeClient(
        {
            "GetMergeRequestMergeability": {
                "Mergeability": {
                    "Mergeable": False,
                    "UnmergeableReason": "conflict",
                    "UnmergeableReasonDetail": "content conflict",
                }
            }
        }
    )
    state = _state(client)
    payload = {
        "Action": "pushed",
        "Repository": {"Id": "repo-1", "CloneURL": "https://code.test/r.git"},
        "MergeRequest": _mr(),
        "OldCommitId": "old",
        "NewCommitId": "new",
    }

    routed = await _handle_mr_conflict(state, payload)

    assert routed is True
    assert client.calls[0] == (
        "GetMergeRequestMergeability",
        {"RepoId": "repo-1", "Number": 42},
        True,
    )
    state.comment_router.route_auto_conflict.assert_awaited_once()


@pytest.mark.asyncio
async def test_merge_request_routes_conflict_for_participating_repository():
    client = FakeClient(
        {
            "GetMergeRequestMergeability": {
                "Mergeability": {
                    "Mergeable": False,
                    "UnmergeableReason": "conflict",
                }
            }
        }
    )
    state = _state(client)
    payload = {
        "Action": "pushed",
        "Repository": {"Id": "repo-1"},
        "MergeRequest": _mr(),
    }

    routed = await _handle_mr_conflict(state, payload)

    assert routed is True
    state.task_repo.get_repo_config.assert_awaited_once()
    state.comment_router.route_auto_conflict.assert_awaited_once()


@pytest.mark.asyncio
async def test_target_branch_push_lists_open_mrs_and_rechecks_conflicts():
    client = FakeClient(
        {
            "ListRepoMergeRequests": {"MergeRequests": [_mr()]},
            "GetMergeRequestMergeability": {
                "Mergeability": {
                    "Mergeable": False,
                    "UnmergeableReason": "conflict",
                }
            },
        }
    )
    state = _state(client)
    payload = {
        "Sender": {"Email": "pusher@example.com"},
        "Repository": {"Id": "repo-1", "CloneURL": "https://code.test/r.git"},
        "Ref": "refs/heads/test/target",
        "Before": "old",
        "After": "new",
        "Commits": [],
        "TotalCommitsCount": 1,
        "Created": False,
        "Deleted": False,
        "Forced": False,
    }

    processed = await _handle_target_branch_push(state, payload)

    assert processed == 1
    assert client.calls[0][0] == "ListRepoMergeRequests"
    assert client.calls[0][1]["TargetBranch"] == "test/target"
    state.comment_router.route_auto_conflict.assert_awaited_once()
