from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from conciliator.api.aeolus import _snapshot, router
from conciliator.domain.aeolus_task import InvalidAeolusWebhook
from conciliator.utils.retry import TransientError


class FakeService:
    def __init__(self):
        self.feedback = None
        self.evaluation = None

    async def accept_webhook(self, payload):
        if payload.get("signature") == "bad":
            raise InvalidAeolusWebhook("invalid webhook signature")
        return SimpleNamespace(task_id="task-1"), True

    async def find_by_node(self, project_key, work_item_id, node_id):
        return SimpleNamespace(
            task_id="task-1",
            project_key=project_key,
            work_item_id=work_item_id,
            state_key=node_id,
            node_uuid="node-uuid",
            user_key="user-1",
            state="review",
            action_token="action-token",
            requirement_url="",
            evaluation=self.evaluation,
            mr_url="https://code.example/mr/1",
            coco_session_id="session-1",
            coco_run_id="run-1",
            coco_last_event_id="event-1",
            rounds=[],
            last_error="",
            created_at=None,
            updated_at=None,
        )

    async def get(self, task_id):
        return await self.find_by_node("space", 1, "node")

    async def submit_feedback(self, task_id, feedback, request_id):
        self.feedback = (task_id, feedback, request_id)
        return await self.get(task_id)

    async def evaluation_feedback(self, task_id, feedback, request_id):
        self.feedback = (task_id, feedback, request_id)
        return await self.get(task_id)

    async def confirm_evaluation(self, task_id, request_id):
        self.confirmed_evaluation = (task_id, request_id)
        return await self.get(task_id)

    async def retry_coding(self, task_id, request_id):
        self.retried_coding = (task_id, request_id)
        return await self.get(task_id)


class FakeDocumentAccess:
    def __init__(self):
        self.call = None

    async def read_markdown(self, *, task_id, capability, url):
        self.call = (task_id, capability, url)
        return "# Requirement"


@pytest.mark.asyncio
async def test_coding_retry_requires_action_token_and_request_id():
    app, service = _app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        url = "/api/aeolus/tasks/task-1/retry-coding"
        assert (await client.post(url, json={"request_id": "retry-1"})).status_code == 401
        headers = {"X-Aeolus-Task-Token": "action-token"}
        assert (await client.post(url, headers=headers, json={})).status_code == 422
        response = await client.post(url, headers=headers, json={"request_id": "retry-1"})
        assert response.status_code == 202
        assert response.json()["can_retry_coding"] is False
        assert "action_token" not in response.json()
        assert service.retried_coding == ("task-1", "retry-1")


def _app():
    app = FastAPI()
    app.state.aeolus_service = FakeService()
    app.state.aeolus_document_access_service = FakeDocumentAccess()
    app.include_router(router)
    return app, app.state.aeolus_service


def test_snapshot_hides_internal_queued_round_status_from_card_clients():
    task = SimpleNamespace(
        state="recoding",
        rounds=[{"round_no": 2, "status": "queued"}],
        evaluation={"valid": True},
        platform_status_done=False,
        completion_requested=False,
    )

    assert _snapshot(task)["rounds"][0]["status"] == "coding"


@pytest.mark.asyncio
async def test_document_proxy_requires_capability_and_returns_markdown():
    app, _ = _app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = "/api/aeolus/tasks/task-1/lark-document"
        assert (
            await client.post(url, json={"url": "https://x/docx/doc1"})
        ).status_code == 422
        response = await client.post(
            url,
            headers={"X-Aeolus-Document-Capability": "capability"},
            json={"url": "https://bytedance.larkoffice.com/docx/doc1"},
        )
    assert response.status_code == 200
    assert response.json()["markdown"] == "# Requirement"
    assert app.state.aeolus_document_access_service.call == (
        "task-1",
        "capability",
        "https://bytedance.larkoffice.com/docx/doc1",
    )


@pytest.mark.asyncio
async def test_webhook_rejects_bad_signature_and_acknowledges_valid_event():
    app, _ = _app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (
            await client.post(
                "/hooks/meegle/ai-node", json={"signature": "bad"}
            )
        ).status_code == 401
        response = await client.post(
            "/hooks/meegle/ai-node", json={"signature": "ok"}
        )
        assert response.status_code == 200
        assert response.json() == {
            "ok": True,
            "started": True,
            "task_id": "task-1",
        }


@pytest.mark.asyncio
async def test_lookup_returns_action_token_but_not_internal_request_ids():
    app, _ = _app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/aeolus/tasks/by-node",
            params={
                "space_id": "space",
                "work_item_id": 42,
                "node_id": "state-1",
            },
        )
        assert response.status_code == 200
        assert response.json()["action_token"] == "action-token"
        assert response.json()["coco_run_id"] == "run-1"
        assert response.json()["user_key"] == "user-1"
        assert response.json()["authorization_required"] is False
        assert "claimed_request_ids" not in response.json()


@pytest.mark.asyncio
async def test_lookup_reports_authorization_required_only_for_blocked_rounds():
    app, service = _app()
    service.evaluation = {
        "valid": False,
        "missing_items": ["当前用户尚未完成飞书授权或授权已失效，请先完成飞书授权后重试"],
        "document_read_error_code": 131006,
        "document_authorization_required": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/aeolus/tasks/by-node",
            params={"space_id": "space", "work_item_id": 42, "node_id": "state-1"},
        )
    assert response.status_code == 200
    assert response.json()["authorization_required"] is True


@pytest.mark.asyncio
async def test_lookup_reports_authorization_required_for_blocked_coding_run():
    app, service = _app()
    service.evaluation = {"valid": True, "coding_authorization_required": True}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/aeolus/tasks/by-node",
            params={"space_id": "space", "work_item_id": 42, "node_id": "state-1"},
        )
    assert response.status_code == 200
    assert response.json()["authorization_required"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [True, False, None])
async def test_lookup_checks_authorization_without_refreshing_tokens(available):
    app, _ = _app()
    tokens = SimpleNamespace(
        authorization_available=AsyncMock(return_value=available),
        user_access_token=AsyncMock(side_effect=AssertionError("GET must not refresh")),
    )
    if available is None:
        tokens.authorization_available.side_effect = TransientError("store unavailable")
    app.state.lark_user_token_service = tokens
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/aeolus/tasks/by-node", params={
            "space_id": "space", "work_item_id": 42, "node_id": "state-1",
        })
    assert response.status_code == 200
    assert response.json()["lark_authorized"] is available
    tokens.authorization_available.assert_awaited_once_with("user-1")
    tokens.user_access_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_feedback_requires_action_token():
    app, service = _app()
    body = {"feedback": "请补测试", "request_id": "request-1"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (
            await client.post("/api/aeolus/tasks/task-1/feedback", json=body)
        ).status_code == 401
        response = await client.post(
            "/api/aeolus/tasks/task-1/feedback",
            json=body,
            headers={"X-Aeolus-Task-Token": "action-token"},
        )
        assert response.status_code == 202
        assert service.feedback == ("task-1", "请补测试", "request-1")


@pytest.mark.asyncio
async def test_evaluation_actions_require_token_and_dispatch():
    app, service = _app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path, body in (
            ("evaluation-feedback", {"feedback": "补充验收标准", "request_id": "e-1"}),
            ("confirm-evaluation", {"request_id": "e-2"}),
        ):
            url = f"/api/aeolus/tasks/task-1/{path}"
            assert (await client.post(url, json=body)).status_code == 401
            response = await client.post(
                url, json=body, headers={"X-Aeolus-Task-Token": "action-token"}
            )
            assert response.status_code == 202
            assert "evaluation_rounds" in response.json()
        assert service.feedback == ("task-1", "补充验收标准", "e-1")
        assert service.confirmed_evaluation == ("task-1", "e-2")
