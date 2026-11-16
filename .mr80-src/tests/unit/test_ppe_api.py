from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from conciliator.api.aeolus import router
from conciliator.domain.aeolus_task import DeploymentVerification
from conciliator.services.aeolus_registry import TaskNotFound


@pytest.fixture
async def api():
    task = SimpleNamespace(
        task_id="one",
        action_token="one-token",
        state="review",
        ppe_capability={"enabled": True, "profile_revision": "c" * 64},
        ppe_deployment=None,
    )
    other = SimpleNamespace(task_id="two", action_token="two-token", state="review")
    service = SimpleNamespace(
        get=AsyncMock(side_effect=lambda task_id: task if task_id == "one" else other),
        confirm=AsyncMock(return_value=task),
    )
    ppe = SimpleNamespace(
        create=AsyncMock(),
        retry=AsyncMock(),
        release=AsyncMock(),
        snapshot=AsyncMock(side_effect=TaskNotFound("deployment")),
    )
    app = FastAPI()
    app.state.aeolus_service = service
    app.state.ppe_deployment_service = ppe
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, ppe, service


@pytest.mark.asyncio
async def test_ppe_writes_and_reads_require_task_bound_token(api):
    client, ppe, _ = api
    cases = [
        (
            "post",
            "/ppe-deployments",
            {"request_id": "id", "round_no": 1, "commit_sha": "a" * 40, "profile_revision": "c" * 64},
        ),
        ("post", "/ppe-deployments/record/retry", {"request_id": "id"}),
        ("post", "/ppe-deployments/record/release", {"environment_id": "a" * 32, "generation": 1}),
        ("get", "/ppe-deployments/record", None),
    ]
    for method, suffix, body in cases:
        url = "/api/aeolus/tasks/one" + suffix
        for token in ("", "two-token"):
            response = await client.request(method, url, headers={"X-Aeolus-Task-Token": token}, json=body)
            assert response.status_code == 401
        response = await client.request(method, url, headers={"X-Aeolus-Task-Token": "one-token"}, json=body)
        assert response.status_code == (404 if method == "get" else 202)
        assert "action_token" not in response.json()
    ppe.create.assert_awaited_once_with("one", "id", 1, "a" * 40, "c" * 64)
    ppe.release.assert_awaited_once_with("one", "record", "a" * 32, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("devbox,ppe", [(True, True), (True, False), (False, True)])
async def test_verified_confirmation_binds_every_available_environment(api, devbox, ppe):
    client, _, service = api
    task = await service.get("one")
    task.deployment = {"deployment_id": "box", "current": devbox}
    task.ppe_deployment = {"deployment_id": "ppe", "can_verify": ppe}
    targets = (
        ([{"target": "devbox", "deployment_id": "box"}] if devbox else [])
        + ([{"target": "ppe", "deployment_id": "ppe"}] if ppe else [])
    )
    response = await client.post(
        "/api/aeolus/tasks/one/confirm", headers={"X-Aeolus-Task-Token": "one-token"},
        json={"request_id": "both", "mode": "verified", "verification_targets": targets},
    )
    assert response.status_code == 202
    service.confirm.assert_awaited_once_with(
        "one", "both", release_authorized=True,
        verification_targets=tuple(DeploymentVerification(**target) for target in targets),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "duplicate", "replaced", "unready"])
async def test_verified_confirmation_rejects_an_incomplete_or_stale_environment_set(api, change):
    client, _, service = api
    task = await service.get("one")
    task.deployment = {"deployment_id": "box", "current": True}
    task.ppe_deployment = {"deployment_id": "ppe", "can_verify": True}
    targets = [{"target": "devbox", "deployment_id": "box"}, {"target": "ppe", "deployment_id": "ppe"}]
    if change == "missing":
        targets.pop()
    elif change == "duplicate":
        targets[1] = targets[0]
    elif change == "replaced":
        task.deployment["deployment_id"] = "new"
    else:
        task.ppe_deployment["can_verify"] = False
    response = await client.post(
        "/api/aeolus/tasks/one/confirm", headers={"X-Aeolus-Task-Token": "one-token"},
        json={"request_id": "both", "mode": "verified", "verification_targets": targets},
    )
    assert response.status_code == 409
    service.confirm.assert_not_awaited()


@pytest.mark.asyncio
async def test_ppe_completion_dispatches_explicit_target_and_id(api):
    client, _, service = api
    response = await client.post(
        "/api/aeolus/tasks/one/confirm",
        headers={"X-Aeolus-Task-Token": "one-token"},
        json={"request_id": "done", "mode": "verified", "deployment_target": "ppe", "deployment_id": "record"},
    )
    assert response.status_code == 202
    service.confirm.assert_awaited_once_with("one", "done", "record", True, "ppe")
