import hashlib
import hmac
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conciliator.api.webhooks import router
from conciliator.config import Settings
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.utils.secrets import StaticSecretsProvider


def _sign(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def create_app():
    """Exercise the real signature/handler with injected persistence and no workers."""
    settings = Settings(_env_file=None, env="dev", nextcode_app_id="app1")
    nc = NextCodeClient(settings, StaticSecretsProvider({
        settings.kms_secret_nextcode_app_secret_alias: "secret123",
        settings.kms_secret_nextcode_webhook_secret_alias: "webhook123",
    }))

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await nc.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.state.settings = settings
    app.state.nextcode_client = nc
    app.state.task_repo = SimpleNamespace(
        claim_nextcode_webhook=AsyncMock(return_value=True),
        complete_nextcode_webhook=AsyncMock(),
    )
    return app


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key, value in {
        "ENV": "dev",
        "SERVICE_PSM": "x",
        "KMS_SECRET_NEXTCODE_APP_SECRET_ALIAS": (
            "test/credentials/nextcode_app_secret"
        ),
        "KMS_SECRET_NEXTCODE_WEBHOOK_SECRET_ALIAS": (
            "test/credentials/nextcode_webhook_secret"
        ),
        "KMS_SECRET_LARK_APP_SECRET_ALIAS": "test/credentials/lark_app_secret",
        "NEXTCODE_APP_ID": "app1",
        "COCO_API_BASE": "x",
        "COCO_WORKSPACE_ID": "workspace-1",
        "COCO_DEFAULT_AGENT_ID": "x",
        "LARK_APP_ID": "x",
        "LARK_MARKDOWN_FOLDER_TOKEN": "x",
        "MYSQL_PSM": "x",
        "NEXTCODE_APP_SECRET": "secret123",
        "NEXTCODE_WEBHOOK_SECRET": "webhook123",
        "LARK_APP_SECRET": "ls",
    }.items():
        monkeypatch.setenv(key, value)
    from conciliator.config import get_settings

    get_settings.cache_clear()


def test_codebase_webhook_rejects_missing_event_header():
    app = create_app()
    body = b'{"Action":"created"}'
    with TestClient(app) as client:
        response = client.post(
            "/hooks/codebase",
            content=body,
            headers={"X-Vecode-Signature": _sign(body, "webhook123")},
        )
    assert response.status_code == 400


def test_codebase_webhook_rejects_bad_signature():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/hooks/codebase",
            content=b'{"Action":"created"}',
            headers={
                "X-Vecode-Event": "comment",
                "X-Vecode-Signature": "sha256=invalid",
            },
        )
    assert response.status_code == 401


def test_codebase_webhook_accepts_signed_unhandled_event():
    app = create_app()
    body = b'{"Action":"updated"}'
    with TestClient(app) as client:
        response = client.post(
            "/hooks/codebase",
            content=body,
            headers={
                "X-Vecode-Event": "repository",
                "X-Vecode-App-Id": "app1",
                "X-Vecode-Signature": _sign(body, "webhook123"),
                "X-Vecode-Event-Id": "unhandled-event",
            },
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "processed": 0}


def test_codebase_webhook_requires_delivery_identity():
    app = create_app()
    body = b'{"Action":"updated"}'
    with TestClient(app) as client:
        response = client.post(
            "/hooks/codebase",
            content=body,
            headers={
                "X-Vecode-Event": "repository",
                "X-Vecode-App-Id": "app1",
                "X-Vecode-Signature": _sign(body, "webhook123"),
            },
        )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "missing webhook event and delivery ids"
    }


def test_legacy_coco_webhook_is_not_exposed():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/hooks/coco",
            json={"task_id": "1", "event_type": "task.started", "event_id": "e1"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert response.status_code == 404
