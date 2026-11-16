"""Standalone HTTP/database smoke; the production app always starts all features."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from conciliator.api.health import router


def test_http_and_database_health_without_external_platform_clients():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    app = FastAPI()
    app.include_router(router)
    app.state.engine = engine
    try:
        with TestClient(app) as client:
            for path in ("/healthz", "/readyz"):
                response = client.get(path)
                assert response.status_code == 200
                assert response.json() == {"status": "ok", "app": "conciliator"}
    finally:
        engine.dispose()


@pytest.mark.parametrize("failure", ["missing_engine", "connect", "execute"])
def test_database_unavailable_returns_503_without_failing_liveness(failure):
    app = FastAPI()
    app.include_router(router)
    if failure != "missing_engine":
        engine = MagicMock()
        error = RuntimeError("database unavailable")
        if failure == "connect":
            engine.connect.side_effect = error
        else:
            engine.connect.return_value.__enter__.return_value.execute.side_effect = error
        app.state.engine = engine

    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json() == {"status": "degraded", "app": "conciliator"}
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "app": "conciliator"}
