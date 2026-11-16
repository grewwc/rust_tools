"""Adding physical primary keys must preserve business-key ORM operations."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from conciliator.repositories.models import Deployment, DeploymentWorker
from conciliator.repositories.ppe_models import PpeDeployment, PpeEnvironment


@pytest.mark.parametrize(
    "model,key,values",
    [
        (DeploymentWorker, "worker_id", {"session_id": "s", "last_seen_at": datetime(2026, 1, 1), "data": {}}),
        (
            Deployment,
            "deployment_id",
            {
                "task_id": "task",
                "worker_id": "worker",
                "request_id": "req",
                "round_no": 1,
                "commit_sha": "a" * 40,
                "status": "pending",
                "created_at": datetime(2026, 1, 1),
                "data": {},
            },
        ),
        (
            PpeEnvironment,
            "environment_id",
            {
                "task_id": "task",
                "generation": 1,
                "name": "ppe_test",
                "resource_state": "planned",
                "profile_revision": "profile",
                "created_at": datetime(2026, 1, 1),
                "data": {},
            },
        ),
        (
            PpeDeployment,
            "deployment_id",
            {
                "task_id": "task",
                "environment_id": "env",
                "request_id": "req",
                "sequence": 1,
                "round_no": 1,
                "commit_sha": "a" * 40,
                "profile_revision": "profile",
                "status": "pending",
                "stage": "queued",
                "created_at": datetime(2026, 1, 1),
                "next_poll_at": datetime(2026, 1, 1),
                "error": "",
                "data": {},
            },
        ),
    ],
)
def test_business_identity_survives_physical_primary_key(model, key, values):
    engine = create_engine("sqlite://")
    model.__table__.create(engine)
    assert inspect(engine).get_pk_constraint(model.__tablename__)["constrained_columns"] == ["id"]
    try:
        with Session(engine) as session:
            session.add(model(**{key: "business-id", **values}))
            session.commit()
        # A fresh session forces a SQL lookup rather than an identity-map hit.
        with Session(engine) as session:
            record = session.get(model, "business-id")
            assert record is not None and record.id > 0
            record.data = {"updated": True}
            session.commit()
        with Session(engine) as session:
            record = session.get(model, "business-id", with_for_update=True)
            assert record.data == {"updated": True}
            session.delete(record)
            session.commit()
            assert session.get(model, "business-id") is None
    finally:
        engine.dispose()
