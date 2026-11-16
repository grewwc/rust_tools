"""PPE resources belong to a task; deployment history belongs to individual requests."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import Mapped, mapped_column

from conciliator.repositories.models import Base


class PpeEnvironment(Base):
    __tablename__ = "aeolus_ppe_environments"
    __table_args__ = (
        UniqueConstraint("task_id", "generation", name="uk_aeolus_ppe_generation"),
        UniqueConstraint("environment_id", name="uk_aeolus_ppe_environment_id"),
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True, comment="主键ID"
    )
    environment_id: Mapped[str] = mapped_column(String(32), nullable=False)
    # Preserve business-key Session.get() lookups independently of the physical primary key.
    __mapper_args__ = {"primary_key": [environment_id]}
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(30), nullable=False, unique=True)
    platform_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_state: Mapped[str] = mapped_column(String(32), nullable=False)
    active_deployment_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    current_deployment_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    profile_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)


class PpeDeployment(Base):
    __tablename__ = "aeolus_ppe_deployments"
    __table_args__ = (
        UniqueConstraint("task_id", "request_id", name="uk_aeolus_ppe_request"),
        UniqueConstraint("task_id", "sequence", name="uk_aeolus_ppe_sequence"),
        Index("idx_aeolus_ppe_poll", "status", "next_poll_at"),
        UniqueConstraint("deployment_id", name="uk_aeolus_ppe_deployment_id"),
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True).with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True, comment="主键ID"
    )
    deployment_id: Mapped[str] = mapped_column(String(32), nullable=False)
    # Preserve business-key Session.get() lookups independently of the physical primary key.
    __mapper_args__ = {"primary_key": [deployment_id]}
    task_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    environment_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    round_no: Mapped[int] = mapped_column(Integer, nullable=False)
    commit_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    profile_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    next_poll_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(32), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
