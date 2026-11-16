import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# 让 alembic 能 import 项目代码
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conciliator.config import (  # noqa: E402
    MYSQL_DB_NAME_DEFAULT,
    MYSQL_PSM_DEFAULT,
)
from conciliator.repositories.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_dsn() -> str:
    """同步 DSN。生产走 bytedmysql(db_psm),dev 走 MYSQL_DSN。"""
    env = os.getenv("ENV", "dev")
    if env == "prod":
        from bytedmysql import sqlalchemy_init
        sqlalchemy_init()
        psm = os.getenv("MYSQL_PSM", MYSQL_PSM_DEFAULT)
        db = os.getenv("MYSQL_DB_NAME", MYSQL_DB_NAME_DEFAULT)
        return f"mysql+pymysql://user:password@/{db}?charset=utf8mb4&db_psm={psm}"
    dsn = os.getenv("MYSQL_DSN") or os.getenv("ALEMBIC_MYSQL_DSN", "")
    if not dsn:
        raise RuntimeError(
            "MYSQL_DSN or ALEMBIC_MYSQL_DSN required for migrations in dev"
        )
    return dsn


def run_migrations_offline() -> None:
    context.configure(
        url=_get_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_get_dsn(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        do_run_migrations(connection)
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
