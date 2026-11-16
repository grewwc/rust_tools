from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from conciliator.config import Settings


def create_engine_from_settings(settings: Settings) -> Engine:
    """bytedmysql 同步引擎,走 db_psm 服务发现。"""
    # 生产:走 bytedmysql 的服务发现,host 参数失效
    if settings.env == "prod":
        from bytedmysql import sqlalchemy_init
        sqlalchemy_init()
        url = (
            f"mysql+pymysql://user:password@/{settings.mysql_db_name}"
            f"?charset=utf8mb4&db_psm={settings.mysql_psm}"
        )
    else:
        # 开发/测试:用普通 mysql,DSN 由 MYSQL_DSN 提供。
        import os
        dsn = os.getenv("MYSQL_DSN") or (
            f"mysql+pymysql://root:root@127.0.0.1:3306/{settings.mysql_db_name}"
        )
        url = dsn

    return create_engine(
        url,
        pool_size=settings.mysql_pool_size,
        max_overflow=settings.mysql_pool_max_overflow,
        pool_pre_ping=True,
        # asyncio cancellation cannot interrupt a blocked synchronous driver.
        connect_args={"connect_timeout": 10, "read_timeout": 10, "write_timeout": 10},
        future=True,
    )


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
