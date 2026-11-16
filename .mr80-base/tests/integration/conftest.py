import pytest
from sqlalchemy import create_engine
from testcontainers.mysql import MySqlContainer


@pytest.fixture(scope="session")
def mysql_container():
    with MySqlContainer("mysql:8.0") as mysql:
        host_port = mysql.get_exposed_port(3306)
        dsn = (
            f"mysql+pymysql://{mysql.username}:{mysql.password}"
            f"@{mysql.get_container_host_ip()}:{host_port}/{mysql.dbname}"
        )
        mysql.dsn = dsn
        yield mysql


@pytest.fixture
def clean_db(mysql_container):
    """需要 DB 的测试显式 depend 此 fixture,在测试前清空所有表。"""
    from conciliator.repositories.models import Base
    engine = create_engine(mysql_container.dsn)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()
    yield
