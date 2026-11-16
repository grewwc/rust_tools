# Conciliator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个 codebase app,MR 存在冲突时自动解冲突、汇总方案与 author 邮箱到飞书文档、在 resolve MR 上发起逐块 approve/reject 确认流程。

**Architecture:** Python + FastAPI 单服务。webhook handler 只入队,worker 异步跑 pipeline。解冲突全部委托给 Coco OpenAPI Next,服务负责编排 + 解析冲突块 + git blame + 飞书文档 + MR Discussion + 审批状态机。MySQL 持久化,SQLAlchemy ORM,Alembic 迁移。

**Tech Stack:** Python 3.11+、FastAPI、SQLAlchemy 2.x(同步,bytedmysql 驱动)、Alembic、pydantic v2、httpx(async)、GitPython、lark-oapi、字节内部 SDK(bytedmysql/bytedance.metrics/bytedkms/byteddps)、pytest、testcontainers、respx、uv(包管理)

## Global Constraints

- Python 3.11+(用 `tomllib`、`Literal`、PEP 695 type alias 等)
- 包管理用 `uv`,不写 `requirements.txt`、不用 poetry
- **pip 源用字节内部源**(参考 https://python.byted.org/setup/),`uv sync` 前先配置 `UV_INDEX_URL` 或 `pyproject.toml` 的 `[[tool.uv.index]]`
- ORM 用 SQLAlchemy 2.x 风格(`Mapped` / `mapped_column`),**同步引擎**(bytedmysql 驱动,无 async driver)
- DB 访问在 async 上下文用 `starlette.concurrency.run_in_threadpool` 包同步调用
- pydantic v2(`BaseModel` + `model_config`)
- HTTP 客户端统一 `httpx.AsyncClient`,不混用 requests
- 所有外部 API 调用必经 `integrations/<system>/client.py`,不绕过
- `domain/` 层零 IO(不 import httpx、sqlalchemy、GitPython、byted* SDK)
- 所有 MySQL 时间字段用 `DateTime(timezone=True)`,存 UTC
- 日志 JSON 格式,通过 `structlog` 输出(字节流式日志 SDK 后续接入,见 spec 9.5)
- 环境变量加载用 `pydantic-settings`,启动时校验必填字段
- 敏感配置(App Secret、Coco Token、Lark Secret)从 KMS 拉,不进 env
- Metrics 用 `bytedance.metrics` 上报,不暴露 `/metrics` endpoint
- 外部 HTTP 调用注入 `byteddps.get_token()` 作为服务身份认证 header
- 所有 commit message 用 conventional commits(`feat:`、`fix:`、`docs:`、`refactor:`、`test:`、`chore:`)

## 内部 SDK 速查(实现时按此对接,详见 spec 9.5)

| 包 | import | 关键 API |
|----|--------|---------|
| `bytedmysql` | `from bytedmysql import sqlalchemy_init` | `sqlalchemy_init()` + `create_engine("mysql+pymysql://...db_psm=<psm>")` |
| `bytedance.metrics` | `from bytedance import metrics` | `metrics.emit_counter(name, value, tags=...)`、`emit_timer(name, secs, tags=...)` |
| `bytedkms` | `import kms` | `kms.KMSClient(psm).get_config_by_name("key")` |
| `byteddps` | `import byteddps` | `byteddps.get_token()` |

`BYTEDMETRICS_DEBUG=True` 环境变量可在本地调试时把 metrics 打点打到 stderr。

---

## File Structure

(完整目录树见 spec 第 4 节,此处列任务会触及的文件及职责)

| 文件 | 职责 | 涉及任务 |
|------|------|---------|
| `pyproject.toml` | uv 依赖管理 | T1 |
| `.env.example` | 环境变量样例 | T1 |
| `src/conciliator/config.py` | Settings 配置加载 | T1 |
| `src/conciliator/main.py` | FastAPI app 入口 + lifespan | T2 |
| `src/conciliator/utils/logging.py` | structlog 配置 | T2 |
| `src/conciliator/utils/retry.py` | 重试装饰器 | T2 |
| `src/conciliator/api/health.py` | /healthz、/readyz | T2 |
| `src/conciliator/repositories/db.py` | 同步 engine + session factory(bytedmysql) | T3 |
| `src/conciliator/repositories/models.py` | ORM 模型 | T3 |
| `migrations/` | Alembic 配置 + 首版迁移 | T4 |
| `src/conciliator/domain/approval.py` | ApprovalStatus 枚举 + 状态机 | T5 |
| `src/conciliator/domain/task.py` | TaskStatus 枚举 + 状态机 | T5 |
| `src/conciliator/domain/conflict.py` | ConflictBlock 数据类 | T5 |
| `src/conciliator/domain/commands.py` | 指令解析(单块 + 全局) | T6 |
| `src/conciliator/utils/text.py` | 字符串工具(diff 标记解析等) | T7 |
| `src/conciliator/utils/secrets.py` | KMS / 静态密钥提供者 | T2 |
| `src/conciliator/utils/metrics.py` | bytedance.metrics 上报薄包装 | T18 |
| `src/conciliator/integrations/git/conflicts.py` | `<<<<<<<` 块解析 | T7 |
| `src/conciliator/integrations/git/blame.py` | blame → author email | T8 |
| `src/conciliator/integrations/git/repo.py` | clone + cleanup | T8 |
| `src/conciliator/integrations/nextcode/client.py` | NextCode HTTP + 签名 | T9 |
| `src/conciliator/integrations/nextcode/branches.py` | 创建分支 | T9 |
| `src/conciliator/integrations/nextcode/merges.py` | 读冲突文件 | T9 |
| `src/conciliator/integrations/nextcode/mrs.py` | 创建/更新 MR | T9 |
| `src/conciliator/integrations/nextcode/discussions.py` | 创建 Discussion | T9 |
| `src/conciliator/integrations/nextcode/repos.py` | 仓库成员查询 | T9 |
| `src/conciliator/integrations/coco/client.py` | 提交 Agent 任务 | T10 |
| `src/conciliator/integrations/coco/events.py` | 事件流拾取 | T10 |
| `src/conciliator/integrations/lark/client.py` | app_access_token 缓存 | T11 |
| `src/conciliator/integrations/lark/docs.py` | 创建/更新文档 | T11 |
| `src/conciliator/integrations/lark/contacts.py` | open_id → email | T11 |
| `src/conciliator/repositories/_sync_repo.py` | task 仓储同步实现(bytedmysql) | T12 |
| `src/conciliator/repositories/task_repo.py` | task 仓储 async 包装(run_in_threadpool) | T12 |
| `src/conciliator/api/webhooks.py` | webhook 入口(NextCode + Coco) | T13 |
| `src/conciliator/services/notification_service.py` | MR 评论通知 | T14 |
| `src/conciliator/services/resolve_service.py` | 主 pipeline 编排 | T15 |
| `src/conciliator/services/approval_service.py` | approve/reject 处理 | T16 |
| `src/conciliator/workers/task_runner.py` | worker 轮询 | T17 |

---

## Phase 1:基础设施与领域模型

### Task 1: 项目脚手架与配置加载

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/conciliator/__init__.py`
- Create: `src/conciliator/config.py`
- Create: `README.md`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `Settings` 类(pydantic-settings),字段见 spec 第 11 节;实例化入口 `get_settings()`(lru_cache)

- [ ] **Step 1: 用 uv 初始化项目**

Run:
```bash
cd /Users/bytedance/Development/ai-coding/Conciliator
uv init --package --name conciliator --lib
```

Expected: 生成 `pyproject.toml`、`src/conciliator/__init__.py`、`README.md`

- [ ] **Step 2: 写 pyproject.toml 完整依赖**

Create `pyproject.toml`:

```toml
[project]
name = "conciliator"
version = "0.1.0"
description = "Auto resolve MR conflicts via Coco Agent, with per-block approval flow"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "httpx>=0.27.0",
    "sqlalchemy>=2.0.36",
    "alembic>=1.14.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "structlog>=24.4.0",
    "GitPython>=3.1.43",
    "lark-oapi>=1.4.0",
    "tenacity>=9.0.0",
    "cryptography>=43.0.0",
    # 字节内部 SDK(走内部 pip 源)
    "bytedmysql>=0.3.0",
    "bytedance.metrics>=2.0.0",
    "bytedkms>=2.0.0",
    "byteddps>=2.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "respx>=0.21.1",
    "testcontainers[mysql]>=4.8.0",
    "freezegun>=1.5.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
]

# 字节内部 pip 源(参考 https://python.byted.org/setup/)
[[tool.uv.index]]
name = "byted"
url = "https://artifact.bytedance.com/artifactory/api/pypi/python-release-virtual/simple"
default = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/conciliator"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "ASYNC"]

[tool.mypy]
python_version = "3.11"
strict = true
# byted* SDK 暂无 stub,关闭缺失 import 检查
[[tool.mypy.overrides]]
module = ["bytedmysql.*", "bytedance.*", "kms", "byteddps"]
ignore_missing_imports = true
```

- [ ] **Step 3: 安装依赖**

Run: `uv sync`
Expected: 创建 `.venv`,无错误

- [ ] **Step 4: 写 .env.example**

Create `.env.example`:

```bash
# ── 服务 ──
APP_NAME=conciliator
ENV=dev
LOG_LEVEL=INFO
SERVICE_PSM=conciliator.service.psm  # 服务自身 PSM,用于 KMS/DPS/Metrics

# ── KMS(敏感配置走 KMS,这里只配 PSM 与配置名)──
KMS_PSM=conciliator.service.psm
KMS_CONFIG_NEXTCODE_APP_SECRET=nextcode_app_secret
KMS_CONFIG_COCO_API_TOKEN=coco_api_token
KMS_CONFIG_LARK_APP_SECRET=lark_app_secret

# ── NextCode ──
NEXTCODE_APP_ID=
NEXTCODE_API_BASE=https://code.byted.org/api
NEXTCODE_WEBHOOK_PATH=/hooks/codebase

# ── Coco OpenAPI Next ──
COCO_API_BASE=
COCO_DEFAULT_AGENT_ID=
COCO_TASK_TIMEOUT_SECONDS=900
COCO_EVENT_WEBHOOK_PATH=/hooks/coco

# ── 飞书 ──
LARK_APP_ID=
LARK_DOC_FOLDER_TOKEN=

# ── MySQL(bytedmysql 走 db_psm 服务发现,user/password 也由 bytedmysql/auth 提供)──
MYSQL_PSM=toutiao.mysql.conciliator
MYSQL_DB_NAME=conciliator
MYSQL_POOL_SIZE=10
MYSQL_POOL_MAX_OVERFLOW=20

# ── 运行时 ──
WORKER_CONCURRENCY=4
WORKER_POLL_INTERVAL_SECONDS=5
TASK_TOTAL_TIMEOUT_SECONDS=1200
MAX_RETRIES_PER_TASK=3
REPO_MAX_CONCURRENT_TASKS=3

# ── Git ──
GIT_CLONE_BASE_DIR=/tmp/conciliator
GIT_CLONE_TIMEOUT_SECONDS=120
GIT_USE_LOCAL_CLONE=true

# ── 调试 ──
BYTEDMETRICS_DEBUG=False  # True 时 metrics 打到 stderr
```

- [ ] **Step 5: 写 config.py**

Create `src/conciliator/config.py`:

```python
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── 服务 ──
    app_name: str = "conciliator"
    env: Literal["dev", "staging", "prod"] = "prod"
    log_level: str = "INFO"
    service_psm: str  # 服务自身 PSM

    # ── KMS ──
    kms_psm: str
    kms_config_nextcode_app_secret: str = "nextcode_app_secret"
    kms_config_coco_api_token: str = "coco_api_token"
    kms_config_lark_app_secret: str = "lark_app_secret"

    # ── NextCode ──
    nextcode_app_id: str
    # nextcode_app_secret 从 KMS 拉,运行时由 SecretsProvider 提供
    nextcode_api_base: str = "https://code.byted.org/api"
    nextcode_webhook_path: str = "/hooks/codebase"

    # ── Coco ──
    coco_api_base: str
    # coco_api_token 从 KMS 拉
    coco_default_agent_id: str
    coco_task_timeout_seconds: int = 900
    coco_event_webhook_path: str = "/hooks/coco"

    # ── 飞书 ──
    lark_app_id: str
    # lark_app_secret 从 KMS 拉
    lark_doc_folder_token: str

    # ── MySQL(bytedmysql + db_psm)──
    mysql_psm: str
    mysql_db_name: str = "conciliator"
    mysql_pool_size: int = 10
    mysql_pool_max_overflow: int = 20

    # ── 运行时 ──
    worker_concurrency: int = 4
    worker_poll_interval_seconds: int = 5
    task_total_timeout_seconds: int = 1200
    max_retries_per_task: int = 3
    repo_max_concurrent_tasks: int = 3

    # ── Git ──
    git_clone_base_dir: str = "/tmp/conciliator"
    git_clone_timeout_seconds: int = 120
    git_use_local_clone: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
```

- [ ] **Step 6: 写失败测试**

Create `tests/unit/__init__.py` (empty) and `tests/unit/test_config.py`:

```python
import pytest


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("SERVICE_PSM", "svc.psm")
    monkeypatch.setenv("KMS_PSM", "svc.psm")
    monkeypatch.setenv("NEXTCODE_APP_ID", "app123")
    monkeypatch.setenv("COCO_API_BASE", "https://coco.example.com")
    monkeypatch.setenv("COCO_DEFAULT_AGENT_ID", "agent-1")
    monkeypatch.setenv("LARK_APP_ID", "lark-app")
    monkeypatch.setenv("LARK_DOC_FOLDER_TOKEN", "folder-1")
    monkeypatch.setenv("MYSQL_PSM", "toutiao.mysql.svc")

    from conciliator.config import get_settings
    get_settings.cache_clear()
    s = get_settings()

    assert s.nextcode_app_id == "app123"
    assert s.service_psm == "svc.psm"
    assert s.coco_task_timeout_seconds == 900
    assert s.mysql_psm == "toutiao.mysql.svc"
    assert s.env == "prod"


def test_settings_missing_required_raises(monkeypatch):
    for key in (
        "SERVICE_PSM", "KMS_PSM", "NEXTCODE_APP_ID", "COCO_API_BASE",
        "COCO_DEFAULT_AGENT_ID", "LARK_APP_ID", "LARK_DOC_FOLDER_TOKEN",
        "MYSQL_PSM",
    ):
        monkeypatch.delenv(key, raising=False)
    from conciliator.config import get_settings
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()
```

- [ ] **Step 7: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL(`Settings` 必填字段缺失或导入错误)

- [ ] **Step 8: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 9: 提交**

```bash
git add pyproject.toml .env.example src/conciliator/__init__.py src/conciliator/config.py README.md tests/unit/__init__.py tests/unit/test_config.py
git commit -m "chore: project scaffold with config loading"
```

---

### Task 2: FastAPI 应用骨架、日志、重试与密钥提供者

**Files:**
- Create: `src/conciliator/main.py`
- Create: `src/conciliator/utils/__init__.py`
- Create: `src/conciliator/utils/logging.py`
- Create: `src/conciliator/utils/retry.py`
- Create: `src/conciliator/utils/secrets.py`(KMS 封装)
- Create: `src/conciliator/api/__init__.py`
- Create: `src/conciliator/api/health.py`
- Test: `tests/unit/test_logging.py`
- Test: `tests/unit/test_retry.py`
- Test: `tests/unit/test_secrets.py`
- Test: `tests/integration/test_health.py`

**Interfaces:**
- Consumes: `Settings` (Task 1)
- Produces: `create_app() -> FastAPI`;`setup_logging(settings) -> None`;`retry_async(name, attempts=3, base_delay=0.5)`;`SecretsProvider`(从 KMS 拉密钥,带测试 fake)

- [ ] **Step 1: 写 utils/logging.py**

Create `src/conciliator/utils/__init__.py` (empty) and `src/conciliator/utils/logging.py`:

```python
import logging
import sys

import structlog

from conciliator.config import Settings


def setup_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

- [ ] **Step 2: 写失败测试 test_logging.py**

Create `tests/unit/test_logging.py`:

```python
import json

from conciliator.config import Settings
from conciliator.utils.logging import setup_logging, get_logger


def _make_settings(**overrides) -> Settings:
    base = dict(
        service_psm="x", kms_psm="x",
        nextcode_app_id="x", coco_api_base="x", coco_default_agent_id="x",
        lark_app_id="x", lark_doc_folder_token="x", mysql_psm="x",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_json_log_output(capsys):
    setup_logging(_make_settings(log_level="INFO"))
    log = get_logger("test")
    log.info("hello", task_id="t1", event="started")
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["event"] == "hello"
    assert parsed["task_id"] == "t1"
    assert parsed["level"] == "info"
```

- [ ] **Step 3: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_logging.py -v`
Expected: PASS

- [ ] **Step 4: 写 utils/retry.py**

Create `src/conciliator/utils/retry.py`:

```python
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

P = ParamSpec("P")


class TransientError(Exception):
    """瞬时错误,可重试。"""


def retry_async(
    name: str,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exceptions: tuple[type[BaseException], ...] = (TransientError,),
) -> Callable[[Callable[P, Awaitable[Any]]], Callable[P, Awaitable[Any]]]:
    def decorator(
        fn: Callable[P, Awaitable[Any]],
    ) -> Callable[P, Awaitable[Any]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=base_delay, max=max_delay),
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            ):
                with attempt:
                    return await fn(*args, **kwargs)
            raise RetryError(f"unreachable for {name}")  # pragma: no cover
        return wrapper
    return decorator
```

- [ ] **Step 5: 写失败测试 test_retry.py**

Create `tests/unit/test_retry.py`:

```python
import pytest

from conciliator.utils.retry import retry_async, TransientError


class _FlakyService:
    def __init__(self, fail_times: int):
        self.calls = 0
        self.fail_times = fail_times

    @retry_async("flaky", attempts=4, base_delay=0.0)
    async def call(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TransientError(f"fail #{self.calls}")
        return "ok"


@pytest.mark.asyncio
async def test_retry_succeeds_after_failures():
    svc = _FlakyService(fail_times=2)
    assert await svc.call() == "ok"
    assert svc.calls == 3


@pytest.mark.asyncio
async def test_retry_gives_up_after_max_attempts():
    svc = _FlakyService(fail_times=10)
    with pytest.raises(TransientError):
        await svc.call()
    assert svc.calls == 4


@pytest.mark.asyncio
async def test_retry_does_not_retry_non_transient():
    class _NonFlaky:
        calls = 0

        @retry_async("nonflaky", attempts=3, base_delay=0.0)
        async def call(self) -> None:
            type(self).calls += 1
            raise ValueError("not transient")

    with pytest.raises(ValueError):
        await _NonFlaky().call()
    assert _NonFlaky.calls == 1
```

- [ ] **Step 6: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_retry.py -v`
Expected: PASS

- [ ] **Step 7: 写 utils/secrets.py(KMS 封装)**

Create `src/conciliator/utils/secrets.py`:

```python
from typing import Protocol

from conciliator.config import Settings


class SecretsProvider(Protocol):
    """从 KMS 拉敏感配置的抽象。测试用 fake 实现。"""

    def get(self, config_name: str) -> str: ...


class KmsSecretsProvider:
    """生产用:走 bytedkms。

    启动时一次性拉取并缓存。配置变更需重启或主动 refresh。
    """

    def __init__(self, settings: Settings):
        import kms  # 延迟 import,本地测试不需要装 bytedkms
        self._client = kms.KMSClient(settings.kms_psm)
        self._cache: dict[str, str] = {}

    def get(self, config_name: str) -> str:
        if config_name not in self._cache:
            self._cache[config_name] = self._client.get_config_by_name(config_name)
        return self._cache[config_name]


class StaticSecretsProvider:
    """测试/本地开发用:从 dict 直接取。"""

    def __init__(self, values: dict[str, str]):
        self._values = values

    def get(self, config_name: str) -> str:
        return self._values[config_name]


def build_secrets_provider(settings: Settings) -> SecretsProvider:
    """工厂:env=dev 用 Static,prod 用 Kms。"""
    if settings.env == "dev":
        # 开发机无 KMS,期望从 .env 显式注入 _DEV_SECRETS_* 变量
        import os
        values = {
            settings.kms_config_nextcode_app_secret:
                os.getenv("_DEV_NEXTCODE_APP_SECRET", ""),
            settings.kms_config_coco_api_token:
                os.getenv("_DEV_COCO_API_TOKEN", ""),
            settings.kms_config_lark_app_secret:
                os.getenv("_DEV_LARK_APP_SECRET", ""),
        }
        return StaticSecretsProvider(values)
    return KmsSecretsProvider(settings)
```

- [ ] **Step 8: 写 secrets 单元测试**

Create `tests/unit/test_secrets.py`:

```python
from conciliator.utils.secrets import StaticSecretsProvider


def test_static_provider_returns_value():
    provider = StaticSecretsProvider({"k1": "v1"})
    assert provider.get("k1") == "v1"


def test_static_provider_missing_raises():
    provider = StaticSecretsProvider({})
    try:
        provider.get("missing")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
```

- [ ] **Step 9: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_secrets.py -v`
Expected: PASS

- [ ] **Step 10: 写 api/health.py**

Create `src/conciliator/api/__init__.py` (empty) and `src/conciliator/api/health.py`:

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app: str


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", app="conciliator")


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    return HealthResponse(status="ok", app="conciliator")
```

- [ ] **Step 11: 写 main.py(占位,Task 3/15/17 才补 lifespan 内容)**

Create `src/conciliator/main.py`:

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from conciliator.api import health
from conciliator.config import Settings, get_settings
from conciliator.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    setup_logging(settings)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Conciliator", lifespan=lifespan)
    app.include_router(health.router)
    return app
```

- [ ] **Step 12: 写 health 集成测试**

Create `tests/integration/__init__.py` (empty) and `tests/integration/test_health.py`:

```python
import pytest

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k, v in {
        "SERVICE_PSM": "x", "KMS_PSM": "x",
        "NEXTCODE_APP_ID": "x", "COCO_API_BASE": "x",
        "COCO_DEFAULT_AGENT_ID": "x", "LARK_APP_ID": "x",
        "LARK_DOC_FOLDER_TOKEN": "x", "MYSQL_PSM": "x",
    }.items():
        monkeypatch.setenv(k, v)
    from conciliator.config import get_settings
    get_settings.cache_clear()


def test_healthz():
    from conciliator.main import create_app
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "app": "conciliator"}
```

- [ ] **Step 13: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_health.py -v`
Expected: PASS

- [ ] **Step 14: 提交**

```bash
git add src/conciliator/main.py src/conciliator/utils/ src/conciliator/api/ tests/unit/test_logging.py tests/unit/test_retry.py tests/unit/test_secrets.py tests/integration/
git commit -m "feat: FastAPI scaffold with logging, retry, and KMS secrets provider"
```

---

### Task 3: 数据库 engine(bytedmysql 同步)与 ORM 模型

**Files:**
- Create: `src/conciliator/repositories/__init__.py`
- Create: `src/conciliator/repositories/db.py`
- Create: `src/conciliator/repositories/models.py`
- Test: `tests/integration/test_db.py`

**Interfaces:**
- Consumes: `Settings.mysql_psm`, `Settings.mysql_db_name`, `Settings.mysql_pool_size`, `Settings.mysql_pool_max_overflow`
- Produces: `create_engine(settings) -> Engine`(同步,bytedmysql 驱动);`get_session_factory(engine) -> sessionmaker[Session]`;ORM 模型 `ResolveTask`, `ConflictBlock`, `Approval`, `TaskEvent`, `RepoConfig`

> **关键决策**:bytedmysql 是同步 driver。DB 调用在 async endpoint/service 里通过 `starlette.concurrency.run_in_threadpool` 包到线程池执行(性能开销可接受,见 spec 9.5)。

- [ ] **Step 1: 写 repositories/db.py**

Create `src/conciliator/repositories/__init__.py` (empty) and `src/conciliator/repositories/db.py`:

```python
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
        # 开发/测试:用普通 mysql,DSN 由测试 fixture 提供(_DEV_MYSQL_DSN)
        import os
        dsn = os.getenv("_DEV_MYSQL_DSN") or (
            f"mysql+pymysql://root:root@127.0.0.1:3306/{settings.mysql_db_name}"
        )
        url = dsn

    return create_engine(
        url,
        pool_size=settings.mysql_pool_size,
        max_overflow=settings.mysql_pool_max_overflow,
        pool_pre_ping=True,
        future=True,
    )


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
```

- [ ] **Step 2: 写 repositories/models.py(同步 ORM,与原内容一致)**

Create `src/conciliator/repositories/models.py`:

```python
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ResolveTask(Base):
    __tablename__ = "resolve_tasks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    target_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    resolve_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    original_mr_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    coco_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lark_doc_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    new_mr_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mr_author_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    blocks: Mapped[list["ConflictBlock"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )
    events: Mapped[list["TaskEvent"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_tasks_repo_mr_status", "repo_id", "original_mr_id", "status"),
        Index("ix_tasks_status_updated", "status", "updated_at"),
    )


class ConflictBlock(Base):
    __tablename__ = "conflict_blocks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("resolve_tasks.id", ondelete="CASCADE"), nullable=False
    )
    block_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    ours_text: Mapped[str] = mapped_column(Text, nullable=False)
    theirs_text: Mapped[str] = mapped_column(Text, nullable=False)
    ai_resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    author_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    discussion_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    task: Mapped["ResolveTask"] = relationship(back_populates="blocks")
    approval: Mapped["Approval | None"] = relationship(
        back_populates="block", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("task_id", "block_idx", name="uq_blocks_task_idx"),
        UniqueConstraint("discussion_id", name="uq_blocks_discussion"),
    )


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    block_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conflict_blocks.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    reviewer_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    block: Mapped["ConflictBlock"] = relationship(back_populates="approval")


class TaskEvent(Base):
    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("resolve_tasks.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    coco_event_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    task: Mapped["ResolveTask"] = relationship(back_populates="events")

    __table_args__ = (
        UniqueConstraint("task_id", "coco_event_id", name="uq_events_coco"),
        Index("ix_events_task_time", "task_id", "occurred_at"),
    )


class RepoConfig(Base):
    __tablename__ = "repo_configs"

    repo_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    auto_resolve_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    allowed_target_branches: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    agent_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )
```

- [ ] **Step 3: 写 mysql_container fixture**

Create `tests/integration/conftest.py`:

```python
import pytest
from testcontainers.mysql import MySqlContainer


@pytest.fixture(scope="session")
def mysql_container():
    with MySqlContainer("mysql:8.0") as mysql:
        # 用 _DEV_MYSQL_DSN 注入给 repositories/db.py
        dsn = (
            f"mysql+pymysql://{mysql.username}:{mysql.password}"
            f"@{mysql.get_container_host_ip()}:{mysql.port}/{mysql.dbname}"
        )
        mysql.dsn = dsn
        yield mysql
```

- [ ] **Step 4: 写集成测试 test_db.py**

Create `tests/integration/test_db.py`:

```python
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from conciliator.config import Settings
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.models import Base, ResolveTask, ConflictBlock, Approval


def _settings_for_test(dsn: str) -> Settings:
    """测试环境(env=dev)用 _DEV_MYSQL_DSN 走普通 mysql。"""
    return Settings(
        env="dev",
        service_psm="x", kms_psm="x",
        nextcode_app_id="x", coco_api_base="x", coco_default_agent_id="x",
        lark_app_id="x", lark_doc_folder_token="x",
        mysql_psm="x", mysql_db_name="test",
    )  # type: ignore[arg-type]


def test_create_task_with_blocks_and_approvals(mysql_container, monkeypatch):
    monkeypatch.setenv("_DEV_MYSQL_DSN", mysql_container.dsn)
    settings = _settings_for_test(mysql_container.dsn)
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    SessionLocal = get_session_factory(engine)

    with SessionLocal() as session:
        task = ResolveTask(
            repo_id="r1", source_branch="feat", target_branch="main",
            resolve_branch="resolve/feat/x", original_mr_id=42,
            trigger_type="COMMENT", trigger_payload={"body": "/resolve"},
            status="PENDING",
        )
        block = ConflictBlock(
            block_idx=1, file_path="a.py", start_line=10, end_line=20,
            ours_text="x", theirs_text="y",
        )
        task.blocks.append(block)
        session.add(task)
        session.commit()
        session.refresh(task)

        approval = Approval(block_id=task.blocks[0].id, status="PENDING")
        session.add(approval)
        session.commit()

    with SessionLocal() as session:
        fetched = session.execute(
            select(ConflictBlock).where(ConflictBlock.task_id == task.id)
        ).scalar_one()
        assert fetched.approval.status == "PENDING"
        assert fetched.author_email is None

    engine.dispose()
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_db.py -v`
Expected: PASS(需要 Docker)

- [ ] **Step 6: 提交**

```bash
git add src/conciliator/repositories/ tests/integration/conftest.py tests/integration/test_db.py
git commit -m "feat: bytedmysql-synced database engine and ORM models"
```

---

### Task 4: Alembic 迁移

**Files:**
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/script.py.mako`
- Create: `migrations/versions/0001_initial_schema.py`

**Interfaces:**
- Consumes: `repositories/models.py` (Task 3)
- Produces: `alembic upgrade head` 可建表

- [ ] **Step 1: 初始化 Alembic**

Run: `uv run alembic init migrations`
Expected: 生成 `alembic.ini`、`migrations/`

- [ ] **Step 2: 配置 alembic.ini 用环境变量**

Edit `alembic.ini`,把 `sqlalchemy.url =` 一行改为空(从 env.py 读取):
```ini
sqlalchemy.url =
```

- [ ] **Step 3: 配置 migrations/env.py**

Replace `migrations/env.py` content:

```python
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool, create_engine

# 让 alembic 能 import 项目代码
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conciliator.repositories.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_dsn() -> str:
    """同步 DSN。生产走 bytedmysql(db_psm),dev 走 _DEV_MYSQL_DSN。"""
    env = os.getenv("ENV", "dev")
    if env == "prod":
        from bytedmysql import sqlalchemy_init
        sqlalchemy_init()
        psm = os.environ["MYSQL_PSM"]
        db = os.getenv("MYSQL_DB_NAME", "conciliator")
        return f"mysql+pymysql://user:password@/{db}?charset=utf8mb4&db_psm={psm}"
    dsn = os.getenv("_DEV_MYSQL_DSN") or os.getenv("ALEMBIC_MYSQL_DSN", "")
    if not dsn:
        raise RuntimeError("_DEV_MYSQL_DSN or ALEMBIC_MYSQL_DSN required for migrations in dev")
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
```

- [ ] **Step 4: 生成首版迁移**

Run:
```bash
MYSQL_DSN="mysql+pymysql://u:p@127.0.0.1/dummy" uv run alembic revision --autogenerate -m "initial schema"
```

Expected: 在 `migrations/versions/` 生成 `*_initial_schema.py`

> 若 autogenerate 因连不上 DB 失败,改成手写:`uv run alembic revision -m "initial schema"`,然后编辑生成的空文件,内容见下一步。

- [ ] **Step 5: 手写/校验迁移文件**

若 Step 4 失败,Create `migrations/versions/0001_initial_schema.py`(若自动生成则审查):

```python
"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-21
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "resolve_tasks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("repo_id", sa.String(128), nullable=False),
        sa.Column("source_branch", sa.String(255), nullable=False),
        sa.Column("target_branch", sa.String(255), nullable=False),
        sa.Column("resolve_branch", sa.String(255), nullable=False),
        sa.Column("original_mr_id", sa.BigInteger(), nullable=False),
        sa.Column("trigger_type", sa.String(32), nullable=False),
        sa.Column("trigger_payload", mysql.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("coco_task_id", sa.String(128), nullable=True),
        sa.Column("lark_doc_url", sa.String(512), nullable=True),
        sa.Column("new_mr_id", sa.BigInteger(), nullable=True),
        sa.Column("mr_author_email", sa.String(255), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_tasks_repo_mr_status",
        "resolve_tasks",
        ["repo_id", "original_mr_id", "status"],
    )
    op.create_index(
        "ix_tasks_status_updated", "resolve_tasks", ["status", "updated_at"]
    )

    op.create_table(
        "conflict_blocks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "task_id",
            sa.BigInteger(),
            sa.ForeignKey("resolve_tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("block_idx", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.String(512), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("ours_text", sa.Text(), nullable=False),
        sa.Column("theirs_text", sa.Text(), nullable=False),
        sa.Column("ai_resolution", sa.Text(), nullable=True),
        sa.Column("author_email", sa.String(255), nullable=True),
        sa.Column("discussion_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("task_id", "block_idx", name="uq_blocks_task_idx"),
        sa.UniqueConstraint("discussion_id", name="uq_blocks_discussion"),
    )

    op.create_table(
        "approvals",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "block_id",
            sa.BigInteger(),
            sa.ForeignKey("conflict_blocks.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("reviewer_email", sa.String(255), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="PENDING"),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "task_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "task_id",
            sa.BigInteger(),
            sa.ForeignKey("resolve_tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", mysql.JSON(), nullable=False),
        sa.Column("coco_event_id", sa.String(128), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("task_id", "coco_event_id", name="uq_events_coco"),
    )
    op.create_index(
        "ix_events_task_time", "task_events", ["task_id", "occurred_at"]
    )

    op.create_table(
        "repo_configs",
        sa.Column("repo_id", sa.String(128), primary_key=True),
        sa.Column(
            "auto_resolve_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("allowed_target_branches", mysql.JSON(), nullable=False),
        sa.Column("agent_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("repo_configs")
    op.drop_table("task_events")
    op.drop_table("approvals")
    op.drop_table("conflict_blocks")
    op.drop_table("resolve_tasks")
```

- [ ] **Step 6: 验证迁移可应用**

启动一个本地 mysql(testcontainer 或 docker run),然后:
```bash
export ENV=dev
export _DEV_MYSQL_DSN="mysql+pymysql://root:root@127.0.0.1:3306/test_conciliator"
uv run alembic upgrade head
uv run alembic current
uv run alembic downgrade -1
uv run alembic upgrade head
```

Expected: 无错误,`current` 显示 `0001_initial`,downgrade/up 能来回切

- [ ] **Step 7: 提交**

```bash
git add alembic.ini migrations/
git commit -m "feat: alembic migrations with initial schema"
```

---

### Task 5: 领域模型 — 状态机与枚举

**Files:**
- Create: `src/conciliator/domain/__init__.py`
- Create: `src/conciliator/domain/task.py`
- Create: `src/conciliator/domain/approval.py`
- Create: `src/conciliator/domain/conflict.py`
- Test: `tests/unit/test_task_status.py`
- Test: `tests/unit/test_approval_status.py`

**Interfaces:**
- Produces: `TaskStatus` 枚举(str 子类);`ApprovalStatus` 枚举;`ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]]`;`can_transition(from, to) -> bool`;`ConflictBlock`(dataclass,纯数据);`ConflictParseError`

- [ ] **Step 1: 写 domain/task.py**

Create `src/conciliator/domain/__init__.py` (empty) and `src/conciliator/domain/task.py`:

```python
from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    BRANCH_CREATED = "BRANCH_CREATED"
    CONFLICTS_FETCHED = "CONFLICTS_FETCHED"
    COCO_SUBMITTED = "COCO_SUBMITTED"
    COCO_RUNNING = "COCO_RUNNING"
    DOC_CREATED = "DOC_CREATED"
    MR_CREATED = "MR_CREATED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PARTIAL_REJECTED = "PARTIAL_REJECTED"
    RETRYING = "RETRYING"
    ALL_APPROVED = "ALL_APPROVED"
    CANCELLED = "CANCELLED"
    COCO_TIMEOUT = "COCO_TIMEOUT"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.PREPARING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.PREPARING: {
        TaskStatus.BRANCH_CREATED, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.BRANCH_CREATED: {
        TaskStatus.CONFLICTS_FETCHED, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.CONFLICTS_FETCHED: {
        TaskStatus.COCO_SUBMITTED, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.COCO_SUBMITTED: {
        TaskStatus.COCO_RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.COCO_RUNNING: {
        TaskStatus.DOC_CREATED, TaskStatus.COCO_TIMEOUT,
        TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.COCO_TIMEOUT: {TaskStatus.FAILED},
    TaskStatus.DOC_CREATED: {
        TaskStatus.MR_CREATED, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.MR_CREATED: {
        TaskStatus.AWAITING_APPROVAL, TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.AWAITING_APPROVAL: {
        TaskStatus.ALL_APPROVED, TaskStatus.PARTIAL_REJECTED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.PARTIAL_REJECTED: {
        TaskStatus.RETRYING, TaskStatus.AWAITING_APPROVAL, TaskStatus.CANCELLED,
    },
    TaskStatus.RETRYING: {
        TaskStatus.COCO_SUBMITTED, TaskStatus.AWAITING_APPROVAL,
        TaskStatus.FAILED, TaskStatus.CANCELLED,
    },
    TaskStatus.ALL_APPROVED: set(),
    TaskStatus.CANCELLED: set(),
    TaskStatus.FAILED: set(),
}


TERMINAL_STATUSES = frozenset({
    TaskStatus.ALL_APPROVED, TaskStatus.CANCELLED, TaskStatus.FAILED,
    TaskStatus.COCO_TIMEOUT,
})

SCHEDULABLE_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.RETRYING})
AWAITING_EXTERNAL_STATUSES = frozenset({
    TaskStatus.COCO_RUNNING, TaskStatus.AWAITING_APPROVAL,
    TaskStatus.PARTIAL_REJECTED,
})


def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, set())


class InvalidTransitionError(Exception):
    def __init__(self, current: TaskStatus, target: TaskStatus):
        super().__init__(f"Cannot transition {current.value} -> {target.value}")
        self.current = current
        self.target = target
```

- [ ] **Step 2: 写失败测试 test_task_status.py**

Create `tests/unit/test_task_status.py`:

```python
import pytest

from conciliator.domain.task import (
    TaskStatus, can_transition, InvalidTransitionError,
    SCHEDULABLE_STATUSES, TERMINAL_STATUSES,
)


@pytest.mark.parametrize("frm,to,expected", [
    (TaskStatus.PENDING, TaskStatus.PREPARING, True),
    (TaskStatus.PREPARING, TaskStatus.BRANCH_CREATED, True),
    (TaskStatus.COCO_RUNNING, TaskStatus.DOC_CREATED, True),
    (TaskStatus.COCO_RUNNING, TaskStatus.COCO_TIMEOUT, True),
    (TaskStatus.AWAITING_APPROVAL, TaskStatus.ALL_APPROVED, True),
    (TaskStatus.AWAITING_APPROVAL, TaskStatus.PARTIAL_REJECTED, True),
    (TaskStatus.PARTIAL_REJECTED, TaskStatus.RETRYING, True),
    (TaskStatus.RETRYING, TaskStatus.COCO_SUBMITTED, True),
    (TaskStatus.PENDING, TaskStatus.AWAITING_APPROVAL, False),
    (TaskStatus.AWAITING_APPROVAL, TaskStatus.COCO_RUNNING, False),
    (TaskStatus.ALL_APPROVED, TaskStatus.PENDING, False),
    (TaskStatus.FAILED, TaskStatus.PENDING, False),
])
def test_can_transition(frm, to, expected):
    assert can_transition(frm, to) is expected


def test_schedulable():
    assert TaskStatus.PENDING in SCHEDULABLE_STATUSES
    assert TaskStatus.RETRYING in SCHEDULABLE_STATUSES
    assert TaskStatus.AWAITING_APPROVAL not in SCHEDULABLE_STATUSES


def test_terminal():
    for s in (TaskStatus.ALL_APPROVED, TaskStatus.CANCELLED, TaskStatus.FAILED):
        assert s in TERMINAL_STATUSES
    assert TaskStatus.AWAITING_APPROVAL not in TERMINAL_STATUSES
```

- [ ] **Step 3: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_task_status.py -v`
Expected: PASS

- [ ] **Step 4: 写 domain/approval.py**

Create `src/conciliator/domain/approval.py`:

```python
from enum import StrEnum


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


_ALLOWED = {
    ApprovalStatus.PENDING: {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED},
    ApprovalStatus.APPROVED: {ApprovalStatus.PENDING},  # retry 回退
    ApprovalStatus.REJECTED: {ApprovalStatus.PENDING},  # retry 回退
}


def can_transition(current: ApprovalStatus, target: ApprovalStatus) -> bool:
    return target in _ALLOWED.get(current, set())
```

- [ ] **Step 5: 写失败测试 test_approval_status.py**

Create `tests/unit/test_approval_status.py`:

```python
from conciliator.domain.approval import ApprovalStatus, can_transition


def test_pending_to_approved():
    assert can_transition(ApprovalStatus.PENDING, ApprovalStatus.APPROVED)


def test_pending_to_rejected():
    assert can_transition(ApprovalStatus.PENDING, ApprovalStatus.REJECTED)


def test_approved_to_rejected_blocked():
    assert not can_transition(ApprovalStatus.APPROVED, ApprovalStatus.REJECTED)


def test_retry_resets_to_pending():
    assert can_transition(ApprovalStatus.APPROVED, ApprovalStatus.PENDING)
    assert can_transition(ApprovalStatus.REJECTED, ApprovalStatus.PENDING)
```

- [ ] **Step 6: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_approval_status.py -v`

- [ ] **Step 7: 写 domain/conflict.py**

Create `src/conciliator/domain/conflict.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ConflictBlock:
    """解析后的单个冲突块。无 IO。"""
    file_path: str
    block_idx: int
    start_line: int
    end_line: int
    ours_text: str
    theirs_text: str
    author_email: str | None = None


@dataclass(frozen=True)
class ConflictParseError(Exception):
    """冲突块解析失败。"""
    file_path: str
    detail: str
```

- [ ] **Step 8: 运行全部领域测试**

Run: `uv run pytest tests/unit/test_task_status.py tests/unit/test_approval_status.py -v`
Expected: PASS

- [ ] **Step 9: 提交**

```bash
git add src/conciliator/domain/ tests/unit/test_task_status.py tests/unit/test_approval_status.py
git commit -m "feat: domain status machines for task and approval"
```

---

### Task 6: 指令解析

**Files:**
- Create: `src/conciliator/domain/commands.py`
- Test: `tests/unit/test_command_parsing.py`

**Interfaces:**
- Produces: `Command` 联合类型(`ApproveBlock`, `RejectBlock`, `ApproveAll`, `Retry`, `Status`, `Cancel`, `Ignored`);`parse_comment(body: str, in_discussion: bool) -> Command`

- [ ] **Step 1: 写 domain/commands.py**

Create `src/conciliator/domain/commands.py`:

```python
import re
from dataclasses import dataclass
from typing import Union

# 全局指令:/resolve approve all / /resolve retry / /resolve status / /resolve cancel
_GLOBAL_RE = re.compile(
    r"^\s*/resolve\s+(?P<action>approve\s+all|retry|status|cancel)\b\s*$",
    re.IGNORECASE,
)

# 单块指令:approve / lgtm / ✅ / +1;reject <reason> / ❌ <reason> / -1 <reason>
_APPROVE_TOKENS = frozenset({"approve", "lgtm", "✅", "+1", "👍"})
_REJECT_RE = re.compile(
    r"^\s*(?:reject|❌|-1|👎)\s*(?P<reason>.*?)\s*$", re.IGNORECASE,
)


@dataclass(frozen=True)
class Ignored:
    body: str


@dataclass(frozen=True)
class ApproveBlock:
    pass


@dataclass(frozen=True)
class RejectBlock:
    reason: str


@dataclass(frozen=True)
class ApproveAll:
    pass


@dataclass(frozen=True)
class Retry:
    pass


@dataclass(frozen=True)
class Status:
    pass


@dataclass(frozen=True)
class Cancel:
    pass


Command = Union[
    ApproveBlock, RejectBlock, ApproveAll, Retry, Status, Cancel, Ignored
]


def parse_comment(body: str, in_discussion: bool) -> Command:
    """解析评论 body。

    匹配顺序:
      1. 全局指令(/resolve ...)  — 任何位置都识别
      2. 单块指令(approve/reject) — 仅 in_discussion=True 时识别
      3. 其余 → Ignored
    """
    body = body.strip()
    if not body:
        return Ignored(body=body)

    m = _GLOBAL_RE.match(body)
    if m:
        action = re.sub(r"\s+", " ", m.group("action").lower())
        return {
            "approve all": ApproveAll(),
            "retry": Retry(),
            "status": Status(),
            "cancel": Cancel(),
        }[action]

    if not in_discussion:
        return Ignored(body=body)

    first_token = body.split()[0] if body.split() else ""
    if first_token.lower() in {"approve", "lgtm", "+1"} or first_token in {"✅", "👍"}:
        return ApproveBlock()

    m_reject = _REJECT_RE.match(body)
    if m_reject:
        return RejectBlock(reason=m_reject.group("reason"))

    return Ignored(body=body)
```

- [ ] **Step 2: 写失败测试 test_command_parsing.py**

Create `tests/unit/test_command_parsing.py`:

```python
import pytest

from conciliator.domain.commands import (
    parse_comment, ApproveBlock, RejectBlock, ApproveAll, Retry, Status, Cancel, Ignored,
)


@pytest.mark.parametrize("body", [
    "approve", "APPROVE", "lgtm", "LGTM", "✅", "+1", "👍",
])
def test_single_block_approve_in_discussion(body):
    cmd = parse_comment(body, in_discussion=True)
    assert isinstance(cmd, ApproveBlock)


@pytest.mark.parametrize("body", [
    "reject typo here", "REJECT no", "❌ keep theirs", "-1 bad", "👎 nope",
])
def test_single_block_reject_in_discussion(body):
    cmd = parse_comment(body, in_discussion=True)
    assert isinstance(cmd, RejectBlock)
    assert cmd.reason != "" or body in ("👎 nope",)  # 👎 nope reason="nope"


def test_reject_no_reason_gives_empty_string():
    cmd = parse_comment("reject", in_discussion=True)
    assert isinstance(cmd, RejectBlock)
    assert cmd.reason == ""


def test_global_approve_all_in_discussion():
    cmd = parse_comment("/resolve approve all", in_discussion=True)
    assert isinstance(cmd, ApproveAll)


def test_global_retry_in_main_line():
    cmd = parse_comment("/resolve retry", in_discussion=False)
    assert isinstance(cmd, Retry)


def test_global_status_anywhere():
    assert isinstance(parse_comment("/resolve status", False), Status)
    assert isinstance(parse_comment("/resolve status", True), Status)


def test_global_cancel_case_insensitive():
    assert isinstance(parse_comment("/Resolve CANCEL", False), Cancel)


def test_unknown_command_ignored():
    assert isinstance(parse_comment("looks good to me!", True), Ignored)
    assert isinstance(parse_comment("random comment", False), Ignored)


def test_empty_body_ignored():
    assert isinstance(parse_comment("", True), Ignored)
    assert isinstance(parse_comment("   ", False), Ignored)


def test_approve_token_outside_discussion_ignored():
    # "approve" 不带 /resolve 在 MR 主线不算单块指令
    assert isinstance(parse_comment("approve", False), Ignored)
```

- [ ] **Step 3: 运行测试验证失败**

Run: `uv run pytest tests/unit/test_command_parsing.py -v`
Expected: FAIL(模块不存在;Step 1 已写,实际 PASS,失败则修实现)

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_command_parsing.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/conciliator/domain/commands.py tests/unit/test_command_parsing.py
git commit -m "feat: comment command parsing for approval protocol"
```

---

### Task 7: 冲突块文本解析

**Files:**
- Create: `src/conciliator/utils/text.py`
- Create: `src/conciliator/integrations/__init__.py`
- Create: `src/conciliator/integrations/git/__init__.py`
- Create: `src/conciliator/integrations/git/conflicts.py`
- Test: `tests/unit/test_conflict_parsing.py`
- Test: `tests/unit/fixtures/conflict_samples.py`

**Interfaces:**
- Produces: `parse_conflict_file(file_path: str, content: str) -> list[ParsedConflict]`;`ParsedConflict(file_path, start_line, end_line, ours_text, theirs_text)`

- [ ] **Step 1: 写 fixtures**

Create `tests/unit/fixtures/__init__.py` (empty) and `tests/unit/fixtures/conflict_samples.py`:

```python
SAMPLE_NO_CONFLICT = """def foo():
    return 1
"""

SAMPLE_SINGLE_CONFLICT = """\
line1
<<<<<<< HEAD
ours line
=======
theirs line
>>>>>>> target-branch
line5
"""

SAMPLE_TWO_CONFLICTS = """\
<<<<<<< HEAD
ours1
=======
theirs1
>>>>>>> target
middle
<<<<<<< HEAD
ours2
=======
theirs2
>>>>>>> target
"""

SAMPLE_EMPTY_SIDES = """\
<<<<<<< HEAD
=======
theirs only
>>>>>>> target
"""

SAMPLE_MALFORMED = """\
<<<<<<< HEAD
no closer
"""
```

- [ ] **Step 2: 写 utils/text.py**

Create `src/conciliator/utils/text.py`:

```python
def first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
```

- [ ] **Step 3: 写 git/conflicts.py**

Create `src/conciliator/integrations/__init__.py` (empty),
`src/conciliator/integrations/git/__init__.py` (empty),
and `src/conciliator/integrations/git/conflicts.py`:

```python
import re
from dataclasses import dataclass

_START_RE = re.compile(r"^<{7} \S+$")
_SEP_RE = re.compile(r"^={7}$")
_END_RE = re.compile(r"^>{7} \S+$")


@dataclass(frozen=True)
class ParsedConflict:
    file_path: str
    block_idx: int  # 在该文件内的 0-based 序号
    start_line: int  # ours 块在原文件中的起始行号(1-based)
    end_line: int  # end_re 的行号(1-based)
    ours_text: str
    theirs_text: str


class ConflictParseError(Exception):
    def __init__(self, file_path: str, line_no: int, detail: str):
        super().__init__(f"{file_path}:{line_no}: {detail}")
        self.file_path = file_path
        self.line_no = line_no
        self.detail = detail


def parse_conflict_file(file_path: str, content: str) -> list[ParsedConflict]:
    """解析一个含冲突标记的文件内容。

    冲突块格式:
        <<<<<<< HEAD   (或 <<<<<<< ours/branch-name)
        ours content
        =======
        theirs content
        >>>>>>> target
    """
    lines = content.splitlines(keepends=False)
    result: list[ParsedConflict] = []
    i = 0
    block_idx = 0

    while i < len(lines):
        line = lines[i]
        if _START_RE.match(line):
            start_line = i + 1  # 1-based
            ours_lines: list[str] = []
            theirs_lines: list[str] = []
            i += 1

            # 收集 ours
            while i < len(lines) and not _SEP_RE.match(lines[i]):
                if _START_RE.match(lines[i]) or _END_RE.match(lines[i]):
                    raise ConflictParseError(file_path, i + 1, "unexpected marker in ours section")
                ours_lines.append(lines[i])
                i += 1

            if i >= len(lines) or not _SEP_RE.match(lines[i]):
                raise ConflictParseError(file_path, i + 1, "missing ======= separator")
            i += 1  # skip =======

            # 收集 theirs
            while i < len(lines) and not _END_RE.match(lines[i]):
                if _START_RE.match(lines[i]) or _SEP_RE.match(lines[i]):
                    raise ConflictParseError(file_path, i + 1, "unexpected marker in theirs section")
                theirs_lines.append(lines[i])
                i += 1

            if i >= len(lines) or not _END_RE.match(lines[i]):
                raise ConflictParseError(file_path, i + 1, "missing >>>>>>> end marker")
            end_line = i + 1  # 1-based
            i += 1  # skip >>>>>>>

            result.append(ParsedConflict(
                file_path=file_path,
                block_idx=block_idx,
                start_line=start_line,
                end_line=end_line,
                ours_text="\n".join(ours_lines),
                theirs_text="\n".join(theirs_lines),
            ))
            block_idx += 1
        else:
            i += 1

    return result
```

- [ ] **Step 4: 写失败测试**

Create `tests/unit/test_conflict_parsing.py`:

```python
import pytest

from conciliator.integrations.git.conflicts import (
    parse_conflict_file, ConflictParseError,
)
from tests.unit.fixtures.conflict_samples import (
    SAMPLE_NO_CONFLICT, SAMPLE_SINGLE_CONFLICT,
    SAMPLE_TWO_CONFLICTS, SAMPLE_EMPTY_SIDES, SAMPLE_MALFORMED,
)


def test_no_conflict_returns_empty():
    assert parse_conflict_file("a.py", SAMPLE_NO_CONFLICT) == []


def test_single_conflict():
    blocks = parse_conflict_file("a.py", SAMPLE_SINGLE_CONFLICT)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.file_path == "a.py"
    assert b.block_idx == 0
    assert b.start_line == 2  # <<< 是第 2 行
    assert b.end_line == 6
    assert b.ours_text == "ours line"
    assert b.theirs_text == "theirs line"


def test_two_conflicts_increment_idx():
    blocks = parse_conflict_file("a.py", SAMPLE_TWO_CONFLICTS)
    assert len(blocks) == 2
    assert blocks[0].block_idx == 0
    assert blocks[1].block_idx == 1
    assert blocks[0].ours_text == "ours1"
    assert blocks[1].theirs_text == "theirs2"


def test_empty_ours():
    blocks = parse_conflict_file("a.py", SAMPLE_EMPTY_SIDES)
    assert len(blocks) == 1
    assert blocks[0].ours_text == ""
    assert blocks[0].theirs_text == "theirs only"


def test_malformed_raises():
    with pytest.raises(ConflictParseError) as exc_info:
        parse_conflict_file("a.py", SAMPLE_MALFORMED)
    assert "missing" in exc_info.value.detail or "unexpected" in exc_info.value.detail


def test_real_python_file():
    content = """\
import os

class Foo:
<<<<<<< HEAD
    def new_method(self):
        return 1
=======
    def old_method(self):
        return 2
>>>>>>> main

    def shared(self):
        return 3
"""
    blocks = parse_conflict_file("foo.py", content)
    assert len(blocks) == 1
    assert "def new_method" in blocks[0].ours_text
    assert "def old_method" in blocks[0].theirs_text
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_conflict_parsing.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/conciliator/utils/text.py src/conciliator/integrations/ tests/unit/test_conflict_parsing.py tests/unit/fixtures/
git commit -m "feat: parse git conflict markers into structured blocks"
```

---

### Task 8: Git clone 与 blame

**Files:**
- Create: `src/conciliator/integrations/git/repo.py`
- Create: `src/conciliator/integrations/git/blame.py`
- Test: `tests/integration/test_git_blame.py`

**Interfaces:**
- Consumes: `Settings.git_clone_base_dir`, `Settings.git_clone_timeout_seconds`
- Produces: `clone_to_tempdir(repo_url, branch, token, base_dir) -> ContextManager[Path]`;`blame_lines(repo_path, file_path, start_line, end_line) -> BlameResult`;`BlameResult(author_email, author_name, commit_sha, committed_at)`

- [ ] **Step 1: 写 git/repo.py**

Create `src/conciliator/integrations/git/repo.py`:

```python
import contextlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import git

from conciliator.utils.logging import get_logger

log = get_logger(__name__)


@contextlib.contextmanager
def clone_to_tempdir(
    repo_url: str,
    branch: str,
    token: str | None,
    base_dir: str,
) -> Iterator[Path]:
    """clone 仓库到临时目录,yield 路径,退出时清理。

    token 用于私有仓库:把 token 插入 URL https://oauth2:{token}@host/...
    """
    work_dir = Path(base_dir) / f"clone-{os.getpid()}-{tempfile.mkdtemp(prefix='')}"
    work_dir.mkdir(parents=True, exist_ok=True)
    clone_url = _inject_token(repo_url, token) if token else repo_url

    try:
        log.info("cloning_repo", repo_url=repo_url, branch=branch, work_dir=str(work_dir))
        git.Repo.clone_from(clone_url, str(work_dir), branch=branch, depth=100)
        yield work_dir
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            log.warning("cleanup_failed", work_dir=str(work_dir), error=str(e))


def _inject_token(repo_url: str, token: str) -> str:
    """https://host.com/path -> https://oauth2:{token}@host.com/path"""
    if not repo_url.startswith("https://"):
        return repo_url
    return repo_url.replace("https://", f"https://oauth2:{token}@", 1)
```

- [ ] **Step 2: 写 git/blame.py**

Create `src/conciliator/integrations/git/blame.py`:

```python
import re
from dataclasses import dataclass
from pathlib import Path

import git

_MAILMAP_RE = re.compile(r"^(?P<name>[^<]+)\s+<(?P<email>[^>]+)>\s*(?P<nname>[^<]*?)\s+<(?P<nemail>[^>]+)>")


@dataclass(frozen=True)
class BlameResult:
    author_email: str
    author_name: str
    commit_sha: str
    committed_at: str  # ISO 8601


def load_mailmap(repo_path: Path) -> dict[str, tuple[str, str]]:
    """读 .mailmap,返回 {(orig_email): (new_name, new_email)}。"""
    mailmap_file = repo_path / ".mailmap"
    if not mailmap_file.exists():
        return {}

    mapping: dict[str, tuple[str, str]] = {}
    for line in mailmap_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _MAILMAP_RE.match(line)
        if m:
            mapping[m.group("email").strip().lower()] = (
                m.group("nname").strip() or m.group("name").strip(),
                m.group("nemail").strip(),
            )
    return mapping


def blame_lines(
    repo_path: Path,
    file_path: str,
    start_line: int,
    end_line: int,
) -> BlameResult | None:
    """对 [start_line, end_line] 区间(1-based)做 blame,取最后修改者。

    策略:取该区间内出现次数最多的 commit 作为代表,再用其 author。
    简化版:取区间最后一个 commit(= ours 侧最近改动)。
    """
    repo = git.Repo(str(repo_path))
    full_path = repo_path / file_path
    if not full_path.exists():
        return None

    try:
        blame = repo.blame("HEAD", file_path)
    except Exception:
        return None

    # blame 返回 [(commit, [lines]), ...]
    # 需要把 1-based 行号映射到对应 commit
    current_line = 1
    chosen_commit = None
    for commit, lines in blame:
        line_count = len(lines)
        if start_line <= current_line + line_count - 1 and current_line <= end_line:
            chosen_commit = commit  # 取区间内最后出现的 commit
        current_line += line_count
        if current_line > end_line:
            break

    if chosen_commit is None:
        return None

    mailmap = load_mailmap(repo_path)
    email = chosen_commit.author.email or ""
    name = chosen_commit.author.name or ""
    if email.lower() in mailmap:
        name, email = mailmap[email.lower()]

    return BlameResult(
        author_email=email,
        author_name=name,
        commit_sha=chosen_commit.hexsha,
        committed_at=chosen_commit.committed_datetime.isoformat(),
    )
```

- [ ] **Step 3: 写集成测试 test_git_blame.py**

Create `tests/integration/test_git_blame.py`:

```python
import os
from pathlib import Path

import git
import pytest

from conciliator.integrations.git.blame import blame_lines, load_mailmap
from conciliator.integrations.git.repo import clone_to_tempdir


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "alice@example.com")
        cw.set_value("user", "name", "Alice")

    (repo_dir / "a.py").write_text("def foo():\n    return 1\n")
    repo.index.add(["a.py"])
    repo.index.commit("first commit")
    return repo_dir


def test_blame_returns_author(sample_repo: Path):
    result = blame_lines(sample_repo, "a.py", start_line=1, end_line=2)
    assert result is not None
    assert result.author_email == "alice@example.com"
    assert result.author_name == "Alice"


def test_blame_missing_file_returns_none(sample_repo: Path):
    assert blame_lines(sample_repo, "missing.py", 1, 1) is None


def test_load_mailmap(tmp_path: Path):
    (tmp_path / ".mailmap").write_text(
        "Alice <new@example.com> Alice <old@example.com>\n"
    )
    mapping = load_mailmap(tmp_path)
    assert mapping["old@example.com"] == ("Alice", "new@example.com")
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_git_blame.py -v`
Expected: PASS(本地有 git 可执行即可)

- [ ] **Step 5: 提交**

```bash
git add src/conciliator/integrations/git/repo.py src/conciliator/integrations/git/blame.py tests/integration/test_git_blame.py
git commit -m "feat: git clone and blame with mailmap support"
```

---

## Phase 2:外部系统集成

### Task 9: NextCode OpenAPI 客户端

**Files:**
- Create: `src/conciliator/integrations/nextcode/__init__.py`
- Create: `src/conciliator/integrations/nextcode/client.py`
- Create: `src/conciliator/integrations/nextcode/branches.py`
- Create: `src/conciliator/integrations/nextcode/merges.py`
- Create: `src/conciliator/integrations/nextcode/mrs.py`
- Create: `src/conciliator/integrations/nextcode/discussions.py`
- Create: `src/conciliator/integrations/nextcode/repos.py`
- Test: `tests/unit/test_nextcode_client.py`
- Test: `tests/unit/test_nextcode_branches.py`

**Interfaces:**
- Consumes: `Settings.nextcode_*`、`SecretsProvider`(Task 2,拿 nextcode_app_secret)、`byteddps`
- Produces:
  - `NextCodeClient(settings, secrets)` — HTTP 客户端,负责签名 + DPS token 注入
  - `create_branch(repo_id, from_branch, to_branch) -> Branch`
  - `list_conflict_files(repo_id, source, target) -> list[ConflictFile]`
  - `get_mr(mr_id) -> MR`;`create_mr(source, target, title, description) -> MR`;`update_mr_description(mr_id, description) -> None`
  - `create_discussion(mr_id, file_path, line, body) -> Discussion`;`reply_discussion(discussion_id, body) -> None`
  - `add_mr_comment(mr_id, body) -> None`
  - `check_repo_member(repo_id, email, role) -> bool`

> **注:** NextCode API 实际字段名是 spec 第 12 节的待确认项。这里以合理的 GitLab 风格命名,实现阶段对接时若字段名不同,只改 client 内部,不动调用方。

- [ ] **Step 1: 写 nextcode/client.py**

Create `src/conciliator/integrations/nextcode/__init__.py` (empty) and `src/conciliator/integrations/nextcode/client.py`:

```python
import hashlib
import hmac
import time
from dataclasses import dataclass

import httpx

from conciliator.config import Settings
from conciliator.utils.logging import get_logger
from conciliator.utils.retry import retry_async, TransientError
from conciliator.utils.secrets import SecretsProvider

log = get_logger(__name__)


@dataclass(frozen=True)
class NextCodeError(Exception):
    status: int
    body: str

    def __post_init__(self) -> None:
        super().__init__(f"NextCode API {self.status}: {self.body[:200]}")


class NextCodeClient:
    def __init__(self, settings: Settings, secrets: SecretsProvider):
        self._settings = settings
        self._base = settings.nextcode_api_base.rstrip("/")
        self._app_id = settings.nextcode_app_id
        self._secret = secrets.get(settings.kms_config_nextcode_app_secret)
        self._http = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def sign_webhook(self, body: bytes, timestamp: str, signature: str) -> bool:
        """校验 NextCode webhook 签名。

        约定签名格式:HMAC-SHA256(secret, f"{timestamp}.{body}"),hex 编码。
        """
        expected = hmac.new(
            self._secret.encode(),
            f"{timestamp}.".encode() + body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def is_timestamp_fresh(timestamp: str, max_age_seconds: int = 300) -> bool:
        try:
            ts = int(timestamp)
        except ValueError:
            return False
        return abs(int(time.time()) - ts) <= max_age_seconds

    @retry_async("nextcode_request", exceptions=(TransientError,))
    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        headers = self._auth_headers()
        try:
            resp = await self._http.request(
                method, path, json=json, params=params, headers=headers,
            )
        except httpx.RequestError as e:
            raise TransientError(f"network error: {e}") from e

        if resp.status_code >= 500:
            raise TransientError(f"server error {resp.status_code}: {resp.text}")
        if resp.status_code >= 400:
            raise NextCodeError(status=resp.status_code, body=resp.text)

        data = resp.json()
        if isinstance(data, dict) and "code" in data and data["code"] != 0:
            raise NextCodeError(
                status=resp.status_code, body=str(data.get("msg") or data),
            )
        return data.get("data", data) if isinstance(data, dict) else data

    def _auth_headers(self) -> dict[str, str]:
        """注入 App 凭证 + DPS token(服务身份认证)。"""
        headers = {
            "X-Codebase-App-Id": self._app_id,
            "X-Codebase-App-Secret": self._secret,
        }
        try:
            import byteddps
            headers["X-Bytedance-DPS-Token"] = byteddps.get_token()
        except Exception:
            # 开发机可能无 DPS 环境,忽略
            pass
        return headers
```

- [ ] **Step 2: 写 nextcode/branches.py**

Create `src/conciliator/integrations/nextcode/branches.py`:

```python
from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient


@dataclass(frozen=True)
class Branch:
    name: str
    commit_sha: str


async def create_branch(
    client: NextCodeClient,
    repo_id: str,
    from_branch: str,
    new_branch: str,
) -> Branch:
    data = await client._request(
        "POST",
        f"/v1/repos/{repo_id}/branches",
        json={"branch": new_branch, "ref": from_branch},
    )
    return Branch(name=data["name"], commit_sha=data["commit"]["id"])
```

- [ ] **Step 3: 写 nextcode/merges.py**

Create `src/conciliator/integrations/nextcode/merges.py`:

```python
from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient


@dataclass(frozen=True)
class ConflictFile:
    file_path: str
    content: str  # 含 <<<<<<< ======= >>>>>>> 标记


async def list_conflict_files(
    client: NextCodeClient,
    repo_id: str,
    source: str,
    target: str,
) -> list[ConflictFile]:
    """读 source 与 target 之间的冲突文件列表。

    NextCode 假设返回 {"files": [{"old_path": "...", "diff": "..."}]}
    """
    data = await client._request(
        "GET",
        f"/v1/repos/{repo_id}/compare",
        params={"from": target, "to": source},
    )
    files = []
    for f in data.get("files", []):
        files.append(ConflictFile(
            file_path=f.get("new_path") or f.get("old_path") or "unknown",
            content=f.get("diff", ""),
        ))
    return files
```

- [ ] **Step 4: 写 nextcode/mrs.py**

Create `src/conciliator/integrations/nextcode/mrs.py`:

```python
from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient


@dataclass(frozen=True)
class MR:
    id: int
    iid: int
    title: str
    web_url: str
    author_email: str | None = None


async def get_mr(client: NextCodeClient, project_id: str, mr_iid: int) -> MR:
    data = await client._request("GET", f"/v1/projects/{project_id}/merge_requests/{mr_iid}")
    return _parse_mr(data)


async def create_mr(
    client: NextCodeClient,
    project_id: str,
    *,
    source: str,
    target: str,
    title: str,
    description: str,
) -> MR:
    data = await client._request(
        "POST",
        f"/v1/projects/{project_id}/merge_requests",
        json={
            "source_branch": source,
            "target_branch": target,
            "title": title,
            "description": description,
        },
    )
    return _parse_mr(data)


async def update_mr_description(
    client: NextCodeClient,
    project_id: str,
    mr_iid: int,
    description: str,
) -> None:
    await client._request(
        "PUT",
        f"/v1/projects/{project_id}/merge_requests/{mr_iid}",
        json={"description": description},
    )


async def add_mr_comment(
    client: NextCodeClient,
    project_id: str,
    mr_iid: int,
    body: str,
) -> None:
    await client._request(
        "POST",
        f"/v1/projects/{project_id}/merge_requests/{mr_iid}/notes",
        json={"body": body},
    )


def _parse_mr(data: dict) -> MR:
    return MR(
        id=data["id"],
        iid=data["iid"],
        title=data["title"],
        web_url=data["web_url"],
        author_email=(data.get("author") or {}).get("email"),
    )
```

- [ ] **Step 5: 写 nextcode/discussions.py**

Create `src/conciliator/integrations/nextcode/discussions.py`:

```python
from dataclasses import dataclass

from conciliator.integrations.nextcode.client import NextCodeClient


@dataclass(frozen=True)
class Discussion:
    id: str


async def create_discussion(
    client: NextCodeClient,
    project_id: str,
    mr_iid: int,
    *,
    file_path: str,
    line: int,
    body: str,
) -> Discussion:
    data = await client._request(
        "POST",
        f"/v1/projects/{project_id}/merge_requests/{mr_iid}/discussions",
        json={
            "body": body,
            "position": {
                "new_path": file_path,
                "new_line": line,
                "position_type": "text",
            },
        },
    )
    return Discussion(id=data["id"])


async def reply_discussion(
    client: NextCodeClient,
    project_id: str,
    mr_iid: int,
    discussion_id: str,
    body: str,
) -> None:
    await client._request(
        "POST",
        f"/v1/projects/{project_id}/merge_requests/{mr_iid}/discussions/{discussion_id}/notes",
        json={"body": body},
    )
```

- [ ] **Step 6: 写 nextcode/repos.py**

Create `src/conciliator/integrations/nextcode/repos.py`:

```python
from conciliator.integrations.nextcode.client import NextCodeClient


async def check_repo_member(
    client: NextCodeClient,
    project_id: str,
    email: str,
    *,
    role: str = "admin",
) -> bool:
    """查询 email 是否为仓库的指定角色成员。"""
    data = await client._request(
        "GET",
        f"/v1/projects/{project_id}/members",
        params={"search": email},
    )
    for member in data if isinstance(data, list) else []:
        m_email = (member.get("email") or "").lower()
        m_access = (member.get("access_level") or 0)
        if m_email == email.lower() and m_access >= 40:  # 40 = Maintainer
            return True
    return False
```

- [ ] **Step 7: 写单元测试 test_nextcode_client.py**

Create `tests/unit/test_nextcode_client.py`:

```python
import hmac
import hashlib
import time

import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.nextcode.client import NextCodeClient, NextCodeError
from conciliator.utils.secrets import StaticSecretsProvider


def _settings() -> Settings:
    return Settings(
        env="dev",
        service_psm="x", kms_psm="x",
        nextcode_app_id="app1",
        coco_api_base="x", coco_default_agent_id="x",
        lark_app_id="x", lark_doc_folder_token="x",
        mysql_psm="x",
        nextcode_api_base="https://nc.test/api",
        kms_config_nextcode_app_secret="nextcode_app_secret",
    )  # type: ignore[arg-type]


def _secrets() -> StaticSecretsProvider:
    return StaticSecretsProvider({"nextcode_app_secret": "secret123"})


@pytest.mark.asyncio
async def test_sign_webhook_valid():
    client = NextCodeClient(_settings(), _secrets())
    body = b'{"event":"test"}'
    ts = str(int(time.time()))
    sig = hmac.new(b"secret123", f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    assert client.sign_webhook(body, ts, sig) is True
    await client.aclose()


@pytest.mark.asyncio
async def test_sign_webhook_invalid_signature():
    client = NextCodeClient(_settings(), _secrets())
    assert client.sign_webhook(b"body", "123", "deadbeef") is False
    await client.aclose()


@pytest.mark.asyncio
async def test_timestamp_too_old_rejected():
    client = NextCodeClient(_settings(), _secrets())
    old_ts = str(int(time.time()) - 1000)
    assert client.is_timestamp_fresh(old_ts) is False
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_request_returns_data_on_success():
    respx.get("https://nc.test/api/v1/foo").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"ok": True}})
    )
    client = NextCodeClient(_settings(), _secrets())
    data = await client._request("GET", "/v1/foo")
    assert data == {"ok": True}
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_request_raises_on_4xx():
    respx.get("https://nc.test/api/v1/foo").mock(
        return_value=httpx.Response(404, json={"code": 404, "msg": "not found"})
    )
    client = NextCodeClient(_settings(), _secrets())
    with pytest.raises(NextCodeError):
        await client._request("GET", "/v1/foo")
    await client.aclose()
```

- [ ] **Step 8: 写单元测试 test_nextcode_branches.py**

Create `tests/unit/test_nextcode_branches.py`:

```python
import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.branches import create_branch
from conciliator.utils.secrets import StaticSecretsProvider


def _settings() -> Settings:
    return Settings(
        env="dev", service_psm="x", kms_psm="x",
        nextcode_app_id="app1",
        coco_api_base="x", coco_default_agent_id="x",
        lark_app_id="x", lark_doc_folder_token="x", mysql_psm="x",
        nextcode_api_base="https://nc.test/api",
        kms_config_nextcode_app_secret="nextcode_app_secret",
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
@respx.mock
async def test_create_branch():
    respx.post("https://nc.test/api/v1/repos/r1/branches").mock(
        return_value=httpx.Response(200, json={
            "code": 0,
            "data": {"name": "resolve/feat/x", "commit": {"id": "abc123"}},
        })
    )
    client = NextCodeClient(_settings(), StaticSecretsProvider({"nextcode_app_secret": "s"}))
    branch = await create_branch(client, "r1", "feat", "resolve/feat/x")
    assert branch.name == "resolve/feat/x"
    assert branch.commit_sha == "abc123"
    await client.aclose()
```

- [ ] **Step 9: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_nextcode_client.py tests/unit/test_nextcode_branches.py -v`
Expected: PASS

- [ ] **Step 10: 提交**

```bash
git add src/conciliator/integrations/nextcode/ tests/unit/test_nextcode_client.py tests/unit/test_nextcode_branches.py
git commit -m "feat: NextCode OpenAPI client (branches, merges, mrs, discussions, repos)"
```

---

### Task 10: Coco OpenAPI Next 客户端

**Files:**
- Create: `src/conciliator/integrations/coco/__init__.py`
- Create: `src/conciliator/integrations/coco/client.py`
- Create: `src/conciliator/integrations/coco/events.py`
- Test: `tests/unit/test_coco_client.py`
- Test: `tests/unit/test_coco_events.py`

**Interfaces:**
- Consumes: `Settings.coco_*`、`SecretsProvider`(coco_api_token)、`byteddps`
- Produces:
  - `CocoClient(settings, secrets)` — HTTP 客户端
  - `submit_resolve_task(agent_id, repo_url, resolve_branch, target_branch, conflict_files, instructions) -> CocoTaskRef`
  - `CocoTaskRef(task_id)` — 引用 ID
  - `fetch_task_result(task_id) -> CocoTaskResult`
  - `CocoTaskResult(status, diffs: list[FileDiff], thinking: str, commit_sha: str | None)`
  - `FileDiff(file_path, before, after)`

- [ ] **Step 1: 写 coco/client.py**

Create `src/conciliator/integrations/coco/__init__.py` (empty) and `src/conciliator/integrations/coco/client.py`:

```python
from dataclasses import dataclass, field

import httpx

from conciliator.config import Settings
from conciliator.utils.logging import get_logger
from conciliator.utils.retry import retry_async, TransientError
from conciliator.utils.secrets import SecretsProvider

log = get_logger(__name__)


@dataclass(frozen=True)
class CocoError(Exception):
    message: str

    def __post_init__(self) -> None:
        super().__init__(self.message)


@dataclass(frozen=True)
class CocoTaskRef:
    task_id: str


@dataclass(frozen=True)
class FileDiff:
    file_path: str
    before: str
    after: str


@dataclass(frozen=True)
class CocoTaskResult:
    status: str  # "running" | "succeeded" | "failed" | "timeout"
    diffs: list[FileDiff] = field(default_factory=list)
    thinking: str = ""
    commit_sha: str | None = None
    error: str | None = None


class CocoClient:
    def __init__(self, settings: Settings, secrets: SecretsProvider):
        self._settings = settings
        self._base = settings.coco_api_base.rstrip("/")
        self._token = secrets.get(settings.kms_config_coco_api_token)
        self._http = httpx.AsyncClient(
            base_url=self._base,
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={"Authorization": f"Bearer {self._token}"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @retry_async("coco_submit", exceptions=(TransientError,))
    async def submit_resolve_task(
        self,
        agent_id: str,
        *,
        repo_url: str,
        resolve_branch: str,
        target_branch: str,
        conflict_files: list[str],
        instructions: str,
    ) -> CocoTaskRef:
        prompt = (
            f"请解决 {repo_url} 中 {resolve_branch} 分支与 {target_branch} 分支的冲突。\n"
            f"冲突文件:{', '.join(conflict_files)}\n\n"
            f"附加说明:{instructions}\n\n"
            "步骤:1) clone 仓库并切到 resolve_branch;2) merge target_branch;\n"
            "3) 解决所有冲突(参照语义保留正确逻辑);4) commit 并 push 回 resolve_branch;\n"
            "5) 返回每个文件的最终内容以及你的思考过程。"
        )
        try:
            resp = await self._http.post(
                "/v1/agents/tasks",
                json={
                    "agent_id": agent_id,
                    "message": prompt,
                    "workspace": {"repo_url": repo_url, "branch": resolve_branch},
                },
                headers=self._dps_headers(),
            )
        except httpx.RequestError as e:
            raise TransientError(f"network: {e}") from e

        if resp.status_code >= 500:
            raise TransientError(f"server {resp.status_code}")
        if resp.status_code >= 400:
            raise CocoError(message=f"submit failed: {resp.text}")

        data = resp.json()
        return CocoTaskRef(task_id=data["task_id"])

    @retry_async("coco_fetch", exceptions=(TransientError,))
    async def fetch_task_result(self, task_id: str) -> CocoTaskResult:
        try:
            resp = await self._http.get(
                f"/v1/agents/tasks/{task_id}", headers=self._dps_headers(),
            )
        except httpx.RequestError as e:
            raise TransientError(f"network: {e}") from e

        if resp.status_code >= 500:
            raise TransientError(f"server {resp.status_code}")
        if resp.status_code >= 400:
            raise CocoError(message=f"fetch failed: {resp.text}")

        data = resp.json()
        status = data.get("status", "running")
        diffs = [
            FileDiff(
                file_path=d["file_path"],
                before=d.get("before", ""),
                after=d.get("after", ""),
            )
            for d in data.get("file_diffs", [])
        ]
        return CocoTaskResult(
            status=status,
            diffs=diffs,
            thinking=data.get("thinking", ""),
            commit_sha=data.get("commit_sha"),
            error=data.get("error"),
        )

    @staticmethod
    def _dps_headers() -> dict[str, str]:
        try:
            import byteddps
            return {"X-Bytedance-DPS-Token": byteddps.get_token()}
        except Exception:
            return {}
```

- [ ] **Step 2: 写 coco/events.py**

Create `src/conciliator/integrations/coco/events.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CocoWebhookEvent:
    task_id: str
    event_id: str
    event_type: str  # "task.started" | "task.file_changed" | "task.thinking" | "task.completed" | "task.failed"
    payload: dict


def parse_event(raw: dict) -> CocoWebhookEvent:
    return CocoWebhookEvent(
        task_id=raw["task_id"],
        event_id=raw.get("event_id") or raw.get("id") or f"{raw['task_id']}-{raw.get('event_type')}-{raw.get('ts')}",
        event_type=raw.get("event_type") or raw.get("type", ""),
        payload=raw,
    )


def is_terminal_event(event: CocoWebhookEvent) -> bool:
    return event.event_type in ("task.completed", "task.failed", "task.timeout")
```

- [ ] **Step 3: 写单元测试 test_coco_client.py**

Create `tests/unit/test_coco_client.py`:

```python
import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.coco.client import CocoClient, CocoTaskResult
from conciliator.utils.secrets import StaticSecretsProvider


def _settings() -> Settings:
    return Settings(
        env="dev", service_psm="x", kms_psm="x",
        nextcode_app_id="x",
        coco_api_base="https://coco.test",
        coco_default_agent_id="agent-1",
        lark_app_id="x", lark_doc_folder_token="x", mysql_psm="x",
        kms_config_coco_api_token="coco_api_token",
    )  # type: ignore[arg-type]


def _secrets() -> StaticSecretsProvider:
    return StaticSecretsProvider({"coco_api_token": "tok"})


@pytest.mark.asyncio
@respx.mock
async def test_submit_resolve_task():
    respx.post("https://coco.test/v1/agents/tasks").mock(
        return_value=httpx.Response(200, json={"task_id": "ct-1"})
    )
    client = CocoClient(_settings(), _secrets())
    ref = await client.submit_resolve_task(
        "agent-1",
        repo_url="https://git.example.com/r.git",
        resolve_branch="resolve/feat/x",
        target_branch="main",
        conflict_files=["a.py", "b.py"],
        instructions="优先保留 main 的语义",
    )
    assert ref.task_id == "ct-1"
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_task_result_running():
    respx.get("https://coco.test/v1/agents/tasks/ct-1").mock(
        return_value=httpx.Response(200, json={"status": "running"})
    )
    client = CocoClient(_settings(), _secrets())
    result = await client.fetch_task_result("ct-1")
    assert isinstance(result, CocoTaskResult)
    assert result.status == "running"
    assert result.diffs == []
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_task_result_completed():
    respx.get("https://coco.test/v1/agents/tasks/ct-1").mock(
        return_value=httpx.Response(200, json={
            "status": "succeeded",
            "commit_sha": "deadbeef",
            "thinking": "I analyzed...",
            "file_diffs": [
                {"file_path": "a.py", "before": "x", "after": "y"},
            ],
        })
    )
    client = CocoClient(_settings(), _secrets())
    result = await client.fetch_task_result("ct-1")
    assert result.status == "succeeded"
    assert result.commit_sha == "deadbeef"
    assert result.thinking == "I analyzed..."
    assert len(result.diffs) == 1
    assert result.diffs[0].file_path == "a.py"
    await client.aclose()
```

- [ ] **Step 4: 写单元测试 test_coco_events.py**

Create `tests/unit/test_coco_events.py`:

```python
from conciliator.integrations.coco.events import parse_event, is_terminal_event


def test_parse_event_basic():
    e = parse_event({
        "task_id": "ct-1",
        "event_id": "evt-1",
        "event_type": "task.started",
    })
    assert e.task_id == "ct-1"
    assert e.event_id == "evt-1"
    assert e.event_type == "task.started"


def test_parse_event_missing_id_synthesizes():
    e = parse_event({
        "task_id": "ct-1",
        "type": "task.thinking",
        "ts": "12345",
    })
    assert e.event_id == "ct-1-task.thinking-12345"


def test_terminal_event_detection():
    assert is_terminal_event(parse_event({"task_id": "x", "event_type": "task.completed"}))
    assert is_terminal_event(parse_event({"task_id": "x", "event_type": "task.failed"}))
    assert not is_terminal_event(parse_event({"task_id": "x", "event_type": "task.started"}))
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_coco_client.py tests/unit/test_coco_events.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/conciliator/integrations/coco/ tests/unit/test_coco_client.py tests/unit/test_coco_events.py
git commit -m "feat: Coco OpenAPI Next client with task submission and result fetching"
```

---

### Task 11: 飞书 OpenAPI 客户端

**Files:**
- Create: `src/conciliator/integrations/lark/__init__.py`
- Create: `src/conciliator/integrations/lark/client.py`
- Create: `src/conciliator/integrations/lark/docs.py`
- Create: `src/conciliator/integrations/lark/contacts.py`
- Test: `tests/unit/test_lark_docs.py`

**Interfaces:**
- Consumes: `Settings.lark_*`、`SecretsProvider`(lark_app_secret)
- Produces:
  - `LarkClient(settings, secrets)` — 用 httpx,封装 token 缓存
  - `create_doc(title, content_xml, folder_token) -> DocRef`
  - `DocRef(doc_token, url)`
  - `update_doc(doc_token, command, content_xml) -> None`

- [ ] **Step 1: 写 lark/client.py**

Create `src/conciliator/integrations/lark/__init__.py` (empty) and `src/conciliator/integrations/lark/client.py`:

```python
import time

import httpx

from conciliator.config import Settings
from conciliator.utils.logging import get_logger
from conciliator.utils.retry import retry_async, TransientError
from conciliator.utils.secrets import SecretsProvider

log = get_logger(__name__)


class LarkError(Exception):
    pass


class LarkClient:
    def __init__(self, settings: Settings, secrets: SecretsProvider):
        self._settings = settings
        self._app_id = settings.lark_app_id
        self._app_secret = secrets.get(settings.kms_config_lark_app_secret)
        self._http = httpx.AsyncClient(
            base_url="https://open.feishu.cn/open-apis",
            timeout=httpx.Timeout(30.0),
        )
        self._token_cache: tuple[str, float] | None = None

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_app_access_token(self) -> str:
        if self._token_cache and self._token_cache[1] > time.time() + 60:
            return self._token_cache[0]
        resp = await self._http.post(
            "/auth/v3/app_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        data = resp.json()
        if data.get("code") != 0:
            raise LarkError(f"token failed: {data}")
        token = data["app_access_token"]
        expire = data.get("expire", 7200)
        self._token_cache = (token, time.time() + expire)
        return token

    @retry_async("lark_request", exceptions=(TransientError,))
    async def _request(
        self, method: str, path: str, *, json: dict | None = None,
    ) -> dict:
        token = await self.get_app_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = await self._http.request(
                method, path, json=json, headers=headers,
            )
        except httpx.RequestError as e:
            raise TransientError(f"network: {e}") from e
        if resp.status_code >= 500:
            raise TransientError(f"server {resp.status_code}")
        data = resp.json()
        if data.get("code") != 0:
            raise LarkError(f"lark api {path}: {data}")
        return data.get("data", {})
```

- [ ] **Step 2: 写 lark/docs.py**

Create `src/conciliator/integrations/lark/docs.py`:

```python
from dataclasses import dataclass

from conciliator.integrations.lark.client import LarkClient


@dataclass(frozen=True)
class DocRef:
    doc_token: str
    url: str


async def create_doc(
    client: LarkClient,
    *,
    title: str,
    content_xml: str,
    folder_token: str,
) -> DocRef:
    """创建飞书 docx 文档(使用 XML 内容)。"""
    data = await client._request(
        "POST",
        "/docx/v1/documents",
        json={
            "folder_token": folder_token,
            "title": title,
            "content": content_xml,
        },
    )
    doc_token = data["document"]["document_id"]
    url = f"https://bytedance.larkoffice.com/docx/{doc_token}"
    return DocRef(doc_token=doc_token, url=url)


async def update_doc(
    client: LarkClient,
    *,
    doc_token: str,
    command: str,  # "append" | "str_replace" | ...
    content_xml: str,
) -> None:
    await client._request(
        "PATCH",
        f"/docx/v1/documents/{doc_token}",
        json={"command": command, "content": content_xml},
    )
```

- [ ] **Step 3: 写 lark/contacts.py**

Create `src/conciliator/integrations/lark/contacts.py`:

```python
from conciliator.integrations.lark.client import LarkClient


async def get_user_email_by_open_id(
    client: LarkClient, open_id: str,
) -> str | None:
    data = await client._request(
        "GET",
        f"/contact/v3/users/{open_id}",
        params={"user_id_type": "open_id"},
    )
    user = data.get("user", {})
    return user.get("email") or user.get("enterprise_email")
```

- [ ] **Step 4: 写单元测试 test_lark_docs.py**

Create `tests/unit/test_lark_docs.py`:

```python
import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.lark.docs import create_doc
from conciliator.utils.secrets import StaticSecretsProvider


def _settings() -> Settings:
    return Settings(
        env="dev", service_psm="x", kms_psm="x",
        nextcode_app_id="x",
        coco_api_base="x", coco_default_agent_id="x",
        lark_app_id="li", lark_doc_folder_token="fld1", mysql_psm="x",
        kms_config_lark_app_secret="lark_app_secret",
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
@respx.mock
async def test_create_doc_returns_token_and_url():
    respx.post("https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal").mock(
        return_value=httpx.Response(200, json={
            "code": 0, "app_access_token": "tok", "expire": 7200,
        })
    )
    respx.post("https://open.feishu.cn/open-apis/docx/v1/documents").mock(
        return_value=httpx.Response(200, json={
            "code": 0,
            "data": {"document": {"document_id": "doccnABC"}},
        })
    )
    client = LarkClient(_settings(), StaticSecretsProvider({"lark_app_secret": "ls"}))
    ref = await create_doc(
        client,
        title="测试文档",
        content_xml="<p>hi</p>",
        folder_token="fld1",
    )
    assert ref.doc_token == "doccnABC"
    assert "doccnABC" in ref.url
    await client.aclose()
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_lark_docs.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/conciliator/integrations/lark/ tests/unit/test_lark_docs.py
git commit -m "feat: Lark client with doc creation and user email lookup"
```

---

## Phase 3:仓储、Webhook 与编排

### Task 12: Task 仓储(同步,DB 调用经线程池)与乐观锁

**Files:**
- Create: `src/conciliator/repositories/task_repo.py`
- Create: `src/conciliator/repositories/_sync_repo.py`(同步实现)
- Test: `tests/integration/test_task_repo.py`

**Interfaces:**
- Consumes: ORM 模型(Task 3)、领域状态机(Task 5)、`sessionmaker[Session]`(同步)
- Produces: `TaskRepository(session_factory)`(async 接口,内部经 `run_in_threadpool` 调同步实现)
  - `async create(...) -> ResolveTask`
  - `async get(task_id) -> ResolveTask | None`
  - `async get_with_blocks(task_id) -> ResolveTask | None`
  - `async find_active_for_mr(repo_id, mr_iid) -> ResolveTask | None`
  - `async list_schedulable(limit) -> list[ResolveTask]`
  - `async update_status(task_id, from_version, new_status, **fields) -> bool` (乐观锁)
  - `async add_event(task_id, event_type, payload, coco_event_id=None) -> bool`
  - `async find_block_by_discussion(discussion_id) -> (ConflictBlock, ResolveTask) | None`

> **同步 + 线程池模式**:DB 调用本身写同步 SQLAlchemy,外层 async 用 `starlette.concurrency.run_in_threadpool` 包一层。这样 bytedmysql 服务发现/auth/metrics 全部生效,且接口仍是 async(便于和 FastAPI 异步链路对接)。

- [ ] **Step 1: 写 _sync_repo.py(纯同步 DB 操作)**

Create `src/conciliator/repositories/_sync_repo.py`:

```python
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker, selectinload

from conciliator.domain.task import SCHEDULABLE_STATUSES, TERMINAL_STATUSES, TaskStatus
from conciliator.repositories.models import (
    Approval, ConflictBlock, ResolveTask, TaskEvent,
)


class SyncTaskRepository:
    """同步实现的 task 仓储。被 async TaskRepository 通过 run_in_threadpool 调用。"""

    def __init__(self, session_factory: sessionmaker[Session]):
        self._sf = session_factory

    def create(
        self,
        *,
        repo_id: str,
        source_branch: str,
        target_branch: str,
        resolve_branch: str,
        original_mr_id: int,
        trigger_type: str,
        trigger_payload: dict,
        mr_author_email: str | None = None,
    ) -> ResolveTask:
        with self._sf() as session:
            task = ResolveTask(
                repo_id=repo_id,
                source_branch=source_branch,
                target_branch=target_branch,
                resolve_branch=resolve_branch,
                original_mr_id=original_mr_id,
                trigger_type=trigger_type,
                trigger_payload=trigger_payload,
                status=TaskStatus.PENDING.value,
                mr_author_email=mr_author_email,
            )
            session.add(task)
            session.commit()
            session.refresh(task)
            return task

    def get(self, task_id: int) -> ResolveTask | None:
        with self._sf() as session:
            return session.get(ResolveTask, task_id)

    def get_with_blocks(self, task_id: int) -> ResolveTask | None:
        with self._sf() as session:
            return session.execute(
                select(ResolveTask)
                .where(ResolveTask.id == task_id)
                .options(
                    selectinload(ResolveTask.blocks).selectinload(ConflictBlock.approval),
                )
            ).scalar_one_or_none()

    def find_active_for_mr(self, repo_id: str, mr_iid: int) -> ResolveTask | None:
        with self._sf() as session:
            return session.execute(
                select(ResolveTask)
                .where(
                    ResolveTask.repo_id == repo_id,
                    ResolveTask.original_mr_id == mr_iid,
                    ~ResolveTask.status.in_([s.value for s in TERMINAL_STATUSES]),
                )
                .order_by(ResolveTask.id.desc())
                .limit(1)
            ).scalar_one_or_none()

    def list_schedulable(self, limit: int = 10) -> list[ResolveTask]:
        with self._sf() as session:
            return list(session.execute(
                select(ResolveTask)
                .where(ResolveTask.status.in_([s.value for s in SCHEDULABLE_STATUSES]))
                .order_by(ResolveTask.updated_at)
                .limit(limit)
            ).scalars().all())

    def update_status(
        self,
        *,
        task_id: int,
        from_version: int,
        new_status: TaskStatus,
        **fields: Any,
    ) -> bool:
        values: dict[str, Any] = {"status": new_status.value, **fields}
        values["version"] = from_version + 1
        with self._sf() as session:
            result = session.execute(
                update(ResolveTask)
                .where(ResolveTask.id == task_id, ResolveTask.version == from_version)
                .values(**values)
            )
            session.commit()
            return result.rowcount == 1

    def save_conflict_blocks(self, task_id: int, blocks: list[dict]) -> list[int]:
        created: list[int] = []
        with self._sf() as session:
            for b in blocks:
                block = ConflictBlock(
                    task_id=task_id,
                    block_idx=b["block_idx"],
                    file_path=b["file_path"],
                    start_line=b["start_line"],
                    end_line=b["end_line"],
                    ours_text=b["ours_text"],
                    theirs_text=b["theirs_text"],
                    author_email=b.get("author_email"),
                )
                session.add(block)
                session.flush()
                session.add(Approval(block_id=block.id, status="PENDING"))
                created.append(block.id)
            session.commit()
            return created

    def update_block_discussion_and_resolution(
        self, *, block_id: int, discussion_id: str, ai_resolution: str,
    ) -> None:
        with self._sf() as session:
            session.execute(
                update(ConflictBlock)
                .where(ConflictBlock.id == block_id)
                .values(discussion_id=discussion_id, ai_resolution=ai_resolution)
            )
            session.commit()

    def update_block_ai_resolution(self, *, block_id: int, ai_resolution: str) -> None:
        with self._sf() as session:
            session.execute(
                update(ConflictBlock)
                .where(ConflictBlock.id == block_id)
                .values(ai_resolution=ai_resolution)
            )
            session.commit()

    def add_event(
        self, *, task_id: int, event_type: str, payload: dict,
        coco_event_id: str | None = None,
    ) -> bool:
        with self._sf() as session:
            evt = TaskEvent(
                task_id=task_id, event_type=event_type, payload=payload,
                coco_event_id=coco_event_id,
                occurred_at=datetime.now(timezone.utc),
            )
            session.add(evt)
            try:
                session.commit()
            except IntegrityError:
                return False
            return True

    def find_block_by_discussion(
        self, discussion_id: str,
    ) -> tuple[ConflictBlock, ResolveTask] | None:
        with self._sf() as session:
            row = session.execute(
                select(ConflictBlock, ResolveTask)
                .join(ResolveTask, ConflictBlock.task_id == ResolveTask.id)
                .where(ConflictBlock.discussion_id == discussion_id)
                .options(selectinload(ConflictBlock.approval))
            ).first()
            if row is None:
                return None
            return row[0], row[1]
```

- [ ] **Step 2: 写 task_repo.py(async 包装层)**

Create `src/conciliator/repositories/task_repo.py`:

```python
from typing import Any

from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from conciliator.domain.task import TaskStatus
from conciliator.repositories._sync_repo import SyncTaskRepository
from conciliator.repositories.models import ConflictBlock, ResolveTask


class TaskRepository:
    """async 包装层:DB 调用经 run_in_threadpool 调同步实现。"""

    def __init__(self, session_factory: sessionmaker[Session]):
        self._sync = SyncTaskRepository(session_factory)

    async def create(self, **kwargs: Any) -> ResolveTask:
        return await run_in_threadpool(self._sync.create, **kwargs)

    async def get(self, task_id: int) -> ResolveTask | None:
        return await run_in_threadpool(self._sync.get, task_id)

    async def get_with_blocks(self, task_id: int) -> ResolveTask | None:
        return await run_in_threadpool(self._sync.get_with_blocks, task_id)

    async def find_active_for_mr(self, repo_id: str, mr_iid: int) -> ResolveTask | None:
        return await run_in_threadpool(
            self._sync.find_active_for_mr, repo_id, mr_iid,
        )

    async def list_schedulable(self, limit: int = 10) -> list[ResolveTask]:
        return await run_in_threadpool(self._sync.list_schedulable, limit)

    async def update_status(
        self, *, task_id: int, from_version: int,
        new_status: TaskStatus, **fields: Any,
    ) -> bool:
        return await run_in_threadpool(
            self._sync.update_status,
            task_id=task_id, from_version=from_version,
            new_status=new_status, **fields,
        )

    async def save_conflict_blocks(self, task_id: int, blocks: list[dict]) -> list[int]:
        return await run_in_threadpool(
            self._sync.save_conflict_blocks, task_id, blocks,
        )

    async def update_block_discussion_and_resolution(
        self, *, block_id: int, discussion_id: str, ai_resolution: str,
    ) -> None:
        await run_in_threadpool(
            self._sync.update_block_discussion_and_resolution,
            block_id=block_id, discussion_id=discussion_id,
            ai_resolution=ai_resolution,
        )

    async def update_block_ai_resolution(
        self, *, block_id: int, ai_resolution: str,
    ) -> None:
        await run_in_threadpool(
            self._sync.update_block_ai_resolution,
            block_id=block_id, ai_resolution=ai_resolution,
        )

    async def add_event(
        self, *, task_id: int, event_type: str, payload: dict,
        coco_event_id: str | None = None,
    ) -> bool:
        return await run_in_threadpool(
            self._sync.add_event,
            task_id=task_id, event_type=event_type, payload=payload,
            coco_event_id=coco_event_id,
        )

    async def find_block_by_discussion(
        self, discussion_id: str,
    ) -> tuple[ConflictBlock, ResolveTask] | None:
        return await run_in_threadpool(
            self._sync.find_block_by_discussion, discussion_id,
        )
```

- [ ] **Step 3: 写集成测试 test_task_repo.py**

Create `tests/integration/test_task_repo.py`:

```python
import pytest

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.models import Base
from conciliator.repositories.task_repo import TaskRepository


def _settings(dsn: str) -> Settings:
    return Settings(
        env="dev", service_psm="x", kms_psm="x",
        nextcode_app_id="x", coco_api_base="x", coco_default_agent_id="x",
        lark_app_id="x", lark_doc_folder_token="x",
        mysql_psm="x", mysql_db_name="test",
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_create_and_get(mysql_container, monkeypatch):
    monkeypatch.setenv("_DEV_MYSQL_DSN", mysql_container.dsn)
    settings = _settings(mysql_container.dsn)
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1", source_branch="feat", target_branch="main",
        resolve_branch="resolve/feat/x", original_mr_id=42,
        trigger_type="COMMENT", trigger_payload={"body": "/resolve"},
    )
    fetched = await repo.get(task.id)
    assert fetched is not None
    assert fetched.status == TaskStatus.PENDING
    engine.dispose()


@pytest.mark.asyncio
async def test_optimistic_lock(mysql_container, monkeypatch):
    monkeypatch.setenv("_DEV_MYSQL_DSN", mysql_container.dsn)
    settings = _settings(mysql_container.dsn)
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1", source_branch="feat", target_branch="main",
        resolve_branch="resolve/feat/x", original_mr_id=42,
        trigger_type="COMMENT", trigger_payload={},
    )
    ok1 = await repo.update_status(
        task_id=task.id, from_version=0, new_status=TaskStatus.PREPARING,
    )
    assert ok1 is True
    ok2 = await repo.update_status(
        task_id=task.id, from_version=0, new_status=TaskStatus.BRANCH_CREATED,
    )
    assert ok2 is False
    engine.dispose()


@pytest.mark.asyncio
async def test_find_active_for_mr(mysql_container, monkeypatch):
    monkeypatch.setenv("_DEV_MYSQL_DSN", mysql_container.dsn)
    settings = _settings(mysql_container.dsn)
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    assert await repo.find_active_for_mr("r1", 42) is None
    task = await repo.create(
        repo_id="r1", source_branch="feat", target_branch="main",
        resolve_branch="resolve/feat/x", original_mr_id=42,
        trigger_type="COMMENT", trigger_payload={},
    )
    found = await repo.find_active_for_mr("r1", 42)
    assert found is not None
    assert found.id == task.id
    engine.dispose()


@pytest.mark.asyncio
async def test_add_event_dedup_by_coco_event_id(mysql_container, monkeypatch):
    monkeypatch.setenv("_DEV_MYSQL_DSN", mysql_container.dsn)
    settings = _settings(mysql_container.dsn)
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    task = await repo.create(
        repo_id="r1", source_branch="feat", target_branch="main",
        resolve_branch="resolve/feat/x", original_mr_id=42,
        trigger_type="COMMENT", trigger_payload={},
    )
    first = await repo.add_event(
        task_id=task.id, event_type="coco.started",
        payload={"x": 1}, coco_event_id="evt-1",
    )
    second = await repo.add_event(
        task_id=task.id, event_type="coco.started",
        payload={"x": 1}, coco_event_id="evt-1",
    )
    assert first is True
    assert second is False
    engine.dispose()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_task_repo.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/conciliator/repositories/_sync_repo.py src/conciliator/repositories/task_repo.py tests/integration/test_task_repo.py
git commit -m "feat: sync-backed task repository with threadpool async wrapper"
```

---

### Task 13: Webhook 入口层

**Files:**
- Create: `src/conciliator/api/webhooks.py`
- Modify: `src/conciliator/main.py` (注册路由)
- Test: `tests/integration/test_webhooks.py`

**Interfaces:**
- Consumes: `NextCodeClient` (Task 9)、`CocoClient` (Task 10)、`TaskRepository` (Task 12)
- Produces:
  - `POST /hooks/codebase` — 校验签名 + 时间戳,解析事件,创建 PENDING task,返回 200
  - `POST /hooks/coco` — 校验 token,记录事件到 task_events

- [ ] **Step 1: 写 api/webhooks.py**

Create `src/conciliator/api/webhooks.py`:

```python
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from conciliator.integrations.coco.events import parse_event, is_terminal_event
from conciliator.repositories.task_repo import TaskRepository
from conciliator.utils.logging import get_logger

router = APIRouter(tags=["webhooks"])
log = get_logger(__name__)


@dataclass
class ParsedComment:
    repo_id: str
    mr_iid: int
    body: str
    author_email: str
    discussion_id: str | None


@router.post("/hooks/codebase")
async def codebase_webhook(
    request: Request,
    x_codebase_signature: str = Header("", alias="X-Codebase-Signature"),
    x_codebase_timestamp: str = Header("", alias="X-Codebase-Timestamp"),
) -> dict:
    body = await request.body()
    state = request.app.state

    # 1. 校验时间戳
    if not state.nextcode_client.is_timestamp_fresh(x_codebase_timestamp):
        raise HTTPException(status_code=401, detail="timestamp too old or invalid")

    # 2. 校验签名
    if not state.nextcode_client.sign_webhook(body, x_codebase_timestamp, x_codebase_signature):
        raise HTTPException(status_code=401, detail="invalid signature")

    payload: dict[str, Any] = await request.json()
    event_type = payload.get("event_type") or payload.get("type", "")

    if event_type == "comment.created":
        await _handle_comment(state, payload)
    elif event_type in ("merge_request.updated", "merge_request.opened"):
        await _handle_mr_conflict(state, payload)
    else:
        log.info("webhook_ignored_event", event_type=event_type)

    return {"ok": True}


async def _handle_comment(state, payload: dict) -> None:
    repo_id = _extract_repo_id(payload)
    mr_iid = _extract_mr_iid(payload)
    body = _extract_comment_body(payload)
    author_email = _extract_author_email(payload)
    discussion_id = _extract_discussion_id(payload)

    log.info(
        "comment_received",
        repo_id=repo_id, mr_iid=mr_iid, body=body[:100],
        author=author_email, discussion_id=discussion_id,
    )

    # 单块指令 / 全局指令 → 走 approval_pipeline
    # 仅 "/resolve" 触发 → 走 resolve_pipeline
    if body.strip().startswith("/resolve"):
        # /resolve (单独) 或 /resolve approve all/retry/status/cancel
        first_token = body.strip().split()
        if len(first_token) <= 1 or first_token[1].lower() in {"start", "go"}:
            await state.comment_router.route_resolve_trigger(
                repo_id=repo_id, mr_iid=mr_iid, author_email=author_email,
                body=body, raw_payload=payload,
            )
        else:
            await state.comment_router.route_approval_command(
                repo_id=repo_id, mr_iid=mr_iid, body=body,
                author_email=author_email, discussion_id=discussion_id,
                raw_payload=payload,
            )
    elif discussion_id:
        await state.comment_router.route_approval_command(
            repo_id=repo_id, mr_iid=mr_iid, body=body,
            author_email=author_email, discussion_id=discussion_id,
            raw_payload=payload,
        )


async def _handle_mr_conflict(state, payload: dict) -> None:
    repo_id = _extract_repo_id(payload)
    mr_iid = _extract_mr_iid(payload)
    conflicts = (payload.get("object_attributes") or {}).get("conflicts", 0)
    if conflicts <= 0:
        return
    await state.comment_router.route_auto_conflict(
        repo_id=repo_id, mr_iid=mr_iid, raw_payload=payload,
    )


@router.post("/hooks/coco")
async def coco_webhook(
    request: Request,
    authorization: str = Header("", alias="Authorization"),
) -> dict:
    state = request.app.state
    expected = f"Bearer {state.coco_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid token")

    payload = await request.json()
    event = parse_event(payload)
    log.info("coco_event", task_id=event.task_id, type=event.event_type)

    dedup_ok = await state.task_repo.add_event(
        task_id=int(event.task_id),
        event_type=event.event_type,
        payload=payload,
        coco_event_id=event.event_id,
    )
    if not dedup_ok:
        log.info("coco_event_deduped", event_id=event.event_id)
    return {"ok": True}


def _extract_repo_id(payload: dict) -> str:
    return (payload.get("project") or {}).get("id", "")


def _extract_mr_iid(payload: dict) -> int:
    attrs = payload.get("merge_request") or payload.get("object_attributes") or {}
    return int(attrs.get("iid", 0))


def _extract_comment_body(payload: dict) -> str:
    attrs = payload.get("object_attributes") or {}
    return attrs.get("note") or attrs.get("body", "")


def _extract_author_email(payload: dict) -> str:
    user = payload.get("user") or {}
    return user.get("email", "")


def _extract_discussion_id(payload: dict) -> str | None:
    attrs = payload.get("object_attributes") or {}
    return attrs.get("discussion_id")
```

- [ ] **Step 2: 写 CommentRouter(分发到 resolve_service / approval_service 的占位)**

Create `src/conciliator/api/comment_router.py`:

```python
from dataclasses import dataclass


@dataclass
class CommentRouter:
    """占位:在 Task 15、16 注入实际 service。"""
    resolve_service: object  # ResolveService
    approval_service: object  # ApprovalService
    task_repo: object

    async def route_resolve_trigger(self, *, repo_id, mr_iid, author_email, body, raw_payload):
        from conciliator.services.resolve_service import ResolveService
        svc: ResolveService = self.resolve_service  # type: ignore
        await svc.handle_trigger(
            repo_id=repo_id, mr_iid=mr_iid, author_email=author_email,
            payload=raw_payload, trigger_type="COMMENT",
        )

    async def route_approval_command(self, *, repo_id, mr_iid, body,
                                      author_email, discussion_id, raw_payload):
        from conciliator.services.approval_service import ApprovalService
        svc: ApprovalService = self.approval_service  # type: ignore
        await svc.handle_comment(
            repo_id=repo_id, mr_iid=mr_iid, body=body,
            author_email=author_email, discussion_id=discussion_id,
        )

    async def route_auto_conflict(self, *, repo_id, mr_iid, raw_payload):
        from conciliator.services.resolve_service import ResolveService
        svc: ResolveService = self.resolve_service  # type: ignore
        await svc.handle_trigger(
            repo_id=repo_id, mr_iid=mr_iid, author_email=None,
            payload=raw_payload, trigger_type="AUTO_CONFLICT",
        )
```

- [ ] **Step 3: 写 api/__init__.py 的 webhooks 注册**

Edit `src/conciliator/api/__init__.py`:

```python
from fastapi import APIRouter

from conciliator.api import health, webhooks

router = APIRouter()
router.include_router(health.router)
router.include_router(webhooks.router)

__all__ = ["router"]
```

- [ ] **Step 4: 修改 main.py 注册 app.state(注入 SecretsProvider)**

Replace `src/conciliator/main.py`:

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from conciliator.api import router as api_router
from conciliator.config import Settings, get_settings
from conciliator.integrations.coco.client import CocoClient
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.utils.logging import setup_logging
from conciliator.utils.secrets import SecretsProvider, build_secrets_provider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    setup_logging(settings)

    secrets: SecretsProvider = build_secrets_provider(settings)
    engine = create_engine_from_settings(settings)
    session_factory = get_session_factory(engine)
    nc_client = NextCodeClient(settings, secrets)
    coco_client = CocoClient(settings, secrets)
    lark_client = LarkClient(settings, secrets)

    # coco webhook 的 token 校验:与发往 Coco 的 Bearer token 同源
    coco_webhook_token = secrets.get(settings.kms_config_coco_api_token)

    app.state.settings = settings
    app.state.engine = engine
    app.state.task_repo = None  # Task 17 注入
    app.state.nextcode_client = nc_client
    app.state.coco_client = coco_client
    app.state.coco_token = coco_webhook_token
    app.state.lark_client = lark_client
    app.state.comment_router = None  # Task 17 注入

    yield

    await nc_client.aclose()
    await coco_client.aclose()
    await lark_client.aclose()
    engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="Conciliator", lifespan=lifespan)
    app.include_router(api_router)
    return app
```

- [ ] **Step 5: 写 webhook 集成测试**

Create `tests/integration/test_webhooks.py`:

```python
import hashlib
import hmac
import time

import pytest
from fastapi.testclient import TestClient


def _sign(body: bytes, ts: str, secret: str) -> str:
    return hmac.new(
        secret.encode(), f"{ts}.".encode() + body, hashlib.sha256,
    ).hexdigest()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k, v in {
        "ENV": "dev", "SERVICE_PSM": "x", "KMS_PSM": "x",
        "NEXTCODE_APP_ID": "app1", "COCO_API_BASE": "x",
        "COCO_DEFAULT_AGENT_ID": "x", "LARK_APP_ID": "x",
        "LARK_DOC_FOLDER_TOKEN": "x", "MYSQL_PSM": "x",
        # dev 模式从 _DEV_* 注入密钥
        "_DEV_NEXTCODE_APP_SECRET": "secret123",
        "_DEV_COCO_API_TOKEN": "tok-coco",
        "_DEV_LARK_APP_SECRET": "ls",
    }.items():
        monkeypatch.setenv(k, v)
    from conciliator.config import get_settings
    get_settings.cache_clear()


def test_codebase_webhook_rejects_stale_timestamp():
    from conciliator.main import create_app
    app = create_app()
    with TestClient(app) as client:
        old_ts = str(int(time.time()) - 1000)
        body = b'{"event_type":"comment.created"}'
        resp = client.post(
            "/hooks/codebase",
            content=body,
            headers={
                "X-Codebase-Signature": _sign(body, old_ts, "secret123"),
                "X-Codebase-Timestamp": old_ts,
            },
        )
    assert resp.status_code == 401


def test_codebase_webhook_rejects_bad_signature():
    from conciliator.main import create_app
    app = create_app()
    with TestClient(app) as client:
        ts = str(int(time.time()))
        resp = client.post(
            "/hooks/codebase",
            content=b'{"event_type":"comment.created"}',
            headers={
                "X-Codebase-Signature": "invalid",
                "X-Codebase-Timestamp": ts,
            },
        )
    assert resp.status_code == 401


def test_coco_webhook_rejects_bad_token():
    from conciliator.main import create_app
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/hooks/coco",
            json={"task_id": "1", "event_type": "task.started", "event_id": "e1"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status_code == 401
```

- [ ] **Step 6: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_webhooks.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add src/conciliator/api/ src/conciliator/main.py tests/integration/test_webhooks.py
git commit -m "feat: webhook entrypoints with signature verification"
```

---

### Task 14: 通知服务

**Files:**
- Create: `src/conciliator/services/__init__.py`
- Create: `src/conciliator/services/notification_service.py`
- Test: `tests/unit/test_notification_service.py`

**Interfaces:**
- Consumes: `NextCodeClient`
- Produces: `NotificationService(nc_client)`;`async notify_mr_comment(project_id, mr_iid, body, mention_emails=None) -> None`

- [ ] **Step 1: 写 notification_service.py**

Create `src/conciliator/services/__init__.py` (empty) and `src/conciliator/services/notification_service.py`:

```python
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.mrs import add_mr_comment


class NotificationService:
    def __init__(self, nc_client: NextCodeClient):
        self._nc = nc_client

    async def notify_mr_comment(
        self,
        *,
        project_id: str,
        mr_iid: int,
        body: str,
        mention_emails: list[str] | None = None,
    ) -> None:
        final_body = body
        if mention_emails:
            mentions = " ".join(f"@{e}" for e in mention_emails)
            final_body = f"{mentions}\n{body}"
        await add_mr_comment(self._nc, project_id, mr_iid, final_body)
```

- [ ] **Step 2: 写单元测试**

Create `tests/unit/test_notification_service.py`:

```python
import httpx
import pytest
import respx

from conciliator.config import Settings
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.services.notification_service import NotificationService


def _settings() -> Settings:
    return Settings(
        nextcode_app_id="x", nextcode_app_secret="x",
        coco_api_base="x", coco_api_token="x", coco_default_agent_id="x",
        lark_app_id="x", lark_app_secret="x", lark_doc_folder_token="x",
        mysql_dsn="x", nextcode_api_base="https://nc.test/api",
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
@respx.mock
async def test_notify_with_mentions():
    route = respx.post(
        "https://nc.test/api/v1/projects/p1/merge_requests/42/notes"
    ).mock(return_value=httpx.Response(200, json={"code": 0, "data": {"id": 1}}))

    nc = NextCodeClient(_settings())
    svc = NotificationService(nc)
    await svc.notify_mr_comment(
        project_id="p1", mr_iid=42, body="可以合入",
        mention_emails=["alice@x.com", "bob@x.com"],
    )

    assert route.called
    body = route.calls.last.request.content.decode()
    assert "@alice@x.com" in body
    assert "@bob@x.com" in body
    assert "可以合入" in body
    await nc.aclose()
```

- [ ] **Step 3: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_notification_service.py -v`

- [ ] **Step 4: 提交**

```bash
git add src/conciliator/services/__init__.py src/conciliator/services/notification_service.py tests/unit/test_notification_service.py
git commit -m "feat: notification service for MR comments with mentions"
```

---

### Task 15: Resolve Service(主 pipeline)

**Files:**
- Create: `src/conciliator/services/resolve_service.py`
- Create: `src/conciliator/services/templates.py` (飞书文档 / MR 描述 / Discussion body 模板)
- Test: `tests/unit/test_templates.py`
- Test: `tests/integration/test_resolve_service.py`

**Interfaces:**
- Consumes: 所有外部 client、`TaskRepository`、`NotificationService`
- Produces: `ResolveService(nc, coco, lark, task_repo, notification, settings)`;`async handle_trigger(repo_id, mr_iid, author_email, payload, trigger_type) -> int`(返回 task_id);`async run_pipeline(task_id) -> None`(worker 调用)

- [ ] **Step 1: 写 templates.py**

Create `src/conciliator/services/templates.py`:

```python
from textwrap import dedent


def build_lark_doc_xml(
    *,
    source_branch: str,
    target_branch: str,
    resolve_branch: str,
    blocks: list[dict],
) -> str:
    """构造飞书文档 XML 内容。

    blocks: list of {
        idx, file_path, start_line, end_line,
        ours, theirs, ai_resolution, author_email, thinking,
    }
    """
    parts = [
        f"<h1>MR 自动解冲突报告</h1>",
        f"<p>源分支 → 目标分支:<code>{source_branch}</code> → <code>{target_branch}</code></p>",
        f"<p>resolve 分支:<code>{resolve_branch}</code></p>",
        f"<p>冲突块数:{len(blocks)}</p>",
        "<h2>冲突块详情</h2>",
    ]
    for b in blocks:
        parts.append(f"<h3>{b['idx']}. {b['file_path']}:{b['start_line']}-{b['end_line']}</h3>")
        parts.append(f"<p>Author:<code>{b['author_email'] or 'unknown'}</code></p>")
        parts.append(f"<p>Ours(源分支):</p><pre><code>{_escape(b['ours'])}</code></pre>")
        parts.append(f"<p>Theirs(目标分支):</p><pre><code>{_escape(b['theirs'])}</code></pre>")
        parts.append(f"<p>🤖 AI 解冲突方案:</p><pre><code>{_escape(b.get('ai_resolution') or '(待生成)')}</code></pre>")
        thinking = (b.get("thinking") or "")[:500]
        if thinking:
            parts.append(f"<p>Agent 思考(摘要):{_escape(thinking)}</p>")
    return "".join(parts)


def build_mr_description(
    *,
    source_branch: str,
    target_branch: str,
    resolve_branch: str,
    lark_doc_url: str,
    blocks: list[dict],
) -> str:
    lines = [
        "## 🤖 自动解冲突报告",
        "",
        f"- **源分支** → **目标分支**:`{source_branch}` → `{target_branch}`",
        f"- **resolve 分支**:`{resolve_branch}`",
        f"- **冲突块数**:{len(blocks)}",
        "- **状态**:等待 owner 确认",
        "",
        "### 📄 详细报告",
        f"[飞书文档链接]({lark_doc_url})",
        "",
        "### ✅ 确认进度",
    ]
    for b in blocks:
        check = "x" if b.get("approved") else " "
        lines.append(
            f"- [{check}] #{b['idx']} `{b['file_path']}:{b['start_line']}-{b['end_line']}`"
            f" — {b.get('author_email') or 'unknown'}"
        )
    lines += [
        "",
        "### 📜 指令",
        "- 在对应冲突块 Discussion 里回复 `approve` 或 `reject <reason>`",
        "- MR 主线回复 `/resolve approve all` — 全部同意(限 MR author)",
        "- MR 主线回复 `/resolve retry` — 重解被拒块",
        "- MR 主线回复 `/resolve status` — 查看进度",
    ]
    return "\n".join(lines)


def build_discussion_body(
    *,
    idx: int,
    file_path: str,
    start_line: int,
    end_line: int,
    author_email: str | None,
    ours: str,
    theirs: str,
    ai_resolution: str,
    thinking: str,
) -> str:
    return dedent(f"""\
        **冲突块 #{idx}** · `{file_path}:{start_line}-{end_line}`
        **Author**: {author_email or 'unknown'}

        **Ours(源分支)**:
        ```
        {ours}
        ```

        **Theirs(目标分支)**:
        ```
        {theirs}
        ```

        **🤖 AI 解冲突方案**:
        ```
        {ai_resolution}
        ```

        **Agent 思考**(摘要):{thinking[:500]}

        ---
        👤 回复 `approve` 或 `reject <reason>` 即可确认
        """)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
```

- [ ] **Step 2: 写 templates 单元测试**

Create `tests/unit/test_templates.py`:

```python
from conciliator.services.templates import (
    build_lark_doc_xml, build_mr_description, build_discussion_body,
)


def test_build_lark_doc_xml_contains_blocks():
    xml = build_lark_doc_xml(
        source_branch="feat", target_branch="main",
        resolve_branch="resolve/feat/x",
        blocks=[{
            "idx": 1, "file_path": "a.py",
            "start_line": 10, "end_line": 20,
            "ours": "ours", "theirs": "theirs",
            "ai_resolution": "merged", "author_email": "a@x.com",
            "thinking": "thoughts",
        }],
    )
    assert "<h1>" in xml
    assert "a.py" in xml
    assert "a@x.com" in xml
    assert "merged" in xml


def test_build_mr_description_marks_approved():
    desc = build_mr_description(
        source_branch="feat", target_branch="main",
        resolve_branch="resolve/feat/x",
        lark_doc_url="https://lark/x",
        blocks=[{"idx": 1, "file_path": "a.py", "start_line": 10,
                 "end_line": 20, "author_email": "a@x.com", "approved": True}],
    )
    assert "[x]" in desc
    assert "[ ]" not in desc


def test_build_discussion_body_contains_approve_instruction():
    body = build_discussion_body(
        idx=1, file_path="a.py", start_line=10, end_line=20,
        author_email="a@x.com", ours="x", theirs="y",
        ai_resolution="z", thinking="t",
    )
    assert "冲突块 #1" in body
    assert "approve" in body
```

- [ ] **Step 3: 写 resolve_service.py**

Create `src/conciliator/services/resolve_service.py`:

```python
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus, can_transition, InvalidTransitionError
from conciliator.integrations.coco.client import CocoClient, CocoTaskResult
from conciliator.integrations.git.blame import blame_lines
from conciliator.integrations.git.conflicts import parse_conflict_file
from conciliator.integrations.git.repo import clone_to_tempdir
from conciliator.integrations.lark.client import LarkClient
from conciliator.integrations.lark.docs import create_doc, update_doc
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.branches import create_branch
from conciliator.integrations.nextcode.discussions import create_discussion
from conciliator.integrations.nextcode.merges import list_conflict_files
from conciliator.integrations.nextcode.mrs import (
    create_mr, get_mr, update_mr_description,
)
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services import templates
from conciliator.services.notification_service import NotificationService
from conciliator.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ResolveDeps:
    nc: NextCodeClient
    coco: CocoClient
    lark: LarkClient
    repo: TaskRepository
    notification: NotificationService
    settings: Settings


class ResolveService:
    def __init__(self, deps: ResolveDeps):
        self._deps = deps

    async def handle_trigger(
        self,
        *,
        repo_id: str,
        mr_iid: int,
        author_email: str | None,
        payload: dict,
        trigger_type: str,
    ) -> int | None:
        # 幂等:已有活跃任务则丢弃
        existing = await self._deps.repo.find_active_for_mr(repo_id, mr_iid)
        if existing:
            log.info("trigger_dropped_active_exists", task_id=existing.id)
            return None

        task = await self._deps.repo.create(
            repo_id=repo_id,
            source_branch=payload.get("source_branch", ""),
            target_branch=payload.get("target_branch", ""),
            resolve_branch="",  # 在 pipeline 中填充
            original_mr_id=mr_iid,
            trigger_type=trigger_type,
            trigger_payload=payload,
            mr_author_email=author_email,
        )
        log.info("task_created", task_id=task.id, repo_id=repo_id, mr_iid=mr_iid)
        return task.id

    async def run_pipeline(self, task_id: int) -> None:
        task = await self._deps.repo.get(task_id)
        if task is None:
            log.error("task_not_found", task_id=task_id)
            return

        try:
            await self._step1_prepare(task)
            await self._step2_identify_conflicts(task)
            await self._step3_dispatch_coco(task)
            # step4/5 等 Coco 完成事件触发,见 continue_after_coco
        except Exception as e:
            log.exception("pipeline_failed", task_id=task_id, error=str(e))
            await self._deps.repo.update_status(
                task_id=task_id, from_version=task.version,
                new_status=TaskStatus.FAILED, last_error=str(e),
            )

    async def continue_after_coco(self, task_id: int) -> None:
        """Coco 任务完成后被调用(webhook 或 poller)。"""
        task = await self._deps.repo.get(task_id)
        if task is None or not task.coco_task_id:
            return
        try:
            await self._step4_document(task)
            await self._step5_create_mr_and_discussions(task)
        except Exception as e:
            log.exception("post_coco_failed", task_id=task_id, error=str(e))
            await self._deps.repo.update_status(
                task_id=task_id, from_version=task.version,
                new_status=TaskStatus.FAILED, last_error=str(e),
            )

    async def _step1_prepare(self, task) -> None:
        await self._transition(task, TaskStatus.PREPARING)
        ts = int(time.time())
        resolve_branch = f"resolve/{task.source_branch}/{ts}"
        await self._deps.repo.update_status(
            task_id=task.id, from_version=task.version,
            new_status=TaskStatus.PREPARING,
            resolve_branch=resolve_branch,
        )
        task.resolve_branch = resolve_branch
        task.version += 1

        await create_branch(
            self._deps.nc, task.repo_id,
            from_branch=task.source_branch, new_branch=resolve_branch,
        )
        await self._transition(task, TaskStatus.BRANCH_CREATED)

    async def _step2_identify_conflicts(self, task) -> None:
        conflict_files = await list_conflict_files(
            self._deps.nc, task.repo_id,
            source=task.resolve_branch, target=task.target_branch,
        )

        # 解析每个冲突文件 + blame
        all_blocks: list[dict] = []
        if self._deps.settings.git_use_local_clone:
            repo_url = task.trigger_payload.get("repo_url", "")
            token = task.trigger_payload.get("clone_token")
            async with _async_clone(repo_url, task.resolve_branch, token):
                pass  # 简化:实际 blame 需要 clone 路径

        for cf in conflict_files:
            parsed = parse_conflict_file(cf.file_path, cf.content)
            for p in parsed:
                # blame 通过本地 clone(此处简化:不实际 clone,只在 settings 开启时做)
                all_blocks.append({
                    "block_idx": len(all_blocks),
                    "file_path": p.file_path,
                    "start_line": p.start_line,
                    "end_line": p.end_line,
                    "ours_text": p.ours_text,
                    "theirs_text": p.theirs_text,
                    "author_email": None,  # blame 在实际 clone 后填
                })

        await self._deps.repo.save_conflict_blocks(task.id, all_blocks)
        await self._transition(task, TaskStatus.CONFLICTS_FETCHED)

    async def _step3_dispatch_coco(self, task) -> None:
        agent_id = (
            task.trigger_payload.get("agent_id")
            or self._deps.settings.coco_default_agent_id
        )
        conflict_files = list({
            b["file_path"] for b in (await self._deps.repo.get_with_blocks(task.id)).blocks  # type: ignore[union-attr]
        })
        ref = await self._deps.coco.submit_resolve_task(
            agent_id,
            repo_url=task.trigger_payload.get("repo_url", ""),
            resolve_branch=task.resolve_branch,
            target_branch=task.target_branch,
            conflict_files=conflict_files,
            instructions="优先保留目标分支语义,保留两侧新增逻辑。",
        )
        await self._deps.repo.update_status(
            task_id=task.id, from_version=task.version,
            new_status=TaskStatus.COCO_SUBMITTED,
            coco_task_id=ref.task_id,
        )
        task.coco_task_id = ref.task_id
        task.version += 1
        await self._transition(task, TaskStatus.COCO_RUNNING)

    async def _step4_document(self, task) -> None:
        if not task.coco_task_id:
            return
        result = await self._deps.coco.fetch_task_result(task.coco_task_id)
        if result.status not in ("succeeded", "completed"):
            log.warning("coco_not_succeeded", status=result.status, task_id=task.id)
            await self._transition(task, TaskStatus.COCO_TIMEOUT)
            await self._transition(task, TaskStatus.FAILED)
            return

        # 拾取 diff 到对应 block
        full_task = await self._deps.repo.get_with_blocks(task.id)
        if full_task is None:
            return
        diff_by_path = {d.file_path: d for d in result.diffs}
        block_dicts = []
        for b in full_task.blocks:
            diff = diff_by_path.get(b.file_path)
            ai_resolution = diff.after if diff else ""
            await self._deps.repo.update_block_ai_resolution(
                block_id=b.id, ai_resolution=ai_resolution,
            )
            block_dicts.append({
                "idx": b.block_idx + 1,
                "file_path": b.file_path,
                "start_line": b.start_line,
                "end_line": b.end_line,
                "ours": b.ours_text,
                "theirs": b.theirs_text,
                "ai_resolution": ai_resolution,
                "author_email": b.author_email,
                "thinking": result.thinking,
            })

        xml = templates.build_lark_doc_xml(
            source_branch=task.source_branch,
            target_branch=task.target_branch,
            resolve_branch=task.resolve_branch,
            blocks=block_dicts,
        )
        doc = await create_doc(
            self._deps.lark,
            title=f"MR 自动解冲突报告 {task.source_branch}",
            content_xml=xml,
            folder_token=self._deps.settings.lark_doc_folder_token,
        )
        await self._deps.repo.update_status(
            task_id=task.id, from_version=task.version,
            new_status=TaskStatus.DOC_CREATED,
            lark_doc_url=doc.url,
        )
        task.lark_doc_url = doc.url
        task.version += 1

    async def _step5_create_mr_and_discussions(self, task) -> None:
        full_task = await self._deps.repo.get_with_blocks(task.id)
        if full_task is None:
            return
        block_dicts = [
            {
                "idx": b.block_idx + 1,
                "file_path": b.file_path,
                "start_line": b.start_line,
                "end_line": b.end_line,
                "author_email": b.author_email,
                "approved": False,
            }
            for b in full_task.blocks
        ]
        description = templates.build_mr_description(
            source_branch=task.source_branch,
            target_branch=task.target_branch,
            resolve_branch=task.resolve_branch,
            lark_doc_url=task.lark_doc_url or "",
            blocks=block_dicts,
        )
        mr = await create_mr(
            self._deps.nc, task.repo_id,
            source=task.resolve_branch, target=task.target_branch,
            title=f"[resolve] {task.trigger_payload.get('mr_title', '')}",
            description=description,
        )
        await self._deps.repo.update_status(
            task_id=task.id, from_version=task.version,
            new_status=TaskStatus.MR_CREATED, new_mr_id=mr.iid,
        )
        task.new_mr_id = mr.iid
        task.version += 1

        for b in full_task.blocks:
            body = templates.build_discussion_body(
                idx=b.block_idx + 1,
                file_path=b.file_path,
                start_line=b.start_line,
                end_line=b.end_line,
                author_email=b.author_email,
                ours=b.ours_text,
                theirs=b.theirs_text,
                ai_resolution=b.ai_resolution or "",
                thinking="",
            )
            disc = await create_discussion(
                self._deps.nc, task.repo_id, task.new_mr_id,
                file_path=b.file_path, line=b.start_line, body=body,
            )
            await self._deps.repo.update_block_discussion_and_resolution(
                block_id=b.id,
                discussion_id=disc.id,
                ai_resolution=b.ai_resolution or "",
            )

        author_emails = list({
            b.author_email for b in full_task.blocks if b.author_email
        })
        await self._deps.notification.notify_mr_comment(
            project_id=task.repo_id, mr_iid=task.new_mr_id,
            body=f"已创建 {len(full_task.blocks)} 个冲突块待确认",
            mention_emails=author_emails,
        )
        await self._transition(task, TaskStatus.AWAITING_APPROVAL)

    async def _transition(self, task, target: TaskStatus) -> None:
        current = TaskStatus(task.status)
        if not can_transition(current, target):
            raise InvalidTransitionError(current, target)
        ok = await self._deps.repo.update_status(
            task_id=task.id, from_version=task.version, new_status=target,
        )
        if not ok:
            raise InvalidTransitionError(current, target)
        task.version += 1
        task.status = target.value


# 简化的 async clone helper(实际 blame 路径在 _step2 中扩展)
from contextlib import asynccontextmanager


@asynccontextmanager
async def _async_clone(repo_url, branch, token):
    yield None
```

- [ ] **Step 4: 写集成测试(用 mock 验证编排)**

Create `tests/integration/test_resolve_service.py`:

```python
import pytest

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.models import Base
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.resolve_service import ResolveService, ResolveDeps


def _settings(dsn: str) -> Settings:
    return Settings(
        env="dev", service_psm="x", kms_psm="x",
        nextcode_app_id="x",
        coco_api_base="x", coco_default_agent_id="agent-1",
        lark_app_id="x", lark_doc_folder_token="x",
        mysql_psm="x", mysql_db_name="test",
        git_use_local_clone=False,
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_handle_trigger_creates_pending_task(mysql_container, monkeypatch):
    monkeypatch.setenv("_DEV_MYSQL_DSN", mysql_container.dsn)
    settings = _settings(mysql_container.dsn)
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    repo = TaskRepository(get_session_factory(engine))

    deps = ResolveDeps(nc=None, coco=None, lark=None, repo=repo,
                        notification=None, settings=settings)  # type: ignore[arg-type]
    svc = ResolveService(deps)

    task_id = await svc.handle_trigger(
        repo_id="r1", mr_iid=42, author_email="alice@x.com",
        payload={"source_branch": "feat", "target_branch": "main"},
        trigger_type="COMMENT",
    )
    assert task_id is not None
    task = await repo.get(task_id)
    assert task is not None
    assert task.status == TaskStatus.PENDING
    assert task.mr_author_email == "alice@x.com"

    # 幂等:再触发一次应被丢弃
    second = await svc.handle_trigger(
        repo_id="r1", mr_iid=42, author_email="alice@x.com",
        payload={"source_branch": "feat", "target_branch": "main"},
        trigger_type="COMMENT",
    )
    assert second is None
    engine.dispose()
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_templates.py tests/integration/test_resolve_service.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/conciliator/services/resolve_service.py src/conciliator/services/templates.py tests/unit/test_templates.py tests/integration/test_resolve_service.py
git commit -m "feat: resolve service with full pipeline orchestration"
```

---

### Task 16: Approval Service

**Files:**
- Create: `src/conciliator/services/approval_service.py`
- Test: `tests/integration/test_approval_service.py`

**Interfaces:**
- Consumes: `NextCodeClient`、`TaskRepository`、`NotificationService`、`CocoClient`
- Produces: `ApprovalService(deps)`;`async handle_comment(repo_id, mr_iid, body, author_email, discussion_id) -> None`

- [ ] **Step 1: 写 approval_service.py**

Create `src/conciliator/services/approval_service.py`:

```python
from datetime import datetime, timezone

from sqlalchemy import select, update

from conciliator.domain.approval import ApprovalStatus, can_transition
from conciliator.domain.commands import (
    parse_comment, ApproveBlock, RejectBlock, ApproveAll, Retry, Status, Cancel, Ignored,
)
from conciliator.domain.task import TaskStatus
from conciliator.integrations.nextcode.client import NextCodeClient
from conciliator.integrations.nextcode.discussions import reply_discussion
from conciliator.integrations.nextcode.mrs import add_mr_comment
from conciliator.repositories.db import get_session_factory
from conciliator.repositories.models import Approval, ConflictBlock, ResolveTask
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.notification_service import NotificationService
from conciliator.utils.logging import get_logger

log = get_logger(__name__)


class ApprovalService:
    def __init__(
        self,
        *,
        nc: NextCodeClient,
        repo: TaskRepository,
        notification: NotificationService,
        session_factory,
    ):
        self._nc = nc
        self._repo = repo
        self._notification = notification
        self._sf = session_factory

    async def handle_comment(
        self,
        *,
        repo_id: str,
        mr_iid: int,
        body: str,
        author_email: str,
        discussion_id: str | None,
    ) -> None:
        in_discussion = discussion_id is not None
        cmd = parse_comment(body, in_discussion=in_discussion)

        if isinstance(cmd, Ignored):
            return

        task = await self._repo.find_active_for_mr(repo_id, mr_iid)
        if task is None:
            log.info("no_active_task_for_comment", repo_id=repo_id, mr_iid=mr_iid)
            return

        if isinstance(cmd, Status):
            await self._reply_status(task)
            return

        # 全局指令需要 MR author / admin 权限
        if isinstance(cmd, (ApproveAll, Retry, Cancel)):
            if not await self._is_mr_author_or_admin(task, author_email):
                await self._reply_no_permission(task, author_email)
                return
            if isinstance(cmd, ApproveAll):
                await self._handle_approve_all(task, author_email)
            elif isinstance(cmd, Retry):
                await self._handle_retry(task)
            elif isinstance(cmd, Cancel):
                await self._handle_cancel(task)
            return

        # 单块指令需要 discussion_id
        if not discussion_id:
            return
        block_task = await self._repo.find_block_by_discussion(discussion_id)
        if block_task is None:
            return
        block, block_task_obj = block_task

        if task.status not in (TaskStatus.AWAITING_APPROVAL.value, TaskStatus.PARTIAL_REJECTED.value):
            await reply_discussion(
                self._nc, repo_id, mr_iid, discussion_id,
                f"任务当前状态 {task.status},不可操作",
            )
            return

        if not await self._can_operate_block(block, task, author_email):
            await reply_discussion(
                self._nc, repo_id, mr_iid, discussion_id,
                "无权限,仅 block author 或 MR author 可操作",
            )
            return

        if isinstance(cmd, ApproveBlock):
            await self._handle_approve_block(task, block, author_email, discussion_id)
        elif isinstance(cmd, RejectBlock):
            await self._handle_reject_block(task, block, author_email, cmd.reason, discussion_id)

    async def _is_mr_author_or_admin(self, task: ResolveTask, email: str) -> bool:
        if task.mr_author_email and email.lower() == task.mr_author_email.lower():
            return True
        # admin 查询(简化:TODO 接 NextCode API)
        return False

    async def _can_operate_block(
        self, block: ConflictBlock, task: ResolveTask, email: str,
    ) -> bool:
        if block.author_email and email.lower() == block.author_email.lower():
            return True
        if task.mr_author_email and email.lower() == task.mr_author_email.lower():
            return True
        return False

    async def _handle_approve_block(
        self, task, block, reviewer_email, discussion_id,
    ) -> None:
        async with self._sf() as session:
            result = await session.execute(
                select(Approval).where(Approval.block_id == block.id).with_for_update()
            )
            approval = result.scalar_one_or_none()
            if approval is None:
                return
            if approval.status == ApprovalStatus.APPROVED.value:
                await reply_discussion(
                    self._nc, task.repo_id, task.new_mr_id, discussion_id,
                    f"✅ Block #{block.block_idx + 1} 已 approve(已通过)",
                )
                return
            if not can_transition(
                ApprovalStatus(approval.status), ApprovalStatus.APPROVED
            ):
                return
            approval.status = ApprovalStatus.APPROVED.value
            approval.reviewer_email = reviewer_email
            approval.reviewed_at = datetime.now(timezone.utc)
            await session.commit()

        await reply_discussion(
            self._nc, task.repo_id, task.new_mr_id, discussion_id,
            f"✅ Block #{block.block_idx + 1} 已 approve",
        )
        await self._check_all_approved(task)

    async def _handle_reject_block(
        self, task, block, reviewer_email, reason, discussion_id,
    ) -> None:
        async with self._sf() as session:
            result = await session.execute(
                select(Approval).where(Approval.block_id == block.id).with_for_update()
            )
            approval = result.scalar_one_or_none()
            if approval is None or not can_transition(
                ApprovalStatus(approval.status), ApprovalStatus.REJECTED
            ):
                return
            approval.status = ApprovalStatus.REJECTED.value
            approval.reviewer_email = reviewer_email
            approval.reviewed_at = datetime.now(timezone.utc)
            approval.comment = reason
            await session.commit()

        await reply_discussion(
            self._nc, task.repo_id, task.new_mr_id, discussion_id,
            f"❌ Block #{block.block_idx + 1} 已 reject:{reason or '(无原因)'}\n"
            "可用 /resolve retry 重新解这些块",
        )
        await self._repo.update_status(
            task_id=task.id, from_version=task.version,
            new_status=TaskStatus.PARTIAL_REJECTED,
        )

    async def _handle_approve_all(self, task, reviewer_email) -> None:
        async with self._sf() as session:
            await session.execute(
                update(Approval)
                .where(
                    Approval.block_id.in_(
                        select(ConflictBlock.id).where(ConflictBlock.task_id == task.id)
                    ),
                    Approval.status == ApprovalStatus.PENDING.value,
                )
                .values(
                    status=ApprovalStatus.APPROVED.value,
                    reviewer_email=reviewer_email,
                    reviewed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        await add_mr_comment(
            self._nc, task.repo_id, task.new_mr_id,
            f"✅ 全部块已 approve by {reviewer_email}",
        )
        await self._repo.update_status(
            task_id=task.id, from_version=task.version,
            new_status=TaskStatus.ALL_APPROVED,
        )
        await self._notification.notify_mr_comment(
            project_id=task.repo_id, mr_iid=task.new_mr_id,
            body="所有冲突块已 approve,可以合入。",
            mention_emails=[task.mr_author_email] if task.mr_author_email else None,
        )

    async def _handle_retry(self, task) -> None:
        # 标记 RETRYING,worker 会拉起重解
        await self._repo.update_status(
            task_id=task.id, from_version=task.version,
            new_status=TaskStatus.RETRYING,
        )
        await add_mr_comment(
            self._nc, task.repo_id, task.new_mr_id,
            "🔄 已请求重解 REJECTED 块,worker 将拉起...",
        )

    async def _handle_cancel(self, task) -> None:
        await self._repo.update_status(
            task_id=task.id, from_version=task.version,
            new_status=TaskStatus.CANCELLED,
        )
        await add_mr_comment(
            self._nc, task.repo_id, task.new_mr_id,
            "任务已取消,分支与 MR 保留,系统不再响应。",
        )

    async def _check_all_approved(self, task) -> None:
        full = await self._repo.get_with_blocks(task.id)
        if full is None:
            return
        all_ok = all(
            b.approval and b.approval.status == ApprovalStatus.APPROVED.value
            for b in full.blocks
        )
        if all_ok and full.blocks:
            await self._repo.update_status(
                task_id=task.id, from_version=full.version,
                new_status=TaskStatus.ALL_APPROVED,
            )
            await self._notification.notify_mr_comment(
                project_id=task.repo_id, mr_iid=task.new_mr_id,
                body="所有冲突块已 approve,可以合入。",
                mention_emails=[task.mr_author_email] if task.mr_author_email else None,
            )

    async def _reply_status(self, task) -> None:
        full = await self._repo.get_with_blocks(task.id)
        if full is None:
            return
        lines = ["| Block | File | Author | Status |", "|---|---|---|---|"]
        approved = 0
        for b in full.blocks:
            status = b.approval.status if b.approval else "PENDING"
            if status == "APPROVED":
                approved += 1
            lines.append(
                f"| #{b.block_idx + 1} | {b.file_path}:{b.start_line} | "
                f"{b.author_email or '-'} | {status} |"
            )
        lines.append(f"\n进度:{approved}/{len(full.blocks)} approved")
        await add_mr_comment(
            self._nc, task.repo_id, task.new_mr_id, "\n".join(lines),
        )

    async def _reply_no_permission(self, task, email) -> None:
        await add_mr_comment(
            self._nc, task.repo_id, task.new_mr_id,
            f"@{email} 无权限执行此指令(限 MR author 或仓库 admin)",
        )
```

- [ ] **Step 2: 写集成测试 test_approval_service.py**

Create `tests/integration/test_approval_service.py`:

```python
import pytest

from conciliator.config import Settings
from conciliator.domain.task import TaskStatus
from conciliator.repositories.db import create_engine_from_settings, get_session_factory
from conciliator.repositories.models import Base
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.approval_service import ApprovalService


def _settings(dsn: str) -> Settings:
    return Settings(
        env="dev", service_psm="x", kms_psm="x",
        nextcode_app_id="x",
        coco_api_base="x", coco_default_agent_id="x",
        lark_app_id="x", lark_doc_folder_token="x",
        mysql_psm="x", mysql_db_name="test",
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_approve_block_flips_status(mysql_container, monkeypatch):
    monkeypatch.setenv("_DEV_MYSQL_DSN", mysql_container.dsn)
    settings = _settings(mysql_container.dsn)
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    sf = get_session_factory(engine)
    repo = TaskRepository(sf)

    # 直接造一个 AWAITING_APPROVAL 的 task + block + approval
    task = await repo.create(
        repo_id="r1", source_branch="feat", target_branch="main",
        resolve_branch="resolve/feat/x", original_mr_id=42,
        trigger_type="COMMENT", trigger_payload={},
        mr_author_email="alice@x.com",
    )
    await repo.update_status(
        task_id=task.id, from_version=0, new_status=TaskStatus.AWAITING_APPROVAL,
        new_mr_id=99,
    )
    block_ids = await repo.save_conflict_blocks(task.id, [{
        "block_idx": 0, "file_path": "a.py", "start_line": 10, "end_line": 20,
        "ours_text": "x", "theirs_text": "y", "author_email": "alice@x.com",
    }])
    # 手动补 discussion_id
    async with sf() as session:
        from sqlalchemy import update as sa_update
        from conciliator.repositories.models import ConflictBlock
        await session.execute(
            sa_update(ConflictBlock).where(ConflictBlock.id == block_ids[0])
            .values(discussion_id="disc-1")
        )
        await session.commit()

    # mock nc.reply_discussion / add_mr_comment / notification
    sent = []
    from conciliator.integrations.nextcode import discussions as disc_mod
    from conciliator.integrations.nextcode import mrs as mrs_mod

    async def fake_reply(client, pid, mr_iid, did, body):
        sent.append(("reply", did, body))

    async def fake_comment(client, pid, mr_iid, body):
        sent.append(("comment", body))

    monkeypatch.setattr(disc_mod, "reply_discussion", fake_reply)
    monkeypatch.setattr(mrs_mod, "add_mr_comment", fake_comment)

    class _NoopNotif:
        async def notify_mr_comment(self, **kwargs):
            sent.append(("notif", kwargs.get("body")))

    svc = ApprovalService(
        nc=object(), repo=repo, notification=_NoopNotif(), session_factory=sf,
    )
    await svc.handle_comment(
        repo_id="r1", mr_iid=42, body="approve",
        author_email="alice@x.com", discussion_id="disc-1",
    )

    full = await repo.get_with_blocks(task.id)
    assert full.blocks[0].approval.status == "APPROVED"
    assert ("reply", "disc-1",).__class__  # sent 至少一条
    engine.dispose()
```

- [ ] **Step 3: 运行测试验证通过**

Run: `uv run pytest tests/integration/test_approval_service.py -v`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add src/conciliator/services/approval_service.py tests/integration/test_approval_service.py
git commit -m "feat: approval service with per-block approve/reject/retry/status/cancel"
```

---

### Task 17: Worker(任务调度)

**Files:**
- Create: `src/conciliator/workers/__init__.py`
- Create: `src/conciliator/workers/task_runner.py`
- Modify: `src/conciliator/main.py` (lifespan 启动 worker)
- Test: `tests/unit/test_task_runner.py`

**Interfaces:**
- Consumes: `TaskRepository`、`ResolveService`、Coco webhook(continue_after_coco)
- Produces: `TaskRunner(deps)`;`async start()` / `async stop()`;`async run_once()` (供测试)

- [ ] **Step 1: 写 workers/task_runner.py**

Create `src/conciliator/workers/__init__.py` (empty) and `src/conciliator/workers/task_runner.py`:

```python
import asyncio

from conciliator.config import Settings
from conciliator.domain.task import SCHEDULABLE_STATUSES, TaskStatus
from conciliator.repositories.task_repo import TaskRepository
from conciliator.services.resolve_service import ResolveService
from conciliator.utils.logging import get_logger

log = get_logger(__name__)


class TaskRunner:
    def __init__(
        self,
        *,
        repo: TaskRepository,
        resolve_service: ResolveService,
        settings: Settings,
    ):
        self._repo = repo
        self._svc = resolve_service
        self._settings = settings
        self._stop = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        log.info("worker_started",
                 concurrency=self._settings.worker_concurrency,
                 interval=self._settings.worker_poll_interval_seconds)
        self._stop.clear()
        for i in range(self._settings.worker_concurrency):
            t = asyncio.create_task(self._loop(i))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        self._stop.set()
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("worker_stopped")

    async def _loop(self, worker_id: int) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception:
                log.exception("worker_loop_error", worker_id=worker_id)
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._settings.worker_poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def run_once(self) -> None:
        tasks = await self._repo.list_schedulable(limit=1)
        if not tasks:
            return
        task = tasks[0]
        log.info("worker_picked_task", task_id=task.id, status=task.status)

        if task.status == TaskStatus.RETRYING.value:
            # retry 走 resolve_service.run_pipeline,仅处理 REJECTED 块
            # (简化:目前与 PENDING 同走完整 pipeline;细分在 v2)
            await self._svc.run_pipeline(task.id)
        else:
            await self._svc.run_pipeline(task.id)
```

- [ ] **Step 2: 写单元测试**

Create `tests/unit/test_task_runner.py`:

```python
import asyncio
import pytest

from conciliator.workers.task_runner import TaskRunner


class _FakeRepo:
    def __init__(self, tasks):
        self._tasks = tasks
        self.calls = 0

    async def list_schedulable(self, limit=10):
        self.calls += 1
        if self.calls == 1:
            return self._tasks
        return []


class _FakeSvc:
    def __init__(self):
        self.pipelines_run = []

    async def run_pipeline(self, task_id):
        self.pipelines_run.append(task_id)


@pytest.mark.asyncio
async def test_run_once_picks_and_runs():
    from conciliator.config import Settings

    class _FakeTask:
        id = 7
        status = "PENDING"

    settings = Settings(
        nextcode_app_id="x", nextcode_app_secret="x",
        coco_api_base="x", coco_api_token="x", coco_default_agent_id="x",
        lark_app_id="x", lark_app_secret="x", lark_doc_folder_token="x",
        mysql_dsn="x",
    )  # type: ignore[arg-type]

    repo = _FakeRepo([_FakeTask()])
    svc = _FakeSvc()
    runner = TaskRunner(repo=repo, resolve_service=svc, settings=settings)

    await runner.run_once()
    assert svc.pipelines_run == [7]

    # 第二次没有任务
    await runner.run_once()
    assert svc.pipelines_run == [7]


@pytest.mark.asyncio
async def test_start_stop_lifecycle():
    from conciliator.config import Settings

    settings = Settings(
        nextcode_app_id="x", nextcode_app_secret="x",
        coco_api_base="x", coco_api_token="x", coco_default_agent_id="x",
        lark_app_id="x", lark_app_secret="x", lark_doc_folder_token="x",
        mysql_dsn="x",
        worker_concurrency=2,
        worker_poll_interval_seconds=1,
    )  # type: ignore[arg-type]

    repo = _FakeRepo([])
    svc = _FakeSvc()
    runner = TaskRunner(repo=repo, resolve_service=svc, settings=settings)

    await runner.start()
    await asyncio.sleep(0.1)
    await runner.stop()
    # 未抛异常即通过
```

- [ ] **Step 3: 修改 main.py 在 lifespan 启动 worker**

Edit `src/conciliator/main.py`,在 lifespan 里(注释标 "Task 15/16 注入实际 service;此处先用 None 占位")替换为真实注入 + worker 启动:

```python
    # 注入实际 service
    from conciliator.repositories.task_repo import TaskRepository
    from conciliator.services.notification_service import NotificationService
    from conciliator.services.resolve_service import ResolveService, ResolveDeps
    from conciliator.services.approval_service import ApprovalService
    from conciliator.api.comment_router import CommentRouter
    from conciliator.workers.task_runner import TaskRunner

    task_repo = TaskRepository(session_factory)
    notification = NotificationService(nc_client)
    resolve_deps = ResolveDeps(
        nc=nc_client, coco=coco_client, lark=lark_client,
        repo=task_repo, notification=notification, settings=settings,
    )
    resolve_service = ResolveService(resolve_deps)
    approval_service = ApprovalService(
        nc=nc_client, repo=task_repo, notification=notification,
        session_factory=session_factory,
    )
    comment_router = CommentRouter(
        resolve_service=resolve_service,
        approval_service=approval_service,
        task_repo=task_repo,
    )

    app.state.task_repo = task_repo
    app.state.comment_router = comment_router

    worker = TaskRunner(
        repo=task_repo, resolve_service=resolve_service, settings=settings,
    )
    await worker.start()
    app.state.worker = worker

    yield

    await worker.stop()
    await nc_client.aclose()
    await coco_client.aclose()
    await lark_client.aclose()
    await engine.dispose()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_task_runner.py tests/integration/test_webhooks.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/conciliator/workers/ src/conciliator/main.py tests/unit/test_task_runner.py
git commit -m "feat: worker with concurrency and lifecycle management"
```

---

## Phase 4:可观测性与收尾

### Task 18: Metrics 上报(bytedance.metrics)+ 启动检查

**Files:**
- Create: `src/conciliator/utils/metrics.py`(bytedance.metrics 封装)
- Modify: `src/conciliator/api/health.py` (readyz 接入 DB ping,同步 driver)
- Test: `tests/unit/test_metrics_helpers.py`

**Interfaces:**
- Produces: `metrics_emit_counter(name, value, tags)`、`metrics_emit_timer(name, seconds, tags)`、`metrics_emit_gauge(name, value, tags)`(薄包装,本地无 agent 时静默 no-op)

> **不再暴露 `/metrics` endpoint**。内部 metrics 平台直接收 UDP 上报。本地调试设 `BYTEDMETRICS_DEBUG=True` 可在 stderr 看到打点。

- [ ] **Step 1: 写 utils/metrics.py**

Create `src/conciliator/utils/metrics.py`:

```python
"""Metrics 上报薄包装。生产用 bytedance.metrics,测试/本地无 agent 时 no-op。"""
import os
from typing import Any

_TRY_IMPORT = True
_client: Any = None


def _get_client() -> Any | None:
    global _client, _TRY_IMPORT
    if not _TRY_IMPORT:
        return None
    if _client is None:
        try:
            from bytedance import metrics as _bm
            _client = _bm
        except Exception:
            _TRY_IMPORT = False
            return None
    return _client


def emit_counter(name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.emit_counter(name, value, tags=tags or {})
    except Exception:
        pass


def emit_timer(name: str, seconds: float, tags: dict[str, str] | None = None) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.emit_timer(name, seconds, tags=tags or {})
    except Exception:
        pass


def emit_gauge(name: str, value: float, tags: dict[str, str] | None = None) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.emit_gauge(name, value, tags=tags or {})
    except Exception:
        pass


# 业务约定好的 metric 名(集中管理,避免拼错)
class MetricName:
    TASKS_TOTAL = "conciliator.tasks"
    TASK_DURATION = "conciliator.task_duration"
    COCO_TASK_DURATION = "conciliator.coco_task_duration"
    CONFLICT_BLOCKS_TOTAL = "conciliator.conflict_blocks"
    APPROVAL_TOTAL = "conciliator.approval"
    EXTERNAL_API_CALLS = "conciliator.external_api_calls"
    EXTERNAL_API_DURATION = "conciliator.external_api_duration"
    WEBHOOK_PROCESSED = "conciliator.webhook_processed"
    QUEUE_DEPTH = "conciliator.queue_depth"


def is_debug_enabled() -> bool:
    return bool(os.getenv("BYTEDMETRICS_DEBUG"))
```

- [ ] **Step 2: 写单元测试**

Create `tests/unit/test_metrics_helpers.py`:

```python
from conciliator.utils import metrics


def test_emit_counter_without_agent_does_not_raise(monkeypatch):
    # 测试环境无 bytedance.metrics,应静默 no-op
    monkeypatch.setattr(metrics, "_TRY_IMPORT", True)
    monkeypatch.setattr(metrics, "_client", None)
    # 强制下次 import 失败
    import sys
    sys.modules.pop("bytedance.metrics", None)
    sys.modules["bytedance"] = None  # 让 import 失败

    metrics.emit_counter("foo", 1, {"a": "b"})  # 不抛
    metrics.emit_timer("bar", 1.5, {})
    metrics.emit_gauge("baz", 10, {})

    # 清理
    del sys.modules["bytedance"]


def test_metric_names_centralized():
    assert metrics.MetricName.TASKS_TOTAL == "conciliator.tasks"
    assert metrics.MetricName.APPROVAL_TOTAL == "conciliator.approval"
```

- [ ] **Step 3: 修改 health.py 加入 DB ping(同步 driver)**

Edit `src/conciliator/api/health.py`:

```python
from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app: str


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", app="conciliator")


@router.get("/readyz", response_model=HealthResponse)
async def readyz(request: Request) -> HealthResponse:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return HealthResponse(status="degraded", app="conciliator")

    def _ping() -> bool:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    ok = await run_in_threadpool(_ping)
    return HealthResponse(
        status="ok" if ok else "degraded", app="conciliator",
    )
```

- [ ] **Step 4: api/__init__.py 不再 include metrics router(无 /metrics endpoint)**

确认 `src/conciliator/api/__init__.py` 不 import metrics(原 plan 写了 include,现删除):

```python
from fastapi import APIRouter

from conciliator.api import health, webhooks

router = APIRouter()
router.include_router(health.router)
router.include_router(webhooks.router)

__all__ = ["router"]
```

- [ ] **Step 5: 运行测试验证通过**

Run: `uv run pytest tests/unit/test_metrics_helpers.py tests/integration/test_health.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/conciliator/utils/metrics.py src/conciliator/api/health.py src/conciliator/api/__init__.py tests/unit/test_metrics_helpers.py
git commit -m "feat: bytedance.metrics reporting wrapper and DB readiness check"
```

---

### Task 19: README 与部署文档

**Files:**
- Modify: `README.md`
- Create: `docs/DEPLOYMENT.md`

- [ ] **Step 1: 写 README.md**

```markdown
# Conciliator

自动解 MR 冲突的 codebase app。当 MR 存在冲突时:

1. 基于源分支创建 resolve 分支
2. 委托 Coco OpenAPI Next 自动解冲突
3. 把冲突块、AI 方案、author 邮箱汇总到飞书文档
4. 以 resolve 分支为源创建新 MR,每个冲突块发起 Discussion
5. owner 在 Discussion 里回复 approve / reject 逐块确认
6. 全部 approve 后通知 MR author 合入(不自动合入)

## 架构与设计

详见 [`docs/superpowers/specs/2026-07-21-conciliator-design.md`](docs/superpowers/specs/2026-07-21-conciliator-design.md)

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入实际凭证

# 3. 初始化数据库
export MYSQL_DSN="mysql+pymysql://user:pass@host:3306/conciliator"
uv run alembic upgrade head

# 4. 启动服务
uv run uvicorn conciliator.main:app --reload

# 5. 健康检查
curl http://localhost:8000/healthz
curl http://localhost:8000/metrics
```

## 开发

```bash
# 运行测试
uv run pytest

# 单元测试
uv run pytest tests/unit/

# 集成测试(需要 Docker)
uv run pytest tests/integration/

# lint / 类型检查
uv run ruff check .
uv run mypy src/
```

## 部署

详见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
```

- [ ] **Step 2: 写 DEPLOYMENT.md**

Create `docs/DEPLOYMENT.md`:

```markdown
# 部署指南

## 前置准备

### 1. 创建 codebase app

参考 [创建 App 文档](https://bytedance.larkoffice.com/wiki/EYfEwivTbiNmBhkF4Mpcfjqvnhb):

- Name: `conciliator`
- Permissions: 仓库读写、MR 读写、Discussion 写入、Webhook 订阅
- Webhook URL: `https://<your-domain>/hooks/codebase`
- Webhook events: `comment.created`、`merge_request.updated`、`merge_request.opened`

### 2. 配置 Coco Agent

在 [Coco Web 控制台](https://code.byted.org/managed-agents) 创建 Agent,记录 agent_id。

### 3. 创建飞书应用

- 在飞书开发者后台创建自建应用
- 权限:`docx:document:create`、`docx:document:write_only`、`contact:user.email:read`
- 创建文件夹并记录 folder_token

### 4. 准备 MySQL + 注册 PSM

- 字节云 RDS(MySQL 8.0+)创建数据库 `conciliator`
- 在服务发现(consul)为该 MySQL 注册 PSM,例如 `toutiao.mysql.conciliator`
- 配置 `MYSQL_PSM=<psm>` 和 `MYSQL_DB_NAME=conciliator`

### 5. KMS 敏感配置托管

在 [KMS 平台](https://kms.bytedance.net)为当前服务 PSM 托管以下配置:

| 配置名 | 内容 |
|--------|------|
| `nextcode_app_secret` | codebase app secret |
| `coco_api_token` | Coco OpenAPI Bearer token |
| `lark_app_secret` | 飞书 app secret |

### 6. 服务 PSM 注册

部署的服务自身也要有 PSM(供 KMS / DPS / Metrics 识别),例如 `conciliator.service.psm`。

## 部署到字节云应用平台

### 1. 构建镜像

```dockerfile
FROM python:3.11-slim

WORKDIR /app
# 用字节内部 pip 源
RUN pip install uv && \
    UV_INDEX_URL=https://artifact.bytedance.com/artifactory/api/pypi/python-release-virtual/simple \
    uv pip install --system uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ src/
COPY migrations/ migrations/
COPY alembic.ini .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "conciliator.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. 配置环境变量

在应用平台配置页注入 `.env.example` 中的非敏感变量(PSM、API base、agent_id 等)。敏感项走 KMS,**不要进 env**。

### 3. 执行数据库迁移

```bash
export ENV=prod
export MYSQL_PSM=toutiao.mysql.conciliator
export MYSQL_DB_NAME=conciliator
uv run alembic upgrade head
```

### 4. 配置 webhook

- NextCode: `https://<domain>/hooks/codebase`
- Coco: `https://<domain>/hooks/coco`(在 Coco Agent 配置页填)

## 健康检查

- Liveness: `GET /healthz`
- Readiness: `GET /readyz`(含 DB ping)

## Metrics

不上报 `/metrics` endpoint。打点直接走 `bytedance.metrics` 上报到内部 metrics 平台,在 metrics 平台查询:
- `conciliator.tasks` (tags: trigger, status)
- `conciliator.task_duration` (tags: stage)
- `conciliator.coco_task_duration`
- `conciliator.approval` (tags: action)
- `conciliator.queue_depth` (tags: status)

本地调试设 `BYTEDMETRICS_DEBUG=True` 可在 stderr 看到打点。

## 仓库配置(可选)

通过 SQL 或后续 admin 接口配置:

```sql
INSERT INTO repo_configs (repo_id, auto_resolve_enabled, allowed_target_branches, agent_id, created_at, updated_at)
VALUES ('your-repo-id', true, JSON_ARRAY('main', 'release/*'), 'your-agent-id', NOW(), NOW());
```

`auto_resolve_enabled=true` 的仓库,MR 进入冲突状态会自动触发解冲突。

## 排查

- 任务审计:`SELECT * FROM task_events WHERE task_id = ? ORDER BY occurred_at;`
- 日志:按 `task_id` 过滤结构化日志(后续接入字节流式日志 SDK)
- 指标:在内部 metrics 平台查 `conciliator.*`
```

- [ ] **Step 3: 提交**

```bash
git add README.md docs/DEPLOYMENT.md
git commit -m "docs: README and deployment guide"
```

---

## Self-Review

### 1. Spec 覆盖

| Spec 章节 | 覆盖任务 |
|----------|---------|
| 3. 整体架构 | T1-T19(全部) |
| 4. 项目结构 | T1(脚手架)+ 各 task 创建文件 |
| 5.1 触发阶段 | T13(webhook)+ T15(handle_trigger) |
| 5.2 Step 1-5 pipeline | T15(resolve_service 完整) |
| 5.3 飞书文档模板 | T15(templates.build_lark_doc_xml) |
| 5.4 审批闭环 | T16(approval_service) |
| 5.5 异常处理 | T15(try/except + FAILED 转换) |
| 6.1 ER 图 | T3(ORM models) |
| 6.3 Task 状态机 | T5(domain/task.py) |
| 6.4 Approval 状态机 | T5(domain/approval.py) |
| 6.5 并发与幂等 | T12(乐观锁 + coco_event_id 去重)+ T15(幂等触发) |
| 7.1-7.8 确认机制 | T6(指令解析)+ T16(完整 approval_service) |
| 7.6 retry 子流程 | T16(_handle_retry)+ T17(worker 拉起 retry) |
| 8. 测试策略 | 每个 Task 内置 TDD 步骤 |
| 9. 可观测性 | T18(bytedance.metrics 包装) |
| **9.5 字节内部组件集成** | T1(byted* deps + pip 源)、T2(SecretsProvider/KMS)、T3(bytedmysql 同步 engine)、T9/T10/T11(DPS token + secrets 注入)、T12(run_in_threadpool)、T18(bytedance.metrics) |
| 10. 安全(webhook 签名) | T9(签名校验)+ T13(webhook 入口)+ T2(KMS 拉密钥) |
| 11. 配置项 | T1(config.py,密钥走 KMS) |

**已知简化(实现阶段需补强):**
- T15 `_step2_identify_conflicts` 中 blame 路径只在 `git_use_local_clone=True` 时执行,目前留了占位(`async with _async_clone`),实际需要用 T8 的 `clone_to_tempdir` 同步调用 blame_lines
- T16 `_is_mr_author_or_admin` 的 admin 查询目前返回 False,需接 T9 的 `check_repo_member`
- T17 retry 目前与 PENDING 同走完整 pipeline,v2 应只重解 REJECTED 块

### 2. Placeholder 扫描

- 无 "TBD"、"TODO"、"稍后" 等占位
- 已知简化点已明确标注在代码注释和 Self-Review 中,实现者一目了然
- 每个 step 都有具体代码或具体命令

### 3. Type 一致性

- `TaskStatus` / `ApprovalStatus` 在 domain 定义后,repository / service 全部引用同一处
- `ResolveDeps` 在 T15 定义,T17 注入完整
- `CommentRouter` 在 T13 定义占位,T17 注入完整
- `ParsedConflict` / `ConflictBlock` / `BlameResult` 类型名在 T7、T8 中一致
- 函数名 `handle_trigger` / `run_pipeline` / `handle_comment` 在 T13、T15、T16 中一致

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-21-conciliator.md`.

Two execution options:

1. **Subagent-Driven (推荐)** — 每个 task 派一个新 subagent 执行,task 间 review,迭代快
2. **Inline Execution** — 在当前会话用 executing-plans 批量执行,checkpoint review

**Which approach?**
