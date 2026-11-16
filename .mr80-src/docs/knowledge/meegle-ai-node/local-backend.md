---
title: Meegle AI node local backend
description: Install dependencies, configure local credentials, migrate MySQL, and start the backend with ZTI.
keywords: [.env, MySQL, dev_run.sh, KMS, ZTI, doas, local-deployment]
last_verified: 2026-11-06
---

# 本地后端

在 Linux DevBox 上运行 Conciliator。业务部署另见 [Devbox 部署](devbox-deployment.md)，生产服务见[生产部署](../../DEPLOYMENT.md)。

## 安装依赖

需要 Python 3.11+、uv、Git、MySQL 8 和字节内网；完整联调还需要已登录的 doas。
以下命令均在仓库根目录执行：

```bash
uv sync --extra dev
uv pip install --python .venv/bin/python \
  --index-url https://bytedpypi.byted.org/simple/ \
  -r byted-requirements.txt
cp -n .env.example .env
```

内部 SDK 单独安装，不进入 `uv.lock`。重新执行 `uv sync` 后，如内部包被清除，需补装。
worker 与业务服务使用各自的解释器和依赖，不能用后端 `.venv` 替代业务运行环境。

## 开发配置

以根目录 [`.env.example`](../../../.env.example) 为准。设置 `ENV=dev`，填写：

| 配置 | 要求 |
| --- | --- |
| `MYSQL_DSN` | 指向开发库，默认示例为本机 `aeolus_govern` |
| `NEXTCODE_APP_SECRET`、`NEXTCODE_WEBHOOK_SECRET` | 与 `NEXTCODE_APP_ID` 配套，分别用于 API 和回调验签 |
| `LARK_APP_ID`、`LARK_APP_SECRET` | 同一个独立测试 App 的 ID/Secret；应用须有文档、通讯录及消息权限 |
| `LARK_OAUTH_REDIRECT_URI` | 用户授权启用时填写测试后端地址，并在测试 App 登记；留空关闭授权入口 |
| `MEEGLE_PLUGIN_ID` | 本地测试使用 `MII_6AAF4159C3C0CCDB`；代码默认 ID 为正式插件 |
| `MEEGLE_PLUGIN_SECRET`、`MEEGLE_WEBHOOK_TOKEN` | 同一测试插件的 API Secret 和回调 Token，不能互换 |
| `BYTECLOUD_AK`、`BYTECLOUD_SK` | 同一 ByteCloud 应用的配套 AK/SK，供 MAAT/PPE 共用 |

稳定 ID、地址和运行默认值在 [Settings](../../../src/conciliator/config.py) 中维护，按需覆盖。
环境凭据仅在 `ENV=dev` 使用；staging/prod 从 KMS 读取，
ByteCloud 使用 `conciliator/credentials/bytecloud_credentials` 中的 `{"ak":"<ak>","sk":"<sk>"}`。
完整生产路径及轮转要求见[配置清单](../../production-configuration-checklist.md)，不要把 Secret 真值填入 `KMS_SECRET_*_ALIAS`。

测试卡片与测试节点的 webhook 必须同时指向本地后端；构建与地址设置见[卡片发布](card-webhook-publish.md)。
测试和生产使用不同插件身份及飞书应用。飞书长连接按同一 App 的连接随机投递回调，
不会按后端域名、数据库或 HTTP 灰度比例区分环境。`ENV=dev` 且使用长连接时，
配置校验拒绝代码中的生产 App ID，必须填写测试 `LARK_APP_ID` 及配套 `LARK_APP_SECRET`。
测试 App 需要单独开启机器人、配置可见范围、申请权限并订阅 `card.action.trigger`。
用户 OAuth 回调也须指向测试后端；使用独立开发库，不能混用不同 App 签发的用户 Token。
若测试 App 无权访问默认 Markdown 上传目录，覆盖 `LARK_MARKDOWN_FOLDER_TOKEN`。

## 数据库与启动

已有 MySQL 时复用开发实例；没有实例时可创建与 `.env.example` 一致的库：

```bash
docker run -d --name conciliator-mysql \
  -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=aeolus_govern \
  -p 3306:3306 mysql:8.0
```

先确认数据库可连接，再迁移开发库。Alembic 不自动加载 `.env`，因此显式导出配置：

```bash
(
  set -a
  source .env
  set +a
  uv run alembic current
  uv run alembic upgrade head
)
```

当前目标版本为 `0019_ppe_deployments`。生产 RDS 走[增量 SQL 工单](../../../ddl/README.md)，不能用本地迁移步骤代替。

使用 Conciliator 的 BOE ZTI 身份启动：

```bash
doas --no-interaction -V 0 -t zti -e boe \
  -p data.aeolus.conciliator bash run/dev_run.sh
```

BOE 是身份环境，`.env` 中的 `ENV=dev` 是配置来源，两者独立。
IDE Run 可在本地配置为执行同一命令；IDE 环境配置不纳入版本控制。
doas 登录过期时先完成登录；普通终端不注入 ZTI，不能满足默认的 Codebase 鉴权要求。

启动脚本加载 `.env`，重启同端口 Conciliator，清空 `logs/app.log`，监听 `0.0.0.0:${PORT0:-8000}` 并热重载 `src`。
重启前确认该端口进程属于当前仓库。数据库迁移独立执行，缺少必需凭据会阻止启动。

## 检查与排障

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
# 只读核验 ByteCloud JWT 和 MAAT 基线，不创建部署：
uv run python scripts/verify_maat_access.py
```

两个健康响应都应为 HTTP 200、`status=ok`。数据库未初始化或不可用时，
`/readyz` 返回 HTTP 503、`status=degraded`；API 文档位于 `http://127.0.0.1:8000/docs`。

飞书默认使用长连接，复用 Lark App Secret；应用开启机器人并订阅 `card.action.trigger`。
在本机日志中确认 `aeolus_lark_ws_connected`，不要公开完整日志、任务查询响应或任何凭据。
若改用 HTTP 回调，还需配置 `LARK_CARD_VERIFICATION_TOKEN`、按需配置 `LARK_CARD_ENCRYPT_KEY`，
并登记 `/hooks/lark/aeolus-card`。用户 OAuth 回调地址须在飞书应用中登记，按环境覆盖 `LARK_OAUTH_REDIRECT_URI`。

SDK 缺失先检查内部依赖安装；鉴权失败检查 doas 身份、配套 ID/Secret、平台权限和可见范围。
健康检查与只读鉴权不代表业务闭环通过，流程回归见[端到端验证与排障](e2e-validation.md)。
