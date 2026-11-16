# 生产部署

Conciliator 通过 SCM → Atum/ICM → TCE 镜像发布。
本地开发见[本地后端](knowledge/meegle-ai-node/local-backend.md)，凭据和上线门槛统一见[生产配置清单](production-configuration-checklist.md)。

## 发布前准备

| 项目 | 要求 |
| --- | --- |
| 服务身份 | `data.aeolus.conciliator`，与 `atum.yaml` 及平台授权一致 |
| 生产数据库 | 写 PSM `toutiao.mysql.aeolus_govern_write`，库 `aeolus_govern` |
| 凭据 | 配置清单中的六项 KMS Secret，授权服务身份读取；ByteCloud AK/SK 存同一个 JSON Secret |
| 业务权限 | Codebase/Coco、飞书/Meegle、MAAT、BITS/SCM/ICM/ENV/TCE；按实际调用身份核验 |
| 模板与资源 | MAAT/PPE 模板、流水线和资源规格随代码管理；Devbox worker 单独安装 |
| 进程与实例 | 默认 `WORKER_NUM=4`，生产至少 2 个进程；请求数达到上限后由 Uvicorn 父进程补充 worker |

Codebase webhook 使用 `/hooks/codebase`，Meegle AI 节点使用 `/hooks/meegle/ai-node`。
浏览器卡片与平台 webhook 都须能访问后端；插件绑定与构建见[正式节点](knowledge/meegle-ai-node/production-node.md)。
飞书默认长连接无需 HTTP 回调地址，OAuth 地址须在飞书应用中登记。

## 构建与启动工件

| 文件 | 用途 |
| --- | --- |
| [`atum.yaml`](../atum.yaml) | SCM 构建配置，安装依赖、打包到 `/opt/tiger/conciliator` 并生成镜像 |
| [`byted-requirements.txt`](../byted-requirements.txt) | 独立安装的内部 SDK；不进入 `uv.lock` |
| [`run/prod_run.sh`](../run/prod_run.sh) | 设置 `src` 的 PYTHONPATH，使用 Uvicorn 监听 TCE `PORT0` |
| [CI 配置](../.codebase/pipelines/ci.yaml) | 测试、lint、类型检查与 schema 静态检查；不连接生产 RDS |
| [`migrations/`](../migrations/) / [`ddl/`](../ddl/) | 迁移定义与供 SQL 工单执行的增量文件 |

SCM 使用 Atum 编译，配置文件为 `atum.yaml`，开启镜像构建；TCE 选择实际 ICM 镜像仓库和版本。
构建时将开源依赖与内部 SDK 安装到同一个系统 Python，避免混用 `.venv` 与系统包。
Atum 配置参考[官方文档](https://bytedance.larkoffice.com/wiki/wikcn90DilLa7eGT7VijZbpc3Sd)。

## 发版顺序

1. 代码通过 CI 和 MR 审核；涉及 schema 时审核 DDL 影响。
2. 查询目标库 `alembic_version`，沿 `down_revision` 链执行缺失的增量 SQL 工单并核验版本。
3. 核对 KMS 凭据、服务 ACL、插件绑定和生产配置。
4. SCM 构建当前 commit/tag 的镜像，在 TCE 发布该版本并观察启动。
5. 验证健康、数据库、平台鉴权和真实业务流程，再确认发布结果。

当前 schema head 为 `0019_ppe_deployments`。已有生产库不能执行 `ddl/all.sql`；
该空库快照仅到 `0007_task_trace`，迁移步骤见 [DDL 说明](../ddl/README.md)。
MySQL DDL 可能自动提交，执行器须遇错即停，失败后按实际结构对账。
启动脚本和 CI 均不执行生产迁移；仓库没有 `.codebase/pipelines/migrate.yaml`。

## 检查与验收

- `/healthz` 验证进程存活；`/readyz` 检查数据库，可用时返回 HTTP 200、`status=ok`，未初始化或不可用时返回 HTTP 503、`status=degraded`。TCE 就绪探针使用 `/readyz`。
- 日志、任务响应和 token 只在有权限的环境中查看，不公开原文。
- Metrics 直接上报内部平台，无 `/metrics` HTTP 入口。
- HTTP/RDS 和只读鉴权检查不能代替 MR 审批、Coding、部署、升级、业务路由及释放验收。

Coding 验收见[端到端验证](knowledge/meegle-ai-node/e2e-validation.md)，
Devbox 的个人 worker JSON 同时保存机器配置和 runtime profile，无需新建 manifest 文件；
安装与运行环境预检见 [worker README](../deployment-worker/README.md)。Devbox 见[部署与恢复](knowledge/meegle-ai-node/devbox-deployment.md)，
PPE 见[产品验收](knowledge/meegle-ai-node/ppe-product-e2e-validation.md)。

## Argos StreamLog(华北)

服务 PSM 为 `data.aeolus.conciliator`，目标区域为 `China-North`。应用日志仍然由
`structlog` 输出单行 JSON；生产镜像安装 `bytedlogger` 后，`setup_logging()` 会额外挂载
`StreamLogHandler`，把同一份 JSON 转发给 TCE 内的 ttlogagent。

Argos 只接收应用日志和 uvicorn 服务日志，第三方 logger 不会上报：

- 只转发应用日志（`conciliator.*`）和 uvicorn 服务日志（`uvicorn.error`，每条记录
  只转发一次）。`httpx`、SQLAlchemy 等第三方 logger 不转发，因为它们会输出完整 URL
  或 SQL 参数。
- `structlog` 应用日志在 stdout 和 Argos 中是同一份 JSON，两者都带 `trace_id`
  （在 trace scope 内）和 `task_id`（业务事件或 trace scope 内），可直接按
  `task_id` / `trace_id` / `level` 检索。少数直接用 stdlib logger 的模块
  （如 `conciliator.lark_oauth`）照常转发，但不带这两个字段。
- uvicorn 服务日志在 Argos 中经 `foreign_pre_chain` 渲染为 JSON；uvicorn 自身输出到
  stderr 的仍是纯文本，两者不逐字相同。

- SDK 版本固定在 `byted-requirements.txt`：`bytedlogger==0.15.2`（`0.16.0` 已被上游
  yank）；同时固定 `bytedbackgrounds==0.0.6`，它提供 `bytedlogger` 运行时 import 的
  `backgrounds` 模块。
- 本地和 CI 不安装内部 SDK，`setup_logging()` 会自动降级为仅 stdout，不影响测试和开发。
- 生产可通过 `ARGOS_LOG_ENABLED=false` 关闭上报；默认开启。
- `StreamLogHandler` 的 `_psm` / `_taskName` tag 固定取 `SERVICE_PSM`，避免 TCE 环境变量
  缺失或错误时把日志打到其他 PSM。
- `uvicorn.access` 不转发到 Argos：访问日志包含 query string，
  `/oauth/lark/callback` 的 `code` / `state` 会因此进入检索系统。应用业务日志仍由
  `structlog` 正常上报。
- 业务线程不直接调用 SDK：记录先在调用线程渲染成 JSON，再放入上限 1024 条的有界队列，
  由专用线程（`argos-log-drain`）交给 `StreamLogHandler`。`emit` 里的
  `bytedance.metrics` 计数和 socket 写入因此都不会落在 webhook、解冲突 worker 或响应
  路径上；SDK 自身的队列上限同为 1024，队列满时由 `_ArgosDropGuard` 丢弃，不会在
  drain 线程上同步 flush socket。
- 日志投递是 best effort，任何失败都不抛给调用方：JSON 渲染失败、入队失败，以及交接
  过程中的其他异常，都只把记录计入 `dropped_records`；只有 drain 线程启动失败
  （`Thread.start()` 抛 `RuntimeError`）不计数——记录仍留在队列里，等待下次 `emit`
  重试或退出时的 `flush()`。以上情况都不会改变 webhook 或解冲突 worker 的行为。
- 队列满时记录被丢弃并计入 `dropped_records`，诊断信息 best-effort 输出一次
  （`argos log queue full`）：由 drain 线程写出，若队列满时该线程尚未启动或已停止，则由
  退出路径的 `flush()` 补写。调用线程既不写 stderr、也不等 ttlogagent 或 metrics，容器
  stderr 回压或写入失败同样不会影响业务代码。进程退出（`atexit`、`logging.shutdown()`）
  会先把队列排空再交给 SDK 自己的 `atexit` flush；handler 被关闭后再次记录会重新启动
  drain 线程。
- `bytedlogger` 只发布 sdist（含 `proto` / `protov3` 两个 C 扩展），基础镜像不带
  编译器，`atum.yaml` 在安装依赖前先装 `build-essential`。

Argos 侧需要存在该 PSM 的华北数据源（当前检索链接中的
`data_source_uid=2360603472`）。发版并产生流量后，可用以下方式确认接入：

```bash
bytedcli --vregion China-North log search-psm-log \
  --psm data.aeolus.conciliator \
  --start <epoch-seconds> --end <epoch-seconds> \
  --max-logs 20 --output console --yes
```

也可以直接打开 Argos StreamLog 的 keyword search 页面，按
`psm=data.aeolus.conciliator`、`region=China-North` 过滤；启动阶段可先搜
`Application startup complete`，业务日志可搜 `task_id` 或接口路径。
