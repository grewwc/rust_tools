# 本地测试配置检查记录

检查日期：2026-11-09，约 10:35–10:40（UTC+8）。

本文是历史检查快照，其中 manifest 指当时的旧版运行环境配置。
当前接入已改为单个 worker JSON 内维护解释器和依赖路径；环境版本标识默认生成，旧文件摘要和路径配置仅为兼容保留；
新接入及迁移以 [worker README](../deployment-worker/README.md) 为准。

检查对象：`/data00/home/huaguojin/Conciliator` 的本地工作区配置、实际运行进程、开发数据库、卡片构建配置及 Devbox deployment worker。记录包含检查时未提交的测试 App 配置校验改动；本次 MR 文档更新未包含这些代码改动。授权入口暂停实施，参见[本地测试 TODO](local-test-environment-todo.md)。

后续更新：同日 10:45 已补齐专用 worker 环境并验证重启；以下初次检查中的安装环境偏差已解决，详情见末节。

## 测试能力与证据边界

当前环境具备后端、数据库、测试飞书长连接和 Devbox worker 的基础联调条件；已有 VQS 实例通过本机健康及进程归属检查。完整的卡片 → 文档评估 → Coding → 新部署 → MAAT 业务验证链路，尚不能据此认定通过。

| 测试内容 | 当前判断 | 依据与限制 |
| --- | --- | --- |
| 本地后端/API、数据库访问 | 已确认可用 | `/healthz`、`/readyz` 均返回 200/ok，数据库 `SELECT 1` 成功 |
| 测试飞书 App 的长连接 | 已确认连接 | 当前日志有连接成功标记；本轮会话此前已验证测试消息发送和邮箱解析 |
| Meegle 测试插件与本地后端身份一致 | 已确认本地一致 | 后端与本地前端工程均使用 `MII_6AAF4159C3C0CCDB`；远端点位和线上卡片版本未核验 |
| 应用身份读取实际需求文档 | 仍需验证 | 此前文档评估被文档权限阻断；应用级接口权限探测不等于实际文档可读 |
| 从卡片完成用户授权并恢复评估 | 当前阻断 | 卡片缺授权入口，OAuth 回调指向生产；开发库用户 Token 行数为 0 |
| Agent 读取引用文档 | 当前地址不匹配 | 文档代理 origin 从 OAuth 回调推导，目前也是生产域名 |
| Devbox worker 注册、心跳、领取轮询 | 已确认在线 | v2 worker `ready=true`，心跳新鲜；最近心跳和领取轮询均为 200 |
| Aeolus/VQS 运行环境预检 | 已通过 | 三套 profile 的版本、目录和 manifest 校验通过；不等于所有业务依赖已完成运行验证 |
| 已有 VQS 实例本机健康 | 已通过 | 动态端口 18943 返回 200，响应匹配，PID 启动时间与 journal 一致，监听归属检查通过 |
| 新 Coding 产物部署、MAAT 页面、释放恢复 | 尚未验证 | 本轮未触发 Coding、新部署、MAAT 写入或释放操作；历史成功记录不能替代当前链路验收 |
| PPE 构建与平台部署 | 不由本地 worker 结论覆盖 | 使用独立平台链路；ByteCloud 凭据存在不等于 SCM/ICM/ENV/TCE 权限与流程可用 |

运行结论是检查时的快照。`/readyz` 只执行数据库探活，不检查数据库版本、飞书连接、Codebase、MAAT 或业务页面，见[健康检查实现](../src/conciliator/api/health.py)。

## 后端配置与环境隔离

实际生效配置通过当前 Uvicorn 进程环境结合 `Settings` 核对；输出只包含非敏感字段和凭据存在性，不包含凭据真值。

| 配置项 | 当前值或状态 | 评价 |
| --- | --- | --- |
| `ENV` / 凭据来源 | `dev` / `StaticSecretsProvider` | 适合本地环境；与 BOE ZTI 身份环境独立 |
| 服务身份 | `data.aeolus.conciliator` | 启动配置使用该身份；复用进程环境的 SDK 预检能取得非空 JWT，未输出 Token |
| `NEXTCODE_ZTI_REQUIRED` | `true` | 保留真实平台鉴权条件；SDK 与 Token 可用不等于所有仓库操作获授权 |
| 飞书 App | `cli_aa3e63e495b9dbe9` | 独立测试 App，区别于代码生产默认值 |
| 飞书回调模式 | `websocket` | 适合本地卡片回调；生产 App 防误连校验已存在 |
| Meegle 插件 | `MII_6AAF4159C3C0CCDB` | 测试身份，后端与本地插件工程一致 |
| OAuth 回调 | `https://aeolus-conciliator.byted.org/oauth/lark/callback` | 与当前本地测试环境不匹配 |
| Markdown 上传目录 | 沿用代码默认目录 | 测试 App 是否有该目录写权限尚未验证 |
| 所需本地凭据 | NextCode API/回调、飞书、Meegle API/回调、ByteCloud AK/SK 均非空 | 存在性检查通过；不能代替平台权限、凭据配对和资源访问验证 |
| `LOG_LEVEL` | `INFO` | 符合当前本地调试需求 |
| `COCO_TASK_TIMEOUT_SECONDS` | `7200` | 允许长评估/Coding；快速故障测试等待较长，非启动阻断项 |
| `WORKER_CONCURRENCY` / 轮询间隔 | `4` / `5s` | 控制 MR 解冲突 TaskRunner；需要串行观察 MR 时可考虑改为 1，不控制 Devbox 实例数量 |
| 飞书通知 worker 并发 | `1` | 已为串行通知处理 |
| 数据库连接池 | `10`，额外连接上限 `20` | 可用于本地测试；连接按需创建，不能理解为启动即占用 30 个连接 |

配置结构将 dev 的环境凭据与 staging/prod 的 KMS 来源分开，基本合理，见[配置实现](../src/conciliator/config.py)和[凭据来源](../src/conciliator/utils/secrets.py)。生产默认值由源码维护，因此 `ENV=dev` 并不会自动替换所有业务 ID、上传目录或公开地址。

NextCode/Coco 地址仍为 `https://codebase-api.byted.org/v2/`，Coco workspace、Agent 和凭据库仍复用既有配置。环境隔离目前主要覆盖飞书 App、Meegle 插件和开发数据库；本地模式仍访问真实平台。实际 Coding、MR 或部署测试应使用明确的测试任务和目标资源。

## 数据库与启动

当前使用显式 `MYSQL_DSN`，目标为本机 `127.0.0.1:3306/aeolus_govern`，驱动为 `mysql+pymysql`。未输出连接串或数据库密码。Docker 中 `conciliator-mysql` 正在运行，镜像为 `hub.byted.org/mysql:8.0`。

实库 `alembic_version` 为 `0019_ppe_deployments`，与当前迁移 head 一致，任务、部署、飞书用户 Token 等表存在。因此数据库已满足当前代码的基本结构要求，而不只是连接可用。

数据库实现实际使用的变量名是 `MYSQL_DSN`，见[引擎创建](../src/conciliator/repositories/db.py)。Alembic 需单独导出配置执行；启动脚本不会迁移数据库。

IDE 已保存后端启动命令：

```bash
doas --no-interaction -V 0 -t zti -e boe -p data.aeolus.conciliator bash run/dev_run.sh
```

当前后端监听 `0.0.0.0:8000`。两个 Uvicorn 相关进程对应热重载监督与应用运行角色，不能解释为启动了两个业务 worker。启动脚本加载 `.env`、清空应用日志并开启源码热重载，适合交互开发；定位重启恢复或长任务行为时，应留意源码修改触发重载。参见[启动脚本](../run/dev_run.sh)和[TaskRunner](../src/conciliator/workers/task_runner.py)。

启动脚本通过命令及端口匹配停止旧 Uvicorn，没有进一步按 worktree 目录检查归属。多个 worktree 联调时应使用不同端口并先确认进程归属，避免同端口启动互相影响。

## 卡片、回调与文档代理

| 入口 | 当前配置 | 判断 |
| --- | --- | --- |
| Devbox worker → 后端 | `http://127.0.0.1:8000` | 同机访问合理，已验证轮询成功 |
| 卡片 dev profile | `http://10.37.248.120:8000` | IP 与本机一致，本机 TCP 可达；浏览器及 Meegle 服务端可达性仍需各自核验 |
| 当前根前端生成配置 | `https://aeolus-conciliator.byted.org` | 根工程下一次直接使用该配置构建会访问生产后端 |
| OAuth 回调 | 生产 `/oauth/lark/callback` | 本地授权不能在该配置下形成同站点闭环 |
| Agent 引用文档代理 | 由 OAuth 回调 origin 推导，当前为生产域名 | 本地任务和能力签名应由本地后端验证，当前请求目标不匹配 |
| 已发布测试卡片及远端 AI 节点 webhook | 本轮未读取远端 | 不能从本地配置或截图推定远端版本与地址一致 |

`npm run build`、`pack` 默认使用 prod；`build:dev`、`pack:dev` 才生成 dev 地址。`frontend/src/build-config.json` 会被构建脚本覆盖。测试插件 ID 与构建 profile 是独立维度，选择测试插件不会自动切换 API 地址。隔离构建目录可能有自己的生成配置，本轮根工程检查不代表所有隔离产物或线上产物。参见[构建 profiles](../frontend/build.config.json)、[构建脚本](../frontend/scripts/lpm.cjs)和[卡片入口](../frontend/src/features/ai_node_web_aeolus_card/App.tsx)。

OAuth start 设置的 state cookie 绑定发起授权的站点，callback 要同时验证 query state 与 cookie。当前本地 HTTP 起点配生产 HTTPS callback，既有跨站点问题，cookie 还会按 callback 的 HTTPS scheme 标记 Secure。单纯新增按钮不足以完成授权。应确定浏览器和 Agent 均可访问的测试后端 origin，再将 OAuth 起点、callback、飞书后台登记地址和文档代理统一。参见[OAuth 接口](../src/conciliator/api/oauth.py)及[文档代理](../src/conciliator/services/aeolus_document_access.py)。

后端已提供 `authorization_required`、`lark_authorized`，卡片尚未消费；应用身份读取优先，失败后才尝试操作者身份。当前库没有用户 Token，因此需要操作者身份的文档无法走该回退路径。用户授权后的自动恢复逻辑存在，但完整授权回调尚未验证。授权请求包含 `offline_access docx:document:readonly wiki:node:read`，仍需在测试 App 后台核对用户身份权限及实际文档访问。

对两个已配置飞书 origin 的 CORS 预检均成功：

- `https://meego.larkoffice.com`
- `https://meegoplg.larkofficepkg.com`

`http://localhost:3000` 预检返回 400。当前配置支持列出的飞书卡片 origin；另起独立前端开发页时需核对实际 origin。HTTPS 页面访问 HTTP Devbox 还可能受 mixed content 或浏览器内网访问限制，CORS 成功不能证明浏览器请求一定成功。

## Devbox worker 与业务运行配置

当前 worker 为 `aeolus-devbox-huaguojin-v2`，协议版本 2，稳定 root 为 `.deployment-worker/aeolus-devbox-huaguojin-v2`。配置权限为 `0600`，root 权限为 `0700`。`advertised_host=10.37.248.120` 与本机网卡地址一致；没有固定服务端口，符合动态端口部署模式。

数据库中该 worker `ready=true`，最近心跳距查询不足数秒；最近 10 次心跳及领取轮询均返回 200。旧 worker 记录的时间为 10 月 28 日，不能只按历史 `ready=true` 判断在线；实现使用 30 秒心跳窗口。

| 仓库 / profile | Python | runtime revision | 本轮校验 |
| --- | --- | --- | --- |
| `dp/aeolus` / `default` | 3.10，既有 Aeolus pyenv | `aeolus-local-py310-r1` | 版本、额外路径及 1 项文件校验通过 |
| `dp/VizQueryService` / `default` | 3.7，既有 VQS venv | `vqs-local-py37-r1` | 版本、依赖路径及 1 项文件校验通过 |
| `dp/VizQueryService` / `py311` | 3.11，独立业务 venv | `vqs-py311-r1` | 版本、依赖路径及 1380 项文件校验通过 |

VQS 本地 source 为 `/data00/work/vqs`，已确认是 Git 仓库。Aeolus 未配置本地 source，部署需从远端获取固定提交；VQS 本地获取失败也会回退远端。当前 manifest 校验通过，只代表声明的版本和文件一致；两个 default manifest 仅覆盖 1 项文件，不足以证明整个依赖环境完整。

三套 profile 的 `base_branches` 均为空。Aeolus 只有一套配置，可按单 profile 回退规则选择；VQS 有两套，用户需明确选择，不能期待按 base branch 自动区分 Python 3.7/3.11。该配置用于人工测试合理；若需分支自动选择，再补充明确映射。参见[profile 校验](../deployment-worker/src/conciliator_deployment_worker/profiles.py)、[runtime 校验](../deployment-worker/src/conciliator_deployment_worker/multi_worker.py)和[选择规则](../src/conciliator/services/deployment_multi.py)。

### worker 启动环境偏差

实际运行命令使用后端虚拟环境：

```bash
/data00/home/huaguojin/Conciliator/.venv/bin/python -m conciliator_deployment_worker \
  --config /data00/home/huaguojin/Conciliator/.deployment-worker/huaguojin.v2.json
```

该环境安装了 v2 模块。与当前源码比较，`worker.py`、`multi_worker.py` 的差异只有注释和错误文案，未发现这两处差异引入行为变化。

文档推荐的 `deployment-worker/.venv` 安装包仍缺少 `profiles` 等 v2 模块，直接导入已出现 `ModuleNotFoundError`。两套安装都标为 `0.1.0`，版本号不足以识别实际代码。按推荐专用环境重启存在失败风险；后端环境后续 `uv sync` 也可能清除额外安装的 worker 包。

应在后续恢复实施时统一 worker 的安装环境与启动入口，并将其加入可复用启动配置。目前 IDE Run 只保存了后端 action，不能承担冷启动整个联调环境。参见[worker 安装说明](../deployment-worker/README.md)。

### 现有实例与外部入口

当前 v2 root 中有 8 份实例 journal：7 份记录失败结果，1 份记录成功且仍在运行；端口账本保留 7 个预留。成功实例为 VQS，动态端口 18943。只读复核确认进程启动身份匹配、监听属于其进程组、健康响应匹配且连续健康失败数为 0。

这些记录不应直接删除以“重置测试环境”。失败结果、预留端口和实际释放状态需按 journal、进程与后端状态对账。当前检查没有执行清理，也没有把历史实例全部判定为已释放。

`advertised_host` 匹配本机，只能证明注册地址配置合理；本机访问后端成功，不能证明 MAAT 到动态业务端口可达。MAAT 登录后的业务页面、路由、当前目标提交和释放恢复，仍需单独验收。PPE 不使用 Devbox worker，也不能从 worker 在线推出 PPE 可部署。

## 后续配置收口顺序

| 顺序 | 工作 | 完成条件 |
| --- | --- | --- |
| 1 | 统一测试公开地址及授权入口 | 测试卡片、OAuth 起点/回调、Agent 文档代理、测试 App 登记地址一致；实际需求及引用文档可读 |
| 2 | 核验远端测试卡片和 AI 节点 | 发布版本、卡片 API origin、webhook、插件身份及限定测试空间一致 |
| 3 | 统一 worker 安装和启动 | 专用环境能加载当前 v2 包；后端与 worker 冷启动后健康、注册和心跳正常 |
| 4 | 完成新的业务部署验收 | 固定 Coding 提交，选择正确 profile，核验本机、MAAT 页面、人工业务验证及释放恢复 |
| 5 | 按测试场景调整调试参数 | 需要串行 MR 时降低 TaskRunner 并发；超时和热重载按测试目的选取 |

授权 UI、配置和远端调整继续保持暂停。详细待办见[本地测试 TODO](local-test-environment-todo.md)。初次检查仅新增/补充检查文档，没有修改运行配置、安装包或业务代码，也没有提交、发布或创建部署。后续 worker 处理如下。

## 专用 worker 环境修复与启动验证

复核时间：2026-11-09 10:45（UTC+8）。根据后续指令，已通过 `uv pip install --reinstall-package conciliator-deployment-worker` 将当前源码安装到 `deployment-worker/.venv`。七个 Python 模块均与源码逐文件校验一致，`profiles`、`multi_worker` 可导入，公共 CLI 的 `--help` 正常。

确认旧监督进程属于当前仓库和 v2 配置后，使用 SIGTERM 停止监督进程，再从专用环境启动 worker。没有停止业务实例、清理 journal 或释放端口。

| 复核项 | 结果 |
| --- | --- |
| 当前启动解释器 | `deployment-worker/.venv/bin/python` |
| 当前 worker 配置 | 原 `.deployment-worker/huaguojin.v2.json`，身份、root、注册地址保持不变 |
| 监督进程数量 | 当前仓库仅 1 个 |
| 注册 | 新注册请求返回 200，协议 v2 |
| 在线 | `ready=true`，查询时心跳年龄 2 秒 |
| 心跳与领取轮询 | 最近 5 次均返回 200 |
| 后端 | `/healthz`、`/readyz` 均为 200/ok |
| 已有 VQS | 原 PID 与 18943 端口保持不变；进程身份、监听归属、健康响应通过，健康失败数为 0 |

已将验证过的启动命令保存为本地 IDE Run 的 `deployment-worker` action，后端 action 和默认入口保持不变。`.trae/environments/run.json` 及 `run.md` 按仓库约定仅在本地保存；可复用安装更新方法见 [worker README](../deployment-worker/README.md)。本次完成专用环境安装和 worker 重启；整个环境冷启动、新业务部署、MAAT 页面与释放恢复仍需后续验收。
