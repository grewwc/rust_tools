# Conciliator 生产配置与上线清单

核验日期：2026-11-06。当前使用 KMS，不接入 TCC。本文依据本地实现，不代表远端配置或真实业务验收已经完成。

## 配置维护方式

Devbox 使用者只需下载本仓库并维护个人 worker JSON：填写连接信息、业务解释器与依赖路径；
示例使用 Aeolus Python 3.10、VQS Python 3.7，无需额外 manifest、文件摘要或手填环境版本号。
具体接入、预检和旧配置兼容见 [worker README](../deployment-worker/README.md)。

稳定配置统一维护在 [`Settings`](../src/conciliator/config.py)，随代码版本发布：App/Plugin/Agent ID、凭据库 ID、接口地址、PSM、数据库名、KMS alias、MAAT 基线/操作人。PPE 模板、SCM/流水线 ID 和资源规格继续随代码管理。worker 地址在其 JSON 配置的 `advertised_host` 中维护，注册后随部署保存；后端不维护 worker/IP 白名单。

同名环境变量可按需覆盖 Settings 字段；无需在 TCE 或 `.env` 重复填写全部默认值。MAAT 基线、操作人和认证代理为代码常量，不接受环境覆盖。更改配置后重启服务，已有 PPE 请求继续使用冻结快照。

正式服务始终初始化 Resolve、Coding、Devbox、PPE 和飞书通知。已删除 `INTEGRATIONS_ENABLED`、`NEXTCODE_ENABLED`、`PIPELINE_ENABLED`、`AEOLUS_AI_NODE_ENABLED`、`AEOLUS_DEPLOYMENT_ENABLED`、`AEOLUS_PPE_DEPLOYMENT_ENABLED`、`AEOLUS_LARK_NOTIFICATIONS_ENABLED` 及 MAAT 根据凭据是否为空决定启用的逻辑。旧变量不会再影响行为，应从运行配置移除。

同时删除九项无消费者字段：`APP_NAME`、`NEXTCODE_WEBHOOK_PATH`、`LARK_DOC_FOLDER_TOKEN`、`TASK_TOTAL_TIMEOUT_SECONDS`、`MAX_RETRIES_PER_TASK`、`REPO_MAX_CONCURRENT_TASKS`、`GIT_CLONE_BASE_DIR`、`GIT_CLONE_TIMEOUT_SECONDS`、`GIT_USE_LOCAL_CLONE`。它们此前不改变实际行为；真实仓库策略仍由数据库维护。

必需配置或凭据缺失时启动失败。所有客户端和必需凭据初始化成功后才启动后台 worker；测试使用 Fake provider、平台客户端和 worker 注入。HTTP/RDS 探活使用独立测试应用，不增加正式服务的功能开关。

服务始终启动 Resolve 不改变已有 `repo_configs` 仓库参与策略、允许的目标分支或仓库并发限制。

## 生产环境需要提供

| 配置 | 用途 |
| --- | --- |
| `ENV=prod` | KMS 与生产 bytedmysql 服务发现；Atum 已配置 |
| `WORKER_NUM=4` | Uvicorn 默认 4 个进程，生产启动脚本要求至少 2；父进程负责回收和重启达到请求上限的 worker |
| `LARK_OAUTH_REDIRECT_URI`（按需覆盖） | 默认 `https://aeolus-conciliator.byted.org/oauth/lark/callback`；用于用户授权后按操作者权限读取需求文档，须在飞书应用中登记相同地址；其他环境可覆盖，显式设为空关闭新授权入口 |
| 环境差异与运行参数（按需） | 身份/资源迁移时的 ID、PSM、alias 覆盖，以及日志、并发、超时、连接池 |

Codebase、飞书、Meegle 使用 `ID + Secret`，稳定 ID 保留在 Settings，只有 Secret 进入 KMS。ByteCloud AK/SK 作为成对轮转的凭据，整对进入同一个 KMS Secret。`MYSQL_PSM` 为目标数据库，`SERVICE_PSM` 为调用服务身份；`WORKER_NUM` 为进程数，`WORKER_CONCURRENCY` 为 Resolve 任务并发，不能混用。

多进程会分别启动后台轮询和飞书长连接；任务竞争依赖数据库租约与请求去重。发布验收需覆盖回调跨进程投递、重复投递、worker 回收及在途任务恢复，并按进程数核算数据库连接池和任务并发。Coding 网络异常可能重复启动 Run 的已知 P1 风险见[上线风险记录](release-risks.md)，本次只记录，尚未修复。

Meegle 默认插件 ID 已切换为正式节点 `MII_6AED80FF53004C14`（Aeolus 研发助手）。
生产 KMS 中的 Plugin Secret 和 Webhook Token 必须与此正式插件配套；
本地测试在 `.env` 设置 `MEEGLE_PLUGIN_ID=MII_6AAF4159C3C0CCDB`，并填写同一测试插件的
`MEEGLE_PLUGIN_SECRET`、`MEEGLE_WEBHOOK_TOKEN`。环境凭据仅在 `ENV=dev` 使用，
不会覆盖 staging/prod 的 KMS 凭据。

## 六项已接线的 KMS Secret

| 用途 | 默认 KMS alias |
| --- | --- |
| Codebase App Secret | `conciliator/credentials/nextcode_app_secret` |
| Codebase Webhook Secret | `conciliator/credentials/nextcode_webhook_secret` |
| 飞书 App Secret | `conciliator/credentials/lark_app_secret` |
| Meegle Plugin Secret | `conciliator/credentials/meegle_plugin_secret` |
| Meegle AI 节点 Webhook Token | `conciliator/credentials/meegle_webhook_token` |
| ByteCloud AK/SK（JSON，整对轮转） | `conciliator/credentials/bytecloud_credentials` |

每项通过 `KMS_SECRET_<用途>_ALIAS` 字段按需覆盖定位信息。真实服务身份 `data.aeolus.conciliator` 须有读取权限；核验时只输出成功/失败，不输出敏感值。Meegle Plugin Secret 与 Webhook Token 不能互换。

Meegle 回调凭据统一使用 `MEEGLE_WEBHOOK_TOKEN`（dev）及
`KMS_SECRET_MEEGLE_WEBHOOK_TOKEN_ALIAS`（KMS 定位配置）。默认 KMS 路径为
`conciliator/credentials/meegle_webhook_token`。尚未创建 Secret 时直接使用新路径；
已使用旧路径 `conciliator/credentials/aeolus_webhook_token` 的环境，须先在新路径
保存相同 Token 并授予服务身份读取权限，再部署新代码。仅调整存储路径不改变
Token 值，无需修改 Meegle 点位的 Token。代码更新不会自动迁移远端 KMS Secret。

ByteCloud AK/SK 由 MAAT/PPE 共用 `SecretsProvider`，staging/prod 从同一个 KMS Secret 读取 JSON：

```json
{"ak": "<bytecloud_application_ak>", "sk": "<bytecloud_application_sk>"}
```

定位配置为 `KMS_SECRET_BYTECLOUD_CREDENTIALS_ALIAS`，默认使用上表路径。轮转时提交完整的新 AK/SK 对；启动时读取并缓存同一 JSON 快照，更新后重启服务。空值、缺字段或格式错误会阻止启动，错误不会输出凭据值。

本地 `ENV=dev` 使用 `BYTECLOUD_AK`、`BYTECLOUD_SK`。所有环境凭据均支持 `.env` 和进程环境变量，并从 Settings 的 repr 与序列化中排除；staging/prod 仅使用 KMS。示例见 [`.env.example`](../.env.example)。

旧的 `KMS_SECRET_BYTECLOUD_SK_ALIAS` 已不再读取。已使用 `conciliator/credentials/bytecloud_sk` 的生产环境，须先在新路径保存配套的 AK/SK JSON 并授权服务读取，再部署新代码。代码不会自动迁移远端 Secret。

Coco 复用 Codebase 身份；飞书 OAuth 与长连接复用 Lark App Secret。ByteCloud JWT、ZTI SVID、tenant/plugin token 和任务 action token 是运行数据，不作为静态配置逐个存进 KMS。

## 仍需完成的上线工作

1. **远端配置与权限核验。** 创建/核对六项 Secret，尤其 ByteCloud AK/SK JSON 对应应用；授予 KMS、Codebase、Coco、飞书、Meegle、数据库、BITS/SCM/ICM/ENV/TCE 权限。代码接线不代表远端配置已完成。
2. **数据库迁移。** 当前 head 为 `0019_ppe_deployments`；读取目标库版本后安排缺失的增量 SQL 工单。仓库没有 `.codebase/pipelines/migrate.yaml`，迁移不能依赖该文件，也不放入容器启动脚本。
3. **就绪探针配置。** TCE 使用 `/readyz`：数据库可用时返回 HTTP 200，未初始化或不可用时返回 HTTP 503；`/healthz` 仅用于存活检查。
4. **入口保护。** 卡片任务查询与 Devbox worker 身份鉴权见下方 TODO。核对 Trace 入口保护；`TRACE_VIEWERS` 是用户名到 Basic Auth 密码的映射，当前为空时跳过该项认证。
5. **按实际模式处理其他敏感值。** 默认飞书长连接无需 HTTP 验签 Token/Encrypt Key；若改为 HTTP，现有卡片凭据仍需补 KMS 托管。Trace 密码也尚未接入 KMS。用户 OAuth token 的存储保护见下方 TODO；这些运行 token 不应逐个搬进 KMS。
6. **真实业务验收。** 按多进程配置完整启动后，验证 MR 冲突处理/审批/重试、Coding/飞书操作、Devbox 部署、PPE 两服务同泳道访问、升级、回收和重启恢复，再推进合入与发布。历史 HTTP/RDS 验证不覆盖这些业务流程。

本轮配置调整未创建远端 Secret 或修改线上配置；代码提交与推送不代表生产服务已经发布。

## 待后续处理的 TODO

以下三项已记录，本次仅提交文档，尚未修复；生产发布前须完成对应保护与验收。

- [ ] **卡片任务查询鉴权。** [`GET /api/aeolus/tasks/by-node`](../src/conciliator/api/aeolus.py) 当前未校验调用者身份或工作项权限，响应包含 `action_token`；已知节点标识的调用者可取得该 token 并调用受其保护的写接口。补充调用者身份与工作项访问权限校验，验收未授权查询不能取得任务或 token，合法卡片仍可查询和操作。
- [ ] **Devbox worker 身份鉴权。** [`worker 注册接口`](../src/conciliator/api/deployments.py) 当前允许匿名注册，[`注册服务`](../src/conciliator/services/deployment_multi.py) 会为相同 worker ID 替换并返回 session token，存在冒充 worker 的风险。注册时校验机器身份，并核对心跳、任务领取和结果回传的身份绑定；验收未授权调用不能注册、替换 session 或提交结果。
- [ ] **用户 OAuth token 存储保护。** 默认已配置飞书 OAuth 授权入口；[`用户 token repository`](../src/conciliator/repositories/lark_user_token_repo.py) 将 access/refresh token 写入数据库 [`Text 字段`](../src/conciliator/repositories/models.py)，授权实际使用后存在明文存储风险。落实持久化加密、密钥管理与轮转、数据库最小访问权限，验收落库值不含原始 token，刷新和撤销仍可用；目前未证实存在匿名读取这些 token 的接口。
