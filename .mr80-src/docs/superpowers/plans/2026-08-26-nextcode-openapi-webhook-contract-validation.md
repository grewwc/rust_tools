# NextCode OpenAPI / Webhook 契约改造与验证记录

日期：2026-08-26
运行验证更新：2026-08-27（KMS 脚本与 Webhook 持久化去重）
基线分支：`feat/infra-only-mode`（本地基线提交 `18a1d1a928cdd90611f1a832b3e7c312e0ed7ab6`）
工作分支：`feat/nextcode-contract-verification`
验证仓库：`xiahaitao/Conciliator`（Repo ID `1106267`）
App：`aeolus_conciliator_test`（App ID `792763146702616`，绑定 PSM `data.aeolus.conciliator`）

## 1. 结论摘要

本轮只验证任务依赖的单个 NextCode OpenAPI 和 Webhook，不做完整业务链路。

- 所需 App scopes 已安装到 Conciliator 仓库：
  `repo.metadata:read`、`repo.content:read`、`repo.content:write`、
  `repo.merge_request:read`、`repo.merge_request:write`、
  `repo.comment:read`、`repo.comment:write`、`repo:download`。
- OpenAPI 已从假想的 GitLab REST 路径改为官方统一格式：
  `POST https://codebase-api.byted.org/v2/?Action=<Action>`，请求体使用
  Thrift/IDL 的 CamelCase 字段，响应读取 `ResponseMetadata` 与 `Result`。
- OpenAPI 鉴权头改为 `X-Code-App-ID`、`X-Code-App-Secret` 和动态
  `X-Code-ZTI-Token`。OpenAPI 协议并不要求 KMS；此前真实验证通过时，由受控
  验证进程注入 App Secret。TCE 生产运行仍需一种合规的密钥注入方案，本计划选择
  KMSv2 托管 App Secret，不把明文写入代码、TCE 环境变量、测试、文档或日志。
  官方 `bytedztijwthelper==0.0.23` 已加入 TCE 内部依赖。
- Webhook 改用 `X-Vecode-Event`、`X-Vecode-Signature`、
  `X-Vecode-App-Id`、`X-Vecode-Event-Id`、`X-Vecode-Delivery`；签名算法为
  `sha256=` + `HMAC-SHA256(Webhook Secret token, raw body)` 的十六进制值，
  没有 timestamp 字段。
- Codebase App 配置现已完成。Webhook 验签从独立的 KMS alias 读取 token；当前
  配置按实际需要与 App Secret 使用相同值。两类用途分别托管在 KMS alias
  `conciliator/credentials/nextcode_app_secret` 和
  `conciliator/credentials/nextcode_webhook_secret`，代码和配置均不保存明文。
  KMS 控制面存在性和 ACL 已核对，仍需在 TCE Pod 内完成一次真实读取 smoke。
- 已通过 Netlink 创建稳定 HTTPS 入口 `https://aeolus-conciliator.byted.org`：
  TLB 集群为 `internal_api_lf_lb_v3`，路由使用 `/` 前缀，后端通过 Consul
  发现 `data.aeolus.conciliator/default` 的 TCE 实例。开发机 DNS 已生效，
  TLS 证书校验通过。
- 经稳定入口请求 `GET /healthz` 返回 HTTP 200 和
  `{"status":"ok","app":"conciliator"}`；请求 `POST /hooks/codebase`
  返回 HTTP 503 和 `{"detail":"external integrations are disabled"}`。
  该 503 是 `INTEGRATIONS_ENABLED=false` 下的预期应用响应，证明
  DNS、HTTPS、TLB、TCE 和 Conciliator receiver 的网络链路已经打通。
- 之前使用 `http://data.aeolus.conciliator/hooks/codebase` 的真实 Codebase
  Delivery 仍作为历史失败证据保留；新的 HTTPS 入口尚未在启用 NextCode receiver 后
  完成一次真实、带签名的 Codebase Delivery 2xx 验收。因此当前可以声明
  **Webhook 网络入口验证通过**，不能声明 **真实 Webhook 应用级处理验证通过**。
- `webhook_event.thrift` 没有 `conflict_changed` Action。冲突状态变更只能在
  `merge_request` 的 `opened/reopened/edited/pushed` 后查询
  `GetMergeRequestMergeability`，以及在目标分支 `push` 后先调用
  `ListRepoMergeRequests` 找到受影响的 open MR，再逐个查询 mergeability。
- Codebase UI 的“Comment review”不是 `comment` 事件：真实验证中它发出
  `merge_request_review`、`Action=commented`，正文位于 `Review.Content`。
  `CreateComment` OpenAPI 创建的评论则发 `comment/created`，载荷同时可能有
  `Comment` 和 `Comments`。接收端必须覆盖两条路径。
- `Comment.CreatedBy` 在真实 `comment/created` 中可能只有 ID，邮箱为空；
  作者邮箱需要回退到顶层 `Sender.Email`。行位置不在 Comment webhook 中，
  需要使用 `ThreadId` 调用 `GetThread` 读取 `Positions`。
- 当前 App OpenAPI service 未暴露 `ListRepoMembers` Action。仓库
  owner/master 列表能力 **无法实现、无法验证**；不能用
  `GetRepository.Creator` 冒充当前 admin。
- `repo:download` scope 已具备，但官方文档只允许 App 服务账号 clone 用于
  Bits 流水线。普通 TCE/本地容器 clone 本轮按合规要求 **主动不执行**；
  blame 使用合规的 `GetBlame` OpenAPI 并已验证。
- TLB 当前会把普通请求分配给“单机房”和“小流量”两个 TCE 实例；该问题不影响
  本轮 OpenAPI/Webhook 契约验证，暂不作为开启真实集成的阻塞项。Argos 应用日志
  接入和 TLB Metrics2.0 数据验证同样暂缓，本轮只使用 HTTP 响应和 Codebase
  Recent Deliveries 作为验收证据。

安全提醒：浏览器核对配置时 App Secret 曾短暂出现在自动化工具回显中，虽然没有
写入任何仓库文件，仍应在合入/上线前轮换 App Secret；同时生成一个与 App Secret
无关的新 Webhook Secret token。两者分别写入 KMS，轮换后重新执行本文的
OpenAPI/Webhook 验证。

## 2. 代码改动

### 2.1 OpenAPI client

- `src/conciliator/integrations/nextcode/client.py`
  - 统一 Action endpoint、官方鉴权头、ZTI 动态 token；
  - 解析 `ResponseMetadata.Error`、`RequestId`、`Result`；
  - 只对只读 Action 做瞬时错误重试，写 Action 不盲目重试；
  - Webhook 按 raw body 验证 `sha256=<hex>` HMAC。
- `src/conciliator/config.py`、`.env.example`、`byted-requirements.txt`
  - App ID 默认值设为 `792763146702616`；
  - endpoint 设为 `https://codebase-api.byted.org/v2/`；
  - KMS provider 已改为官方 `kmsv2.KmsClient().get_secret(full_alias)`，不再传
    `KMS_PSM` 或调用旧 `get_config_by_name`；
  - App Secret 与 Webhook Secret 使用两个必填、不同的完整 KMS alias，仓库配置
    不保存秘密明文；
  - 增加 `bytedztijwthelper==0.0.23`。

### 2.2 Action wrappers

- `branches.py`：`CreateBranch`、`CreateCommit`。
- `repos.py`：`GetRepository`、`GetBlame`；admin 查询显式抛出
  `RepositoryMembersOpenAPIUnavailable`，不再调用不存在的 members REST。
- `mrs.py`：`GetMergeRequest`、`CreateMergeRequest`、
  `UpdateMergeRequest`、`GetMergeRequestMergeability`、
  `ListRepoMergeRequests`，并解析 MR 作者与最新 Version。
- `merges.py`：`ListMergeRequestConflicts`、`GetMergeRequestConflict`；
  按 Thrift `binary` 的 JSON 表示对 `Content` 做 Base64 解码。
- `discussions.py`：`CreateComment`、`GetThread`；创建行评论时使用最新 MR
  Version 的 `BaseCommitId`（fallback `TargetCommitId`）和
  `SourceCommitId`，Position 为 `Type=content`、`Side=new`。
- `resolve_service.py`：冲突查询改用原 MR Number，而不是假想的
  source/target compare REST。

### 2.3 Webhook receiver

- `api/webhooks.py`
  - 使用独立的 `NEXTCODE_ENABLED` 和 `PIPELINE_ENABLED`：前者控制 KMS/NextCode
    receiver，后者控制 Coco、飞书、Worker 和写入型业务处理；
  - NextCode-only 模式仍执行验签、IDL 解析、MR/thread/mergeability 等只读补查，
    但跳过命令路由和自动解冲突副作用；Coco webhook 在 pipeline 关闭时返回 503；
  - 仅按官方 headers 和 CamelCase body 解析；
  - 覆盖 `comment`、`merge_request_review/commented`、`merge_request`、
    `push`；
  - `Comment` 与 `Comments` 去重读取；
  - 通过 `CommentableId` 调用 `GetMergeRequest`，得到仓库内 MR Number；
  - 通过 `GetThread` 补充评论文件/行位置；
  - 用 MR/Push 事件触发 mergeability 查询，而不读取不存在的
    webhook conflict 字段；
  - 以 `X-Vecode-Delivery`（缺失时回退到 `X-Vecode-Event-Id`）作为持久化
    去重键。成功投递重复返回 `deduplicated=true` 且不重复执行；失败记录允许
    Codebase 使用同一 Delivery ID 重投。
- `domain/commands.py`
  - 除原协议外，补充需求中明确写出的 `/approval` 与
    `/reject <reason>`。

### 2.4 可重复验证资产

- `scripts/verify_nextcode_contracts.py`：只输出脱敏结果；可只读查询，也可
  创建一组独立冲突分支/MR并逐项运行全部相关 OpenAPI；与服务共用
  `build_secrets_provider()`，不再读取已废弃的明文 `NEXTCODE_APP_SECRET`。
- `scripts/replay_nextcode_webhook.py`：从文件读取 raw body，按官方算法生成
  headers 并重放到接收端；从独立 Webhook Secret KMS alias 取值，可用
  `--delivery-id` 固定投递 ID 来重复验证去重。
- `scripts/verify_kms_secrets.py`：默认保留本地明文落盘验证能力；Pod 内使用
  `--redacted-only`，两项 secret 均可读取且非空即通过，同时脱敏报告二者是否
  相同，不写出明文。
- `migrations/versions/0002_nextcode_webhook_dedup.py`、
  `ddl/0002_nextcode_webhook_dedup.sql`：增加 Webhook Delivery 去重状态表；部署
  新应用版本前必须先升级数据库到 `0002_webhook_dedup`。
- `tests/fixtures/nextcode/`：Comment、Review Comment、MR edited、Push 的
  最小 CamelCase 接收端 fixtures；不含凭据。
- `tests/unit/test_nextcode_*.py`：固定 Action 名、CamelCase body、响应 envelope、
  HMAC、Base64 conflict、MR version position、Comment/Review/Push 路由。

## 3. 真实 OpenAPI 验证

保留的远程验证资源：

- MR：[!9](https://code.byted.org/xiahaitao/Conciliator/merge_requests/9)
- source：`verify/nextcode-20260826-1313-test1`
- target：`verify/nextcode-20260826-1313-test2`
- 冲突文件：`docs/nextcode-contract-fixtures/20260826-1313.txt`
- MR internal ID：`793125163377574`

| Action / 能力 | 真实结果 | 证据/结论 |
|---|---|---|
| `GetRepository` | 通过 | Path 查询得到 Repo ID `1106267`、默认分支 `main`；带官方 ZTI SDK 的保留脚本也成功 |
| `CreateBranch` | 通过 | 从 `main` 创建上述 test1/test2，初始 SHA 均为 `4439ba43...` |
| `CreateCommit` | 通过 | 两侧在同一路径创建不同内容；之后又在 target 更新一次以触发签名 Push webhook |
| `CreateMergeRequest` | 通过 | 创建 source -> target 的 MR !9，并写入 Description |
| `GetMergeRequest` | 通过 | 返回 repo、source/target branch、Version、URL、作者；本次作者为 App 账号，证明 `CreatedBy` 字段可取 |
| `UpdateMergeRequest` | 通过 | Description 更新后回读完全一致 |
| `ListRepoMergeRequests` | 通过 | 按 source/target/status=open 查询仅返回 MR !9 |
| `GetMergeRequestMergeability` | 通过 | 创建后及 target push 后均返回 `Mergeable=false`、`UnmergeableReason=conflict` |
| `ListMergeRequestConflicts` | 通过 | 返回 1 个文件和 `ConflictsId`；`Content` 是 Base64 字符串 |
| `GetMergeRequestConflict` | 通过 | 单文件详情解码内容与 List API 一致 |
| 冲突文件/行/块内容 | 通过 | 解码后有 `<<<<<<<`/`=======`/`>>>>>>>`，标记行分别为 1/3/5；应用 parser 可得冲突块起止行 |
| `GetBlame` | 通过 | source 分支 1-2 行返回 commit `af3354f2...` 与 App author 邮箱 |
| `CreateComment` 普通评论 | 通过 | comment/thread ID `793125257801783` |
| `CreateComment` 文件行评论 | 通过 | comment/thread ID `793125270362037`，目标文件第 1 行 |
| `GetThread` | 通过 | `Positions[0]` 精确回读 Path、StartLine=1、EndLine=1、base/source commit IDs |
| `repo:download` / clone | 未执行 | scope 已有；普通 TCE/本地 App clone 不符合文档限定，不能据此宣称 clone 已验证 |
| 仓库 owner/master | 不支持 | IDL 中有 request model，但 App OpenAPI service 没有对应 Action；无可调用 API |

## 4. 真实 Webhook 验证

App 订阅当前为 All events；验证所需 Push、Merge request、Review、Comment
均处于选中状态。下表记录的是创建稳定入口之前，使用
`http://data.aeolus.conciliator/hooks/codebase` 得到的历史 Delivery：

| Delivery | 事件 | 契约验证 | 接收结果 |
|---|---|---|---|
| `793125473781813` | `comment/created` | 有签名；同时包含 `Comment`、`Comments`，正文正确；真实 `CreatedBy.Email` 为空 | `Response 0`，失败 |
| `793125476376462` | `merge_request/edited` | 有签名；`Action=edited`，包含 MR Number、source/target branch、Diffs | `Response 0`，失败 |
| `793125476900750` | `push` | 有签名；Ref 为 target branch，含 Before/After/Commits/Created/Deleted/Forced | `Response 0`，失败 |
| `793125838689125` | `merge_request_review/commented` | 有签名；人类作者在 `Sender`/`Review.CreatedBy`，命令 `/resolve status` 在 `Review.Content` | `Response 0`，失败 |

`Response 0` 表示当时 Recent Deliveries 没有拿到 HTTP 响应（没有 response
headers/body）。这些记录继续用于证明 Codebase 侧事件、签名、headers 和 body
schema，但不再代表当前网络入口状态。

### 4.1 稳定 HTTPS 入口验证

- 入口：`https://aeolus-conciliator.byted.org`
- TLB 集群：`internal_api_lf_lb_v3`
- TLB 路由：前缀 `/`
- 后端：PSM `data.aeolus.conciliator`，TCE 逻辑集群 `default`
- DNS：开发机解析成功
- TLS：证书校验通过

功能探测结果：

```text
GET /healthz
HTTP 200
{"status":"ok","app":"conciliator"}

POST /hooks/codebase
HTTP 503
{"detail":"external integrations are disabled"}
```

结论：客户端到 Conciliator receiver 的 DNS、HTTPS、TLB 和 TCE 链路已经验证
通过，原“Webhook 接收联通失败”问题已解决。HTTP 503 来自应用自身的
infra-only 门禁，而非网络、证书或路由故障。

尚未完成的验收是：部署本分支契约改造、配置凭据、只开启 NextCode receiver 后，
由 Codebase 向新 HTTPS URL 发出真实签名 Delivery，并在 Recent Deliveries 中
观察到接收端的 2xx 响应。不能直接开启当前全局 `INTEGRATIONS_ENABLED`，否则会
连带初始化本轮不验证的 Coco、飞书和 Worker。

### 4.2 当前非阻塞的运行问题

- 普通请求仍会命中“单机房”和“小流量”两个 TCE 实例。采样 20 次中，13 次
  命中 `...:11495`，7 次命中 `...:11254`。当前契约改造部署到两个实例时，
  两者均可处理请求，因此暂不调整服务发现或权重；后续若要求只由单机房实例承载
  生产流量，再单独修复。
- `/healthz` 没有显式应用请求日志，当前 Uvicorn stdout 也未在 Argos 中检索到；
  应用日志中间件和 TCE stdout/Argos 采集暂缓，不作为真实 Webhook 验收门禁。
- `TLB Metrics2.0` 当前暂无数据，Argos 指标查询曾超时。功能探测已通过，但不能
  声明 Metrics 验证通过；后续作为可观测性任务处理。

## 5. 重复验证方法

### 5.1 本地确定性测试

```bash
uv run pytest tests/unit -q
uv run ruff check src tests scripts
```

### 5.2 App OpenAPI smoke

验证进程必须配置两个 KMS alias；脚本通过与服务相同的 KMS provider 读取
App Secret，不接受 `NEXTCODE_APP_SECRET` 明文环境变量。只读验证：

```bash
uv run python scripts/verify_nextcode_contracts.py
```

创建一套新的隔离冲突资源并验证全部 Action：

```bash
uv run python scripts/verify_nextcode_contracts.py \
  --provision-conflict-fixture \
  --run-id <unique-run-id>
```

该脚本不会修改 `main`，会从 `main` 创建
`verify/nextcode-<run-id>-test1/test2`。脚本输出可记录 MR URL、IDs 与每项
检查结果，但不会输出 App Secret/ZTI token。

### 5.3 Webhook 本地重放

启动只启用 NextCode receiver 的本地服务，并以安全方式向验证进程注入与
Codebase App Settings 一致的 Webhook Secret token，然后执行：

```bash
uv run python scripts/replay_nextcode_webhook.py \
  --event merge_request_review \
  --payload tests/fixtures/nextcode/merge_request_review_commented.json \
  --delivery-id repeatable-review-001

# 使用完全相同的 delivery-id 再执行一次，第二次预期：
# {"ok":true,"processed":0,"deduplicated":true}
uv run python scripts/replay_nextcode_webhook.py \
  --event merge_request_review \
  --payload tests/fixtures/nextcode/merge_request_review_commented.json \
  --delivery-id repeatable-review-001

uv run python scripts/replay_nextcode_webhook.py \
  --event merge_request \
  --payload tests/fixtures/nextcode/merge_request_edited.json

uv run python scripts/replay_nextcode_webhook.py \
  --event push \
  --payload tests/fixtures/nextcode/push.json
```

Comment fixture 中的 Thread ID 是占位值；要验证 `GetThread` 位置查询，应使用
真实 `CreateComment` 输出或 Recent Deliveries 捕获值替换。对 Codebase -> TCE
真实验证，触发新事件后必须在 Recent Deliveries 中同时确认：

1. `X-Vecode-Signature` 存在；
2. event header/body 与 IDL 一致；
3. Response 为 2xx，而不是历史的 `0` 或 infra-only 状态下的 503；
4. 接收端产生了与事件相符且无重复副作用的业务结果。

通过 Argos 按 Event ID/LogID 定位应用日志属于后续可观测性验收；在日志采集尚未
接通期间，不阻塞使用 Recent Deliveries、HTTP 响应和业务结果完成契约验证。

## 6. 明确未通过项与后续动作

1. **Webhook 网络入口已通过，应用级验收待完成**：稳定 HTTPS 入口已能访问
   `/healthz` 和 `/hooks/codebase`；KMS provider 和功能开关均已在代码中修正，
   下一步构建并部署本分支，然后只开启 NextCode receiver。其运行时 smoke 通过后，再把 App
   Webhook URL 更新为 `https://aeolus-conciliator.byted.org/hooks/codebase`，
   要求真实签名 Delivery 返回 2xx；本轮保持业务 pipeline 关闭。
2. **OpenAPI 契约已通过，KMS 代码已修正但 TCE 运行时尚未验证**：之前的真实
   OpenAPI 验证通过受控进程环境注入 App Secret，验证结论不受 KMS 影响。
   NextCode 协议只要求 `X-Code-App-Secret`，并不规定必须使用 KMS；Webhook
   验签需要的是 App Webhook 设置中的独立 Secret token，也不是协议上必须复用
   App Secret。生产 provider 已按 `bytedkmsv2==0.10.55` 改为
   `kmsv2.KmsClient().get_secret("keyring/key/secret")`；TCE 已集成 ZTI，client
   不再接收 `KMS_PSM`。两个独立 alias 已创建并授权 PSM
   `data.aeolus.conciliator` 读取；尚需在 TCE Pod 内执行 `--redacted-only` smoke，
   不能把明文写进 TCE 环境变量、文件或日志。
3. **代码开关已拆分，TCE 尚未配置**：`NEXTCODE_ENABLED` 现在独立控制 KMS 与
   NextCode receiver，`PIPELINE_ENABLED` 独立控制 Coco、飞书、Worker 和写入型
   业务处理；NextCode-only 启动测试已通过。旧 `INTEGRATIONS_ENABLED` 仅在两个
   新开关未显式配置时作为兼容默认值。TCE 当前尚未部署该版本，也尚未显式配置两个
   新开关。
4. **git clone 未验证**：合规文档仅允许 App 服务账号在 Bits pipeline clone。
   若业务必须在 TCE clone，需先取得 Codebase/安全合规书面确认并确定合规方案；
   当前实现不应开启普通 TCE App clone。
5. **仓库 admin 未实现**：没有公开 App Action。应向 Codebase OpenAPI oncall
   申请开放 repo member/admin API；开放前权限判断必须 fail closed，不能用 creator
   或 MR reviewer 推断 admin。
6. **没有原生冲突状态事件**：当前实现是事件后主动查询的合成语义。若需要严格的
   “从无冲突变为冲突/从冲突恢复”边沿事件，还需要持久化上次 mergeability 状态。
7. **数据库迁移是部署前置条件**：新 receiver 会写
   `nextcode_webhook_deliveries`；应用版本发布到 TCE 前必须先应用
   `ddl/0002_nextcode_webhook_dedup.sql` 或执行 Alembic upgrade，并确认当前 revision
   为 `0002_webhook_dedup`。
8. **凭据轮换**：本轮浏览器工具回显曾短暂显示 App Secret。上线前必须完成轮换，
   且不应继续使用本轮验证时的旧值。

## 7. 稳定入口完成后的详细执行方案

### 7.1 固化只读基线

开始修改控制台前，先保存以下基线，后续每次操作均记录时间、操作者、版本和结果：

- NextCode App 管理：<https://code.byted.org/profile/applications>
- Conciliator 仓库 App 安装：
  <https://code.byted.org/xiahaitao/Conciliator/settings/apps>
- KMSv2 中国区控制面：<https://security.bytedance.net/kmsv2>
- SCM 仓库 `581577`：
  <https://cloud.bytedance.net/scm/detail/581577/versions?x-bc-region-id=bytedance&x-resource-account=public>
- TCE 服务 `210017612`：
  <https://cloud.bytedance.net/tce/services/210017612?control_plane=cn&module=cluster&x-resource-account=public&x-bc-region-id=bytedance>

当前已知部署基线为 SCM `1.0.0.4`、提交
`627200738a303b892cc882cce848542138dbba7c`、TCE image version `1.0.0.7`；
TCE 服务 PSM 为 `data.aeolus.conciliator`，逻辑集群为 `default`。当前生产环境
`INTEGRATIONS_ENABLED=false`，且 `MountJwtBundles=true`、
`EnableZTIToken=false`。`feat/nextcode-contract-verification` 尚无 SCM 构建版本。

这里要明确两条结论：OpenAPI 接口格式与权限已经真实验证成功；下面的 KMS 工作只
解决 TCE 生产运行时如何安全取得 App Secret，不推翻也不重做 OpenAPI 契约结论。
Webhook 则另需一个独立 Secret token，供接收端 HMAC 验签。

### 7.2 第一阶段：先完成代码侧安全改造（KMS provider 与开关拆分已完成）

在修改 KMS、NextCode App 或 TCE 控制台前，先完成以下代码改造和测试：

1. **改正 KMSv2 provider**
   - 使用官方当前接口 `from kmsv2 import KmsClient`；
   - 使用 `KmsClient()` 创建 client，不向构造函数传 PSM；
   - 使用 `get_secret("keyring_alias/key_alias/secret_alias")` 读取敏感配置；
   - 删除对旧接口 `kmsv2.v1.KMSClient(psm).get_config_by_name(name)` 的依赖；
   - 不再把 `KMS_PSM` 作为 KMS client 的必填参数；TCE 身份由平台 ZTI 提供；
   - client 应在应用退出时关闭，启动错误只记录 alias 和错误类型，不记录秘密内容、
     长度、摘要或请求 header。

2. **为两类秘密使用不同配置项**
   - 例如 `KMS_SECRET_NEXTCODE_APP_SECRET_ALIAS`；
   - 例如 `KMS_SECRET_NEXTCODE_WEBHOOK_SECRET_ALIAS`；
   - 两者值都是完整的 `keyring/key/secret` alias，只是定位信息，不是秘密明文；
   - 禁止默认让 Webhook alias 回退到 App Secret alias。

3. **拆开功能开关**
   - 新增独立的 NextCode receiver 开关，例如 `NEXTCODE_ENABLED`；
   - 新增业务处理开关，例如 `PIPELINE_ENABLED`；
   - 当 `NEXTCODE_ENABLED=true`、`PIPELINE_ENABLED=false` 时，receiver 仍需完成
     验签、IDL 解析、事件去重和安全的只读补充查询，并返回 2xx，但不得启动
     Coco/Lark/Worker，也不得触发创建分支、MR、评论等业务副作用；
   - 保留旧 `INTEGRATIONS_ENABLED` 仅作兼容迁移：只在两个新开关均未显式配置时
     提供默认值；不能继续让 NextCode 验证依赖 dummy Coco/Lark 配置。

4. **增加事件幂等保护**
   - 以 `X-Vecode-Event-Id` 或 `X-Vecode-Delivery` 建立唯一键；
   - 相同 Delivery 重投时返回 2xx 和 `deduplicated` 结果，不重复执行副作用；
   - 幂等记录必须早于任何可能的写 OpenAPI 或消息发送。

5. **保留 TCE 内运行时 smoke 脚本**
   - 通过生产 secret provider 读取两类 secret，不接收命令行明文；
   - 动态取得 ZTI token，不打印 token；
   - 只读调用 `GetRepository`、`GetMergeRequest`、
     `GetMergeRequestMergeability`、`GetThread`；
   - 可在本机内存构造 raw body 并验证 HMAC，但不打印 secret、header 或摘要；
   - 输出只包含检查项 `OK/FAILED`、非敏感资源 ID 和 RequestId。

6. **代码验收**

   ```bash
   uv run pytest tests/unit -q
   uv run ruff check src tests scripts
   ```

   除已有 NextCode 契约测试外，还要覆盖实际 KMS client API、完整 alias、两个秘密不
   回退复用、NextCode-only 启动、pipeline-off 无副作用、签名失败和 Delivery 重投。

### 7.3 第二阶段：在 KMSv2 创建两个独立 Secret

代码侧配置名确定后，再进入 KMSv2 控制面：

1. 选择或创建 Conciliator 专用 keyring。
2. 创建用途为“敏感配置”的 key；不要把普通加解密 key 当成敏感配置容器。
3. 给 TCE PSM `data.aeolus.conciliator` 授予包含 `GetSecret` 的最小只读权限。
   控制台权限名称可能调整，应以页面上实际显示的“读取敏感配置/GetSecret”能力为准，
   不授予写、删除或 key 管理权限。
4. 在该 key 下分别创建两个 secret alias，例如：
   - `.../.../nextcode_app_secret`
   - `.../.../nextcode_webhook_secret`
5. KMS 页面中的 secret 名称可能只填写最后一段；代码/TCE 配置中必须使用
   `keyring_alias/key_alias/secret_alias` 完整 alias。
6. KMS 敏感配置写入后，控制台不会再次展示明文。先确认 alias 和授权，再保存；
   不要通过聊天、工单、shell history 或 TCE 环境变量传递秘密。

App Secret 和 Webhook Secret token 的来源分别为：

- **OpenAPI App Secret**：在 NextCode App 管理页轮换此前曾回显的 App Secret，
  将新值直接写入 `nextcode_app_secret`，不复制到仓库或 `.env.example`。
- **Webhook Secret token**：用公司批准的密码生成/保管工具生成一个独立随机值，
  写入 `nextcode_webhook_secret`；稍后把同一个值配置到 App 的 Webhook Secret
  token。它不应等于 App Secret。

NextCode 协议不强制使用 KMS。如果已有另一个经确认合规、支持 TCE 运行时注入且
不会暴露明文的 secret provider，可以用它替代 KMS；当前仓库没有这样的 provider，
因此本计划默认 KMSv2。动态 ZTI token 不需要、也不应保存到 KMS。

### 7.4 第三阶段：核对 NextCode App 与仓库安装

1. 在 App 管理页确认 App ID 仍为 `792763146702616`，完成 App Secret 轮换。
2. 在 Conciliator 仓库 App 设置页确认该 App 已安装，且仍有任务要求的 read/write、
   comment、merge request、content 和 `repo:download` scopes。
3. Webhook 事件只需包含本轮会验证的 Comment、Merge request、Review 和 Push；
   不需要为了方便订阅与任务无关的事件。
4. 此阶段先不要触发验证事件。Webhook URL 和独立 Secret token 应在新版本部署、
   KMS smoke 通过、NextCode-only receiver 就绪后再一起更新，避免产生预期失败投递。

### 7.5 第四阶段：构建可追溯的 SCM 版本

1. 在本地完成代码检查后，提交并推送 `feat/nextcode-contract-verification`。
2. 在 SCM `581577` 从该分支的**精确提交**创建新版本，不部署浮动分支头。
3. 等待构建状态为 `build_ok`，记录 SCM version、Git commit、镜像版本和构建时间。
4. 检查构建产物确实包含 `bytedkmsv2` 与 `bytedztijwthelper`。当前已部署
   `1.0.0.4` 的依赖记录中没有 `bytedztijwthelper`，因此不能直接用旧产物验证。
5. 保留 SCM `1.0.0.4` / image `1.0.0.7` 作为回滚基线。

### 7.6 第五阶段：关闭功能开关部署新版本

先让两个会接收流量的实例运行相同新版本，但保持 NextCode 和业务 pipeline 关闭。
当前小流量实例仍会承接普通请求，因此即使暂不调整权重，也必须一起滚动成功。

TCE 的环境变量更新语义是整组替换；修改时必须带上现有全部变量，不能只提交新增项，
否则被省略的旧变量会被删除。建议的新配置如下，实际变量名以 7.2 的最终代码为准：

```text
ENV=prod
SERVICE_PSM=data.aeolus.conciliator
INTEGRATIONS_ENABLED=false
NEXTCODE_ENABLED=false
PIPELINE_ENABLED=false
MYSQL_PSM=toutiao.mysql.aeolus_govern_write
MYSQL_DB_NAME=aeolus_govern
PYTHONPATH=/opt/tiger/conciliator/src
LOG_LEVEL=INFO
NEXTCODE_APP_ID=792763146702616
NEXTCODE_API_BASE=https://codebase-api.byted.org/v2/
NEXTCODE_ZTI_REQUIRED=true
NEXTCODE_WEBHOOK_PATH=/hooks/codebase
KMS_SECRET_NEXTCODE_APP_SECRET_ALIAS=<keyring/key/nextcode_app_secret>
KMS_SECRET_NEXTCODE_WEBHOOK_SECRET_ALIAS=<keyring/key/nextcode_webhook_secret>
```

不要在这里填写 App Secret、Webhook Secret 或静态 ZTI token。部署完成后确认：

- 两个实例均为 Running，版本和 commit 完全一致；
- `GET /healthz` 返回 200；
- `GET /readyz` 返回 200；
- `POST /hooks/codebase` 仍返回 503，这是 `NEXTCODE_ENABLED=false` 下的预期结果。

### 7.7 第六阶段：在 TCE Pod 内做生产凭据 smoke

使用 TCE 实例页 WebShell，在两个会承接流量的实例中运行 7.2 保留的 smoke 脚本。
至少要确认：

1. `kmsv2.KmsClient` 可导入；
2. 两个完整 alias 均可取得非空 secret，但脚本不输出任何秘密信息；
3. `bytedztijwthelper` 可动态取得 token；
4. `GetRepository` 可读取 Repo ID `1106267`；
5. `GetMergeRequest` 和 `GetMergeRequestMergeability` 可读取 MR !9；
6. `GetThread` 可读取保留的 thread；
7. 用 KMS 中 Webhook token 对本地 raw body 做一次 HMAC 验证成功。

常见失败应按下面方向处理，不要退回明文凭据：

| 失败 | 优先检查 |
|---|---|
| `ImportError` | SCM 构建产物是否包含正确依赖 |
| alias 不存在 | TCE 配置是否使用完整 `keyring/key/secret` alias |
| KMS `Forbidden` | PSM 是否具有最小 `GetSecret` 权限 |
| ZTI/Unauthorized | TCE 身份挂载、SDK 运行环境和 NextCode App 绑定 PSM；不要设置静态 token |
| `ResponseMetadata.Error` | Action body、App scope、仓库安装状态和 RequestId |

当前 `EnableZTIToken=false` 不应在没有证据时直接修改。KMS 官方说明 TCE 已提供 ZTI
身份；先以实际 SDK smoke 为准，若 ZTI helper 确实失败，再根据具体错误和平台文档
确认该 feature gate 是否相关。

### 7.8 第七阶段：只开启 NextCode receiver

1. 通过一次完整 TCE 环境变量更新设置 `NEXTCODE_ENABLED=true`，保持
   `PIPELINE_ENABLED=false`、旧 `INTEGRATIONS_ENABLED=false`。
2. 滚动两个实例并重新检查 `/healthz`、`/readyz`。
3. 向 `/hooks/codebase` 发送一个没有签名的空请求。此时预期从原 503 变为 401，
   证明请求已进入签名校验器；不能把无签名请求返回 2xx 当作成功。
4. 在 Codebase App 设置中把 URL 更新为
   `https://aeolus-conciliator.byted.org/hooks/codebase`，并写入与 KMS
   `nextcode_webhook_secret` 完全一致的独立 Webhook Secret token。
5. 再次确认健康检查正常。不要在页面或日志截图中暴露 token。

### 7.9 第八阶段：逐项验证真实 Webhook

建议按从无副作用到有状态的顺序验证：

1. 在 MR !9 写一条不以命令开头的普通评论，验证 `comment/created`；
2. 写一条非命令的文件行 Review Comment，验证
   `merge_request_review/commented`、`Review.Content` 和位置补查；
3. 修改 MR Description，验证 `merge_request/edited`；
4. 向隔离的 target test branch push 一个提交，验证 `push` 和随后只读
   mergeability 查询；
5. pipeline 关闭时可验证 `/resolve status` 的识别和只读响应；不要在本阶段执行会
   创建分支/MR的 `/resolve`，也不要伪造 cancel/retry/approval/reject 的业务状态。

每次在 Recent Deliveries 和接收端响应中记录：事件类型、Action、Event ID、
Delivery ID、签名头是否存在、body schema、HTTP 状态码和脱敏处理结果。通过标准为：

- 签名正确的真实 Delivery 返回 2xx；
- 签名错误或缺失返回 401；
- body 与 `webhook_event.thrift` 一致；
- pipeline 关闭时没有创建分支、MR、评论或发送飞书消息；
- 对同一个 Delivery 执行 Redeliver，仍返回 2xx，并明确为 `deduplicated`，无重复
  副作用。

Argos 日志尚未接通不阻塞本阶段；HTTP 响应、Recent Deliveries 和 Codebase 资源
状态足以作为契约验证证据。

### 7.10 第九阶段：在 TCE 运行路径复核 OpenAPI

OpenAPI 已经真实验证成功，本阶段不是重新确认 schema，而是确认 TCE 生产路径中的
KMS、动态 ZTI、网络和新构建产物组合可用：

1. 先保留 7.7 的只读 smoke 结果；
2. 再使用 `scripts/verify_nextcode_contracts.py --provision-conflict-fixture`
   创建带唯一 run-id 的隔离 test1/test2 分支和 MR；
3. 复核 `CreateBranch`、`CreateCommit`、`CreateMergeRequest`、Description、冲突
   列表/详情、行与冲突块、`GetBlame`、普通评论、文件行评论和 `GetThread`；
4. 不修改 `main`，不复用业务 MR，不执行不合规的 TCE git clone；
5. 保存 MR URL、分支名、非敏感 ID、RequestId 和逐项结果，不保存 secret/ZTI。

### 7.11 本轮保持 pipeline 关闭

本轮目标是单个 NextCode OpenAPI/Webhook 契约验证，不是完整 resolve 链路。只有在
Coco、飞书、Worker、仓库白名单、命令状态机和副作用幂等均另行验证后，才能考虑
`PIPELINE_ENABLED=true`。在此之前，真实 Webhook 只做验证、解析、事件去重和允许的
只读查询。

### 7.12 失败停止条件与回滚

- KMS、ZTI 或 OpenAPI smoke 失败：保持 `NEXTCODE_ENABLED=false`，修正后重试；
- receiver 启动或健康检查失败：回滚到 SCM `1.0.0.4` / image `1.0.0.7`；
- 真实 Webhook 非 2xx：先关闭 `NEXTCODE_ENABLED`，保留 Delivery ID、响应和
  RequestId，再定位签名、schema 或下游只读调用；
- 发现 pipeline-off 仍有写副作用：立即关闭 receiver，并修正开关和幂等边界；
- 任何失败都不得改用 TCE 明文 secret、代码常量或静态 ZTI token 绕过。

### 7.13 完成标准

本轮只有同时满足下列条件才算完成：

- 两个 TCE 实例运行同一个可追溯 SCM 版本；
- App Secret 与 Webhook Secret token 已轮换且分开托管；
- TCE Pod 内 KMS、动态 ZTI 和只读 OpenAPI smoke 全部通过；
- 四类真实 Webhook 均带签名并返回 2xx；
- 重投同一 Delivery 可去重；
- TCE 生产路径中的隔离 OpenAPI 回归通过；
- 文档记录每项结果，并明确 clone、repo admin 等未支持/未验证能力；
- `PIPELINE_ENABLED` 仍保持 false。

小流量实例权重、Argos 应用日志以及 TLB Metrics2.0 不进入上述 P0 顺序；完成真实
Webhook/OpenAPI 契约验收后再分别建立运行治理和可观测性任务。
