# NextCode OpenAPI / Webhook 契约验证记录（修订版）

日期：2026-08-27
最后更新：2026-08-27
状态：本轮要求的 TCE OpenAPI 与真实 Webhook 契约验证已完成
工作分支：`feat/nextcode-contract-verification`
TCE 验证提交：`9f118143abdbd2230524e9f6d75797b8be40c332`
SCM / TCE 版本：`1.0.0.9`
验证仓库：`xiahaitao/Conciliator`（Repo ID `1106267`）
App：`aeolus_conciliator_test`（App ID `792763146702616`，PSM `data.aeolus.conciliator`）

本文承接 `2026-08-26-nextcode-openapi-webhook-contract-validation.md` 的过程记录，
是截至 2026-08-27 的权威修订版。
旧文件保留过程记录，其中“尚未部署”“真实 Webhook 待验证”“Delivery 优先去重”
等结论已经过时，应以本文为准。

## 1. 最终结论

本轮只验证 Conciliator 需要的单个 NextCode OpenAPI 和 Webhook，不验证完整的
AI 解冲突业务链路。结论如下：

- TCE Pod 已能通过两个 KMS alias 读取非空敏感配置、动态获取 ZTI token，并以
  App 身份调用 NextCode OpenAPI。两个 alias 当前按项目实际配置保存相同值；这是
  已确认的配置选择，不构成验证失败，脚本也不再要求二者必须不同。
- 在 TCE Pod 内创建了一套新的隔离冲突资源 MR !10，并逐项跑通本轮所需的读写
  OpenAPI。这次结果来自生产 TCE 运行时，覆盖了 KMS、ZTI、网络和构建产物的实际
  组合，不沿用此前的本地验证结论。
- Codebase 到稳定 HTTPS 入口的真实签名 Webhook 已返回 HTTP 200。Comment 正文、
  作者、Thread ID、文件路径和行号均已验证；MR、source push、target push、评论更新、
  Review 忽略和 Event-ID 去重也均有真实 Delivery 证据。
- 当前 `PIPELINE_ENABLED=false`，所以 MR 冲突检查成功后返回 `processed=0` 是预期
  结果：receiver 会查询 mergeability，但不会创建 resolve 分支、MR、评论或发送
  飞书消息。
- `webhook_event.thrift` 没有原生 `conflict_changed` 事件。当前实现以
  `merge_request/opened|reopened|edited|pushed` 和目标分支 `push` 为触发器，主动
  重新查询 MR 冲突状态。
- `repo:download` scope 已安装，但普通 TCE 容器内 clone 不符合现有文档限定，本轮
  未执行；`GetBlame` 已通过 OpenAPI 验证。
- 当前 App OpenAPI 未暴露仓库成员/admin 查询 Action，owner/master 获取能力仍然
  不支持，不能用仓库 creator 或 MR reviewer 冒充 admin。

## 2. 当前运行基线

TCE 当前配置：

```text
ENV=prod
SERVICE_PSM=data.aeolus.conciliator
INTEGRATIONS_ENABLED=false
NEXTCODE_ENABLED=true
PIPELINE_ENABLED=false
MYSQL_PSM=toutiao.mysql.aeolus_govern_write
MYSQL_DB_NAME=aeolus_govern
PYTHONPATH=/opt/tiger/conciliator/src
LOG_LEVEL=INFO
KMS_SECRET_NEXTCODE_APP_SECRET_ALIAS=conciliator/credentials/nextcode_app_secret
KMS_SECRET_NEXTCODE_WEBHOOK_SECRET_ALIAS=conciliator/credentials/nextcode_webhook_secret
```

已确认：

- 稳定入口：`https://aeolus-conciliator.byted.org`
- Webhook URL：`https://aeolus-conciliator.byted.org/hooks/codebase`
- `/healthz`：HTTP 200
- `/readyz`：HTTP 200
- `0002_webhook_dedup` DDL 工单已执行，真实重投验证已成功写入并查询去重表
- `NEXTCODE_ENABLED=true`：启用验签、事件解析、去重和 NextCode 补充查询
- `PIPELINE_ENABLED=false`：禁用完整业务 pipeline 及其写副作用

App 当前具备本轮使用的 `repo.metadata:read`、`repo.content:read/write`、
`repo.merge_request:read/write`、`repo.comment:read/write` 和 `repo:download`。
Webhook 控制面目前订阅 All events，范围大于 receiver 实际处理集合；receiver 会按
第 4.1 节的白名单处理或忽略事件。App Secret、Webhook Secret 和 ZTI token 的值
均不写入本文或仓库；保留脚本也不会打印这些值。

## 3. TCE OpenAPI 完整验证

### 3.1 验证方法与资源

在生产 TCE Pod 内运行：

```bash
cd /opt/tiger/conciliator
/opt/tiger/toutiao/runtime/bin/python \
  scripts/verify_nextcode_contracts.py \
  --provision-conflict-fixture \
  --run-id 20260827-tce-2148
```

脚本退出码为 `0`，结果中没有 `failure`。本次创建的隔离资源：

- MR：[!10](https://code.byted.org/xiahaitao/Conciliator/merge_requests/10)
- MR global ID：`793371490678684`
- source：`verify/nextcode-20260827-tce-2148-test1`
- target：`verify/nextcode-20260827-tce-2148-test2`
- 冲突文件：`docs/nextcode-contract-fixtures/20260827-tce-2148.txt`
- base commit：`4439ba43f27069b19c0e3dae1b893be0af7cb777`
- source commit：`2719174a97e348ce05a38cdcb0863c2979da3410`
- target commit：`6218f10ccc6c147bd4abf1712615ee01b42f80ff`

### 3.2 逐项结果

| Action / 能力 | TCE 结果 | 证据/结论 |
|---|---|---|
| `GetRepository` | 通过 | Path 查询得到 Repo ID `1106267`；验证资源从 `main` 创建 |
| `CreateBranch` | 通过 | 从 `main` 创建上述 source/target 两个隔离分支 |
| `CreateCommit` | 通过 | 两侧在同一路径写入不同内容，得到上述 source/target commit |
| `CreateMergeRequest` | 通过 | 创建 source -> target 的 MR !10，并写入 Description |
| `GetMergeRequest` | 通过 | 回读 repo、source/target branch、Version、URL 和 `CreatedBy`；作者为 App 账号 `aeolus_conciliator_test@bytedance.com` |
| `UpdateMergeRequest` | 通过 | Description 增加 `Description update API: verified.` 后回读一致 |
| `ListRepoMergeRequests` | 通过 | 按 source、target、`status=open` 查询仅返回 MR !10 |
| `GetMergeRequestMergeability` | 通过 | `Mergeable=false`、`UnmergeableReason=conflict`，详情为 code conflict |
| `ListMergeRequestConflicts` | 通过 | 返回 1 个冲突文件、`ConflictsId` 和 Base64 `Content` |
| `GetMergeRequestConflict` | 通过 | 单文件详情解码内容与 List API 一致 |
| 冲突文件/行/块内容 | 通过 | marker 行为 1/3/5；应用 parser 得到 1 个 `start_line=1`、`end_line=5` 的冲突块，ours/theirs 分别为 source/target 内容 |
| `GetBlame` | 通过 | source 分支 1-2 行返回 source commit 和 App author 邮箱 |
| `CreateComment` 普通评论 | 通过 | comment/thread ID `793371492764957` |
| `CreateComment` 文件行评论 | 通过 | comment/thread ID `793371492792661`，目标文件第 1 行 |
| `GetThread` | 通过 | 精确回读 Path、`StartLine=1`、`EndLine=1`、base/source commit IDs；Position 为 `Type=content`、`Side=new` |
| `repo:download` / clone | 未执行 | scope 已有；普通 TCE App clone 仍需 Codebase/安全合规确认，不能宣称已验证 |
| 仓库 owner/master | 不支持 | 当前 App OpenAPI service 没有可调用的 repo member/admin Action |

除 App 自身的脚本结果外，还使用 `bytedcli codebase` 独立回读了 MR !10、mergeability、
评论列表、行位置和 blame，结果一致。

## 4. 真实 Webhook 验证

### 4.1 当前接收规则

receiver 当前只按以下规则处理事件：

| 事件 | Action | 当前行为 |
|---|---|---|
| `comment` | `created` | 唯一评论/命令入口；读取正文、作者、Thread ID，并通过 `GetThread` 补充文件行位置 |
| `comment` | `updated` | 忽略并返回 200，避免编辑评论重复执行命令 |
| `merge_request` | `opened/reopened/edited/pushed` | 重新调用 `GetMergeRequestMergeability` |
| `push` | 目标分支 push | 查询以该分支为 target 的 open MR，再逐个查询 mergeability |
| `merge_request_review` | 任意 | 忽略并返回 200；不再作为第二条评论业务链路 |

`comment/created` 承载 `/resolve`、`/resolve cancel`、`/resolve retry`、
`/resolve approve all`、`/resolve status`、`/approval` 和 `/reject <reason>`。
本轮 pipeline 关闭，因此只验证接收、补查和去重，不执行这些命令的完整业务副作用。

### 4.2 Comment 正文、作者、Thread 与位置

MR !9 上的人类文件行评论验证结果：

- Delivery：`793370356097220`
- HTTP 200：`{"ok":true,"processed":1}`
- `Action=created`
- comment ID：`793370356103237`
- Thread ID：`793125270362037`
- 正文：`NextCode inline webhook validation 2026-08-27: verify body author thread path line`
- 作者：`cuijiawei.2026@bytedance.com`
- 文件：`docs/nextcode-contract-fixtures/20260826-1313.txt`
- 位置：`Side=new`，`StartLine=1`，`EndLine=1`

真实 payload 中 `Comment.CreatedBy.Email` 为空，顶层 `Sender.Email` 完整；receiver
按设计回退到 `Sender.Email`，因此人类作者获取通过。Webhook 只给出 Thread ID，
文件名和行号由 `GetThread` 补查得到，结果与评论创建位置完全一致。

本次 TCE OpenAPI 脚本创建 MR !10 后，又自动产生了两条真实签名 Comment Delivery：

| Delivery | Comment | HTTP 结果 | 结论 |
|---|---|---|---|
| `793371493851929` | 普通评论 `793371492764957` | 200，`processed=1` | 正文、Thread ID 和 App 作者回退正常 |
| `793371492794946` | 文件行评论 `793371492792661` | 200，`processed=1` | receiver 成功调用 `GetThread`；TCE 回读路径和第 1 行准确 |

伴生的 `merge_request_review` Delivery `793371492819490` 返回 HTTP 200、
`processed=0`，证明 Review 事件被安全忽略，没有形成第二条评论处理链路。

位置补查还有一个需要后续加固的边界：当前实现取 `thread.positions[0]`。真实 MR !9
在分支更新后同一 Thread 返回了两个 version position，本次两者路径和行号相同，
所以结果正确；若未来评论位置随 MR version 移动，应明确按最新版本选择 Position，
不能长期假设第一个元素就是最新位置。

### 4.3 MR 与 Push 触发冲突复查

已验证：

- MR !10 创建产生 `merge_request/opened` Delivery `793371491049869`，HTTP 200。
- 更新 MR !10 Description 产生 `merge_request/edited` Delivery
  `793371492826572`，HTTP 200、`processed=0`。
- MR !9 source 新增 commit `e5538291978a1367294713785e9a94e9640d2c51` 后，
  收到 source `push` 和 `merge_request/pushed`；MR Delivery
  `793370517779218` 为 HTTP 200。
- MR !9 target 新增 commit `a8ec9e7c8823824bbcb5a2e32ec0cdfe0f4fc461` 后，
  target `push` Delivery `793370608159172` 为 HTTP 200，Ref 精确指向 target 分支。
- source/target 更新后再次查询 MR !9，仍为
  `Mergeable=false`、`UnmergeableReason=conflict`。

代码在 MR/target-push 处理路径中先查询 mergeability，再检查
`PIPELINE_ENABLED`。因此上述 `processed=0` 表示“冲突补查完成但 pipeline 关闭”，
不是事件漏收或 OpenAPI 失败。

当前实现提供的是“收到可能影响冲突状态的事件后重新查询”的合成语义，不是原生
冲突变化事件。如果以后需要只在 `mergeable -> conflict` 或
`conflict -> mergeable` 的边沿触发动作，还需持久化上一次 mergeability 状态。

### 4.4 评论更新、Review 忽略和传输层去重

- Comment updated Delivery `793369257217645`：HTTP 200、`processed=0`。
- Review Comment Delivery：HTTP 200、`processed=0`。
- 首次 Delivery `793369016108619`：HTTP 200、`processed=1`。
- Codebase 重投 Delivery `793369118779383`：携带与首次相同的
  `X-Vecode-Event-Id`，返回
  `{"ok":true,"processed":0,"deduplicated":true}`。
- 数据库中该事件只有 1 条去重记录，key 为 `event:<event-id>`。

当前正确的去重优先级是：

1. 优先使用 `X-Vecode-Event-Id`；
2. 仅在 Event-ID 缺失时使用 `X-Vecode-Delivery`；
3. 两者都缺失时记录告警，但不能建立传输层唯一键。

旧文档中“Delivery 优先、Event-ID 兜底”的说法已经失效。由于 Comment 现在只有
`comment/created` 一条业务入口，暂不增加 Comment ID 业务级幂等；单个 payload 同时
出现 `Comment` 和 `Comments` 时，receiver 仍会按 Comment ID 做本次请求内去重。

## 5. 新会话验证手册

新会话应先读本节，再看第 3、4 节的历史证据。当前证据对应 SCM/TCE `1.0.0.9`
和提交 `9f118143abdbd2230524e9f6d75797b8be40c332`。以后部署了新版本，不要直接沿用
这里的“通过”结论；先记录新版本，再完整执行本节。

### 5.1 验证范围和安全边界

本方案会在 `xiahaitao/Conciliator` 创建测试分支、commit、MR 和评论，但不会修改
`main`。开始前必须确认：

- 只使用 `verify/nextcode-<run-id>-test1/test2` 这一类隔离分支；
- `INTEGRATIONS_ENABLED=false`、`NEXTCODE_ENABLED=true`、
  `PIPELINE_ENABLED=false`；
- 不显示、不复制、不打印 App Secret、Webhook Secret 或 ZTI token；
- KMS 或 ZTI 失败时，不得改用明文环境变量、代码常量或静态 token 绕过；
- 不在普通 TCE Pod 中执行 git clone；
- 不自动删除旧测试 MR 和分支，先保留证据；
- 写 OpenAPI 没有文档化幂等键。失败后不要用同一个 run ID 盲目重跑，应先查看
  已经创建了哪些资源，再换新的唯一 run ID。

本轮不运行 `/resolve`、`/approval` 或 `/reject` 等命令。验证评论时使用不以 `/`
开头的普通文本，避免开关配置错误时误触发业务动作。

### 5.2 新会话先打开这些入口

- 本地仓库：`/Users/bytedance/Documents/ChatGPT/Conciliator`
- 远程仓库：<https://code.byted.org/xiahaitao/Conciliator>
- NextCode App：<https://code.byted.org/profile/applications>
- 仓库 App 安装：<https://code.byted.org/xiahaitao/Conciliator/settings/apps>
- TCE 实例：<https://cloud.bytedance.net/tce/services/210017612?control_plane=cn&module=instance&x-resource-account=public&x-bc-region-id=bytedance>
- SCM 版本：<https://cloud.bytedance.net/scm/detail/581577/versions?x-bc-region-id=bytedance&x-resource-account=public>
- RDS 查询：<https://cloud.bytedance.net/rds/detail/db/global/aeolus_govern/autoSQL?x-global-site=cn&x-global-vregion=China-North&x-global-vdc=cn&x-global-detail-tab=China-North&x-resource-account=public&x-bc-region-id=bytedance&x-bc-vregion=ChinaSinf-North&x-bc-vdc=multicloud>
- OpenAPI 接入指南：<https://bytedance.larkoffice.com/wiki/PKoiwOfVniSnO6kEomWcett0nIc>
- Webhook 接入指南：<https://bytedance.larkoffice.com/wiki/DOOqwgVBWiJ5Zukjjf2cdRJJnOg>
- Webhook IDL：`/Users/bytedance/Downloads/backend/api/kitex/webhook_event.thrift`

不要保存或复用带 `webshell_token` 的签名 URL。每次从 TCE 实例页点击当前实例的
WebShell 进入。若文档、代码和观察到的 schema 不一致，以 NextCode 接入文档和
`webhook_event.thrift` 为准，不能猜字段。

### 5.3 记录部署基线

先在本地记录仓库状态：

```bash
cd /Users/bytedance/Documents/ChatGPT/Conciliator
git status --short --branch
git rev-parse HEAD
git log -1 --format='%H %s'
```

再从 SCM/TCE 页面记录：

- SCM version 和对应 Git commit；
- 两个 Pod 的镜像/版本、运行状态和更新时间；
- 当前完整环境变量，但不要记录任何 secret 明文；
- App ID、Webhook URL、SSL 校验状态和已安装 scopes。

两个会承接流量的 Pod 必须运行同一精确版本。若版本不同，先停止验证并完成滚动
更新。工作区有用户改动时只记录，不要 reset、checkout 或覆盖。

### 5.4 两个 Pod 都要做的只读 smoke

分别进入两个 Running Pod。服务端口读取 `$PORT0`，不能写死为 8000：

```bash
echo "PORT0=${PORT0:?PORT0 is not set}"
curl -fsS "http://127.0.0.1:${PORT0}/healthz"
curl -fsS "http://127.0.0.1:${PORT0}/readyz"

cd /opt/tiger/conciliator
/opt/tiger/toutiao/runtime/bin/python - <<'PY'
from importlib.metadata import version

from kmsv2 import KmsClient  # noqa: F401
from ztijwthelper import ZTIJwtHelper

for package in ("bytedkmsv2", "bytedztijwthelper"):
    print(f"{package}={version(package)}")

token = str(ZTIJwtHelper().get_jwt_svid())
assert token
print("imports=ok")
print("zti_token=ok")
PY

/opt/tiger/toutiao/runtime/bin/python \
  scripts/verify_kms_secrets.py --redacted-only

/opt/tiger/toutiao/runtime/bin/python \
  scripts/verify_nextcode_contracts.py
```

每个 Pod 的预期结果：

| 检查 | 预期 |
|---|---|
| `/healthz` | HTTP 200，`status=ok` |
| `/readyz` | HTTP 200，`status=ok` |
| SDK | `bytedkmsv2`、`bytedztijwthelper` 可查询版本并成功 import |
| ZTI | 动态取得非空 token，只打印 `zti_token=ok` |
| KMS | 两个 alias 均为非空可读；值相同或不同都不算失败；不生成明文文件 |
| 只读 OpenAPI | `GetRepository` 成功，Repo ID 为 `1106267`，脚本退出码为 0 |

任一 Pod 的 health、ready、KMS、ZTI 或只读 OpenAPI 失败，都应停止后续写验证。
记录 Pod 名、时间、错误类型、NextCode RequestId 和脱敏响应，不要把秘密写入日志。

### 5.5 只在一个同版本 Pod 运行完整 OpenAPI 验证

两个 Pod 的只读 smoke 都通过后，选一个 Pod 运行完整脚本：

```bash
cd /opt/tiger/conciliator
NEXTCODE_RUN_ID="$(date +%Y%m%d)-tce-$(date +%H%M%S)"
NEXTCODE_RESULT="/tmp/nextcode-openapi-${NEXTCODE_RUN_ID}.json"

/opt/tiger/toutiao/runtime/bin/python \
  scripts/verify_nextcode_contracts.py \
  --provision-conflict-fixture \
  --run-id "${NEXTCODE_RUN_ID}" \
  > "${NEXTCODE_RESULT}"
NEXTCODE_EXIT=$?

sed -n '1,400p' "${NEXTCODE_RESULT}"
echo "exit=${NEXTCODE_EXIT} result=${NEXTCODE_RESULT}"
test "${NEXTCODE_EXIT}" -eq 0
```

输出中必须保存以下非敏感信息：run ID、MR URL、MR Number/global ID、source/target
分支、base/source/target commit、两个 comment/thread ID 和行评论 Position。

逐项通过标准：

| Action / 检查 | 通过标准 |
|---|---|
| `GetRepository` | Repo ID `1106267`、Path `xiahaitao/Conciliator` |
| `CreateBranch` | 从 `main` 创建唯一 source/target 分支 |
| `CreateCommit` | 两侧 commit 成功，SHA 非空且不同 |
| `CreateMergeRequest` | source -> target 的新 MR，Description 含 run ID |
| `GetMergeRequest` | repo、分支、Version、URL、`CreatedBy` 完整 |
| `UpdateMergeRequest` | 更新后的 Description 回读一致 |
| `ListRepoMergeRequests` | source、target、open 条件只命中新 MR |
| `GetMergeRequestMergeability` | `Mergeable=false`、reason 为 `conflict` |
| Conflict List/Detail | 同一个冲突文件和 `ConflictsId`，Base64 内容可解码 |
| 冲突 parser | 包含三种 marker；能得到准确 start/end line 和 ours/theirs 文本 |
| `GetBlame` | source 文件 1-2 行返回刚创建的 source commit 和 App 作者 |
| 普通评论 | comment/thread ID 非空 |
| 文件行评论 | comment/thread ID 非空，目标为冲突文件第 1 行 |
| `GetThread` | Path 正确，`StartLine=1`、`EndLine=1`，commit IDs 与 MR Version 一致 |

脚本退出码非 0、出现 `failure`，或任何 `ok` 不是 `true`，都算失败。保留输出和
已经创建的资源，不要为了得到绿色结果立刻重跑。

### 5.6 用独立客户端回读 OpenAPI 结果

完整脚本通过后，在本地用它输出的 MR global ID、分支和文件路径做独立回读：

```bash
bytedcli codebase mr get \
  --mr-id <mr-global-id> \
  -R xiahaitao/Conciliator

bytedcli codebase mr status \
  --mr-id <mr-global-id> \
  -R xiahaitao/Conciliator

bytedcli codebase mr comment list \
  --mr-id <mr-global-id> \
  -R xiahaitao/Conciliator

bytedcli codebase repo blame \
  <fixture-path> \
  -R xiahaitao/Conciliator \
  --revision <source-branch> \
  --start-line 1 \
  --end-line 2
```

回读结果应与 TCE 脚本一致。这里主要排除“脚本只检查自己构造的数据，却没有真实
落到 Codebase”的假阳性。

### 5.7 验证真实 Webhook

先从本地验证稳定入口和验签门禁：

```bash
curl -fsS https://aeolus-conciliator.byted.org/healthz
curl -sS \
  -o /tmp/nextcode-unsigned-response.json \
  -w '%{http_code}\n' \
  -X POST \
  -H 'Content-Type: application/json' \
  --data '{}' \
  https://aeolus-conciliator.byted.org/hooks/codebase
sed -n '1,20p' /tmp/nextcode-unsigned-response.json
```

`/healthz` 应为 HTTP 200；无签名 Webhook 应为 HTTP 401，并返回签名校验失败，而不是
旧版本的 503。随后打开 NextCode App 的 Recent Deliveries，按下面顺序核对。

#### 脚本自动产生的事件

完整 OpenAPI 脚本会产生 MR opened/edited、Push、普通评论、文件行评论及伴生 Review
事件。以 run ID、MR ID、comment/thread ID 关联 Delivery，不要只按时间猜。

| 事件 | 预期响应 | 额外检查 |
|---|---|---|
| `merge_request/opened` | HTTP 200、`processed=0` | MR Number、source/target 与脚本输出一致 |
| `merge_request/edited` | HTTP 200、`processed=0` | `Diffs.Description` 是脚本写入的变更 |
| `comment/created` 普通评论 | HTTP 200、`processed=1` | 正文、Comment ID、Thread ID 一致 |
| `comment/created` 文件行评论 | HTTP 200、`processed=1` | Thread ID 一致；`GetThread` 回读文件和第 1 行 |
| `merge_request_review` | HTTP 200、`processed=0` | receiver 明确忽略，不走评论业务链路 |

MR 事件的 `processed=0` 在 pipeline 关闭时是正确结果。`comment/created` 若返回
`processed=0`，则不符合预期。

#### 人类评论、作者回退和评论更新

在新 MR 的冲突文件第 1 行写一条带 run ID 的普通人类评论，例如：

```text
NextCode human inline verification <run-id>: body author thread path line
```

检查对应 `comment/created`：

- HTTP 200、`processed=1`；
- `Comment.Content` 与输入完全一致；
- `Comment.ThreadId` 非空；
- 顶层 `Sender.Email` 是实际操作人；
- `Comment.CreatedBy.Email` 允许为空，receiver 应回退到 `Sender.Email`；
- 用 `GetThread` 回读的 Path、Side、StartLine、EndLine 和 commit IDs 与页面位置一致。

随后编辑同一条评论，只改普通文本，不改成命令。对应 `comment/updated` 必须返回
HTTP 200、`processed=0`，避免编辑评论重复执行。

#### source push 和 target push

完整脚本创建 MR 后，再分别给 source 和 target 分支增加一个不与冲突文件重名的
测试文件。可以在 Codebase 页面切换到对应分支后提交：

```text
docs/nextcode-contract-fixtures/<run-id>-source-push.txt
docs/nextcode-contract-fixtures/<run-id>-target-push.txt
```

source 分支提交后应看到 `push` 和 `merge_request/pushed`；target 分支提交后应看到
目标分支 `push`。这些 Delivery 都应为 HTTP 200、`processed=0`。最后再次调用
`GetMergeRequestMergeability`，原冲突 MR 应继续返回 conflict。

若要覆盖 `merge_request/reopened`，只操作本轮测试 MR：先关闭再重新打开，确认
`reopened` Delivery 返回 HTTP 200、`processed=0`。2026-08-27 的历史结果已覆盖
opened/edited/pushed，尚未保留真实 reopened Delivery；后续需要该 Action 时补验。

### 5.8 验证 Event-ID 去重

从一条首次响应为 `processed=1` 的 `comment/created` 开始：

1. 记录原 `X-Vecode-Event-Id` 和 `X-Vecode-Delivery`；
2. 在 Recent Deliveries 点击 Resend；
3. 确认重投使用相同 Event-ID；Delivery ID 可以变化；
4. 重投应返回 HTTP 200、`processed=0`、`deduplicated=true`；
5. 在 RDS 查询同一 Event-ID，必须只有一条 `event:<event-id>` 记录。

查询 SQL：

```sql
SELECT
  delivery_key,
  event_id,
  delivery_id,
  event_type,
  status,
  processed_count,
  last_error,
  created_at,
  updated_at
FROM nextcode_webhook_deliveries
WHERE event_id = '<event-id>';
```

预期 `delivery_key='event:<event-id>'`、`status='COMPLETED'`、`processed_count=1`、
`last_error` 为空，并且查询只返回一行。这证明 Event-ID 优先于变化后的 Delivery ID。

### 5.9 本地重放和确定性测试

本地或开发机可以用保留脚本重放 fixture：

```bash
uv run python scripts/replay_nextcode_webhook.py \
  --event comment \
  --payload tests/fixtures/nextcode/comment_created.json \
  --delivery-id repeatable-comment-001
```

重复执行同一命令，第二次应为 `deduplicated=true`。脚本从 KMS provider 读取
Webhook Secret，不在命令行或输出中暴露明文。

代码回归：

```bash
uv run pytest tests/unit -q
uv run ruff check src tests scripts
git diff --check
```

至少要覆盖当前已有的 NextCode client、Action wrapper、Webhook contract、Comment
created/updated、Review ignore、MR/Push 路由和 Event-ID 优先去重测试。

### 5.10 失败时怎么处理

| 失败点 | 先做什么 | 不要做什么 |
|---|---|---|
| 两个 Pod 版本不同 | 停止测试，先统一精确 SCM 版本 | 不要只在其中一个 Pod 得出生产结论 |
| health/ready 非 200 | 保存 Pod、端口和响应，检查进程与数据库 | 不要继续创建远程测试资源 |
| KMS 失败 | 检查完整 alias、PSM ACL 和 SDK 错误 | 不要注入明文 secret |
| ZTI 失败 | 检查 TCE 身份、SDK 和 App 绑定 PSM | 不要改用静态 token |
| OpenAPI 失败 | 保存 Action、RequestId、脱敏错误和已创建资源 | 不要用相同 run ID 盲目重试写 Action |
| Webhook 401 | 核对 App ID、raw body 和签名配置 | 不要关闭验签 |
| Webhook 5xx | 保存 Event-ID、Delivery、响应和相关 MR/thread | 不要反复 Resend 制造噪声 |
| Comment created 为 `processed=0` | 检查 Action、CommentableType、Thread 和 receiver 版本 | 不要把 200 当成已处理成功 |
| MR/Push 为 `processed=0` | 在 pipeline 关闭时继续核对 mergeability | 不要把预期的 0 误判成失败 |

### 5.11 每次验证要留下的记录

完成后在本文追加一个新的日期小节，不覆盖旧证据。至少记录：

```text
验证时间：
本地 Git commit：
SCM/TCE version：
两个 Pod 名称与版本：
环境开关：
OpenAPI run ID：
MR URL / Number / global ID：
source / target / fixture：
OpenAPI 逐项结果：
Webhook Event-ID / Delivery / Action / HTTP / processed：
Comment 正文 / 作者 / Thread / Path / Line：
重投去重与数据库查询结果：
未验证、失败或与预期不符的项目：
```

只保留非敏感 ID、分支、commit、路径、状态和脱敏错误。验证结束后应明确写出：
哪些结论来自真实 TCE/Codebase，哪些只来自单元测试，哪些仍未验证。

## 6. 仍未验证或不支持的项目

| 项目 | 状态 | 说明 |
|---|---|---|
| TCE `git clone` | 未验证 | `repo:download` 已有；普通 TCE clone 仍需书面合规确认 |
| 仓库 owner/master | 不支持 | 当前 App OpenAPI 没有 repo member/admin Action |
| 原生冲突状态变化事件 | 不存在 | 当前用 MR/Push 事件后主动查询代替 |
| mergeability 边沿去重 | 未实现 | 若业务只应在状态变化时触发，需要持久化上一次状态 |
| 多 Position 最新版本选择 | 待加固 | 当前取 `Positions[0]`；本次路径/行号正确，但版本变化时有潜在歧义 |
| 完整 `/resolve` 业务链路 | 本轮不验证 | `PIPELINE_ENABLED=false`，符合本轮只验单项契约的范围 |
| 小流量实例权重 | 暂缓 | 两个实例运行同版本时不阻塞本轮契约验收 |
| Argos/TLB Metrics | 暂缓 | 当前以 HTTP 响应、Recent Deliveries 和资源回读作为证据 |

## 7. 本轮完成标准核对

- [x] TCE Pod 内 KMS、动态 ZTI 和 OpenAPI 组合可用
- [x] TCE 生产路径完整读写 OpenAPI 回归通过
- [x] Comment 正文、作者、Thread ID、文件名和行号通过真实事件验证
- [x] `comment/updated` 忽略并返回 200
- [x] `merge_request_review` 忽略并返回 200
- [x] MR opened/edited/pushed 和 target push 可触发冲突复查路径
- [x] Event-ID 优先的真实重投去重通过
- [x] `PIPELINE_ENABLED=false`，没有执行完整业务副作用
- [x] clone、repo admin、原生 conflict-changed 等边界已明确标注

因此，本轮要求的 NextCode OpenAPI 与 Webhook 单项契约验证可以收敛为：
**已通过；保留上述已知边界，不等同于完整解冲突业务链路已上线。**
