# 任务明细页面实现方案

在现有 FastAPI 服务中增加 `/trace` 页面和只读查询 API，将任务状态、外部调用和任务日志写入现有 MySQL。两个 Pod 共用数据库，访问任意 Pod 都能查到同一份任务历史。

首版已完成数据库迁移和 TCE 发布，正式域名上的页面可访问。Chrome 验证发现重置筛选失效，修复已在本地验证，待补发；真实新任务全过程尚未验收。下文保留历史操作记录。

- 分支：`feat/task-trace-plan-refresh`。
- 基于本地分支：`feat/coco-validation-repair`。
- 源码基线：`981d8806fedd74dc6cc37be08113bb04825f117e`。
- 页面入口：[任务明细](https://aeolus-conciliator.byted.org/trace)，已通过 Chrome 验证可访问。
- 本文按 2026-09-11 的代码和本轮补充要求整理，旧的未提交草稿已移除，后续实施以本文为准。

## 实施前提

用户已确认当前没有线上流量、没有正在执行的任务，**本次采用停服改造**。不设计旧版本兼容、灰度、双写过渡或历史回填，也不新增为这些目的服务的开关。部署时停止两个 Pod 的旧服务，完成数据库变更后统一发布新版本，再开放使用。

| 资源 | 本次使用方式 | 证据范围 |
| --- | --- | --- |
| [TCE 服务 210017612](https://cloud.bytedance.net/tce/services/210017612?control_plane=cn&module=instance&x-resource-account=public&x-bc-region-id=bytedance) | 现有两个 Pod，页面随 Python 服务部署 | 本轮 Chrome 已读到两个实例运行中、健康分 100；无流量条件由用户提供 |
| [MySQL aeolus_govern](https://cloud.bytedance.net/rds/detail/db/global/aeolus_govern/overview?x-global-site=cn&x-global-vregion=China-North&x-global-vdc=cn&x-global-detail-tab=China-North&x-resource-account=public&x-bc-region-id=bytedance&x-bc-vregion=ChinaSinf-North&x-bc-vdc=multicloud) | 新表与任务表使用同一个库 | Chrome 已核实 China-North、NDB 8.0、utf8mb4、非分片，写 PSM 为 `toutiao.mysql.aeolus_govern_write`；磁盘总空间显示暂无数据，容量仍待核实 |
| `https://aeolus-conciliator.byted.org` | 复用现有 HTTPS 和 `/` 前缀路由 | 仓库有 Netlink 配置及历史访问记录；证书、P4 类型由用户补充；身份验证尚无证据 |
| Pod 监听端口 | 使用 Pod 自己的 `$PORT0` | 用户说明端口随机；`run/prod_run.sh` 也使用 `PORT0` |

前两次 Chrome 操作报错 `Unable to load browser request-header policy`；后续重试已恢复，TCE 页面可读。该错误表示浏览器工具未加载到请求头策略，仅凭报错无法确认具体根因，不能归因于 Conciliator。WebShell 最初要求 SSO 二次验证，认证完成后已进入 Pod，并完成下述 curl 验证。部署前仍需检查目标库、容量和索引，不删除已有数据。

**只展示新任务，旧历史任务不展示。** “全部历史”指新任务从创建起的所有状态与明细，不包括上线前的任务、快照或事件。旧数据保留在数据库中，不导入、不回填，也不通过页面或 trace API 暴露。

用 `resolve_tasks.trace_schema_version INT NOT NULL DEFAULT 0` 明确界定范围：迁移后已有任务保持 0，新服务创建任务时在创建事务中显式写 1。页面及全部 trace API 只查询该值大于等于 1 的任务，不依赖发布时间、Pod 时钟、最大任务 ID 或“是否已有事件”猜测新旧。该标记创建后保持不变，旧任务后续收到操作也不变成新任务。无流量条件下由一次停服迁移完成，不需要兼容过渡。

旧任务不绑定新的 trace 明细；采集器在确定归属后跳过它们。任务确定前的临时记录只有关联到新任务后才进入可查询范围。直接请求旧 task_id、旧任务的 event_id 或正文接口返回 404，不能只在前端隐藏。页面空状态提示“暂无新任务”，并注明只收录本次采集启用后新建的任务。

## 现有域名配置与验证方法

仓库已记录配置过程和验证结果：

- [2026-08-26 Webhook 方案](2026-08-26-nextcode-openapi-webhook-contract-validation.md) 的“4.1 稳定 HTTPS 入口验证”记录通过 Netlink 创建入口，TLB 集群为 `internal_api_lf_lb_v3`，路由为 `/` 前缀，后端通过 Consul 发现 `data.aeolus.conciliator/default`。当时开发机 DNS 解析成功、TLS 校验通过，`GET /healthz` 返回 HTTP 200 和 `{"status":"ok","app":"conciliator"}`。
- [2026-08-27 修订方案](2026-08-27-nextcode-openapi-webhook-contract-validation-revised.md) 的“5.7 验证真实 Webhook”保留了 `curl -fsS https://aeolus-conciliator.byted.org/healthz` 命令。

用户本轮补充的配置如下，原文档没有完整记录证书类型和 P4 名称，不能将这些细节写成旧文档已有内容：

| 配置项 | 值 |
| --- | --- |
| 域名 | `aeolus-conciliator.byted.org` |
| HTTPS | 开启 |
| 证书 | `*.byted.org` RSA 证书 |
| 路由类型 | 普通前缀匹配（通用匹配 P4） |
| 路径 | `/` |
| 后端 | 历史文档记录为 `data.aeolus.conciliator` / `default` |

按这个 `/` 前缀规则，`/trace`、`/trace/assets/*` 和 `/api/trace/*` 应复用已有转发，无需为页面另注册域名或单独增加路径规则。实施时检查是否存在更高优先级规则或路径重写，再做访问验收。HTTPS 证书与路径路由不能证明已经配置了用户登录鉴权。

验证时先从能使用内网 DNS 的环境访问稳定域名，再在所选 Pod 内检查监听端口：

```bash
# 内网客户端：验证 DNS、TLS 和入口转发，不跳过证书校验
curl -sS --connect-timeout 5 --max-time 15 -w '\nHTTP %{http_code}\n' \
  https://aeolus-conciliator.byted.org/healthz

# Pod WebShell：使用该 Pod 的运行时端口
curl -fsS "http://127.0.0.1:${PORT0}/healthz"
curl -fsS "http://127.0.0.1:${PORT0}/readyz"
```

2026-09-11 本轮实测：Chrome 访问域名返回 `net::ERR_NAME_NOT_RESOLVED`，本机 curl 返回 `Could not resolve host`，尚未到达 HTTP 服务。开发机 SSH 返回 GSSAPI 认证失败，未能从开发机重验。

随后通过 Chrome WebShell 进入 `fed-dp-6b9e96188f-68c6cf446b-b6rpl`，使用其 `$PORT0` 执行只读 curl，结果如下。该次运行时端口为 10330，仅作为验证快照，后续仍使用 `$PORT0`。

| 请求 | 实测结果 |
| --- | --- |
| `http://127.0.0.1:${PORT0}/healthz` | HTTP 200，`{"status":"ok","app":"conciliator"}` |
| `http://127.0.0.1:${PORT0}/readyz` | HTTP 200，`{"status":"ok","db":"ok"}` |
| `https://aeolus-conciliator.byted.org/healthz` | HTTP 200，`{"status":"ok","app":"conciliator"}`；未使用 `-k` |

这次验证确认所选 Pod 服务与数据库就绪检查正常，且从该 Pod 访问稳定域名的 DNS、TLS 和 HTTP 链路可用。域名请求可能由负载均衡转发到任意后端，不能据此认定两个 Pod 都已逐一验证。此前本机 DNS 失败仍未解决，不能将 Pod 内成功等同于当前用户 Chrome 已可直接访问域名。

### 本机 Chrome 的 DNS 排查

后续已读取用户提供的 Netlink 文档索引、域名控制台以及本机飞连状态，定位到解析路径与区域差异：

| 检查项 | 2026-09-11 实测 |
| --- | --- |
| 飞连 | VPN 开启，极速模式，服务器 `JP-2`，显示 DNS `10.103.255.12` |
| macOS 解析配置 | 存在高优先级的 `127.0.0.1` resolver；Wi-Fi 分配的 DNS 为 `10.88.128.1` |
| 查询 `127.0.0.1` | 域名 A、AAAA 均返回 NXDOMAIN |
| 查询 `10.88.128.1` | 返回 `inner-rd.internal-api-lf-lb-v3.hl.byted.org`，A 为 `10.8.6.73`，AAAA 为 `fdbd:dc02:fe:2013::1` |
| 本机临时指定已解析地址 curl | HTTPS `/healthz` 返回 200，`ssl_verify_result=0`；未修改 hosts、系统 DNS 或 Netlink |
| Netlink CN 解析 | `inner_v2`，存在 `*`、hl、lq 线路，CNAME 指向 HL / LQ TLB，TTL 1 分钟 |
| Netlink 海外解析 | `inner_i18n` 区域暂无数据 |
| TLB 接入类型 | `inner_rd`，当前集群标明“国内内网 api 普通集群（仅研发访问）” |

上述证据表明当前本机查询走了与 CN 办公网不同的解析路径；飞连 JP 节点与域名缺少海外解析相吻合。不能把它笼统归为 Chrome 缓存问题，也无需为此修改 `/` 路由、证书或 Pod 监听端口。

用户随后自行切换到 CN 区域，提供普通 curl `/healthz` 返回 HTTP 200、响应 `{"status":"ok","app":"conciliator"}` 以及 Chrome 成功打开的截图，再切回 JP。后续目标是保持 JP 网络也能正常访问，不再由代理修改飞连设置。

按用户授权，已通过 Chrome 在 Netlink 新增海外 `inner_i18n` 默认 CNAME 草稿，并于 2026-09-11 16:28:11 提交[工单 651301](https://cloud.bytedance.net/netlink/v2/main/business/ticket/651301?ti_business_id=9357128&x-resource-account=public&x-bc-region-id=bytedance)：

- 线路：`default`；类型：`CNAME`；TTL：60 秒。
- 目标：`inner-rd.internal-api-lf-lb-v3.hl.byted.org`，通过表单选择现有 `internal_api_lf_lb_v3 / HL` TLB 入口生成。
- 提交后的工单差异只包含这条新增记录；CN 解析、TLB 路由与 HTTPS 配置未变。
- 提交时状态为“业务线 reviewer 审核进行中”，审核人为 `xiahaitao`，“域名解析”阶段等待中，尚未生效。

变更前已确认 JP DNS `10.103.255.12` 能解析上述目标 CNAME 为 `10.8.6.73`，也能解析 LQ 入口。工单审核并发布后，应在 JP 网络下用默认解析的普通 curl 和 Chrome 重新访问 `/healthz`；只有成功后才能确认该网络的访问问题解决。其他海外网络需分别验证，不把临时 `--resolve`、固定 hosts 或关闭证书校验当成交付方案。

同事访问条件：

- 目标为国内研发同事时，继续使用现有域名和 `/` 前缀；访问者需处于具备研发访问权限的办公网络或合适的 CN VPN。页面自身仍需实现应用访问控制。
- 若同事需要保持海外网络接入，需按实际区域设计受支持的海外入口与访问路径。CN 和海外解析分开管理，不能直接复制 CN 私网地址就承诺海外可达；应先核实跨区访问规则，再提对应解析 / 接入变更。
- 若包含非研发序列同事，现有 `.byted.org` 不能默认覆盖全员。可按官方流程申请研发网络权限，或评估 `.bytedance.net` 的内网全员访问入口；这属于访问范围调整，另行确定，不在 DNS 排查中自动扩大权限。

依据：[DNS 解析管理](https://cloud.bytedance.net/api/v1/cloud_developer/docs/netlink/cn/66fa47556d8a0302fcba690f.md) 明确区分 CN 与海外解析；[TLB 集群说明](https://cloud.bytedance.net/api/v1/cloud_developer/docs/netlink/cn/67f8da84262d810528df74af.md) 区分研发、全员与外网入口；[非技术序列访问说明](https://cloud.bytedance.net/api/v1/cloud_developer/docs/netlink/cn/67f8e10f477ccf0535eb5293.md) 提供权限申请指引；[域名故障排查](https://cloud.bytedance.net/api/v1/cloud_developer/docs/netlink/cn/68b9600c6888fd04f92c1931.md) 提供 dig、curl 及连通性诊断方法。

页面上线验收增加一项：当前用户和一名目标同事使用各自 Chrome，通过正常域名解析访问 `/healthz` 和 `/trace`。使用临时 IP 绑定成功只能用于定位问题，不算浏览器访问验收通过。

## 最终实现范围

用户在实施时将 Coco 输入输出改为 Session 链接。首版不提供源码或 diff 视图，也不将 Coco prompt、AgentMessage 或结果正文复制到 trace 表；Coco 的业务处理和原有业务结果存储保持原样。页面链接为：

```text
https://code.byted.org/managed-agents/conciliator-service/session/{session_id}
```

每条 Coco 调用保存当时获得的 Session / Run 等标识。非 Coco 记录可导航到任务当前 Session，页面明确标为“当前 Session”，不把它伪装成历史时刻的 Session。

Netlink 工单完成后，用户确认保持 JP 节点时普通 curl 和 Chrome 均可访问 `/healthz`。本轮重新执行普通 curl 返回 HTTP 200 和 `{"status":"ok","app":"conciliator"}`；Chrome 自动化两次返回 `ERR_BLOCKED_BY_CLIENT`，未据此推翻用户的手动浏览器验证。这里的客户端拦截与此前 NXDOMAIN 不同。

## 数据与采集

新增 migration `0007_task_trace`，为任务表增加 `original_mr_url` 和 `trace_schema_version`，并建立三张表：

| 表 | 内容 |
| --- | --- |
| `task_trace_events` | 追加式明细、状态、调用标识、结果、摘要、正文采集状态 |
| `task_trace_event_bodies` | 脱敏正文，UTF-8 每块最多 256 KiB，通过事件、正文类型和块序号定位 |
| `task_trace_links` | 关联创建任务之前的请求与最终确定的新任务 |

三张表均使用自增 BIGINT 主键；正文块和关联表另有组合唯一约束。DDL 保留时间、任务、调用和 trace_id 索引。任务创建时显式写入采集版本 1，已有任务保持默认值 0，全部 trace API 都排除版本 0；不会导入旧任务。

Repository 的初始状态、ORM 状态变更和成功 CAS 状态变更都在业务事务内写入历史，失败一同回滚。`TASK_CREATED` 也已移入创建事务；其他现有业务事件通过事务内映射写入明细，业务事件表仍负责原来的恢复逻辑。相同状态的字段更新不产生状态迁移，CAS 失败不写成功迁移。

NextCode、Coco、Lark 在每次实际 HTTP 请求处记录请求和响应，网络异常另有一行，沿用原来的重试策略。请求和响应通过 `call_id` 配对。NextCode 的逻辑调用及 Lark `_request` 重试共用 `operation_id`，每次尝试有独立编号。Coco SSE 记录连接、事件标识和结束，不持久化流正文。

NextCode 的文件、冲突内容和提交代码字段在采集副本中省略。Lark 认证正文不采集 token；Markdown 上传保存文件名、类型和去掉围栏代码后的文本，不读取文件流。结构化源码、diff 字段和围栏代码不进入展示副本；这不是任意自由文本源码识别器，业务日志仍不应直接打印源码。凭证脱敏覆盖嵌套 JSON、字符串内 JSON、URL 参数、Authorization、token、secret、签名、JWT 和数据库密码。

HTTP 和日志使用非阻塞有界队列，最多 1000 条、正文总计 64 MiB；单条普通正文最多 32 MiB，单条日志最多 256 KiB。正文超过上限明确标为截断。存储失败最多重试相同 `event_uid` 一次，不能触发外部业务调用重发。正常关停限时 flush。队列丢弃、写库失败和各进程心跳持久化后由健康接口汇总；强杀或数据库中断可能造成缺失，页面会显示已知异常及失联状态。

请求记录表示已进入一次调用尝试，不保证对方收到或执行。外部返回与 MySQL 无法原子提交，不能根据没有响应行推断“对方没有执行”。首版不采样、不自动清理历史。

## 页面与 API

FastAPI 同时提供 HTML、CSS、JavaScript 和只读 API，静态资源随 Python wheel 部署，不新增 Node.js 运行时。

- 默认“全部明细”，一行一个事件；“任务列表”一行一个新任务，即使还没有明细也能查询。
- 支持任务 ID、当前状态、记录时状态、仓库、原 / 新 MR、原 MR URL、时间、外部系统、事件类型、操作、调用结果和调用 ID。
- 当前状态来自任务当前快照；记录时状态来自采集时观察值。MR 列标明链接来自当前快照。
- 点击任务 ID 查询时间线；`/trace?task_id=123` 可直达。筛选条件写入 URL。
- 默认每页 50 条，上限 200 条；时间和 ID 联合游标分页。列表不查询大正文，也不全库 count 或使用无限 OFFSET。
- 正文按需加载，支持分段和已加载文本查找。用户可以选择每 15 秒刷新首页；查看历史页时不自动换页。刷新可重新查询迟到记录。
- 所有 ID 在 JSON 中使用字符串，时间返回 UTC，页面按 Asia/Shanghai 显示。

| 路由 | 内容 |
| --- | --- |
| `GET /trace` | 页面 |
| `GET /trace/assets/trace.js`、`trace.css` | 白名单静态资源 |
| `GET /api/trace/options` | 状态与事件类型枚举 |
| `GET /api/trace/tasks` | 新任务分页 |
| `GET /api/trace/tasks/{task_id}` | 当前快照与链接 |
| `GET /api/trace/events` | 跨任务或单任务事件分页 |
| `GET /api/trace/events/{event_id}` | 事件信息与正文目录 |
| `GET /api/trace/events/{event_id}/body?kind=response&chunk=0` | 正文块 |
| `GET /api/trace/health` | 各采集进程心跳和丢弃情况 |

页面从 MySQL 读取不需要 SSO 协议。首版沿用现有研发内网入口；另外提供可选的 `TRACE_VIEWERS` 只读 HTTP Basic 账号映射，凭证通过运行时配置注入，未配置时依赖内网入口的访问范围，不具备个人身份鉴别。没有实现或假定公司 SSO，也不信任任意客户端提交的身份头。不能把默认模式迁移到公网入口。

页面与 API 同源、禁止缓存，并设置 CSP。正文用 `textContent` / textarea 渲染，不执行 HTML。MR 和 Session 链接只允许 HTTPS 的 `code.byted.org`，新窗口设置 `noopener noreferrer`。

## 验证与部署

本地验证已覆盖新旧任务隔离、事务回滚、同状态更新、CAS 失败、ORM cancel、业务事件映射、创建前关联、历史状态筛选、时间及游标参数、大正文 UTF-8 分块、查询不读正文、可选鉴权、外部调用配对、重试编号、超时、并发 contextvars、队列满、幂等写入、脱敏和 Coco 正文不落 trace 表。

本地 Chrome 使用独立 SQLite 演示库验证任务筛选、正文展开与 Session 链接。演示任务不来自线上，也未调用真实 MR、Coco 或飞书业务接口。

最终本地检查：291 个单元测试通过；`src`、`tests`、`scripts` 和新增 migration 的 Ruff 检查通过；mypy 检查 64 个源码文件通过；JavaScript 语法检查通过。wheel 构建成功，其中 HTML、CSS、JavaScript 与工作区文件逐字节一致。测试有一条 Starlette TestClient 的依赖弃用警告，不影响结果。

MySQL 集成测试、目标库容量与执行计划、两个 Pod 一致性和真实任务全过程仍是发布验收项。本机没有 Docker，不能把 SQLite 测试称为 MySQL 验证。

发布步骤：

1. 按无流量约定停止旧服务，检查目标库的 `alembic_version`。从 `0006_processing_version` 升级时只执行 `ddl/0007_task_trace.sql`，不要在已有库执行 `ddl/all.sql`，不要重复执行非幂等 DDL。
2. 发布包含页面资源的新版本到两个 TCE Pod。服务启动不会自动运行迁移；先迁移再启动新版本。
3. 在各 Pod WebShell 使用 `http://127.0.0.1:${PORT0}/healthz` 和 `/readyz` 检查，不硬编码端口。验证 `/trace`、静态资源和 `/api/trace/health`。
4. 用普通 Chrome 访问稳定域名 `/trace`，在专用 MR 创建新任务，核对状态、NextCode / Lark 请求响应、Session 链接、MR 链接和日志。验证旧任务查询返回 404。
5. 从两个 Pod 查询同一新任务，并在采集结束后从头分页核对完整记录。验证重解、更换 Session、审批、取消和通知；不能把本地演示或 `/healthz` 的 HTTP 200 当成上述业务验收。

## RDS 迁移执行记录（2026-09-11）

通过用户 Chrome 在 `aeolus_govern` 的 China-North / cn 入口提单。迁移前查询版本为 `0006_processing_version`，共有 19 个任务，其中 #18 为 `AWAITING_APPROVAL`；此次不修改这些任务。该结果更新了先前“没有未结束任务”的前提。

平台将 DDL 与 DML 分开受理，因此合并为两张工单：

- [DDL 113195048](https://cloud.bytedance.net/rds/worksheet-bpm/detail/cn/113195048)：两次加列合并为一次 `ALTER TABLE ... ADD COLUMN ..., ADD COLUMN ...,ALGORITHM=INSTANT`，与三条建表 SQL 一起提交。禁止 KILL 长事务，审批后自动执行。SQL 校验已通过，等待李文帅完成 DB Owner/协作者审批。
- [DML 113195366](https://cloud.bytedance.net/rds/worksheet-bpm/detail/cn/aeolus_govern/113195366)：只登记迁移版本，设置为手动执行。SQL 校验通过，计划使用唯一索引扫描并更新 1 行；当前等待夏海涛审批。必须等四条 DDL 全部成功并核对表、列和索引后才能执行。

```sql
UPDATE alembic_version
SET version_num = '0007_task_trace'
WHERE version_num = '0006_processing_version';
```

DDL 初次平台校验要求 TEXT 字段允许 NULL，在原工单中将 `task_trace_event_bodies.content` 改为可空后通过校验，没有新增 DDL 工单。Migration、ORM 和导出 DDL 已同步；采集器仍显式写入正文，查询接口将异常空值返回为空字符串。修正后的 24 个 trace / DDL 导出测试、Ruff 和 mypy 检查通过。

上述审批状态是提单时的记录。后续发布验证已查询到 `0007_task_trace`，三个新表及两个新字段均可查询；19 个旧任务的 `trace_schema_version` 都为 0。迁移已落库，不要重跑迁移文件。

## 首次发布验证（2026-09-11）

[TCE 工单 390381708](https://cloud.bytedance.net/tce/deployment_new/390381708?service_id=210017612) 于 18:14:38（UTC+8）完成，耗时 9 分 11 秒。SCM 版本 `1.0.0.27` 对应提交 `3ae0051251eef1600d7908d16abc6abe65045ecf`。检查时人工确认节点已显示完成，发布已进入生产实例升级，没有重复点击确认。

Chrome 通过正常域名打开 `/trace`，页面、查询选项和样式加载成功，采集状态显示 8 个进程在线。查询旧任务 #18 返回“任务未收录”。当前没有新任务，默认明细和任务列表为空，未创建真实业务任务来验收外部调用全过程。

操作检查发现“重置”按钮的 `id="reset"` 遮蔽了 `form.reset()`。修复改为显式调用 `HTMLFormElement.prototype.reset.call(form)`，同时处理浏览器历史导航中的重置路径。本地 Chrome 演示库验证：筛选 #102 得到 3 条记录，点击重置后清空任务 ID、移除 URL 中的 task_id，并恢复全部 9 条记录。JavaScript 语法检查通过。该修复不在 `1.0.0.27` 中，需要补发后生效。
