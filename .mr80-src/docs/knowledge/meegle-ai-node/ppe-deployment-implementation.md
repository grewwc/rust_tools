# Meego 一键部署 PPE：实现与运行约定

当前实现支持固定 Coding SHA 的 BITS → SCM → ICM 构建、任务专属 Aeolus + VQS 完整 PPE 环境、同任务环境复用升级、重试、恢复扫描及释放。Aeolus 双集群平台契约见 [PPE 平台结论](ppe-environment-conclusions.md)，VQS 配置和路由边界见 [VQS 运行约定](vqs-ppe.md)，产品验证步骤见 [端到端验收](ppe-product-e2e-validation.md)。

## 运行配置

- PPE 和 Devbox 均为正式服务必需能力，启动时始终初始化，无逐功能开关。
- 目标库先应用 `0019_ppe_deployments`，增量 SQL 为 `ddl/0019_ppe_deployments.sql`；服务启动不执行迁移。
- 平台调用复用 ByteCloud 应用 JWT：AK/SK 在 staging/prod 从同一个 KMS alias `conciliator/credentials/bytecloud_credentials` 获取 JSON 对，本地 dev 使用 `BYTECLOUD_AK`、`BYTECLOUD_SK`。整对轮转，更新后重启服务。BITS `Username`、ICM `username` 使用服务端固定且经授权的 `huaguojin`；前端不能指定操作人。后端不依赖 bytedcli 或个人 CLI 登录缓存。
- 模板为 `src/conciliator/integrations/ppe/aeolus_cn.json`：代码仓 `dp/aeolus`、主制品仓 `data/dp/aeolus_ppe`、PSM `data.aeolus.api_v3`、CN / `online_cn` / LF / `PPE/ipv6`。
- Python 2.7 使用独立的 `aeolus_py27_cn.json`、主 SCM `4863` 和 SCM-only 流水线 `1254638941442`；配方、来源和验收见 [Python 2.7 PPE](ppe-py27.md)。PPE 与 Devbox 对齐为代码证据推荐及手动选择，分支只作提示；部署和重试冻结原运行时，见[运行时选择](runtime-selection.md)。
- `online` 与 `query_online_mirror` 各 8 CPU / 32 GiB / 1 实例。固定全部 11 项 SCM 依赖、安装路径和 Python 3 镜像配方，仅替换本轮 SHA 对应的主仓成功制品。ICM 显式使用 `pip3`。
- 环境名为 `ppe_ai_ae_<环境 UUID 前 16 位>`。同任务复用；释放后重建使用新 UUID、环境名及 generation。参考集群仅用于读取配置，不作为更新或删除目标。
- VQS 代码仓 `dp/VizQueryService` 使用 `vqs_cn.json`、SCM `53349` 和流水线 `1251819549442`；VQS 的 LF / IPv6 `default` 为 16 CPU / 32 GiB / 1 实例，任务环境前缀为 `ppe_ai_vqs_`。
- 两种代码仓均在同一个 ENV 部署 `data.aeolus.api_v3` 的两个集群和 `data.vqs.api` 的一个集群，共 32 CPU / 96 GiB / 3 实例。应用身份必须同时拥有两侧 PPE 部署权限；平台按实际 PSM 与身份判断配额和审批。
- VQS 创建及补缺时强制覆盖 `new_env_vars.AEOLUS_SERVICE_DISCOVERY="0"`，其余来源变量保留且仅驻内存。这样 VQS 使用 Aeolus 域名入口，保留 PPE 请求头，经入口代理进行同泳道调度；直接服务发现不能作为已命中 PPE 的证据。
- 不构建业务前端；不自动续期或启用热部署。

## 构建与部署

创建 API 返回 202，数据库记录驱动后台执行：

1. 校验最新成功 Review 轮次、完整 SHA、模板及来源配置。
2. 查询精确 SHA 的成功 SCM 制品；没有可复用制品时，预检并触发固定 BITS SCM-only 流水线，保存原 run ID。
3. 等待原运行，独立核验 SCM version ID、SHA、`test` 类型、`x86_64` 架构及 `build_ok`。CN PPE 不消费仅供 BOE 的 `offline` 制品；精确 SHA 没有 test 制品时触发固定 BITS 流水线。
4. 提交 ICM 并保存 version ID，按精确 ID 核验 LF 镜像状态与完整配方。
5. 另一侧直接复用固定的生产成品镜像，不触发 BITS/ICM 构建。部署前独立核验 SCM version ID、SHA、`online` 类型，以及 ICM 精确 version ID、LF / x86_64、成功状态、内容摘要、启动路径和镜像 manifest。基准不可用则阻止部署，不自动切换 latest；当前 Coding 制品只接受 `test`。
6. ENV 预检后一次提交两侧服务的创建动作。已有 ENV 分服务判断创建、补缺或升级；镜像和依赖未变的配套服务保持原状。补缺工单完成后再次检查，必要时升级原有集群。
7. 等待原 ENV/TCE 工单，核验三个集群的身份、实际镜像、依赖、资源规格和对应服务的 Pod healthz，并从 VQS 的 TCE `runtime.env_list` 核验 `AEOLUS_SERVICE_DISCOVERY=0`。所有条件同时满足才允许人工确认。

SCM、镜像、工单、实例和人工业务验证分别有独立证据。ENV `Succeed` 不等于实例就绪；健康接口通过不等于浏览器业务请求命中本轮代码。

### 更新生产基准

[production_baselines.json](../../../src/conciliator/integrations/ppe/production_baselines.json) 集中保存两侧生产基准，使用生产常规集群实际部署的镜像：

| 服务 | 生产来源集群 | 固定 ICM 镜像 / version ID | 主 SCM |
| --- | --- | --- | --- |
| Aeolus | `online`、`query_online_mirror` | `217258` / `1.0.2.107` / `176877628` | `4863` / `1.0.11.8884` / `172012735` |
| VQS | `default` | `191197` / `1.0.0.2118` / `176877640` | `53349` / `1.0.0.6079` / `172012752` |

生产镜像、当前 Coding 构建配方分别配置。Aeolus 生产基准是 Atum 成品，主仓为 `data/dp/aeolus`，启动 `/opt/tiger/toutiao/app/data/aeolus/bootstrap.sh`；用于替换的 `scm_repos` 为空，但镜像 `extras.Scm` 包含完整内嵌依赖。按已固定 ID 和摘要读取该清单，制品快照保存名称、版本、绝对安装路径，并与 ENV 的实际 `repo_info` 完整比较；非主仓在 Atum 清单中的 `remote_id` 为 0，因此按唯一仓库名称比对。配套部署关闭 `scm_replace`，不会把 `aeolus_ppe` 的主仓、旧依赖或启动脚本覆盖进生产镜像。改动侧仍使用自己的 SCM-only 流水线与 PPE 配方。

按需更新时，先从生产 TCE 常规集群读取实际 `runtime.image_version`，再按该版本核对 ICM 的精确 ID 与 manifest，并同步更新 JSON 中的镜像摘要、SCM 身份、启动路径、解析后的基础镜像和 `scm_repos`；这些字段是同一镜像的完整快照，不能只改版本号。文件不保存环境变量或凭据。生产后来发布新版本不会自动改变基准。

基准变更随配置重新加载生效，并改变使用该配套基准的模板 revision；已有部署继续读取原快照。同任务旧环境需先释放，再用新模板重新部署。

## 持久化与恢复

`aeolus_ppe_environments` 保存资源身份、generation 和当前部署指针；`aeolus_ppe_deployments` 保存每次请求的轮次、SHA、模板、外部 ID、意图、租约、阶段和通知投递状态。PPE 不使用虚构 Devbox worker。

- `(task_id, request_id)` 去重；同 ID 不同参数返回冲突。task 内 sequence 决定记录顺序。
- 同环境的远端变更串行。数据库租约不能撤销已提交工单，必须等待前一工单结束或完成对账后再更新。
- 外部写请求前持久化意图，取得外部身份后清除；响应丢失或写期间进程退出进入“结果待确认”，不盲目重放。晚到响应仍可保存原外部 ID。
- 重启只续查原构建/工单 ID。等待超时后的 BITS 重试继续查询尚未终结的原 run，明确失败后才允许显式重试。
- 同一部署请求对每个服务最多提交一次升级。原工单完成后版本仍不匹配则明确失败并保留原工单，不循环提交相同镜像。旧 schema 3 制品缺少 Atum 内嵌清单时，只补读原固定镜像并校验身份，不改变模板 revision 或要求释放环境。
- BITS 构建及旧模板制品等待限 30 分钟；平台工单等待超过 2 小时进入待确认；实例核验超过 20 分钟失败并保留资源和产物。具体值以当前源码为准。
- 已就绪环境继续定期复查；健康或一致性检查不通过时显示“暂未就绪”并暂停人工验证，保持轮询，恢复后自动清除提示。查询异常同样暂停验证；不累计失败次数，也不因一次健康检查失败要求重新部署。
- 同任务、SHA、模板一致时可复用成功制品；部署前仍按 ID 和配方复核镜像。
- 同任务、完整模板版本一致时，下一轮复用配套服务的生产基准镜像，并按固定 ID 和内容摘要复核。历史 schema 2 记录已受理的配套构建仍只查询原 ID，使用原写前意图和超时机制。
- 新部署冻结包含两服务及生产基准的 schema 3 模板；旧记录继续按原模板恢复。模板版本变化时须释放旧环境再创建，不能直接混用旧环境。VQS 变量缺失或漂移会阻止就绪；当前仅支持释放后重建，不猜测 ENV 的变量升级字段。
- 新反馈使旧部署失去验证资格，保留旧环境供下一轮复用。旧轮次迟到结果不能推进新部署。
- 更新失败保留旧版本提示、访问方式及实际资源状态；部分集群成功不使整体取得验证资格。
- 平台响应仅保存白名单字段。来源敏感变量驻内存；Pod 探测使用不带平台 JWT 的独立 HTTP 客户端。

`meta.status=need_cluster` 不单独判失败，`podNotReady=None` 不单独判未就绪。资源是否已删除需接受 ENV 的正数 `deleted_at` 墓碑，并继续核验已记录的 TCE 服务和集群不存在。

## API、验证与释放

所有 PPE API 使用绑定任务的 `X-Aeolus-Task-Token`，不接受客户端指定任意仓库、PSM、环境名或操作人。

```text
POST /api/aeolus/tasks/{task_id}/ppe-deployments
GET  /api/aeolus/tasks/{task_id}/ppe-deployments/{deployment_id}
POST /api/aeolus/tasks/{task_id}/ppe-deployments/{deployment_id}/retry
POST /api/aeolus/tasks/{task_id}/ppe-deployments/{deployment_id}/release
```

创建参数为 `request_id, round_no, commit_sha, profile_revision, runtime_profile_id`；
服务端验证候选及配置版本。旧客户端省略候选 ID 时，以已提交的精确模板 revision 识别选择。
释放校验 `environment_id, generation` 及当前 deployment ID。

完成确认只提供“验证通过”和“跳过验证”。验证通过通过结构化 `verification_targets` 绑定快照中全部可用环境的目标类型与部署 ID；后端拒绝遗漏、重复或已变化的集合。registry 在同一事务中检查每个目标、记录人工验证并请求释放，任一目标失效则整次回滚。PPE 要求最新轮次、当前资源身份和 30 秒内的就绪观察；单目标兼容参数以当前 API 定义为准。

节点完成与资源释放分别记录。释放等待在途 ENV 工单结束后才删除；删除失败保留身份并显示清理失败，需显式重试。台账累计保存两侧曾观察到的服务和集群身份，即使平台后续只返回部分资源也不丢弃旧 ID。只有 ENV 及全部已记录 TCE 服务/集群均确认不存在，才标记已释放。SCM/ICM 制品保留供诊断。

“结果待确认”由维护者依据原 SCM、ICM、ENV/TCE ID 和资源归属对账；当前没有强制重放或跳过隔离的接口。

## 卡片与通知

Web 和飞书展示当前阶段、版本、访问入口及可追溯构建信息，重试、部署和释放以服务端资格为准。PPE 不依赖 Devbox 列表，节点完成后仍轮询资源释放进度。

“查看 PPE 环境”在后端确认远端环境 ID 后显示；首次构建及环境尚未创建时隐藏，构建失败通知同样不提供空入口。管理页可用于查看部署进度，业务访问提示和 `X-TT-ENV` 则等首次集群核验就绪后展示。已有环境更新时保留管理入口、访问方式和上一版提示；请求释放后隐藏入口及访问提示。展示资格分别由快照的 `can_view_environment` 和 `can_access_environment` 给出，不能代替人工验证资格 `can_verify`。

当前轮次成功且仍为当前环境的 PPE 不显示普通重复部署按钮；新轮次、已释放后的部署和失败重试保留。更新期间醒目显示上一版镜像、短 SHA 和 `X-TT-ENV`，直至新部署成功。

部署通知只发送简短的成功/失败结果，保留“结果待确认”和“释放失败”等需人工干预提醒。稳定 UUID 去重，发送失败保留待投递标记。布局与输入限制见 [卡片布局约定](card-layout-verification.md)。
