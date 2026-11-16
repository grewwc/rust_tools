# VQS PPE 运行约定

VQS 使用线上普通 Python 3 构建。代码仓为 `dp/VizQueryService`（Codebase `69970`），SCM 主仓为 `data/dp/vqs`（`53349`），PSM 为 `data.vqs.api`。不使用 `vqs_atum_py311`、Python 3.11 升级分支或 Aeolus 镜像配方。

## 固定模板

[vqs_cn.json](../../../src/conciliator/integrations/ppe/vqs_cn.json) 冻结以下配置：

- 镜像配方来自线上 ICM 镜像 `191197` / `1.0.0.2118`，基础镜像为 `toutiao.debian.python3:latest`；启动脚本为 `toutiao/app/data/vqs/run/prod_run.sh`，根目录为 `/opt/tiger`，服务端口为 `12355`。
- 保留线上完整的 18 项 SCM 依赖及路径、系统包和两组 `pip` 安装项。每轮仅替换主仓成功制品版本，不从 ENV 默认依赖或 ICM latest 推断配方。
- 集群配置参考 [ppe_liwenshuai_test](https://bits.bytedance.net/env/life/ppe/online_cn/detail/ppe_liwenshuai_test/service) 的 `default`（`217131828`）：LF、16 CPU / 32 GiB / 1 实例，保留网络和 Sidecar 开关。
- 目标资源池为 `China-North-LF/PPE/ipv6`。参考环境使用 `PPE/amd`；本次 amd 内存配额预检被拦截，ipv6 预检和实际部署通过。来源集群 ID 只用于读取配置，不能用于升级或释放。
- 来源环境变量仅驻内存，创建和补缺时显式覆盖 `AEOLUS_SERVICE_DISCOVERY=0`。已核验 ENV suggest 的 `new_env_vars` 返回字典，TCE 的 `runtime.env_list` 返回 `{env_name, env_value}` 列表；就绪检查使用后者确认实际配置，敏感值不写入台账。
- 平台 health check 之外，独立直连 `/healthz` 必须返回 HTTP 200 和 JSON `code=aeolus/ok`。`/version` 可核验 `git_revision`、`scm_version`、PSM 和集群。

## 流水线与任务路由

[meego_vqs_scm_build](https://bits.bytedance.net/devops/284273157378/pipeline/detail/1251819549442) 为 SCM-only 流水线，仅包含一个 `job_atoms/scm_compile_beta`，构建 `data/dp/vqs`，`pub_base=commit_base`、`revision={{custom.commit_hash}}`、类型为 `test`；不包含部署步骤或自动触发器。

后端按任务评估中的 `code_context.repository` 选择主构建仓；客户端不能指定任意 PSM、操作人或流水线。两种任务均在一个 ENV 内部署 Aeolus 双集群 + VQS `default`。改动侧使用本轮 SHA 构建，配套侧直接复用 `production_baselines.json` 中固定的生产成品镜像，按精确 ID、摘要、SCM 身份及 manifest 核验，不重建配套镜像。VQS 任务环境名仍为 `ppe_ai_vqs_<UUID 前 16 位>`；Aeolus 任务使用 `ppe_ai_ae_`。基准来源和按需更新方法见 [PPE 实现](ppe-deployment-implementation.md#更新生产基准)。

模板、配套基线和 revision 随部署记录冻结，恢复使用原记录。升级仍将镜像放在服务 meta，并传本任务的数字 cluster ID；不更新未变化的配套侧。幂等、恢复、串行变更和释放规则沿用 [PPE 实现](ppe-deployment-implementation.md)。

## VQS → Aeolus 路由

参考泳道 `ppe_liwenshuai_test` 的 VQS 关闭直接服务发现，使用 `https://data.bytedance.net/aeolus/api/v3/queryEngine/query` 域名入口。请求从 VQS 再经过 `inf.env.proxy`，根据保留的 `X-TT-ENV` 调度到同泳道 Aeolus。TCC 的基准值为 `AEOLUS_SERVICE_DISCOVERY=1`，必须由 VQS 环境变量 `0` 覆盖。此前 `ppe_aeolus_huaguojin_test` 虽包含 VQS 与 Aeolus，却缺少该覆盖，实际调用直接发现到生产 IP，未经过此入口代理。

域名入口仍受平台域名路由、目标集群与回退规则影响，变量正确和三个 Pod 健康不能单独证明业务隔离；现场需用真实请求的 trace 核验 VQS → 代理 → PPE Aeolus 的完整链路。

## 已验证与边界

应用身份已通过真实任务 API 完成 BITS 固定 SHA 构建、SCM 制品独立核验、ICM 新镜像构建和单服务部署。主制品提交、镜像完整配方、18 项依赖、资源规格、Pod `/healthz` 和 `/version` 均有独立证据。单服务释放已确认 ENV 删除墓碑及记录的 TCE 服务、集群不存在；手工参考环境不属于任务的自动释放目标。

应用身份必须在 `data.vqs.api`（服务树节点 `133365`）具备 `tce.operator_ppe.cn`。缺失时 ENV 创建实测返回 HTTP 401；确认原环境不存在后才能对账并重试。测试授权有有效期，持续使用需保持授权有效。应用身份的 VQS 16 CPU 工单没有 `ConfirmStep` 资源审批步骤，不能将其他 PSM 或个人身份的审批规则推广至此。

Web 和飞书根据服务端能力开放 VQS 部署与重试入口，参数绑定本轮 SHA 和模板。浏览器已验证宽屏、窄屏的可操作性。

完整双服务环境已通过离线创建、部分服务/集群补缺、同任务升级、配套镜像重启恢复、变量漂移拦截及两侧资源释放测试。真实双服务部署已核验三个集群的镜像、依赖、资源、健康接口和 VQS 路由开关；旧 Atum 制品补读原镜像内嵌清单后，原资源无需再次升级即可进入业务验证。双服务第二提交升级、联合释放、真实浏览器下游路由和人工业务验证仍需现场验收。
