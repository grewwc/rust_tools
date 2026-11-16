# PPE 卡片端到端验收

实现与恢复规则见 [运行约定](ppe-deployment-implementation.md)，平台契约见 [PPE 平台结论](ppe-environment-conclusions.md)。

## 验证范围

验收覆盖卡片发起、双服务创建/补缺/升级、旧轮次拦截、业务路由、人工确认及资源释放。
离线测试和历史平台探活只作回归依据，部署现场须核验当前版本和实际资源。

联合模式仍需现场证明浏览器请求命中 PPE、VQS → `inf.env.proxy` → 同泳道 Aeolus 的完整路由、
第二提交升级和两侧删除回查。热部署、无缓存的全部依赖下载权限尚未验证；自动续期和平台取消未实现。

## 开始之前

1. 目标库已应用 `0019_ppe_deployments`，ByteCloud 应用凭据及权限齐备；PPE 始终初始化，启动不执行迁移。
2. `/healthz`、`/readyz` 的响应体均为 `status=ok`。平台使用 ByteCloud 应用 JWT，BITS `Username` / ICM `username` 使用已授权操作人。
3. Meego 使用已核验上架版本，`API_BASE` 与 webhook 指向可访问后端。
4. 测试任务为 `dp/aeolus` 或 `dp/VizQueryService`，最新成功 Coding 轮次进入 Review，提交为完整 SHA；`coding_blocked` 不提供新部署。
5. 使用测试账号、测试数据和可设置请求头的浏览器。PPE 与线上存储不隔离，实例环境标签不能证明下游路由隔离。
6. 应用身份同时拥有 `data.aeolus.api_v3` 与 `data.vqs.api` 的 PPE 操作权限，LF 资源池满足三个集群的配额。旧模板任务先释放环境再使用新模板。

## 操作与预期

| 操作 | 预期结果 |
| --- | --- |
| Review 发起 PPE 部署 | 任务专属环境、当前 SHA 和阶段可见，无 Devbox 也可操作 |
| 等待 SCM/ICM | 主仓精确 SHA 的成功制品可复用，否则触发 SCM-only BITS；另一侧直接复用固定生产成品镜像，核验精确 ID 与摘要，不触发配套构建 |
| 等待 ENV/TCE | 同 ENV 包含两个 PSM、三个集群；Aeolus 两集群各 8 CPU / 32 GiB / 1 实例，VQS 单集群 16 CPU / 32 GiB / 1 实例，分别核验依赖和镜像 |
| 三个 Pod healthz 及 VQS 变量检查通过 | 显示部署完成、待业务验证；VQS `AEOLUS_SERVICE_DISCOVERY=0`，未自动记录人工验证 |
| 打开 Aeolus 业务前端 | 按环境名设置 `X-TT-ENV`，检查完整 trace 中 VQS → 域名入口代理 → 同 PPE Aeolus 的调用和版本证据，证明命中目标代码；请求头或健康响应不能单独证明 |
| 新反馈后部署第二 SHA | 环境名、generation、服务和集群 ID 不变；主仓制品更新，配套服务基线和镜像保持，VQS 变量不丢失；旧部署不能确认新轮次验证 |
| 更新进行中或失败 | 上一版镜像、提交与访问方式持续可见；不将旧环境当作新代码验证依据 |
| 断网或重复提交 | 同任务/操作/轮次/SHA/模板/重试目标复用 request ID；参数改变换新 ID，不重复创建 |
| 选择“验证通过”并完成 | 确认快照中的全部可用环境；目标过期/变化时阻止提交；任一目标失效整次回滚 |
| 等待后台释放 | 节点完成后卡片仍刷新；ENV（含删除墓碑）及已记录 TCE 服务/集群确认不存在后才显示已释放 |

独立释放保留 Coding 任务；释放后重建产生新身份和 generation。删除失败保留资源记录及显式重试入口。不要用参考环境或其他 POC 代替本任务验收。

## 本地构建与发布

在 `frontend/` 执行 `npm run build:dev`（本地测试）或 `npm run build:prod`（正式地址），产物为 `dist/ai_node_web_aeolus_card/`。构建使用 CLI 内置 schema 时只能验证本地源码和点位，不代表远端配置已核验。

前端类型配置使用 `bundler`、现代标准库及 `noEmit`。发布前核对完整远端基线、真实 artifact version 和目标空间，按仓库规则取得明确确认；操作方式见 [发布流程](card-webhook-publish.md)。
