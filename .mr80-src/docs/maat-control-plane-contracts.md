# MAAT 调用契约与模板维护

安装、凭据和启动见[本地后端](knowledge/meegle-ai-node/local-backend.md)，
业务操作见[Devbox 部署](knowledge/meegle-ai-node/devbox-deployment.md)。
当前后端使用公开 OpenAPI 创建、回读并软删除 MAAT 环境和代理记录。

## 认证与调用

MAAT/PPE 共用 ByteCloud AK/SK，官方 `bytedance.cloud-sdk-python` 换取和刷新应用 JWT。
SDK 版本以 [`byted-requirements.txt`](../byted-requirements.txt) 为准；不复用个人 Cookie 或 CLI 登录缓存。
MAAT 的项目操作人为 Settings 中固定且经授权的 `maat_action_user`，不能由客户端指定，也不能直接以应用账号名替代。

网关为 `https://bc-cn-gw.bytedance.net`，请求携带 `X-Jwt-Token`、
`domain: maat;v1` 和 `x-action-user`。仅向绑定的 HTTPS origin 发送 JWT，拒绝重定向；
写入失败不自动重放。

| 当前使用的公开路径 | 用途与约束 |
| --- | --- |
| `/api/open/devApi/create` | 创建本次实例代理；port 为字符串，返回 ID、名称和 endpoint 须匹配 |
| `/api/open/devEnv/create` | 提交完整模板与新代理绑定；写前保存意图，响应返回资源 ID |
| `/api/open/devEnv/list` | 按精确名称和 ID 回读完整配置；`pageNo`、`pageSize` 为 JSON 整数 |
| `/api/open/devEnv/update` | 更新已有环境；完整创建后不紧接着重复更新 |
| `/api/open/devConfig/get` | 获取项目域名配置 |
| `/api/open/devEnv/remove` | 仅删除本次部署创建的环境记录，成功后从环境列表回查消失 |
| `/api/open/devApi/list` | 用 `name` 查询代理，按精确名称、项目、ID、创建人和端点核对归属 |
| `/api/open/devApi/getUsedEnvList` | 删除代理前确认有效环境引用为空 |
| `/api/open/devApi/remove` | 删除本次部署创建且无引用的代理记录，成功后回查消失 |

接口字段以 [`control_plane.py`](../src/conciliator/integrations/maat/control_plane.py) 为准。
公开模式不依赖受限的 `devEnv/detail`，不调用控制台文件发布、网关发布或控制台删除路径；
控制台客户端及其离线测试不能作为已接入能力的证明。
新增接口需要对当前 ByteCloud 应用单独开通调用审批；HTTP 200 仍须检查业务 `code`。

## 实例与模板

- 一次部署对应独立 devApi/devEnv，名称为 `hua-ai-<deployment_id 前 12 位>-g<generation>`。
- `advertised_host` 来自 worker 配置，端口由实例动态分配并保存到部署快照；不使用静态 worker/IP 白名单。
- 按精确 `pathname`、匹配类型、原 `apiId` 和模板摘要替换本服务规则，其余代理和固定 SCM 配置保留。
- 新环境不继承参考环境 ID、状态或订阅群；同名环境查询命中时停止，不覆盖或认领。
- 预查不能代替服务端唯一约束。接口重名幂等语义未确认，响应丢失时保留意图和 ID，按原资源对账。
- 当前支持 CN、generic 域名，入口为 `https://<name>.maat.bytedance.net`，可追加已校验的默认路径。

模板位于 [`baseline.json`](../src/conciliator/integrations/maat/baseline.json)，基线名称、摘要、操作人和代理在 Settings 固定维护。
启动只读本地模板；身份、SHA 或必需路由不匹配时拒绝启动。模板更新不会自动修改已有环境或共享代理资源。

只读核验或显式导出基线时，在开发配置就绪的仓库根目录执行：

```bash
uv run python scripts/verify_maat_access.py
# 创建新快照，不覆盖已有文件：
uv run python scripts/verify_maat_access.py \
  --save-snapshot /tmp/maat-baseline-reviewed.json
```

脚本仅输出阶段、HTTP/business code 和匹配数量；导出文件权限为 0600。
审查完整环境配置和域名策略后更新 `baseline.json` 及 `maat_baseline_revision`，重启后端加载。

## 就绪与回收

创建后直接回读代理绑定、固定 SCM 版本和环境状态；入口显示“待验证”，
产物完成仅显示“待业务验证”，不自动记录人工验证通过。
本机监听归属、MAAT 网络转发、登录后业务页面与当前 Coding SHA 都需要各自的证据。

取消、超时或释放由 worker 停止所属进程并回报 ACK。本机停止已确认后可继续 Coding 或重新部署。
收到 ACK 后，后端依次软删除环境、回查环境消失、核验代理零引用、删除代理并回查消失。
每次删除沿用持久化写入意图；失败或响应丢失时保留资源信息供维护者对账，不自动重放。
清理成功仅表示环境和代理记录已删除，不设置 `maat_cleanup_verified`，也不释放端口账本。
软删除不会保证撤销网关路由或删除部署产物，因此仍保留远端 ID 和旧端口预留。
worker 地址和实例一致性校验不构成调用者鉴权，生产入口保护仍须落实。

2026-11-12 使用应用 JWT 在 `aeolus` 项目完成独立测试对象的创建、引用检查和删除回查：
被引用的代理删除返回 HTTP 200 / `code=100` 且记录保留；环境删除返回 `code=0` / `删除成功`，
随后代理引用为空，代理删除返回 `code=0` / `ok`；环境和代理均从有效记录列表消失。
该验证未检查网关路由撤销，不涉及既有部署的删除。

[未回收测试资源台账](maat-real-validation-2026-11-02.json)保留用于维护者对账；
其中临时探针的转发结果不能替代真实业务或 worker 生命周期验收。

## 契约来源

- [官方 MAAT v1 目录](https://cloud.bytedance.net/api/v2/openapi/llms/maat/v1/llms.txt)
- [环境软删除契约](https://cloud.bytedance.net/api/v2/openapi/llms/maat/v1/api/b805ac17bbd2e04dd70e9636e7b50238ec3a67ec.md)
- [代理删除契约](https://cloud.bytedance.net/api/v2/openapi/llms/maat/v1/api/351fa323ca3b4a06ee2d4de25045d755f43a5b91.md)
- [平台创建/更新调用方](https://code.byted.org/devops/maat_dev_env_operator/blob/e9fcce9a5342c6b6672e9b514b375c382c78a880/atomic/example/plugin.go)
- [官方 ByteCloud SDK](https://code.byted.org/paas/cloud-sdk-python/tree/v0.1.14)
