# Devbox 部署与恢复

Devbox 上的机器级 worker 可同时运行多个 Aeolus/VQS 实例。
每次部署使用独立 checkout、进程组、动态空闲端口和 journal，不再按服务槽位排队或占用机器。
同一个 Coding 任务重新部署前，须先确认其旧实例已在本机停止；其他任务不受影响。

## 接入

1. 接已有线上后端时，无需在 Devbox 安装后端或数据库；按 [worker README](../../../deployment-worker/README.md) 准备环境并检查入口。
   如需连接本地后端，按[本地后端](local-backend.md)启动 Conciliator，并将开发库升级至当前代码的迁移 head。
2. 后端始终初始化 Coding 和 Devbox/MAAT 部署。
   生产 AK/SK 从同一个 `conciliator/credentials/bytecloud_credentials` Secret 读取 JSON 对，本地 `ENV=dev` 使用 `BYTECLOUD_AK`、`BYTECLOUD_SK`。
   模板随包分发于 `src/conciliator/integrations/maat/baseline.json`，
   参考环境 `xiaorui-aeolus-hua`、操作人等随代码固定维护。
3. 按 [worker README](../../../deployment-worker/README.md) 克隆本仓库并安装，复制
   [配置示例](../../../deployment-worker/examples/deployment_worker.v2.example.json)，示例使用 Aeolus Python 3.10、VQS Python 3.7；删除不需要的服务配置。
   `protocol_version` 必须为 2，`advertised_host` 必须能从 MAAT 访问，不能填 localhost。
   worker 从该配置读取地址并注册，不自动探测 IP；后端不维护 worker/IP 白名单。
   地址随每次部署固定，上报端点必须与注册地址一致；有未释放部署时不能修改注册地址。
4. 预先准备业务解释器和依赖，将解释器路径、Python 版本及依赖路径统一写在 worker 的 runtime profile 中，无需额外 manifest 或文件摘要。部署不会安装业务依赖。
   Aeolus 使用 `data.aeolus.api_v3`，VQS 使用 `data.vqs.api`；确认 worker 可以停止自己启动的进程组。

## 配置与执行

- 每个仓库可配置具名 runtime_profile，填写 Python 版本、解释器和依赖路径；环境版本标识默认使用 profile 名称，无需手填。
  旧配置中的 `runtime_revision`、`files`、`runtime_manifest` 保留兼容，新接入无需填写。
  推荐读取本轮精确 SHA 的 Python 声明和依赖；证据不足或多个兼容候选时手动选择，
  明确不兼容的配置由服务端拒绝。`base_branches` 和 `base_release_max` 仅作分支提示，
  不限制手选；完整契约见[运行时选择](runtime-selection.md)。
  worker 的 Python 3.11 环境不变，业务代码仍来自独立的固定 SHA checkout。
- 卡片展示机器总部署数、运行中、部署中、停止待确认、状态未知和清理待处理，数量仅作压力参考。
- Coding 结果的完整 SHA 随部署固定，worker 拉取该提交到独立 checkout。
  Aeolus/VQS 的子模块（含嵌套子模块）按该提交固定的 SHA 自动拉取，
  不维护运行配置中的子模块名单。若 `dependency_paths` 显式配置 `aeolus_expr_parse`，
  跳过根目录同名子模块并复制准备好的 SCM parser；未配置时仍使用固定子模块版本。
  其他额外运行库不得覆盖 Git 子模块源码。
  动态端口不复用账本中仍被保留的端口，readiness 还须核验监听属于本实例的进程组。
- 后端创建本次部署专用 devApi/devEnv，名称为 `hua-ai-<deployment_id 前 12 位>-g<generation>`。
  Aeolus 部署替换 Aeolus 路由，VQS 部署替换 `aeolus/vqs`，其余路由继承本地模板。
- MAAT 创建携带完整配置后直接回读，不紧接着重复更新。飞书在创建待验证、产物部署完成和失败时推送，
  相同状态的轮询不重复推送。卡片分别选择 Devbox 和对应运行配置，切换机器不会创建部署。
- 本机服务就绪到 MAAT 入口创建期间显示部署进度，不写入部署错误；真实的 MAAT 写入异常和产物部署失败仍显示错误。
- MAAT 创建并回读后先显示“入口待验证”；产物完成后显示“MAAT 环境部署完成，待业务验证”。
  健康检查不代表登录后的完整业务页面可用，
  未取得入口验证证据时不能标记验证通过。

## 释放与恢复

“释放环境（保留 Coding 任务）”仅针对指定 deployment_id/generation。
worker 停止所属进程并移除隔离 checkout，保留部署日志、journal、assignment 和准备好的运行环境。
旧消息不能释放后续部署；重复停止 ACK 不重复发通知。

本机停服 ACK 后，后端自动软删除本次部署的 MAAT 环境，回查消失后删除无引用的代理并回查。
自动清理期间显示“本机已释放，MAAT 清理中”，不发送待人工清理通知。
环境和代理删除回查通过后，飞书发送一次“Devbox 释放完成”通知；后续轮询不重复发送。
清理成功显示“环境已释放”；失败或写入结果未知时保留记录供维护者对账。
软删除不保证网关路由撤销，因此清理后仍保留远程资源 ID 和旧端口预留；
可继续 Coding、重新部署或完成节点。

升级 worker 包后重启监督进程，会根据逐实例 journal 与后端对账，不能删除 ports.json 或实例记录。
离线、监听归属无法确认、停服未确认时保留资源状态，不杀其他服务来腾端口。

MAAT 调用契约与模板维护见[控制面说明](../../maat-control-plane-contracts.md)。
PPE 不使用本地 worker，按[PPE 运行约定](ppe-deployment-implementation.md)执行。
