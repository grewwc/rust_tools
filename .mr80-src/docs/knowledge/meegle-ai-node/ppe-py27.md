# Aeolus Python 2.7 PPE

Python 2.7 与 Python 3.10 复用任务 API、BITS → SCM → ICM → ENV/TCE 流程，
以及升级、重试、持久恢复、就绪复查和释放机制，无需新增数据库迁移或卡片点位配置。
PPE 与 Devbox 按精确 SHA 推荐运行时并支持手动选择，见[运行时选择](runtime-selection.md)。

## 配方和来源

[aeolus_py27_cn.json](../../../src/conciliator/integrations/ppe/aeolus_py27_cn.json) 保存独立配方：

- 主 SCM 为 `data/dp/aeolus` / `4863`；Python 3 配方的 `572600` 不能替代。
- [SCM-only 流水线](https://bits.bytedance.net/devops/284273157378/pipeline/detail/1254638941442)
  使用精确提交、`scm_type=test`，没有自动触发器或部署步骤。
- 镜像配方取自 `data.aeolus.api_v3` / `1.0.2.907`，完整保留 26 项 SCM 依赖、
  安装路径及 apt/pip 包；基础镜像为 `debian.jessie.jdk8_yarn:latest`，
  启动 `toutiao/app/data/aeolus/run/prod_run.sh`。
- 解析器为 `aeolus_expr_parse_py2` / `256483`，MySQLdb 位于 Python 2.7 的
  site-packages。`pytos` 固定 `1.2.11.1333`，包清单包含 `sqlparse==0.3.1`；
  旧参考配方的 `pytos:1.0.0.92` 已被平台禁用，不能重新构建。
- 来源为 `ppe_aeolus_jhc_py2`，`online` / `query_online_mirror` 来源集群为
  `219217817` / `219217814`。目标各 8 CPU / 32 GiB / 1 实例，LF / PPE / IPv6。
  健康检查必须返回 `Hello Aeolus!`，不能沿用来源中的 `exit 0`。
- 配套 VQS 使用固定生产镜像及 `AEOLUS_SERVICE_DISCOVERY=0`；
  完整环境为 32 CPU / 96 GiB / 3 实例。

每轮仅替换主仓的成功 `test` 制品，独立核验 SHA 和镜像配方。
CN PPE 不能使用 BOE 的 `offline` 制品；精确 SHA 没有 test 制品时走固定 BITS 流水线。
重试不复用已知 offline 缓存；旧记录缺少类型时，镜像写入前再次查询 SCM 详情。

模板与选择随部署记录冻结，恢复和重试使用原配方。更换运行时或模板 revision
须先释放旧环境。来源仅用于读配置，更新与释放只操作任务自己的资源。
2026-11-11 查询到来源服务租期为 2026-11-14；来源被回收或身份改变会阻止新建，
维护时需重新核验并更新完整模板，不自动续租他人的环境。

## 实机验证

2026-11-11 使用应用身份和隔离台账完成真实部署、第二 SHA 升级、跨进程恢复、
无缓存业务查询和联合回收。两个 Aeolus 集群各 8 CPU，新建及升级工单均无 ConfirmStep。
实例健康、镜像、完整依赖、资源规格及以下精确 SHA 均通过独立核验：

- 首轮：`b2b392f16817921f0c90a29fcb6b0b94e61b27f1`。
- 第二轮：`0340e0c863072607d731ce5ee48098bfbf68bae2`，
  test SCM `1.0.11.9689` / `174428825`，镜像 `1.0.2.1086` / `179979516`。
- VQS 固定基线：`dd854438c71627b76419aabca52ade725437eb60`。

业务域名请求关闭 query/schema 缓存，log ID
`202611112137023259DCD310621E2FC8` 的服务日志、代理目标及 TCE Pod 身份
确认入口 → 本任务 VQS → 代理 → 同任务 py2.7 Aeolus 链路。
BytedTrace 未返回 span 树，浏览器业务页面人工验收未执行。

三个 generation 的环境、两服务及三集群均已独立回查删除，没有遗留运行资源；
成功 SCM/ICM 制品保留。219 项 PPE 测试通过，新增运行时选择的验证见对应文档。
