# Aeolus 研发助手：正式 AI 节点

正式插件为 **Aeolus 研发助手**，类型为 `ai_node`。已上架，限定可见空间：

- `6aabe23024218882059f74a4`
- `6ac733772f03a7ffb5b3952d`

[插件安装链接](https://meego.larkoffice.com/openapp/plugin_share?appKey=MII_6AED80FF53004C14)

| 配置 | 正式环境 |
| --- | --- |
| 站点 | `https://meego.larkoffice.com` |
| 插件 ID | `MII_6AED80FF53004C14` |
| AI 节点 key | `ai_node_aeolus` |
| Web 卡片资源 | `ai_node_web_aeolus_card` |
| 卡片 API origin | `https://aeolus-conciliator.byted.org` |
| Webhook | `https://aeolus-conciliator.byted.org/hooks/meegle/ai-node` |
| 输入属性 | `design_doc`：需求文档，关联 1 个工作项字段 |
| 本地插件工程 | `/data00/home/huaguojin/Conciliator-meegle-production/aeolus` |

原测试插件 ID 为 `MII_6AAF4159C3C0CCDB`，仍位于 `frontend/`。正式工程独立维护
插件身份和点位配置；卡片源码以本仓库 `frontend/` 为准，每次构建同步到正式工程。

## 构建正式卡片

在 `frontend/` 目录执行：

```bash
MEEGLE_PLUGIN_ROOT=/data00/home/huaguojin/Conciliator-meegle-production/aeolus npm run build:prod
```

脚本同步源码、依赖声明和构建脚本，不覆盖目标工程的 `plugin.config.json` 或
`point.config.local.json`。地址写入目标工程自己的 `src/build-config.json`；
正式构建不会改变测试工程生成的地址。目标工程依赖变更后需执行 `npm ci`。

## 生产后端绑定

代码已默认使用正式插件 ID，生产无需重复设置；如需显式指定：

```text
MEEGLE_PLUGIN_ID=MII_6AED80FF53004C14
```

正式节点拥有独立的 Plugin Secret 和 Webhook Token，启用前需通过 Meegle 开发者后台
将二者分别同步到生产 KMS；不要复制到源码或聊天记录：

| KMS Secret | 对应正式插件中的凭据 |
| --- | --- |
| `conciliator/credentials/meegle_plugin_secret` | Plugin Secret，用于 Meegle OpenAPI 换取 token |
| `conciliator/credentials/meegle_webhook_token` | AI 节点 Webhook Token，用于回调验签 |

授予 `data.aeolus.conciliator` 读取权限，重启生产服务加载凭据。
测试服务在 `.env` 配置原插件 ID 和对应凭据，配置见[本地后端](local-backend.md#开发配置)。
staging/prod 始终读取 KMS；不同插件的 Secret 和 Token 不能混用。

## 发布与启用

完成生产后端绑定并核验线上域名后，获得发布确认，再在正式工程执行：

```bash
lpm --cwd /data00/home/huaguojin/Conciliator-meegle-production/aeolus release --source-type local
# 以本次 release 实际返回的 artifactVersion 为准：
lpm --cwd /data00/home/huaguojin/Conciliator-meegle-production/aeolus publish \
  --artifact-version <artifactVersion> \
  --release-notes "Aeolus 研发助手正式节点" \
  --project-key 6aabe23024218882059f74a4 6ac733772f03a7ffb5b3952d \
  --overwrite-projects
```

发布后在目标空间安装该正式插件，把 AI 节点加入工作流，并将“需求文档”输入映射到
工作项的真实需求文档字段。使用测试工作项核验回调验签、卡片查询、评估确认、
Coding 和完成节点；不要仅凭构建成功认定整条链路已可用。

首次发布不传 `--project-key` 时会对全部空间可见。发布前需明确确认可见范围；
后续发布不传空间参数会继承上一版范围。如需更改，使用目标空间真实的 project key，
不能凭空间名称猜测。
