---
title: Meegle AI node card, webhook, and publishing
description: Change the card API origin, safely update webhook point configuration, build, and publish to a test space.
keywords: [API_BASE, webhook, local-config, build, release, publish, artifactVersion]
last_verified: 2026-11-06
---

# 卡片、webhook 与发布

## 区分两个地址

- 卡片 `API_BASE`：浏览器访问后端的 origin，例如 `http://<devbox-ip>:8000`。
- AI 节点 webhook：Meegle 服务端调用的完整 URL，例如 `http://<devbox-ip>:8000/hooks/meegle/ai-node`。

两者可复用域名和端口，但配置互相独立。浏览器和 Meegle 服务端都必须能访问对应地址；`127.0.0.1` 通常不能跨机器联调。如果 HTTPS 页面阻止 HTTP mixed content，应使用 HTTPS 暴露后端。

## 修改卡片 `API_BASE`

地址由 `frontend/build.config.json` 的 dev/prod 配置在构建时确定。
默认 dev 为 `http://10.37.248.120:8000`，prod 为 `https://aeolus-conciliator.byted.org`。
卡片从构建脚本生成的 `frontend/src/build-config.json` 导入 `API_BASE`，无需修改组件源码。

```bash
cd /path/to/Conciliator/frontend
npm test
npm run build:dev   # 测试插件 / AI 节点
npm run build:prod  # 生产插件 / AI 节点；npm run build 同样使用 prod
```

临时覆盖地址时设置 `MEEGLE_CARD_API_BASE`，只填写 origin，不附加 API 路径；
生产构建只接受 HTTPS：

```bash
MEEGLE_CARD_API_BASE=https://<reachable-host> npm run build:dev
```

dev/prod 共用源码和构建目录，每次构建覆盖上一份产物。发布前核对构建输出的
`API_BASE` 与目标插件环境一致。测试和生产插件身份、AI 节点 webhook 仍需分别配置；
构建命令不创建插件，也不修改远端 webhook。

远程主机需要热更新时运行以下命令，并使用 CLI 输出的调试地址；不要加自动打开浏览器的 `--auto`：

```bash
cd /path/to/Conciliator/frontend
npm run dev
# 临时更换本地测试后端：
MEEGLE_CARD_API_BASE=http://<reachable-host>:8000 npm run dev
```

## 按需修改 webhook

地址或 token 没变时，不更新点位配置。需要变更时使用：

```text
URL:   http://<reachable-host>:8000/hooks/meegle/ai-node
Token: 与 MEEGLE_WEBHOOK_TOKEN 完全一致（ENV=dev）
```

生产环境从 `KMS_SECRET_MEEGLE_WEBHOOK_TOKEN_ALIAS` 指定的 KMS Secret 读取 Token，
默认路径为 `conciliator/credentials/meegle_webhook_token`。KMS 路径调整时保持 Token
值不变，无需修改 Meegle 点位的 Token；新路径须先授予服务身份读取权限。

点位配置是全量替换。先获取远端完整基线：

```bash
PLUGIN_ROOT=/path/to/Conciliator/frontend

lpm --cwd "$PLUGIN_ROOT" local-config get --remote
lpm --cwd "$PLUGIN_ROOT" schema
```

1. 按命令输出路径复制完整远端配置作为 draft。
2. 只修改目标 AI 节点现有 `listen_event` 中的 URL/token；字段结构以 schema 和 `lpm ai docs` 为准。
3. 保留所有其他点位和属性。
4. 人工检查完整 draft 后再执行：

```bash
lpm --cwd "$PLUGIN_ROOT" local-config set --from <完整-draft-json>
lpm --cwd "$PLUGIN_ROOT" local-config diff
lpm --cwd "$PLUGIN_ROOT" update --source-type=local
```

以上命令会影响远端配置，执行前必须获得用户明确确认。检查 `[ADDED]`、`[MODIFIED]`、`[DELETED]`；如果 CLI 因删除返回 exit 2，停止操作，由用户在自己的交互终端按 CLI 提示确认，不绕过删除闸口。

## 发布到限定测试空间

发布是外部可见变更，确认构建结果和可见范围后手工执行：

```bash
PLUGIN_ROOT=/path/to/Conciliator/frontend

lpm --cwd "$PLUGIN_ROOT" release
# 复制真实输出：Artifact version: <artifactVersion>

lpm --cwd "$PLUGIN_ROOT" publish \
  --artifact-version <artifactVersion> \
  --release-notes "更新 aeolus 编码助手本地联调版本" \
  --project-key <test_space_project_key>
```

- `artifactVersion` 必须来自本次 `release` 输出，不能自行构造。
- `build` 仅构建本地产物，不上传、不更新线上卡片。源码变更后必须重新 `release`，再 `publish` 本次返回的 artifactVersion；复用旧 artifactVersion 会继续展示旧代码。发布成功后核对线上 `front_version` 与本次 artifactVersion 一致，不能只检查版本为已上架。
- 不传 `--version` 时平台自动生成下一个 patch 版本。
- 本地联调不要使用 `--all-projects`。
- 发布后检查测试空间是否已升级或重新选择新插件/AI 节点版本；固定版本不会自动替换。
