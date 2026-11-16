---
title: Meegle local tooling and documentation service
description: Install LPM, register Meegle skills, authenticate, and query MCP-backed official documentation.
keywords: [LPM, meegle-plugin, skills, MCP, ai-docs, login]
last_verified: 2026-09-24
---

# 工具与知识服务

## 基础依赖

- Python 3.11 或更高版本、[uv](https://docs.astral.sh/uv/)
- Node.js、npm
- MySQL 8.x
- 能访问 `codebase-api.byted.org`、`open.feishu.cn` 和目标 Meegle 站点的网络

```bash
cd /path/to/Conciliator
uv sync --extra dev

cd frontend
npm ci
```

## 安装 LPM 和 Meegle skills

```bash
npm install -g @lark-project/cli@latest
lpm --help
```

正常安装时，LPM 的 `postinstall` 自动注册三个全局 skills：

- `meegle-plugin`
- `meegle-plugin-cli`
- `meegle-plugin-backend`

检查注册结果：

```bash
npx skills list -g
```

如果安装时使用了 `--ignore-scripts`，或设置了 `CI=true` / `LPM_SKIP_SKILLS`，自动注册会被跳过。POSIX 环境可手动注册：

```bash
npx skills add "$(npm root -g)/@lark-project/cli/skills" -g --all
```

注册完成后重启正在使用的编码工具，使其重新发现 skills。

## 登录与官方文档

```bash
lpm login --site-domain https://meego.larkoffice.com
lpm whoami
```

当前 LPM 不需要单独安装或常驻 `lpm mcp` 服务。Meegle 官方文档通过 MCP-backed 的 `lpm ai docs` 入口查询：

```bash
# 插件目录内查询；完整结果缓存在 frontend/.lpm-cache/mcp/
lpm --cwd /path/to/Conciliator/frontend \
  ai docs "AI 节点结果卡片 webhook 调试" --format json

# 后端目录等非插件目录中查询；结果直接输出
lpm ai docs --site-domain https://meego.larkoffice.com \
  "完成 AI 节点 OpenAPI"
```

`lpm login` 的凭据只保存在本机。不要在命令历史中使用 `--token`，也不要把 Developer Token 或 `plugin.config.json` 中的插件凭据复制到代码、文档和日志。

## Agent 查询规则

- 查点位结构：先运行 `lpm schema`，再按需查询 `lpm ai docs`。
- 查 webhook 验签：查询“兼容性说明与验签说明”或“webhook 签名校验”。
- 查 OpenAPI：使用接口中文名加 `OpenAPI`，例如“完成 AI 节点 OpenAPI”。
- 插件目录内的查询结果会缓存；后续优先按章节读取缓存，不重复加载整份资料。
