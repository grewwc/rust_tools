# Meegle Coding 前端

独立的 LPM 插件工程。开发、测试和构建在本目录执行，不依赖 Conciliator 的 Python 环境。

## 安装与开发

需要 Node.js、npm 和已安装的 `@lark-project/cli`（`lpm`）。

```bash
cd frontend
npm ci
npm test
npm run dev
```

`npm run dev` 使用 LPM 启动调试服务；使用 CLI 输出的调试地址访问。
浏览器访问的后端 origin 由 `build.config.json` 按构建环境选择：

| 命令 | 后端 origin |
| --- | --- |
| `npm run dev` / `npm run build:dev` / `npm run pack:dev` | `http://10.37.248.120:8000` |
| `npm run build` / `npm run build:prod` / `npm run pack` | `https://aeolus-conciliator.byted.org` |

临时更换测试地址：

```bash
MEEGLE_CARD_API_BASE=http://<reachable-host>:8000 npm run dev
MEEGLE_CARD_API_BASE=https://<reachable-host> npm run build:dev
```

只填写 origin，不附加 API 路径；生产构建必须使用 HTTPS。
脚本在启动调试或构建前生成 `src/build-config.json`，卡片直接导入该模块，
地址随产物固定。该文件无需提交，也不包含后端凭据。
修改配置后需重新启动调试或构建。AI 节点 webhook 是另一项平台配置，不随构建改变。

## 构建与部署

```bash
npm run build
npm run pack
# 保留测试地址的构建 / 打包：
npm run build:dev
npm run pack:dev
```

产物位于 `dist/`，`pack` 额外生成 zip；每次构建会覆盖上一份产物。
测试插件使用 dev 产物，生产插件使用 prod 产物，两者共用源码。
发布前核对构建输出的 `API_BASE` 并完成平台联调；上传和发布由维护者在本目录显式执行：

正式插件工程独立存放于 `/data00/home/huaguojin/Conciliator-meegle-production/aeolus`。
在本目录执行以下命令，将当前源码同步到该工程并构建；保留它自己的插件身份和点位配置：

```bash
MEEGLE_PLUGIN_ROOT=/data00/home/huaguojin/Conciliator-meegle-production/aeolus npm run build:prod
```

目标工程首次使用或依赖变更时，需同步 `package.json` / `package-lock.json` 并在目标工程
执行 `npm ci`。正式节点的配置与启用步骤见 [正式 AI 节点](../docs/knowledge/meegle-ai-node/production-node.md)。

```bash
lpm --cwd . release
# 使用上一条命令实际返回的 artifactVersion。
lpm --cwd . publish --artifact-version <artifactVersion> --release-notes "<发布说明>"
```

`plugin.config.json` 和 `point.config.local.json` 保留原插件身份与点位。
移动工程不需要创建插件或推送点位配置。插件凭据、LPM 缓存及含凭据的配置不得进入分发包。
完整发布约定见 [卡片、webhook 与发布](../docs/knowledge/meegle-ai-node/card-webhook-publish.md)。
