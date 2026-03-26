## 用 AI Agent 搜索飞书记录（云文档）

这个仓库的 `a` 支持 MCP（Model Context Protocol）工具扩展。你可以启动一个 MCP server，把飞书开放平台的“云文档搜索”接口变成一个可调用工具，让 agent 直接搜索你在飞书里的文档标题/记录。

### 1) 构建 MCP Server

在仓库根目录执行：

```bash
cargo build --bin mcp_feishu
```

产物路径：

- macOS/Linux: `target/debug/mcp_feishu`

### 2) 准备飞书访问凭证

云文档搜索接口 `docs-api/search/object` 要求使用 **user_access_token**（用户授权凭证），仅用 `tenant_access_token` 会报错（常见表现：400 + `Invalid access token for authorization`）。

因此需要两步：

1) 先准备应用凭据（app_id/app_secret）用于换取 `app_access_token`（给 OAuth 换 token 用）  
2) 再走一次 OAuth 获取 `user_access_token`（之后可用 refresh_token 自动刷新）

运行时通过环境变量提供（不要写进仓库）：

- `FEISHU_USER_ACCESS_TOKEN`（推荐）
- 或 `FEISHU_ACCESS_TOKEN`（兼容）
- `FEISHU_APP_ID` + `FEISHU_APP_SECRET`（用于 OAuth 兑换/刷新 user_access_token）
- 可选：`FEISHU_BASE_URL`
  - 国内默认：`https://open.feishu.cn`
  - 国际可改为：`https://open.larksuite.com`

### 3) 配置 MCP

编辑 `~/.config/mcp.json`：

```json
{
  "mcpServers": {
    "feishu": {
      "command": "/ABS/PATH/TO/rust_tools/target/debug/mcp_feishu",
      "args": [],
      "env": {
        "FEISHU_USER_ACCESS_TOKEN": "u-***",
        "FEISHU_BASE_URL": "https://open.feishu.cn"
      },
      "request_timeout_ms": 20000,
      "disabled": false
    }
  }
}
```

如果你不想放 user_access_token（首次），可以先只放 app_id/app_secret：

```json
{
  "mcpServers": {
    "feishu": {
      "command": "/ABS/PATH/TO/rust_tools/target/debug/mcp_feishu",
      "args": [],
      "env": {
        "FEISHU_APP_ID": "cli_***",
        "FEISHU_APP_SECRET": "***",
        "FEISHU_BASE_URL": "https://open.feishu.cn"
      },
      "request_timeout_ms": 20000,
      "disabled": false
    }
  }
}
```

### 3.5) 获取 user_access_token（OAuth，一次性）

你需要在飞书开发者后台把重定向 URL 配进应用（例如：`http://127.0.0.1:8711/callback`）。

然后在 `a` 里调用 MCP 工具完成授权：

1) 生成授权链接：

- `mcp_feishu_oauth_authorize_url`
  - `redirect_uri`: `http://127.0.0.1:8711/callback`
  - `scope`: 建议至少包含 `offline_access`（需要 refresh_token）

2) 监听本地回调拿 code：

- `mcp_feishu_oauth_wait_local_code`
  - `port`: `8711`

3) 用 code 换 user_access_token：

- `mcp_feishu_oauth_exchange_code`
  - `code`: 上一步拿到的 code

拿到 `user_access_token` 和 `refresh_token` 后，把它们写进 `~/.config/mcp.json` 的 env（或用环境变量注入），后续就能直接搜索。

### 4) 启动 a 并验证工具已加载

```bash
cargo run --bin a -- --list-mcp-tools
```

你应该能看到一个工具（名字会被自动加前缀）：

- `mcp_feishu_docs_search`

### 4.5) 自动授权（推荐）

当你第一次让 agent 调用飞书云文档搜索工具时，如果本地没有可用的 user_access_token / refresh_token，程序会在终端自动弹出授权流程（确认 -> 打开浏览器 -> 本地回调 -> 换 token -> 保存）。

也可以手动触发一次授权（排查时用）：

- `/feishu-auth`

它会在终端里：

- 询问 scope（默认 `offline_access`）
- 生成授权链接并可选择自动打开浏览器
- 本地监听回调拿 code
- 自动调用换 token，并把 token 写入 `~/.config/rust_tools/feishu_token.json`

### 5) 在对话中怎么用

直接运行 `a` 即可：当问题涉及飞书云文档时，提示词会引导模型优先调用 Feishu MCP 工具。

然后直接问：

- “帮我在飞书云文档里搜索关键词：项目复盘，返回前 10 条”
- “搜索飞书里和 ‘面试记录’ 相关的文档，只要 sheet 类型”

工具参数（docs_search）：

- `search_key`（必填）
- `count`（0-50）
- `offset`（offset + count < 200）
- `docs_types`：`doc` / `sheet` / `slides` / `bitable` / `mindnote` / `file`
- `owner_ids`：Open ID 列表
- `chat_ids`：群 ID 列表

### 6) 数据来源与限制

- 当前实现只接了“云文档搜索”接口，返回的是文档列表（title/type/token/owner_id）。
- 如果你想进一步“读取文档内容并做全文检索”，已支持 doc/docx 纯文本抓取与导出：
  - `mcp_feishu_docs_get_text`：给定 docs_token + docs_type（doc/docx）直接返回纯文本
  - `mcp_feishu_docs_export_text`：导出为本地 txt 文件（默认到 `~/.config/rust_tools/feishu_docs_text/`），便于用 `grep_search` 做全文检索

示例流程：

1) 先搜索拿到 `docs_token`（比如 `dox...` / `doc...`）与 `docs_type`
2) 导出文本：调用 `mcp_feishu_docs_export_text`
3) 全文检索：调用内置工具 `grep_search` 在导出目录里搜关键字
