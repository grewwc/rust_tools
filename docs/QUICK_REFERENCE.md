# 快速参考 - 飞书导出功能

## 工具列表

| 工具名称 | 功能 | 必需参数 | 可选参数 |
|---------|------|---------|---------|
| `sheet_create_from_csv` | 创建表格 | title, csv_content | folder_token |
| `doc_create_from_markdown` | 创建文档 | title, markdown_content | folder_token |

## 快速示例

### CSV → 表格

```json
{
  "name": "sheet_create_from_csv",
  "arguments": {
    "title": "我的表格",
    "csv_content": "A,B,C\n1,2,3\n4,5,6"
  }
}
```

### Markdown → 文档

```json
{
  "name": "doc_create_from_markdown",
  "arguments": {
    "title": "我的文档",
    "markdown_content": "# 标题\n\n- 列表项 1\n- 列表项 2"
  }
}
```

## 支持的 Markdown 语法

| 语法 | 效果 | 示例 |
|-----|------|------|
| `# ` | 一级标题 | `# 标题` |
| `## ` | 二级标题 | `## 副标题` |
| `### ` | 三级标题 | `### 小标题` |
| `- ` 或 `* ` | 列表项 | `- 项目` |
| `> ` | 引用块 | `> 引用` |
| 其他 | 普通段落 | `这是文本` |

## CSV 格式说明

- 使用逗号 `,` 分隔字段
- 使用双引号 `"` 包裹包含特殊字符的字段
- 使用双引号 `""` 表示字段中的单个引号

示例：
```csv
姓名，年龄，备注
张三，25,"喜欢""编程"""
李四，30,普通文本
```

## 认证流程

首次使用前需要认证：

```bash
# 在 a.rs agent 中运行
feishu-auth
```

## 常见错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `title is required` | 缺少 title 参数 | 添加 title 参数 |
| `csv_content is required` | 缺少 csv_content 参数 | 添加 csv_content 参数 |
| `markdown_content is required` | 缺少 markdown_content 参数 | 添加 markdown_content 参数 |
| `Missing user_access_token` | 未认证 | 运行 `feishu-auth` |
| `Failed to create spreadsheet` | API 调用失败 | 检查网络和权限 |

## 返回格式

成功时返回：
```
Created spreadsheet: https://xxx.feishu.cn/sheets/TOKEN
Token: TOKEN
```

或：
```
Created document: https://xxx.feishu.cn/docx/ID
ID: ID
```

## 文件位置

- 主代码：`/Users/bytedance/rust_tools/src/bin/mcp_feishu.rs`
- 功能文档：`/Users/bytedance/rust_tools/docs/mcp-feishu-export-features.md`
- 使用示例：`/Users/bytedance/rust_tools/docs/mcp-feishu-export-examples.md`
- 实现总结：`/Users/bytedance/rust_tools/docs/IMPLEMENTATION_SUMMARY.md`

## 相关命令

```bash
# 编译 mcp_feishu
cargo build --bin mcp_feishu

# 编译整个项目
cargo build

# 运行 a.rs agent
./target/debug/a

# 检查代码
cargo check --bin mcp_feishu
```
