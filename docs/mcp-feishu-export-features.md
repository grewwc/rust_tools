# 飞书 MCP 工具新增功能

## 概述

为 `mcp_feishu` MCP 服务器添加了两个新工具，支持将本地内容导出到飞书云平台。

## 新增工具

### 1. sheet_create_from_csv

**功能**：创建新的飞书电子表格，并将 CSV 内容写入表格。

**输入参数**：
- `title` (必需): 电子表格标题
- `csv_content` (必需): CSV 格式的内容
- `folder_token` (可选): 存储电子表格的文件夹 token

**返回结果**：
- 创建成功的电子表格 URL
- 电子表格 token

**示例用法**：
```json
{
  "name": "sheet_create_from_csv",
  "arguments": {
    "title": "销售数据",
    "csv_content": "姓名，年龄，城市\n张三，25，北京\n李四，30，上海",
    "folder_token": ""
  }
}
```

**实现细节**：
1. 调用飞书 API `POST /open-apis/sheets/v3/spreadsheets` 创建电子表格
2. 解析 CSV 内容为二维数组
3. 调用飞书 API `PUT /open-apis/sheets/v2/spreadsheets/{token}/values/batchUpdate` 批量写入数据
4. 返回电子表格 URL 和 token

### 2. doc_create_from_markdown

**功能**：创建新的飞书文档（docx），并将 Markdown 内容写入文档。

**输入参数**：
- `title` (必需): 文档标题
- `markdown_content` (必需): Markdown 格式的内容
- `folder_token` (可选): 存储文档的文件夹 token

**返回结果**：
- 创建成功的文档 URL
- 文档 ID

**示例用法**：
```json
{
  "name": "doc_create_from_markdown",
  "arguments": {
    "title": "项目文档",
    "markdown_content": "# 项目概述\n\n## 目标\n- 目标 1\n- 目标 2\n\n> 重要提示\n\n这是正文内容。",
    "folder_token": ""
  }
}
```

**实现细节**：
1. 调用飞书 API `POST /open-apis/docx/v1/documents` 创建文档
2. 将 Markdown 转换为 docx blocks 格式：
   - `# ` → heading1
   - `## ` → heading2
   - `### ` → heading3
   - `- ` 或 `* ` → bullet
   - `> ` → quote
   - 其他 → text
3. 调用飞书 API `POST /open-apis/docx/v1/documents/{id}/blocks/batch-update` 批量更新文档内容
4. 返回文档 URL 和 ID

## 辅助函数

### parse_csv_line

解析单行 CSV 内容为字符串数组，支持：
- 逗号分隔
- 引号包裹的字段
- 引号转义（双引号表示单个引号）

### convert_markdown_to_docx_blocks

将 Markdown 文本转换为飞书 docx blocks 格式，支持：
- 三级标题（#, ##, ###）
- 列表项（-, *）
- 引用块（>）
- 普通段落

## 认证要求

两个工具都需要用户完成飞书 OAuth 认证流程：
1. 首次使用前需要执行 `feishu-auth` 命令完成 OAuth 认证
2. 用户访问令牌会自动缓存和刷新
3. 需要配置飞书应用的 app_id 和 app_secret

## 错误处理

- 参数验证：检查必需参数是否存在
- API 调用错误：返回详细的错误信息和响应内容
- 认证错误：提示用户完成 OAuth 认证流程

## 文件修改

- **修改文件**: `/Users/bytedance/rust_tools/src/bin/mcp_feishu.rs`
- **新增行数**: 约 450 行
- **新增函数**: 4 个
  - `feishu_sheet_create_from_csv`
  - `feishu_doc_create_from_markdown`
  - `parse_csv_line`
  - `convert_markdown_to_docx_blocks`

## 测试建议

1. 测试 CSV 导出功能：
   - 简单 CSV 数据
   - 包含特殊字符的 CSV 数据
   - 包含引号的 CSV 数据
   - 空 CSV 数据（应报错）

2. 测试 Markdown 导出功能：
   - 包含各级标题的 Markdown
   - 包含列表的 Markdown
   - 包含引用块的 Markdown
   - 混合格式的 Markdown
   - 空 Markdown 内容（应报错）

3. 测试文件夹功能：
   - 指定有效的 folder_token
   - 不指定 folder_token（默认行为）

## 注意事项

1. **API 限制**：飞书 API 可能有调用频率限制，大批量导入时需要注意
2. **内容大小**：单次 API 调用的内容大小有限制，超大内容需要分批处理
3. **Markdown 支持**：当前实现仅支持基础 Markdown 语法，复杂格式（如表格、代码块、图片）暂不支持
4. **CSV 格式**：当前实现假设 CSV 使用逗号分隔，其他分隔符需要额外处理

## 后续改进建议

1. 增强 Markdown 解析器，支持更多语法（表格、代码块、图片链接等）
2. 支持自定义 CSV 分隔符
3. 添加进度反馈和错误恢复机制
4. 支持从文件读取内容而不是仅支持字符串输入
5. 添加批量导入功能，支持多个文件
