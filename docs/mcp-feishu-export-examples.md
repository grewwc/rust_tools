# 使用示例

## 前置条件

1. 确保已配置飞书 MCP 服务
2. 完成 OAuth 认证（运行 `feishu-auth` 命令）
3. 确保有合适的 app_id 和 app_secret 配置

## 示例 1：导出 CSV 到飞书表格

### 简单示例

```rust
// 在 a.rs agent 中询问：
请帮我创建一个销售数据表格，包含以下数据：
姓名，年龄，城市
张三，25，北京
李四，30，上海
王五，28，深圳
```

Agent 会自动调用 `sheet_create_from_csv` 工具创建表格。

### 编程方式调用

```json
{
  "name": "sheet_create_from_csv",
  "arguments": {
    "title": "2024 年销售数据",
    "csv_content": "产品，销量，销售额\n手机，1000,500000\n电脑，500,2500000\n平板，800,1600000",
    "folder_token": ""
  }
}
```

**预期输出**：
```
Created spreadsheet: https://app.feishu.cn/sheets/xxxxx
Token: xxxxx
```

## 示例 2：导出 Markdown 到飞书文档

### 简单示例

```rust
// 在 a.rs agent 中询问：
请帮我创建一个项目文档，内容如下：

# 项目概述

## 目标
- 完成产品开发
- 上线测试
- 用户反馈收集

## 时间线
> 重要：必须在 Q4 前完成

项目预计在下个月启动。
```

Agent 会自动调用 `doc_create_from_markdown` 工具创建文档。

### 编程方式调用

```json
{
  "name": "doc_create_from_markdown",
  "arguments": {
    "title": "项目计划文档",
    "markdown_content": "# 项目计划\n\n## 阶段一\n- 需求分析\n- 技术方案设计\n\n## 阶段二\n- 开发实现\n- 单元测试\n\n> 注意：需要定期同步进度\n\n具体细节待讨论。",
    "folder_token": ""
  }
}
```

**预期输出**：
```
Created document: https://app.feishu.cn/docx/xxxxx
ID: xxxxx
```

## 示例 3：指定文件夹存储

如果要将创建的文档/表格存储到特定文件夹，需要提供 `folder_token`：

```json
{
  "name": "sheet_create_from_csv",
  "arguments": {
    "title": "团队数据",
    "csv_content": "姓名，部门\n张三，技术部\n李四，产品部",
    "folder_token": "FOLDER_TOKEN_HERE"
  }
}
```

## 示例 4：在 a.rs agent 中的完整工作流

```rust
// 1. 首先完成认证（如果还未认证）
feishu-auth

// 2. 创建 CSV 表格
请帮我把以下数据创建成飞书表格：
日期，收入，支出
2024-01-01,10000,5000
2024-01-02,12000,6000
2024-01-03,11000,5500

// 3. 创建 Markdown 文档
请帮我创建一个会议记录文档：

# 周会记录

## 参会人员
- 张三
- 李四
- 王五

## 讨论内容
> 重点：下周发布新版本

1. 进度汇报
2. 问题讨论
3. 下周计划

## 行动计划
- 张三：完成功能开发
- 李四：准备测试用例
```

## 错误处理示例

### 缺少必需参数

```json
{
  "name": "sheet_create_from_csv",
  "arguments": {
    "title": "测试表格"
    // 缺少 csv_content
  }
}
```

**错误响应**：
```
Invalid params: csv_content is required
```

### 认证失败

如果未完成 OAuth 认证，会收到类似错误：
```
Missing user_access_token. Create spreadsheet requires OAuth once.
```

**解决方案**：运行 `feishu-auth` 命令完成认证。

## 最佳实践

1. **数据验证**：在调用前验证 CSV/Markdown 内容格式
2. **错误处理**：捕获并处理可能的 API 错误
3. **文件夹管理**：使用 folder_token 将相关文档组织到同一文件夹
4. **标题命名**：使用清晰、有意义的标题便于后续查找
5. **内容大小**：避免单次导入过大的内容，必要时分批处理

## 故障排查

### 问题：创建失败，提示认证错误

**解决方案**：
1. 运行 `feishu-auth` 重新认证
2. 检查 app_id 和 app_secret 配置是否正确
3. 确认飞书应用权限配置正确

### 问题：内容未正确显示

**解决方案**：
1. 检查 CSV 格式是否正确（逗号分隔、引号使用）
2. 检查 Markdown 语法是否符合支持的格式
3. 查看返回的错误信息获取详细原因

### 问题：文件夹 token 无效

**解决方案**：
1. 确认文件夹 token 是否正确
2. 确认有权限访问该文件夹
3. 尝试不指定 folder_token 使用默认位置
