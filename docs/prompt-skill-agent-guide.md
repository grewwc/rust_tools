# Prompt-Skill Agent 使用指南

## 概述

`prompt-skill` 是一个专门用于优化和生成 prompt/skill 的 AI agent，具备提示词工程和技能设计的专业知识。

## 核心能力

### 1. Prompt 优化
- 分析现有 prompt 的问题和不足
- 应用提示词工程最佳实践
- 提高清晰度和可执行性
- 优化结构和组织方式

### 2. Skill 生成
- 创建符合规范的 skill 文件
- 设计准确的描述用于路由
- 配置适当的工具权限
- 设置合理的优先级

### 3. 模板设计
- 提供常见模式的模板
- 生成示例和最佳实践
- 确保 YAML front matter 语法正确

## 使用方法

### 基本调用

```bash
cargo run --bin a -- --agent prompt-skill "帮我优化这个 prompt"
```

### 典型场景

#### 1. 优化现有 Prompt

```
请优化以下 prompt，使其更清晰和结构化：

[粘贴需要优化的 prompt]
```

#### 2. 创建新 Skill

```
我需要创建一个用于代码审查的 skill，主要功能是：
- 检查代码风格
- 发现潜在 bug
- 提供改进建议

请帮我生成完整的 skill 文件。
```

#### 3. 分析 Skill 质量

```
请分析这个 skill 文件的优缺点，并提出改进建议：

[粘贴 skill 文件内容]
```

## 设计原则

### Prompt 设计

1. **角色定义**：明确 agent 的身份和职责
2. **能力列表**：清晰列出可以执行的操作
3. **行为准则**：定义如何响应用户请求
4. **工具策略**：说明何时使用哪些工具
5. **输出格式**：指定响应的结构和风格

### Skill 设计

1. **命名规范**：简洁、描述性强（如 `code-review.skill`）
2. **描述准确**：能够被路由系统正确匹配
3. **工具最小化**：只授予必要的工具权限
4. **优先级合理**：根据重要性设置 priority
5. **结构清晰**：YAML front matter + Markdown body

## 示例

### 优化前的 Prompt

```
你是一个编程助手，可以帮助用户写代码。
```

### 优化后的 Prompt

```markdown
You are a coding-focused AI assistant specialized in software development tasks.

Core capabilities:
- Write, read, and modify source code files
- Execute build commands and run tests
- Search and navigate codebases efficiently
- Debug compilation errors and runtime issues

Guidelines:
- Always read existing code before making changes
- Prefer minimal, targeted modifications over rewrites
- Use apply_patch for precise edits when possible
- Verify changes by running tests or builds
- Explain your reasoning for significant changes

Tool usage:
- Use code_search first for code exploration
- For structural code search, use `code_search(operation=structural, intent=find_functions|find_classes|find_methods|find_calls, ...)`
- Do not use `find_functions` or `find_calls` directly as the `operation` value
- Apply read_file_lines + apply_patch for modifications
- Run cargo check/build after Rust code changes
- Clean up any temporary files created during debugging
```

### 生成的 Skill 文件

```yaml
---
name: api-designer
description: API 设计专家：专注于 RESTful API 设计、OpenAPI 规范、接口文档生成和最佳实践
author: user
version: 1.0.0
tools:
  - write_file
  - read_file_lines
  - apply_patch
tool_groups:
  - builtin
priority: 70
triggers:
  - api
  - rest
  - openapi
  - swagger
---

# API Design Guidelines

## Core Principles
- RESTful resource modeling
- Consistent naming conventions
- Proper HTTP status codes
- Comprehensive error handling
- Versioning strategy

## Documentation Standards
- OpenAPI 3.0 specification
- Clear endpoint descriptions
- Request/response examples
- Authentication requirements

## Best Practices
- Use nouns for resources (not verbs)
- Implement pagination for collections
- Support filtering and sorting
- Rate limiting and throttling
- Security considerations (CORS, auth)
```

## 最佳实践

### 1. 迭代优化
- 先分析当前版本的问题
- 提出具体的改进方案
- 应用修改并验证效果
- 收集反馈继续优化

### 2. 保持一致性
- 使用统一的术语和风格
- 遵循项目现有的约定
- 保持文档格式的一致性

### 3. 测试有效性
- 从用户角度审视 prompt
- 检查是否有歧义或遗漏
- 验证工具权限是否充分
- 确保描述能触发正确的路由

### 4. 文档化决策
- 记录重要的设计选择
- 说明为什么采用某种结构
- 提供替代方案的对比

## 常见问题

### Q: 如何让 skill 更容易被路由系统选中？

A: 
- 编写详细且准确的 description
- 添加相关的 routing_tags
- 在 description 中包含常见的同义词
- 使用中英文双语描述关键概念

### Q: 应该给 skill 多少工具权限？

A:
- 遵循最小权限原则
- 只授予完成任务必需的工具
- 对于只读操作，不要授予写入权限
- 考虑安全风险，避免过度授权

### Q: 如何评估 prompt 的质量？

A:
- 清晰度：指令是否无歧义
- 完整性：是否覆盖边界情况
- 可执行性：agent 能否按要求行动
- 效率：是否能快速得到期望结果
- 一致性：不同输入下表现是否稳定

## 相关资源

- [Agent 架构文档](../src/bin/ai/agents.rs)
- [Skill 加载机制](../src/bin/ai/skills.rs)
- [内置 Agents](../src/bin/ai/builtin_agents/)
- [内置 Skills](../src/bin/ai/builtin_skills/)
