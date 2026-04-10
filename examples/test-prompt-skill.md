# Prompt-Skill Agent 功能测试

## 测试场景 1: 优化一个简单的 prompt

### 输入
```
你是一个代码助手。你可以帮用户写代码、改代码、调试代码。你会用各种工具来完成这些任务。

你要小心一点，不要随便删除用户的代码。如果不确定就问用户。

你可以做的事情：
- 读文件
- 写文件
- 运行命令
- 搜索代码

记住要先看看现有的代码再修改。
```

### 期望输出
一个结构化的 prompt，包含：
- 清晰的角色定义
- 明确的能力列表
- 详细的行为准则
- 工具使用策略
- 最佳实践指导

---

## 测试场景 2: 创建一个新的 skill

### 输入
```
我需要创建一个用于数据库设计的 skill，主要功能包括：
- 设计数据库 schema
- 生成 SQL 迁移脚本
- 提供性能优化建议
- 检查数据完整性约束

请帮我生成完整的 skill 文件。
```

### 期望输出
一个完整的 `.skill` 文件，包含：
- YAML front matter（name, description, tools, priority 等）
- Markdown body（guidelines, examples, best practices）
- 适当的工具权限配置
- 准确的描述用于路由

---

## 测试场景 3: 分析现有 skill 的质量

### 输入
```
请分析以下 skill 文件的优缺点，并提出改进建议：

---
name: test
description: 测试技能
tools:
  - read_file
priority: 50
---

这是一个测试技能。
```

### 期望输出
详细的分析报告，包括：
- 当前版本的问题（描述过于简单、缺少作者信息、工具权限不足等）
- 改进建议（添加更详细的描述、补充必要字段、优化工具配置等）
- 优化后的完整版本

---

## 验证方法

运行以下命令测试：

```bash
# 测试场景 1
cargo run --bin a -- --agent prompt-skill "请优化这个 prompt: [粘贴上面的输入]"

# 测试场景 2
cargo run --bin a -- --agent prompt-skill "我需要创建一个用于数据库设计的 skill..."

# 测试场景 3
cargo run --bin a -- --agent prompt-skill "请分析以下 skill 文件的质量..."
```
