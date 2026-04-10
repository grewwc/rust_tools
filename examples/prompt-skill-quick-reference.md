# Prompt-Skill Agent 快速参考

## 🚀 快速开始

```bash
# 启动 agent
cargo run --bin a -- --agent prompt-skill

# 或者直接传入任务
cargo run --bin a -- --agent prompt-skill "帮我优化这个 prompt"
```

## 📋 常用命令

### 优化 Prompt
```
请优化以下 prompt，使其更清晰和结构化：

[粘贴需要优化的内容]
```

### 创建 Skill
```
创建一个用于 [用途] 的 skill，主要功能：
- [功能1]
- [功能2]
- [功能3]

请生成完整的 skill 文件。
```

### 分析质量
```
请分析这个 skill/prompt 的优缺点：

[粘贴内容]
```

### 生成模板
```
给我一个 [类型] 的 prompt/skill 模板，用于 [场景]
```

## 🎯 核心能力

| 能力 | 说明 | 示例 |
|------|------|------|
| Prompt 优化 | 改进现有 prompt 的质量 | "优化这个代码助手的 prompt" |
| Skill 生成 | 创建新的 skill 文件 | "创建一个 API 设计的 skill" |
| 质量评估 | 分析 prompt/skill 的问题 | "这个 skill 有什么可以改进的？" |
| 模板设计 | 提供标准化的模板 | "给我一个通用的 agent 模板" |
| 最佳实践 | 提供设计和优化建议 | "如何写好 skill 的描述？" |

## 💡 提示词工程要点

### 好的 Prompt 特征
✅ 角色定义清晰  
✅ 能力列表具体  
✅ 行为准则明确  
✅ 工具策略合理  
✅ 输出格式规范  

### 差的 Prompt 特征
❌ 模糊不清的指令  
❌ 缺少边界条件  
❌ 没有错误处理  
❌ 工具权限不当  
❌ 结构混乱  

## 🔧 Skill 设计要点

### YAML Front Matter 必填字段
```yaml
name: 简洁的名称（如 code-review）
description: 详细的描述（用于路由匹配）
tools: 必需的工具列表
priority: 优先级（0-100）
```

### 推荐的可选字段
```yaml
author: 作者信息
version: 版本号（如 1.0.0）
tool_groups: 工具组（如 builtin）
triggers: 触发关键词
routing_tags: 路由标签
```

### Markdown Body 结构
```markdown
# 标题

## 核心原则
- 原则1
- 原则2

## 最佳实践
- 实践1
- 实践2

## 示例
### 好的示例
...

### 差的示例
...
```

## 📊 优先级设置指南

| 优先级 | 使用场景 | 示例 |
|--------|----------|------|
| 90-100 | 关键技能，应该优先匹配 | debugger, build |
| 70-89 | 重要技能，经常使用 | code-review, refactor |
| 50-69 | 一般技能，特定场景使用 | api-designer, db-designer |
| 30-49 | 辅助技能，偶尔使用 | formatter, linter |
| 0-29 | 实验性技能 | test-skill |

## 🛠️ 工具权限配置

### 只读操作
```yaml
tools:
  - read_file
  - read_file_lines
  - code_search
  - search_files
```

### 读写操作
```yaml
tools:
  - read_file
  - write_file
  - apply_patch
  - code_search
```

### 完整权限
```yaml
tools:
  - read_file
  - write_file
  - apply_patch
  - execute_command
  - code_search
  - search_files
```

### `code_search` 参数提示
- 做结构化代码搜索时，固定使用 `operation=structural`
- 具体意图写在 `intent`：`find_functions`、`find_classes`、`find_methods`、`find_calls`
- 不要把 `find_functions`、`find_classes`、`find_methods`、`find_calls` 直接放进 `operation`
- 正确示例：`code_search(operation=structural, intent=find_functions, path=/repo, name=run)`

## 🎨 路由标签建议

### 英文标签
- `code`, `debug`, `build`, `test`
- `review`, `refactor`, `optimize`
- `design`, `architect`, `document`
- `prompt`, `skill`, `engineer`

### 中文标签
- `代码`, `调试`, `构建`, `测试`
- `审查`, `重构`, `优化`
- `设计`, `架构`, `文档`
- `提示词`, `技能`, `工程`

## ⚡ 快速模板

### 最小化 Skill 模板
```yaml
---
name: my-skill
description: 简要但准确的描述
tools:
  - read_file
  - apply_patch
priority: 50
---

# 核心指南

## 原则
- 原则1
- 原则2
```

### 完整 Skill 模板
```yaml
---
name: comprehensive-skill
description: 详细的描述，包含所有相关关键词和功能说明
author: your-name
version: 1.0.0
tools:
  - read_file
  - write_file
  - apply_patch
  - code_search
tool_groups:
  - builtin
priority: 70
triggers:
  - keyword1
  - keyword2
routing_tags:
  - tag1
  - tag2
---

# 技能名称

## 核心原则
- 原则1：说明
- 原则2：说明

## 最佳实践
### 推荐做法
1. 步骤1
2. 步骤2

### 避免做法
1. 错误1
2. 错误2

## 示例

### 好的示例
描述：...
步骤：
  - 步骤1
  - 步骤2

### 差的示例
问题：...
改进：...
```

## 🔍 常见问题

**Q: 如何让 skill 更容易被选中？**  
A: 编写详细准确的 description，添加相关的 routing_tags 和 triggers

**Q: 应该给多少工具权限？**  
A: 遵循最小权限原则，只授予完成任务必需的工具

**Q: 如何测试 skill 的效果？**  
A: 从用户角度审视，检查是否有歧义，验证工具权限是否充分

**Q: priority 应该设多少？**  
A: 根据重要性和使用频率，关键技能 90+，一般技能 50-70

## 📚 相关资源

- 完整指南: `docs/prompt-skill-agent-guide.md`
- 更新说明: `PROMPT_SKILL_AGENT_UPDATE.md`
- 测试用例: `examples/test-prompt-skill.md`
- Agent 源码: `src/bin/ai/builtin_agents/prompt-skill.agent`
